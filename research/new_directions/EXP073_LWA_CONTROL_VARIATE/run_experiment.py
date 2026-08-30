from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
OUT = ROOT / "research" / "new_directions" / "EXP073_LWA_CONTROL_VARIATE"
EXP072_SCRIPT = ROOT / "research" / "new_directions" / "EXP072_LWA_TAB" / "run_experiment.py"
EXP072_CONFIG = ROOT / "research" / "new_directions" / "EXP072_LWA_TAB" / "config.json"
EXP072_PILOT = ROOT / "research" / "new_directions" / "EXP072_LWA_TAB" / "pilot_metrics.json"

DESIGN_FOLD = "2025-10-16"
CONFIRM_FOLD = "2025-10-02"
ALPHA_GRID = np.asarray([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00])
PARITY_TOL = 2e-6


def load_exp072():
    spec = importlib.util.spec_from_file_location("exp072_reused", EXP072_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(EXP072_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E72 = load_exp072()


def log(*parts: object) -> None:
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fit_lgb(X: np.ndarray, y: np.ndarray, seed: int):
    import lightgbm as lgb

    if len(X) != len(y) or X.shape[1] != 227 or not np.isfinite(y).all():
        raise AssertionError(f"invalid LightGBM matrix {X.shape}/{y.shape}")
    params = dict(E72.LGB_PARAMS)
    params["seed"] = int(seed)
    params["num_threads"] = int(os.environ.get("LGB_THREADS", params.get("num_threads", 12)))
    dataset = lgb.Dataset(X, label=y, params=params, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=E72.LGB_ROUNDS)
    del dataset
    return model


def donor_raw_preprocess(raw: np.ndarray, donor_mask: np.ndarray) -> dict[str, Any]:
    donor = np.asarray(raw[donor_mask], float)
    if len(donor) == 0 or not np.isfinite(donor).all():
        raise AssertionError("empty/nonfinite donor preprocessing sample")
    lo, hi = np.quantile(donor, [0.005, 0.995])
    center = float(np.clip(donor, lo, hi).mean())
    return {
        "q005": float(lo),
        "q995": float(hi),
        "center": center,
        "n_donor": int(len(donor)),
    }


def apply_raw_preprocess(raw: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.clip(np.asarray(raw, float), params["q005"], params["q995"]) - params["center"]


def score(y: np.ndarray, z: np.ndarray) -> float:
    return E72.calibrate(np.asarray(y, float), np.asarray(z, float))[1]


def stage0() -> dict[str, Any]:
    started = time.time()
    metrics_path = OUT / "stage0_metrics.json"
    vector_path = OUT / "stage0_design_rows.parquet"
    if metrics_path.exists() or vector_path.exists() or (OUT / "report.md").exists():
        raise FileExistsError("Stage 0 outputs already exist; refusing to overwrite")
    if sha256(EXP072_CONFIG) != "8e2c1509cee02959cc0667131209154f59c13192b456594724a58326ada9c6c1":
        raise AssertionError("EXP072 config changed")
    if sha256(EXP072_PILOT) != "095d768bd344274c9f04e7b9bbefabc734e58bfc708acc5ee3d7d54ccf555185":
        raise AssertionError("EXP072 pilot metrics changed")
    exp072_pilot = json.loads(EXP072_PILOT.read_text(encoding="utf-8"))

    schema = list(pl.read_parquet_schema(E72.feature_path(E72.PILOT_FOLD)).keys())
    columns = [name for name in schema if name != "user_id"]
    if len(columns) != 227:
        raise AssertionError(f"expected frozen 227 features, got {len(columns)}")

    log("Stage 0: load canonical raw events and reconstruct EXP072 legal metadata")
    events = pl.read_parquet(E72.RAW, columns=["user_id", "event_date", "gmv"])
    clean_cutoffs = E72.clean_cutoffs(E72.PILOT_FOLD)
    if len(clean_cutoffs) != 24:
        raise AssertionError(f"design CLEAN cutoff count changed: {len(clean_cutoffs)}")
    clean_meta: list[dict[str, Any]] = []
    for index, cutoff in enumerate(clean_cutoffs, 1):
        item = E72.cutoff_positive_metadata(events, cutoff)
        clean_meta.append(item)
        log(f"CLEAN {index:02d}/24 {cutoff}: positive={item['positive_rows']:,}")
    extra_meta: list[dict[str, Any]] = []
    for index, cutoff in enumerate(E72.EXTRA_CUTOFFS, 1):
        item = E72.cutoff_positive_metadata(events, cutoff)
        extra_meta.append(item)
        log(f"EXTRA {index:02d}/13 {cutoff}: positive={item['positive_rows']:,}")

    validation = E72.load_validation_frames(columns)
    predictions = {
        arm: {name: np.full(len(validation[name]["uid"]), np.nan, float) for name in E72.FOLD_NAMES}
        for arm in ("CLEAN", "FRESH", "VOL")
    }
    design_all: dict[int, dict[str, np.ndarray]] = {}
    fit_audit: list[dict[str, Any]] = []
    vol_hashes: dict[str, str] = {}

    for donor_side in (0, 1):
        log(f"Stage 0: assemble donor side {donor_side}")
        data = E72.assemble_side(donor_side, clean_meta, extra_meta, columns)
        n_clean = len(data["y_clean"])
        n_extra = len(data["y_extra"])
        early = np.flatnonzero(data["slot_clean"] < max(1, len(clean_cutoffs) // 3))
        vol_draw = np.random.default_rng(42).choice(early, size=n_extra, replace=True)
        vol_hashes[str(donor_side)] = hashlib.sha256(np.ascontiguousarray(vol_draw).view(np.uint8)).hexdigest()
        builders = {
            "CLEAN": lambda: (data["X_clean"], data["y_clean"]),
            "FRESH": lambda: (
                np.concatenate([data["X_clean"], data["X_extra"]]),
                np.concatenate([data["y_clean"], data["y_extra"]]),
            ),
            "VOL": lambda: (
                np.concatenate([data["X_clean"], data["X_clean"][vol_draw]]),
                np.concatenate([data["y_clean"], data["y_clean"][vol_draw]]),
            ),
        }
        design_all[donor_side] = {}
        for arm in ("CLEAN", "FRESH", "VOL"):
            X_train, y_train = builders[arm]()
            expected = n_clean if arm == "CLEAN" else n_clean + n_extra
            if len(y_train) != expected:
                raise AssertionError(f"{arm} row matching failed: {len(y_train)} != {expected}")
            fit_started = time.time()
            log(f"fit seed=42 donor={donor_side} arm={arm} rows={len(y_train):,}")
            model = fit_lgb(X_train, y_train, 42)
            fit_seconds = time.time() - fit_started
            fit_audit.append({
                "seed": 42,
                "donor_side": donor_side,
                "recipient_side": 1 - donor_side,
                "arm": arm,
                "rows": int(len(y_train)),
                "rounds": int(E72.LGB_ROUNDS),
                "seconds": fit_seconds,
            })
            for fold_name in E72.FOLD_NAMES:
                recipient = validation[fold_name]["side"] == (1 - donor_side)
                mu_recipient = np.asarray(model.predict(validation[fold_name]["X"][recipient]), float)
                predictions[arm][fold_name][recipient] = mu_recipient
            if arm in {"FRESH", "VOL"}:
                latest_X = validation[DESIGN_FOLD]["X"]
                design_all[donor_side][arm] = np.asarray(model.predict(latest_X), float)
            del model
            if arm != "CLEAN":
                del X_train, y_train
            gc.collect()
            log(f"finished seed=42 donor={donor_side} arm={arm} in {fit_seconds:.1f}s")
        del data
        gc.collect()

    for arm in predictions:
        for fold_name, values in predictions[arm].items():
            if not np.isfinite(values).all():
                raise AssertionError(f"incomplete cross-fit predictions: {arm}/{fold_name}")

    # Exact EXP072 parity chain: arm-specific mu_arm-mu_clean, temporal donor-fold
    # preprocessing on the first three folds, then multiplication by p_dist.
    arm_raw: dict[str, list[np.ndarray]] = {}
    arm_params: dict[str, dict[str, Any]] = {}
    arm_correction: dict[str, list[np.ndarray]] = {}
    for arm in ("FRESH", "VOL"):
        arm_raw[arm] = [predictions[arm][name] - predictions["CLEAN"][name] for name in E72.FOLD_NAMES]
        arm_params[arm] = E72.donor_preprocess(arm_raw[arm][:-1])
        arm_correction[arm] = [
            E72.apply_correction(arm_raw[arm][i], validation[name]["p_dist"], arm_params[arm])
            for i, name in enumerate(E72.FOLD_NAMES)
        ]

    latest = validation[DESIGN_FOLD]
    z_base = latest["z_base"]
    y = latest["y"]
    d_fresh = arm_correction["FRESH"][-1]
    d_vol = arm_correction["VOL"][-1]
    parity_scores = {
        "EXP037": score(y, z_base),
        "FRESH": score(y, z_base + d_fresh),
        "VOL": score(y, z_base + d_vol),
    }
    parity_errors = {name: parity_scores[name] - float(exp072_pilot["scores"][name]) for name in parity_scores}
    parity_pass = bool(max(abs(value) for value in parity_errors.values()) <= PARITY_TOL)

    # EXP073 chain: multiply first, fit winsor/center on the donor user-side,
    # transfer unchanged to the opposite side. No arm-specific preprocessing.
    d_semantic = np.full(len(y), np.nan, float)
    semantic_params: dict[str, dict[str, Any]] = {}
    raw_semantic_crossfit = np.full(len(y), np.nan, float)
    side = latest["side"]
    for donor_side in (0, 1):
        raw_all = latest["p_dist"] * (
            design_all[donor_side]["FRESH"] - design_all[donor_side]["VOL"]
        )
        donor = side == donor_side
        recipient = side == (1 - donor_side)
        params = donor_raw_preprocess(raw_all, donor)
        semantic_params[str(donor_side)] = params
        raw_semantic_crossfit[recipient] = raw_all[recipient]
        d_semantic[recipient] = apply_raw_preprocess(raw_all[recipient], params)
    if not np.isfinite(d_semantic).all() or not np.isfinite(raw_semantic_crossfit).all():
        raise AssertionError("incomplete EXP073 design correction")

    d_arm_difference = d_fresh - d_vol
    corr_fresh_vol = float(np.corrcoef(d_fresh, d_vol)[0, 1])
    semantic_rms = float(np.sqrt(np.mean(d_semantic * d_semantic)))
    arm_difference_rms = float(np.sqrt(np.mean(d_arm_difference * d_arm_difference)))
    z_target = np.log1p(y)
    residual_spec = z_target - z_base
    A = float(np.mean(residual_spec * d_semantic))
    Q = float(np.mean(d_semantic * d_semantic))
    alpha_star = float(A / Q) if Q > 0 else float("nan")
    baseline_theory = float(np.sqrt(np.mean(residual_spec * residual_spec)))
    optimal_theory = float(np.sqrt(max(0.0, np.mean(residual_spec * residual_spec) - A * A / Q))) if Q > 0 else float("nan")
    theoretical_delta = optimal_theory - baseline_theory
    theoretical_gain = -theoretical_delta

    curve_rows: list[dict[str, Any]] = []
    base_score = score(y, z_base)
    for alpha in ALPHA_GRID:
        candidate_score = score(y, z_base + float(alpha) * d_semantic)
        curve_rows.append({
            "fold": DESIGN_FOLD,
            "seed": 42,
            "alpha": float(alpha),
            "score": candidate_score,
            "delta_vs_exp037": candidate_score - base_score,
        })

    gates = {
        "parity": {"pass": parity_pass, "tolerance": PARITY_TOL, "max_abs_error": max(abs(v) for v in parity_errors.values())},
        "corr_fresh_vol": {"pass": bool(corr_fresh_vol >= 0.865), "value": corr_fresh_vol, "threshold": 0.865},
        "theoretical_optimal_gain": {"pass": bool(theoretical_gain >= 0.00020), "value": theoretical_gain, "threshold": 0.00020},
        "A_positive": {"pass": bool(A > 0), "value": A, "threshold": 0.0},
    }
    passed = bool(all(value["pass"] for value in gates.values()))
    verdict = "PASS_HEADROOM" if passed else "REJECT_HEADROOM"

    design_frame = pl.DataFrame({
        "user_id": latest["uid"],
        "target": y,
        "z_target": z_target,
        "z_exp037": z_base,
        "mu_clean": predictions["CLEAN"][DESIGN_FOLD],
        "mu_fresh": predictions["FRESH"][DESIGN_FOLD],
        "mu_vol": predictions["VOL"][DESIGN_FOLD],
        "p_dist": latest["p_dist"],
        "d_fresh": d_fresh,
        "d_vol": d_vol,
        "d_fresh_minus_vol": d_arm_difference,
        "d_semantic_seed42_raw": raw_semantic_crossfit,
        "d_semantic_seed42": d_semantic,
        "recipient_side": side,
    })
    design_frame.write_parquet(vector_path, compression="zstd")
    write_csv_new(OUT / "stage0_alpha_curve.csv", curve_rows)

    result = {
        "verdict": verdict,
        "fold": DESIGN_FOLD,
        "seed": 42,
        "parity": {
            "expected_scores": {name: exp072_pilot["scores"][name] for name in parity_scores},
            "reproduced_scores": parity_scores,
            "errors": parity_errors,
            "tolerance": PARITY_TOL,
            "pass": parity_pass,
            "exp072_config_sha256": sha256(EXP072_CONFIG),
            "exp072_pilot_sha256": sha256(EXP072_PILOT),
        },
        "headroom": {
            "corr_d_fresh_d_vol": corr_fresh_vol,
            "rms_d_fresh_minus_d_vol_arm_preprocessed": arm_difference_rms,
            "rms_d_semantic_direct_preprocessed": semantic_rms,
            "corr_direct_vs_arm_difference": float(np.corrcoef(d_semantic, d_arm_difference)[0, 1]),
            "A": A,
            "Q": Q,
            "analytic_alpha_star": alpha_star,
            "theoretical_optimal_delta": theoretical_delta,
            "theoretical_optimal_gain": theoretical_gain,
            "theoretical_baseline_rms": baseline_theory,
            "theoretical_optimal_rms": optimal_theory,
            "score_curve": curve_rows,
        },
        "gates": gates,
        "preprocessing": {
            "exp072_arm_parity": arm_params,
            "exp073_semantic_seed42_by_donor_side": semantic_params,
        },
        "fit_audit": fit_audit,
        "vol_draw_index_hash_by_donor_side": vol_hashes,
        "row_vector": {
            "path": str(vector_path),
            "rows": design_frame.height,
            "columns": design_frame.columns,
            "sha256": sha256(vector_path),
        },
        "leakage_audit": {
            "canonical_baseline_only": "pred_exp037",
            "clean_T_plus_30_le_V": bool(all(c + E72.dt.timedelta(days=30) <= E72.PILOT_FOLD for c in clean_cutoffs)),
            "positive_target_training_only": bool(all(np.all(item["z"] > 0) for item in clean_meta + extra_meta)),
            "frozen_227_features": True,
            "frozen_exp069_p_dist": True,
            "opposite_recipient_prediction": True,
            "donor_user_side_preprocessing": True,
            "vol_sample_fixed_rng42": True,
            "public_lb_used": False,
            "reconstructed_incumbent_oof_used": False,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json_new(metrics_path, result)
    if not passed:
        write_reject_headroom_report(result)
    log(json.dumps({"verdict": verdict, "gates": gates, "runtime_seconds": result["runtime_seconds"]}, indent=2))
    return result


def write_reject_headroom_report(result: dict[str, Any]) -> None:
    h = result["headroom"]
    failed = [name for name, gate in result["gates"].items() if not gate["pass"]]
    report = f"""# EXP073 — LWA FRESH−VOL Control-Variate Correction

## Verdict

**REJECT_HEADROOM**. Failed Stage 0 gate(s): `{', '.join(failed)}`. The protocol stops before the untouched confirmation fold, seed 43, full OOF, TEST inference, and submission creation.

## Stage 0 headroom (`2025-10-16`)

- EXP072 parity: **{'PASS' if result['parity']['pass'] else 'FAIL'}**; maximum score error `{max(abs(v) for v in result['parity']['errors'].values()):.3e}` (tolerance `2e-6`).
- `corr(d_fresh,d_vol)`: `{h['corr_d_fresh_d_vol']:.6f}`.
- RMS of arm-preprocessed `d_fresh-d_vol`: `{h['rms_d_fresh_minus_d_vol_arm_preprocessed']:.6f}`.
- RMS of direct EXP073 correction: `{h['rms_d_semantic_direct_preprocessed']:.6f}`.
- `A`: `{h['A']:+.9e}`; `Q`: `{h['Q']:.9e}`; analytic `alpha*`: `{h['analytic_alpha_star']:+.6f}`.
- Theoretical optimal delta: `{h['theoretical_optimal_delta']:+.9f}` (gain `{h['theoretical_optimal_gain']:+.9f}`).

The complete frozen alpha curve is in `stage0_alpha_curve.csv`; row-level parity and correction vectors are in `stage0_design_rows.parquet`.

## Confirmation and full canonical validation

Not run by gate. There is no confirmatory estimate, four-fold wCV/LOFO result, REAL-vs-null estimate, fold-sign count, bootstrap interval, standalone/EXP069 combination, or TEST diversity/span measurement.

## Production audit

TEST inference was not authorized. No `exp073_lwa_cv_TEST.*`, builder, or `SUBMIT_EXP073_LWA_CONTROL_VARIATE.csv` was created. Existing artifacts were not overwritten. Public-LB information and reconstructed incumbent OOF were not used.

## Estimated public gain

Not estimated: the headroom gate failed, and the public leaderboard is not a model-selection source.
"""
    path = OUT / "report.md"
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stage0"])
    args = parser.parse_args()
    {"stage0": stage0}[args.command]()


if __name__ == "__main__":
    main()
