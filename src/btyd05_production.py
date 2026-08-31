"""EXP-050: exact production resolution for BTYD05_FRESH1.

This script performs no research or neural training.  It first audits whether
the frozen EXP-040 encoder trajectory exists for production.  If it does not,
it builds only the authorised BTYD05 fallback with the exact EXP-047
two-sided hash cross-fit at cutoff 2026-02-13.

Run from the repository root::

    python src/btyd05_production.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.block4_saf import BASE_TEST, L_STAR, _strongest_test, level_shift
from src.btyd_day_bgnbd import (
    K_MONETARY,
    NMAX,
    ORIGIN,
    bgnbd_count_distribution,
    cutoff_safety_audit,
    event_audit,
    fit_bgnbd,
    history_summary,
    metric_sum_moments,
    monetary_parameters,
    scored_monetary,
    sha256_array,
    splitmix64,
    user_group,
    user_universe,
)
from src.config import ARTIFACTS, ROOT, SAMPLE_SUBMIT, SUBMISSIONS


EXP_NUM = 50
EXP_ID = "BTYD05-PRODUCTION-RESOLUTION"
PREFIX = "BTYD05_PROD_EXP050"
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
PROD_CUTOFF = dt.date(2026, 2, 13)
OOF_PATH = ARTIFACTS / "BTYD_DAY_BGNBD_EXP047_V2" / "oof_raw.npz"
STRONGEST_SUBMISSION = SUBMISSIONS / "submission_STRONGEST_CURRENT.csv"
OUTPUT = SUBMISSIONS / "submission_BTYD05.csv"
FRESH_OUTPUT = SUBMISSIONS / "submission_BTYD05_FRESH1.csv"
EXPECTED_STRONGEST_SHA256 = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"
SUPPORT_RANGE = (0.6, 1.4)
QUANTILES = (0.0, 0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99,
             0.995, 0.999, 1.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def distribution(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise AssertionError("distribution input is empty or non-finite")
    return {
        "n": len(x), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "max": float(x.max()),
        "quantiles": {f"{q:g}": float(np.quantile(x, q)) for q in QUANTILES},
    }


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": path, "sha256": sha256_file(path), "val": obj["val"],
            "cfg": obj["cfg"]}


def fresh_parity_audit() -> dict[str, Any]:
    """Resolve EXP-040 parity without training or attempting an approximation."""
    fold_tags = ("0904", "0918", "1002", "1016")
    fold_paths = [ARTIFACTS / f"model_SEQ-D3A-BASE-S42-V{tag}.pt"
                  for tag in fold_tags]
    missing_fold = [str(p.resolve()) for p in fold_paths if not p.exists()]
    fold_meta = [load_checkpoint_metadata(p) for p in fold_paths if p.exists()]
    recipe_keys = ("hidden", "blocks", "kernel", "dropout", "batch", "chunk",
                   "lr", "wd", "epochs", "warmup", "seed", "workers", "compile",
                   "aug")
    recipes = [{k: m["cfg"].get(k) for k in recipe_keys} for m in fold_meta]
    cfg_equal = bool(recipes) and all(r == recipes[0] for r in recipes[1:])
    production_checkpoint = ARTIFACTS / "model_SEQ-D3A-BASE-S42-TEST.pt"
    saved_heads = sorted(str(p.resolve()) for p in ARTIFACTS.glob(
        "*FRESH_CONTRAST*head*.pt"))
    alternatives = sorted(str(p.resolve()) for p in ARTIFACTS.glob(
        "model_SEQ-*-TEST.pt"))
    exact = (not missing_fold and cfg_equal and production_checkpoint.exists())
    reasons = []
    if missing_fold:
        reasons.append("one or more exact EXP-040 fold encoders are missing")
    if fold_meta and not cfg_equal:
        reasons.append("EXP-040 fold encoder configurations differ")
    if not production_checkpoint.exists():
        reasons.append("exact SEQ-D3A-BASE seed-42 production checkpoint is absent")
    if not saved_heads:
        reasons.append("EXP-040 conditional head weights were not saved")
    if not exact:
        reasons.append("available TEST checkpoints have a different seed/model identity")
    audit = {
        "status": "PASS" if exact else "FAIL",
        "exact_encoder_family": "SEQ-D3A-BASE frozen TCN encoder used by EXP-040",
        "required_seed": 42,
        "required_training_recipe": recipes[0] if recipes else None,
        "required_feature_channels": "src.seq stored sequence channels; pooled [last,mean,max]",
        "required_depth_policy": 289,
        "required_donor_split": "splitmix64(user_id)&1; donor B->recipient A and donor A->recipient B",
        "required_heads": "CLEAN/VOL/FRESH conditional heads; GLOBAL uses FRESH-CLEAN",
        "required_preprocessing": "donor-fold 0.5/99.5% winsorization, then GLOBAL gate, then centering",
        "required_alpha": 1.0,
        "fold_checkpoints": fold_meta,
        "missing_fold_checkpoints": missing_fold,
        "production_checkpoint": production_checkpoint,
        "production_checkpoint_exists": production_checkpoint.exists(),
        "saved_conditional_head_weights": saved_heads,
        "incompatible_test_checkpoint_inventory": alternatives,
        "reasons": reasons,
        "action": "use exact FRESH" if exact else "STOP FRESH; build BTYD05 only",
    }
    write_json(RESULTS / "fresh_parity_audit.json", audit)
    return audit


def score_btyd(summary: pl.DataFrame) -> tuple[dict[str, np.ndarray], list[dict[str, Any]],
                                               list[dict[str, Any]]]:
    users = summary["user_id"].to_numpy().astype(np.int64)
    groups = summary["group"].to_numpy().astype(np.int8)
    n = len(users)
    T = (PROD_CUTOFF - ORIGIN).days
    outputs = {
        "user_id": users, "group": groups,
        "x": summary["x"].to_numpy().astype(np.int32),
        "t_x": summary["t_x"].to_numpy().astype(np.int32),
        "p_alive": np.empty(n, dtype=np.float64),
        "expected_count_30": np.empty(n, dtype=np.float64),
        "mu_u": np.empty(n, dtype=np.float64),
        "sigma_population": np.empty(n, dtype=np.float64),
        "z_btyd": np.empty(n, dtype=np.float64),
        "pmf_tail_30": np.empty(n, dtype=np.float64),
        "hash_side": np.empty(n, dtype=np.uint64),
    }
    fits: list[dict[str, Any]] = []
    monetary: list[dict[str, Any]] = []
    for donor_group in (0, 1):
        recipient_group = 1 - donor_group
        donor = summary.filter(pl.col("group") == donor_group)
        fit = fit_bgnbd(donor["x"].to_numpy(), donor["t_x"].to_numpy(), T,
                        PROD_CUTOFF.isoformat(), donor_group)
        fits.append(fit)
        pop = monetary_parameters(summary, donor_group)
        pop.update(fold=PROD_CUTOFF.isoformat(), recipient_group=recipient_group)
        monetary.append(pop)
        recipient = groups == recipient_group
        recipient_frame = summary.filter(pl.col("group") == recipient_group)
        if not np.array_equal(recipient_frame["user_id"].to_numpy(), users[recipient]):
            raise AssertionError("BTYD recipient order mismatch")
        params = np.asarray([fit["parameters"][k] for k in ("r", "alpha", "a", "b")])
        alive, pmf, expected = bgnbd_count_distribution(
            outputs["x"][recipient], outputs["t_x"][recipient], T, params)
        mu, sigma = scored_monetary(recipient_frame, pop)
        z = np.empty(int(recipient.sum()), dtype=np.float64)
        for start in range(0, len(z), 30_000):
            stop = min(start + 30_000, len(z))
            moments = metric_sum_moments(mu[start:stop], sigma[start:stop])
            z[start:stop] = np.sum(pmf[start:stop] * moments, axis=1)
        outputs["p_alive"][recipient] = alive
        outputs["expected_count_30"][recipient] = expected
        outputs["mu_u"][recipient] = mu
        outputs["sigma_population"][recipient] = sigma
        outputs["z_btyd"][recipient] = z
        outputs["pmf_tail_30"][recipient] = pmf[:, NMAX]
        outputs["hash_side"][recipient] = splitmix64(users[recipient])
    if not all(np.all(np.isfinite(v)) for k, v in outputs.items()
               if k not in {"user_id", "group", "x", "t_x", "hash_side"}):
        raise AssertionError("non-finite BTYD production output")
    if not np.array_equal(groups, user_group(users)):
        raise AssertionError("production split differs from splitmix64")
    return outputs, fits, monetary


def centered_oof_correction(oof: Any) -> np.ndarray:
    correction = 0.05 * (np.asarray(oof["z_btyd"], float)
                         - np.asarray(oof["z_strongest"], float))
    folds = np.asarray(oof["cutoff"], dtype="U10")
    centered = correction.copy()
    for fold in np.unique(folds):
        mask = folds == fold
        centered[mask] -= centered[mask].mean()
    return centered


def align_values(uid: np.ndarray, values: np.ndarray,
                 sample_uid: np.ndarray) -> np.ndarray:
    uid = np.asarray(uid)
    values = np.asarray(values)
    if len(uid) != len(values) or len(np.unique(uid)) != len(uid):
        raise AssertionError("invalid production user ids")
    order = np.argsort(uid)
    pos = np.searchsorted(uid[order], sample_uid)
    if np.any(pos >= len(uid)) or not np.array_equal(uid[order][pos], sample_uid):
        raise AssertionError("production users differ from sample submission")
    return values[order][pos]


def build_submission(uid: np.ndarray, z_raw: np.ndarray) -> dict[str, Any]:
    sample = pl.read_csv(SAMPLE_SUBMIT)
    reference = pl.read_csv(STRONGEST_SUBMISSION)
    if sample.columns != reference.columns or sample.columns != ["user_id", "predict"]:
        raise AssertionError("submission schema differs from project reference")
    sample_uid = sample["user_id"].to_numpy().astype(np.int64)
    if reference["user_id"].to_list() != sample["user_id"].to_list():
        raise AssertionError("STRONGEST_CURRENT order differs from sample")
    if sha256_file(STRONGEST_SUBMISSION) != EXPECTED_STRONGEST_SHA256:
        raise AssertionError("STRONGEST_CURRENT CSV hash changed")
    ordered_z = align_values(uid, z_raw, sample_uid)
    shift = level_shift(ordered_z, L_STAR)
    z_final = np.maximum(ordered_z + shift, 0.0)
    pred = np.maximum(np.expm1(z_final), 0.0)
    frame = pl.DataFrame({"user_id": sample_uid, "predict": pred})
    if frame.height != sample.height or frame["user_id"].n_unique() != sample.height:
        raise AssertionError("submission row or duplicate invariant failed")
    if not np.all(np.isfinite(pred)) or np.any(pred < 0):
        raise AssertionError("submission contains invalid predictions")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(OUTPUT, float_precision=6)
    disk = pl.read_csv(OUTPUT)
    disk_pred = disk["predict"].to_numpy()
    mean_level = float(np.log1p(disk_pred).mean())
    if disk.columns != reference.columns or disk.height != reference.height:
        raise AssertionError("written submission schema or row count changed")
    if disk["user_id"].to_list() != reference["user_id"].to_list():
        raise AssertionError("written submission user order changed")
    if abs(mean_level - L_STAR) > 1e-6:
        raise AssertionError("written submission misses target level")
    return {
        "path": OUTPUT, "sha256": sha256_file(OUTPUT), "rows": disk.height,
        "columns": disk.columns, "level_shift": shift,
        "mean_log1p_disk": mean_level, "zero_fraction": float(np.mean(disk_pred == 0)),
        "prediction": distribution(disk_pred), "duplicate_users": 0,
        "missing_users": 0, "finite": True, "nonnegative": True,
        "reference_path": STRONGEST_SUBMISSION,
        "reference_sha256": EXPECTED_STRONGEST_SHA256,
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if FRESH_OUTPUT.exists():
        raise AssertionError("unexpected combined submission exists before parity resolution")
    fresh = fresh_parity_audit()
    if fresh["status"] == "PASS":
        raise NotImplementedError(
            "exact FRESH checkpoint unexpectedly exists; this fixed runner must be extended "
            "with the registered EXP-040 conditional-head production path before submission")

    audit = event_audit()
    universe = user_universe()
    sample = pl.read_csv(SAMPLE_SUBMIT)
    sample_uid = sample["user_id"].to_numpy().astype(np.int64)
    if universe.height != 250_000 or sample.height != 250_000:
        raise AssertionError("production universe row count changed")
    if not np.array_equal(universe["user_id"].to_numpy(), np.sort(sample_uid)):
        raise AssertionError("raw-data universe differs from sample submission")
    duplicate = int(audit["duplicate_user_days"]["groups"])
    summary = history_summary(PROD_CUTOFF, universe, duplicate)
    safety = cutoff_safety_audit(PROD_CUTOFF, summary, universe)
    try:
        production, fits, monetary = score_btyd(summary)
    except RuntimeError as exc:
        prefix = "TECHNICAL_FAIL_UNSTABLE_MLE: "
        detail = str(exc)
        unstable_fit = json.loads(detail[len(prefix):]) if detail.startswith(prefix) else {
            "error": detail}
        blocker = {
            "status": "FAIL_UNSTABLE_MLE",
            "cutoff": PROD_CUTOFF,
            "origin": ORIGIN,
            "history_inclusive_max": PROD_CUTOFF,
            "event_audit": audit,
            "cutoff_safety": safety,
            "failed_fit": unstable_fit,
            "action": "fail-fast; no BTYD predictions, support composition, or submission",
            "forbidden_rescues_not_run": [
                "different optimizer starts", "relaxed stability gates", "parameter selection",
                "fold-fit averaging", "latest-fold parameters", "full-population refit",
                "different BTYD family", "weight tuning"],
            "artifact_hashes": {
                "raw_train": sha256_file(ROOT / "data" / "raw" / "train.parquet"),
                "sample_submit": sha256_file(SAMPLE_SUBMIT),
                "oof_raw": sha256_file(OOF_PATH),
            },
        }
        write_json(RESULTS / "production_support.json", blocker)
        summary_out = {
            "experiment": EXP_ID,
            "validation_status": "PREFERRED from EXP-049; no new validation",
            "fresh_parity_status": fresh["status"],
            "btyd_production_status": "FAIL_UNSTABLE_MLE",
            "test_support_status": "NOT_REACHED",
            "submission_status": "NOT_CREATED",
            "failed_donor_group": unstable_fit.get("donor_group"),
            "reasons": ["EXP-047 numerical stability gates failed at production cutoff"],
        }
        write_json(RESULTS / "summary.json", summary_out)
        print(json.dumps(jsonable(summary_out), ensure_ascii=False, indent=2))
        return
    uid_strongest, z_strongest = _strongest_test()
    if not np.array_equal(production["user_id"], uid_strongest):
        order = np.argsort(uid_strongest)
        pos = np.searchsorted(uid_strongest[order], production["user_id"])
        if (np.any(pos >= len(order))
                or not np.array_equal(uid_strongest[order][pos], production["user_id"])):
            raise AssertionError("STRONGEST_CURRENT and BTYD user sets differ")
        z_strongest = z_strongest[order][pos]
    z_btyd = production["z_btyd"]
    correction_test = 0.05 * (z_btyd - z_strongest)
    correction_test_centered = correction_test - correction_test.mean()
    oof = np.load(OOF_PATH, allow_pickle=False)
    correction_oof_centered = centered_oof_correction(oof)
    var_oof = float(np.var(correction_oof_centered))
    var_test = float(np.var(correction_test_centered))
    variance_ratio = var_test / var_oof

    oof_p_alive = np.asarray(oof["p_alive"], float)
    oof_count = np.asarray(oof["expected_count_30"], float)
    oof_btyd = np.asarray(oof["z_btyd"], float)
    support_reasons = []
    if not all(f["stable"] for f in fits):
        support_reasons.append("one or more production BG/NBD fits are unstable")
    if not (SUPPORT_RANGE[0] <= variance_ratio <= SUPPORT_RANGE[1]):
        support_reasons.append(
            f"correction variance ratio {variance_ratio:.6f} is outside {SUPPORT_RANGE}")
    if not np.all(np.isfinite(z_btyd)):
        support_reasons.append("BTYD predictions are not finite")
    if len(np.unique(production["user_id"])) != len(production["user_id"]):
        support_reasons.append("duplicate production users")

    fit_rows = []
    for fit in fits:
        best = fit["starts"][fit["best_start_index"]]
        fit_rows.append({
            "donor_group": fit["donor_group"], **fit["parameters"],
            "gradient_norm": best["gradient_norm"],
            "mean_nll_spread": fit["mean_nll_spread"],
            "max_log_parameter_spread": fit["max_log_parameter_spread"],
            "stable": fit["stable"], "n_users": fit["n_users"],
        })
    diagnostics = {
        "status": "PASS" if not support_reasons else "FAIL",
        "support_range": SUPPORT_RANGE, "reasons": support_reasons,
        "cutoff": PROD_CUTOFF, "origin": ORIGIN, "history_inclusive_max": PROD_CUTOFF,
        "event_audit": audit, "cutoff_safety": safety,
        "user_alignment": {
            "rows": len(production["user_id"]), "unique": len(np.unique(production["user_id"])),
            "sample_set_exact": True, "strongest_set_exact": True,
            "group0": int(np.sum(production["group"] == 0)),
            "group1": int(np.sum(production["group"] == 1)),
            "hash_side_exact": bool(np.array_equal(
                production["hash_side"], splitmix64(production["user_id"]))),
        },
        "bgnbd_fits": fit_rows, "monetary": monetary,
        "p_alive": {"oof": distribution(oof_p_alive),
                    "test": distribution(production["p_alive"])},
        "expected_count_30": {"oof": distribution(oof_count),
                              "test": distribution(production["expected_count_30"])},
        "z_btyd": {"oof": distribution(oof_btyd),
                   "test": distribution(z_btyd)},
        "correction": {
            "definition": "0.05*(z_BTYD-z_STRONGEST), centered per OOF fold / once on test",
            "oof": distribution(correction_oof_centered),
            "test": distribution(correction_test_centered),
            "variance_oof": var_oof, "variance_test": var_test,
            "variance_ratio": variance_ratio,
        },
        "extremes_and_clipping": {
            "direct_z_btyd_clip_fraction": 0.0,
            "mu_below_qmc_grid_fraction": float(np.mean(production["mu_u"] < -1.0)),
            "mu_above_qmc_grid_fraction": float(np.mean(production["mu_u"] > 9.0)),
            "sigma_below_qmc_grid_fraction": float(np.mean(production["sigma_population"] < .2)),
            "sigma_above_qmc_grid_fraction": float(np.mean(production["sigma_population"] > 3.0)),
            "mean_pmf_tail_at_cap30": float(production["pmf_tail_30"].mean()),
            "fraction_pmf_tail_gt_1e_6": float(np.mean(production["pmf_tail_30"] > 1e-6)),
        },
        "artifact_hashes": {
            "oof_raw": sha256_file(OOF_PATH),
            "raw_train": sha256_file(Path(audit.get("source_path", ROOT / "data/raw/train.parquet")))
            if audit.get("source_path") else sha256_file(ROOT / "data" / "raw" / "train.parquet"),
            "sample_submit": sha256_file(SAMPLE_SUBMIT),
            "strongest_components": {
                name: {"z": sha256_file(ARTIFACTS / f"ztest_{name}.npy"),
                       "uid": sha256_file(ARTIFACTS / f"uid_{name}.npy")}
                for name in BASE_TEST
            },
        },
    }
    write_json(RESULTS / "production_support.json", diagnostics)
    write_json(RESULTS / "btyd_fit_details.json", fits)
    write_json(RESULTS / "btyd_monetary.json", monetary)
    np.savez_compressed(
        RUN_DIR / "test_raw.npz", **production,
        z_strongest=z_strongest, correction_raw=correction_test,
        correction_centered=correction_test_centered)

    if diagnostics["status"] != "PASS":
        summary_out = {
            "experiment": EXP_ID, "fresh_parity": fresh["status"],
            "btyd_production": "PASS", "test_support": diagnostics["status"],
            "submission": None, "reasons": support_reasons,
        }
        write_json(RESULTS / "summary.json", summary_out)
        print(json.dumps(jsonable(summary_out), ensure_ascii=False, indent=2))
        return

    candidate_raw = 0.95 * z_strongest + 0.05 * z_btyd
    submission = build_submission(production["user_id"], candidate_raw)
    submission["raw_candidate_sha256"] = sha256_array(candidate_raw)
    submission["raw_btyd_sha256"] = sha256_array(z_btyd)
    submission["raw_strongest_sha256"] = sha256_array(z_strongest)
    write_json(RESULTS / "submission_verification.json", submission)
    summary_out = {
        "experiment": EXP_ID,
        "validation_status": "PREFERRED from EXP-049; no new validation",
        "fresh_parity_status": fresh["status"],
        "btyd_production_status": "PASS",
        "test_support_status": diagnostics["status"],
        "submission_status": "CREATED_BTYD05_ONLY",
        "variance_ratio": variance_ratio,
        "recipe": "0.95*STRONGEST_CURRENT + 0.05*BTYD; final level 2.3293",
        "submission": submission,
    }
    write_json(RESULTS / "summary.json", summary_out)
    print(json.dumps(jsonable(summary_out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
