from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OUT = ROOT / "research" / "new_directions" / "EXP069_BTYD05_FRESH1_PROD"
OOF_PATH = GEO / "gpt_pro_research_packet" / "06_ALIGNED_OOF.parquet"
FRESH_PATH = OLD / "artifacts" / "oof_FRESH_CONTRAST_MOE.npz"
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0])
ALPHA_GRID = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
BTYD_GRID = np.asarray([0.0, 0.025, 0.05, 0.10, 0.15])
BOOTSTRAPS = 500
SEED = 42


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def rmsle_raw(y: np.ndarray, z: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y) - np.maximum(z, 0.0)) ** 2)))


def calibrate(y: np.ndarray, z: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    ly = np.log1p(y)
    weights = np.ones(len(y), float) if weights is None else np.asarray(weights, float)
    denom = weights.sum()
    d = float(np.dot(weights, ly - z) / denom)
    for _ in range(25):
        active = z + d > 0
        wa = weights[active]
        if not len(wa) or wa.sum() == 0:
            break
        d_new = float(np.dot(wa, ly[active] - z[active]) / wa.sum())
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    err = ly - np.maximum(z + d, 0.0)
    return d, float(np.sqrt(np.dot(weights, err * err) / denom))


def evaluate(y: np.ndarray, z: np.ndarray, fold: np.ndarray) -> dict:
    raw, cal, offsets, sizes = [], [], [], []
    for f in FOLDS:
        m = fold == f
        off, score = calibrate(y[m], z[m])
        raw.append(rmsle_raw(y[m], z[m]))
        cal.append(score)
        offsets.append(off)
        sizes.append(int(m.sum()))
    raw = np.asarray(raw)
    cal = np.asarray(cal)
    return {
        "raw": raw,
        "cal": cal,
        "offset": np.asarray(offsets),
        "sizes": sizes,
        "wcv": float(FOLD_WEIGHTS @ cal / FOLD_WEIGHTS.sum()),
        "wcv_raw": float(FOLD_WEIGHTS @ raw / FOLD_WEIGHTS.sum()),
    }


def load_component(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def nested_scale(
    name: str,
    y: np.ndarray,
    fold: np.ndarray,
    base: np.ndarray,
    direction: np.ndarray,
    grid: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    rows: list[dict] = []
    held_scores = np.empty(4)
    base_scores = evaluate(y, base, fold)["cal"]
    for h, heldout in enumerate(FOLDS):
        donors = [i for i in range(4) if i != h]
        ranked: list[tuple[float, float, list[float]]] = []
        for scale in grid:
            donor_scores = []
            for i in donors:
                m = fold == FOLDS[i]
                donor_scores.append(calibrate(y[m], base[m] + float(scale) * direction[m])[1])
            donor_weights = FOLD_WEIGHTS[donors]
            score = float(donor_weights @ np.asarray(donor_scores) / donor_weights.sum())
            ranked.append((score, float(scale), donor_scores))
        selection_score, selected, donor_scores = min(ranked, key=lambda x: (x[0], x[1]))
        m = fold == heldout
        held_score = calibrate(y[m], base[m] + selected * direction[m])[1]
        held_scores[h] = held_score
        rows.append({
            "comparison": name,
            "heldout_fold": heldout,
            "donor_folds": json.dumps([FOLDS[i] for i in donors]),
            "grid": json.dumps(grid.tolist()),
            "selected_scale": selected,
            "selection_wcv": selection_score,
            "donor_scores": json.dumps(donor_scores),
            "heldout_score": held_score,
            "heldout_baseline_score": float(base_scores[h]),
            "heldout_delta": held_score - base_scores[h],
        })
    return rows, held_scores


def half_metrics(y: np.ndarray, fold: np.ndarray, side: np.ndarray, candidates: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for s in (0, 1):
        keep = side == s
        base = evaluate(y[keep], candidates["EXP037"][keep], fold[keep])
        for name, z in candidates.items():
            ev = evaluate(y[keep], z[keep], fold[keep])
            for i, f in enumerate(FOLDS):
                rows.append({"user_half": "A" if s == 0 else "B", "candidate": name, "fold": f,
                             "n": int(np.sum(keep & (fold == f))), "rmsle_cal": ev["cal"][i],
                             "delta_vs_exp037": ev["cal"][i] - base["cal"][i]})
            rows.append({"user_half": "A" if s == 0 else "B", "candidate": name, "fold": "wCV",
                         "n": int(keep.sum()), "rmsle_cal": ev["wcv"],
                         "delta_vs_exp037": ev["wcv"] - base["wcv"]})
    return pd.DataFrame(rows)


def bootstrap_metrics(
    y: np.ndarray,
    fold: np.ndarray,
    user_id: np.ndarray,
    candidates: dict[str, np.ndarray],
    point_deltas: dict[str, float],
) -> pd.DataFrame:
    unique_users, user_inverse = np.unique(user_id, return_inverse=True)
    fold_indices = [np.flatnonzero(fold == f) for f in FOLDS]
    rng = np.random.default_rng(SEED)
    names = [n for n in candidates if n != "EXP037"]
    samples = {n: np.empty(BOOTSTRAPS) for n in names}
    samples["REAL_FRESH_MINUS_VOL"] = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        counts = np.bincount(rng.integers(0, len(unique_users), len(unique_users)), minlength=len(unique_users))
        row_weights = counts[user_inverse].astype(float)
        fold_scores: dict[str, np.ndarray] = {}
        for name, z in candidates.items():
            scores = []
            for idx in fold_indices:
                scores.append(calibrate(y[idx], z[idx], row_weights[idx])[1])
            fold_scores[name] = np.asarray(scores)
        base = fold_scores["EXP037"]
        for name in names:
            samples[name][b] = float(FOLD_WEIGHTS @ (fold_scores[name] - base) / FOLD_WEIGHTS.sum())
        samples["REAL_FRESH_MINUS_VOL"][b] = samples["FRESH"][b] - samples["VOL"][b]
    rows = []
    for name, values in samples.items():
        point = (point_deltas["FRESH"] - point_deltas["VOL"]
                 if name == "REAL_FRESH_MINUS_VOL" else point_deltas[name])
        rows.append({
            "candidate": name,
            "n_bootstrap": BOOTSTRAPS,
            "point_delta_wcv": point,
            "p02_5": float(np.quantile(values, 0.025)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "p97_5": float(np.quantile(values, 0.975)),
            "p_delta_lt_0": float(np.mean(values < 0)),
        })
    return pd.DataFrame(rows)


def ridge_projection(
    fold: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    targets: dict[str, np.ndarray],
    alpha: float = 1.0,
) -> dict:
    result: dict[str, object] = {"ridge_alpha_fixed": alpha, "features": feature_names, "targets": {}}
    for target_name, target in targets.items():
        fold_rows = []
        all_resid, all_target = [], []
        for h, heldout in enumerate(FOLDS):
            test = fold == heldout
            train = ~test
            mean = X[train].mean(axis=0)
            scale = X[train].std(axis=0)
            scale[scale < 1e-12] = 1.0
            Xt = (X[train] - mean) / scale
            Xh = (X[test] - mean) / scale
            ym = float(target[train].mean())
            yc = target[train] - ym
            gram = Xt.T @ Xt
            coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), Xt.T @ yc)
            pred = ym + Xh @ coef
            resid = target[test] - pred
            var_target = float(np.var(target[test]))
            ratio = float(np.var(resid) / var_target) if var_target > 0 else float("nan")
            rms = float(np.sqrt(np.mean(resid * resid)))
            fold_rows.append({"fold": heldout, "n": int(test.sum()),
                              "unexplained_variance_ratio": ratio, "unexplained_RMS": rms})
            all_resid.append(resid)
            all_target.append(target[test])
        resid = np.concatenate(all_resid)
        target_all = np.concatenate(all_target)
        result["targets"][target_name] = {
            "folds": fold_rows,
            "pooled_unexplained_variance_ratio": float(np.var(resid) / np.var(target_all)),
            "pooled_unexplained_RMS": float(np.sqrt(np.mean(resid * resid))),
            "weighted_unexplained_variance_ratio": float(FOLD_WEIGHTS @ np.asarray([r["unexplained_variance_ratio"] for r in fold_rows]) / FOLD_WEIGHTS.sum()),
            "weighted_unexplained_RMS": float(FOLD_WEIGHTS @ np.asarray([r["unexplained_RMS"] for r in fold_rows]) / FOLD_WEIGHTS.sum()),
        }
    return result


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(OOF_PATH)
    fresh = np.load(FRESH_PATH, allow_pickle=False)
    user_id = frame["user_id"].to_numpy().astype(np.int64)
    fold = frame["fold"].to_numpy()
    y = frame["target"].to_numpy().astype(float)
    z0 = np.log1p(frame["pred_exp037"].to_numpy().astype(float))
    zb = np.log1p(frame["pred_btyd"].to_numpy().astype(float))
    d_fresh = fresh["fresh_processed_nested"].astype(float)
    d_vol = fresh["vol_processed_nested"].astype(float)
    raw_fresh = fresh["d_fresh"].astype(float)
    raw_vol = fresh["d_vol"].astype(float)
    z_btyd05 = 0.95 * z0 + 0.05 * zb
    z_fresh = z0 + d_fresh
    z_vol = z0 + d_vol
    z_combined = z_btyd05 + d_fresh

    required = ["user_id", "fold", "target", "pred_exp037", "pred_fresh_contrast", "pred_btyd", "pred_btyd05"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise AssertionError(f"missing aligned columns: {missing}")
    duplicate_keys = frame.select(["fold", "user_id"]).is_duplicated().sum()
    fold_sizes = [int(np.sum(fold == f)) for f in FOLDS]
    prediction_columns = [c for c in frame.columns if c.startswith("pred_")]
    predictions_ok = all(np.isfinite(frame[c].to_numpy()).all() and np.all(frame[c].to_numpy() >= 0) for c in prediction_columns)
    if frame.height != 770_616 or duplicate_keys != 0 or fold_sizes != [188_518, 191_025, 193_694, 197_379] or not predictions_ok:
        raise AssertionError("canonical row/schema audit failed")
    if not np.array_equal(user_id, fresh["uid"]) or not np.array_equal(fold, fresh["cutoff"]):
        raise AssertionError("FRESH row alignment failed")
    if not np.array_equal(y.astype(np.float32), fresh["y"]):
        raise AssertionError("FRESH targets differ")

    component_map = {
        "pred_cap": (OLD / "artifacts" / "oof_S1-E03a.npz", 0.10),
        "pred_unc": (OLD / "artifacts" / "oof_S1-E02.npz", 0.20),
        "pred_dist": (OLD / "artifacts" / "oof_S1-DIST.npz", 0.25),
        "pred_seq_avg3": (OLD / "artifacts" / "oof_SEQ-AVG3.npz", 0.225),
        "pred_etx_avg3": (OLD / "artifacts" / "oof_ETX-AVG3.npz", 0.225),
    }
    reconstructed = np.zeros(len(y), float)
    component_audits = []
    for column, (path, weight) in component_map.items():
        artifact = load_component(path)
        source_z = np.empty(len(y), float)
        source_fold = np.asarray(artifact["cutoff"], dtype="U10")
        source_user = np.asarray(artifact["user_id"], np.int64)
        source_y = np.asarray(artifact["y"], np.float32)
        source_raw_z = np.asarray(artifact["z"], float)
        for f in FOLDS:
            dst_idx = np.flatnonzero(fold == f)
            src_idx = np.flatnonzero(source_fold == f)
            dst_order = np.argsort(user_id[dst_idx])
            src_order = np.argsort(source_user[src_idx])
            if not np.array_equal(user_id[dst_idx][dst_order], source_user[src_idx][src_order]):
                raise AssertionError(f"{path.name} keys differ on {f}")
            if not np.array_equal(y[dst_idx][dst_order].astype(np.float32), source_y[src_idx][src_order]):
                raise AssertionError(f"{path.name} targets differ on {f}")
            source_z[dst_idx[dst_order]] = source_raw_z[src_idx][src_order]
        aligned_z = np.log1p(frame[column].to_numpy().astype(float))
        component_audits.append({"column": column, "path": str(path), "max_aligned_log_error": float(np.max(np.abs(source_z - aligned_z)))})
        reconstructed += weight * source_z

    exp037_reconstruction_error = float(np.max(np.abs(reconstructed - z0)))
    btyd05_error = float(np.max(np.abs(np.log1p(frame["pred_btyd05"].to_numpy().astype(float)) - z_btyd05)))
    fresh_saved_error = float(np.max(np.abs(np.log1p(frame["pred_fresh_contrast"].to_numpy().astype(float)) - z_fresh)))
    if max(exp037_reconstruction_error, btyd05_error, fresh_saved_error) > 1e-6:
        raise AssertionError("baseline reconstruction tolerance failed")

    candidates = {
        "EXP037": z0,
        "FRESH": z_fresh,
        "BTYD05": z_btyd05,
        "BTYD05_FRESH1_FIXED": z_combined,
        "VOL": z_vol,
    }
    evaluations = {name: evaluate(y, z, fold) for name, z in candidates.items()}
    base = evaluations["EXP037"]
    point_deltas = {name: ev["wcv"] - base["wcv"] for name, ev in evaluations.items()}

    metric_rows = []
    for name, ev in evaluations.items():
        for i, f in enumerate(FOLDS):
            metric_rows.append({"candidate": name, "fold": f, "n": ev["sizes"][i],
                                "rmsle_raw": ev["raw"][i], "rmsle_cal": ev["cal"][i],
                                "offset": ev["offset"][i], "delta_vs_exp037": ev["cal"][i] - base["cal"][i],
                                "improved": bool(ev["cal"][i] < base["cal"][i])})
        metric_rows.append({"candidate": name, "fold": "wCV", "n": len(y),
                            "rmsle_raw": ev["wcv_raw"], "rmsle_cal": ev["wcv"], "offset": np.nan,
                            "delta_vs_exp037": ev["wcv"] - base["wcv"],
                            "improved": bool(ev["wcv"] < base["wcv"])})
    pd.DataFrame(metric_rows).to_csv(OUT / "fold_metrics.csv", index=False)

    nested_rows = []
    nested_summaries = {}
    specs = [
        ("FRESH_STANDALONE_NESTED", z0, d_fresh, ALPHA_GRID),
        ("FRESH_ON_BTYD05_NESTED", z_btyd05, d_fresh, ALPHA_GRID),
        ("VOL_ON_EXP037_NESTED", z0, d_vol, ALPHA_GRID),
        ("BTYD_WEIGHT_DIAGNOSTIC_NESTED", z0, zb - z0, BTYD_GRID),
    ]
    for name, nb, direction, grid in specs:
        rows, held = nested_scale(name, y, fold, nb, direction, grid)
        nested_rows.extend(rows)
        nb_ev = evaluate(y, nb, fold)
        delta = held - nb_ev["cal"]
        nested_summaries[name] = {"heldout_scores": held.tolist(), "fold_deltas": delta.tolist(),
                                  "delta_wcv": float(FOLD_WEIGHTS @ delta / FOLD_WEIGHTS.sum()),
                                  "improved_folds": int(np.sum(delta < 0))}
    for weight in BTYD_GRID:
        ev = evaluate(y, z0 + float(weight) * (zb - z0), fold)
        nested_rows.append({"comparison": "BTYD_FIXED_DIAGNOSTIC_CURVE", "heldout_fold": "wCV",
                            "donor_folds": "", "grid": json.dumps(BTYD_GRID.tolist()),
                            "selected_scale": float(weight), "selection_wcv": ev["wcv"],
                            "donor_scores": json.dumps(ev["cal"].tolist()), "heldout_score": ev["wcv"],
                            "heldout_baseline_score": base["wcv"], "heldout_delta": ev["wcv"] - base["wcv"]})
    pd.DataFrame(nested_rows).to_csv(OUT / "nested_selection.csv", index=False)

    half = half_metrics(y, fold, fresh["group"].astype(np.int8), candidates)
    half.to_csv(OUT / "user_half_metrics.csv", index=False)

    boot = bootstrap_metrics(y, fold, user_id, candidates, point_deltas)
    boot.to_csv(OUT / "bootstrap_metrics.csv", index=False)

    fresh_vs_vol = []
    for i, f in enumerate(FOLDS):
        fd = evaluations["FRESH"]["cal"][i] - base["cal"][i]
        vd = evaluations["VOL"]["cal"][i] - base["cal"][i]
        fresh_vs_vol.append({"fold": f, "fresh_delta": fd, "vol_delta": vd, "real_minus_vol": fd - vd})
    fresh_vs_vol.append({"fold": "wCV", "fresh_delta": point_deltas["FRESH"], "vol_delta": point_deltas["VOL"],
                         "real_minus_vol": point_deltas["FRESH"] - point_deltas["VOL"]})
    pd.DataFrame(fresh_vs_vol).to_csv(OUT / "fresh_vs_vol.csv", index=False)

    z_candidate = z_combined
    p_candidate = np.expm1(np.maximum(z_candidate, 0.0))
    ly = np.log1p(y)
    candidate_ev = evaluations["BTYD05_FRESH1_FIXED"]
    diversity_rows = []
    for source in prediction_columns:
        z_source = np.log1p(frame[source].to_numpy().astype(float))
        p_source = frame[source].to_numpy().astype(float)
        source_ev = evaluate(y, z_source, fold)
        source_corr = z_source - z0
        new_corr = z_candidate - z0
        diversity_rows.append({
            "source": source,
            "pearson_prediction": float(np.corrcoef(p_candidate, p_source)[0, 1]),
            "pearson_log_prediction": float(np.corrcoef(z_candidate, z_source)[0, 1]),
            "residual_correlation": float(np.corrcoef(ly - z_candidate, ly - z_source)[0, 1]),
            "correction_correlation_vs_exp037": (float(np.corrcoef(new_corr, source_corr)[0, 1]) if np.std(source_corr) > 0 else np.nan),
            "rms_log_prediction_difference": float(np.sqrt(np.mean((z_candidate - z_source) ** 2))),
            "source_wcv": source_ev["wcv"],
            "candidate_wcv": candidate_ev["wcv"],
            "incremental_wcv_gain": candidate_ev["wcv"] - source_ev["wcv"],
        })
    pd.DataFrame(diversity_rows).to_csv(OUT / "diversity_oof.csv", index=False)

    projection_sources = [c for c in prediction_columns if c not in {"pred_exp037", "pred_fresh_contrast"}]
    X = np.column_stack([np.log1p(frame[c].to_numpy().astype(float)) - z0 for c in projection_sources])
    projection = ridge_projection(
        fold, X, projection_sources,
        {"FRESH_CORRECTION": d_fresh, "BTYD05_FRESH1_CORRECTION": z_combined - z0},
        alpha=1.0,
    )
    projection["exclusion_note"] = "pred_fresh_contrast is the new historical OOF direction itself and is excluded; BTYD/BTYD05 and every other aligned source are included."
    write_json("oof_projection_metrics.json", projection)

    bridge_fold_rows = []
    bridge_corr = []
    for h, heldout in enumerate(FOLDS):
        donors = fold != heldout
        held = fold == heldout
        q005, q995 = np.quantile(raw_fresh[donors], [0.005, 0.995])
        center = float(np.mean(np.clip(raw_fresh[donors], q005, q995)))
        corr = np.clip(raw_fresh[held], q005, q995) - center
        hist = d_fresh[held]
        bridge_corr.append(corr)
        bridge_fold_rows.append({"fold": heldout, "q005": float(q005), "q995": float(q995), "center": center,
                                 "rms_difference_vs_saved": float(np.sqrt(np.mean((corr - hist) ** 2))),
                                 "mean_difference_vs_saved": float(np.mean(corr - hist)),
                                 "std_difference_vs_saved": float(np.std(corr - hist))})
    bridge_corr = np.concatenate(bridge_corr)
    bridge_ev = evaluate(y, z0 + bridge_corr, fold)
    q005, q995 = np.quantile(raw_fresh, [0.005, 0.995])
    center = float(np.mean(np.clip(raw_fresh, q005, q995)))
    preprocessing = {
        "method": "predeclared all-honest-OOF production bridge",
        "order": ["raw FRESH-CLEAN", "winsorize to frozen q005/q995", "GLOBAL", "center using frozen OOF winsorized mean", "alpha=1"],
        "q005": float(q005), "q995": float(q995), "center": center,
        "lofo_emulation": bridge_fold_rows,
        "historical_wcv": evaluations["FRESH"]["wcv"],
        "bridge_wcv": bridge_ev["wcv"],
        "bridge_minus_historical_wcv": bridge_ev["wcv"] - evaluations["FRESH"]["wcv"],
        "historical_fold_signs": np.sign(evaluations["FRESH"]["cal"] - base["cal"]).astype(int).tolist(),
        "bridge_fold_signs": np.sign(bridge_ev["cal"] - base["cal"]).astype(int).tolist(),
        "pass": bool(abs(bridge_ev["wcv"] - evaluations["FRESH"]["wcv"]) <= 2e-5 and np.array_equal(np.sign(bridge_ev["cal"] - base["cal"]), np.sign(evaluations["FRESH"]["cal"] - base["cal"]))),
    }
    write_json("preprocessing_parameters.json", preprocessing)
    if not preprocessing["pass"]:
        raise RuntimeError("TECHNICAL_BLOCK: production preprocessing bridge failed")

    three = np.asarray([1.0, 2.0, 4.0])
    three_delta = float(three @ (evaluations["BTYD05_FRESH1_FIXED"]["cal"][:3] - base["cal"][:3]) / three.sum())
    historical_refs = {
        "baseline_wcv": 1.7475098625201952,
        "fresh_delta_wcv": -0.00022495613042393297,
        "btyd05_delta_wcv": -0.000320983015,
        "combined_delta_wcv": -0.000466939738,
    }
    observed = {
        "baseline_wcv": base["wcv"],
        "fresh_delta_wcv": point_deltas["FRESH"],
        "btyd05_delta_wcv": point_deltas["BTYD05"],
        "combined_delta_wcv": point_deltas["BTYD05_FRESH1_FIXED"],
    }
    discrepancies = {k: observed[k] - historical_refs[k] for k in observed}
    parity_pass = max(abs(v) for v in discrepancies.values()) <= 2e-5
    baseline_parity = {
        "status": "PASS" if parity_pass else "FAIL",
        "rows": frame.height,
        "unique_fold_user_keys": frame.select(["fold", "user_id"]).unique().height,
        "duplicate_keys": int(duplicate_keys),
        "fold_sizes": fold_sizes,
        "targets_match_fresh_exact_float32": bool(np.array_equal(y.astype(np.float32), fresh["y"])),
        "target_sha256_float32": array_hash(y.astype(np.float32)),
        "all_aligned_predictions_finite_nonnegative": predictions_ok,
        "component_audits": component_audits,
        "exp037_reconstruction_max_log_error": exp037_reconstruction_error,
        "btyd05_reconstruction_max_log_error": btyd05_error,
        "fresh_saved_reconstruction_max_log_error": fresh_saved_error,
        "historical_references": historical_refs,
        "observed": observed,
        "discrepancies": discrepancies,
        "three_fold_same_fold_audit_delta_1_2_4": three_delta,
        "three_fold_signs": np.sign(evaluations["BTYD05_FRESH1_FIXED"]["cal"][:3] - base["cal"][:3]).astype(int).tolist(),
    }
    write_json("baseline_parity.json", baseline_parity)
    if not parity_pass:
        raise RuntimeError("TECHNICAL_BLOCK: historical OOF parity failed")

    fixed_combo_delta = point_deltas["BTYD05_FRESH1_FIXED"]
    nested_combo_delta = nested_summaries["FRESH_ON_BTYD05_NESTED"]["delta_wcv"] + point_deltas["BTYD05"]
    half_wcv = half[(half["candidate"] == "BTYD05_FRESH1_FIXED") & (half["fold"] == "wCV")]
    interim_predictive = bool(
        min(fixed_combo_delta, nested_combo_delta) <= -0.00035
        and np.sum(evaluations["BTYD05_FRESH1_FIXED"]["cal"] - base["cal"] < 0) >= 3
        and evaluations["BTYD05_FRESH1_FIXED"]["cal"][-1] < base["cal"][-1]
        and point_deltas["FRESH"] - point_deltas["VOL"] <= -0.00010
        and np.all(half_wcv["delta_vs_exp037"].to_numpy() < 0)
    )
    config = {
        "experiment": "EXP069_BTYD05_FRESH1_PROD",
        "hypothesis": "Historical FRESH conditional-positive residual signal is complementary to fixed BTYD05 and can be produced for TEST as a new geometry basis vector.",
        "folds": FOLDS,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "alpha_grid": ALPHA_GRID.tolist(),
        "btyd_weight_grid": BTYD_GRID.tolist(),
        "primary_recipe": "0.95*z_exp037 + 0.05*z_btyd + d_fresh",
        "metric_prediction": "expm1(max(z,0))",
        "bootstrap_user_clusters": BOOTSTRAPS,
        "seed": SEED,
        "encoder": {
            "family": "SEQ-D3A-BASE-S42", "sequence_length": 365,
            "stored_channels": ["present", "cat", "buy", "ponly", "searches", "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv"],
            "generated_channels": ["avail", "dow_sin", "dow_cos"],
            "hidden": 64, "blocks": 8, "kernel": 3, "dropout": 0.1,
            "optimizer": "AdamW betas=(0.9,0.98)", "learning_rate": 0.003,
            "weight_decay": 0.01, "epochs": 4, "batch_size": 1024,
            "warmup_steps": 300, "workers": 3, "seed": 42,
            "depth_policy": "no depth augmentation; observed clean cutoff depth",
            "test_depth_clip": 289, "static_inputs": "none",
            "calendar_inputs": ["dow_sin", "dow_cos"],
            "normalization_cutoff": "2025-07-31",
            "training_cutoffs": "project 7-day clean cutoff grid through 2025-10-16",
        },
        "conditional_heads": {
            "architecture": "Linear(192,64)-GELU-Dropout(0.1)-Linear(64,1)",
            "initialization": "PyTorch default first layer; final weight zero; final bias zero",
            "learning_rate": 0.001, "weight_decay": 0.01, "batch_size": 8192,
            "epochs": 4, "head_seeds": [42, 43, 44], "target": "log1p(GMV30), y30>0 only",
            "crossfit": "splitmix64(user_id)&1; two donor-side head sets; TEST log prediction average",
        },
        "preprocessing": preprocessing,
        "production_regime_gates": {
            "processed_test_to_oof_std_ratio": [0.35, 2.0],
            "test_winsor_clipped_fraction_max": 0.10,
            "absolute_processed_test_mean_max": 0.05,
            "donor_side_raw_correction_correlation_min": 0.25,
            "donor_side_raw_correction_rms_difference_max": 0.10,
            "note": "Fixed before TEST conditional-head outputs were observed; broad plausibility gates account for two-side/three-seed averaging.",
        },
        "interim_predictive_gate": interim_predictive,
        "public_lb_used": False,
    }
    write_json("config.json", config)

    fresh_oof = pl.DataFrame({
        "user_id": user_id,
        "fold": fold,
        "target": y.astype(np.float32),
        "predict": np.expm1(np.maximum(z_fresh, 0.0)),
        "z_predict": z_fresh,
        "z_base": z0,
        "correction": d_fresh,
        "candidate_name": np.repeat("FRESH_CONDITIONAL", len(y)),
        "raw_correction": raw_fresh,
        "vol_correction": d_vol,
        "raw_vol_correction": raw_vol,
        "z_cond_clean": fresh["z_clean"].astype(float),
        "z_cond_fresh": fresh["z_fresh"].astype(float),
        "z_cond_vol": fresh["z_vol"].astype(float),
        "user_side": fresh["group"].astype(np.int8),
    })
    fresh_oof.write_parquet(OUT / "fresh_conditional_OOF.parquet", compression="zstd")

    runtime = {
        "oof_analysis_seconds": time.time() - started,
        "interim_predictive_gate": interim_predictive,
        "nested_summaries": nested_summaries,
        "fixed_deltas": point_deltas,
    }
    write_json("oof_analysis_summary.json", runtime)
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
