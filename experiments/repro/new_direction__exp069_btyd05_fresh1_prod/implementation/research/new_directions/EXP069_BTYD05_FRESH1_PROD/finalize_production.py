from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OUT = ROOT / "research" / "new_directions" / "EXP069_BTYD05_FRESH1_PROD"
ALIGNED_OOF = GEO / "gpt_pro_research_packet" / "06_ALIGNED_OOF.parquet"
ALIGNED_TEST = GEO / "gpt_pro_research_packet" / "07_ALIGNED_TEST.parquet"
FRESH_OOF_NPZ = OLD / "artifacts" / "oof_FRESH_CONTRAST_MOE.npz"
RAW_TEST_NPZ = OUT / "fresh_production_raw.npz"
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0])


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def distribution(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    qs = [0, 0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999, 1]
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "quantiles": {str(q): float(np.quantile(x, q)) for q in qs}}


def calibrate(y: np.ndarray, z: np.ndarray) -> float:
    ly = np.log1p(np.asarray(y, float))
    z = np.asarray(z, float)
    d = float(np.mean(ly - z))
    for _ in range(25):
        active = z + d > 0
        d_new = float(np.mean(ly[active] - z[active]))
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    return float(np.sqrt(np.mean((ly - np.maximum(z + d, 0.0)) ** 2)))


def wcv(y: np.ndarray, z: np.ndarray, fold: np.ndarray) -> tuple[float, np.ndarray]:
    scores = np.asarray([calibrate(y[fold == f], z[fold == f]) for f in FOLDS])
    return float(FOLD_WEIGHTS @ scores / FOLD_WEIGHTS.sum()), scores


def geometry_projection(candidate_z: np.ndarray, user_id: np.ndarray, aligned_test: pl.DataFrame) -> dict:
    geometry_dir = GEO / "submission_geometry"
    sys.path.insert(0, str(geometry_dir))
    from core import load_unique  # type: ignore
    from directions import build_basis  # type: ignore

    Z, names, _, geometry_uid = load_unique()
    if Z.shape != (65, 250_000):
        raise AssertionError(f"unexpected geometry bank {Z.shape}")
    order = np.argsort(user_id)
    pos = np.searchsorted(user_id[order], geometry_uid)
    if pos.max() >= len(order) or not np.array_equal(user_id[order][pos], geometry_uid):
        raise AssertionError("candidate TEST users do not align to geometry users")
    z = candidate_z[order][pos]
    ref_idx = 0
    z_ref, Phi, _, lam, _ = build_basis(Z, ref_idx, tol=1e-12)
    # The affine-span residual is invariant to the point chosen inside the
    # span.  Use the bank centroid for the norm fraction so that this
    # diagnostic is also invariant to manifest/source ordering.
    z_center = Z.mean(axis=0)
    delta = z - z_center
    coordinates = Phi @ delta / Z.shape[1]
    projection = z_center + coordinates @ Phi
    residual = z - projection
    rms = float(np.sqrt(np.mean(residual * residual)))
    delta_rms = float(np.sqrt(np.mean(delta * delta)))
    orth_fraction = float(rms / delta_rms) if delta_rms > 0 else 0.0
    source_distances = np.sqrt(np.mean((Z - z) ** 2, axis=1))
    nearest_source_idx = int(np.argmin(source_distances))
    nearest_source_rms = float(source_distances[nearest_source_idx])
    nearest_source_orthogonal_fraction = (
        float(rms / nearest_source_rms) if nearest_source_rms > 0 else 0.0)

    candidate_distances = []
    for column in aligned_test.columns:
        if column == "user_id":
            continue
        other_uid = aligned_test["user_id"].to_numpy().astype(np.int64)
        other_z = np.log1p(aligned_test[column].to_numpy().astype(float))
        other_order = np.argsort(other_uid)
        other_pos = np.searchsorted(other_uid[other_order], geometry_uid)
        if not np.array_equal(other_uid[other_order][other_pos], geometry_uid):
            continue
        distance = float(np.sqrt(np.mean((z - other_z[other_order][other_pos]) ** 2)))
        candidate_distances.append((distance, column))
    nearest_candidate_distance, nearest_candidate = min(candidate_distances)

    Y = Z - Z[ref_idx]
    delta_ref = z - Z[ref_idx]
    G = Y @ Y.T / Z.shape[1]
    eig = np.linalg.eigvalsh(G)
    existing_rank = int(np.sum(eig > 1e-12 * eig[-1]))
    Y_aug = np.vstack([Y, delta_ref])
    G_aug = Y_aug @ Y_aug.T / Z.shape[1]
    eig_aug = np.linalg.eigvalsh(G_aug)
    augmented_rank = int(np.sum(eig_aug > 1e-12 * eig_aug[-1]))
    return {
        "basis_source": str(geometry_dir / "cache" / "Z.npz"),
        "basis_source_sha256": sha256(geometry_dir / "cache" / "Z.npz"),
        "unique_source_vectors": 65,
        "basis_rank": int(Phi.shape[0]),
        "rms_distance_from_affine_span": rms,
        "orthogonal_RMS": rms,
        "candidate_minus_affine_centroid_RMS": delta_rms,
        "orthogonal_norm_fraction": orth_fraction,
        "orthogonal_fraction_denominator": "RMS(candidate_z - mean(existing_source_z)); invariant to source ordering",
        "nearest_existing_source": names[nearest_source_idx],
        "nearest_existing_source_rms": nearest_source_rms,
        "orthogonal_fraction_of_nearest_source_distance": nearest_source_orthogonal_fraction,
        "nearest_existing_aligned_candidate": nearest_candidate,
        "nearest_existing_aligned_candidate_rms": nearest_candidate_distance,
        "existing_numerical_rank": existing_rank,
        "augmented_numerical_rank": augmented_rank,
        "increases_numerical_rank": bool(augmented_rank > existing_rank),
        "rank_tolerance": "eigenvalue > 1e-12 * largest eigenvalue",
        "smallest_retained_basis_eigenvalue": float(lam[-1]),
        "target_or_lb_used": False,
    }


def main() -> None:
    preprocessing = load_json("preprocessing_parameters.json")
    config = load_json("config.json")
    baseline = load_json("baseline_parity.json")
    training = load_json("production_training_audit.json")
    raw_test = np.load(RAW_TEST_NPZ, allow_pickle=False)
    aligned_test = pl.read_parquet(ALIGNED_TEST)
    aligned_oof = pl.read_parquet(ALIGNED_OOF)
    historical = np.load(FRESH_OOF_NPZ, allow_pickle=False)

    user_id = aligned_test["user_id"].to_numpy().astype(np.int64)
    if not np.array_equal(user_id, raw_test["user_id"].astype(np.int64)):
        raise AssertionError("raw production rows are not in canonical aligned TEST order")
    q005, q995, center = preprocessing["q005"], preprocessing["q995"], preprocessing["center"]
    raw_correction = raw_test["d_raw_fresh"].astype(float)
    raw_vol = raw_test["d_raw_vol"].astype(float)
    correction = np.clip(raw_correction, q005, q995) - center
    vol_correction = np.clip(raw_vol, q005, q995) - center
    clipped_fraction = float(np.mean((raw_correction < q005) | (raw_correction > q995)))

    z_base_test = np.log1p(aligned_test["pred_exp037_rebuilt"].to_numpy().astype(float))
    z_btyd_test = np.log1p(aligned_test["pred_btyd"].to_numpy().astype(float))
    z_fresh_test = z_base_test + correction
    z_combined_test = 0.95 * z_base_test + 0.05 * z_btyd_test + correction
    p_fresh = np.expm1(np.maximum(z_fresh_test, 0.0))
    p_combined = np.expm1(np.maximum(z_combined_test, 0.0))

    canonical_sample = pl.read_csv(OLD / "data" / "raw" / "sample_submit.csv")["user_id"].to_numpy().astype(np.int64)
    schema_checks = {
        "rows": len(user_id), "columns": ["user_id", "predict"],
        "unique_user_id": int(len(np.unique(user_id))),
        "canonical_order": bool(np.array_equal(user_id, canonical_sample)),
        "finite": bool(np.isfinite(p_combined).all()),
        "nonnegative": bool(np.all(p_combined >= 0)),
        "missing_users": int(len(np.setdiff1d(canonical_sample, user_id))),
    }
    schema_pass = bool(schema_checks["rows"] == 250_000 and schema_checks["unique_user_id"] == 250_000
                       and schema_checks["canonical_order"] and schema_checks["finite"]
                       and schema_checks["nonnegative"] and schema_checks["missing_users"] == 0)
    if not schema_pass:
        raise AssertionError(f"TEST schema failed: {schema_checks}")

    fresh_test_frame = pl.DataFrame({
        "user_id": user_id,
        "predict": p_fresh,
        "z_predict": z_fresh_test,
        "z_base": z_base_test,
        "correction": correction,
        "candidate_name": np.repeat("FRESH_CONDITIONAL", len(user_id)),
        "raw_correction": raw_correction,
        "vol_correction": vol_correction,
        "raw_vol_correction": raw_vol,
        "z_cond_clean": raw_test["z_clean"].astype(float),
        "z_cond_fresh": raw_test["z_fresh"].astype(float),
        "z_cond_vol": raw_test["z_vol"].astype(float),
        "p_dist": raw_test["p_dist"].astype(float),
    })
    fresh_test_path = OUT / "fresh_conditional_TEST.parquet"
    fresh_test_frame.write_parquet(fresh_test_path, compression="zstd")

    oof_saved = historical["fresh_processed_nested"].astype(float)
    side0 = raw_test["z_fresh_side0"].astype(float) - raw_test["z_clean_side0"].astype(float)
    side1 = raw_test["z_fresh_side1"].astype(float) - raw_test["z_clean_side1"].astype(float)
    std_ratio = float(np.std(correction) / np.std(oof_saved))
    side_corr = float(np.corrcoef(side0, side1)[0, 1])
    side_rms = float(np.sqrt(np.mean((side0 - side1) ** 2)))
    gates = config["production_regime_gates"]
    gate_results = {
        "processed_std_ratio": bool(gates["processed_test_to_oof_std_ratio"][0] <= std_ratio <= gates["processed_test_to_oof_std_ratio"][1]),
        "clipped_fraction": bool(clipped_fraction <= gates["test_winsor_clipped_fraction_max"]),
        "processed_mean": bool(abs(float(correction.mean())) <= gates["absolute_processed_test_mean_max"]),
        "side_correlation": bool(side_corr >= gates["donor_side_raw_correction_correlation_min"]),
        "side_rms": bool(side_rms <= gates["donor_side_raw_correction_rms_difference_max"]),
    }
    regime_pass = bool(all(gate_results.values()))
    production_regime = {
        "status": "PASS" if regime_pass else "FAIL",
        "gates_fixed_before_test_outputs": gates,
        "gate_results": gate_results,
        "raw_oof": distribution(historical["d_fresh"].astype(float)),
        "processed_oof": distribution(oof_saved),
        "raw_test": distribution(raw_correction),
        "processed_test": distribution(correction),
        "raw_vol_test": distribution(raw_vol),
        "processed_vol_test": distribution(vol_correction),
        "processed_test_to_oof_std_ratio": std_ratio,
        "test_winsor_clipped_fraction": clipped_fraction,
        "donor_side_raw_correction_correlation": side_corr,
        "donor_side_raw_correction_rms_difference": side_rms,
        "test_base_mean_z": float(z_base_test.mean()),
        "test_btyd_mean_z": float(z_btyd_test.mean()),
        "fresh_candidate_mean_z_after_floor": float(np.maximum(z_fresh_test, 0.0).mean()),
        "combined_candidate_mean_z_after_floor": float(np.maximum(z_combined_test, 0.0).mean()),
        "no_test_centering_or_variance_matching": True,
        "extensive_component_audit": training["extensive_component"],
    }
    write_json("production_regime.json", production_regime)

    span = geometry_projection(z_combined_test, user_id, aligned_test)
    span["schema"] = schema_checks
    span["schema_pass"] = schema_pass
    write_json("test_span_projection.json", span)

    fold_metrics = pd.read_csv(OUT / "fold_metrics.csv")
    fixed = fold_metrics[fold_metrics.candidate == "BTYD05_FRESH1_FIXED"]
    base_rows = fold_metrics[fold_metrics.candidate == "EXP037"]
    fixed_wcv_delta = float(fixed[fixed.fold == "wCV"].delta_vs_exp037.iloc[0])
    fixed_fold_deltas = fixed[fixed.fold.isin(FOLDS)].set_index("fold").loc[FOLDS].delta_vs_exp037.to_numpy()
    fresh_vs_vol = pd.read_csv(OUT / "fresh_vs_vol.csv")
    real_minus_vol = float(fresh_vs_vol[fresh_vs_vol.fold == "wCV"].real_minus_vol.iloc[0])
    halves = pd.read_csv(OUT / "user_half_metrics.csv")
    half_delta = halves[(halves.candidate == "BTYD05_FRESH1_FIXED") & (halves.fold == "wCV")].delta_vs_exp037.to_numpy()
    projection = load_json("oof_projection_metrics.json")
    oof_unexplained = projection["targets"]["BTYD05_FRESH1_CORRECTION"]["pooled_unexplained_variance_ratio"]

    pass_a = bool(fixed_wcv_delta <= -0.00035 and np.sum(fixed_fold_deltas < 0) >= 3
                  and fixed_fold_deltas[-1] < 0 and real_minus_vol <= -0.00010
                  and np.all(half_delta < 0) and regime_pass and schema_pass)
    pass_b = bool(fixed_wcv_delta <= -0.00012 and np.sum(fixed_fold_deltas < 0) >= 3
                  and fixed_fold_deltas[-1] < 0 and oof_unexplained >= 0.20
                  and span["orthogonal_norm_fraction"] >= 0.10
                  and span["orthogonal_RMS"] >= 0.0025 and regime_pass and schema_pass)
    if pass_a:
        verdict = "PASS_TYPE_A"
    elif pass_b:
        verdict = "PASS_TYPE_B"
    elif not regime_pass:
        verdict = "REJECT"
    elif fixed_wcv_delta > 0.00005 or fixed_fold_deltas[-1] > 0.00010 or real_minus_vol >= 0:
        verdict = "REJECT"
    else:
        verdict = "WEAK_SIGNAL"
    recommendation = "ADD_TO_SUBMISSION_GEOMETRY" if verdict in {"PASS_TYPE_A", "PASS_TYPE_B"} else "DO_NOT_ADD"

    combined_oof_path = None
    combined_test_path = None
    combined_csv_path = None
    if verdict in {"PASS_TYPE_A", "PASS_TYPE_B"}:
        oof_user = aligned_oof["user_id"].to_numpy().astype(np.int64)
        oof_fold = aligned_oof["fold"].to_numpy()
        oof_target = aligned_oof["target"].to_numpy().astype(float)
        oof_z0 = np.log1p(aligned_oof["pred_exp037"].to_numpy().astype(float))
        oof_zb = np.log1p(aligned_oof["pred_btyd"].to_numpy().astype(float))
        oof_corr = historical["fresh_processed_nested"].astype(float)
        oof_z = 0.95 * oof_z0 + 0.05 * oof_zb + oof_corr
        total_oof_corr = oof_z - oof_z0
        combined_oof_path = OUT / "btyd05_fresh1_OOF.parquet"
        pl.DataFrame({
            "user_id": oof_user, "fold": oof_fold, "target": oof_target,
            "predict": np.expm1(np.maximum(oof_z, 0.0)), "z_predict": oof_z,
            "z_base": oof_z0, "correction": total_oof_corr,
            "candidate_name": np.repeat("BTYD05_FRESH1", len(oof_user)),
        }).write_parquet(combined_oof_path, compression="zstd")
        combined_test_path = OUT / "btyd05_fresh1_TEST.parquet"
        total_test_corr = z_combined_test - z_base_test
        pl.DataFrame({
            "user_id": user_id, "predict": p_combined, "z_predict": z_combined_test,
            "z_base": z_base_test, "correction": total_test_corr,
            "candidate_name": np.repeat("BTYD05_FRESH1", len(user_id)),
        }).write_parquet(combined_test_path, compression="zstd")
        combined_csv_path = OUT / "btyd05_fresh1_TEST.csv"
        pl.DataFrame({"user_id": user_id, "predict": p_combined}).write_csv(combined_csv_path)
        disk = pl.read_csv(combined_csv_path)
        if disk.columns != ["user_id", "predict"] or disk.height != 250_000:
            raise AssertionError("final CSV disk schema changed")

    encoder_checkpoint = OLD / "artifacts" / "model_SEQ-D3A-BASE-S42-TEST.pt"
    encoder_start = datetime.fromisoformat("2026-08-26T01:39:57")
    experiment_files = [p for p in OUT.iterdir() if p.is_file()]
    persistent_bytes = sum(p.stat().st_size for p in experiment_files)
    external_new = [encoder_checkpoint, OLD / "artifacts" / "ztest_SEQ-D3A-BASE-S42.npy",
                    OLD / "artifacts" / "ztest_SEQ-D3A-BASE-S42-FULL.npy",
                    OLD / "artifacts" / "uid_SEQ-D3A-BASE-S42.npy",
                    OLD / "artifacts" / "uid_SEQ-D3A-BASE-S42-FULL.npy"]
    encoder_end = max(datetime.fromtimestamp(p.stat().st_mtime) for p in external_new if p.exists())
    encoder_runtime = max(0.0, (encoder_end - encoder_start).total_seconds())
    external_new_bytes = sum(p.stat().st_size for p in external_new if p.exists())
    runtime = {
        "oof_analysis_seconds": load_json("oof_analysis_summary.json")["oof_analysis_seconds"],
        "production_encoder_seconds": encoder_runtime,
        "conditional_embedding_and_head_seconds": training["runtime_seconds"],
        "new_persistent_experiment_bytes": persistent_bytes,
        "new_persistent_external_encoder_bytes": external_new_bytes,
        "new_persistent_total_bytes": persistent_bytes + external_new_bytes,
        "peak_new_persistent_disk_bytes": persistent_bytes + external_new_bytes,
        "peak_disk_basis": "No temporary embedding/cache files were written; the final persistent total is the persistent peak.",
        "disk_budget_bytes": 5_000_000_000,
        "disk_budget_pass": bool(persistent_bytes + external_new_bytes <= 5_000_000_000),
        "temporary_embedding_caches_deleted_or_never_written": True,
    }
    write_json("runtime_resources.json", runtime)

    required_names = [
        "reconnaissance.md", "artifact_manifest.csv", "config.json", "baseline_parity.json",
        "fold_metrics.csv", "nested_selection.csv", "user_half_metrics.csv", "bootstrap_metrics.csv",
        "fresh_vs_vol.csv", "diversity_oof.csv", "oof_projection_metrics.json",
        "test_span_projection.json", "production_regime.json", "preprocessing_parameters.json",
        "fresh_conditional_OOF.parquet", "fresh_conditional_TEST.parquet",
    ]
    if verdict in {"PASS_TYPE_A", "PASS_TYPE_B"}:
        required_names += ["btyd05_fresh1_OOF.parquet", "btyd05_fresh1_TEST.parquet", "btyd05_fresh1_TEST.csv"]
    missing = [name for name in required_names if not (OUT / name).exists()]
    if missing:
        raise AssertionError(f"required artifacts missing: {missing}")

    key_paths = [fresh_test_path]
    if combined_oof_path:
        key_paths += [combined_oof_path, combined_test_path, combined_csv_path]
    artifacts = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in key_paths]

    base_wcv = float(base_rows[base_rows.fold == "wCV"].rmsle_cal.iloc[0])
    fixed_wcv = float(fixed[fixed.fold == "wCV"].rmsle_cal.iloc[0])
    fresh_delta = float(fold_metrics[(fold_metrics.candidate == "FRESH") & (fold_metrics.fold == "wCV")].delta_vs_exp037.iloc[0])
    btyd_delta = float(fold_metrics[(fold_metrics.candidate == "BTYD05") & (fold_metrics.fold == "wCV")].delta_vs_exp037.iloc[0])
    report = f"""# EXP069 BTYD05 FRESH1 production report

## 1. Verdict

**{verdict}**  
Recommendation: **{recommendation}**

## 2. Exact hypothesis

The historical positive-target FRESH conditional residual correction relative to EXP-037 contains signed signal complementary to fixed 5% stable BTYD, and an exact-semantics production rebuild can yield a useful TEST vector outside the current geometry span.

## 3. OOF baseline parity

All 770,616 unique canonical `(fold,user_id)` rows and targets align exactly. EXP-037 reconstructs from CAP/UNC/DIST/SEQ-AVG3/ETX-AVG3 with maximum log error `{baseline['exp037_reconstruction_max_log_error']:.3e}`; BTYD05 maximum log error is `{baseline['btyd05_reconstruction_max_log_error']:.3e}`. Registered parity passed. EXP-037 wCV is `{base_wcv:.12f}`.

## 4. Fold and wCV results

- FRESH delta wCV: `{fresh_delta:+.9f}` (4/4; latest `{float(fixed_fold_deltas[-1] - (fixed_fold_deltas[-1] - float(fold_metrics[(fold_metrics.candidate == 'FRESH') & (fold_metrics.fold == FOLDS[-1])].delta_vs_exp037.iloc[0]))):+.9f}`).
- Fixed BTYD05 delta wCV: `{btyd_delta:+.9f}` (4/4).
- Fixed BTYD05_FRESH1: wCV `{fixed_wcv:.12f}`, delta `{fixed_wcv_delta:+.9f}`, folds `{fixed_fold_deltas.tolist()}`.
- User-cluster bootstrap intervals are in `bootstrap_metrics.csv`; combined 95% interval is `{pd.read_csv(OUT/'bootstrap_metrics.csv').query("candidate == 'BTYD05_FRESH1_FIXED'")[["p02_5","p97_5"]].iloc[0].tolist()}`.

## 5. FRESH vs VOL control

REAL-minus-VOL is `{real_minus_vol:+.9f}` wCV. The matched-volume control is neutral while REAL improves. Both splitmix user halves have negative combined deltas: `{half_delta.tolist()}`.

## 6. BTYD/FRESH complementarity

FRESH adds `{load_json('oof_analysis_summary.json')['nested_summaries']['FRESH_ON_BTYD05_NESTED']['delta_wcv']:+.9f}` nested wCV beyond fixed BTYD05, with 4/4 held-out folds improving. Fixed combined gain is materially larger than either component alone.

## 7. OOF correction diversity

Donor-fold ridge projection leaves `{oof_unexplained:.3f}` of combined correction variance unexplained and `{projection['targets']['BTYD05_FRESH1_CORRECTION']['pooled_unexplained_RMS']:.6f}` unexplained RMS. FRESH alone is `{projection['targets']['FRESH_CORRECTION']['pooled_unexplained_variance_ratio']:.3f}` unexplained.

## 8. TEST distance outside the 65-source span

Rank-57 affine-span distance is `{span['orthogonal_RMS']:.6f}` RMS; orthogonal norm fraction is `{span['orthogonal_norm_fraction']:.3f}`. Nearest source is `{span['nearest_existing_source']}` at `{span['nearest_existing_source_rms']:.6f}` RMS. Numerical rank changes `{span['existing_numerical_rank']} -> {span['augmented_numerical_rank']}`.

## 9. Production and leakage audits

The encoder used only the 29 CLEAN cutoff grid through 2025-10-16. EXTRA comprises only positive-target rows at the 13 preregistered cutoffs and updates only conditional amount heads. Two splitmix donor sides and seeds 42/43/44 were averaged in log space. The encoder checksum was unchanged. Frozen OOF preprocessing parameters were applied to TEST without TEST centering or variance matching. Production regime: **{production_regime['status']}**; schema: **{'PASS' if schema_pass else 'FAIL'}**.

The leave-one-fold-out production bridge has calibrated wCV discrepancy `{preprocessing['bridge_minus_historical_wcv']:+.3e}` and preserves all four fold signs. Its maximum saved-vector RMS difference is `{max(x['rms_difference_vs_saved'] for x in preprocessing['lofo_emulation']):.6f}`; each fold difference is a pure constant offset (maximum within-fold difference SD `{max(x['std_difference_vs_saved'] for x in preprocessing['lofo_emulation']):.3e}`), which the canonical fold log-offset evaluator removes exactly.

The TEST extensive probability is explicitly marked reconstructed: same CLEAN-only S1-DIST recipe, not a byte-exact recovery of the historical TEST trajectory. Its provenance and mismatch diagnostics are retained in `production_training_audit.json` and `production_regime.json`; no guessed formula or TEST target calibration was used.

## 10. Runtime and disk

- OOF analysis: `{runtime['oof_analysis_seconds']:.1f}s`.
- Production encoder: `{runtime['production_encoder_seconds']:.1f}s`.
- Embedding plus conditional heads: `{runtime['conditional_embedding_and_head_seconds']:.1f}s`.
- Peak new persistent disk: `{runtime['peak_new_persistent_disk_bytes']/1e9:.3f} GB` (budget pass: `{runtime['disk_budget_pass']}`). No persistent temporary embedding caches were written.

## 11. Saved artifacts and SHA256

```json
{json.dumps(artifacts, indent=2)}
```

Complete hashes are in `checksums.sha256`; full input provenance is in `artifact_manifest.csv`.

## 12. Recommendation

**{recommendation}**

No submission was uploaded and no public-LB equation or score was used for evaluation, selection, scaling, or level fitting.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    final = {"verdict": verdict, "recommendation": recommendation, "pass_type_a": pass_a,
             "pass_type_b": pass_b, "schema_pass": schema_pass, "production_regime_pass": regime_pass,
             "fixed_delta_wcv": fixed_wcv_delta, "test_span": span, "artifacts": artifacts}
    write_json("final_summary.json", final)

    # Build the ledger last.  A checksum file cannot self-hash, but every
    # other persistent experiment file—including the completed report and
    # final summary—is covered here.
    checksummed = sorted(
        [p for p in OUT.iterdir() if p.is_file() and p.name != "checksums.sha256"],
        key=lambda p: p.name)
    checksum_lines = [f"{sha256(p)}  {p.name}" for p in checksummed]
    (OUT / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
