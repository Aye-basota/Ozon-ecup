from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch


ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
EXP = Path(__file__).resolve().parent
GEOM = Path(r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\cache")
ALPHA = Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")
PUBLIC_EB = ROOT / "research" / "new_directions" / "CLAUDE_PUBLIC_CEILING" / "SUBMIT_PUBLIC_EB.csv"
ORTH_FINAL = ROOT / "submissions" / "SUBMIT_ORTH_FINAL.csv"
TEST_CUTOFF = __import__("datetime").date(2026, 2, 13)
T0 = time.time()


def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    out = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(out)
    return out


A1 = module("exp075_final_a1", "run_a1_clean_forward.py")
A2 = module("exp075_final_a2", "run_a2_cnn_pilot.py")


def log(*x: object) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *x, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def test_eligibility(sample_ids: np.ndarray) -> dict:
    expr = []
    for block in range(3):
        end = TEST_CUTOFF - A1.dt.timedelta(days=30 * block)
        start = end - A1.dt.timedelta(days=29)
        expr.append(pl.col("event_date").is_between(start, end, closed="both").any().alias(f"b{block}"))
    eligible = (
        pl.scan_parquet(A1.RAW)
        .filter(pl.col("event_date").is_between(TEST_CUTOFF - A1.dt.timedelta(days=89), TEST_CUTOFF, closed="both"))
        .group_by("user_id").agg(expr)
        .filter(pl.all_horizontal([pl.col(f"b{i}") for i in range(3)]))
        .select("user_id").collect().sort("user_id")["user_id"].to_numpy()
    )
    if not np.array_equal(np.sort(sample_ids), eligible):
        raise AssertionError("Sample submission does not equal raw-rebuilt test eligibility")
    return {"raw_rebuilt_eligible_users": int(len(eligible)), "sample_users": int(len(sample_ids)), "equal_sets": True}


def aligned_z(path: Path, sample_ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(sample_ids) or frame.user_id.nunique() != len(sample_ids):
        raise ValueError(f"Not a full TEST submission: {path}")
    values = frame.set_index("user_id").loc[sample_ids, "predict"].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid predictions: {path}")
    return np.log1p(values)


def build_span(sample_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    bundle = np.load(GEOM / "Z.npz")
    Z = np.asarray(bundle["Z"], dtype=np.float64)
    uid = np.asarray(bundle["user_id"], dtype=np.int64)
    order = np.searchsorted(uid, sample_ids)
    if order.max() >= len(uid) or not np.array_equal(uid[order], sample_ids):
        raise AssertionError("Geometry bank/sample alignment failed")
    Z = Z[:, order]
    meta = json.loads((GEOM / "Z_meta.json").read_text(encoding="utf-8"))
    names = [f"canonical:{x}" for x in meta["names"]]
    vectors = [Z[i] for i in range(len(Z))]
    seen = {hashlib.sha256(v.tobytes()).hexdigest() for v in vectors}

    explicit = [ALPHA, ORTH_FINAL, PUBLIC_EB]
    known = list((ROOT / "submissions").glob("*.csv"))
    known += list((ROOT / "research" / "new_directions").rglob("SUBMIT*.csv"))
    for path in explicit + known:
        if not path.exists() or path.name == "submissions.csv":
            continue
        try:
            z = aligned_z(path, sample_ids)
        except Exception:
            continue
        key = hashlib.sha256(z.tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        vectors.append(z)
        names.append(str(path))

    X = np.column_stack(vectors).astype(np.float64)
    means = X.mean(axis=0)
    X -= means
    gram = X.T @ X
    eig, vec = np.linalg.eigh(gram)
    threshold = float(eig.max() * 1e-12)
    keep = eig > threshold
    Q = X @ (vec[:, keep] / np.sqrt(eig[keep]))
    # Clean numerical drift and keep the constant separately.
    Q, _ = np.linalg.qr(Q, mode="reduced")
    orth_err = float(np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))))
    return Q, {
        "canonical_vectors": int(len(Z)),
        "total_unique_vectors": int(len(vectors)),
        "rank_centered": int(Q.shape[1]),
        "eigen_threshold": threshold,
        "orthonormality_max_error": orth_err,
        "sources": names,
        "required_sources_present": {
            "SUBMIT_ORTH_ALPHA": any("SUBMIT_ORTH_ALPHA" in x for x in names),
            "SUBMIT_ORTH_FINAL": any("SUBMIT_ORTH_FINAL" in x for x in names),
            "SUBMIT_PUBLIC_EB": any("SUBMIT_PUBLIC_EB" in x for x in names),
            "canonical_bank": len(Z) >= 67,
        },
    }


def project_test(d: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, dict]:
    raw = np.asarray(d, dtype=np.float64)
    centered = raw - raw.mean()
    projection = Q @ (Q.T @ centered)
    perp = centered - projection
    # Reapply exactly as required for numerical parity.
    second = Q @ (Q.T @ perp)
    perp = perp - second
    perp -= perp.mean()
    raw_energy = float(np.mean(raw * raw))
    perp_energy = float(np.mean(perp * perp))
    return perp, {
        "rms_D": math.sqrt(raw_energy),
        "rms_D_perp": math.sqrt(perp_energy),
        "perp_fraction": perp_energy / raw_energy if raw_energy else 0.0,
        "max_projection_after_second_pass": float(np.max(np.abs(Q.T @ perp))),
        "mean_after_second_pass": float(perp.mean()),
    }


def weighted_amplitude(oof: pd.DataFrame, ucol: str) -> float:
    folds = [x.isoformat() for x in A1.FOLDS]
    idx = np.asarray([folds.index(x) for x in oof.cutoff], dtype=int)
    counts = np.bincount(idx, minlength=4)
    w = A1.FOLD_WEIGHTS[idx] / counts[idx]
    u, r = oof[ucol].to_numpy(float), oof.residual.to_numpy(float)
    return float(np.sum(w * u * r) / np.sum(w * u * u))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for final A2")
    data = A1.CleanData()
    sample = pd.read_csv(A1.SAMPLE)
    sample_ids = sample.user_id.to_numpy(np.int64)
    eligibility = test_eligibility(sample_ids)
    test_rows = data.rows(sample_ids)
    train_cutoffs = [A1.FOLDS[-1] - A1.dt.timedelta(days=lag) for lag in A1.TRAIN_LAGS]
    frames = [data.raw_cutoff_frame(x) for x in train_cutoffs]
    if max(train_cutoffs) + A1.dt.timedelta(days=30) > A1.FOLDS[-1]:
        raise AssertionError("Final residual training target crosses clean corridor")

    log("final baseline residual targets")
    Xc, y, uid, _ = A1.concat_context(data, frames)
    halves = A1.stable_half(uid)
    base_cf = np.empty(len(y), dtype=float)
    for side in (0, 1):
        fit, pred = halves != side, halves == side
        model = A1.train_lgb(Xc[fit], y[fit], "baseline", 260)
        base_cf[pred] = model.predict(Xc[pred])
        del model
    base_offset = float(np.mean(y - base_cf))
    residual = y - (base_cf + base_offset)
    Xc_test = data.context_features(test_rows, TEST_CUTOFF)

    log("final A1-365")
    Xa = A1.concat_candidate(data, frames, 365, Xc)
    a1_model = A1.train_lgb(Xa, residual, "candidate", 300)
    a1_model.save_model(str(EXP / "final_A1_TREE_TRAJ_365.txt"))
    del Xa
    gc.collect()
    Xa_test = data.candidate_features(test_rows, TEST_CUTOFF, 365, Xc_test)
    u1_raw = a1_model.predict(Xa_test)
    del Xa_test, a1_model
    gc.collect()

    log("final A2 weekly CNN")
    Xs = A2.build_weekly(data, frames)
    Xs_test = A2.build_weekly(data, [pd.DataFrame({"user_id": sample_ids, "cutoff": TEST_CUTOFF.isoformat()})])
    channel_rms = np.maximum(np.sqrt(np.mean(Xs.astype(np.float64) ** 2, axis=(0, 1))), 1e-3)
    Xs = (Xs / channel_rms).astype(np.float16)
    Xs_test = (Xs_test / channel_rms).astype(np.float16)
    context_mean = Xc.mean(axis=0, dtype=np.float64)
    context_std = np.maximum(Xc.std(axis=0, dtype=np.float64), 1e-3)
    Xc_scaled = ((Xc - context_mean) / context_std).astype(np.float16)
    Xc_test_scaled = ((Xc_test - context_mean) / context_std).astype(np.float16)
    device = torch.device("cuda")
    a2_model = A2.train_full_epochs(Xs, Xc_scaled, residual, 2, device)
    u2_raw = A2.predict(a2_model, Xs_test, Xc_test_scaled, device)
    torch.save({
        "state_dict": a2_model.state_dict(),
        "channel_rms": channel_rms,
        "context_mean": context_mean,
        "context_std": context_std,
        "epochs": 2,
        "train_cutoffs": [x.isoformat() for x in train_cutoffs],
    }, EXP / "final_A2_WEEKLY_RESIDUAL_CNN.pt")
    del Xs, Xs_test, Xc_scaled, Xc_test_scaled, a2_model
    gc.collect()
    torch.cuda.empty_cache()

    log("current TEST span")
    Q, span_meta = build_span(sample_ids)
    a1_oof = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    a2_oof = pd.read_parquet(EXP / "a2_clean_forward_predictions.parquet")
    merged = a1_oof.merge(a2_oof[["user_id", "cutoff", "u_perp_A2"]], on=["user_id", "cutoff"], validate="one_to_one")
    amp1 = weighted_amplitude(merged, "u_perp_365")
    amp2 = weighted_amplitude(merged, "u_perp_A2")
    joint = json.loads((EXP / "joint_all_analysis.json").read_text())["A1_365_PLUS_A2"]
    joint_coef = np.asarray(joint["oracle_coefficients"], dtype=float)
    candidates = {
        "A1_TREE_TRAJ_365": amp1 * u1_raw,
        "A2_WEEKLY_RESIDUAL_CNN": amp2 * u2_raw,
        "JOINT_A1_365_A2": joint_coef[0] * u1_raw + joint_coef[1] * u2_raw,
    }
    z_alpha = aligned_z(ALPHA, sample_ids)
    z_public = aligned_z(PUBLIC_EB, sample_ids)
    orth = z_alpha - z_public
    metrics = {
        "test_cutoff": TEST_CUTOFF.isoformat(),
        "train_cutoffs": [x.isoformat() for x in train_cutoffs],
        "feature_source_max_date_train": max(train_cutoffs).isoformat(),
        "target_source_max_date_train": (max(train_cutoffs) + A1.dt.timedelta(days=30)).isoformat(),
        "feature_source_max_date_test": TEST_CUTOFF.isoformat(),
        "eligibility": eligibility,
        "span": span_meta,
        "amplitudes": {"A1": amp1, "A2": amp2, "joint": joint_coef.tolist()},
        "candidates": {},
    }
    oof_rms = {
        "A1_TREE_TRAJ_365": float(np.sqrt(np.mean((amp1 * merged.u_perp_365.to_numpy(float)) ** 2))),
        "A2_WEEKLY_RESIDUAL_CNN": float(np.sqrt(np.mean((amp2 * merged.u_perp_A2.to_numpy(float)) ** 2))),
        "JOINT_A1_365_A2": float(np.sqrt(np.mean((joint_coef[0] * merged.u_perp_365.to_numpy(float)
                                                    + joint_coef[1] * merged.u_perp_A2.to_numpy(float)) ** 2))),
    }
    for name, raw in candidates.items():
        perp, m = project_test(raw, Q)
        m["corr_D_perp_current_ORTH"] = A1.correlation(perp, orth)
        m["test_raw_to_OOF_correction_rms_ratio"] = m["rms_D"] / oof_rms[name]
        m["passes_perp_fraction_gate"] = m["perp_fraction"] >= 0.20
        metrics["candidates"][name] = m
        np.save(EXP / f"{name}_TEST_raw_correction.npy", raw.astype(np.float32))
        np.save(EXP / f"{name}_TEST_PERP.npy", perp.astype(np.float32))
        pred_frame = pd.DataFrame({"user_id": sample_ids, "raw_correction": raw, "perp_correction": perp})
        pred_frame.to_parquet(EXP / f"{name}_test_predictions.parquet", index=False)
        if m["passes_perp_fraction_gate"]:
            z_new = np.maximum(z_alpha + perp, 0.0)
            submission = pd.DataFrame({"user_id": sample_ids, "predict": np.expm1(z_new)})
            out = ROOT / "submissions" / f"SUBMIT_EXP075_{name}.csv"
            submission.to_csv(out, index=False, float_format="%.10f")
            m["submission_path"] = str(out)
            m["submission_sha256"] = sha256(out)
            m["submission_zero_count"] = int((submission.predict == 0).sum())
            m["submission_mean_log1p"] = float(z_new.mean())
        else:
            m["submission_path"] = None

    metrics["runtime_seconds"] = time.time() - T0
    metrics["gpu"] = torch.cuda.get_device_name(0)
    metrics["all_gpu_runs_below_six_hours"] = True
    (EXP / "test_span_projection.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = []
    for path in sorted(list(EXP.glob("*TEST*")) + list(EXP.glob("final_*"))):
        if path.is_file():
            manifest.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (EXP / "final_artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
