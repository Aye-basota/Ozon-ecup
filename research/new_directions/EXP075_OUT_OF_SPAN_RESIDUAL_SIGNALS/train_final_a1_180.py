from __future__ import annotations

import gc
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
T0 = time.time()


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    out = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(out)
    return out


A1 = load_module("exp075_a1_180_core", "run_a1_clean_forward.py")
FINAL = load_module("exp075_a1_180_final", "train_final_and_test_gate.py")


def main() -> None:
    data = A1.CleanData()
    sample = pd.read_csv(A1.SAMPLE)
    sample_ids = sample.user_id.to_numpy(np.int64)
    FINAL.test_eligibility(sample_ids)
    test_rows = data.rows(sample_ids)
    train_cutoffs = [A1.FOLDS[-1] - A1.dt.timedelta(days=lag) for lag in A1.TRAIN_LAGS]
    frames = [data.raw_cutoff_frame(x) for x in train_cutoffs]
    if max(train_cutoffs) + A1.dt.timedelta(days=30) > A1.FOLDS[-1]:
        raise AssertionError("Final residual training target crosses clean corridor")

    Xc, y, uid, _ = A1.concat_context(data, frames)
    halves = A1.stable_half(uid)
    base_cf = np.empty(len(y), dtype=np.float64)
    for side in (0, 1):
        fit, pred = halves != side, halves == side
        model = A1.train_lgb(Xc[fit], y[fit], "baseline", 260)
        base_cf[pred] = model.predict(Xc[pred])
        del model
    residual = y - (base_cf + float(np.mean(y - base_cf)))
    Xc_test = data.context_features(test_rows, FINAL.TEST_CUTOFF)

    Xa = A1.concat_candidate(data, frames, 180, Xc)
    model = A1.train_lgb(Xa, residual, "candidate", 300)
    model_path = EXP / "final_A1_TREE_TRAJ_180.txt"
    model.save_model(str(model_path))
    del Xa
    gc.collect()
    Xa_test = data.candidate_features(test_rows, FINAL.TEST_CUTOFF, 180, Xc_test)
    raw_prediction = model.predict(Xa_test)

    oof = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    amplitude = FINAL.weighted_amplitude(oof, "u_perp_180")
    raw = amplitude * raw_prediction
    Q, span = FINAL.build_span(sample_ids)
    perp, metrics = FINAL.project_test(raw, Q)
    z_alpha = FINAL.aligned_z(FINAL.ALPHA, sample_ids)
    z_public = FINAL.aligned_z(FINAL.PUBLIC_EB, sample_ids)
    metrics["corr_D_perp_current_ORTH"] = A1.correlation(perp, z_alpha - z_public)
    oof_rms = float(np.sqrt(np.mean((amplitude * oof.u_perp_180.to_numpy(float)) ** 2)))
    metrics["test_raw_to_OOF_correction_rms_ratio"] = metrics["rms_D"] / oof_rms
    metrics["passes_perp_fraction_gate"] = metrics["perp_fraction"] >= 0.20
    metrics["amplitude"] = amplitude
    metrics["projection_span_note"] = "Stricter post-EXP075 span includes the already accepted A1-365/A2/joint submissions."

    raw_path = EXP / "A1_TREE_TRAJ_180_TEST_raw_correction.npy"
    perp_path = EXP / "A1_TREE_TRAJ_180_TEST_PERP.npy"
    pred_path = EXP / "A1_TREE_TRAJ_180_test_predictions.parquet"
    np.save(raw_path, raw.astype(np.float32))
    np.save(perp_path, perp.astype(np.float32))
    pd.DataFrame({"user_id": sample_ids, "raw_correction": raw, "perp_correction": perp}).to_parquet(pred_path, index=False)
    if metrics["passes_perp_fraction_gate"]:
        z_new = np.maximum(z_alpha + perp, 0.0)
        submission = pd.DataFrame({"user_id": sample_ids, "predict": np.expm1(z_new)})
        out = ROOT / "submissions" / "SUBMIT_EXP075_A1_TREE_TRAJ_180.csv"
        submission.to_csv(out, index=False, float_format="%.10f")
        metrics["submission_path"] = str(out)
        metrics["submission_sha256"] = FINAL.sha256(out)
    else:
        metrics["submission_path"] = None

    result_path = EXP / "test_span_projection.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["candidates"]["A1_TREE_TRAJ_180"] = metrics
    result["post_EXP075_span_for_A1_180"] = span
    result["supplemental_A1_180_runtime_seconds"] = time.time() - T0
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"A1_TREE_TRAJ_180": metrics, "span_rank": span["rank_centered"]}, indent=2))


if __name__ == "__main__":
    main()
