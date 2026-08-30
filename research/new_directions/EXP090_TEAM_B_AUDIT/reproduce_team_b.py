"""Exact-code reproduction of the Team-B exp024 handoff TEST vector.

This script does not modify Team-B sources.  It imports the frozen handoff API,
trains its five published components, and stores component/final log vectors for
the EXP090 audit.  The missing original exp024 CSV means byte parity cannot be
claimed; the output is a fresh execution of the supplied primary code.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "new_directions" / "EXP090_TEAM_B_AUDIT"
TEAM_B = ROOT / "team-b"
RAW = TEAM_B / "data" / "raw" / "train.parquet"
SAMPLE = TEAM_B / "data" / "raw" / "sample_submit.csv"
EXPECTED_RAW_SHA256 = "5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0"
EXPECTED_SAMPLE_SHA256 = "06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    if sha256(RAW) != EXPECTED_RAW_SHA256:
        raise AssertionError("unexpected Team-B raw train parquet")
    if sha256(SAMPLE) != EXPECTED_SAMPLE_SHA256:
        raise AssertionError("unexpected Team-B sample submission")

    # Team-B and the clean workspace both expose a package called `src`.
    sys.path.insert(0, str(TEAM_B))
    from src.predict import _component_log_prediction, predict_log
    from src.train import HANDOFF_LEVEL, HANDOFF_WEIGHTS, TEST_CUTOFF, train_models

    started = time.time()
    models, meta = train_models()
    component_z: dict[str, np.ndarray] = {}
    uid: np.ndarray | None = None
    for name in HANDOFF_WEIGHTS:
        z = _component_log_prediction(models[name], TEST_CUTOFF)
        if uid is None:
            uid = z.index.to_numpy(np.int64)
        elif not np.array_equal(uid, z.index.to_numpy(np.int64)):
            raise AssertionError(f"component row order mismatch: {name}")
        component_z[name] = z.to_numpy(np.float64)
    assert uid is not None

    sample_uid = pd.read_csv(SAMPLE, usecols=["user_id"])["user_id"].to_numpy(np.int64)
    if not np.array_equal(uid, sample_uid):
        raise AssertionError("Team-B TEST row order differs from sample_submit")
    if len(uid) != 250_000 or len(np.unique(uid)) != 250_000:
        raise AssertionError("Team-B TEST user cardinality failure")

    weights = meta["weights"]
    total_weight = float(sum(weights.values()))
    z_unleveled = sum(component_z[name] * weight for name, weight in weights.items()) / total_weight
    z_formula = np.maximum(z_unleveled + (meta["level"] - float(z_unleveled.mean())), 0.0)
    z_api = predict_log(models, meta=meta, cutoff_date=TEST_CUTOFF, level=meta["level"]).to_numpy(np.float64)
    formula_api_max_abs = float(np.max(np.abs(z_formula - z_api)))
    if formula_api_max_abs > 1e-12:
        raise AssertionError(f"handoff formula/API mismatch: {formula_api_max_abs}")
    predict = np.maximum(np.expm1(z_api), 0.0)

    frame = pd.DataFrame({"user_id": uid})
    for name, z in component_z.items():
        frame[f"z_{name}"] = z
    frame["z_unleveled"] = z_unleveled
    frame["z_final"] = z_api
    frame["predict"] = predict
    parquet_path = EXP / "team_b_reproduction_predictions.parquet"
    frame.to_parquet(parquet_path, index=False)

    npz_path = EXP / "team_b_reproduced_vectors.npz"
    np.savez_compressed(
        npz_path,
        user_id=uid,
        z_unleveled=z_unleveled,
        z_final=z_api,
        predict=predict,
        **{f"z_{name}": z for name, z in component_z.items()},
    )

    elapsed = time.time() - started
    audit = {
        "status": "FRESH_CODE_REPRODUCTION_ORIGINAL_EXP024_MISSING",
        "original_submission_present": False,
        "byte_exact_parity_test_possible": False,
        "missing_artifact": "team-b/submissions/exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_level_e19.csv",
        "raw_train": {"path": str(RAW.resolve()), "sha256": sha256(RAW)},
        "sample_submit": {"path": str(SAMPLE.resolve()), "sha256": sha256(SAMPLE)},
        "train_cutoffs": meta["train_cutoffs"],
        "test_cutoff": TEST_CUTOFF,
        "weights": weights,
        "component_scales": {
            name: float(models[name]["scale"]) for name in weights
        },
        "component_feature_sets": {
            name: models[name]["feature_set"] for name in weights
        },
        "component_kinds": {
            name: models[name]["kind"] for name in weights
        },
        "component_feature_counts": {
            name: len(models[name]["features"]) for name in weights
        },
        "level": float(HANDOFF_LEVEL),
        "formula": "z_i=log1p(clip(expm1(model_i_z)*component_scale,0,inf)); z=sum(w_i*z_i); z=max(z+(2.370966-mean(z)),0)",
        "rows": int(len(uid)),
        "unique_user_id": int(len(np.unique(uid))),
        "sample_order_exact": bool(np.array_equal(uid, sample_uid)),
        "finite": bool(np.isfinite(predict).all()),
        "nonnegative": bool(np.all(predict >= 0)),
        "fraction_clipped_final_z": float(np.mean(z_formula == 0.0)),
        "mean_z_unleveled": float(z_unleveled.mean()),
        "mean_z_final": float(z_api.mean()),
        "level_shift_before_clipping": float(meta["level"] - z_unleveled.mean()),
        "rms_level_realized_correction": rms(z_api - z_unleveled),
        "formula_api_max_abs": formula_api_max_abs,
        "runtime_seconds": elapsed,
        "predictions_parquet": {"path": str(parquet_path.resolve()), "sha256": sha256(parquet_path)},
        "vectors_npz": {"path": str(npz_path.resolve()), "sha256": sha256(npz_path)},
    }
    (EXP / "reproduction_run.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
