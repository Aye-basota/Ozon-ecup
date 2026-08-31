# final_team_b_ensemble

## Catalogue metadata

- **Catalogue ID:** `team_a_final__final_team_b_ensemble`
- **Namespace:** `team_a_final`
- **Experiment ID:** `final_team_b_ensemble`
- **Original source:** `src/final_team_b_ensemble.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** requested final partial-slot ensemble
- **Model:** sequence model, ensemble, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** from src.validation import calibrate
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** level_shift = TARGET_LOG_LEVEL - float(z_candidate_test.mean())
- **Submission:** "submission": submission_metrics,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# final_team_b_ensemble

Original script: `src/final_team_b_ensemble.py`

```python
"""Build the final STRONGEST-CURRENT + team-b-B2 ensemble.

The team weight is selected only on canonical S1 OOF predictions.  Public LB
is deliberately not used.  The selected weight is then audited with LOFO and
an OOF-to-TEST correction-variance check before the submission is written.

Run:
    python src/final_team_b_ensemble.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blend import aligned
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, ROOT, VAL_FOLDS_S1
from src.validation import calibrate


EXPERIMENT = "FINAL_TEAM_B_ENSEMBLE_EXP071"
RESULTS = ROOT / "research" / "strategies" / "results" / EXPERIMENT
TEAM_RUN = ARTIFACTS / "TEAM_B_B2_EXP069"
SUBMISSION = ROOT / "submissions" / "submission_FINAL_CAP_UNC_DIST_SEQ_ETX_TEAM_B.csv"
SAMPLE = ROOT / "data" / "raw" / "sample_submit.csv"
TEAM_SHA = "88dc69163b1f39aaac55ddfbfc9986e2203cfbdf"
TARGET_LOG_LEVEL = 2.3293

OOF_COMPONENTS = ["S1-E03a", "S1-E02", "S1-DIST", "SEQ-AVG3", "ETX-AVG3"]
BASE_WEIGHTS = np.array([0.10, 0.20, 0.25, 0.225, 0.225], dtype=np.float64)
TAB_WEIGHTS = BASE_WEIGHTS[:3]
TAB_SLOT_WEIGHT = float(TAB_WEIGHTS.sum())
TEST_COMPONENTS = {
    "CAP": ["S1-CAP"],
    "UNC": ["S1-UNC"],
    "DIST": ["S1-DIST"],
    "SEQ": ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"],
    "ETX": ["ETX-01-S42-DCW", "ETX-01-S43-DCW", "ETX-01-S44-DCW"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_scores(y: np.ndarray, z: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    return np.array([calibrate(y[mask], z[mask])[1] for mask in masks], dtype=np.float64)


def weighted_score(scores: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.dot(scores, weights) / weights.sum())


def optimise_alpha(
    y: np.ndarray,
    z_tab: np.ndarray,
    z_team: np.ndarray,
    z_seq: np.ndarray,
    z_etx: np.ndarray,
    masks: list[np.ndarray],
    weights: np.ndarray,
) -> tuple[float, float]:
    """Select the fraction of the fixed 55% tabular slot assigned to team B."""

    def objective(alpha: float) -> float:
        z = (1.0 - alpha) * z_tab + TAB_SLOT_WEIGHT * alpha * z_team
        z = z + BASE_WEIGHTS[3] * z_seq + BASE_WEIGHTS[4] * z_etx
        return weighted_score(fold_scores(y, z, masks), weights)

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    candidates = [(0.0, objective(0.0)), (1.0, objective(1.0)),
                  (float(result.x), float(result.fun))]
    return min(candidates, key=lambda item: item[1])


def load_team_oof(
    canonical_uid: np.ndarray,
    canonical_cutoff: np.ndarray,
    canonical_y: np.ndarray,
) -> np.ndarray:
    parts = [np.load(TEAM_RUN / f"oof_{fold.strftime('%Y%m%d')}.npz")
             for fold in VAL_FOLDS_S1]
    uid = np.concatenate([part["user_id"] for part in parts])
    cutoff = np.concatenate([part["cutoff"].astype("U10") for part in parts])
    y = np.concatenate([part["y"] for part in parts]).astype(np.float64)
    z = np.concatenate([part["z"] for part in parts]).astype(np.float64)
    key = np.char.add(cutoff, uid.astype("U20"))
    order = np.argsort(key)
    canonical_key = np.char.add(canonical_cutoff, canonical_uid.astype("U20"))
    if not np.array_equal(key[order], canonical_key):
        raise RuntimeError("team B and canonical OOF keys differ")
    if not np.allclose(y[order], canonical_y, rtol=1e-7, atol=1e-6):
        raise RuntimeError("team B and canonical OOF targets differ")
    return z[order]


def load_oof() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix, y, cutoff = aligned(OOF_COMPONENTS)
    base = np.load(ARTIFACTS / f"oof_{OOF_COMPONENTS[0]}.npz")
    base_key = np.char.add(base["cutoff"].astype("U10"), base["user_id"].astype("U20"))
    order = np.argsort(base_key)
    uid = base["user_id"][order]
    if not np.array_equal(base["cutoff"].astype("U10")[order], cutoff):
        raise RuntimeError("failed to recover canonical OOF user order")
    z_team = load_team_oof(uid, cutoff, y)
    return matrix.astype(np.float64), y.astype(np.float64), cutoff, z_team


def load_test_component(names: list[str], expected_uid: np.ndarray) -> np.ndarray:
    predictions = []
    for name in names:
        uid = np.load(ARTIFACTS / f"uid_{name}.npy")
        z = np.load(ARTIFACTS / f"ztest_{name}.npy").astype(np.float64)
        if not np.array_equal(uid, expected_uid):
            raise RuntimeError(f"{name}: TEST user order differs from sample_submit")
        if len(z) != len(expected_uid) or not np.isfinite(z).all():
            raise RuntimeError(f"{name}: invalid TEST predictions")
        predictions.append(z)
    return np.mean(predictions, axis=0)


def load_team_test(expected_uid: np.ndarray) -> np.ndarray:
    data = np.load(TEAM_RUN / "test_predictions.npz")
    frame = pd.DataFrame({"user_id": data["user_id"], "z": data["z"]})
    if len(frame) != len(expected_uid) or frame["user_id"].nunique() != len(expected_uid):
        raise RuntimeError("invalid team B TEST user keys")
    aligned_frame = pd.DataFrame({"user_id": expected_uid}).merge(
        frame, on="user_id", how="left", validate="one_to_one",
    )
    z = aligned_frame["z"].to_numpy(dtype=np.float64)
    if not np.isfinite(z).all():
        raise RuntimeError("missing or invalid team B TEST predictions")
    return z


def validate_submission(path: Path, expected_uid: np.ndarray) -> dict[str, object]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["user_id", "predict"]:
        raise RuntimeError("submission schema must be user_id,predict")
    if len(frame) != 250_000 or frame["user_id"].nunique() != 250_000:
        raise RuntimeError("submission must contain 250,000 unique users")
    if not np.array_equal(frame["user_id"].to_numpy(), expected_uid):
        raise RuntimeError("submission order differs from sample_submit")
    pred = frame["predict"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise RuntimeError("submission contains invalid predictions")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "unique_users": int(frame["user_id"].nunique()),
        "mean_log1p": float(np.log1p(pred).mean()),
        "mean_predict": float(pred.mean()),
        "min": float(pred.min()),
        "max": float(pred.max()),
        "zero_fraction": float((pred == 0).mean()),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)

    matrix, y, cutoff, z_team = load_oof()
    z_tab = np.average(matrix[:3], axis=0, weights=TAB_WEIGHTS) * TAB_SLOT_WEIGHT
    z_seq, z_etx = matrix[3], matrix[4]
    z_base = z_tab + BASE_WEIGHTS[3] * z_seq + BASE_WEIGHTS[4] * z_etx
    folds = [fold.isoformat() for fold in VAL_FOLDS_S1]
    masks = [cutoff == fold for fold in folds]
    fold_weights = np.asarray(FOLD_WEIGHTS_S1, dtype=np.float64)

    alpha, candidate_wcv = optimise_alpha(
        y, z_tab, z_team, z_seq, z_etx, masks, fold_weights,
    )
    z_candidate = ((1.0 - alpha) * z_tab + TAB_SLOT_WEIGHT * alpha * z_team
                   + BASE_WEIGHTS[3] * z_seq + BASE_WEIGHTS[4] * z_etx)
    base_fold_scores = fold_scores(y, z_base, masks)
    candidate_fold_scores = fold_scores(y, z_candidate, masks)
    base_wcv = weighted_score(base_fold_scores, fold_weights)

    lofo_alphas = []
    lofo_fold_scores = []
    for held_out in range(len(masks)):
        keep = [index for index in range(len(masks)) if index != held_out]
        held_alpha, _ = optimise_alpha(
            y,
            z_tab,
            z_team,
            z_seq,
            z_etx,
            [masks[index] for index in keep],
            fold_weights[keep],
        )
        lofo_alphas.append(held_alpha)
        held_z = ((1.0 - held_alpha) * z_tab[masks[held_out]]
                  + TAB_SLOT_WEIGHT * held_alpha * z_team[masks[held_out]]
                  + BASE_WEIGHTS[3] * z_seq[masks[held_out]]
                  + BASE_WEIGHTS[4] * z_etx[masks[held_out]])
        lofo_fold_scores.append(calibrate(y[masks[held_out]], held_z)[1])
    lofo_fold_scores_array = np.asarray(lofo_fold_scores, dtype=np.float64)
    lofo_wcv = weighted_score(lofo_fold_scores_array, fold_weights)

    curve = []
    for probe_alpha in np.linspace(0.0, 1.0, 201):
        probe_z = ((1.0 - probe_alpha) * z_tab + TAB_SLOT_WEIGHT * probe_alpha * z_team
                   + BASE_WEIGHTS[3] * z_seq + BASE_WEIGHTS[4] * z_etx)
        scores = fold_scores(y, probe_z, masks)
        curve.append({
            "team_fraction_of_tab_slot": float(probe_alpha),
            "team_absolute_weight": float(TAB_SLOT_WEIGHT * probe_alpha),
            "wcv": weighted_score(scores, fold_weights),
        })
    pd.DataFrame(curve).to_csv(RESULTS / "weight_curve.csv", index=False)

    sample = pd.read_csv(SAMPLE)
    expected_uid = sample["user_id"].to_numpy()
    test = {name: load_test_component(names, expected_uid)
            for name, names in TEST_COMPONENTS.items()}
    z_team_test = load_team_test(expected_uid)
    z_tab_test = (BASE_WEIGHTS[0] * test["CAP"] + BASE_WEIGHTS[1] * test["UNC"]
                  + BASE_WEIGHTS[2] * test["DIST"])
    z_base_test = (z_tab_test + BASE_WEIGHTS[3] * test["SEQ"]
                   + BASE_WEIGHTS[4] * test["ETX"])
    z_candidate_test = ((1.0 - alpha) * z_tab_test
                        + TAB_SLOT_WEIGHT * alpha * z_team_test
                        + BASE_WEIGHTS[3] * test["SEQ"] + BASE_WEIGHTS[4] * test["ETX"])
    correction_oof = z_candidate - z_base
    correction_test = z_candidate_test - z_base_test
    variance_ratio = float(np.var(correction_test) / np.var(correction_oof))
    if not 0.6 <= variance_ratio <= 1.2:
        raise RuntimeError(f"OOF-to-TEST correction regime failed: {variance_ratio:.6f}")

    level_shift = TARGET_LOG_LEVEL - float(z_candidate_test.mean())
    final_z = np.maximum(z_candidate_test + level_shift, 0.0)
    final_pred = np.maximum(np.expm1(final_z), 0.0)
    pd.DataFrame({"user_id": expected_uid, "predict": final_pred}).to_csv(
        SUBMISSION, index=False, float_format="%.6f",
    )
    submission_metrics = validate_submission(SUBMISSION, expected_uid)

    final_weights = {
        "CAP": float((1.0 - alpha) * BASE_WEIGHTS[0]),
        "UNC": float((1.0 - alpha) * BASE_WEIGHTS[1]),
        "DIST": float((1.0 - alpha) * BASE_WEIGHTS[2]),
        "TEAM_B": float(TAB_SLOT_WEIGHT * alpha),
        "SEQ_AVG3": float(BASE_WEIGHTS[3]),
        "ETX_AVG3_DCW": float(BASE_WEIGHTS[4]),
    }
    metrics = {
        "experiment": EXPERIMENT,
        "team_source_sha": TEAM_SHA,
        "selection": "continuous canonical S1 wCV optimum; LB unused",
        "folds": folds,
        "fold_weights": list(map(float, FOLD_WEIGHTS_S1)),
        "team_fraction_of_tab_slot": alpha,
        "final_absolute_weights": final_weights,
        "base": {"wcv": base_wcv, "fold_scores": base_fold_scores.tolist()},
        "candidate": {
            "wcv": candidate_wcv,
            "delta_vs_base": candidate_wcv - base_wcv,
            "fold_scores": candidate_fold_scores.tolist(),
            "fold_deltas": (candidate_fold_scores - base_fold_scores).tolist(),
            "better_folds": int(np.sum(candidate_fold_scores < base_fold_scores)),
        },
        "lofo": {
            "held_out_team_fractions": lofo_alphas,
            "fold_scores": lofo_fold_scores,
            "wcv": lofo_wcv,
            "delta_vs_base": lofo_wcv - base_wcv,
            "better_folds": int(np.sum(lofo_fold_scores_array < base_fold_scores)),
        },
        "production_regime": {
            "correction_variance_oof": float(np.var(correction_oof)),
            "correction_variance_test": float(np.var(correction_test)),
            "test_oof_variance_ratio": variance_ratio,
            "gate": "PASS",
            "target_mean_log1p_before_floor": TARGET_LOG_LEVEL,
            "pre_level_mean_log1p": float(z_candidate_test.mean()),
            "global_log_shift": level_shift,
        },
        "submission": submission_metrics,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

```
