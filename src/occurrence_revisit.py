"""EXP-063: nested-LOFO occurrence-member audit against exact exp_037.

Run: python src/occurrence_revisit.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, ROOT, VAL_FOLDS_S1


GRID = np.asarray([0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20])
EXPECTED_BASE_WCV = 1.7475098625
RESULTS = ROOT / "research" / "strategies" / "results" / "OCCURRENCE_REVISIT_EXP063"


def calibrated_score(y: np.ndarray, z: np.ndarray) -> float:
    return float(np.std(np.log1p(y) - z))


def fold_curve(y: np.ndarray, base: np.ndarray, member: np.ndarray,
               cutoff: np.ndarray, folds: list[str], grid: np.ndarray = GRID) -> np.ndarray:
    out = np.empty((len(grid), len(folds)), dtype=float)
    for ai, alpha in enumerate(grid):
        z = (1.0 - alpha) * base + alpha * member
        for fi, fold in enumerate(folds):
            mask = cutoff == fold
            out[ai, fi] = calibrated_score(y[mask], z[mask])
    return out


def nested_lofo(curve: np.ndarray, fold_weights: np.ndarray,
                grid: np.ndarray = GRID) -> tuple[np.ndarray, np.ndarray]:
    """Select alpha without each held fold; ascending grid makes ties conservative."""
    selected = np.empty(curve.shape[1], dtype=int)
    held = np.empty(curve.shape[1], dtype=float)
    for h in range(curve.shape[1]):
        donor = np.arange(curve.shape[1]) != h
        weights = fold_weights[donor] / fold_weights[donor].sum()
        donor_score = curve[:, donor] @ weights
        best = int(np.flatnonzero(donor_score <= donor_score.min() + 1e-12)[0])
        selected[h] = best
        held[h] = curve[best, h]
    return selected, held


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_aligned() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    aligned_path = ARTIFACTS / "RESDISC_053" / "aligned_oof.parquet"
    base = pd.read_parquet(aligned_path, columns=["cutoff", "user_id", "y_true", "z_strong_raw"])
    base = base.sort_values(["cutoff", "user_id"], kind="mergesort").reset_index(drop=True)
    members: dict[str, np.ndarray] = {}
    audits: dict = {
        "aligned_oof_sha256": sha256(aligned_path),
        "rows": int(len(base)),
        "sources": {},
    }
    base_keys = base[["cutoff", "user_id"]].astype({"cutoff": str}).to_records(index=False)
    for name in ("S1-E11", "S1-E10"):
        path = ARTIFACTS / f"oof_{name}.npz"
        data = np.load(path)
        order = np.lexsort((data["user_id"], data["cutoff"].astype("U10")))
        keys = np.rec.fromarrays(
            [data["cutoff"][order].astype("U10"), data["user_id"][order].astype(np.int64)],
            names="cutoff,user_id",
        )
        assert np.array_equal(keys, base_keys), f"{name}: (cutoff,user_id) alignment failed"
        target_diff = float(np.max(np.abs(data["y"][order].astype(float) - base["y_true"].to_numpy(float))))
        assert target_diff <= 1e-6, f"{name}: target mismatch {target_diff}"
        z = data["z"][order].astype(float)
        assert np.isfinite(z).all(), f"{name}: non-finite prediction"
        members[name] = z
        audits["sources"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "target_max_abs_diff": target_diff,
            "finite": True,
        }
    audits["unique_keys"] = int(base.drop_duplicates(["cutoff", "user_id"]).shape[0]) == len(base)
    audits["finite_base"] = bool(np.isfinite(base["z_strong_raw"].to_numpy(float)).all())
    return base, members, audits


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    base_df, members, audits = load_aligned()
    folds = [d.isoformat() for d in VAL_FOLDS_S1]
    cutoff = base_df["cutoff"].astype(str).to_numpy()
    y = base_df["y_true"].to_numpy(float)
    base = base_df["z_strong_raw"].to_numpy(float)
    weights = np.asarray(FOLD_WEIGHTS_S1, dtype=float)
    weights /= weights.sum()
    fold_sizes = [int((cutoff == f).sum()) for f in folds]
    base_scores = np.asarray([calibrated_score(y[cutoff == f], base[cutoff == f]) for f in folds])
    base_wcv = float(base_scores @ weights)
    assert abs(base_wcv - EXPECTED_BASE_WCV) <= 2e-7, (base_wcv, EXPECTED_BASE_WCV)
    audits.update({
        "folds": folds,
        "fold_sizes": fold_sizes,
        "base_fold_scores": base_scores.tolist(),
        "base_wcv": base_wcv,
        "base_replay_abs_error": abs(base_wcv - EXPECTED_BASE_WCV),
        "pass": True,
    })

    curve_rows: list[dict] = []
    arm_results: dict[str, dict] = {}
    for arm, name in (("REAL_E11", "S1-E11"), ("DIRECT_CONTROL_E10", "S1-E10")):
        curve = fold_curve(y, base, members[name], cutoff, folds)
        fixed_wcv = curve @ weights
        selected_idx, held = nested_lofo(curve, weights)
        nested_delta_folds = held - base_scores
        result = {
            "source": name,
            "standalone_fold_scores": [calibrated_score(y[cutoff == f], members[name][cutoff == f]) for f in folds],
            "selected_alpha_by_held_fold": GRID[selected_idx].tolist(),
            "selected_grid_index_by_held_fold": selected_idx.tolist(),
            "nested_fold_scores": held.tolist(),
            "nested_fold_deltas": nested_delta_folds.tolist(),
            "nested_wcv": float(held @ weights),
            "nested_delta_wcv": float(nested_delta_folds @ weights),
            "nested_improved_folds": int((nested_delta_folds < 0).sum()),
            "latest_delta": float(nested_delta_folds[-1]),
            "best_fixed_alpha": float(GRID[int(np.argmin(fixed_wcv))]),
            "best_fixed_wcv": float(fixed_wcv.min()),
            "best_fixed_delta_wcv": float(fixed_wcv.min() - base_wcv),
            "prediction_difference_variance_oof": float(np.var(members[name] - base)),
            "residual_correlation": float(np.corrcoef(np.log1p(y) - members[name], np.log1p(y) - base)[0, 1]),
        }
        arm_results[arm] = result
        for ai, alpha in enumerate(GRID):
            for fi, fold in enumerate(folds):
                curve_rows.append({
                    "arm": arm, "alpha": alpha, "cutoff": fold,
                    "score": curve[ai, fi], "delta": curve[ai, fi] - base_scores[fi],
                    "fixed_wcv": fixed_wcv[ai],
                    "selected_when_held": bool(selected_idx[fi] == ai),
                })

    real = arm_results["REAL_E11"]
    control = arm_results["DIRECT_CONTROL_E10"]
    real_idx = np.asarray(real["selected_grid_index_by_held_fold"])
    control_separation = float(real["nested_delta_wcv"] - control["nested_delta_wcv"])
    success = bool(
        real["nested_delta_wcv"] <= -0.0010
        and real["nested_improved_folds"] >= 3
        and real["latest_delta"] < 0
        and min(real["selected_alpha_by_held_fold"]) > 0
        and int(real_idx.max() - real_idx.min()) <= 1
        and control_separation <= -0.0005
    )
    development = bool(
        -0.0010 < real["nested_delta_wcv"] <= -0.0005
        and real["nested_improved_folds"] >= 3
        and real["latest_delta"] < 0
    )
    verdict = "CONTINUE_TO_TEST_AUDIT" if success else ("CONTINUE_DEVELOPMENT_ONLY" if development else "REJECT")
    summary = {
        "experiment_id": 63,
        "prefix": "OCCURRENCE_REVISIT_EXP063",
        "development_reference": "STRONGEST-CURRENT / exp_037",
        "grid": GRID.tolist(),
        "audits": audits,
        "arms": arm_results,
        "real_minus_direct_control_nested_delta_wcv": control_separation,
        "decision": {
            "success_gate_passed": success,
            "development_gate_passed": development,
            "verdict": verdict,
            "next": "test regime audit" if success else ("record only; no reference change" if development else "close direct occurrence integration"),
        },
    }
    pd.DataFrame(curve_rows).to_csv(RESULTS / "fixed_and_nested_curves.csv", index=False)
    (RESULTS / "provenance_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
