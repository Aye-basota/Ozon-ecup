"""Minimal error analysis for the confirmed Strategy 2 OOF predictions.

Run:
    python src/strategy_2_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_features
from src.strategy_2 import ARTIFACTS, _load_s1_oof, rmsle_z


ROOT = Path(__file__).resolve().parent.parent


def _segment_metrics(y: np.ndarray, z_s2: np.ndarray, z_s1: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "n": int(mask.sum()),
        "s2": rmsle_z(y[mask], z_s2[mask]),
        "s1": rmsle_z(y[mask], z_s1[mask]),
        "delta_s2_minus_s1": rmsle_z(y[mask], z_s2[mask]) - rmsle_z(y[mask], z_s1[mask]),
    }


def main() -> None:
    s2 = np.load(ARTIFACTS / "s2_oof_best.npz")
    order = np.argsort(np.char.add(
        s2["cutoff"].astype("U10"), np.char.zfill(s2["user_id"].astype("U10"), 10)))
    users = s2["user_id"][order]
    cutoffs = s2["cutoff"][order]
    y = s2["y"][order]
    z_s2 = s2["z_K5"][order].astype(np.float64)

    s1_dir = ROOT.parent / "OZON-E-CUP" / "artifacts"
    s1_users, s1_cutoffs, s1_y, z_s1 = _load_s1_oof(s1_dir)
    s1_keys = np.char.add(s1_cutoffs.astype("U10"), np.char.zfill(s1_users.astype("U10"), 10))
    s2_keys = np.char.add(cutoffs.astype("U10"), np.char.zfill(users.astype("U10"), 10))
    positions = np.searchsorted(s1_keys, s2_keys)
    assert np.array_equal(s1_keys[positions], s2_keys)
    assert np.allclose(s1_y[positions], y)
    z_s1 = z_s1[positions]

    buy_days = np.empty(len(users), dtype=np.float32)
    for cutoff in np.unique(cutoffs):
        mask = cutoffs == cutoff
        feature = build_features(str(cutoff)).select("user_id", "w180_days_buy")
        joined = pl.DataFrame({"user_id": users[mask]}).join(feature, on="user_id", how="left")
        buy_days[mask] = joined["w180_days_buy"].to_numpy()

    buckets = {
        "0": buy_days == 0,
        "1": buy_days == 1,
        "2-3": (buy_days >= 2) & (buy_days <= 3),
        "4-7": (buy_days >= 4) & (buy_days <= 7),
        "8-15": (buy_days >= 8) & (buy_days <= 15),
        "16+": buy_days >= 16,
    }
    result = {
        "overall": _segment_metrics(y, z_s2, z_s1, np.ones(len(y), dtype=bool)),
        "residual_correlation": float(np.corrcoef(np.log1p(y) - z_s2,
                                                   np.log1p(y) - z_s1)[0, 1]),
        "by_outcome": {
            "y=0": _segment_metrics(y, z_s2, z_s1, y == 0),
            "y>0": _segment_metrics(y, z_s2, z_s1, y > 0),
        },
        "by_w180_days_buy": {
            name: _segment_metrics(y, z_s2, z_s1, mask) for name, mask in buckets.items()
        },
        "by_fold": {
            str(cutoff): _segment_metrics(y, z_s2, z_s1, cutoffs == cutoff)
            for cutoff in np.unique(cutoffs)
        },
    }
    output = ARTIFACTS / "s2_error_analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
