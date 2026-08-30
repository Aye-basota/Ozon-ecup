"""Two-fold out-of-time validation for the public team-b-final API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.predict import CURRENT_WEIGHT, TEAM_WEIGHT, predict_current_log
from src.team_model import build_target, predict_log as predict_team_log
from src.train import train_final_models


ROOT = Path(__file__).resolve().parent
FOLDS = [
    ("2025-11-15", "2025-12-15"),
    ("2025-12-15", "2026-01-14"),
]

# Production difference between team-b-final and the incumbent level.  This is
# fixed before validation and is used only as a calibration diagnostic.
PRODUCTION_LEVEL_SHIFT = 2.4512064373460523 - 2.3297898398410823


def rmsle_from_log(y: np.ndarray, z: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y) - z) ** 2)))


def main() -> None:
    df_raw = pd.read_parquet(ROOT / "data" / "train.parquet")
    metrics = []
    predictions = []

    for train_cutoff, val_cutoff in FOLDS:
        print(f"fold train={train_cutoff} val={val_cutoff}", flush=True)
        models = train_final_models(df_raw, train_cutoff)

        z_current = predict_current_log(
            models["current"],
            df_raw,
            val_cutoff,
            models["current_features"],
        )
        z_team = predict_team_log(
            models["team"],
            models["team_meta"],
            df_raw,
            cutoff_date=val_cutoff,
            level=models["team_meta"]["level"],
        ).reindex(z_current.index).fillna(z_current)
        z_final = CURRENT_WEIGHT * z_current + TEAM_WEIGHT * z_team
        z_shifted = pd.Series(
            np.maximum(z_final.to_numpy() - PRODUCTION_LEVEL_SHIFT, 0.0),
            index=z_final.index,
        )

        y = build_target(df_raw, val_cutoff).reindex(z_final.index).fillna(0.0)
        yv = y.to_numpy(np.float64)
        zy = np.log1p(yv)

        fold_row = {
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
            "rows": int(len(y)),
            "target_mean_log": float(zy.mean()),
            "current_rmsle": rmsle_from_log(yv, z_current.to_numpy()),
            "team_rmsle": rmsle_from_log(yv, z_team.to_numpy()),
            "final_rmsle": rmsle_from_log(yv, z_final.to_numpy()),
            "final_shifted_rmsle": rmsle_from_log(yv, z_shifted.to_numpy()),
            "current_bias": float((z_current.to_numpy() - zy).mean()),
            "team_bias": float((z_team.to_numpy() - zy).mean()),
            "final_bias": float((z_final.to_numpy() - zy).mean()),
            "final_shifted_bias": float((z_shifted.to_numpy() - zy).mean()),
            "current_team_prediction_corr": float(np.corrcoef(z_current, z_team)[0, 1]),
            "current_team_error_corr": float(
                np.corrcoef(z_current.to_numpy() - zy, z_team.to_numpy() - zy)[0, 1]
            ),
        }
        metrics.append(fold_row)
        print(json.dumps(fold_row, ensure_ascii=False), flush=True)

        predictions.append(pd.DataFrame({
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
            "user_id": z_final.index.to_numpy(),
            "target": yv,
            "z_current": z_current.to_numpy(),
            "z_team": z_team.to_numpy(),
            "z_final": z_final.to_numpy(),
            "z_final_shifted": z_shifted.to_numpy(),
        }))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(ROOT / "validation_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        ROOT / "validation_predictions.parquet",
        index=False,
    )

    summary = {
        "folds": metrics,
        "mean": {
            col: float(metrics_df[col].mean())
            for col in [
                "current_rmsle",
                "team_rmsle",
                "final_rmsle",
                "final_shifted_rmsle",
                "current_bias",
                "team_bias",
                "final_bias",
                "final_shifted_bias",
                "current_team_prediction_corr",
                "current_team_error_corr",
            ]
        },
        "production_level_shift": PRODUCTION_LEVEL_SHIFT,
        "current_weight": CURRENT_WEIGHT,
        "team_weight": TEAM_WEIGHT,
    }
    (ROOT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["mean"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
