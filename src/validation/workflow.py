"""Minimal end-to-end tabular experiment workflow.

This module intentionally contains infrastructure only: it delegates all feature
construction to ``build_features``/``make_xy`` and all model behavior to the
registered model family.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data import sample_submission
from src.features.canonical import feature_names, make_xy, to_numpy
from src.models import fit_model, predict_model
from src.settings import ROOT, competition
from src.utils.artifacts import load_oof, write_json, write_oof, write_test
from src.validation.evaluate import compare_oof, evaluate_oof
from src.validation.folds import canonical_folds, cutoff_grid


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    name: str
    family: str
    hypothesis: str
    parent_baseline: str = "EXP_037_STRONGEST_CURRENT"
    history_days: int | None = None
    normalize_long_windows: bool = True
    min_history_days: int = 90
    cutoff_step_days: int = 7
    train_panel_blocks: int = 1
    validation_panel_blocks: int = 3
    rounds: int = 600
    model_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentSpec":
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected mapping in {path}")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed - {"validation", "success_gate", "exact_change"}
        if unknown:
            raise ValueError(f"Unknown experiment config keys: {sorted(unknown)}")
        return cls(**{key: value[key] for key in allowed if key in value})


def _assemble_training(spec: ExperimentSpec, cutoffs: tuple[dt.date, ...], columns: list[str]):
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for cutoff in cutoffs:
        frame, y_true = make_xy(
            cutoff,
            history_days=spec.history_days,
            panel_blocks=spec.train_panel_blocks,
            with_target=True,
            normalize_long=spec.normalize_long_windows,
        )
        assert y_true is not None
        feature_blocks.append(to_numpy(frame, columns))
        target_blocks.append(y_true)
    return np.vstack(feature_blocks), np.concatenate(target_blocks)


def run_cv(spec: ExperimentSpec, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    user_ids: list[np.ndarray] = []
    cutoffs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    columns: list[str] | None = None

    for fold in canonical_folds(spec.min_history_days, spec.cutoff_step_days):
        validation_frame, y_validation = make_xy(
            fold.validation_cutoff,
            history_days=spec.history_days,
            panel_blocks=spec.validation_panel_blocks,
            with_target=True,
            normalize_long=spec.normalize_long_windows,
        )
        assert y_validation is not None
        if columns is None:
            columns = feature_names(validation_frame)
        x_train, y_train = _assemble_training(spec, fold.train_cutoffs, columns)
        model = fit_model(
            spec.family,
            x_train,
            y_train,
            rounds=spec.rounds,
            params=spec.model_params,
        )
        x_validation = to_numpy(validation_frame, columns)
        z_pred = np.maximum(predict_model(spec.family, model, x_validation), 0.0)
        user_ids.append(validation_frame["user_id"].to_numpy())
        cutoffs.append(np.full(len(y_validation), fold.validation_cutoff.isoformat(), dtype="U10"))
        targets.append(y_validation)
        predictions.append(z_pred)
        del x_train, y_train, x_validation, model
        gc.collect()

    assert columns is not None
    cutoff_all = np.concatenate(cutoffs)
    user_all = np.concatenate(user_ids)
    y_all = np.concatenate(targets)
    z_all = np.concatenate(predictions)
    artifact = write_oof(spec.experiment_id, cutoff_all, user_all, y_all, z_all)
    challenger = load_oof(artifact)
    baseline_cfg = competition()["offline_baseline"]
    baseline = load_oof(ROOT / str(baseline_cfg["artifact"]))
    comparison = compare_oof(challenger, baseline)
    result = {
        "experiment_id": spec.experiment_id,
        "name": spec.name,
        "family": spec.family,
        "hypothesis": spec.hypothesis,
        "parent_baseline": spec.parent_baseline,
        "validation": "canonical_4fold_wCV_1_2_4_8",
        "wCV": comparison["challenger"]["wCV"],
        "baseline_wCV": comparison["baseline"]["wCV"],
        "delta_wCV": comparison["delta_wCV"],
        "folds_positive": comparison["folds_positive"],
        "fold_deltas": comparison["fold_deltas"],
        "runtime_seconds": round(time.time() - started, 1),
        "status": None,
        "oof_path": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "test_path": None,
        "n_features": len(columns),
        "feature_columns": columns,
        "config": {
            key: value
            for key, value in spec.__dict__.items()
            if key not in {"hypothesis", "name"}
        },
        "evaluation": comparison["challenger"],
    }
    write_json(output_dir / "metrics.json", result)
    _write_report(output_dir / "report.md", result)
    return result


def run_test(spec: ExperimentSpec, output_dir: Path) -> Path:
    cfg = competition()
    test_cutoff = dt.date.fromisoformat(str(cfg["data"]["test_cutoff"]))
    test_frame, _ = make_xy(
        test_cutoff,
        history_days=spec.history_days,
        panel_blocks=spec.validation_panel_blocks,
        with_target=False,
        normalize_long=spec.normalize_long_windows,
    )
    columns = feature_names(test_frame)
    train_cutoffs = tuple(cutoff_grid(spec.min_history_days, spec.cutoff_step_days))
    x_train, y_train = _assemble_training(spec, train_cutoffs, columns)
    model = fit_model(spec.family, x_train, y_train, rounds=spec.rounds, params=spec.model_params)
    z_pred = np.maximum(predict_model(spec.family, model, to_numpy(test_frame, columns)), 0.0)

    expected = sample_submission().select("user_id").to_pandas()
    predicted = pd.DataFrame({"user_id": test_frame["user_id"].to_numpy(), "z_pred": z_pred})
    aligned = expected.merge(predicted, on="user_id", how="left", validate="one_to_one")
    if aligned["z_pred"].isna().any():
        raise ValueError("TEST prediction is missing sample-submission users")
    artifact = write_test(spec.experiment_id, aligned["user_id"], aligned["z_pred"])

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["test_path"] = str(artifact.relative_to(ROOT)).replace("\\", "/")
        write_json(metrics_path, metrics)
    return artifact


def _write_report(path: Path, result: dict[str, Any]) -> None:
    fold_lines = "\n".join(
        f"- `{row['cutoff']}`: {row['rmsle_cal']:.9f}"
        for row in result["evaluation"]["per_fold"]
    )
    text = f"""# {result['experiment_id']} — result

## Hypothesis

{result['hypothesis']}

## Validation result

- wCV: `{result['wCV']:.12f}`
- Baseline wCV: `{result['baseline_wCV']:.12f}`
- Delta wCV: `{result['delta_wCV']:+.12f}`
- Positive folds: `{result['folds_positive']}/4`
- Runtime: `{result['runtime_seconds']}` seconds
- Verdict: pending; set exactly one of `PASS`, `WEAK_SIGNAL`, `REJECT`, or `INVALID` after review.

{fold_lines}

## Artifacts

- OOF: `{result['oof_path']}`
- TEST: `{result['test_path'] or 'not generated'}`
"""
    path.write_text(text, encoding="utf-8")
