from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OUT = ROOT / "research" / "new_directions" / "EXP072_LWA_TAB"
PROCESSED = OLD / "data" / "processed"
RAW = OLD / "data" / "raw" / "train.parquet"
SAMPLE = OLD / "data" / "raw" / "sample_submit.csv"
ALIGNED_OOF = GEO / "gpt_pro_research_packet" / "06_ALIGNED_OOF.parquet"
ALIGNED_TEST = GEO / "gpt_pro_research_packet" / "07_ALIGNED_TEST.parquet"
EXP069 = ROOT / "research" / "new_directions" / "EXP069_BTYD05_FRESH1_PROD"
EXP071 = ROOT / "research" / "new_directions" / "EXP071_ETX_FRESH_CONTRAST"
NEXT069 = ROOT / "research" / "new_directions" / "NEXT_SUBMISSION_AFTER_EXP069"

FOLDS = [dt.date(2025, 9, 4), dt.date(2025, 9, 18), dt.date(2025, 10, 2), dt.date(2025, 10, 16)]
FOLD_NAMES = [str(x) for x in FOLDS]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0])
FOLD_SIZES = [188_518, 191_025, 193_694, 197_379]
PILOT_FOLD = FOLDS[-1]
EXTRA_CUTOFFS = [
    dt.date(2025, 10, 22), dt.date(2025, 10, 29), dt.date(2025, 11, 5),
    dt.date(2025, 11, 12), dt.date(2025, 11, 19), dt.date(2025, 11, 26),
    dt.date(2025, 12, 3), dt.date(2025, 12, 10), dt.date(2025, 12, 17),
    dt.date(2025, 12, 24), dt.date(2025, 12, 31), dt.date(2026, 1, 7),
    dt.date(2026, 1, 14),
]
TEST_CUTOFF = dt.date(2026, 2, 13)
ALPHA_GRID = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
ADD_ONE_GRID = np.round(np.arange(0.0, 2.0001, 0.05), 10)
SEED = 42
ARMS = ("CLEAN", "FRESH", "FRESH_NOOV", "VOL")
BOOTSTRAPS = 1_000
PILOT_BUDGET_SECONDS = 50 * 60
PILOT_HARD_STOP_SECONDS = int(1.5 * PILOT_BUDGET_SECONDS)

# Recovered verbatim from OLD/experiments/exp_013_s1_e11_two_part.md and
# OLD/src/config.py:LGB_PARAMS.  The positive regressor used 600 rounds.
LGB_ROUNDS = 600
LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": SEED,
    "num_threads": int(os.environ.get("LGB_THREADS", "12")),
    "max_bin": 63,
    "force_row_wise": True,
    "deterministic": True,
}


def log(*parts: object) -> None:
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)


def write_json_new(path: Path, value: object) -> None:
    write_text_new(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "EMPTY"}]
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v
                             for k, v in row.items()})


def feature_path(cutoff: dt.date) -> Path:
    return PROCESSED / f"feat_{cutoff:%Y%m%d}_LnormNone.parquet"


def panel_path(cutoff: dt.date, blocks: int = 3) -> Path:
    return PROCESSED / f"panel_{cutoff:%Y%m%d}_b{blocks}.parquet"


def weekly_grid(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    value = start
    while value <= end:
        out.append(value)
        value += dt.timedelta(days=7)
    return out


CLEAN_GRID = weekly_grid(dt.date(2025, 4, 3), dt.date(2025, 10, 16))
REQUIRED_FEATURE_CUTOFFS = sorted(set(CLEAN_GRID + EXTRA_CUTOFFS + FOLDS + [TEST_CUTOFF]))


def clean_cutoffs(validation_cutoff: dt.date) -> list[dt.date]:
    return [x for x in CLEAN_GRID if x + dt.timedelta(days=30) <= validation_cutoff]


def noov_cutoffs(validation_cutoff: dt.date) -> list[dt.date]:
    # The protocol gives both an interval formula and an explicit controlling
    # membership/count.  They disagree for V=2025-10-02: the literal formula
    # also overlaps 2025-10-29 for three days, but the fixed specification says
    # 12 and drops only 2025-10-22.  Because arms may not be redesigned, use the
    # explicit membership exactly as preregistered.
    drops = {
        dt.date(2025, 9, 4): set(),
        dt.date(2025, 9, 18): set(),
        dt.date(2025, 10, 2): {dt.date(2025, 10, 22)},
        dt.date(2025, 10, 16): {
            dt.date(2025, 10, 22), dt.date(2025, 10, 29),
            dt.date(2025, 11, 5), dt.date(2025, 11, 12),
        },
    }
    if validation_cutoff not in drops:
        raise ValueError(f"NOOV membership is not preregistered for {validation_cutoff}")
    return [cutoff for cutoff in EXTRA_CUTOFFS if cutoff not in drops[validation_cutoff]]


def user_side(user_ids: np.ndarray) -> np.ndarray:
    x = np.asarray(user_ids).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(1)).astype(np.int8)


def calibrate(y: np.ndarray, z: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    ly = np.log1p(y)
    weights = np.ones(len(y), float) if weights is None else np.asarray(weights, float)
    denominator = float(weights.sum())
    offset = float(np.dot(weights, ly - z) / denominator)
    for _ in range(25):
        active = z + offset > 0
        active_weights = weights[active]
        if not len(active_weights) or float(active_weights.sum()) == 0.0:
            break
        updated = float(np.dot(active_weights, ly[active] - z[active]) / active_weights.sum())
        if abs(updated - offset) < 1e-12:
            offset = updated
            break
        offset = updated
    residual = ly - np.maximum(z + offset, 0.0)
    score = float(np.sqrt(np.dot(weights, residual * residual) / denominator))
    return offset, score


def evaluate(y: np.ndarray, z: np.ndarray, fold: np.ndarray) -> dict[str, Any]:
    scores, offsets, sizes = [], [], []
    for fold_name in FOLD_NAMES:
        mask = fold == fold_name
        offset, score = calibrate(y[mask], z[mask])
        offsets.append(offset)
        scores.append(score)
        sizes.append(int(mask.sum()))
    values = np.asarray(scores)
    return {
        "fold_scores": values,
        "offsets": offsets,
        "sizes": sizes,
        "wcv": float(FOLD_WEIGHTS @ values / FOLD_WEIGHTS.sum()),
    }


def align_npz_component(
    artifact: dict[str, np.ndarray], user_id: np.ndarray, fold: np.ndarray, y: np.ndarray
) -> np.ndarray:
    source_fold = np.asarray(artifact["cutoff"], dtype="U10")
    source_user = np.asarray(artifact["user_id"], np.int64)
    source_y = np.asarray(artifact["y"], np.float32)
    source_z = np.asarray(artifact["z"], float)
    aligned = np.empty(len(y), float)
    for name in FOLD_NAMES:
        dst = np.flatnonzero(fold == name)
        src = np.flatnonzero(source_fold == name)
        dst_order = np.argsort(user_id[dst])
        src_order = np.argsort(source_user[src])
        if not np.array_equal(user_id[dst][dst_order], source_user[src][src_order]):
            raise AssertionError(f"component keys differ on {name}")
        if not np.array_equal(y[dst][dst_order].astype(np.float32), source_y[src][src_order]):
            raise AssertionError(f"component targets differ on {name}")
        aligned[dst[dst_order]] = source_z[src][src_order]
    return aligned


def nested_add_one(y: np.ndarray, fold: np.ndarray, z_base: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
    base = evaluate(y, z_base, fold)
    held_scores = np.empty(4)
    selected: list[float] = []
    for h, heldout in enumerate(FOLD_NAMES):
        donors = [i for i in range(4) if i != h]
        ranked: list[tuple[float, float]] = []
        for alpha in ADD_ONE_GRID:
            donor_scores = []
            for i in donors:
                mask = fold == FOLD_NAMES[i]
                donor_scores.append(calibrate(y[mask], z_base[mask] + float(alpha) * direction[mask])[1])
            donor_weights = FOLD_WEIGHTS[donors]
            ranked.append((float(donor_weights @ np.asarray(donor_scores) / donor_weights.sum()), float(alpha)))
        _, alpha = min(ranked, key=lambda item: (item[0], item[1]))
        mask = fold == heldout
        held_scores[h] = calibrate(y[mask], z_base[mask] + alpha * direction[mask])[1]
        selected.append(alpha)
    deltas = held_scores - base["fold_scores"]
    return {
        "selected_alpha": selected,
        "heldout_scores": held_scores.tolist(),
        "heldout_deltas": deltas.tolist(),
        "delta_wcv": float(FOLD_WEIGHTS @ deltas / FOLD_WEIGHTS.sum()),
        "improved_folds": int(np.sum(deltas < 0)),
    }


def derive_panel_b3(events: pl.DataFrame, cutoff: dt.date) -> pl.DataFrame:
    age = (pl.lit(cutoff) - pl.col("event_date")).dt.total_days()
    return (
        events.lazy()
        .filter((pl.col("event_date") >= cutoff - dt.timedelta(days=89)) & (pl.col("event_date") <= cutoff))
        .with_columns((age // 30).cast(pl.Int8).alias("_block"))
        .group_by("user_id")
        .agg(pl.col("_block").n_unique().alias("_n_blocks"))
        .filter(pl.col("_n_blocks") == 3)
        .select("user_id")
        .sort("user_id")
        .collect()
    )


def load_or_derive_panel_b3(events: pl.DataFrame, cutoff: dt.date) -> tuple[pl.DataFrame, str]:
    path = panel_path(cutoff, 3)
    if path.exists():
        return pl.read_parquet(path).select("user_id").sort("user_id"), "canonical_cache"
    return derive_panel_b3(events, cutoff), "canonical_in_memory_rebuild"


def manifest_paths() -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    required_text = [
        ROOT / "README.md", ROOT / "config" / "competition.yaml", ROOT / "config" / "paths.local.yaml",
        ROOT / "src" / "features" / "canonical.py", ROOT / "src" / "data" / "loaders.py",
        ROOT / "src" / "validation" / "folds.py", ROOT / "src" / "validation" / "evaluate.py",
        ROOT / "src" / "validation" / "workflow.py", ROOT / "src" / "metrics" / "rmsle.py",
        ROOT / "src" / "models" / "tabular.py", ROOT / "registry" / "experiments.csv",
        ROOT / "registry" / "models.csv", ROOT / "registry" / "submissions.csv",
        EXP069 / "reconnaissance.md", EXP069 / "report.md", EXP069 / "config.json",
        EXP069 / "run_oof_analysis.py", EXP069 / "train_production_fresh.py",
        EXP069 / "preprocessing_parameters.json", EXP069 / "production_training_audit.json",
        EXP071 / "report.md", EXP071 / "reconnaissance.md", NEXT069 / "report.md",
        NEXT069 / "oof_component_metrics.csv", NEXT069 / "oof_nested_alpha.csv",
        NEXT069 / "exp069_component_decomposition.csv", OLD / "src" / "fresh_contrast.py",
        OLD / "src" / "seq_cond.py", OLD / "src" / "config.py", OLD / "src" / "models.py",
        OLD / "experiments" / "exp_013_s1_e11_two_part.md",
        GEO / "gpt_pro_research_packet" / "06_ALIGNED_OOF_COLUMNS.md",
        GEO / "gpt_pro_research_packet" / "07_ALIGNED_TEST_COLUMNS.md",
        GEO / "gpt_pro_research_packet" / "15_VALIDATION_PROTOCOL.md",
        GEO / "gpt_pro_research_packet" / "16_DO_NOT_REPEAT.md",
        GEO / "submission_geometry" / "core.py", GEO / "submission_geometry" / "directions.py",
        GEO / "submission_geometry" / "geomlib.py", GEO / "submission_geometry" / "cache" / "Z_meta.json",
    ]
    paths.extend((path, "required_read") for path in required_text)
    paths.extend([(RAW, "canonical_raw_events"), (SAMPLE, "canonical_sample_submission")])
    paths.extend([(ALIGNED_OOF, "canonical_aligned_oof"), (ALIGNED_TEST, "canonical_aligned_test")])
    paths.extend([
        (EXP069 / "fresh_conditional_OOF.parquet", "exp069_saved_oof_correction"),
        (EXP069 / "fresh_conditional_TEST.parquet", "exp069_frozen_test_p_dist"),
        (GEO / "submission_geometry" / "cache" / "Z.npz", "target_free_test_span"),
    ])
    for fold in FOLDS:
        paths.append((OLD / "artifacts" / f"FRESH_CONTRAST_MOE_fold_{fold:%Y%m%d}.npz", "exp069_exact_oof_p_dist"))
    for name in ["oof_S1-E03a.npz", "oof_S1-E02.npz", "oof_S1-DIST.npz", "oof_SEQ-AVG3.npz", "oof_ETX-AVG3.npz"]:
        paths.append((OLD / "artifacts" / name, "exp037_component_reconstruction"))
    for cutoff in REQUIRED_FEATURE_CUTOFFS:
        paths.append((feature_path(cutoff), "frozen_227_feature_cache"))
        cached_panel = panel_path(cutoff, 3)
        if cached_panel.exists():
            paths.append((cached_panel, "canonical_b3_panel_cache"))
    # Stable order and no duplicate paths.
    unique: dict[str, tuple[Path, str]] = {}
    for path, role in paths:
        unique.setdefault(str(path.resolve()).lower(), (path, role))
    return list(unique.values())


def path_shape(path: Path) -> tuple[int | None, int | None]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
            return rows, len(pl.read_parquet_schema(path))
        if suffix == ".csv":
            frame = pl.read_csv(path)
            return frame.height, frame.width
        if suffix == ".npz":
            data = np.load(path, allow_pickle=False)
            if "Z" in data.files:
                return int(data["Z"].shape[0]), int(data["Z"].shape[1])
            for key in ("uid", "user_id", "cutoff", "y"):
                if key in data.files:
                    return int(len(data[key])), len(data.files)
            return None, len(data.files)
    except Exception:
        return None, None
    return None, None


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (path, role) in enumerate(manifest_paths(), 1):
        if not path.exists():
            raise FileNotFoundError(path)
        log(f"hash {index}/{len(manifest_paths())}: {path.name}")
        n_rows, n_columns = path_shape(path)
        stat = path.stat()
        rows.append({
            "path": str(path.resolve()), "role": role, "bytes": stat.st_size,
            "rows": n_rows, "columns": n_columns, "sha256": sha256(path),
        })
    return rows


def correction_parity() -> dict[str, Any]:
    path = EXP069 / "fresh_conditional_OOF.parquet"
    frame = pl.read_parquet(path)
    fold = frame["fold"].to_numpy()
    raw = frame["raw_correction"].to_numpy().astype(float)
    saved = frame["correction"].to_numpy().astype(float)
    reproduced = np.empty_like(raw)
    donor_bridge = np.empty_like(raw)
    rows = []
    for name in FOLD_NAMES:
        held = fold == name
        donor = ~held
        lo, hi = np.quantile(raw[donor], [0.005, 0.995])
        clipped_held = np.clip(raw[held], lo, hi)
        held_center = float(clipped_held.mean())
        donor_center = float(np.clip(raw[donor], lo, hi).mean())
        reproduced[held] = clipped_held - held_center
        donor_bridge[held] = clipped_held - donor_center
        rows.append({
            "fold": name, "q005_from_other_folds": float(lo), "q995_from_other_folds": float(hi),
            "saved_historical_center_from_held_clipped_mean": held_center,
            "new_protocol_donor_center": donor_center,
            "max_abs_error_historical_reproduction": float(np.max(np.abs(reproduced[held] - saved[held]))),
            "rms_difference_donor_center_bridge_vs_saved": float(np.sqrt(np.mean((donor_bridge[held] - saved[held]) ** 2))),
        })
    raw_from_saved_cond = frame["z_cond_fresh"].to_numpy().astype(float) - frame["z_cond_clean"].to_numpy().astype(float)
    side_error = int(np.sum(frame["user_side"].to_numpy().astype(np.int8) != user_side(frame["user_id"].to_numpy())))
    result = {
        "status": "PASS",
        "source": str(path),
        "source_sha256": sha256(path),
        "rows": frame.height,
        "historical_saved_chain": [
            "raw_correction = z_cond_fresh - z_cond_clean",
            "raw already equals p_dist * (mu_fresh - mu_clean)",
            "fit q0.005/q0.995 on the other three folds",
            "clip held-fold raw correction",
            "GLOBAL variant (no HIGH16 gate)",
            "subtract the held fold's clipped mean",
            "alpha = 1",
        ],
        "new_exp072_chain": [
            "raw_correction = mu_arm - mu_clean",
            "fit q0.005/q0.995 and clipped mean on donor folds",
            "clip held-fold raw correction and subtract donor clipped mean",
            "multiply by frozen p_dist",
        ],
        "max_abs_error_saved_correction": float(np.max(np.abs(reproduced - saved))),
        "max_abs_error_raw_vs_z_cond_difference": float(np.max(np.abs(raw - raw_from_saved_cond))),
        "splitmix_side_mismatches": side_error,
        "folds": rows,
        "important_distinction": (
            "The saved EXP069 OOF vector centered each held fold on itself. EXP072 follows the newly "
            "specified donor-derived center. The difference is foldwise constant before p_dist only in "
            "the historical already-multiplied representation."
        ),
    }
    if result["max_abs_error_saved_correction"] > 1e-9 or side_error:
        result["status"] = "FAIL"
        raise AssertionError(json.dumps(result, indent=2))
    return result


def reconnaissance() -> dict[str, Any]:
    started = time.time()
    for name in ("reconnaissance.md", "config.json", "parity_exp069_replication.json", "artifact_manifest.csv"):
        if (OUT / name).exists():
            raise FileExistsError(f"refusing to overwrite {OUT / name}")

    # First substantive computation: exact EXP069 saved correction reproduction.
    parity = correction_parity()

    frame = pl.read_parquet(ALIGNED_OOF)
    fold = frame["fold"].to_numpy()
    user_id = frame["user_id"].to_numpy().astype(np.int64)
    y = frame["target"].to_numpy().astype(float)
    prediction_columns = [c for c in frame.columns if c.startswith("pred_")]
    duplicate_keys = int(frame.select(["fold", "user_id"]).is_duplicated().sum())
    sizes = [int(np.sum(fold == name)) for name in FOLD_NAMES]
    predictions_ok = all(
        np.isfinite(frame[c].to_numpy()).all() and np.all(frame[c].to_numpy() >= 0)
        for c in prediction_columns
    )
    if frame.height != 770_616 or duplicate_keys != 0 or sizes != FOLD_SIZES or not predictions_ok:
        raise AssertionError("canonical aligned OOF schema audit failed")

    z_base = np.log1p(frame["pred_exp037"].to_numpy().astype(float))
    base = evaluate(y, z_base, fold)
    expected_folds = np.asarray([1.7668834, 1.7605096, 1.7486292, 1.7412786])
    if abs(base["wcv"] - 1.747509867) > 5e-8 or np.max(np.abs(base["fold_scores"] - expected_folds)) > 5e-7:
        raise AssertionError(f"canonical evaluator parity failed: {base}")

    nested: dict[str, Any] = {}
    expectations = {
        "pred_fresh_contrast": (-0.000225, 4, 5e-6),
        "pred_btyd": (-0.000269, 4, 5e-6),
        "pred_hurdle_e11": (0.000005, 1, 1e-5),
    }
    for column, (expected_delta, expected_improved, tolerance) in expectations.items():
        z_source = np.log1p(frame[column].to_numpy().astype(float))
        result = nested_add_one(y, fold, z_base, z_source - z_base)
        result["expected_delta_rounded"] = expected_delta
        result["delta_error_vs_expected"] = result["delta_wcv"] - expected_delta
        nested[column] = result
        if abs(result["delta_wcv"] - expected_delta) > tolerance or result["improved_folds"] != expected_improved:
            raise AssertionError(f"add-one parity failed for {column}: {result}")

    component_map = {
        "pred_cap": (OLD / "artifacts" / "oof_S1-E03a.npz", 0.10),
        "pred_unc": (OLD / "artifacts" / "oof_S1-E02.npz", 0.20),
        "pred_dist": (OLD / "artifacts" / "oof_S1-DIST.npz", 0.25),
        "pred_seq_avg3": (OLD / "artifacts" / "oof_SEQ-AVG3.npz", 0.225),
        "pred_etx_avg3": (OLD / "artifacts" / "oof_ETX-AVG3.npz", 0.225),
    }
    reconstructed = np.zeros(len(y), float)
    component_errors = []
    for column, (path, weight) in component_map.items():
        component = dict(np.load(path, allow_pickle=False))
        source_z = align_npz_component(component, user_id, fold, y)
        aligned_z = np.log1p(frame[column].to_numpy().astype(float))
        component_errors.append({"column": column, "path": str(path),
                                 "max_aligned_log_error": float(np.max(np.abs(source_z - aligned_z)))})
        reconstructed += weight * source_z
    reconstruction_error = float(np.max(np.abs(reconstructed - z_base)))
    if reconstruction_error > 1e-6:
        raise AssertionError(f"EXP037 reconstruction error {reconstruction_error}")

    schemas: list[dict[str, Any]] = []
    reference_schema: list[tuple[str, str]] | None = None
    missing_features = []
    for cutoff in REQUIRED_FEATURE_CUTOFFS:
        path = feature_path(cutoff)
        if not path.exists():
            missing_features.append(str(path))
            continue
        schema = [(name, str(dtype)) for name, dtype in pl.read_parquet_schema(path).items()]
        rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        if reference_schema is None:
            reference_schema = schema
        if schema != reference_schema:
            raise AssertionError(f"feature schema/order/dtype mismatch at {cutoff}")
        if len(schema) != 228 or schema[0][0] != "user_id":
            raise AssertionError(f"expected user_id + 227 frozen features at {cutoff}, got {len(schema)}")
        schemas.append({"cutoff": str(cutoff), "path": str(path), "rows": rows, "columns_total": len(schema),
                        "feature_columns": len(schema) - 1})
    if missing_features:
        raise FileNotFoundError(f"missing feature caches (must rebuild only through build_features): {missing_features}")

    # The review packet claimed all EXTRA b3 caches existed. They do not. Rebuild
    # the exact canonical eligibility set in memory and verify the adapter against
    # two existing canonical caches. Nothing is written outside EXP072.
    events_for_panel = pl.read_parquet(RAW, columns=["user_id", "event_date"])
    panel_findings: list[dict[str, Any]] = []
    for cutoff in EXTRA_CUTOFFS:
        path = panel_path(cutoff, 3)
        panel, source = load_or_derive_panel_b3(events_for_panel, cutoff)
        panel_findings.append({"cutoff": str(cutoff), "expected_path": str(path), "cache_exists": path.exists(),
                               "source_used": source, "rows": panel.height})
    for cutoff in (PILOT_FOLD, TEST_CUTOFF):
        derived = derive_panel_b3(events_for_panel, cutoff)
        cached = pl.read_parquet(panel_path(cutoff, 3)).select("user_id").sort("user_id")
        if not np.array_equal(derived["user_id"].to_numpy(), cached["user_id"].to_numpy()):
            raise AssertionError(f"in-memory canonical panel adapter differs at {cutoff}")
    del events_for_panel
    gc.collect()

    fresh_oof = pl.read_parquet(EXP069 / "fresh_conditional_OOF.parquet")
    fresh_test = pl.read_parquet(EXP069 / "fresh_conditional_TEST.parquet")
    aligned_test_schema = pl.read_parquet_schema(ALIGNED_TEST)
    if fresh_oof.height != 770_616 or fresh_oof.width != 15:
        raise AssertionError(f"EXP069 OOF prerequisite differs: {fresh_oof.shape}")
    if fresh_test.height != 250_000 or fresh_test.width != 13 or "p_dist" not in fresh_test.columns:
        raise AssertionError(f"EXP069 TEST prerequisite differs: {fresh_test.shape}, {fresh_test.columns}")
    if int(pl.scan_parquet(ALIGNED_TEST).select(pl.len()).collect().item()) != 250_000:
        raise AssertionError("aligned TEST row count differs")

    geometry = np.load(GEO / "submission_geometry" / "cache" / "Z.npz", allow_pickle=False)
    meta = json.loads((GEO / "submission_geometry" / "cache" / "Z_meta.json").read_text(encoding="utf-8"))
    names = list(meta["names"])
    kept = [name for name in names if name not in {"C_lgbm_exp015_regen.csv", "submission_BTYD05.csv"}]
    if geometry["Z"].shape != (67, 250_000) or len(names) != 67 or len(kept) != 65:
        raise AssertionError("geometry cache prerequisite differs")

    clean_counts = {str(fold_date): len(clean_cutoffs(fold_date)) for fold_date in FOLDS}
    noov_counts = {str(fold_date): len(noov_cutoffs(fold_date)) for fold_date in FOLDS}
    if list(clean_counts.values()) != [18, 20, 22, 24] or list(noov_counts.values()) != [13, 13, 12, 9]:
        raise AssertionError(f"cutoff construction differs: clean={clean_counts}, noov={noov_counts}")

    manifest = build_manifest()
    config = {
        "experiment": "EXP072_LWA_TAB",
        "hypothesis": (
            "Replacing the frozen-encoder conditional-positive head with full-capacity tabular LightGBM "
            "on the frozen 227 LnormNone features scales the legal late-window amount channel."
        ),
        "folds": FOLD_NAMES, "fold_weights": FOLD_WEIGHTS.tolist(), "pilot_fold": str(PILOT_FOLD),
        "clean_cutoff_counts": clean_counts, "extra_cutoffs": [str(x) for x in EXTRA_CUTOFFS],
        "noov_cutoff_counts": noov_counts, "arms": list(ARMS),
        "target": "z_pos=log1p(y30), restricted to y30>0; no per-cutoff target centering",
        "features": "227 frozen feat_{T}_LnormNone columns, unchanged order/dtype, cast to float32 only for LightGBM matrix input",
        "crossfit": "splitmix64(user_id)&1; donor side s predicts recipient side 1-s",
        "vol_control": "same number of rows as EXTRA; replacement sampling from earliest floor(n_clean_cutoffs/3) cutoff slots; RNG seed 42",
        "correction": "winsor raw mu difference with donor-fold q0.005/q0.995; subtract donor clipped mean; multiply frozen EXP069 p_dist",
        "estimator_provenance": {
            "source": str(OLD / "experiments" / "exp_013_s1_e11_two_part.md"),
            "parameter_source": str(OLD / "src" / "config.py"),
            "description": "historical S1-E11 positive-part LightGBM regressor, recovered verbatim",
        },
        "lightgbm": {"num_boost_round": LGB_ROUNDS, **LGB_PARAMS},
        "alpha_grid_nested": ALPHA_GRID.tolist(), "bootstrap_user_clusters": BOOTSTRAPS,
        "panel_cache_adaptation": (
            "13 EXTRA b3 cache files claimed by the review packet are absent. Exact canonical b3 eligibility "
            "is reconstructed in memory from raw events; no external cache is written."
        ),
        "noov_membership_note": (
            "The stated interval formula would retain 11 EXTRA cutoffs for V=2025-10-02, but the protocol "
            "explicitly fixes 12 and drops only 2025-10-22. The explicit 13/13/12/9 membership controls."
        ),
        "public_lb_used": False, "test_inference_authorized": False,
    }

    result = {
        "status": "PASS", "runtime_seconds": time.time() - started,
        "aligned_oof": {"rows": frame.height, "duplicate_keys": duplicate_keys, "fold_sizes": sizes,
                        "prediction_columns": prediction_columns, "all_predictions_finite_nonnegative": predictions_ok},
        "evaluator": {"wcv": base["wcv"], "per_fold": base["fold_scores"].tolist(),
                      "expected_wcv": 1.747509867, "expected_per_fold_rounded": expected_folds.tolist()},
        "add_one_nested_reproduction": nested,
        "exp037_reconstruction_max_log_error": reconstruction_error,
        "component_alignment": component_errors,
        "exp069_correction_replication": parity,
        "features": {"required_cache_count": len(REQUIRED_FEATURE_CUTOFFS), "missing": missing_features,
                     "schema_identical": True, "columns_total": 228, "feature_columns": 227, "cutoffs": schemas},
        "panels": {"extra_cache_discrepancy_count": int(sum(not r["cache_exists"] for r in panel_findings)),
                   "adapter_parity_checks": [str(PILOT_FOLD), str(TEST_CUTOFF)], "extra": panel_findings},
        "prerequisites": {"fresh_oof_shape": fresh_oof.shape, "fresh_test_shape": fresh_test.shape,
                          "aligned_test_columns": len(aligned_test_schema), "geometry_Z_shape": geometry["Z"].shape,
                          "geometry_names": len(names), "geometry_unique_after_drops": len(kept)},
        "manifest_inputs": len(manifest), "public_lb_used": False,
    }

    lines = [
        "# EXP072 LWA TAB — reconnaissance", "", "## Status", "", "**PASS**", "",
        "All evaluator, row-key, feature-schema, cutoff-count, EXP-037 reconstruction, and EXP069 correction-parity gates passed. No model was trained during reconnaissance.", "",
        "## First substantive check: EXP069 correction", "",
        f"The saved correction was reproduced with maximum absolute error `{parity['max_abs_error_saved_correction']:.3e}` (required `<=1e-9`).",
        "The saved `raw_correction` equals `z_cond_fresh-z_cond_clean` exactly and already contains `p_dist`. Historical saved OOF used donor-fold winsor bounds but held-fold centering. EXP072 uses the newly specified donor-derived center and multiplies frozen `p_dist` after processing the raw mu difference.", "",
        "## Canonical OOF/evaluator parity", "",
        f"- Rows: `{frame.height:,}`; duplicate `(fold,user_id)` keys: `{duplicate_keys}`; fold sizes: `{sizes}`.",
        f"- EXP-037 wCV: `{base['wcv']:.12f}`; folds: `{base['fold_scores'].tolist()}`.",
        f"- EXP-037 reconstruction maximum log error: `{reconstruction_error:.3e}`.",
    ]
    for column, value in nested.items():
        lines.append(f"- Add-one `{column}`: nested delta `{value['delta_wcv']:+.9f}`, improved `{value['improved_folds']}/4`, alphas `{value['selected_alpha']}`.")
    lines.extend([
        "", "## Frozen features and cutoff construction", "",
        f"- `{len(REQUIRED_FEATURE_CUTOFFS)}` required feature caches exist; each has `user_id + 227` columns with byte-identical column order and dtype schema.",
        f"- CLEAN cutoff counts are `{clean_counts}`; NOOV counts are `{noov_counts}`.",
        "- NOOV specification discrepancy: literal interval intersection would keep 11 cutoffs for `2025-10-02`, but the fixed arm definition requires 12 and drops only `2025-10-22`; the explicit membership was used.",
        "- Every feature cache is the canonical `feat_*_LnormNone.parquet` output. No feature was rebuilt or written.", "",
        "## Filesystem adaptation: EXTRA b3 panels", "",
        f"The review packet stated that the 13 EXTRA `panel_*_b3.parquet` caches existed, but `{sum(not r['cache_exists'] for r in panel_findings)}` are absent. The experiment therefore applies the exact `panel_users` three-block rule in memory from the canonical raw events. The adapter was equality-checked against the existing `{PILOT_FOLD}` and `{TEST_CUTOFF}` b3 caches. No cache is written outside EXP072.", "",
        "| EXTRA cutoff | b3 rows | source |", "|---|---:|---|",
    ])
    lines.extend(f"| {r['cutoff']} | {r['rows']:,} | {r['source_used']} |" for r in panel_findings)
    lines.extend([
        "", "## Estimator", "",
        "The S1-E11 positive regressor configuration was recovered from `exp_013_s1_e11_two_part.md` and `src/config.py`: the frozen default LightGBM regression parameters for 600 rounds. The same configuration is fixed for all four arms and both user sides; no sweep or early stopping is permitted.", "",
        "## Input integrity", "",
        f"`artifact_manifest.csv` records `{len(manifest)}` independently SHA256-hashed inputs with row/column counts where applicable. Geometry scores and weights were not read or used. Public-LB use: **false**.", "",
    ])

    write_json_new(OUT / "parity_exp069_replication.json", parity)
    write_json_new(OUT / "config.json", config)
    write_csv_new(OUT / "artifact_manifest.csv", manifest)
    write_text_new(OUT / "reconnaissance.md", "\n".join(lines))
    log(json.dumps({"status": "PASS", "runtime_seconds": result["runtime_seconds"],
                    "wcv": base["wcv"], "nested": nested}, indent=2))
    return result


def target_for_users(events: pl.DataFrame, cutoff: dt.date, users: pl.DataFrame) -> np.ndarray:
    observed = (
        events.lazy()
        .filter((pl.col("event_date") >= cutoff + dt.timedelta(days=1))
                & (pl.col("event_date") <= cutoff + dt.timedelta(days=30))
                & (pl.col("gmv") > 0))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("y_true"))
        .collect()
    )
    return (
        users.select("user_id")
        .join(observed, on="user_id", how="left")
        .with_columns(pl.col("y_true").fill_null(0.0))
        .sort("user_id")["y_true"].to_numpy().astype(float)
    )


def cutoff_positive_metadata(events: pl.DataFrame, cutoff: dt.date) -> dict[str, Any]:
    users, source = load_or_derive_panel_b3(events, cutoff)
    y = target_for_users(events, cutoff, users)
    uid = users["user_id"].to_numpy().astype(np.int64)
    positive = y > 0
    return {
        "cutoff": cutoff, "uid": uid[positive], "z": np.log1p(y[positive]).astype(np.float32),
        "side": user_side(uid[positive]), "panel_rows": len(uid), "positive_rows": int(positive.sum()),
        "panel_source": source,
    }


def load_feature_rows(cutoff: dt.date, uid: np.ndarray, columns: list[str]) -> np.ndarray:
    requested = pl.DataFrame({"user_id": np.asarray(uid, np.int64)})
    features = pl.read_parquet(feature_path(cutoff), columns=["user_id", *columns])
    aligned = requested.join(features, on="user_id", how="left")
    if aligned.height != len(uid) or not np.array_equal(aligned["user_id"].to_numpy(), uid):
        raise AssertionError(f"feature alignment failed at {cutoff}")
    return aligned.select(columns).to_numpy().astype(np.float32, copy=False)


def load_validation_frames(columns: list[str]) -> dict[str, dict[str, Any]]:
    aligned = pl.read_parquet(ALIGNED_OOF)
    out: dict[str, dict[str, Any]] = {}
    for fold_date in FOLDS:
        name = str(fold_date)
        frame = aligned.filter(pl.col("fold") == name)
        uid = frame["user_id"].to_numpy().astype(np.int64)
        p_source = np.load(OLD / "artifacts" / f"FRESH_CONTRAST_MOE_fold_{fold_date:%Y%m%d}.npz", allow_pickle=False)
        p_uid = p_source["uid"].astype(np.int64)
        order = np.argsort(p_uid)
        pos = np.searchsorted(p_uid[order], uid)
        if np.any(pos >= len(order)) or not np.array_equal(p_uid[order][pos], uid):
            raise AssertionError(f"p_dist alignment failed at {fold_date}")
        p_dist = p_source["p_dist"].astype(float)[order][pos]
        if not np.all(np.isfinite(p_dist)) or not np.all((p_dist >= 0) & (p_dist <= 1)):
            raise AssertionError(f"p_dist invalid at {fold_date}")
        out[name] = {
            "uid": uid, "side": user_side(uid), "X": load_feature_rows(fold_date, uid, columns),
            "y": frame["target"].to_numpy().astype(float),
            "z_base": np.log1p(frame["pred_exp037"].to_numpy().astype(float)),
            "p_dist": p_dist,
            "frame": frame,
        }
    return out


def assemble_side(
    side: int, clean_meta: list[dict[str, Any]], extra_meta: list[dict[str, Any]], columns: list[str]
) -> dict[str, np.ndarray]:
    clean_x, clean_y, clean_slot = [], [], []
    extra_x, extra_y, extra_slot = [], [], []
    for slot, item in enumerate(clean_meta):
        keep = item["side"] == side
        uid = item["uid"][keep]
        clean_x.append(load_feature_rows(item["cutoff"], uid, columns))
        clean_y.append(item["z"][keep])
        clean_slot.append(np.full(len(uid), slot, np.int16))
    for slot, item in enumerate(extra_meta):
        keep = item["side"] == side
        uid = item["uid"][keep]
        extra_x.append(load_feature_rows(item["cutoff"], uid, columns))
        extra_y.append(item["z"][keep])
        extra_slot.append(np.full(len(uid), slot, np.int16))
    return {
        "X_clean": np.concatenate(clean_x), "y_clean": np.concatenate(clean_y),
        "slot_clean": np.concatenate(clean_slot), "X_extra": np.concatenate(extra_x),
        "y_extra": np.concatenate(extra_y), "slot_extra": np.concatenate(extra_slot),
    }


def fit_lgb(X: np.ndarray, y: np.ndarray):
    import lightgbm as lgb
    if len(X) != len(y) or X.shape[1] != 227 or not np.all(np.isfinite(y)):
        raise AssertionError(f"invalid LightGBM training matrix {X.shape}/{y.shape}")
    dataset = lgb.Dataset(X, label=y, params=LGB_PARAMS, free_raw_data=True)
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=LGB_ROUNDS)
    del dataset
    return model


def donor_preprocess(raws: list[np.ndarray]) -> dict[str, Any]:
    pooled = np.concatenate([np.asarray(x, float) for x in raws])
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    center = float(np.clip(pooled, lo, hi).mean())
    return {"q005": float(lo), "q995": float(hi), "center": center, "n_donor": len(pooled)}


def apply_correction(raw_mu_difference: np.ndarray, p_dist: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    processed_mu = np.clip(raw_mu_difference, params["q005"], params["q995"]) - params["center"]
    return p_dist * processed_mu


def least_squares_unexplained(candidate: np.ndarray, source_matrix: np.ndarray) -> dict[str, Any]:
    y = np.asarray(candidate, float) - float(np.mean(candidate))
    X = np.asarray(source_matrix, float)
    X = X - X.mean(axis=0, keepdims=True)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coef
    ratio = float(np.var(residual) / np.var(y))
    return {"unexplained_variance_ratio": ratio, "coefficients": coef.tolist(),
            "candidate_rms_centered": float(np.sqrt(np.mean(y * y))),
            "residual_rms": float(np.sqrt(np.mean(residual * residual)))}


def rss_bytes() -> int:
    import psutil
    return int(psutil.Process().memory_info().rss)


def pilot() -> dict[str, Any]:
    started = time.time()
    if not (OUT / "reconnaissance.md").exists() or "**PASS**" not in (OUT / "reconnaissance.md").read_text(encoding="utf-8"):
        raise RuntimeError("reconnaissance has not passed")
    if (OUT / "pilot_metrics.json").exists() or (OUT / "report.md").exists():
        raise FileExistsError("refusing to overwrite pilot/final artifacts")

    schema = list(pl.read_parquet_schema(feature_path(PILOT_FOLD)).keys())
    columns = [name for name in schema if name != "user_id"]
    if len(columns) != 227:
        raise AssertionError(f"expected 227 features, got {len(columns)}")
    for cutoff in REQUIRED_FEATURE_CUTOFFS:
        current = [(name, str(dtype)) for name, dtype in pl.read_parquet_schema(feature_path(cutoff)).items()]
        reference = [(name, str(dtype)) for name, dtype in pl.read_parquet_schema(feature_path(PILOT_FOLD)).items()]
        if current != reference:
            raise AssertionError(f"feature schema changed at {cutoff}")

    log("loading canonical raw events for exact b3 panels and y30")
    events = pl.read_parquet(RAW, columns=["user_id", "event_date", "gmv"])
    pilot_clean = clean_cutoffs(PILOT_FOLD)
    pilot_noov = noov_cutoffs(PILOT_FOLD)
    if len(pilot_clean) != 24 or len(pilot_noov) != 9:
        raise AssertionError("pilot cutoff counts changed")
    clean_meta = []
    for index, cutoff in enumerate(pilot_clean, 1):
        item = cutoff_positive_metadata(events, cutoff)
        clean_meta.append(item)
        log(f"CLEAN metadata {index}/24 {cutoff}: panel={item['panel_rows']:,}, positive={item['positive_rows']:,}")
    extra_meta = []
    for index, cutoff in enumerate(EXTRA_CUTOFFS, 1):
        item = cutoff_positive_metadata(events, cutoff)
        extra_meta.append(item)
        log(f"EXTRA metadata {index}/13 {cutoff}: panel={item['panel_rows']:,}, positive={item['positive_rows']:,}")

    validation = load_validation_frames(columns)
    predictions = {arm: {name: np.full(len(validation[name]["uid"]), np.nan, float) for name in FOLD_NAMES}
                   for arm in ARMS}
    fit_rows: list[dict[str, Any]] = []
    peak_rss = rss_bytes()

    for donor_side in (0, 1):
        log(f"assembling frozen feature matrices for donor side {donor_side}")
        data = assemble_side(donor_side, clean_meta, extra_meta, columns)
        peak_rss = max(peak_rss, rss_bytes())
        n_clean, n_extra = len(data["y_clean"]), len(data["y_extra"])
        noov_slots = np.asarray([EXTRA_CUTOFFS[i] in pilot_noov for i in data["slot_extra"]], bool)
        early = np.flatnonzero(data["slot_clean"] < max(1, len(pilot_clean) // 3))
        if not len(early):
            raise AssertionError("VOL early-third pool is empty")
        rng = np.random.default_rng(SEED)
        vol_draw = rng.choice(early, size=n_extra, replace=True)
        arm_builders = {
            "CLEAN": lambda: (data["X_clean"], data["y_clean"]),
            "FRESH": lambda: (np.concatenate([data["X_clean"], data["X_extra"]]),
                              np.concatenate([data["y_clean"], data["y_extra"]])),
            "FRESH_NOOV": lambda: (np.concatenate([data["X_clean"], data["X_extra"][noov_slots]]),
                                   np.concatenate([data["y_clean"], data["y_extra"][noov_slots]])),
            "VOL": lambda: (np.concatenate([data["X_clean"], data["X_clean"][vol_draw]]),
                            np.concatenate([data["y_clean"], data["y_clean"][vol_draw]])),
        }
        for arm in ARMS:
            if time.time() - started > PILOT_HARD_STOP_SECONDS:
                raise RuntimeError("pilot exceeded 150% of its 50-minute budget; hard stop")
            X_train, y_train = arm_builders[arm]()
            expected = n_clean + (0 if arm == "CLEAN" else (int(noov_slots.sum()) if arm == "FRESH_NOOV" else n_extra))
            if len(y_train) != expected:
                raise AssertionError(f"{arm} matched-row construction failed")
            fit_started = time.time()
            log(f"fit side={donor_side} arm={arm}: rows={len(y_train):,}, features={X_train.shape[1]}, rounds={LGB_ROUNDS}")
            model = fit_lgb(X_train, y_train)
            fit_seconds = time.time() - fit_started
            fit_rows.append({"donor_side": donor_side, "recipient_side": 1 - donor_side, "arm": arm,
                             "rows": len(y_train), "rounds": LGB_ROUNDS, "seconds": fit_seconds})
            for fold_name in FOLD_NAMES:
                recipient = validation[fold_name]["side"] == (1 - donor_side)
                mu = np.asarray(model.predict(validation[fold_name]["X"][recipient]), float)
                predictions[arm][fold_name][recipient] = mu
            del model
            if arm != "CLEAN":
                del X_train, y_train
            gc.collect()
            peak_rss = max(peak_rss, rss_bytes())
            log(f"finished side={donor_side} arm={arm} in {fit_seconds:.1f}s")
        del data
        gc.collect()

    for arm in ARMS:
        for fold_name in FOLD_NAMES:
            if not np.all(np.isfinite(predictions[arm][fold_name])):
                raise AssertionError(f"incomplete cross-fit vector {arm}/{fold_name}")
    # Same exact p_dist vector is referenced by all arms; no arm-specific copy is fitted.
    p_hashes = {name: hashlib.sha256(np.ascontiguousarray(validation[name]["p_dist"]).view(np.uint8)).hexdigest()
                for name in FOLD_NAMES}

    raw: dict[str, list[np.ndarray]] = {}
    processed: dict[str, list[np.ndarray]] = {}
    preprocess: dict[str, dict[str, Any]] = {}
    for arm in ("FRESH", "FRESH_NOOV", "VOL"):
        raw[arm] = [predictions[arm][name] - predictions["CLEAN"][name] for name in FOLD_NAMES]
        preprocess[arm] = donor_preprocess(raw[arm][:-1])
        processed[arm] = [apply_correction(raw[arm][i], validation[name]["p_dist"], preprocess[arm])
                          for i, name in enumerate(FOLD_NAMES)]

    latest = validation[str(PILOT_FOLD)]
    base_score = calibrate(latest["y"], latest["z_base"])[1]
    scores = {"EXP037": base_score}
    for arm in ("FRESH", "FRESH_NOOV", "VOL"):
        scores[arm] = calibrate(latest["y"], latest["z_base"] + processed[arm][-1])[1]
    g1_value = scores["FRESH"] - scores["VOL"]
    g2_value = scores["FRESH"] - scores["EXP037"]
    g3_left = scores["FRESH_NOOV"] - scores["VOL"]
    g3_right = 0.5 * (scores["FRESH"] - scores["VOL"])

    source_columns = [
        "pred_cap", "pred_unc", "pred_dist", "pred_etx_avg3", "pred_seq_avg3",
        "pred_seq_d3a_avg3", "pred_ridge15", "pred_hurdle_e11", "pred_mhz_full",
        "pred_holiday_yoy", "pred_block4_saf", "pred_fresh_contrast", "pred_btyd",
    ]
    latest_frame = latest["frame"]
    directions = np.column_stack([
        np.log1p(latest_frame[c].to_numpy().astype(float)) - latest["z_base"] for c in source_columns
    ])
    projection = least_squares_unexplained(processed["FRESH"][-1], directions)
    d_exp069 = np.log1p(latest_frame["pred_fresh_contrast"].to_numpy().astype(float)) - latest["z_base"]
    corr_exp069 = float(np.corrcoef(processed["FRESH"][-1], d_exp069)[0, 1])

    gates = {
        "G1": {"pass": bool(g1_value <= -0.00010), "value": g1_value, "threshold": -0.00010,
               "margin": -0.00010 - g1_value, "formula": "score(FRESH)-score(VOL)"},
        "G2": {"pass": bool(g2_value < 0), "value": g2_value, "threshold": 0.0,
               "margin": -g2_value, "formula": "score(FRESH)-score(EXP037)"},
        "G3": {"pass": bool(g3_left <= g3_right), "left": g3_left, "right": g3_right,
               "margin": g3_right - g3_left,
               "formula": "score(FRESH_NOOV)-score(VOL) <= 0.5*(score(FRESH)-score(VOL))"},
        "G4_unexplained": {"pass": bool(projection["unexplained_variance_ratio"] >= 0.50),
                           "value": projection["unexplained_variance_ratio"], "threshold": 0.50,
                           "margin": projection["unexplained_variance_ratio"] - 0.50},
        "G4_corr_exp069": {"pass": bool(abs(corr_exp069) <= 0.85), "value": corr_exp069,
                           "absolute_value": abs(corr_exp069), "threshold": 0.85,
                           "margin": 0.85 - abs(corr_exp069)},
    }
    passed = all(item["pass"] for item in gates.values())
    pilot_result = {
        "status": "PASS" if passed else "REJECT", "fold": str(PILOT_FOLD), "alpha": 1.0,
        "scores": scores,
        "deltas": {"fresh_minus_vol": g1_value, "fresh_minus_exp037": g2_value,
                   "noov_minus_vol": g3_left, "half_fresh_minus_vol": g3_right},
        "gates": gates, "preprocessing": preprocess, "projection": projection,
        "corr_lwa_exp069_fresh": corr_exp069, "projection_sources": source_columns,
        "correction_rms": {arm: float(np.sqrt(np.mean(processed[arm][-1] ** 2)))
                           for arm in ("FRESH", "FRESH_NOOV", "VOL")},
        "raw_mu_difference_rms": {arm: float(np.sqrt(np.mean(raw[arm][-1] ** 2)))
                                  for arm in ("FRESH", "FRESH_NOOV", "VOL")},
        "fit_audit": fit_rows,
        "cutoff_row_audit": {
            "clean": [{k: (str(v) if k == "cutoff" else v) for k, v in item.items()
                       if k in {"cutoff", "panel_rows", "positive_rows", "panel_source"}} for item in clean_meta],
            "extra": [{k: (str(v) if k == "cutoff" else v) for k, v in item.items()
                       if k in {"cutoff", "panel_rows", "positive_rows", "panel_source"}} for item in extra_meta],
            "clean_cutoffs": len(pilot_clean), "extra_cutoffs": len(EXTRA_CUTOFFS), "noov_cutoffs": len(pilot_noov),
        },
        "leakage_assertions": {
            "feature_cache_cutoff_safe_by_canonical_builder": True,
            "all_clean_T_plus_30_le_V": all(x + dt.timedelta(days=30) <= PILOT_FOLD for x in pilot_clean),
            "extra_positive_only": all(np.all(item["z"] > 0) for item in extra_meta),
            "opposite_side_crossfit_full_coverage": True,
            "donor_fold_preprocessing": True,
            "p_dist_frozen_array_hash_by_fold": p_hashes,
            "p_dist_same_object_for_all_arms": True,
            "public_lb_used": False,
        },
        "runtime_seconds": time.time() - started, "budget_seconds": PILOT_BUDGET_SECONDS,
        "hard_stop_seconds": PILOT_HARD_STOP_SECONDS, "within_50_min_budget": time.time() - started <= PILOT_BUDGET_SECONDS,
        "peak_observed_rss_bytes": peak_rss,
    }
    write_json_new(OUT / "pilot_metrics.json", pilot_result)
    if not passed:
        finalize_pilot_reject(pilot_result)
    else:
        log("pilot PASS: full validation is authorized")
    del events, validation, predictions, clean_meta, extra_meta
    gc.collect()
    return pilot_result


def finalize_pilot_reject(result: dict[str, Any]) -> None:
    failed = [name for name, item in result["gates"].items() if not item["pass"]]
    score_rows = []
    for name, score in result["scores"].items():
        score_rows.append({"fold": str(PILOT_FOLD), "scope": "pilot_seed42_alpha1", "candidate": name,
                           "rmsle_cal": score, "delta_vs_exp037": score - result["scores"]["EXP037"]})
    write_csv_new(OUT / "fold_metrics.csv", score_rows)
    write_csv_new(OUT / "real_vs_vol.csv", [{"fold": str(PILOT_FOLD), "scope": "pilot_seed42_alpha1",
                                               "fresh_score": result["scores"]["FRESH"],
                                               "vol_score": result["scores"]["VOL"],
                                               "real_minus_vol": result["deltas"]["fresh_minus_vol"]}])
    denominator = result["deltas"]["fresh_minus_vol"]
    retention = (result["deltas"]["noov_minus_vol"] / denominator) if denominator != 0 else float("nan")
    write_csv_new(OUT / "noov_control.csv", [{"fold": str(PILOT_FOLD), "scope": "pilot_seed42_alpha1",
                                               "noov_minus_vol": result["deltas"]["noov_minus_vol"],
                                               "fresh_minus_vol": denominator, "retention_ratio": retention}])
    write_csv_new(OUT / "orthogonal_metrics.csv", [{"fold": str(PILOT_FOLD), "scope": "pilot",
                                                      "unexplained_variance_ratio": result["projection"]["unexplained_variance_ratio"],
                                                      "corr_lwa_exp069_fresh": result["corr_lwa_exp069_fresh"]}])
    skipped = [{"status": "SKIPPED_AFTER_PILOT_REJECT", "reason": ",".join(failed)}]
    for name in ("nested_selection.csv", "bootstrap_metrics.csv", "user_half_metrics.csv", "seed_robustness.csv", "diversity_oof.csv"):
        write_csv_new(OUT / name, skipped)
    write_json_new(OUT / "oof_projection_metrics.json", {
        "status": "SKIPPED_AFTER_PILOT_REJECT", "reason": failed,
        "pilot_target_free_projection": result["projection"], "corr_lwa_exp069_fresh": result["corr_lwa_exp069_fresh"],
    })
    runtime = {
        "reconnaissance_seconds": None,
        "pilot_seconds": result["runtime_seconds"], "pilot_budget_seconds": PILOT_BUDGET_SECONDS,
        "pilot_within_budget": result["within_50_min_budget"], "peak_observed_rss_bytes": result["peak_observed_rss_bytes"],
        "platform": platform.platform(), "python": sys.version, "public_lb_used": False,
    }
    write_json_new(OUT / "runtime_resources.json", runtime)
    gate_lines = []
    for name, item in result["gates"].items():
        gate_lines.append(f"- `{name}`: **{'PASS' if item['pass'] else 'FAIL'}** — `{json.dumps(item, ensure_ascii=False)}`")
    report = f"""# EXP072 LWA TAB — final report

## 1. Verdict

**REJECT** — pilot gate failure: `{', '.join(failed)}`.

The legal late-window conditional-amount channel does not scale in the preregistered full-capacity tabular form. Full four-fold validation, seed robustness, bootstrap analysis, OOF vector production, and all TEST inference were stopped exactly at the pilot gate.

Recommendation: **CLOSE_FAMILY**.

## 2. Pilot metrics and gate arithmetic

- EXP-037: `{result['scores']['EXP037']:.12f}`
- FRESH: `{result['scores']['FRESH']:.12f}`; delta vs EXP-037 `{result['deltas']['fresh_minus_exp037']:+.9f}`
- VOL: `{result['scores']['VOL']:.12f}`
- FRESH_NOOV: `{result['scores']['FRESH_NOOV']:.12f}`
- REAL - VOL: `{result['deltas']['fresh_minus_vol']:+.9f}`
- NOOV - VOL: `{result['deltas']['noov_minus_vol']:+.9f}`; required `<= {result['deltas']['half_fresh_minus_vol']:+.9f}`
- Unexplained variance: `{result['projection']['unexplained_variance_ratio']:.6f}`
- `corr(d_LWA,d_FRESH_EXP069)`: `{result['corr_lwa_exp069_fresh']:+.6f}`

{chr(10).join(gate_lines)}

The fixed-alpha pilot is the only predictive estimate authorized after rejection. Canonical four-fold wCV, nested delta, span-orthogonal nested delta, bootstrap interval, user-half full-validation results, and seed spread are **not estimated**.

## 3. Controls and diversity

The VOL arm used exactly the same number of additional rows as EXTRA, drawn with replacement from the earliest one-third of CLEAN positive cutoff slots with RNG seed 42. NOOV retained 9/13 EXTRA cutoffs. The pilot target-free least-squares projection used the 13 nonredundant aligned OOF directions and left `{result['projection']['unexplained_variance_ratio']:.3f}` of centered variance unexplained.

## 4. Leakage and provenance

All CLEAN cutoffs satisfy `T+30<=V`; EXTRA contributes only positive-target rows; each donor splitmix side predicts only the opposite recipient side; the exact EXP069 OOF `p_dist` vectors are frozen and shared byte-identically by all arms; bounds and centers come only from the three donor panels; no public-LB value, score, geometry weight, or reconstructed champion OOF entered a label, weight, bound, level, projection coefficient, or selection.

The review packet's 13 claimed EXTRA `panel_*_b3.parquet` caches were absent. Exact canonical three-block eligibility was reconstructed in memory from raw events and equality-checked against two existing b3 caches. No external cache was written.

## 5. Runtime and artifacts

Pilot wall time: `{result['runtime_seconds']:.1f}s` (budget `{PILOT_BUDGET_SECONDS}s`; within budget `{result['within_50_min_budget']}`). Peak observed RSS: `{result['peak_observed_rss_bytes']:,}` bytes. Temporary data remained in memory and was released; no persistent model or feature cache was created.

Input hashes and row counts are in `artifact_manifest.csv`. Output hashes are in `checksums.sha256`.

No `lwa_tab_OOF.parquet`, `lwa_tab_TEST.parquet`, `lwa_tab_TEST.csv`, or any other TEST vector exists.

## 6. Limitations

The run stopped on the single latest-fold pilot, so it cannot estimate four-fold stability, bootstrap uncertainty, user-half aggregate effects, seed spread, TEST span distance, or OOF/TEST magnitude parity. The frozen OOF `p_dist` is the exact EXP069 fold source; the saved EXP069 TEST `p_dist` was never read for inference. The canonical folds still end four months and one holiday season before TEST, but the early stop prevents any TEST claim.

## 7. Recommendation and next measurement

**CLOSE_FAMILY.** The single most informative next measurement would be a genuinely different legal late-window data channel with a preregistered matched-row control; another model, seed, round count, or feature manipulation on this same conditional-positive tabular channel is not authorized by this experiment.
"""
    write_text_new(OUT / "report.md", report)
    # Hash every newly produced artifact last. The checksum file does not hash itself.
    artifacts = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in artifacts]
    write_text_new(OUT / "checksums.sha256", "\n".join(checksum_lines) + "\n")


def auto() -> None:
    if not (OUT / "reconnaissance.md").exists():
        reconnaissance()
    result = pilot()
    if result["status"] == "PASS":
        raise NotImplementedError("pilot passed; invoke the separately reviewed full-validation continuation")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP072 fixed LWA tabular experiment")
    parser.add_argument("command", choices=["recon", "pilot", "auto"])
    args = parser.parse_args()
    {"recon": reconnaissance, "pilot": pilot, "auto": auto}[args.command]()


if __name__ == "__main__":
    main()
