"""Reproducible preflight for EXP074.

This program is deliberately read-only.  EXP074 may not train a replacement
encoder: the requested protocol requires the exact historical SEQ-AVG3 member
checkpoints and mandates TECHNICAL_BLOCK when they are absent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXP_ID = "EXP074"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configured_paths() -> tuple[Path, Path]:
    cfg = yaml.safe_load((ROOT / "config" / "paths.local.yaml").read_text(encoding="utf-8"))
    return Path(cfg["old_repo_root"]), Path(cfg["submission_geometry_root"])


def registry_hits(path: Path) -> list[int]:
    hits: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_no, row in enumerate(csv.reader(stream), start=1):
            normalized = "".join(row).upper().replace("_", "").replace("-", "")
            if EXP_ID in normalized:
                hits.append(line_no)
    return hits


def compare_oof(reference: Path, candidate: Path) -> dict[str, Any]:
    a = np.load(reference, allow_pickle=True)
    b = np.load(candidate, allow_pickle=True)
    assert np.array_equal(a["user_id"], b["user_id"])
    assert np.array_equal(a["y"], b["y"])
    d = np.asarray(b["z"], np.float64) - np.asarray(a["z"], np.float64)
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "rows": int(len(d)),
        "user_id_equal": True,
        "target_equal": True,
        "prediction_equal": bool(np.array_equal(a["z"], b["z"])),
        "max_abs_log_error": float(np.max(np.abs(d))),
        "rms_log_error": float(np.sqrt(np.mean(d * d))),
    }


def aligned_inputs_audit(packet: Path, sample_submit: Path) -> dict[str, Any]:
    oof_path = packet / "06_ALIGNED_OOF.parquet"
    test_path = packet / "07_ALIGNED_TEST.parquet"
    oof_required = [
        "user_id", "fold", "target", "pred_exp037", "pred_dist",
        "pred_seq_avg3", "pred_etx_avg3", "pred_seq_d3a_avg3",
        "pred_fresh_contrast", "pred_btyd", "pred_hurdle_e11",
        "pred_mhz_full", "pred_holiday_yoy",
    ]
    test_required = [
        "user_id", "pred_current_1_6466079084", "pred_dist",
        "pred_seq_avg3", "pred_etx_avg3",
    ]
    oof = pd.read_parquet(oof_path, columns=oof_required)
    test = pd.read_parquet(test_path, columns=test_required)
    sample = pd.read_csv(sample_submit, usecols=["user_id"])

    folds = tuple(sorted(oof["fold"].astype(str).unique()))
    oof_numeric = oof.drop(columns=["fold"]).to_numpy(dtype=np.float64)
    test_numeric = test.to_numpy(dtype=np.float64)
    assert len(oof) == 770_616
    assert folds == FOLDS
    assert not oof.duplicated(["fold", "user_id"]).any()
    assert np.isfinite(oof_numeric).all()
    assert (oof["target"] >= 0).all()
    assert (oof.filter(like="pred_") >= 0).all().all()
    assert len(test) == 250_000
    assert test["user_id"].is_unique
    assert np.isfinite(test_numeric).all()
    assert (test.filter(like="pred_") >= 0).all().all()
    assert np.array_equal(test["user_id"].to_numpy(), sample["user_id"].to_numpy())
    return {
        "oof_path": str(oof_path),
        "oof_sha256": sha256(oof_path),
        "oof_rows": int(len(oof)),
        "oof_folds": list(folds),
        "oof_required_columns_present": True,
        "oof_unique_keys": True,
        "oof_finite_nonnegative": True,
        "test_path": str(test_path),
        "test_sha256": sha256(test_path),
        "test_rows": int(len(test)),
        "test_required_columns_present": True,
        "test_unique_users": True,
        "test_finite_nonnegative": True,
        "test_matches_sample_order": True,
        "sample_submit": str(sample_submit),
        "sample_submit_sha256": sha256(sample_submit),
    }


def geometry_audit(geometry_root: Path) -> dict[str, Any]:
    geom = geometry_root / "submission_geometry"
    basis = geom / "cache" / "Z.npz"
    dropped = pd.read_csv(geom / "dropped_duplicates.csv")
    with np.load(basis, allow_pickle=False) as data:
        shape = list(data["Z"].shape)
        uid_shape = list(data["user_id"].shape)
    assert shape == [67, 250_000]
    assert uid_shape == [250_000]
    assert len(dropped) == 2
    return {
        "basis_path": str(basis),
        "basis_sha256": sha256(basis),
        "source_vectors": 67,
        "exact_duplicates": 2,
        "unique_sources": 65,
        "documented_difference_rank": 57,
        "rows": 250_000,
        "incumbent_path": str(geom / "SUBMIT_NEXT_BEST.csv"),
        "incumbent_sha256": sha256(geom / "SUBMIT_NEXT_BEST.csv"),
    }


def run() -> dict[str, Any]:
    old_root, geometry_root = configured_paths()
    artifacts = old_root / "artifacts"
    archive = old_root / "weights_archives" / "TCN_SEQ-01_weights.zip"
    packet = geometry_root / "gpt_pro_research_packet"
    registries = [ROOT / "registry" / "experiments.csv", packet / "02_EXPERIMENT_REGISTRY.csv"]
    registry_result = {str(path): registry_hits(path) for path in registries}
    assert all(not hits for hits in registry_result.values()), "EXP074 is already occupied"

    with zipfile.ZipFile(archive) as zf:
        archived = set(zf.namelist())
    expected_seed42 = [f"model_SEQ-01-S42-V{fold[5:7]}{fold[8:10]}.pt" for fold in FOLDS]
    expected_seed43 = [f"model_SEQ-01-S43-V{fold[5:7]}{fold[8:10]}.pt" for fold in FOLDS]
    expected_seed44 = [f"model_SEQ-01-S44-V{fold[5:7]}{fold[8:10]}.pt" for fold in FOLDS]
    expected_seed42_production = ["model_SEQ-01-S42-TEST.pt", "model_SEQ-C289-S42-TEST.pt"]

    def availability(names: list[str]) -> dict[str, bool]:
        return {name: bool((artifacts / name).exists() or name in archived) for name in names}

    seed42 = availability(expected_seed42)
    seed43 = availability(expected_seed43)
    seed44 = availability(expected_seed44)
    seed42_production = availability(expected_seed42_production)
    assert not any(seed42.values())
    assert not any(seed42_production.values())
    assert not seed43["model_SEQ-01-S43-V1016.pt"]
    assert all(seed44.values())

    substitutes = [
        compare_oof(
            artifacts / "oof_SEQ-01-S42-V1016.npz",
            artifacts / "oof_SEQ-D3A-BASE-S42-V1016.npz",
        ),
        compare_oof(
            artifacts / "oof_SEQ-01-S42-V1016.npz",
            artifacts / "oof_SEQ-03A-BASE-S42-V1016.npz",
        ),
        compare_oof(
            artifacts / "oof_SEQ-01-S42-V0904.npz",
            artifacts / "oof_SEQ-01C-S42-V0904.npz",
        ),
    ]
    assert all(x["max_abs_log_error"] > 1e-6 for x in substitutes)

    sources = {
        name: {
            "path": str(old_root / "src" / name),
            "sha256": sha256(old_root / "src" / name),
        }
        for name in ("models.py", "config.py", "seq.py", "seq_cond.py")
    }
    aligned = aligned_inputs_audit(packet, old_root / "data" / "raw" / "sample_submit.csv")
    geometry = geometry_audit(geometry_root)

    result = {
        "experiment_id": EXP_ID,
        "verdict": "TECHNICAL_BLOCK",
        "registry": {"status": "FREE", "hits": registry_result},
        "historical_dist16": {
            "bins": 16,
            "zero_edge": 1e-9,
            "positive_edges": "np.quantile(z[z>0], np.linspace(0,1,16)[1:-1])",
            "labels": "np.searchsorted(edges, z, side='right')",
            "centres": "training-row mean z per class; empty class inherits nearest non-empty class on the left",
            "prediction": "probabilities @ centres",
            "rounds": 250,
            "params": {
                "learning_rate": 0.05,
                "num_leaves": 127,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l2": 5.0,
                "seed": 42,
                "max_bin": 63,
                "force_row_wise": True,
                "dist_objective": "multiclass",
                "dist_metric": "multi_logloss",
                "num_class": 16,
                "direct_objective": "regression",
                "direct_metric": "rmse",
            },
        },
        "seq_representation": {
            "encoder": "historical plain SEQ-01 dilated TCN (not D3A)",
            "channels": 17,
            "channel_order": "present,cat,buy,ponly,searches,search_to_cart,search_to_ord,cat_to_cart,cat_to_ord,to_cart,to_ord,gmv_search,gmv_cat,gmv,avail,dow_sin,dow_cos",
            "hidden": 64,
            "blocks": 8,
            "final_embedding": "concat(last, mean_over_time, max_over_time) of final LayerNorm output",
            "embedding_width": 192,
            "existing_hook": "src.seq_cond._pool + embed under torch.no_grad",
        },
        "checkpoint_archive": {
            "path": str(archive),
            "sha256": sha256(archive),
            "seed42_fold_availability": seed42,
            "seed43_fold_availability": seed43,
            "seed44_fold_availability": seed44,
            "seed42_production_checkpoint_availability": seed42_production,
            "exact_seed42_fold_checkpoint_available": False,
            "exact_seed42_production_checkpoint_available": False,
        },
        "substitute_parity": substitutes,
        "source_files": sources,
        "aligned_inputs": aligned,
        "geometry": geometry,
        "stop_reason": (
            "The exact fold-specific historical SEQ-01 seed-42 checkpoints used by current "
            "SEQ-AVG3 were never saved. Available retrained/D3A/compiled checkpoints are not "
            "parity-equivalent. Encoder parity and fold-safe embedding extraction therefore "
            "cannot be established without retraining or substituting the encoder, both forbidden."
        ),
        "stages_not_run": ["encoder parity", "pilot", "full folds", "multi-seed", "TEST", "submission"],
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
