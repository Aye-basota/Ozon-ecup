from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp074_recon", HERE / "run_reconnaissance.py")
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_exp074_is_free_in_both_registries():
    old_root, geometry_root = AUDIT.configured_paths()
    del old_root
    registries = [
        AUDIT.ROOT / "registry" / "experiments.csv",
        geometry_root / "gpt_pro_research_packet" / "02_EXPERIMENT_REGISTRY.csv",
    ]
    assert all(AUDIT.registry_hits(path) == [] for path in registries)


def test_exact_historical_seed42_checkpoints_are_absent():
    old_root, _ = AUDIT.configured_paths()
    archive = old_root / "weights_archives" / "TCN_SEQ-01_weights.zip"
    import zipfile

    with zipfile.ZipFile(archive) as stream:
        names = set(stream.namelist())
    for fold in AUDIT.FOLDS:
        name = f"model_SEQ-01-S42-V{fold[5:7]}{fold[8:10]}.pt"
        assert not (old_root / "artifacts" / name).exists()
        assert name not in names
    for name in ("model_SEQ-01-S42-TEST.pt", "model_SEQ-C289-S42-TEST.pt"):
        assert not (old_root / "artifacts" / name).exists()
        assert name not in names


def test_available_seed42_substitutes_fail_prediction_parity():
    old_root, _ = AUDIT.configured_paths()
    art = old_root / "artifacts"
    checks = [
        ("oof_SEQ-01-S42-V1016.npz", "oof_SEQ-D3A-BASE-S42-V1016.npz"),
        ("oof_SEQ-01-S42-V1016.npz", "oof_SEQ-03A-BASE-S42-V1016.npz"),
        ("oof_SEQ-01-S42-V0904.npz", "oof_SEQ-01C-S42-V0904.npz"),
    ]
    for reference, candidate in checks:
        result = AUDIT.compare_oof(art / reference, art / candidate)
        assert result["user_id_equal"] and result["target_equal"]
        assert not result["prediction_equal"]
        assert result["max_abs_log_error"] > 1e-6


def test_historical_dist16_semantics_from_exact_source():
    old_root, _ = AUDIT.configured_paths()
    sys.path.insert(0, str(old_root))
    try:
        from src import models

        positive = np.arange(1, 31, dtype=np.float64) / 10
        z = np.concatenate(([0.0, 0.0], positive))
        edges = models.z_bins(z)
        labels = models.bin_labels(z, edges)
        centres = models.bin_centroids(z, labels)
        assert len(edges) == 15 and len(centres) == 16
        assert edges[0] == 1e-9
        assert np.array_equal(labels[:2], np.zeros(2, dtype=np.int32))
        one_hot = np.eye(16, dtype=np.float64)[labels]
        pred = models.dist_expectation(one_hot, centres)
        assert np.allclose(pred, centres[labels])
    finally:
        sys.path.pop(0)


def test_aligned_inputs_and_rank57_basis_are_present():
    old_root, geometry_root = AUDIT.configured_paths()
    aligned = AUDIT.aligned_inputs_audit(
        geometry_root / "gpt_pro_research_packet",
        old_root / "data" / "raw" / "sample_submit.csv",
    )
    geometry = AUDIT.geometry_audit(geometry_root)
    assert aligned["oof_rows"] == 770_616
    assert aligned["test_rows"] == 250_000
    assert aligned["test_matches_sample_order"]
    assert geometry["unique_sources"] == 65
    assert geometry["documented_difference_rank"] == 57
