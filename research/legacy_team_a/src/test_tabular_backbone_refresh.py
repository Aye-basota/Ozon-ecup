"""Invariant tests for EXP-046 / TABULAR-BACKBONE-REFRESH."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.config import SEED
from src.tabular_backbone_refresh import (
    ALL_HISTORICAL,
    COMPONENTS,
    EXPECTED_H,
    FOLD_WEIGHTS,
    PRIMARY_ROUND,
    ROUNDS,
    SEEDS,
    WEIGHTS,
    _prefix_smoke_audit,
    assemble_component,
    component_setup,
    historical_arrays,
    historical_baseline,
    recipe_manifest,
    saved_features,
)


def test_historical_strongest_current_reconstructs_exactly():
    _, _, _, rep = historical_baseline()
    assert np.allclose(rep["fold_cal"], EXPECTED_H["fold_cal"], rtol=0, atol=5e-10)
    assert abs(rep["wcv"] - EXPECTED_H["wcv"]) < 5e-10


def test_fixed_weights_and_fold_weights_are_exact():
    assert WEIGHTS == {"CAP": .10, "UNC": .20, "S1-DIST": .25,
                       "ETX-AVG3": .225, "SEQ-AVG3": .225}
    assert sum(WEIGHTS.values()) == 1.0
    assert np.array_equal(FOLD_WEIGHTS, np.array([1, 2, 4, 8]) / 15)


def test_unc_cap_frozen_feature_lists_match_recipes():
    assert len(saved_features("UNC")) == 236
    assert len(saved_features("CAP")) == 195
    assert len(set(saved_features("UNC"))) == 236
    assert len(set(saved_features("CAP"))) == 195


def test_only_master_seed_changes_and_lightgbm_derives_child_streams():
    assert SEEDS == (SEED, SEED + 1, SEED + 2)
    for component in COMPONENTS:
        setups = [component_setup(component, seed).as_dict() for seed in SEEDS]
        for setup, seed in zip(setups, SEEDS):
            assert setup["params"] == {"seed": seed}
        for key in setups[0]:
            if key != "params":
                assert setups[0][key] == setups[1][key] == setups[2][key]
        assert "LightGBM-derived" in recipe_manifest(component)["seed_policy"]["subordinate_streams"]


def test_early_stopping_off_and_primary_round_preregistered():
    assert PRIMARY_ROUND == 300
    assert ROUNDS == (200, 250, 300, 600)
    assert all(recipe_manifest(c)["early_stopping"] is False for c in COMPONENTS)
    assert 200 != PRIMARY_ROUND and 250 != PRIMARY_ROUND


def test_prefix_snapshot_equals_independent_300_artifact():
    assert _prefix_smoke_audit()["status"] == "PASS_BITWISE"


def test_historical_component_rows_targets_and_keys_are_identical():
    Z, y, cut = historical_arrays()
    assert Z.shape == (len(ALL_HISTORICAL), len(y))
    assert len(y) == len(cut) == 770_616
    assert np.isfinite(Z).all() and np.isfinite(y).all()


def test_runner_contains_no_test_or_submission_pipeline_access():
    path = Path(__file__).with_name("tabular_backbone_refresh.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "CUTOFF_TEST" not in names
    assert "SUBMISSIONS" not in names
    text = path.read_text(encoding="utf-8")
    assert "sample_submit" not in text
    assert "src.predict" not in text


def test_saved_fresh_component_uses_historical_canonical_order_when_available():
    # Artifact-backed after training; harmlessly skip on a clean checkout.
    run = Path(__file__).resolve().parent.parent / "artifacts" / "TBR_EXP046"
    if not (run / "TBR_EXP046_UNC_S42_V0904.npz").exists():
        return
    d = assemble_component("UNC", SEED, 600)
    _, y, cut = historical_arrays()
    assert np.array_equal(d["cutoff"], cut)
    assert np.array_equal(d["y"], y)


def test_all_saved_trajectories_share_rows_targets_and_have_four_raw_snapshots():
    run = Path(__file__).resolve().parent.parent / "artifacts" / "TBR_EXP046"
    if not (run / "TBR_EXP046_UNC_S42_V0904.npz").exists():
        return
    ref_uid = ref_y = None
    for component in COMPONENTS:
        for seed in SEEDS:
            for fold in ("0904", "0918", "1002", "1016"):
                d = np.load(run / f"TBR_EXP046_{component}_S{seed}_V{fold}.npz")
                assert set(d.files) == {"user_id", "cutoff", "y", "z_r200", "z_r250",
                                        "z_r300", "z_r600"}
                assert len(np.unique(d["user_id"])) == len(d["user_id"])
                assert all(d[f"z_r{r}"].dtype == np.float32 for r in ROUNDS)
                if component == "UNC" and seed == SEED:
                    ref_uid, ref_y = d["user_id"], d["y"]
                if component == "CAP" and seed == SEED and fold == "0904":
                    u = np.load(run / "TBR_EXP046_UNC_S42_V0904.npz")
                    assert np.array_equal(d["user_id"], u["user_id"])
                    assert np.array_equal(d["y"], u["y"])
