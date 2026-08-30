"""Invariant tests for EXP-044 fresh conditional fine-tuning."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import fresh_cond_ft as ft
from src.seq import build_model, gather


def _pair_artifacts(seed: int = ft.SEEDS[0]):
    npz, meta_path = ft.pair_plan_paths(seed)
    if not (npz.exists() and meta_path.exists()):
        pytest.skip("paired plan has not been materialized yet")
    return np.load(npz, allow_pickle=False), json.loads(meta_path.read_text(encoding="utf-8"))


def _arm_results(seed: int = ft.SEEDS[0]):
    paths = [ft.ARMS / ft.arm_name(seed, arm) / "result.json" for arm in ft.ARMS_ALLOWED]
    if not all(path.exists() for path in paths):
        pytest.skip("full paired arms have not completed yet")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_01_plain_baseline_cfg_has_no_depth_augmentation():
    for seed in ft.SEEDS:
        cfg = ft.baseline_cfg(seed)
        assert cfg["depth_aug"] == 0.0
        assert cfg["aug"] == "none"


def test_02_d3a_checkpoint_or_name_is_rejected():
    cfg = ft.baseline_cfg(ft.SEED)
    cfg["depth_aug"] = 0.5
    with pytest.raises(AssertionError):
        ft.validate_plain_baseline_cfg(cfg, "SEQ-D3A-S42-V1016")
    with pytest.raises(AssertionError):
        ft.validate_plain_baseline_cfg(ft.baseline_cfg(ft.SEED), "SEQ-D3A-S42-V1016")


def test_03_one_baseline_checkpoint_is_named_for_both_arms():
    _, meta = _pair_artifacts()
    assert meta["baseline_name"] == ft.baseline_name(ft.SEED)
    assert meta["arms"] == [ft.arm_name(ft.SEED, ft.ARM_VOL),
                            ft.arm_name(ft.SEED, ft.ARM_FRESH)]
    assert "D3A" not in meta["baseline_checkpoint"].upper()


def test_04_initial_model_hashes_inside_completed_pair_are_equal():
    vol, fresh = _arm_results()
    assert vol["initial"]["model_sha256"] == fresh["initial"]["model_sha256"]


def test_05_initial_conditional_head_hashes_inside_pair_are_equal():
    vol, fresh = _arm_results()
    assert (vol["initial"]["conditional_head_sha256"]
            == fresh["initial"]["conditional_head_sha256"])


def test_06_initial_optimizer_hashes_inside_pair_are_equal():
    vol, fresh = _arm_results()
    assert vol["initial"]["optimizer_sha256"] == fresh["initial"]["optimizer_sha256"]


def test_07_direct_clean_plan_is_shared():
    vol, fresh = _arm_results()
    assert vol["direct_plan_sha256"] == fresh["direct_plan_sha256"]


def test_08_common_clean_positive_plan_is_shared():
    vol, fresh = _arm_results()
    assert vol["common_plan_sha256"] == fresh["common_plan_sha256"]


def test_09_added_conditional_slot_counts_are_equal():
    plan, meta = _pair_artifacts()
    assert meta["n_added_slots"] == meta["n_common_slots"]
    assert plan["common_index"].shape == plan["added_slot"].shape


def test_10_conditional_batch_shapes_are_equal():
    vol, fresh = _arm_results()
    assert vol["conditional_batch_shape"] == fresh["conditional_batch_shape"]


def test_11_optimizer_steps_and_lr_sequences_are_equal():
    plan, meta = _pair_artifacts()
    assert len(plan["encoder_lr"]) == len(plan["conditional_lr"]) == meta["n_steps"]
    assert np.array_equal(plan["encoder_lr"], ft.ENCODER_LR * plan["lr_multiplier"])
    assert np.array_equal(plan["conditional_lr"], ft.COND_LR * plan["lr_multiplier"])


def test_12_direct_head_is_frozen_and_receives_no_gradients():
    cfg = dict(ft.baseline_cfg(ft.SEED), z0=2.0)
    model = build_model(cfg)
    ft.freeze_direct_head(model)
    x = torch.zeros(2, 17, 365)
    model(x).sum().backward()
    assert all(not p.requires_grad and p.grad is None for p in model.head.parameters())
    assert any(p.grad is not None for name, p in model.named_parameters()
               if not name.startswith("head."))


def test_13_direct_head_hash_does_not_change_when_frozen():
    cfg = dict(ft.baseline_cfg(ft.SEED), z0=2.0)
    model = build_model(cfg)
    ft.freeze_direct_head(model)
    before = ft.direct_head_hash(model)
    opt = torch.optim.SGD(ft.encoder_parameters(model), lr=0.01)
    model(torch.randn(2, 17, 365)).sum().backward()
    opt.step()
    assert ft.direct_head_hash(model) == before


def test_14_extra_is_declared_conditional_only_never_direct():
    _, meta = _pair_artifacts()
    assert meta["fresh_semantics"].endswith("conditional loss only")
    vol, fresh = _arm_results()
    assert not vol["extra_in_direct_loss"] and not fresh["extra_in_direct_loss"]


def test_15_every_extra_row_is_positive():
    plan, _ = _pair_artifacts()
    assert len(plan["extra_z"]) > 0
    assert bool((plan["extra_z"] > 0).all())


def test_16_extra_donors_are_group_b_not_validation_group_a():
    plan, _ = _pair_artifacts()
    extra_uid = ft.panel()[2][plan["extra_rows"]]
    assert bool((ft.user_group(extra_uid) == 1).all())
    assert bool((ft.user_group(plan["val_user_id"]) == 0).all())
    assert np.intersect1d(np.unique(extra_uid), plan["val_user_id"]).size == 0


def test_17_extra_input_depth_clip_is_289():
    _, meta = _pair_artifacts()
    assert meta["extra_depth_clip"] == 289


def test_18_target_centering_does_not_read_validation_or_test():
    _, meta = _pair_artifacts()
    assert meta["centering"] == {
        "clean": "positive CLEAN training rows grouped by their cutoff",
        "extra": "positive EXTRA group-B training rows grouped by their cutoff",
        "other_arm_used": False,
        "test_used": False,
        "validation_used": False,
    }


def test_19_primary_inference_does_not_use_conditional_head():
    source = inspect.getsource(ft.predict_pair_outputs)
    assert "zo = model.head(pooled)" in source
    assert "inference_uses_conditional_head=False" in inspect.getsource(ft._snapshot_prediction)


def test_20_two_100_step_vol_processes_match_bitwise():
    comparison = ft.REPLAY / "comparison.json"
    if not comparison.exists():
        pytest.skip("integration replay has not completed yet")
    data = json.loads(comparison.read_text(encoding="utf-8"))
    assert data["technical_pass"]
    assert all(data["checks"].values())
    assert data["var_delta_z"] == data["max_abs_delta_z"] == 0.0


def test_21_validation_order_is_identical_for_baseline_vol_and_fresh():
    vol, fresh = _arm_results()
    baseline = json.loads((ft.BASELINES / ft.baseline_name(ft.SEED) / "result.json").read_text(
        encoding="utf-8"))
    assert (baseline["validation_order_sha256"]
            == vol["validation_order_sha256"]
            == fresh["validation_order_sha256"])


def test_22_inputs_end_at_their_own_cutoff():
    # The experiment delegates all three streams to the already anti-lookahead
    # audited gather().  Guard the exact left-closed implementation contract so
    # a future rewrite cannot silently substitute a different input builder.
    source = inspect.getsource(gather)
    assert "lo, hi = max(0, d - SEQ_L + 1), d + 1" in source
    assert "p[rows, lo:hi" in source
    arm_source = inspect.getsource(ft.run_pair_arm)
    assert "gather_index_rows" in arm_source and "gather_extra_rows" in arm_source


def test_pair_plan_has_fixed_snapshot_endpoints():
    plan, meta = _pair_artifacts()
    steps = set(int(x) for x in plan["snapshot_steps"])
    assert {0, 1, 100, 1000, meta["n_steps"] // 2, meta["n_steps"]} <= steps


def test_pair_plan_uses_exact_requested_extra_cutoffs():
    plan, _ = _pair_artifacts()
    assert [str(x) for x in plan["extra_cuts"]] == [x.isoformat() for x in ft.EXTRA_CUTOFFS]


def test_decision_boundaries_match_preregistration():
    decision, promote, _ = ft.decision_from_metrics(
        mean_delta=-0.0008, negative_seeds=2, mean_positive_delta=-0.001,
        positive_better_seeds=2, mean_auc_delta=-0.0001, pooled_var=0.049,
        mean_aux_delta=-0.001, max_abs_delta=0.1, technical_pass=True,
        mean_fresh_vs_baseline=0.0004)
    assert decision == "SIGNAL PASS" and promote
    decision, promote, _ = ft.decision_from_metrics(
        mean_delta=-0.0005, negative_seeds=3, mean_positive_delta=-0.001,
        positive_better_seeds=3, mean_auc_delta=0.0, pooled_var=0.01,
        mean_aux_delta=0.0, max_abs_delta=0.1, technical_pass=True,
        mean_fresh_vs_baseline=0.0)
    assert decision == "INCONCLUSIVE" and not promote
