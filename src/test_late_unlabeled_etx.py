"""Causal, paired-arm and reproducibility contract for EXP-056.

The small structural tests run before the CPU cache exists.  Artifact-backed
checks become active after ``late_unlabeled_etx.py audit/replay/pilot`` and are
the executable audit of the actual materialized experiment.
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
import json

import numpy as np
import pytest

from src import etx
from src import late_unlabeled_etx as exp
from src import seq
from src.config import SEED, TARGET_DAYS


def _plan():
    if not exp.PLAN_NPZ.exists() or not exp.PLAN_JSON.exists():
        pytest.skip("EXP-056 CPU audit/cache has not been materialized")
    return np.load(exp.PLAN_NPZ), json.loads(exp.PLAN_JSON.read_text(encoding="utf-8"))


def _arm_results():
    paths = [exp.OUT / "arms" / name / "result.json" for name in ("CONTROL", "LATE")]
    if not all(p.exists() for p in paths):
        pytest.skip("EXP-056 paired pilot has not been materialized")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_fixed_corridors_are_equal_weekly_thursday_grids():
    assert len(exp.CONTROL_CUTS) == len(exp.LATE_CUTS) == 11
    assert exp.CONTROL_CUTS[0] == dt.date(2025, 5, 22)
    assert exp.CONTROL_CUTS[-1] == dt.date(2025, 7, 31)
    assert exp.LATE_CUTS[0] == dt.date(2025, 8, 7)
    assert exp.LATE_CUTS[-1] == exp.PRIMARY_VAL == dt.date(2025, 10, 16)
    assert all(d.weekday() == exp.THURSDAY for d in exp.CONTROL_CUTS + exp.LATE_CUTS)
    assert np.array_equal(np.diff([d.toordinal() for d in exp.CONTROL_CUTS]), np.full(10, 7))
    assert np.array_equal(np.diff([d.toordinal() for d in exp.LATE_CUTS]), np.full(10, 7))


def test_clean_rehearsal_targets_end_before_starting_validation():
    cuts = exp.clean_cutoffs()
    assert cuts[0] == dt.date(2025, 4, 3) and cuts[-1] == dt.date(2025, 7, 31)
    assert all(c + dt.timedelta(days=TARGET_DAYS) <= exp.START_VAL for c in cuts)


def test_mask_is_deterministic_real_token_only_and_exact_rate_rule():
    counts = np.array([0, 1, 6, 7, 20, exp.N_TOK], np.int32)
    a = exp.make_mask(counts, SEED)
    b = exp.make_mask(counts, SEED)
    assert np.array_equal(a, b)
    for row, n in zip(a, counts, strict=True):
        expected = 0 if n == 0 else max(1, int(np.floor(exp.MASK_RATE * n + 0.5)))
        assert int(row.sum()) == expected
        assert not row[int(n):].any()


def test_exposed_forward_is_exact_saved_etx_forward():
    import torch

    torch.manual_seed(SEED)
    cfg = dict(etx.DEFAULT_CFG, d_model=32, blocks=2, heads=4, head_dim=8,
               ffn=64, dropout=0.0, z0=2.0)
    model = etx.build_model(cfg).eval()
    tok = torch.randn(5, 11, etx.N_TOK_FEAT)
    static = torch.randn(5, etx.N_STATIC)
    age = torch.rand(5, 11) * 100
    n = torch.tensor([0, 1, 4, 8, 11])
    with torch.no_grad():
        expected = model(tok, static, age, n)
        actual, pooled, hidden, real = exp.forward_parts(model, tok, static, age, n)
    assert torch.equal(expected, actual)
    assert pooled.shape == (5, 3 * cfg["d_model"])
    assert hidden.shape == (5, 11, cfg["d_model"])
    assert torch.equal(real.sum(1), n)


def test_cli_has_no_prediction_test_submission_or_lb_action():
    tree = ast.parse(inspect.getsource(exp.main))
    commands = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_parser" and node.args
                and isinstance(node.args[0], ast.Constant)):
            commands.append(node.args[0].value)
    assert set(commands) == {"audit", "embed", "domain", "replay", "pilot", "arm", "analyze"}
    forbidden = {"predict", "test", "submit", "submission", "leaderboard", "lb"}
    assert forbidden.isdisjoint(commands)


def test_materialized_ssl_contains_no_label_and_direct_targets_are_legal():
    plan, meta = _plan()
    ssl_keys = {k for k in plan.files if k.startswith(("control_", "late_"))}
    assert ssl_keys == {"control_cut", "control_row", "late_cut", "late_row"}
    assert not any("target" in k or k.endswith("_y") for k in ssl_keys)
    assert meta["forbidden"]["ssl_targets"] is False
    latest_direct_target = max(exp.date_from_day(int(d)) for d in plan["direct_cut"])
    assert latest_direct_target + dt.timedelta(days=TARGET_DAYS) <= exp.START_VAL


def test_materialized_arms_are_exactly_matched_except_source_history():
    plan, meta = _plan()
    n = len(plan["control_row"])
    assert n == len(plan["late_row"]) == meta["n_ssl_examples"]
    assert len(plan["control_cut"]) == len(plan["late_cut"]) == len(plan["token_count"])
    assert len(plan["control_row"]) == len(plan["late_row"]) == len(plan["activity_bin"])
    expected_control = np.array([seq.day_index(d) for d in exp.CONTROL_CUTS], np.int16)
    expected_late = np.array([seq.day_index(d) for d in exp.LATE_CUTS], np.int16)
    assert np.array_equal(np.unique(plan["control_cut"]), expected_control)
    assert np.array_equal(np.unique(plan["late_cut"]), expected_late)
    assert np.all(plan["late_cut"].astype(np.int32) - plan["control_cut"].astype(np.int32) == 77)
    # Exact token count is stricter than the requested token-count decile match.
    rng = np.random.default_rng(SEED)
    sample = rng.choice(n, min(n, 4096), replace=False)
    for key, rows, cuts in [
        ("CONTROL", plan["control_row"][sample], plan["control_cut"][sample]),
        ("LATE", plan["late_row"][sample], plan["late_cut"][sample]),
    ]:
        _, cnt, _ = exp._indices_for_batch(cuts, rows)
        assert np.array_equal(cnt, plan["token_count"][sample]), key
        actual_bin = np.empty(len(sample), np.int8)
        for d in np.unique(cuts):
            m = cuts == d
            _, buy = exp.selected_counts_and_activity(exp.date_from_day(int(d)), rows[m])
            actual_bin[m] = exp.activity_bin(buy)
        assert np.array_equal(actual_bin, plan["activity_bin"][sample]), key
    assert len(plan["mask_seed"]) == len(plan["lr"]) == len(plan["ssl_batch_ptr"]) - 1


def test_no_selected_event_is_after_its_materialized_cutoff():
    plan, _ = _plan()
    _, event_day, _, _ = etx.events()
    rng = np.random.default_rng([SEED, 5601])
    for cut_key, row_key in [("control_cut", "control_row"), ("late_cut", "late_row"),
                             ("direct_cut", "direct_row")]:
        take = rng.choice(len(plan[row_key]), min(4096, len(plan[row_key])), replace=False)
        idx, cnt, cut = exp._indices_for_batch(plan[cut_key][take], plan[row_key][take])
        for i, n in enumerate(cnt):
            days = event_day[idx[i, :n]].astype(np.int64)
            assert not len(days) or int(days.max()) <= int(cut[i])
            assert not len(days) or int(days.min()) > int(cut[i]) - exp.DEPTH_CAP


def test_materialized_direct_batches_and_mask_plan_are_common_and_hashed():
    plan, meta = _plan()
    direct_hash = exp.sha256_json({k: exp.sha256_array(plan[k]) for k in
                                   ("direct_cut", "direct_row", "direct_y", "direct_batch_ptr",
                                    "holdout_cut", "holdout_row", "holdout_y")})
    assert direct_hash == meta["direct_plan_sha256"]
    ptr = plan["ssl_batch_ptr"]
    assert len(meta["mask_plan_sha256"]) == 64
    for i in (0, len(plan["mask_seed"]) // 2, len(plan["mask_seed"]) - 1):
        seed = plan["mask_seed"][i]
        a, b = int(ptr[i]), int(ptr[i + 1])
        left = exp.make_mask(plan["token_count"][a:b], int(seed))
        right = exp.make_mask(plan["token_count"][a:b], int(seed))
        assert np.array_equal(left, right)
        real = np.arange(exp.N_TOK)[None, :] < plan["token_count"][a:b, None]
        assert not left[~real].any()


def test_exact_validation_order_matches_saved_canonical_oof():
    _, meta = _plan()
    uid, _ = exp.fixed_validation(False)
    canonical = np.load(exp.ARTIFACTS / "oof_ETX-01-S42-V1016.npz")
    assert np.array_equal(uid, canonical["user_id"])
    assert exp.sha256_array(uid) == meta["validation_order_sha256"]


def test_paired_runs_share_initial_model_head_optimizer_rng_and_plan():
    control, late = _arm_results()
    keys = ["checkpoint_sha256", "checkpoint_state_sha256", "plan_sha256",
            "direct_plan_sha256", "mask_plan_sha256", "lr_plan_sha256",
            "initial_combined_state_sha256", "initial_optimizer_sha256",
            "initial_rng_sha256", "direct_head_sha256", "lambda_ssl", "mask_rate",
            "depth_cap", "weekday"]
    assert all(control["common"][k] == late["common"][k] for k in keys)
    assert control["n_steps"] == late["n_steps"]
    assert control["final_direct_head_sha256"] == late["final_direct_head_sha256"]


def test_deterministic_100_step_replay_artifact():
    if not exp.REPLAY_JSON.exists():
        pytest.skip("EXP-056 deterministic replay has not been run")
    replay = json.loads(exp.REPLAY_JSON.read_text(encoding="utf-8"))
    assert replay["steps"] == 100 and replay["pass_replay"]
    assert replay["snapshots_equal"] and all(replay["hashes_equal"].values())
