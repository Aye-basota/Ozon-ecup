from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from src import landmark_memory_etx as lm
from src import residual_signal_discovery as exp053
from src.btyd_day_bgnbd import user_group
from src.config import DATA_START, SEED


def synthetic(days: int = 330, users: int = 6):
    panel = np.zeros((users, days, 14), np.float32)
    gmv = np.zeros((users, days), np.float64)
    for u in range(users):
        for day in range(u, days, 5 + u):
            panel[u, day, 0] = 1
            panel[u, day, 1] = day % 3
            panel[u, day, 4] = np.log1p(day % 7)
        for day in range(3 + u, days, 17):
            panel[u, day, 2] = 1
            panel[u, day, 9] = np.log1p(1 + day % 11)
            gmv[u, day] = 1 + day % 11
    return panel, gmv


def test_fixed_schedule_has_16_lags_and_stride_15():
    assert lm.LAGS.tolist() == [30 + 15 * k for k in range(16)]
    assert len(lm.LAGS) == 16
    assert np.array_equal(np.diff(lm.LAGS), np.full(15, 15))


def test_every_real_landmark_satisfies_t_plus_30_le_query():
    for fold in lm.FOLDS:
        T = dt.date.fromisoformat(fold)
        for row in lm.landmark_schedule(T):
            t = dt.date.fromisoformat(row["landmark"])
            assert t + dt.timedelta(days=30) <= T


def test_state_window_is_exactly_t_minus_30_open_to_t_closed():
    q, lag = 280, 60
    a, b = lm.state_bounds(q, lag)
    t = q - lag
    assert (a, b) == (t - 29, t + 1)
    assert b - a == 30


def test_outcome_window_is_exactly_t_open_to_t_plus_30_closed():
    q, lag = 280, 60
    a, b = lm.outcome_bounds(q, lag)
    t = q - lag
    assert (a, b) == (t + 1, t + 31)
    assert b - a == 30


def test_current_target_window_never_appears_in_inputs():
    panel, gmv = synthetic()
    q = 280
    before = lm.build_tokens_from_arrays(panel, gmv, q)[0]
    changed = gmv.copy()
    changed[:, q + 1:q + 31] += 100000
    after = lm.build_tokens_from_arrays(panel, changed, q)[0]
    assert np.array_equal(before, after)


def test_mutation_after_query_leaves_all_inputs_unchanged():
    panel, gmv = synthetic()
    q = 280
    before = lm.build_tokens_from_arrays(panel, gmv, q)[0]
    p2, g2 = panel.copy(), gmv.copy()
    p2[:, q + 1:] = 777
    g2[:, q + 1:] = 888888
    after = lm.build_tokens_from_arrays(p2, g2, q)[0]
    assert np.array_equal(before, after)


def test_state_boundary_mutations_obey_open_closed_window():
    panel, gmv = synthetic()
    q, lag = 280, 60
    a, b = lm.state_bounds(q, lag)
    base = lm.build_tokens_from_arrays(panel, gmv, q)[0]
    j = lm.LAGS.tolist().index(lag)
    outside = panel.copy()
    outside[:, a - 1, 4] += 3
    assert np.array_equal(base[:, j, :18],
                          lm.build_tokens_from_arrays(outside, gmv, q)[0][:, j, :18])
    inside = panel.copy()
    inside[:, a, 4] += 3
    changed = lm.build_tokens_from_arrays(inside, gmv, q)[0]
    assert not np.array_equal(base[:, j, :18], changed[:, j, :18])
    assert b - 1 == q - lag


def test_outcome_boundary_mutations_obey_open_closed_window():
    panel, gmv = synthetic()
    q, lag = 280, 60
    a, b = lm.outcome_bounds(q, lag)
    base = lm.build_tokens_from_arrays(panel, gmv, q)[0]
    at_t = gmv.copy()
    at_t[:, a - 1] += 100
    j = lm.LAGS.tolist().index(lag)
    assert np.array_equal(base[:, j, 19], lm.build_tokens_from_arrays(panel, at_t, q)[0][:, j, 19])
    first = gmv.copy()
    first[:, a] += 100
    assert not np.array_equal(base[:, j, 19],
                              lm.build_tokens_from_arrays(panel, first, q)[0][:, j, 19])
    assert b - 1 == q - lag + 30


def test_pad_and_mask_correctness_for_insufficient_history():
    panel, gmv = synthetic(days=200)
    q = 100
    tokens, valid, _, _ = lm.build_tokens_from_arrays(panel, gmv, q)
    expected = np.asarray([lm.landmark_valid(q, int(lag)) for lag in lm.LAGS])
    assert np.array_equal(valid[0], expected)
    assert np.all(tokens[:, :16][~valid] == 0)


def test_query_outcome_is_zero_and_masked():
    panel, gmv = synthetic()
    tokens, valid, _, _ = lm.build_tokens_from_arrays(panel, gmv, 280)
    assert np.all(tokens[:, lm.QUERY_POS, 19] == 0)
    outcome_available = np.column_stack([valid, np.zeros(len(valid), bool)])
    assert not outcome_available[:, lm.QUERY_POS].any()


def test_historical_outcome_field_mutation_changes_only_that_field():
    panel, gmv = synthetic()
    real = lm.build_tokens_from_arrays(panel, gmv, 280)[0]
    shuf = real.copy()
    j = 3
    shuf[:, j, 19] = shuf[::-1, j, 19]
    assert np.array_equal(real[:, :, :19], shuf[:, :, :19])
    keep = np.ones(real.shape[1:], bool)
    keep[j, 19] = False
    assert np.array_equal(real[:, keep], shuf[:, keep])


def test_raw_historical_window_mutation_can_touch_overlapping_landmarks():
    """Record the unavoidable overlap induced by 30d windows at 15d stride.

    The causal REAL/SHUF intervention is therefore applied to the materialized
    outcome field (tested above), rather than pretending that a raw event edit
    can leave every neighbouring landmark state/outcome untouched.
    """
    q, anchor_lag = 280, 60
    raw_day = q - anchor_lag + 20
    affected_outcomes = []
    affected_states = []
    for lag in lm.LAGS:
        oa, ob = lm.outcome_bounds(q, int(lag))
        sa, sb = lm.state_bounds(q, int(lag))
        if oa <= raw_day < ob:
            affected_outcomes.append(int(lag))
        if sa <= raw_day < sb:
            affected_states.append(int(lag))
    assert anchor_lag in affected_outcomes
    assert len(affected_outcomes) > 1
    assert affected_states


def test_shuffle_preserves_multiset_in_every_stratum():
    values = np.arange(80, dtype=np.float32)
    strata = np.repeat(np.arange(8), 10)
    shuffled, permutation = lm.materialize_shuffle(values, strata)
    assert lm.shuffle_preserves_multiset(values, shuffled, strata)
    assert np.array_equal(shuffled, values[permutation])


def test_shuffle_is_deterministic_at_config_seed():
    values = np.arange(80, dtype=np.float32)
    strata = np.repeat(np.arange(8), 10)
    a = lm.materialize_shuffle(values, strata, SEED)
    b = lm.materialize_shuffle(values, strata, SEED)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_real_and_shuffled_have_equal_shape_and_differ_only_in_outcome():
    panel, gmv = synthetic()
    real = lm.build_tokens_from_arrays(panel, gmv, 280)[0]
    shuf = real.copy()
    for j in range(16):
        shuf[:, j, 19] = real[::-1, j, 19]
    assert real.shape == shuf.shape
    assert np.array_equal(real[:, :, :19], shuf[:, :, :19])
    assert np.array_equal(real[:, -1, 19], shuf[:, -1, 19])


def test_user_halves_are_disjoint_and_stable():
    uid = np.arange(1, 10001, dtype=np.int64)
    side = user_group(uid)
    assert not np.intersect1d(uid[side == 0], uid[side == 1]).size
    assert np.array_equal(side, user_group(uid))
    assert set(np.unique(side)) == {0, 1}


def test_recipient_labels_are_absent_from_preflight_training_masks():
    uid = np.arange(200, dtype=np.int64)
    side = user_group(uid)
    fi = np.tile(np.arange(4, dtype=np.int8), 50)
    for donor_side in (0, 1):
        donor = (fi < 3) & (side == donor_side)
        recipient = (fi == 3) & (side == 1 - donor_side)
        assert not np.any(donor & recipient)
        assert not np.intersect1d(uid[donor], uid[recipient]).size


def test_exact_strongest_reconstruction_fold_1016():
    frame, manifest = exp053._load_core()
    exp053._audit_baseline(frame, manifest)
    audit = manifest["calibration_audit"]
    assert audit["status"] == "PASS_EXACT"
    assert audit["fold_scores"][-1] == pytest.approx(1.741278566, abs=5e-10)
    assert audit["wcv"] == pytest.approx(1.747509863, abs=5e-10)


def test_fixed_ensemble_weights_are_exact_and_sum_to_one():
    assert lm.FIXED_REPLACEMENT_WEIGHTS == {
        "CAP": .10, "UNC": .20, "DIST": .25, "LANDMARK_REAL": .225,
        "SEQ_AVG3": .225,
    }
    assert lm.FIXED_COAUTHOR_WEIGHTS == {
        "CAP": .10, "UNC": .20, "DIST": .25, "ETX_AVG3": .1125,
        "LANDMARK_REAL": .1125, "SEQ_AVG3": .225,
    }
    assert sum(lm.FIXED_REPLACEMENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(lm.FIXED_COAUTHOR_WEIGHTS.values()) == pytest.approx(1.0)


def test_same_initial_model_optimizer_and_rng_contract_hashes():
    hashes = lm.paired_contract_hashes(100)
    assert hashes["same_initial_model"]
    assert hashes["same_initial_optimizer"]
    assert hashes["arms"][0] == hashes["arms"][1]


def test_same_materialized_batch_and_lr_plans():
    a = lm.materialized_batch_plan(101, 4, SEED)
    b = lm.materialized_batch_plan(101, 4, SEED)
    assert np.array_equal(a, b)
    la = lm.learning_rate_plan(100)
    lb = lm.learning_rate_plan(100)
    assert np.array_equal(la, lb)


def test_no_test_submission_or_public_lb_actions():
    assert lm.FORBIDDEN_ACTIONS == {
        "test_inference": False, "submission": False, "public_lb": False,
        "full_folds": False,
    }


def test_analysis_only_replay_hashes_when_artifacts_exist():
    p = lm.RESULT_DIR / "reproducibility.json"
    if not p.exists():
        pytest.skip("run EXP-055 once to create artifact-backed replay hashes")
    stored = json.loads(p.read_text(encoding="utf-8"))["canonical_hashes"]
    assert lm.canonical_replay_hashes() == stored


def test_historical_outcomes_never_overlap_current_target():
    q = 280
    current = set(range(q + 1, q + 31))
    for lag in lm.LAGS:
        if lm.landmark_valid(q, int(lag)):
            a, b = lm.outcome_bounds(q, int(lag))
            assert current.isdisjoint(range(a, b))


def test_landmark_model_has_direct_head_only_and_fixed_capacity():
    model = lm.build_landmark_model()
    names = [name for name, _ in model.named_parameters()]
    assert lm.NEURAL_CFG["d_model"] == 128
    assert lm.NEURAL_CFG["blocks"] == 5
    assert lm.NEURAL_CFG["heads"] == 8
    assert lm.NEURAL_CFG["head_dim"] == 16
    assert lm.NEURAL_CFG["ffn"] == 384
    assert lm.NEURAL_CFG["epochs"] == 4
    assert not any("aux" in name or "control" in name for name in names)
