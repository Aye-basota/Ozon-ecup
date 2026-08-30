from __future__ import annotations

import inspect

import numpy as np
import pytest

from src import burst_gap_etx as bg
from src import residual_signal_discovery as exp053
from src.btyd_day_bgnbd import user_group


def _values(n: int) -> np.ndarray:
    x = np.zeros((n, 14), np.float32)
    x[:, 0] = 1.0
    x[:, 2] = np.arange(n) % 2
    x[:, 4] = np.arange(n) + 1
    x[:, 13] = np.arange(n) * 10.0
    return x


def _scale() -> np.ndarray:
    return np.ones(14, np.float32)


def test_cutoff_safety_future_mutation_changes_no_token():
    days = np.array([2, 5, 9, 12], np.int64)
    values = _values(len(days))
    a = bg.build_history_tokens(days, values, cutoff_day=9, available_depth=10, scale=_scale())
    values[3] = 1e9
    b = bg.build_history_tokens(days, values, cutoff_day=9, available_depth=10, scale=_scale())
    assert np.array_equal(a["tokens"], b["tokens"])
    assert np.array_equal(a["types"], b["types"])


def test_fixed_burst_threshold_three():
    seg = bg.segment_days(np.array([0, 3, 7]), cutoff_day=9, available_depth=10)
    bursts = [s for s in seg if s["type"] == bg.BURST]
    assert [s["event_days"].tolist() for s in bursts] == [[0, 3], [7]]


def test_segmentation_is_deterministic():
    days = np.array([1, 2, 6, 8, 20])
    a = bg.segment_days(days, 25, 26)
    b = bg.segment_days(days, 25, 26)
    assert [(s["type"], s["start"], s["end"], s["length"]) for s in a] == [
        (s["type"], s["start"], s["end"], s["length"]) for s in b]
    assert all(np.array_equal(x["event_days"], y["event_days"]) for x, y in zip(a, b))


def test_every_event_day_belongs_to_exactly_one_burst():
    days = np.array([1, 2, 6, 8, 20])
    seg = bg.segment_days(days, 25, 26)
    covered = np.concatenate([s["event_days"] for s in seg if s["type"] == bg.BURST])
    assert np.array_equal(covered, days)
    assert len(np.unique(covered)) == len(days)


def test_internal_gap_lengths_are_date_diff_minus_one():
    seg = bg.segment_days(np.array([1, 5, 11]), cutoff_day=15, available_depth=16)
    gaps = [s["length"] for s in seg if s["type"] == bg.GAP]
    assert gaps == [3, 5, 4]


def test_final_open_gap_is_always_present_including_zero():
    seg = bg.segment_days(np.array([1, 5, 15]), cutoff_day=15, available_depth=16)
    assert seg[-1]["type"] == bg.GAP
    assert seg[-1]["length"] == 0


def test_no_events_is_one_full_history_gap():
    seg = bg.segment_days(np.array([100]), cutoff_day=9, available_depth=10)
    assert len(seg) == 1 and seg[0]["type"] == bg.GAP and seg[0]["length"] == 10


def test_tokens_have_no_target_argument_or_target_like_field():
    signature = inspect.signature(bg.build_history_tokens)
    assert all(token not in name.lower() for name in signature.parameters
               for token in ("target", "future", "y30"))


def test_token_cap_merges_prefix_into_summary():
    days = np.arange(0, 500, 4, dtype=np.int64)
    result = bg.build_history_tokens(days, _values(len(days)), 499, 500, _scale())
    assert len(result["tokens"]) == 191
    assert result["types"][0] == bg.SUMMARY
    assert result["overflow"]


def test_numeric_token_dimension_is_22():
    result = bg.build_history_tokens(np.array([1, 2, 8]), _values(3), 10, 11, _scale())
    assert result["tokens"].shape[1] == 22 == bg.N_TOKEN_NUMERIC


def test_parameter_count_is_within_two_percent():
    audit = bg.parameter_audit()
    assert audit["within_2pct"]
    assert audit["candidate_params"] > audit["baseline_params"]


def test_exact_baseline_reconstruction_from_saved_oof():
    frame, manifest = exp053._load_core()
    exp053._audit_baseline(frame, manifest)
    assert abs(manifest["calibration_audit"]["fold_scores"][-1] - 1.741278566) < 5e-7
    assert abs(manifest["calibration_audit"]["wcv"] - 1.747509863) < 5e-7


def test_joint_shuffle_preserves_strata_and_rows():
    strata = np.array([0, 0, 1, 1, 1, 2])
    values = np.arange(18).reshape(6, 3)
    perm = exp053.permutation_within_strata(strata, np.ones(6, bool), seed=42)
    assert np.array_equal(strata, strata[perm])
    assert sorted(map(tuple, values)) == sorted(map(tuple, values[perm]))


def test_user_halves_are_disjoint():
    uid = np.arange(1, 1001, dtype=np.int64)
    side = user_group(uid)
    assert np.intersect1d(uid[side == 0], uid[side == 1]).size == 0


def test_protocol_has_no_recipient_users_or_labels_in_training():
    uid = np.tile(np.arange(100, dtype=np.int64), 4)
    frame = {"user_id": uid, "fold_index": np.repeat(np.arange(4, dtype=np.int8), 100)}
    for donor_side in (0, 1):
        donor, recipient = bg.protocol_masks(frame, donor_side)
        assert not np.any(donor & recipient)
        assert np.intersect1d(uid[donor], uid[recipient]).size == 0
        assert np.all(frame["fold_index"][donor] < 3)
        assert np.all(frame["fold_index"][recipient] == 3)


def test_fixed_ensemble_weights_are_exact_and_sum_to_one():
    assert bg.FIXED_ENSEMBLE_WEIGHTS == {
        "cap": .10, "unc": .20, "dist": .25, "etx": .225, "seq": .225,
    }
    assert bg.REPLACEMENT_WEIGHTS == {
        "cap": .10, "unc": .20, "dist": .25, "burst": .225, "seq": .225,
    }
    assert bg.COAUTHOR_WEIGHTS == {
        "cap": .10, "unc": .20, "dist": .25,
        "etx": .1125, "burst": .1125, "seq": .225,
    }
    assert all(abs(sum(v.values()) - 1.0) < 1e-12 for v in
               (bg.FIXED_ENSEMBLE_WEIGHTS, bg.REPLACEMENT_WEIGHTS, bg.COAUTHOR_WEIGHTS))


def test_runner_exposes_no_test_or_submission_command():
    source = inspect.getsource(bg.main)
    assert '"predict"' not in source
    assert '"submission"' not in source


def test_analysis_only_hash_replay_when_endpoint_exists():
    if not (bg.OUT_ARTIFACTS / "preflight_verdict.json").exists():
        pytest.skip("endpoint artifacts are built by the experiment runner")
    first = bg.analysis_only()["hashes"]
    second = bg.analysis_only()["hashes"]
    assert first == second
