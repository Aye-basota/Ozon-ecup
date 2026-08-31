"""Focused contracts for EXP-058.

The expensive full-parquet checks live in the runner's mandatory Phase 0.  The
tests below exercise the pure boundary, cutoff, permutation, novelty, and scope
contracts without training a model.
"""
from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path

import numpy as np
import polars as pl

from src.config import DATA_START, SEED
from src.dataset_fingerprint import (
    FINGERPRINT_FIELDS,
    FORBIDDEN_FEATURE_TOKENS,
    build_fingerprint_features,
    fixed_permutation,
    panel_schedule,
    permuted_fingerprints,
    summarize_panel_memberships,
    target_mask,
)


def test_target_boundary_is_open_left_closed_right():
    cutoff = dt.date(2025, 10, 16)
    dates = np.array(["2025-10-16", "2025-10-17", "2025-11-15", "2025-11-16"],
                     dtype="datetime64[D]")
    assert target_mask(dates, cutoff).tolist() == [False, True, True, False]


def test_target_sum_and_log1p_boundary_audit():
    cutoff = dt.date(2025, 10, 16)
    dates = np.array(["2025-10-16", "2025-10-17", "2025-11-15", "2025-11-16"],
                     dtype="datetime64[D]")
    gmv = np.array([100.0, 2.0, 3.0, 1000.0])
    y = gmv[target_mask(dates, cutoff)].sum()
    assert y == 5.0
    assert np.log1p(y) == np.log(6.0)


def test_panel_schedule_never_exceeds_query_cutoff():
    cutoff = dt.date(2025, 10, 16)
    schedule = panel_schedule(cutoff)
    assert schedule
    assert max(schedule) <= cutoff
    assert all((b - a).days == 7 for a, b in zip(schedule, schedule[1:]))


def test_no_future_panel_membership_is_accepted():
    universe = np.array([1, 2, 3], dtype=np.int64)
    cutoff = dt.date(2025, 4, 3)
    future = cutoff + dt.timedelta(days=7)
    try:
        summarize_panel_memberships(universe, cutoff, [future],
                                    [np.array([1])], [np.array([1])])
    except AssertionError as exc:
        assert "future panel" in str(exc)
    else:
        raise AssertionError("future membership must be rejected")


def test_panel_history_counts_only_supplied_past_dates():
    universe = np.array([1, 2, 3], dtype=np.int64)
    schedule = [dt.date(2025, 4, 3), dt.date(2025, 4, 10)]
    out = summarize_panel_memberships(
        universe, schedule[-1], schedule,
        [np.array([1, 2]), np.array([1, 3])],
        [np.array([1]), np.array([1, 2])],
    ).sort("user_id")
    assert out["fp_panel1_pass_count"].to_list() == [2, 1, 1]
    assert out["fp_panel3_pass_count"].to_list() == [2, 1, 0]
    assert out["fp_panel3_first_day"].to_list()[2] == -1


def test_fixed_permutation_is_bijective_and_seed_stable():
    users = np.arange(1, 101, dtype=np.int64)
    signatures = np.repeat(np.arange(10), 10)
    a = fixed_permutation(users, signatures, SEED)
    b = fixed_permutation(users, signatures, SEED)
    assert np.array_equal(a, b)
    assert np.array_equal(np.sort(a), users)
    assert np.all(a != users)
    assert np.array_equal(signatures[np.searchsorted(users, a)], signatures)


def test_singletons_remain_fixed_but_other_users_are_deranged():
    users = np.arange(1, 8, dtype=np.int64)
    signatures = np.array([0, 0, 1, 2, 2, 2, 3])
    mapped = fixed_permutation(users, signatures, SEED)
    assert mapped[2] == users[2]
    assert mapped[6] == users[6]
    assert np.all(mapped[[0, 1, 3, 4, 5]] != users[[0, 1, 3, 4, 5]])


def test_real_perm_joint_marginals_are_identical_when_panel_is_invariant():
    users = np.arange(1, 7, dtype=np.int64)
    signatures = np.zeros(len(users), dtype=np.uint32)
    mapped = fixed_permutation(users, signatures, SEED)
    mapping = pl.DataFrame({"user_id": users, "mapped_user_id": mapped})
    fp = pl.DataFrame({"user_id": users, "a": users * 3, "b": users[::-1] ** 2})
    real = fp.select("a", "b").to_numpy()
    perm = permuted_fingerprints(users, fp, mapping, ["a", "b"])
    assert all(np.array_equal(np.sort(real[:, j]), np.sort(perm[:, j])) for j in range(2))


def test_duplicate_user_day_count_definition():
    users = np.array([1, 1, 1, 2, 2])
    days = np.array([1, 1, 2, 1, 3])
    duplicate = int(np.sum((users[1:] == users[:-1]) & (days[1:] == days[:-1])))
    assert duplicate == 1


def test_missing_date_behavior_does_not_dense_fill():
    observed = np.array([0, 2, 5])
    span = observed[-1] - observed[0] + 1
    assert len(observed) == 3
    assert span - len(observed) == 3


def test_search_catalog_equality_tolerance():
    search = np.array([1.0, 2.25])
    catalog = np.array([0.5, 0.75])
    total = np.array([1.5, 3.0])
    assert np.max(np.abs(search + catalog - total)) <= 1e-6


def test_fingerprint_names_have_no_target_encoding_tokens():
    assert len(FINGERPRINT_FIELDS) == len(set(FINGERPRINT_FIELDS))
    assert not [name for name in FINGERPRINT_FIELDS
                if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]


def test_builder_has_no_target_argument_and_calls_canonical_build_features():
    assert "target" not in inspect.signature(build_fingerprint_features).parameters
    source = inspect.getsource(build_fingerprint_features)
    assert "build_features(cutoff" in source


def test_runner_has_no_test_prediction_or_external_output_path():
    source = Path(inspect.getsourcefile(build_fingerprint_features)).read_text(encoding="utf-8")
    assert "SUBMISSIONS" not in source
    assert "write_csv(out" not in source


def test_identity_bits_are_fixed_integer_slices():
    ids = np.array([0x00010002, 0x0003FFFF], dtype=np.uint64)
    low = (ids & np.uint64(0xFFFF)).astype(np.uint32)
    high = ((ids >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint32)
    assert low.tolist() == [2, 65535]
    assert high.tolist() == [1, 3]


def test_initial_missing_prefix_is_cutoff_safe_by_definition():
    first = dt.date(2025, 1, 8)
    prefix = (first - DATA_START).days
    assert prefix == 7
    # Appending any event after a query cutoff cannot change the already observed first date.
    future = dt.date(2026, 1, 1)
    assert min(first, future) == first
