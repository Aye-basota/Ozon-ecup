from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.exact_anniversary import (
    PRIMARY_CUTOFF,
    aggregate_window,
    align_submission,
    anniversary_window,
    build_features,
    crossfit_arm,
    feature_matrix,
    pre_window,
    recent_window,
    shift_calendar_year,
    shifted_year_window,
    shuffle_anniversary,
    shuffle_strata,
    splitmix64,
    target_window,
    user_group,
)


def tiny_raw() -> pl.DataFrame:
    rows = []
    users = np.arange(1, 65, dtype=np.int64)
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(105)]
    dates += [dt.date(2025, 12, 1) + dt.timedelta(days=i) for i in range(75)]
    for uid in users:
        for j, day in enumerate(dates):
            active = (uid + j) % 5 != 0
            buy = active and (uid * 3 + j) % 11 == 0
            rows.append({
                "user_id": int(uid), "event_date": day,
                "searches": int(active) * ((uid + j) % 4),
                "cat": int(active) * ((uid + 2 * j) % 5),
                "to_cart": int(active and (uid + j) % 3 == 0),
                "to_ord": int(buy), "gmv": float((uid + j) % 37 + 1) if buy else 0.0,
                "gmv_search": float((uid + j) % 19 + 1) if buy else 0.0,
                "gmv_cat": float((uid + j) % 13) if buy else 0.0,
            })
    return pl.DataFrame(rows)


def test_exact_date_boundaries_and_production_example():
    assert target_window(PRIMARY_CUTOFF) == (
        type(target_window(PRIMARY_CUTOFF))(dt.date(2026, 1, 15), dt.date(2026, 2, 13)))
    assert anniversary_window(PRIMARY_CUTOFF).start == dt.date(2025, 1, 15)
    assert anniversary_window(PRIMARY_CUTOFF).end == dt.date(2025, 2, 13)
    prod = dt.date(2026, 2, 13)
    assert target_window(prod).start == dt.date(2026, 2, 14)
    assert target_window(prod).end == dt.date(2026, 3, 15)
    assert anniversary_window(prod).start == dt.date(2025, 2, 14)
    assert anniversary_window(prod).end == dt.date(2025, 3, 15)


def test_calendar_year_shift_handles_leap_day():
    assert shift_calendar_year(dt.date(2024, 2, 29)) == dt.date(2023, 2, 28)
    assert shift_calendar_year(dt.date(2025, 3, 1)) == dt.date(2024, 3, 1)


def test_no_post_cutoff_rows_and_mutation_invariance():
    raw = tiny_raw()
    users = pl.DataFrame({"user_id": np.arange(1, 65, dtype=np.int64)})
    base = build_features(PRIMARY_CUTOFF, raw, users)
    future = pl.DataFrame({
        "user_id": [1], "event_date": [PRIMARY_CUTOFF + dt.timedelta(days=1)],
        "searches": [999999], "cat": [999999], "to_cart": [999999],
        "to_ord": [999999], "gmv": [999999.0], "gmv_search": [999999.0],
        "gmv_cat": [999999.0],
    })
    mutated = build_features(PRIMARY_CUTOFF, pl.concat([raw, future]), users)
    assert base.equals(mutated)
    with pytest.raises(AssertionError, match="crosses cutoff"):
        aggregate_window(raw, users, target_window(PRIMARY_CUTOFF), PRIMARY_CUTOFF, "bad")


def test_shifted_window_is_exactly_thirty_days_later_and_same_length():
    real = anniversary_window(PRIMARY_CUTOFF)
    shifted = shifted_year_window(PRIMARY_CUTOFF)
    assert shifted.start - real.start == dt.timedelta(days=30)
    assert shifted.end - real.end == dt.timedelta(days=30)
    assert shifted.days == real.days == 30
    assert pre_window(shifted, 30).end == shifted.start - dt.timedelta(days=1)


def test_shuffle_preserves_strata_and_all_annual_marginals():
    raw = tiny_raw()
    users = pl.DataFrame({"user_id": np.arange(1, 65, dtype=np.int64)})
    source = build_features(PRIMARY_CUTOFF, raw, users)
    shuffled, audit = shuffle_anniversary(source, PRIMARY_CUTOFF)
    assert audit["strata_preserved"]
    assert np.array_equal(shuffle_strata(source), shuffle_strata(shuffled))
    for name in [c for c in source.columns if c.startswith("old_")]:
        strata = shuffle_strata(source)
        for value in np.unique(strata):
            mask = strata == value
            assert np.array_equal(np.sort(source[name].to_numpy()[mask]),
                                  np.sort(shuffled[name].to_numpy()[mask]))


def test_deterministic_features_shuffle_and_output():
    raw = tiny_raw()
    users = pl.DataFrame({"user_id": np.arange(1, 65, dtype=np.int64)})
    a = build_features(PRIMARY_CUTOFF, raw, users)
    b = build_features(PRIMARY_CUTOFF, raw, users)
    assert feature_matrix(a).equals(feature_matrix(b))
    sa, audit_a = shuffle_anniversary(a, PRIMARY_CUTOFF)
    sb, audit_b = shuffle_anniversary(b, PRIMARY_CUTOFF)
    assert sa.equals(sb)
    assert audit_a == audit_b


def test_crossfit_isolation_and_recipient_does_not_affect_its_model():
    rng = np.random.default_rng(7)
    uid = np.arange(1, 401, dtype=np.int64)
    X = rng.normal(size=(len(uid), 4))
    features = pl.DataFrame({"user_id": uid, **{f"x{i}": X[:, i] for i in range(4)}})
    z = rng.normal(2.0, 0.2, len(uid))
    y = np.expm1(np.maximum(z + 0.2 * X[:, 0] + rng.normal(0, 0.2, len(uid)), 0))
    first = crossfit_arm("REAL", features, uid, y, z)
    groups = user_group(uid)
    y_mut = y.copy()
    y_mut[groups == 0] *= 50
    second = crossfit_arm("REAL", features, uid, y_mut, z)
    # Group A is predicted only by donor B, whose labels were not mutated.
    assert np.allclose(first.correction[groups == 0], second.correction[groups == 0])
    assert set(first.selected_shrink) == {0, 1}


def test_hash_matches_binary_split_and_is_deterministic():
    uid = np.arange(1, 1000, dtype=np.int64)
    assert np.array_equal(splitmix64(uid), splitmix64(uid.copy()))
    assert set(np.unique(user_group(uid))) == {0, 1}


def test_submission_alignment_and_rejections():
    sample = pl.DataFrame({"user_id": [3, 1, 2]})
    pred = pl.DataFrame({"user_id": [1, 2, 3], "predict": [1.0, 2.0, 3.0]})
    out = align_submission(sample, pred)
    assert out["user_id"].to_list() == [3, 1, 2]
    assert out["predict"].to_list() == [3.0, 1.0, 2.0]
    with pytest.raises(AssertionError):
        align_submission(sample, pred.filter(pl.col("user_id") != 2))
    with pytest.raises(AssertionError):
        align_submission(sample, pl.DataFrame({"user_id": [1, 2, 3],
                                               "predict": [1.0, -1.0, 3.0]}))
