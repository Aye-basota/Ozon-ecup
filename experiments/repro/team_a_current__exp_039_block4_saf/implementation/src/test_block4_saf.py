from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src import block4_saf as saf
from src import data, features


def row(uid: int, day: dt.date, *, gmv: float = 0.0, searches: int = 0,
        to_ord: int = 0) -> dict:
    return {
        "user_id": uid, "event_date": day, "searches": searches, "cat": 0,
        "search_to_cart": 0, "search_to_ord": to_ord, "cat_to_cart": 0,
        "cat_to_ord": to_ord, "to_cart": 0, "to_ord": to_ord, "gmv": gmv,
        "gmv_search": gmv, "gmv_cat": gmv,
    }


def patch_data(monkeypatch, tmp_path, rows) -> pl.DataFrame:
    df = pl.DataFrame(rows).with_columns(pl.col("event_date").cast(pl.Date))
    monkeypatch.setattr(data, "load", lambda: df)
    monkeypatch.setattr(features, "load", lambda: df)
    monkeypatch.setattr(features, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(saf, "DATA_PROCESSED", tmp_path)
    return df


def test_target_window_is_open_left_closed_right():
    T = dt.date(2025, 5, 10)
    assert saf.window(T) == (dt.date(2025, 5, 11), dt.date(2025, 6, 9))


def test_activity_boundaries_and_purchase_implies_activity(monkeypatch, tmp_path):
    T = dt.date(2025, 5, 10)
    patch_data(monkeypatch, tmp_path, [
        row(1, T, gmv=100),                         # excluded left boundary
        row(2, T + dt.timedelta(days=1), gmv=10),  # included
        row(3, T + dt.timedelta(days=30), gmv=20), # included right boundary
        row(4, T + dt.timedelta(days=31), gmv=30), # excluded
        row(5, T + dt.timedelta(days=5), searches=1),
    ])
    got = saf.future_all(T).sort("user_id")
    assert got["user_id"].to_list() == [2, 3, 5]
    assert got["activity"].to_list() == [1, 1, 1]
    assert got["y"].to_list() == [10.0, 20.0, 0.0]
    assert got.filter((pl.col("y") > 0) & (pl.col("activity") != 1)).height == 0


def test_panel_users_are_active_in_B2_and_B3(monkeypatch, tmp_path):
    V = dt.date(2025, 10, 16)
    C, F = V - dt.timedelta(days=60), V - dt.timedelta(days=30)
    rows = [
        # user 1 has a row in every panel block
        row(1, C), row(1, C + dt.timedelta(days=1)), row(1, F + dt.timedelta(days=1)),
        # user 2 misses B2 and must not enter P_V
        row(2, C), row(2, F + dt.timedelta(days=1)),
    ]
    patch_data(monkeypatch, tmp_path, rows)
    users = features.panel_users(V, 3)
    assert users["user_id"].to_list() == [1]
    assert saf.future_labels(C, users)["activity"].to_list() == [1]
    assert saf.future_labels(F, users)["activity"].to_list() == [1]


def test_block_features_ignore_future_rows(monkeypatch, tmp_path):
    T = dt.date(2025, 7, 1)
    past = [row(1, T - dt.timedelta(days=1), gmv=10, searches=2, to_ord=1),
            row(1, T - dt.timedelta(days=40), gmv=5, searches=1)]
    patch_data(monkeypatch, tmp_path, past)
    a = features.build_features(T, 180, block_features=True)
    patch_data(monkeypatch, tmp_path, past + [
        row(1, T + dt.timedelta(days=1), gmv=10**9, searches=10**6, to_ord=10**5)
    ])
    b = features.build_features(T, 180, block_features=True)
    assert a.equals(b, null_equal=True)


def test_default_feature_pipeline_is_backward_compatible(monkeypatch, tmp_path):
    T = dt.date(2025, 7, 1)
    patch_data(monkeypatch, tmp_path, [
        row(1, T, gmv=10, searches=2, to_ord=1),
        row(1, T - dt.timedelta(days=35), searches=1),
        row(1, T - dt.timedelta(days=70), gmv=3),
    ])
    old = features.build_features(T, 180)
    new = features.build_features(T, 180, block_features=True)
    assert new.select(old.columns).equals(old, null_equal=True)
    assert "block0_gmv" not in new.columns       # exact duplicate of w30_gmv is forbidden
    assert "block0_gmv_search" in new.columns    # no old equivalent exists
    assert "block1_gmv" in new.columns and "block2_gmv" in new.columns
    assert "block_accel_gmv" in new.columns


def test_user_id_is_not_a_model_feature(monkeypatch, tmp_path):
    T = dt.date(2025, 7, 1)
    patch_data(monkeypatch, tmp_path, [row(1, T)])
    frame = features.build_features(T, 180, block_features=True)
    assert "user_id" not in features.feature_names(frame)


def test_crossfit_groups_are_global_stable_and_disjoint():
    uid = np.arange(1, 1001, dtype=np.uint64)
    a = saf.splitmix_group(uid)
    b = saf.splitmix_group(uid.copy())
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {0, 1}
    assert not np.any((a == 0) & (a == 1))
    assert np.intersect1d(uid[a == 0], uid[a == 1]).size == 0


def test_production_geometry_is_pinned():
    assert saf.PROD_V == dt.date(2026, 2, 13)
    assert saf.PROD_C == dt.date(2025, 12, 15)
    assert saf.PROD_F == dt.date(2026, 1, 14)
    assert saf.PROD_C == saf.PROD_V - dt.timedelta(days=60)
    assert saf.PROD_F == saf.PROD_V - dt.timedelta(days=30)
    assert saf.window(saf.PROD_C) == (dt.date(2025, 12, 16), dt.date(2026, 1, 14))
    assert saf.window(saf.PROD_F) == (dt.date(2026, 1, 15), dt.date(2026, 2, 13))


def test_q_never_uses_late_labels():
    for V in saf.VAL_FOLDS_S1:
        cuts = saf.clean_q_cutoffs(V)
        assert all(T <= saf.CORRIDOR_END for T in cuts)
        assert all(T + dt.timedelta(days=30) <= V for T in cuts)
    assert max(saf.clean_q_cutoffs(None)) == saf.CORRIDOR_END
    assert all(T <= saf.CORRIDOR_END for T in saf.clean_q_cutoffs(None))


def test_model_seeds_come_from_config_seed():
    assert saf.SEEDS == (saf.SEED, saf.SEED + 1, saf.SEED + 2)
    assert saf.SEEDS == (42, 43, 44)
