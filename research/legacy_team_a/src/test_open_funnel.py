import datetime as dt

import polars as pl

import src.features as features


def _tiny_log() -> pl.DataFrame:
    rows = [
        # user 1: buy at age 5, then open search/cart at ages 3 and 1
        (1, dt.date(2025, 1, 5), 0, 0, 0, 0, 100.0),
        (1, dt.date(2025, 1, 7), 2, 0, 0, 0, 0.0),
        (1, dt.date(2025, 1, 9), 1, 1, 1, 0, 0.0),
        (1, dt.date(2025, 1, 10), 999, 999, 0, 0, 0.0),  # after cutoff: forbidden
        # user 2 never bought: all historical funnel rows are open
        (2, dt.date(2025, 1, 8), 3, 2, 1, 0, 0.0),
    ]
    return pl.DataFrame(rows, schema=[
        "user_id", "event_date", "searches", "to_cart", "search_to_cart", "cat_to_cart", "gmv"
    ], orient="row").with_columns([
        pl.lit(0).cast(pl.Int64).alias("cat"),
        pl.lit(0).cast(pl.Int64).alias("search_to_ord"),
        pl.lit(0).cast(pl.Int64).alias("cat_to_ord"),
        pl.lit(0).cast(pl.Int64).alias("to_ord"),
        pl.lit(0.0).alias("gmv_search"),
        pl.lit(0.0).alias("gmv_cat"),
    ])


def test_open_funnel_is_cutoff_safe_and_anchored(monkeypatch):
    monkeypatch.setattr(features, "load", _tiny_log)
    got = features._open_funnel_frame(dt.date(2025, 1, 9)).sort("user_id")
    u1, u2 = got.to_dicts()
    assert u1["of90_searches"] == 3
    assert u1["of90_carts"] == 1
    assert u1["of90_search_days"] == 2
    assert u1["of90_cart_days"] == 1
    assert u1["of90_oldest_search_age"] == 2
    assert u2["of90_searches"] == 3
    assert u2["of90_carts"] == 2


def test_build_features_adds_only_opt_in_columns(monkeypatch):
    monkeypatch.setattr(features, "load", _tiny_log)
    T = dt.date(2025, 1, 9)
    base = features.build_features(T, L=90)
    enriched = features.build_features(T, L=90, open_funnel_features=True)
    assert enriched.columns[: len(base.columns)] == base.columns
    assert enriched.columns[len(base.columns):] == features.OPEN_FUNNEL_COLUMNS
    assert enriched.height == base.height
