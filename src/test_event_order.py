import datetime as dt

import polars as pl

import src.features as features


def _tiny_log() -> pl.DataFrame:
    rows = [
        (1, dt.date(2025, 1, 2), 1, 0, 0, 0.0),
        (1, dt.date(2025, 1, 4), 1, 1, 0, 0.0),
        (1, dt.date(2025, 1, 8), 0, 0, 1, 50.0),
        (1, dt.date(2025, 1, 11), 999, 999, 999, 999.0),  # future: forbidden
        (2, dt.date(2025, 1, 3), 1, 0, 0, 0.0),
        (2, dt.date(2025, 1, 5), 0, 1, 0, 0.0),
        (2, dt.date(2025, 1, 7), 0, 0, 1, 20.0),
        (2, dt.date(2025, 1, 9), 1, 1, 0, 0.0),
    ]
    return pl.DataFrame(rows, schema=[
        "user_id", "event_date", "searches", "to_cart", "to_ord", "gmv"
    ], orient="row").with_columns([
        pl.lit(0).cast(pl.Int64).alias("cat"),
        pl.lit(0).cast(pl.Int64).alias("search_to_cart"),
        pl.lit(0).cast(pl.Int64).alias("search_to_ord"),
        pl.lit(0).cast(pl.Int64).alias("cat_to_cart"),
        pl.lit(0).cast(pl.Int64).alias("cat_to_ord"),
        pl.lit(0.0).alias("gmv_search"),
        pl.lit(0.0).alias("gmv_cat"),
    ])


def test_event_order_real_is_cutoff_safe(monkeypatch):
    monkeypatch.setattr(features, "load", _tiny_log)
    got = features._event_order_frame(dt.date(2025, 1, 10), "real").sort("user_id")
    u1 = got.row(by_predicate=pl.col("user_id") == 1, named=True)
    assert u1["eo90_transition_count"] == 2
    assert u1["eo90_up_count"] == 2
    assert u1["eo90_search_to_cartbuy_count"] == 1
    assert u1["eo90_cart_to_buy_count"] == 1
    assert u1["eo90_nobuy_to_buy_count"] == 1


def test_shuffle_preserves_each_user_state_multiset_and_dates(monkeypatch):
    monkeypatch.setattr(features, "load", _tiny_log)
    T = dt.date(2025, 1, 10)
    real = features._event_order_daily(T, "real")
    shuf = features._event_order_daily(T, "shuffled")
    assert real.select(["user_id", "event_date"]).equals(shuf.select(["user_id", "event_date"]))
    for uid in real["user_id"].unique().to_list():
        a = real.filter(pl.col("user_id") == uid)["_eo_state"].sort().to_list()
        b = shuf.filter(pl.col("user_id") == uid)["_eo_state"].sort().to_list()
        assert a == b


def test_build_features_adds_event_order_only_when_opted_in(monkeypatch):
    monkeypatch.setattr(features, "load", _tiny_log)
    T = dt.date(2025, 1, 10)
    base = features.build_features(T, L=90)
    enriched = features.build_features(T, L=90, event_order_source="real")
    assert enriched.columns[:len(base.columns)] == base.columns
    assert enriched.columns[len(base.columns):] == features.EVENT_ORDER_COLUMNS
    assert enriched.height == base.height
