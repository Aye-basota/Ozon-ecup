import datetime as dt

import numpy as np
import polars as pl

import src.features as features


def _log(with_future: bool) -> pl.DataFrame:
    rows = []
    for day in range(1, 10):
        date = dt.date(2025, 1, day)
        rows += [
            (1, date, day, day % 3, day % 2, float(day * 10)),
            (2, date, 2 * day, (day + 1) % 3, (day + 1) % 2, float(day * 20)),
        ]
    if with_future:
        rows.append((1, dt.date(2025, 1, 11), 999999, 999999, 999999, 1e12))
    return pl.DataFrame(rows, schema=["user_id", "event_date", "searches", "to_cart", "to_ord", "gmv"],
                        orient="row").with_columns([
        pl.lit(0).cast(pl.Int64).alias("cat"),
        pl.lit(0).cast(pl.Int64).alias("search_to_cart"),
        pl.lit(0).cast(pl.Int64).alias("search_to_ord"),
        pl.lit(0).cast(pl.Int64).alias("cat_to_cart"),
        pl.lit(0).cast(pl.Int64).alias("cat_to_ord"),
        pl.lit(0.0).alias("gmv_search"), pl.lit(0.0).alias("gmv_cat"),
    ])


def test_platform_factors_ignore_future(monkeypatch):
    T = dt.date(2025, 1, 9)
    monkeypatch.setattr(features, "panel_users", lambda _: pl.DataFrame({"user_id": [1, 2]}))
    monkeypatch.setattr(features, "load", lambda: _log(False))
    expected = features._platform_detrend_frame(T, "real")
    monkeypatch.setattr(features, "load", lambda: _log(True))
    got = features._platform_detrend_frame(T, "real")
    assert expected.equals(got)


def test_platform_placebo_preserves_factor_multiset(monkeypatch):
    T = dt.date(2025, 1, 9)
    monkeypatch.setattr(features, "panel_users", lambda _: pl.DataFrame({"user_id": [1, 2]}))
    monkeypatch.setattr(features, "load", lambda: _log(False))
    real = features._platform_daily_factors(T, "real")
    shuffled = features._platform_daily_factors(T, "shuffled")
    cols = [c for c in real.columns if c != "event_date"]
    assert np.allclose(np.sort(real.select(cols).to_numpy(), axis=0),
                       np.sort(shuffled.select(cols).to_numpy(), axis=0))


def test_platform_profile_is_opt_in(monkeypatch):
    T = dt.date(2025, 1, 9)
    monkeypatch.setattr(features, "panel_users", lambda _: pl.DataFrame({"user_id": [1, 2]}))
    monkeypatch.setattr(features, "load", lambda: _log(False))
    base = features.build_features(T, L=90)
    enriched = features.build_features(T, L=90, platform_detrend_source="real")
    assert enriched.columns[:len(base.columns)] == base.columns
    assert enriched.columns[len(base.columns):] == features.PLATFORM_DETREND_COLUMNS
