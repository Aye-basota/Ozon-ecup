"""Anti-leak and semantics tests for RENEWAL-01."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src.config import TARGET_DAYS, VAL_FOLDS_S1
from src.renewal import (R0Model, _build_features_from_events, cohort_code,
                         km_curve, platt_crossfit)
from src.validation import get_folds


T = dt.date(2025, 4, 10)


def _events(rows):
    e = pl.DataFrame(rows, schema={"user_id": pl.Int64, "event_date": pl.Date},
                     orient="row").sort(["user_id", "event_date"])
    return e.with_columns(
        gap=pl.col("event_date").diff().over("user_id").dt.total_days().cast(pl.Float32))


def _synthetic(include_future: bool = True):
    rows = [(1, T-dt.timedelta(days=40)), (1, T-dt.timedelta(days=10)),
            (2, T-dt.timedelta(days=5))]
    if include_future:
        rows += [(1, T+dt.timedelta(days=1)), (2, T+dt.timedelta(days=2))]
    all_e = _events(rows)
    search_e = _events([(1, T-dt.timedelta(days=40)), (1, T-dt.timedelta(days=10))] +
                       ([(1, T+dt.timedelta(days=1))] if include_future else []))
    cat_e = _events([(1, T-dt.timedelta(days=10)), (2, T-dt.timedelta(days=5))] +
                    ([(2, T+dt.timedelta(days=2))] if include_future else []))
    users = pl.DataFrame({"user_id": [1, 2, 3]})
    return _build_features_from_events(T, users, all_e, search_e, cat_e)


def test_future_purchase_rows_do_not_change_features():
    a = _synthetic(True)
    b = _synthetic(False)
    assert a.columns == b.columns
    for col in a.columns:
        av, bv = a[col].to_numpy(), b[col].to_numpy()
        if col == "user_id":
            assert np.array_equal(av, bv)
        else:
            assert np.allclose(av, bv, equal_nan=True), col


def test_general_clock_exact_gaps_and_unfinished_interval():
    f = _synthetic().filter(pl.col("user_id") == 1).row(0, named=True)
    assert f["clk_n_events"] == 2
    assert f["clk_n_intervals"] == 1
    assert f["clk_recency"] == 10
    assert f["clk_gap_last1"] == 30
    assert f["clk_gap_mean"] == 30
    assert np.isclose(f["clk_rec_over_median"], 1/3)
    assert f["clk_risk_at_recency"] == 1
    assert f["clk_ends_next30"] == 1
    assert np.isclose(f["clk_share_near_30"], 1)


def test_cold_start_flags_cover_zero_one_two_and_history():
    f = _synthetic().sort("user_id")
    assert f["clk_cold_0"].to_list() == [0.0, 0.0, 1.0]
    assert f["clk_cold_1"].to_list() == [0.0, 1.0, 0.0]
    assert f["clk_cold_2"].to_list() == [1.0, 0.0, 0.0]
    assert f["clk_hist_3plus"].to_list() == [0.0, 0.0, 0.0]


def test_channel_clocks_are_separate_sequences():
    f = _synthetic().filter(pl.col("user_id") == 1).row(0, named=True)
    assert f["sclk_n_events"] == 2
    assert f["sclk_gap_last1"] == 30
    assert f["cclk_n_events"] == 1
    assert f["cclk_gap_last1"] is None


def test_kaplan_meier_uses_right_censoring_without_an_event():
    surv, risk = km_curve(np.array([10, 20]), np.array([True, False]), 25)
    assert risk[10] == 2
    assert risk[20] == 1
    assert surv[9] == 1.0
    assert surv[10] == 0.5
    assert surv[20] == 0.5


def test_r0_conditional_probability_and_individual_shrinkage():
    # S(5)=1, S(35)=0.6 -> population conditional p=0.4.
    curve = np.ones(100)
    curve[20:] = 0.8
    curve[35:] = 0.6
    risk = np.full(100, 1000.0)
    model = R0Model({-1: curve}, {-1: risk}, cold0_prior=0.12,
                    shrinkage=2.0, limit="test")
    frame = pl.DataFrame({
        "clk_n_events": [0, 1, 3], "clk_recency": [99.0, 5.0, 5.0],
        "clk_gap_median": [None, None, 20.0],
        "clk_risk_at_recency": [0, 0, 2], "clk_ends_next30": [0, 0, 1],
    })
    p = model.predict(frame)
    assert np.isclose(p[0], 0.12)
    assert np.isclose(p[1], 0.4)
    assert np.isclose(p[2], (1 + 2*0.4) / 4)


def test_cohort_boundaries_are_stable():
    x = np.array([np.nan, 10, 14, 15, 30, 31, 60, 61, 90, 91], float)
    assert cohort_code(x).tolist() == [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_platt_predictions_are_fold_exclusive_and_finite():
    rng = np.random.default_rng(42)
    cut = np.repeat([d.isoformat() for d in VAL_FOLDS_S1], 100)
    y = rng.integers(0, 2, len(cut))
    raw = np.clip(0.2 + 0.6*y + rng.normal(0, 0.1, len(y)), 0.01, 0.99)
    p, info = platt_crossfit(raw, y, cut)
    assert len(info) == 4
    assert np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()


def test_production_folds_use_the_exact_target_embargo():
    folds = get_folds(min_history=90)
    assert [v for _, v in folds] == VAL_FOLDS_S1
    for cuts, val in folds:
        assert all(t + dt.timedelta(days=TARGET_DAYS) <= val for t in cuts)
        # Grid is anchored at CORRIDOR_END, so the nearest eligible point is
        # 30..36 days behind the validation date rather than necessarily exact.
        gap = (val - max(cuts)).days
        assert TARGET_DAYS <= gap < TARGET_DAYS + 7
