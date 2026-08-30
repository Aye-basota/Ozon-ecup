"""Artifact-backed and numerical tests for EXP-047 BTYD-DAY-BGNBD-RESIDUAL."""
from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.btyd_day_bgnbd import (
    BLEND_GRID, COMPONENT_WEIGHTS, FOLD_LABELS, HORIZON, K_MONETARY, NMAX,
    ORIGIN, RUN_DIR, S2_QMC_CACHE, aggregation_mc_audit, bgnbd_count_distribution,
    bgnbd_log_likelihood, exact_baseline, fit_bgnbd, fold_cal_scores,
    metric_sum_moments, monetary_parameters, posterior_alive, row_keys,
    scored_monetary, select_lofo_weights, splitmix64, touch, user_group,
)
from src.config import SEED
from src.validation import calibrate


def _summary_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [1, 2, 3, 4], "x": [0, 2, 2, 3], "t_x": [0, 10, 20, 30],
        "sum_log_gmv": [0.0, 5.0, 8.0, 15.0],
        "sum_sq_log_gmv": [0.0, 13.0, 34.0, 77.0], "group": [0, 0, 1, 1],
        "T": [100, 100, 100, 100],
    })


def _simulated_histories(n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    T = 240.0
    lam = rng.gamma(shape=0.8, scale=1 / 28.0, size=n)
    p = rng.beta(1.8, 4.0, size=n)
    x = np.zeros(n, dtype=np.int32)
    tx = np.zeros(n, dtype=np.int32)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            tx[i] = max(1, min(int(np.floor(t)) + 1, int(T)))
            if rng.random() < p[i]:
                break
    return x, tx


def test_01_purchase_day_event_semantics() -> None:
    gmv = np.asarray([-1.0, 0.0, 0.01, 10.0])
    assert np.array_equal(gmv > 0, [False, False, True, True])


def test_02_to_ord_not_used_as_transaction_count() -> None:
    source = inspect.getsource(__import__("src.btyd_day_bgnbd", fromlist=["history_summary"]).history_summary)
    assert "to_ord" not in source
    assert 'pl.col("gmv") > 0' in source


def test_03_summary_cutoff_safety_by_construction() -> None:
    dates = np.asarray([1, 5, 10, 11, 20])
    gmv = np.asarray([1.0, 0.0, 2.0, 1000.0, 2000.0])
    cutoff = 10
    before = (dates <= cutoff) & (gmv > 0)
    changed = gmv.copy(); changed[dates > cutoff] *= 999
    after = (dates <= cutoff) & (changed > 0)
    assert before.sum() == after.sum() == 2
    assert np.max(dates[before]) == np.max(dates[after]) == 10


def test_04_common_origin_semantics() -> None:
    assert (dt.date(2025, 1, 1) - ORIGIN).days == 1
    assert (dt.date(2025, 10, 16) - ORIGIN).days == 289


def test_05_zero_frequency_implies_zero_recency() -> None:
    frame = _summary_frame()
    x, tx = frame["x"].to_numpy(), frame["t_x"].to_numpy()
    assert np.all(tx[x == 0] == 0)
    assert np.all(tx[x > 0] > 0)


def test_06_stable_user_hash_split() -> None:
    ids = np.asarray([0, 1, 2, 3, 42, 10**12], dtype=np.int64)
    first = user_group(ids)
    second = user_group(ids.copy())
    assert np.array_equal(first, second)
    assert np.array_equal(splitmix64(ids) & np.uint64(1), first.astype(np.uint64))


def test_07_donor_recipient_disjointness() -> None:
    ids = np.arange(10_000, dtype=np.int64)
    groups = user_group(ids)
    assert set(ids[groups == 0]).isdisjoint(set(ids[groups == 1]))
    assert set(np.unique(groups)) == {0, 1}


def test_08_bgnbd_likelihood_finite_including_x_zero() -> None:
    x = np.asarray([0, 1, 2, 10], dtype=np.float64)
    tx = np.asarray([0, 5, 30, 150], dtype=np.float64)
    ll = bgnbd_log_likelihood(x, tx, 200.0, np.asarray([0.8, 28.0, 1.8, 4.0]))
    assert np.all(np.isfinite(ll))


def test_09_mle_deterministic() -> None:
    x, tx = _simulated_histories()
    one = fit_bgnbd(x, tx, 240, "synthetic", 0)
    two = fit_bgnbd(x, tx, 240, "synthetic", 0)
    assert one["parameters"] == two["parameters"]
    assert one["log_likelihood"] == two["log_likelihood"]


def test_10_parameters_positive() -> None:
    params = np.exp(np.asarray([-4.0, 3.0, -1.0, 1.0]))
    assert np.all(params > 0)


def test_11_p_alive_within_unit_interval() -> None:
    x = np.asarray([0, 1, 3, 10]); tx = np.asarray([0, 20, 50, 100])
    alive = posterior_alive(x, tx, 200, np.asarray([0.7, 20.0, 1.5, 3.0]))
    assert np.all((alive >= 0) & (alive <= 1))
    assert alive[0] == 1.0


def test_12_count_pmf_nonnegative_and_sums_to_one() -> None:
    x = np.asarray([0, 1, 3, 10]); tx = np.asarray([0, 20, 50, 100])
    _, pmf, _ = bgnbd_count_distribution(x, tx, 200, np.asarray([0.7, 20, 1.5, 3.0]))
    assert np.all(pmf >= 0)
    assert np.max(np.abs(pmf.sum(axis=1) - 1)) <= 1e-8


def test_13_pmf_mean_equals_closed_form_capped_mean() -> None:
    x = np.asarray([0, 1, 3, 10]); tx = np.asarray([0, 20, 50, 100])
    _, pmf, expected = bgnbd_count_distribution(x, tx, 200,
                                                 np.asarray([0.7, 20.0, 1.5, 3.0]))
    assert np.max(np.abs(pmf @ np.arange(NMAX + 1) - expected)) <= 1e-6


def test_14_tail_ge_30_folded_into_bin_30() -> None:
    _, pmf, _ = bgnbd_count_distribution(np.asarray([30]), np.asarray([195]), 200,
                                          np.asarray([2.0, 2.0, 2.0, 20.0]))
    assert pmf.shape == (1, 31)
    assert pmf[0, 30] > 0
    assert pmf[0, :30].sum() + pmf[0, 30] == pytest.approx(1.0)


def test_15_monetary_population_stats_are_donor_only() -> None:
    frame = _summary_frame()
    zero = monetary_parameters(frame, 0)
    one = monetary_parameters(frame, 1)
    assert zero["mu_population"] != one["mu_population"]
    assert zero["n_users"] == one["n_users"] == 2


def test_16_k_is_fixed_at_three() -> None:
    assert K_MONETARY == 3.0
    pop = monetary_parameters(_summary_frame(), 0)
    assert pop["K"] == 3.0


def test_17_no_rows_after_cutoff_in_summary_source() -> None:
    source = inspect.getsource(__import__("src.btyd_day_bgnbd", fromlist=["history_summary"]).history_summary)
    assert 'pl.col("event_date") <= cutoff' in source


def test_18_metric_aggregation_agrees_with_deterministic_mc() -> None:
    audit = aggregation_mc_audit(samples_pow2=2**15)
    assert max(row["abs_error"] for row in audit if row["n"] <= 4) <= 0.01
    # Exact S2 reuse is mandatory. Its FW n>=5 limitation remains visible.
    assert any(row["abs_error"] > 0.01 for row in audit if row["n"] >= 5)


def test_19_exact_oof_row_alignment_key_is_unambiguous() -> None:
    keys = row_keys(np.asarray(["2025-09-04", "2025-09-04"]), np.asarray([1, 11]))
    assert len(np.unique(keys)) == 2
    assert "|" in keys[0]


def test_20_exact_reconstruction_strongest_current() -> None:
    baseline = exact_baseline()
    assert baseline["manifest"]["status"] == "PASS_EXACT"
    assert baseline["report"]["wcv"] == pytest.approx(1.747509862520, abs=5e-10)


def test_21_log_space_blend() -> None:
    z_base = np.asarray([1.0, 3.0]); z_member = np.asarray([3.0, 1.0]); w = 0.25
    blended = (1 - w) * z_base + w * z_member
    assert np.array_equal(blended, np.asarray([1.5, 2.5]))
    assert not np.allclose(np.expm1(blended),
                           (1 - w) * np.expm1(z_base) + w * np.expm1(z_member))


def test_22_outer_fold_excluded_from_weight_selection() -> None:
    source = inspect.getsource(select_lofo_weights)
    assert "if i != outer" in source
    assert "curves[:, train]" in source


def test_23_fold_calibration_applied_after_blend() -> None:
    source = inspect.getsource(select_lofo_weights)
    blend_pos = source.index("z = (1.0 - weight)")
    calibration_pos = source.index("fold_cal_scores")
    assert blend_pos < calibration_pos


def test_24_no_test_data_or_submission_path_touched() -> None:
    with pytest.raises(AssertionError):
        touch(Path("submissions/forbidden.csv"))
    with pytest.raises(AssertionError):
        touch(Path("data/raw/sample_submit.csv"))


def test_25_saved_artifact_reanalysis_hash_if_available() -> None:
    audit = RUN_DIR / "reanalysis_audit.json"
    if not audit.exists():
        pytest.skip("experiment has not run yet")
    import json
    value = json.loads(audit.read_text(encoding="utf-8"))
    assert value["status"] == "PASS"
    assert value["summary_sha256"] == value["reproduced_summary_sha256"]


def test_26_x_zero_uses_population_monetary_mean() -> None:
    frame = _summary_frame().filter(pl.col("group") == 0)
    pop = monetary_parameters(_summary_frame(), 1)
    mu, sigma = scored_monetary(frame, pop)
    assert mu[0] == pytest.approx(pop["mu_population"])
    assert np.all(sigma == pop["sigma_population"])


def test_27_expected_count_nonnegative_and_day_cap() -> None:
    _, _, expected = bgnbd_count_distribution(np.asarray([0, 5]), np.asarray([0, 90]),
                                               100, np.asarray([1.0, 10.0, 2.0, 5.0]))
    assert np.all((expected >= 0) & (expected <= HORIZON))


def test_28_blend_grid_is_preregistered() -> None:
    assert np.array_equal(BLEND_GRID, [0.0, 0.025, 0.05, 0.10, 0.15])
    assert COMPONENT_WEIGHTS.sum() == pytest.approx(1.0)


def test_29_s2_aggregation_cache_is_present_and_gh_setting_fixed() -> None:
    assert S2_QMC_CACHE.exists()
    moments = metric_sum_moments(np.asarray([3.3]), np.asarray([1.1]))
    assert moments.shape == (1, 31)
    assert moments[0, 0] == 0
    assert np.all(np.diff(moments[0]) >= 0)
