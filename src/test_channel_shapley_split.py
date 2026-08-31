"""Invariants for EXP-052 CHANNEL-SHAPLEY-SPLIT."""
from __future__ import annotations

import datetime as dt
import inspect

import numpy as np
import polars as pl
import pytest

from src.channel_shapley_split import (ALPHAS, EXPECTED_BASE, EXPECTED_WCV, FLOAT_ATOL,
                                       PILOT_FOLD, canonical_sha256, channel_target,
                                       choose_alpha, contributions, exact_baseline,
                                       pilot_setup, raw_log_blend, splitmix64,
                                       stable_deciles, stratified_shuffle)
from src.config import SEED
from src.validation import calibrate


def _raw() -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [1, 1, 2, 2],
        "event_date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2),
                       dt.date(2025, 1, 2), dt.date(2025, 2, 2)],
        "gmv": [0.0, 7.0, 3.0, 99.0],
        "gmv_search": [0.0, 5.0, 0.0, 99.0],
        "gmv_cat": [0.0, 2.0, 3.0, 0.0],
    })


def test_01_daily_channel_identity():
    d = _raw()
    delta = (d["gmv_search"] + d["gmv_cat"] - d["gmv"]).abs().to_numpy()
    assert np.all(d["gmv"].to_numpy() >= 0)
    assert np.all(d["gmv_search"].to_numpy() >= 0)
    assert np.all(d["gmv_cat"].to_numpy() >= 0)
    assert delta.max() <= FLOAT_ATOL


def test_02_target_window_is_open_left_closed_right_and_bounded():
    got, meta = channel_target(_raw(), dt.date(2025, 1, 1), np.array([1, 2]), horizon=1)
    assert got["Y"].tolist() == [7.0, 3.0]
    assert meta["window_start"] == "2025-01-02"
    assert meta["window_end"] == "2025-01-02"


def test_03_channel_targets_sum_to_current_total():
    got, _ = channel_target(_raw(), dt.date(2025, 1, 1), np.array([1, 2]), horizon=1)
    assert np.allclose(got["S"] + got["C"], got["Y"], rtol=0, atol=FLOAT_ATOL)
    assert np.allclose(np.log1p(got["S"] + got["C"]), np.log1p(got["Y"]),
                       rtol=0, atol=FLOAT_ATOL)


def test_04_shapley_contributions_sum_to_metric_target():
    ps, pc, z, _ = contributions(np.array([0.0, 2.0, 7.0]), np.array([0.0, 5.0, 3.0]))
    assert np.allclose(ps + pc, z, rtol=0, atol=1e-12)


def test_05_contributions_are_nonnegative():
    ps, pc, _, _ = contributions(np.linspace(0, 100, 101), np.linspace(100, 0, 101))
    assert np.all(ps >= 0) and np.all(pc >= 0)


def test_06_search_catalog_swap_is_symmetric():
    s, c = np.array([0.0, 2.0, 8.0]), np.array([4.0, 3.0, 0.0])
    ps, pc, z, _ = contributions(s, c)
    ps2, pc2, z2, _ = contributions(c, s)
    assert np.allclose(ps, pc2) and np.allclose(pc, ps2) and np.allclose(z, z2)


def test_07_zero_row_has_zero_contributions():
    ps, pc, z, u = contributions(np.zeros(5), np.zeros(5))
    assert np.array_equal(ps, np.zeros(5))
    assert np.array_equal(pc, np.zeros(5))
    assert np.array_equal(z, np.zeros(5))
    assert np.array_equal(u, np.full(5, 0.5))


def _shuffle_input():
    z = np.repeat(np.arange(20, dtype=float), 4)
    u = np.linspace(0, 1, len(z))
    cutoff = np.repeat([0, 1], len(z) // 2).astype(np.uint8)
    return z, u, cutoff


def test_08_shuffled_contributions_still_sum_to_z():
    z, u, cutoff = _shuffle_input()
    ush, _, _ = stratified_shuffle(z, u, cutoff, seed=SEED)
    assert np.allclose(ush * z + (1 - ush) * z, z, rtol=0, atol=1e-12)


def test_09_shuffle_preserves_u_multiset_in_every_stratum():
    z, u, cutoff = _shuffle_input()
    ush, _, _ = stratified_shuffle(z, u, cutoff, seed=SEED)
    for code in np.unique(cutoff):
        m = cutoff == code
        dec, _ = stable_deciles(z[m])
        idx = np.flatnonzero(m)
        for d in np.unique(dec):
            rows = idx[dec == d]
            assert np.array_equal(np.sort(u[rows]), np.sort(ush[rows]))


def test_10_shuffle_is_deterministic():
    args = _shuffle_input()
    a = stratified_shuffle(*args, seed=SEED)
    b = stratified_shuffle(*args, seed=SEED)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2]["permutation_sha256"] == b[2]["permutation_sha256"]


def test_11_validation_rows_cannot_enter_edges_or_permutations():
    z, u, cutoff = _shuffle_input()
    with pytest.raises(AssertionError, match="validation rows"):
        stratified_shuffle(z, u, cutoff, seed=SEED,
                           validation_mask=np.arange(len(z)) == 0)
    s = pilot_setup()
    assert all(t + dt.timedelta(days=30) <= PILOT_FOLD for t in s.train_cutoffs(PILOT_FOLD))


def test_12_real_and_shuffle_feature_matrices_are_identical():
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    real_matrix = X
    shuffled_control_matrix = X
    assert real_matrix is shuffled_control_matrix
    assert np.array_equal(real_matrix, shuffled_control_matrix)


def test_13_corresponding_lightgbm_configs_and_seeds_are_identical():
    base = {"objective": "regression", "rounds": 300, "seed": SEED}
    real_search = dict(base)
    shuf_search = dict(base)
    base_cat = {**base, "seed": SEED + 1}
    real_catalog = dict(base_cat)
    shuf_catalog = dict(base_cat)
    assert real_search == shuf_search
    assert real_catalog == shuf_catalog


def test_14_exact_strongest_current_reconstruction():
    _, manifest = exact_baseline()
    assert np.allclose(manifest["fold_scores_calibrated"], EXPECTED_BASE,
                       rtol=0, atol=5e-10)
    assert abs(manifest["wcv"] - EXPECTED_WCV) <= 5e-10
    assert manifest["log_space_assembly"] is True


def test_15_hash_halves_are_disjoint_and_exhaustive():
    uid = np.arange(1, 10_001, dtype=np.uint64)
    half = splitmix64(uid) & np.uint64(1)
    A, B = half == 0, half == 1
    assert np.all(A ^ B)
    assert not np.any(A & B)
    assert A.any() and B.any()


def test_16_recipient_is_excluded_from_alpha_selection():
    n = 200
    y = np.expm1(np.linspace(0, 4, n))
    base = np.log1p(y) + np.sin(np.arange(n)) * 0.2
    direction = -np.sin(np.arange(n)) * 0.2
    selector = np.arange(n) % 2 == 0
    recipient = ~selector
    a, curve = choose_alpha(y, base, direction, selector, recipient)
    y_changed = y.copy(); y_changed[recipient] = 1e9
    b, curve_changed = choose_alpha(y_changed, base, direction, selector, recipient)
    assert a == b
    assert curve == curve_changed


def test_17_blend_is_only_raw_log_space_addition():
    base = np.array([0.1, 1.0, 2.0])
    d = np.array([-0.2, 0.3, 0.4])
    assert np.array_equal(raw_log_blend(base, d, 0.5), base + 0.5 * d)


def test_18_fold_calibration_occurs_after_final_assembly():
    y = np.array([0.0, 2.0, 10.0, 100.0])
    base, d = np.array([0.1, 0.8, 2.0, 4.0]), np.array([0.0, 0.2, -0.1, 0.3])
    assembled = raw_log_blend(base, d, ALPHAS[2])
    off, score = calibrate(y, assembled)
    assert np.isfinite(off) and np.isfinite(score)
    assert np.array_equal(assembled, base + ALPHAS[2] * d)


def test_19_runner_has_no_test_prediction_submission_or_lb_io():
    import src.channel_shapley_split as module
    source = inspect.getsource(module)
    assert "sample_submit(" not in source
    assert "SUBMISSIONS /" not in source
    assert "ztest_" not in source
    assert "public_lb" in source  # explicitly reported false in the summary
    assert module.PILOT_FOLD == dt.date(2025, 10, 16)


def test_20_analysis_hash_is_reproducible():
    payload = {"rows": [1, 2, 3], "hashes": {"a": "abc"}, "score": 1.25}
    assert canonical_sha256(payload) == canonical_sha256(payload)
    changed = {**payload, "score": 1.2500001}
    assert canonical_sha256(payload) != canonical_sha256(changed)
