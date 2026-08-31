import datetime as dt

import numpy as np

from src.selection_mismatch_cv import (COMPETITION_BLOCKS, weighted_calibrate,
                                       window_spec)


def test_future_windows_exclude_target_and_are_exact_30_days():
    V = dt.date(2025, 10, 2)
    s = window_spec(V)
    target = set(s["target"][0] + dt.timedelta(days=i) for i in range(30))
    future = set()
    for name in ("F1", "F2", "F3"):
        a, b = s[name]
        assert (b - a).days == 29
        future |= set(a + dt.timedelta(days=i) for i in range(30))
    assert len(future) == 90
    assert not target & future


def test_last_fold_future_blocks_equal_competition_selection():
    s = window_spec(dt.date(2025, 10, 16))
    assert [s["F1"], s["F2"], s["F3"]] == COMPETITION_BLOCKS


def test_weighted_calibration_matches_unweighted_for_unit_weights():
    y = np.array([0.0, 1.0, 4.0, 20.0])
    z = np.array([0.2, 0.5, 1.8, 2.7])
    from src.validation import calibrate
    a = calibrate(y, z)
    b = weighted_calibrate(y, z, np.ones(len(y)))
    assert np.allclose(a, b, atol=1e-12)


def test_matching_weights_recover_reference_mass():
    k = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    pi_fold = np.bincount(k, minlength=4) / len(k)
    pi_ref = np.array([.2, .3, .1, .4])
    w = np.array([pi_ref[j] / pi_fold[j] for j in k])
    got = np.array([w[k == j].sum() / w.sum() for j in range(4)])
    assert np.allclose(got, pi_ref)


def test_weighted_calibration_invariant_to_weight_scale():
    rng = np.random.default_rng(42)
    y = rng.gamma(2, 3, 100)
    z = rng.normal(1, .5, 100)
    w = rng.uniform(.1, 2, 100)
    assert np.allclose(weighted_calibrate(y, z, w),
                       weighted_calibrate(y, z, 17*w), atol=1e-12)
