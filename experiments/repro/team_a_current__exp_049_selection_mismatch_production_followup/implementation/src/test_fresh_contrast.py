"""Invariants for FRESH-CONTRAST-MOE."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.fresh_contrast import (ALPHAS, add_log_correction, align_to_sample,
                                conditional_contrasts, merge_crossfit, nested_lofo,
                                process_correction, stable_group, validate_crossfit)


def test_stable_hash_split_is_deterministic_and_order_independent():
    uid = np.arange(1, 10001, dtype=np.int64)
    g = stable_group(uid)
    assert np.array_equal(g, stable_group(uid))
    p = np.random.default_rng(42).permutation(len(uid))
    assert np.array_equal(g[p], stable_group(uid[p]))
    assert set(np.unique(g)) == {0, 1}


def test_hash_halves_are_disjoint_and_extra_donor_cannot_receive_own_prediction():
    uid = np.arange(1, 5001, dtype=np.int64)
    g = stable_group(uid)
    a, b = uid[g == 0], uid[g == 1]
    assert np.intersect1d(a, b).size == 0
    validate_crossfit(a, b, recipient_group=0, donor_group=1)
    validate_crossfit(b, a, recipient_group=1, donor_group=0)
    with pytest.raises(AssertionError):
        validate_crossfit(a, a, recipient_group=0, donor_group=0)


def test_full_panel_merge_uses_opposite_donor_head_for_each_half():
    uid = np.arange(1, 101, dtype=np.int64)
    g = stable_group(uid)
    from_b = np.full(len(uid), 11.0)
    from_a = np.full(len(uid), 22.0)
    full = merge_crossfit(g, from_b, from_a)
    assert len(full) == len(uid)
    assert np.all(full[g == 0] == 11.0)
    assert np.all(full[g == 1] == 22.0)


def test_registered_contrasts_are_exact_subtractions():
    clean = np.array([1.0, 2.0, 3.0])
    vol = np.array([1.5, 1.0, 4.0])
    fresh = np.array([0.5, 3.0, 5.0])
    d_fresh, d_vol = conditional_contrasts(clean, vol, fresh)
    assert np.array_equal(d_fresh, fresh - clean)
    assert np.array_equal(d_vol, vol - clean)


def test_high16_uses_only_cutoff_safe_w180_and_is_gated_before_centering():
    raw = np.array([-2.0, -1.0, 1.0, 2.0])
    w180 = np.array([15, 16, 1, 20])
    corr, _ = process_correction(raw, (-10.0, 10.0), "HIGH16", w180)
    expected = np.array([0.0, -1.0, 0.0, 2.0])
    expected -= expected.mean()
    assert np.array_equal(corr, expected)


def _synthetic_folds():
    rng = np.random.default_rng(42)
    out = []
    for i in range(4):
        n = 250
        z = rng.normal(2.0, 0.4, n)
        ly = z + rng.normal(0, 0.5, n)
        y = np.expm1(np.maximum(ly, 0))
        out.append({"y": y, "z_base": z, "w180": rng.integers(0, 25, n),
                    "raw_fresh": rng.normal(0, 0.2, n)})
    return out


def test_nested_lofo_outer_fold_is_absent_from_selection_and_winsor_bounds():
    folds = _synthetic_folds()
    result = nested_lofo(folds, "raw_fresh")
    for h, selected in enumerate(result["selected"]):
        assert h not in selected["selection_folds"]
        assert h not in selected["heldout_bounds_folds"]
        for meta in selected["selection_bounds"].values():
            assert h not in meta["donor_folds"]

    # Mutating the held-out fold's labels and correction cannot alter its choice.
    for h in range(4):
        changed = copy.deepcopy(folds)
        changed[h]["y"] = changed[h]["y"] * 1000
        changed[h]["raw_fresh"] = changed[h]["raw_fresh"] * 10000 + 500
        again = nested_lofo(changed, "raw_fresh")
        a, b = result["selected"][h], again["selected"][h]
        assert (a["variant"], a["alpha"]) == (b["variant"], b["alpha"])


def test_alpha_zero_bitwise_reproduces_strongest_current():
    z = np.array([0.0, 0.1, 2.5, 8.0], dtype=np.float64)
    corr = np.array([100.0, -20.0, 3.0, -7.0])
    got = add_log_correction(z, corr, ALPHAS[0])
    assert np.array_equal(got, z)
    assert got.dtype == z.dtype


def test_test_prediction_order_matches_sample_submission():
    uid = np.array([30, 10, 20])
    z = np.array([3.0, 1.0, 2.0])
    sample = np.array([20, 30, 10])
    assert np.array_equal(align_to_sample(uid, z, sample), [2.0, 3.0, 1.0])
    with pytest.raises(AssertionError):
        align_to_sample(uid, z, np.array([20, 30, 99]))


def test_correction_is_additive_in_log_space_not_raw_prediction_space():
    z = np.array([1.0, 2.0])
    corr = np.array([0.5, -0.25])
    got = add_log_correction(z, corr, 0.5)
    assert np.allclose(got, z + 0.5 * corr)
    assert not np.allclose(np.expm1(got), np.expm1(z) + 0.5 * corr)
