from __future__ import annotations

import numpy as np
import pytest

from src.zero2d_shrink import (LEVEL, amount_bins, apply_log_correction,
                               assert_test_order, calibrated_residuals, fit_mapping,
                               fit_p0_edges, isotonic_negative, mapping_correction,
                               set_log_level)
from src.validation import calibrate


def _toy_mapping_inputs():
    # Three fitting groups and one outer group; every amount/p0 cell is supported
    # when min_cell_rows=1.
    amount = np.tile(np.arange(7), 20)
    p0 = np.linspace(0.001, 0.999, len(amount))
    residual = -0.1 - 0.2 * p0
    weight = np.ones(len(amount))
    outer = np.zeros(len(amount), bool)
    outer[-20:] = True
    return amount, p0, residual, weight, outer


def test_amount_bins_have_exact_registered_boundaries():
    values = np.array([0, .999999, 1, 2.999999, 3, 9.999999, 10,
                       29.999999, 30, 49.999999, 50, 99.999999, 100, 1e9])
    assert np.array_equal(amount_bins(values),
                          [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])


def test_outer_fold_does_not_participate_in_p0_quantiles():
    _, p0, _, _, outer = _toy_mapping_inputs()
    edge_a = fit_p0_edges(p0, ~outer)
    changed = p0.copy()
    changed[outer] = 1.0
    edge_b = fit_p0_edges(changed, ~outer)
    assert np.array_equal(edge_a, edge_b)


def test_outer_fold_does_not_participate_in_residual_correction():
    amount, p0, residual, weight, outer = _toy_mapping_inputs()
    a = fit_mapping(amount, p0, residual, weight, ~outer,
                    strength=10, min_cell_rows=1)
    changed = residual.copy()
    changed[outer] = -1000
    b = fit_mapping(amount, p0, changed, weight, ~outer,
                    strength=10, min_cell_rows=1)
    assert np.array_equal(a["edges"], b["edges"])
    assert np.allclose(a["correction"], b["correction"])


def test_mapping_calibration_is_separate_inside_each_fitting_fold():
    fold = np.repeat(np.arange(3), 5)
    z = np.tile(np.linspace(0.5, 1.5, 5), 3)
    y = np.expm1(z + np.repeat([0.2, -0.1, 0.4], 5))
    residual, offsets, _ = calibrated_residuals(y, z, fold)
    for index in range(3):
        mask = fold == index
        expected_offset = calibrate(y[mask], z[mask])[0]
        assert offsets[index] == pytest.approx(expected_offset)
        assert residual[mask].mean() == pytest.approx(0.0, abs=1e-12)


def test_isotonic_is_monotone_negative_and_keeps_sparse_cells_zero():
    raw = np.array([-0.02, -0.08, -0.04, -0.12, -0.10])
    counts = np.array([1000, 100, 1000, 1000, 1000])
    weights = np.ones(5)
    out = isotonic_negative(raw, counts, weights, min_cell_rows=500)
    assert out[1] == 0.0
    assert np.all(out <= 0)
    assert np.all(np.diff(out) <= 1e-12)


def test_fitted_correction_is_always_non_positive():
    amount, p0, residual, weight, outer = _toy_mapping_inputs()
    residual[::2] = 1.0
    mapping = fit_mapping(amount, p0, residual, weight, ~outer,
                          strength=10, min_cell_rows=1)
    correction = mapping_correction(mapping, amount)
    assert np.all(mapping["correction"] <= 0)
    assert np.all(correction <= 0)


def test_eta_zero_reproduces_baseline_exactly():
    z = np.array([0.2, 1.0, 3.0])
    correction = np.array([-0.4, -0.2, 0.0])
    assert np.array_equal(apply_log_correction(z, correction, 0.0), z)


def test_correction_is_applied_in_log_space():
    raw_prediction = np.array([9.0])
    z = np.log1p(raw_prediction)
    corrected = apply_log_correction(z, np.array([-0.2]), 0.5)
    assert corrected[0] == pytest.approx(z[0] - 0.1)
    assert corrected[0] != pytest.approx(raw_prediction[0] - 0.1)


def test_test_order_must_equal_sample_submission():
    assert_test_order(np.array([3, 1, 2]), np.array([3, 1, 2]))
    with pytest.raises(AssertionError, match="sample submission"):
        assert_test_order(np.array([1, 2, 3]), np.array([3, 1, 2]))


def test_final_log_level_is_exactly_23293():
    z = np.array([-2.0, 0.1, 0.5, 1.0, 5.0])
    adjusted = set_log_level(z)
    assert np.all(adjusted >= 0)
    assert adjusted.mean() == pytest.approx(LEVEL, abs=1e-11)
