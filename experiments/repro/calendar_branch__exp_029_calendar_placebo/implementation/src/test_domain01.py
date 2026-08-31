"""Fast unit tests for DOMAIN-01 invariants and weighting math."""
from __future__ import annotations

import numpy as np

from src.domain01 import (clipped_importance_weights, density_ratio, domain_feature_sets,
                          effective_sample_size, population_stability_index,
                          user_group_fold, weighted_calibrate, weighted_rmsle_z)


def test_user_group_split_is_stable_across_observations():
    users = np.array([10, 20, 10, 30, 20, 10])
    folds = user_group_fold(users, 5)
    assert len(set(folds[users == 10])) == 1
    assert len(set(folds[users == 20])) == 1


def test_density_ratio_is_one_at_domain_prior():
    prior = 0.25
    got = density_ratio(np.array([prior]), np.array([prior]))
    assert np.allclose(got, 1.0)


def test_importance_weights_are_clipped_normalized_and_have_valid_neff():
    w = clipped_importance_weights([1e-9, 0.5, 1.0, 2.0, 1e9], 1.0, 0.25, 4.0)
    assert np.isclose(w.mean(), 1.0)
    assert w.min() >= 0.25
    assert w.max() <= 4.0
    assert 1 <= effective_sample_size(w) <= len(w)


def test_weighted_calibration_cannot_worsen_same_weighted_metric():
    y = np.array([0.0, 0.0, 3.0, 8.0, 20.0])
    z = np.array([0.2, 0.5, 0.4, 1.2, 1.8])
    w = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
    before = weighted_rmsle_z(y, z, w)
    offset, after = weighted_calibrate(y, z, w)
    assert np.isfinite(offset)
    assert after <= before


def test_psi_is_zero_for_identical_and_positive_for_shift():
    x = np.arange(100, dtype=float)
    assert population_stability_index(x, x) < 1e-12
    assert population_stability_index(x, x + 50) > 0.1


def test_primary_feature_contract_excludes_source_metadata():
    feats = ["w30_gmv", "rec_buy", "tenure_frac", "w365_orders", "trend_gmv_30_60"]
    sets = domain_feature_sets(feats)
    assert sets["all"] == feats
    assert "rec_buy" in sets["production_depth"]
    assert "tenure_frac" in sets["production_depth"]
    assert "w30_gmv" in sets["behavioral"]
    assert "w365_orders" not in sets["behavioral"]


def test_forbidden_feature_is_rejected():
    try:
        domain_feature_sets(["w30_gmv", "cutoff_month"])
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("cutoff feature must be rejected")
