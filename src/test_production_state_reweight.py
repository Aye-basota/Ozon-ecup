from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from src import production_state_reweight as exp


def test_exact_unc_and_strongest_replay():
    audit = exp.phase0_audit()
    assert audit["phase0"]["status"] == "PASS_BITWISE"
    assert abs(audit["strongest_current"]["rmsle_cal"] - exp.EXPECTED_LATE) < 5e-10


def test_domain_features_are_target_free_and_semantic():
    names = exp.saved_features("UNC") + ["cutoff_index", "query_weekday", "avail", "w365_gmv"]
    selected = exp.select_domain_features(names)
    assert selected
    assert "rec_buy" in selected
    assert "w180_gmv" in selected
    assert "trend_gmv_60_180" in selected
    assert all("w365" not in name for name in selected)
    assert all(not any(token in name.lower() for token in exp.FORBIDDEN_FEATURE_TOKENS)
               for name in selected)


def test_outer_user_side_is_excluded_from_own_domain_fit():
    source_uid = np.arange(1, 41, dtype=np.int64)
    target_uid = np.arange(1, 41, dtype=np.int64)
    source_X = source_uid[:, None].astype(np.float32)
    target_X = target_uid[:, None].astype(np.float32)
    calls = []

    def fake(sf, tf, ss, ts, donor_side):
        calls.append((donor_side, sf[:, 0].astype(int), ss[:, 0].astype(int)))
        return object(), np.full(len(ss), 0.4 + 0.1 * donor_side, np.float32), np.full(
            len(ts), 0.6 + 0.1 * donor_side, np.float32)

    ps, pt, audit, _ = exp.domain_crossfit(source_X, source_uid, target_X, target_uid, fake)
    assert np.isfinite(ps).all() and np.isfinite(pt).all()
    assert len(audit) == 2
    for _, donor, recipient in calls:
        assert not set(donor) & set(recipient)


def test_weight_clipping_normalization_and_user_cap():
    raw = np.array([0.25, 4.0, 0.5, 2.0, 4.0, 0.25, 1.0, 1.0])
    cutoff = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    users = np.array([1, 1, 2, 3, 1, 1, 2, 4])
    w, audit, uniq, total = exp.normalize_state_weights(raw, cutoff, users)
    assert abs(float(w.mean()) - 1.0) < 1e-7
    assert audit["max_to_median_user_total_final"] <= 2.0 + 1e-8
    assert len(uniq) == len(total)
    assert audit["ess_fraction"] > 0


def test_shuffled_weight_multiset_is_exact_in_every_stratum():
    real = np.arange(1, 25, dtype=np.float32)
    cutoff = np.repeat(np.arange(2), 12)
    rec = np.tile([1, 20, 100], 8)
    buy = np.tile([0, 5, 20, 0], 6)
    shuffled, permutation, strata, rows = exp.shuffle_weights(real, cutoff, rec, buy)
    assert np.array_equal(np.sort(real), np.sort(shuffled))
    assert np.array_equal(np.sort(permutation), np.arange(len(real)))
    assert all(row["multiset_exact"] for row in rows)
    for key in np.unique(strata):
        m = strata == key
        assert np.array_equal(np.sort(real[m]), np.sort(shuffled[m]))


def test_same_rows_configs_and_calibration_after_final_ensemble_contract():
    if exp.PRED_FILE.exists():
        arms = __import__("json").loads((exp.RESULTS / "arms_manifest.json").read_text("utf-8"))
        assert arms["same_rows_order_matrix_params_seed_rounds_threads"]
        assert arms["weight_multiset_exact"]
    y = np.array([0.0, 1.0, 5.0, 10.0])
    base = np.array([0.2, 0.5, 1.2, 2.0])
    candidate = base + np.array([0.1, -0.1, 0.1, -0.1])
    metric = exp.predictor_metrics(y, candidate)
    assert metric["calibration_after_final_assembly"]
    assert np.isfinite(metric["rmsle_cal"])


def test_no_test_or_public_paths_and_analysis_hash_contract():
    paths = (exp.RUN_DIR, exp.RESULTS, exp.WEIGHT_FILE, exp.PRED_FILE)
    text = " ".join(str(p).lower() for p in paths)
    assert "submission" not in text and "public" not in text and "20260213" not in text
    assert exp.PILOT_FOLD == dt.date(2025, 10, 16)
    if exp.PRED_FILE.exists() and exp.WEIGHT_FILE.exists():
        first = exp.analyze()["analysis_replay_sha256"]
        second = exp.analyze()["analysis_replay_sha256"]
        assert first == second


def test_target_day_overlap_audit():
    cuts = [dt.date(2025, 4, 3) + dt.timedelta(days=7 * k) for k in range(4)]
    audit = exp.target_day_audit(cuts)
    assert audit["adjacent_overlap_fraction_mean"] == 23 / 30
    assert audit["unique_target_days"] == 51
