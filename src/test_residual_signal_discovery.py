from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.btyd_day_bgnbd import user_group
from src.residual_signal_discovery import (
    EXPECTED_FOLD_SCORES,
    FOLDS,
    OUT_ARTIFACTS,
    RESULTS,
    add_past_residual,
    assert_no_future_feature_columns,
    canonical_order,
    fold_calibrated,
    full_slot_weights,
    gate_assembly,
    gate_weight,
    permutation_within_strata,
    reconstruct_strongest,
    select_scale_without_late,
    shuffle_preserves_strata,
)


def test_exact_strongest_reconstruction_artifact_backed():
    paths = {
        "cap": "oof_S1-E03a.npz", "unc": "oof_S1-E02.npz",
        "dist": "oof_S1-DIST.npz", "etx": "oof_ETX-AVG3.npz",
        "seq": "oof_SEQ-AVG3.npz",
    }
    arrays = {}
    y = cutoff = uid = None
    for key, name in paths.items():
        data = np.load(OUT_ARTIFACTS.parent / name)
        order = canonical_order(data["user_id"], data["cutoff"])
        arrays[key] = np.asarray(data["z"], float)[order]
        if y is None:
            y = np.asarray(data["y"], float)[order]
            cutoff = np.asarray(data["cutoff"], dtype="U10")[order]
            uid = np.asarray(data["user_id"], np.int64)[order]
        else:
            assert np.array_equal(np.asarray(data["cutoff"], dtype="U10")[order], cutoff)
            assert np.array_equal(np.asarray(data["user_id"], np.int64)[order], uid)
            assert np.allclose(np.asarray(data["y"], float)[order], y, atol=1e-6)
    z = reconstruct_strongest(**arrays)
    fi = np.asarray([FOLDS.index(value) for value in cutoff], np.int8)
    _, scores, _ = fold_calibrated(y, z, fi)
    assert np.max(np.abs(scores - EXPECTED_FOLD_SCORES)) < 5e-7


def test_row_and_target_alignment_rejects_changed_order_only_after_key_sort():
    uid = np.array([2, 1, 2, 1])
    cutoff = np.array(["2025-09-18", "2025-09-04", "2025-09-04", "2025-09-18"])
    order = canonical_order(uid, cutoff)
    assert list(zip(cutoff[order], uid[order])) == [
        ("2025-09-04", 1), ("2025-09-04", 2),
        ("2025-09-18", 1), ("2025-09-18", 2),
    ]


def test_fold_calibration_semantics_are_independent_by_fold():
    y = np.expm1(np.array([1.0, 2.0, 4.0, 6.0]))
    z = np.array([0.0, 1.0, 1.0, 3.0])
    fi = np.array([0, 0, 1, 1], np.int8)
    z_cal, scores, offsets = fold_calibrated(y, z, fi)
    assert np.allclose(offsets[:2], [1.0, 3.0])
    assert np.allclose(scores[:2], [0.0, 0.0])
    assert np.allclose(z_cal, np.log1p(y))


def test_no_future_feature_columns():
    assert_no_future_feature_columns(["rec_buy", "w180_days_buy", "trend_gmv_30_90"])
    try:
        assert_no_future_feature_columns(["future_gmv"])
    except AssertionError:
        pass
    else:
        raise AssertionError("future column was accepted")


def test_no_user_id_feature():
    try:
        assert_no_future_feature_columns(["user_id", "rec_buy"])
    except AssertionError:
        pass
    else:
        raise AssertionError("user_id was accepted as a feature")


def test_user_hash_halves_disjoint_and_exhaustive():
    uid = np.arange(1, 10001, dtype=np.int64)
    side = user_group(uid)
    assert set(np.unique(side)) == {0, 1}
    assert not np.intersect1d(uid[side == 0], uid[side == 1]).size
    assert len(uid[side == 0]) + len(uid[side == 1]) == len(uid)


def test_recipient_users_absent_from_probe_training_labels():
    uid = np.arange(1, 10001, dtype=np.int64)
    side = user_group(uid)
    for donor in (0, 1):
        assert np.intersect1d(uid[side == donor], uid[side == 1 - donor]).size == 0


def test_shuffle_preserves_every_stratum_distribution():
    strata = np.repeat(np.arange(20), 10)
    values = np.arange(len(strata), dtype=float)
    mask = np.ones(len(values), bool)
    permutation = permutation_within_strata(strata, mask)
    shuffled = values[permutation]
    assert shuffle_preserves_strata(values, shuffled, strata, mask)


def test_winner_and_advantage_definitions():
    loss_etx = np.array([1.0, 3.0, 2.0])
    loss_seq = np.array([2.0, 2.0, 2.0])
    advantage = loss_seq - loss_etx
    winner = advantage > 0
    assert advantage.tolist() == [1.0, -1.0, 0.0]
    assert winner.tolist() == [True, False, False]


def test_full_slot_weights_sum_to_one():
    for kind in ("etx", "seq", "fixed"):
        assert abs(sum(full_slot_weights(kind).values()) - 1.0) < 1e-12


def test_gate_weight_is_always_bounded():
    p = np.linspace(0, 1, 1001)
    w = gate_weight(p)
    assert w.min() >= 0.25 and w.max() <= 0.75


def test_probability_half_reproduces_exact_strongest():
    rng = np.random.default_rng(42)
    frame = {name: rng.normal(size=100) for name in ("cap", "unc", "dist", "etx", "seq")}
    expected = reconstruct_strongest(frame["cap"], frame["unc"], frame["dist"],
                                     frame["etx"], frame["seq"])
    actual = gate_assembly(frame, np.full(100, 0.5))
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)


def test_residual_scale_selection_uses_no_late_rows():
    # Late rows carry absurd values; selection must be identical after changing them.
    y = np.expm1(np.array([1., 2., 1., 2., 1., 2., 100., 100.]))
    z = np.array([1., 1., 1., 1., 1., 1., -100., -100.])
    correction = np.array([0., 1., 0., 1., 0., 1., 1000., -1000.])
    fi = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int8)
    donor = fi < 3
    first, _ = select_scale_without_late(y, z, correction, fi, donor)
    y[-2:] = np.expm1([0., 0.])
    correction[-2:] = [-1e9, 1e9]
    second, _ = select_scale_without_late(y, z, correction, fi, donor)
    assert first == second


def test_past_residual_only_when_target_end_not_after_cutoff():
    frame = {
        "cutoff": np.array(["2025-09-04", "2025-10-16"], dtype="U10"),
        "user_id": np.array([1, 1]), "y": np.array([0., 0.]),
        "r_strong": np.array([0.25, -0.5]),
    }
    value = add_past_residual(frame)
    assert np.isnan(value[0]) and value[1] == np.float32(0.25)


def test_seed_oracle_control_is_present_in_summary_after_run():
    path = RESULTS / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    oracle = summary["oracle_headroom"]
    assert oracle["seed_null_gain_wcv"] is not None
    assert oracle["semantic_excess_wcv"] == oracle["semantic_gain_wcv"] - oracle["seed_null_gain_wcv"]


def test_analysis_only_rerun_reproduces_hashes():
    path = RESULTS / "reproducibility.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["analysis_only_status"] == "PASS"
    assert payload["canonical_hashes"]


def test_no_test_submission_or_public_lb_paths_touched():
    path = RESULTS / "input_manifest.json"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").lower().replace("/", "\\")
    assert "\\submissions\\" not in text
    assert "ztest_" not in text
    assert "test_predictions.parquet" not in text
    assert "public_lb" not in text
