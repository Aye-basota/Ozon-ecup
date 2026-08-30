"""Fast invariant and signed-direction tests for CALENDAR-PLACEBO-01."""
from __future__ import annotations

import datetime as dt

import numpy as np

from src.calendar_placebo import (HISTORY_SUPPORT, PLACEBO_PAIRS, PANEL_BLOCKS,
                                  signed_standardized_shift, strict_support_eligible,
                                  validate_feature_contract, validate_task)
from src.config import CUTOFF_TEST, DATA_START
from src.domain01 import user_group_fold


def test_all_placebos_have_equal_full_history_support_and_no_future_cutoff():
    for _, a, b, _ in PLACEBO_PAIRS:
        validate_task(a, b)
        assert strict_support_eligible(a)
        assert strict_support_eligible(b)
        assert a >= DATA_START + dt.timedelta(days=HISTORY_SUPPORT)
        assert b <= CUTOFF_TEST


def test_non_fixed_support_or_panel_is_rejected():
    a, b = dt.date(2025, 7, 3), dt.date(2025, 7, 10)
    for history, blocks in [(90, PANEL_BLOCKS), (HISTORY_SUPPORT, 1)]:
        try:
            validate_task(a, b, history, blocks)
        except ValueError as exc:
            assert "fixed-L180" in str(exc)
        else:
            raise AssertionError("non-comparable history/panel support was accepted")


def test_exact_yoy_cutoff_is_not_eligible_for_l180():
    assert not strict_support_eligible(dt.date(2025, 2, 13))


def test_group_split_keeps_every_user_in_one_fold():
    users = np.array([7, 8, 7, 9, 8, 10, 7])
    folds = user_group_fold(users, 5)
    for user in np.unique(users):
        assert len(np.unique(folds[users == user])) == 1


def test_source_date_and_user_identifiers_are_forbidden():
    validate_feature_contract(["w30_gmv", "rec_buy", "trend_gmv_30_60"])
    for feature in ["user_id", "cutoff_month", "source_marker", "event_date", "fold_id"]:
        try:
            validate_feature_contract(["w30_gmv", feature])
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError(f"identifier {feature} was accepted")


def test_signed_shift_uses_b_minus_a_direction():
    a = np.array([0.0, 1.0, 2.0, np.nan])
    b = np.array([2.0, 3.0, 4.0, np.nan])
    forward = signed_standardized_shift(a, b)
    backward = signed_standardized_shift(b, a)
    assert forward["smd"] > 0
    assert forward["median_shift"] > 0
    assert np.isclose(forward["smd"], -backward["smd"])
    assert np.isclose(forward["median_shift"], -backward["median_shift"])
