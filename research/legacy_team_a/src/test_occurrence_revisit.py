import numpy as np

from src.occurrence_revisit import GRID, calibrated_score, nested_lofo


def test_calibrated_score_ignores_constant_level_shift():
    y = np.asarray([0.0, 1.0, 5.0, 20.0])
    z = np.asarray([0.2, 0.5, 1.5, 2.5])
    assert abs(calibrated_score(y, z) - calibrated_score(y, z + 7.0)) < 1e-12


def test_nested_lofo_never_uses_held_column_for_selection():
    curve = np.ones((len(GRID), 4))
    curve[2, :] = 0.5
    idx1, held1 = nested_lofo(curve, np.asarray([1, 2, 4, 8], float))
    changed = curve.copy()
    changed[:, 0] = np.arange(len(GRID)) * 100.0
    idx2, held2 = nested_lofo(changed, np.asarray([1, 2, 4, 8], float))
    assert idx1[0] == idx2[0] == 2
    assert held1[0] != held2[0]


def test_nested_lofo_tie_chooses_smallest_alpha():
    curve = np.ones((len(GRID), 4))
    idx, _ = nested_lofo(curve, np.asarray([1, 2, 4, 8], float))
    assert np.array_equal(idx, np.zeros(4, dtype=int))
