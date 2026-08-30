"""Fast structural tests for DET-PAIR (the full repeats are experiment artifacts)."""
from __future__ import annotations

import numpy as np

from src.config import SEED
from src.det_pair import (THRESHOLDS, continuation_cfg, learning_rate_plan, pack_plan,
                          sha256_array, unpack_group)


def test_materialized_plan_roundtrip_preserves_chunk_and_batch_order():
    groups = [
        [(1, np.array([7, 3, 5])), (0, np.array([9, 2]))],
        [(2, np.array([4]))],
    ]
    plan = pack_plan(groups)
    for i, expected in enumerate(groups):
        actual = unpack_group(plan, i)
        assert [k for k, _ in actual] == [k for k, _ in expected]
        for (_, a), (_, b) in zip(actual, expected, strict=True):
            assert np.array_equal(a, b)


def test_continuation_only_changes_execution_count_and_policy():
    source = dict(seed=SEED, epochs=4, workers=3, compile=True, lr=3e-3,
                  warmup=300, wd=1e-2, depth_grid=[90, 120], depth_aug=0.5)
    cfg = continuation_cfg(source)
    assert cfg["seed"] == source["seed"] == SEED
    assert cfg["lr"] == source["lr"] and cfg["wd"] == source["wd"]
    assert cfg["depth_aug"] == source["depth_aug"]
    assert cfg["epochs"] == 1 and cfg["workers"] == 1 and cfg["compile"] is False


def test_materialized_lr_plan_is_the_production_formula():
    cfg = dict(lr=3e-3, warmup=300)
    lr = learning_rate_plan(cfg, 1000)
    assert lr.dtype == np.float64 and len(lr) == 1000
    assert lr[0] == cfg["lr"] / cfg["warmup"]
    assert lr[299] < cfg["lr"] and lr[300] < lr[299]
    assert lr[-1] < lr[300]


def test_prediction_hash_uses_exact_value_bytes():
    a = np.array([1.0, 2.0], np.float32)
    b = a.copy()
    assert sha256_array(a) == sha256_array(b)
    b[1] = np.nextafter(b[1], np.float32(np.inf))
    assert sha256_array(a) != sha256_array(b)


def test_requested_tolerances_are_encoded_exactly():
    assert THRESHOLDS == dict(abs_delta_rmsle=1e-4, var_delta_z=1e-5,
                              max_abs_delta_z=1e-3, corr=0.999999)
