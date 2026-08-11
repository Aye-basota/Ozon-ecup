"""Проверки обвязки обучения: сборка матрицы и диспетчеризация постановок.

`assemble` собирает матрицу на 5+ млн строк и является общим кодом всех
экспериментов — молчаливая ошибка здесь испортила бы любой будущий результат.

Запуск: python -m src.test_pipeline
"""
from __future__ import annotations

import numpy as np

from src.config import cutoff_grid
from src.features import feature_names, make_xy, to_np
from src.train import Setup, assemble, infer

RNG = np.random.default_rng(0)


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}", flush=True)
    assert cond, msg


def test_assemble_matches_per_cutoff_reference():
    """Сборка обязана быть побитово равна вертикальной склейке по cutoff'ам."""
    cuts = cutoff_grid(90, 7)[-2:]
    s = Setup(L=0, norm_long=True, train_blocks=1)
    X0, _ = make_xy(cuts[0], s.L, 1, norm_long=True)
    feats = feature_names(X0)
    ref_X, ref_y = [], []
    for T in cuts:
        X, y = make_xy(T, s.L, 1, norm_long=True)
        ref_X.append(to_np(X, feats))
        ref_y.append(y)
    ref_X = np.vstack(ref_X)
    ref_y = np.concatenate(ref_y)

    A, y, w = assemble(cuts, s, feats)
    ok(A.shape == ref_X.shape, f"форма {A.shape} == {ref_X.shape}")
    ok(A.dtype == np.float32, f"dtype {A.dtype} == float32")
    ok(np.array_equal(A, ref_X, equal_nan=True), "матрица побитово совпадает со склейкой")
    ok(np.array_equal(y, ref_y), "таргет совпадает")
    ok(w.shape == (A.shape[0],) and np.all(w == 1.0), "веса по умолчанию единичные")


def test_assemble_applies_exponential_cutoff_weights():
    cuts = cutoff_grid(90, 7)[-2:]
    s = Setup(L=0, norm_long=True, train_blocks=1, weight_tau=30.0)
    X0, _ = make_xy(cuts[0], s.L, 1, norm_long=True)
    A, y, w = assemble(cuts, s, feature_names(X0), V=cuts[-1])
    ok(w.min() < w.max(), "разные cutoff'ы получили разные веса")
    ok(abs(float(w.max()) - 1.0) < 1e-6, "ближайший к val cutoff весит 1.0")


def test_infer_dispatches_dist_to_distribution_head():
    """infer() обязан знать про постановку dist, иначе фолд посчитается молча неверно."""
    from src.models import train_dist

    n, d = 5_000, 4
    X = RNG.normal(size=(n, d)).astype(np.float32)
    y = np.expm1(np.maximum(X[:, 0] + RNG.normal(scale=0.5, size=n), 0.0))
    m = train_dist(X, y, params={"num_threads": 4, "verbose": -1}, rounds=20)
    s = Setup(model="dist")
    from src.models import predict_dist
    ok(np.allclose(infer(s, m, X), predict_dist(m, X)), "infer(dist) == predict_dist")


def test_infer_supports_iteration_snapshot():
    from src.models import train_dist

    n, d = 5_000, 4
    X = RNG.normal(size=(n, d)).astype(np.float32)
    y = np.expm1(np.maximum(X[:, 0] + RNG.normal(scale=0.5, size=n), 0.0))
    m = train_dist(X, y, params={"num_threads": 4, "verbose": -1}, rounds=20)
    s = Setup(model="dist")
    z5, z20 = infer(s, m, X, num_iteration=5), infer(s, m, X)
    ok(not np.array_equal(z5, z20), "срез на 5 раундах отличается от полного")


def _fold_arrays(cutoff: str, n: int, seed: int):
    r = np.random.default_rng(seed)
    y = np.expm1(np.maximum(r.normal(2.0, 1.0, n), 0.0))
    z = np.log1p(y) + r.normal(0, 0.5, n)
    return np.arange(n) + seed * 10_000, np.full(n, cutoff, dtype="U10"), z, y


def test_merge_combines_folds_and_averages_their_scores():
    from src.merge_oof import merge_arrays
    from src.validation import rmsle_z

    a_u, a_c, a_z, a_y = _fold_arrays("2025-09-04", 1000, 1)
    b_u, b_c, b_z, b_y = _fold_arrays("2025-10-16", 1500, 2)
    m = merge_arrays(np.concatenate([a_u, b_u]), np.concatenate([a_c, b_c]),
                     np.concatenate([a_z, b_z]), np.concatenate([a_y, b_y]))
    ok(m["n"] == 2500, f"строк {m['n']}, ожидается 2500")
    ok(m["folds"] == ["2025-09-04", "2025-10-16"], f"фолды {m['folds']}")
    want = [rmsle_z(a_y, a_z), rmsle_z(b_y, b_z)]
    ok(np.allclose(m["fold_scores"], want), "пофолдовые RMSLE считаются независимо")
    ok(abs(m["cv_mean"] - float(np.mean(want))) < 1e-12,
       "CV = невзвешенное среднее по фолдам, а не по строкам")


def test_merge_rejects_the_same_fold_twice():
    """Двойной учёт фолда не упал бы сам собой — он просто дал бы красивое число."""
    from src.merge_oof import merge_arrays

    u, c, z, y = _fold_arrays("2025-09-04", 100, 1)
    try:
        merge_arrays(np.concatenate([u, u]), np.concatenate([c, c]),
                     np.concatenate([z, z]), np.concatenate([y, y]))
    except AssertionError:
        ok(True, "повтор пары (cutoff, user_id) отвергнут")
        return
    ok(False, "повтор пары (cutoff, user_id) прошёл незамеченным")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"== обвязка обучения: {len(tests)} проверок ==")
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print("\nвсе проверки пройдены")
