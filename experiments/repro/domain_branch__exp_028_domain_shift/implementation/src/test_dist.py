"""Проверки головы распределения (постановка `dist`, эксперимент E0).

Голова считает E[z|x] как среднее дискретного распределения по бинам z.
Здесь проверяется ровно та логика, где легко ошибиться молча: раскладка по бинам,
центроиды и то, что ожидание не смещает уровень прогноза.

Запуск: python -m src.test_dist
"""
from __future__ import annotations

import numpy as np

from src.models import (bin_centroids, bin_labels, dist_expectation, dist_targets,
                        make_datasets, predict_dist, train_dist_ds, z_bins)

RNG = np.random.default_rng(0)


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}", flush=True)
    assert cond, msg


def sample_z(n=50_000, p_zero=0.4):
    """Таргет с точечной массой в нуле и тяжёлым правым хвостом — как в задаче."""
    z = RNG.lognormal(mean=1.0, sigma=1.0, size=n)
    z[RNG.random(n) < p_zero] = 0.0
    return z


def test_edges_count_is_one_less_than_bins():
    ok(len(z_bins(sample_z(), 16)) == 15, "z_bins(z, 16) даёт 15 границ -> 16 бинов")


def test_zero_target_goes_to_bin_zero():
    z = sample_z()
    lab = bin_labels(z, z_bins(z, 16))
    ok(set(np.unique(lab[z == 0]).tolist()) == {0}, "все нули попадают в бин 0")


def test_positive_target_never_lands_in_bin_zero():
    z = sample_z()
    lab = bin_labels(z, z_bins(z, 16))
    ok((lab[z > 0] >= 1).all(), "ни один положительный z не попал в бин 0")


def test_all_bins_are_used():
    z = sample_z()
    lab = bin_labels(z, z_bins(z, 16))
    ok(int(lab.max()) + 1 == 16, f"использованы все 16 бинов (получено {int(lab.max()) + 1})")


def test_centroid_of_zero_bin_is_zero():
    z = sample_z()
    lab, cent = dist_targets(np.expm1(z))
    ok(cent[0] == 0.0, f"центроид бина 0 равен нулю (получено {cent[0]})")


def test_centroid_is_mean_of_train_values_in_bin():
    z = sample_z()
    lab = bin_labels(z, z_bins(z, 16))
    cent = bin_centroids(z, lab, 16)
    k = 7
    ok(abs(cent[k] - z[lab == k].mean()) < 1e-9, f"центроид бина {k} = среднему z внутри бина")


def test_empty_bin_centroid_is_finite():
    """Совпадающие квантили дают пустой бин; NaN оттуда отравил бы весь прогноз."""
    z = np.concatenate([np.zeros(100), np.full(100, 2.0), np.full(10, 9.0)])
    lab = bin_labels(z, z_bins(z, 16))
    cent = bin_centroids(z, lab, 16)
    ok(np.isfinite(cent).all(), f"пустые бины не дают NaN (центроиды {np.round(cent, 3)})")


def test_expectation_of_one_hot_is_centroid():
    cent = np.array([0.0, 1.0, 2.5, 7.0])
    P = np.eye(4)
    ok(np.allclose(dist_expectation(P, cent), cent), "на вырожденном распределении ожидание = центроид")


def test_expectation_is_mean_not_mode():
    """Ровно то, ради чего вся постановка: E[z|x], а не самый вероятный бин."""
    cent = np.array([0.0, 1.0, 10.0])
    P = np.array([[0.6, 0.2, 0.2]])              # мода — бин 0, среднее — 2.2
    ok(abs(float(dist_expectation(P, cent)[0]) - 2.2) < 1e-12,
       "ожидание считается как сумма p_k*m_k, а не как argmax")


def test_expectation_preserves_level():
    """Идеально предсказанное распределение обязано воспроизводить mean(z) точно.

    Иначе голова вносит систематический сдвиг уровня, а уровень в этой задаче
    закрыт замером на LB (STATE.md) и портить его нельзя.
    """
    z = sample_z()
    lab, cent = dist_targets(np.expm1(z))
    P = np.eye(len(cent))[lab]                   # «оракульное» распределение
    ok(abs(float(dist_expectation(P, cent).mean()) - float(z.mean())) < 1e-9,
       "mean(E[z|x]) == mean(z) при точном распределении")


def test_train_and_predict_roundtrip():
    """Сквозная проверка обвязки: make_datasets -> train_dist_ds -> predict_dist."""
    n, d = 20_000, 5
    X = RNG.normal(size=(n, d)).astype(np.float32)
    y = np.expm1(np.maximum(X[:, 0] + RNG.normal(scale=0.5, size=n), 0.0))
    dss = make_datasets("dist", X, y, params={"num_threads": 4, "verbose": -1})
    m = train_dist_ds(dss, params={"num_threads": 4, "verbose": -1}, rounds=30)
    z = predict_dist(m, X)
    ok(z.shape == (n,), f"форма прогноза {z.shape}, ожидается ({n},)")
    ok(np.isfinite(z).all(), "в прогнозе нет NaN/inf")
    ok((z >= 0).all(), "прогноз неотрицателен (центроиды z неотрицательны)")
    err = abs(float(z.mean()) - float(np.log1p(y).mean()))
    ok(err < 0.02, f"уровень прогноза воспроизводится на train: |d mean| = {err:.4f}")
    z_half = predict_dist(m, X, num_iteration=10)
    ok(z_half.shape == (n,) and not np.array_equal(z_half, z),
       "num_iteration даёт срез по числу раундов")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"== голова распределения: {len(tests)} проверок ==")
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print("\nвсе проверки пройдены")
