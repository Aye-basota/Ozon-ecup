"""Блендинг OOF-предсказаний нескольких экспериментов.

Усреднение делается **в лог-пространстве** z, а не в сыром GMV: метрика — MSE по z,
а усреднение сырых предсказаний смещает уровень вверх (неравенство Йенсена).

Веса подбираются на OOF и только на OOF. Одновременно печатается корреляция
остатков — она показывает, есть ли смысл в ансамбле вообще.

Запуск: python -m src.blend --exps S1-E02 S1-E03a
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from src.tracking import load_oof
from src.validation import best_offset, rmsle_z


def shifted_rmsle(ly: np.ndarray, z: np.ndarray) -> float:
    """RMSLE после ОПТИМАЛЬНОГО глобального сдвига, аналитически.

    MSE(delta) = mean((ly - z - delta)^2) минимизируется при delta = mean(ly - z),
    и тогда MSE = Var(ly - z). Сеточный поиск здесь не нужен и в 241 раз дороже —
    это важно при переборе тысяч комбинаций весов.
    """
    r = ly - z
    return float(np.sqrt(r.var()))


def aligned(exps: list[str]):
    """Выравнивает OOF по (cutoff, user_id) — порядок строк обязан совпадать."""
    ds = [load_oof(e) for e in exps]
    keys = [np.char.add(np.asarray(d["cutoff"], dtype="U10"),
                        np.asarray(d["user_id"]).astype("U10")) for d in ds]
    base = keys[0]
    order = np.argsort(base)
    y = ds[0]["y"][order]
    Z = []
    for d, k in zip(ds, keys):
        o = np.argsort(k)
        assert np.array_equal(k[o], base[order]), "OOF на разных наборах строк"
        Z.append(d["z"][o])
    return np.vstack(Z), y, ds[0]["cutoff"][order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()
    Z, y, cut = aligned(a.exps)
    print(f"n = {len(y):,} строк OOF, моделей {len(a.exps)}\n")
    print(f"{'модель':>12} {'RMSLE':>9} {'после сдвига':>13}")
    res = []
    for e, z in zip(a.exps, Z):
        o, sc = best_offset(y, z)
        res.append(sc)
        print(f"{e:>12} {rmsle_z(y, z):>9.5f} {sc:>13.5f}  (сдвиг {o:+.3f})")

    ly = np.log1p(y)
    print("\nкорреляция остатков (в лог-пространстве):")
    R = np.vstack([ly - z for z in Z])
    for i, j in itertools.combinations(range(len(a.exps)), 2):
        print(f"  {a.exps[i]:>12} vs {a.exps[j]:<12} {np.corrcoef(R[i], R[j])[0, 1]:.4f}")

    print("\nперебор весов (после оптимального сдвига у каждой смеси):")
    ws = [w for w in itertools.product(np.arange(0, 1 + 1e-9, a.step), repeat=len(a.exps))
          if abs(sum(w) - 1) < 1e-9]
    best = None
    for w in ws:
        z = np.average(Z, axis=0, weights=w)
        o, sc = best_offset(y, z)
        if best is None or sc < best[1]:
            best = (w, sc, o)
        if len(a.exps) == 2 and abs(w[0] * 10 - round(w[0] * 10)) < 1e-6:
            print(f"  w={np.round(w, 2)}  RMSLE={rmsle_z(y, z):.5f}  после сдвига {sc:.5f}")
    print(f"\nлучшая смесь: w={np.round(best[0], 2)}  RMSLE после сдвига {best[1]:.5f} "
          f"(сдвиг {best[2]:+.3f})")
    print(f"выигрыш над лучшей одиночной моделью: {min(res) - best[1]:+.5f}")

    print("\nпо фолдам (лучшая смесь):")
    zb = np.average(Z, axis=0, weights=best[0])
    for c in np.unique(cut):
        m = cut == c
        o, sc = best_offset(y[m], zb[m])
        singles = " ".join(f"{best_offset(y[m], z[m])[1]:.5f}" for z in Z)
        print(f"  {c}: смесь {sc:.5f}   одиночные {singles}")


if __name__ == "__main__":
    main()
