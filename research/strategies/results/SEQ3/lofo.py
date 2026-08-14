"""Этап 4 — нужны ли старые GBDT-компоненты после появления SEQ.

Вопрос не «подобрать красивую сотую веса», а структурный: `SEQ-AVG3` в одиночку
вышел на паритет со всей боевой смесью (`exp_026`), и три из четырёх табличных
членов смеси построены на ОДНИХ И ТЕХ ЖЕ 227 признаках, отличаясь только головой.
`exp_018` уже показал, что расстояние между ними — 1.5-1.8 пола сидов, то есть
половина «разнообразия» смеси это переобучение сида, а не разные функции.

Семейства сравниваются честно: веса подбираются БЕЗ отложенного фолда и
проверяются на нём (LOFO). Опора — уже отправленный `SEQ-01-MIX` (LB 1.6501764),
а не прежняя база `S1-DIST-MIX`: мерить прирост от того, что давно перекрыто,
значит завышать его.

`S1-CAP` (=`S1-E03a`) не обнуляется никогда, минимум 0.10: `MIX-E11` стоил
+0.00023 LB при локальном выигрыше −0.00038 ровно из-за обнуления этой страховки,
и её ценность лежит в неизмеримой оси (`exp_016` §5).

Запуск: PYTHONIOENCODING=utf-8 PYTHONPATH=. python research/strategies/results/SEQ3/lofo.py
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

from src.blend import aligned, shifted_rmsle
from src.config import FOLD_WEIGHTS_S1, VAL_FOLDS_S1

OUT = Path(__file__).parent
FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]
W = np.asarray(FOLD_WEIGHTS_S1, float)
W = W / W.sum()
CAP_MIN = 0.10
STEP = 0.05

# опора: то, что реально стоит на LB (SEQ-01-MIX, 1.6501764)
REF = {"S1-E10": 0.15, "S1-E02": 0.20, "S1-E03a": 0.10, "S1-DIST": 0.25, "SEQ-01-S42": 0.30}

FAMILIES = {
    "полная смесь (E10+E02+CAP+DIST+SEQ)": ["S1-E10", "S1-E02", "S1-E03a", "S1-DIST", "SEQ-AVG3"],
    "CAP+DIST+SEQ": ["S1-E03a", "S1-DIST", "SEQ-AVG3"],
    "CAP+best direct(ROUNDS)+DIST+SEQ": ["S1-E03a", "S1-ROUNDS", "S1-DIST", "SEQ-AVG3"],
    "CAP+E02+DIST+SEQ": ["S1-E03a", "S1-E02", "S1-DIST", "SEQ-AVG3"],
    "CAP+SEQ": ["S1-E03a", "SEQ-AVG3"],
    "контроль: без SEQ (прежняя база)": ["S1-E10", "S1-E02", "S1-E03a", "S1-DIST"],
}


def grid(names: list[str], step: float = STEP):
    """Все веса на сетке, сумма 1, доля `S1-E03a` не ниже CAP_MIN."""
    i_cap = names.index("S1-E03a")
    g = np.arange(0, 1 + 1e-9, step)
    out = []
    for w in itertools.product(g, repeat=len(names)):
        if abs(sum(w) - 1) > 1e-9 or w[i_cap] < CAP_MIN - 1e-9:
            continue
        out.append(tuple(w))
    return out


def fold_scores(Z: np.ndarray, ly: np.ndarray, masks, ws) -> np.ndarray:
    out = np.empty((len(ws), len(masks)))
    for i, w in enumerate(ws):
        z = np.average(Z, axis=0, weights=w)
        for j, m in enumerate(masks):
            out[i, j] = shifted_rmsle(ly[m], z[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=STEP)
    a = ap.parse_args()

    names = sorted({n for v in FAMILIES.values() for n in v} | set(REF))
    Z_all, y, cut = aligned(names)
    ly = np.log1p(y)
    masks = [cut == c for c in FOLDS]
    idx = {n: i for i, n in enumerate(names)}
    print(f"строк OOF {len(y):,}, моделей {len(names)}: {', '.join(names)}\n")

    z_ref = np.average(np.vstack([Z_all[idx[n]] for n in REF]), axis=0,
                       weights=list(REF.values()))
    ref_fc = np.array([shifted_rmsle(ly[m], z_ref[m]) for m in masks])
    print(f"опора SEQ-01-MIX (LB 1.6501764): wCV {float(W @ ref_fc):.5f}   "
          + " ".join(f"{v:.5f}" for v in ref_fc))

    rows, wrows = [], []
    for fam, ns in FAMILIES.items():
        Z = np.vstack([Z_all[idx[n]] for n in ns])
        ws = grid(ns, a.step)
        FC = fold_scores(Z, ly, masks, ws)
        sc = FC @ W
        b_in = int(np.argmin(sc))

        held, picks = np.zeros(len(FOLDS)), []
        for h in range(len(FOLDS)):
            keep = [i for i in range(len(FOLDS)) if i != h]
            wh = W[keep] / W[keep].sum()
            b = int(np.argmin(FC[:, keep] @ wh))
            held[h] = FC[b, h]
            picks.append(ws[b])
        honest = float(W @ (held - ref_fc))
        insample = float(sc[b_in] - W @ ref_fc)
        better = int((held < ref_fc).sum())
        plateau = int((sc <= sc[b_in] + 5e-5).sum())
        # устойчивость: разброс каждого веса между четырьмя LOFO-подборами
        P = np.asarray(picks)
        rows.append(dict(family=fam, n_models=len(ns), honest=honest, insample=insample,
                         folds_better=better, plateau=plateau, n_grid=len(ws),
                         wcv_honest=float(W @ held), wcv_insample=float(sc[b_in]),
                         w_seq_mean=float(P[:, ns.index("SEQ-AVG3")].mean())
                         if "SEQ-AVG3" in ns else 0.0,
                         w_spread=float(np.abs(P.max(0) - P.min(0)).max()),
                         **{f"f{i}": float(held[i] - ref_fc[i]) for i in range(len(FOLDS))}))
        for h, p in enumerate(picks):
            wrows.append(dict(family=fam, held_out=FOLDS[h],
                              **{n: float(v) for n, v in zip(ns, p)}))
        print(f"\n=== {fam}")
        print(f"  в выборке: w={np.round(ws[b_in], 2)}  wCV {sc[b_in]:.5f} "
              f"({insample:+.5f} к опоре), плато {plateau}/{len(ws)}")
        print(f"  {'отложенный фолд':<14}{'веса без него':<32}{'на нём':>10}{'опора':>10}{'дельта':>10}")
        for h in range(len(FOLDS)):
            print(f"  {FOLDS[h]:<14}{str(np.round(picks[h], 2)):<32}{held[h]:>10.5f}"
                  f"{ref_fc[h]:>10.5f}{held[h] - ref_fc[h]:>+10.5f}")
        print(f"  ЧЕСТНЫЙ выигрыш {honest:+.5f} ({better}/4 фолда), "
              f"максимальный разброс одного веса между LOFO-подборами {rows[-1]['w_spread']:.2f}")

    print(f"\n{'семейство':<38}{'честно':>10}{'в выборке':>11}{'фолдов':>8}"
          f"{'доля SEQ':>10}{'разброс w':>11}")
    for r in sorted(rows, key=lambda r: r["honest"]):
        print(f"{r['family']:<38}{r['honest']:>+10.5f}{r['insample']:>+11.5f}"
              f"{r['folds_better']:>6}/4{r['w_seq_mean']:>10.2f}{r['w_spread']:>11.2f}")
    pl.DataFrame(rows).write_csv(OUT / "lofo_families.csv")
    pl.DataFrame(wrows).write_csv(OUT / "lofo_weights.csv")
    print(f"\nзаписано: {OUT}/lofo_families.csv, lofo_weights.csv")


if __name__ == "__main__":
    main()
