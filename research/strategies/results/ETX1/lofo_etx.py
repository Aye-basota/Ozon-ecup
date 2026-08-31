"""ETX1 — честный LOFO слота SEQ с событийным трансформером среди кандидатов.

Протокол ровно тот же, что в `research/strategies/results/MIX9/lofo_mix.py`
(`exp_035`), и это принципиально: опора, сетка весов, правило LOFO и опорный
`SEQ-01-MIX` совпадают дословно, поэтому число ETX сравнивается с уже принятыми
−0.00061 / −0.00055, а не живёт в своей системе координат.

Кандидаты на слот SEQ:

  * `SEQ-AVG3`          — ВЫБРАННЫЙ член (`exp_035`), опора решения;
  * `SEQ-D3A-AVG3`      — проигравший ему кандидат, для калибровки шкалы;
  * `ETX-01-S42`        — событийный трансформер, ОДНА модель против трёх;
  * `ETX+SEQ` при α ∈ {0.25, 0.5, 0.75} — не замена, а СОСУЩЕСТВОВАНИЕ.
    Это главный вопрос `exp_036`: соло ETX выигрывает −0.0033, но ближе к
    табличной части, чем TCN, и в замене даёт ноль. Если ценность есть, она
    обязана проявиться именно здесь.

Запуск: PYTHONPATH=. python research/strategies/results/ETX1/lofo_etx.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from src.blend import aligned, fold_masks, shifted_rmsle
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1

CAP, E02, DIST = "S1-E03a", "S1-E02", "S1-DIST"
REF_EXPS = ["S1-E10", E02, CAP, DIST, "SEQ-01-S42"]      # отправленный SEQ-01-MIX
REF_W = [0.15, 0.20, 0.10, 0.25, 0.30]
FIXED_W = [0.10, 0.20, 0.25, 0.45]                       # CAP, E02, DIST, слот SEQ
SEQ_SHARES = [0.40, 0.45, 0.50]
STEP = 0.05
ALPHAS = [0.25, 0.5, 0.75]
ETX, TCN = "ETX-01-S42", "SEQ-AVG3"


def narrow_grid():
    out = []
    for s in SEQ_SHARES:
        rest = round(1.0 - 0.10 - s, 10)
        n = int(round(rest / STEP))
        for i in range(n + 1):
            e = round(i * STEP, 10)
            out.append((0.10, e, round(rest - e, 10), s))
    return out


def fold_cal(Z, ly, masks, w):
    z = np.average(Z, axis=0, weights=np.asarray(w, float))
    return np.array([shifted_rmsle(ly[m], z[m]) for m in masks])


def main() -> None:
    base = sorted({*REF_EXPS, CAP, E02, DIST, ETX, TCN, "SEQ-D3A-AVG3"})
    Z, y, cut = aligned(base)
    idx = {e: i for i, e in enumerate(base)}
    ly = np.log1p(y)
    folds, masks = fold_masks(cut)
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()
    print(f"n = {len(y):,} строк OOF, фолды {folds}\n")

    ref_fc = fold_cal(Z[[idx[e] for e in REF_EXPS]], ly, masks, REF_W)
    ref_wcv = float(w_f @ ref_fc)
    print(f"ОПОРА  SEQ-01-MIX {REF_W}  wCV={ref_wcv:.5f}   "
          + " ".join(f"{v:.5f}" for v in ref_fc))

    # члены слота: три «чистых» и три смешанных
    members = {m: Z[idx[m]] for m in (TCN, "SEQ-D3A-AVG3", ETX)}
    for a in ALPHAS:
        members[f"{a:g}*ETX+{1 - a:g}*{TCN}"] = a * Z[idx[ETX]] + (1 - a) * Z[idx[TCN]]

    print("\nсоло члена (калиброванный RMSLE пофолдово):")
    for m, zm in members.items():
        fc = fold_cal(zm[None, :], ly, masks, [1.0])
        print(f"  {m:>24} wCV={float(w_f @ fc):.5f}   " + " ".join(f"{v:.5f}" for v in fc))

    z_tab = np.average(Z[[idx[CAP], idx[E02], idx[DIST]]], axis=0,
                       weights=[0.10, 0.20, 0.25])
    print("\nразнообразие против табличной части (CAP+E02+DIST, веса 0.10/0.20/0.25):")
    for m, zm in members.items():
        print(f"  {m:>24} Var(z - z_tab)={np.var(zm - z_tab):.5f}  "
              f"corr остатков={np.corrcoef(ly - zm, ly - z_tab)[0, 1]:.5f}")

    grid = narrow_grid()
    rows = []
    for m, zm in members.items():
        sub = np.vstack([Z[idx[CAP]], Z[idx[E02]], Z[idx[DIST]], zm])
        FC = np.vstack([fold_cal(sub, ly, masks, w) for w in grid])
        sc = FC @ w_f
        fix_fc = fold_cal(sub, ly, masks, FIXED_W)
        d_fix = fix_fc - ref_fc

        held, chosen = np.zeros(len(folds)), []
        for h in range(len(folds)):
            keep = [i for i in range(len(folds)) if i != h]
            wh = w_f[keep] / w_f[keep].sum()
            b = int(np.argmin(FC[:, keep] @ wh))
            held[h] = FC[b, h]
            chosen.append(grid[b])
        d_lofo = held - ref_fc
        best = int(np.argmin(sc))

        print(f"\n===== слот SEQ: {m} =====")
        print(f"  фиксированные {FIXED_W}: wCV={float(w_f @ fix_fc):.5f}  "
              f"дельта к опоре {float(w_f @ fix_fc) - ref_wcv:+.5f}  "
              f"фолдов лучше {int((d_fix < 0).sum())}/4  (10-16: {d_fix[3]:+.5f})")
        print("    пофолдово " + " ".join(f"{v:.5f}" for v in fix_fc)
              + "   дельты " + " ".join(f"{v:+.5f}" for v in d_fix))
        for h, f in enumerate(folds):
            print(f"  {f:<12}{str(list(chosen[h])):<26}{held[h]:>10.5f}"
                  f"{ref_fc[h]:>10.5f}{d_lofo[h]:>+10.5f}")
        print(f"  ЧЕСТНЫЙ LOFO по wCV: {float(w_f @ d_lofo):+.5f}"
              f"   фолдов лучше {int((d_lofo < 0).sum())}/4"
              f"   (в выборке оптимум {list(grid[best])}: {sc[best] - ref_wcv:+.5f})")

        rows.append(dict(member=m, ref_wcv=ref_wcv,
                         fixed_wcv=float(w_f @ fix_fc),
                         fixed_delta=float(w_f @ fix_fc) - ref_wcv,
                         fixed_folds=int((d_fix < 0).sum()),
                         fixed_fold_cal=fix_fc.tolist(), fixed_delta_folds=d_fix.tolist(),
                         lofo_delta=float(w_f @ d_lofo),
                         lofo_folds=int((d_lofo < 0).sum()),
                         lofo_delta_folds=d_lofo.tolist(),
                         lofo_weights=[list(c) for c in chosen],
                         insample_best=list(grid[best]),
                         insample_delta=float(sc[best] - ref_wcv),
                         var_vs_tab=float(np.var(zm - z_tab)),
                         corr_resid_tab=float(np.corrcoef(ly - zm, ly - z_tab)[0, 1])))

    out = ARTIFACTS / "ETX1_lofo.json"
    out.write_text(json.dumps(dict(folds=folds, ref_exps=REF_EXPS, ref_w=REF_W,
                                   ref_fold_cal=ref_fc.tolist(), rows=rows),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаписано: {out}")


if __name__ == "__main__":
    main()
