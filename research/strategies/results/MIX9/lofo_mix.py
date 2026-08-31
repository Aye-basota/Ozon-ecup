"""MIX9: честный LOFO трёх кандидатов смеси на уже готовых OOF.

Опора для всех дельт — ОТПРАВЛЕННЫЙ чемпион `SEQ-01-MIX` (LB 1.6501764):
0.15 `S1-E10` + 0.20 `S1-E02` + 0.10 `S1-E03a` + 0.25 `S1-DIST` + 0.30 `SEQ-01-S42`.
Так требует `exp_027` §4: мерить от того, что уже прошло LB, а не от прежней базы.

Кандидаты — семейство `CAP + E02 + DIST + SEQ` при разном SEQ-члене:

  * `SEQ-01-S42`        — тот же член, что в чемпионе, но в 4-компонентной смеси;
  * `SEQ-AVG3`          — известный кандидат `exp_027` §7 (усреднение 3 сидов SEQ-01);
  * `SEQ-D3A-AVG3`      — основной кандидат (усреднение 3 сидов `SEQ-D3A`, `exp_030c`);
  * `SEQ-D3A-BASE-AVG3` — КОНТРОЛЬ СРЕДЫ: те же 12 прогонов без `--depth-aug`.
    Сиды 43/44 у D3A считались на арендованных A10, а `SEQ-AVG3` целиком локальный;
    `exp_030c` мерил абсолютное расхождение машин в 0.0010-0.0015, то есть больше
    самого эффекта. Разность `D3A-AVG3 − D3A-BASE-AVG3` от среды свободна, разность
    `D3A-AVG3 − SEQ-AVG3` — нет, и без контроля её читать нельзя.

Поиск весов СОЗНАТЕЛЬНО узкий (`MIX-E11`: подбор ради 4-го знака дал локальные
−0.00038 и +0.00023 на LB): `CAP` жёстко 0.10, доля SEQ только из {0.40, 0.45, 0.50},
остаток делят `E02` и `DIST` с шагом 0.05. Основное число кандидата — не оптимум
поиска, а фиксированные 0.10/0.20/0.25/0.45.

Запуск: PYTHONPATH=. python research/strategies/results/MIX9/lofo_mix.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from src.blend import aligned, fold_masks, shifted_rmsle
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1

CAP, E02, DIST = "S1-E03a", "S1-E02", "S1-DIST"
REF_EXPS = ["S1-E10", E02, CAP, DIST, "SEQ-01-S42"]
REF_W = [0.15, 0.20, 0.10, 0.25, 0.30]

SEQ_MEMBERS = ["SEQ-01-S42", "SEQ-AVG3", "SEQ-D3A-AVG3", "SEQ-D3A-BASE-AVG3"]
FIXED_W = [0.10, 0.20, 0.25, 0.45]          # CAP, E02, DIST, SEQ
SEQ_SHARES = [0.40, 0.45, 0.50]
STEP = 0.05


def narrow_grid() -> list[tuple[float, float, float, float]]:
    """Веса (CAP, E02, DIST, SEQ): CAP=0.10, SEQ из {0.40,0.45,0.50}, остаток шагом 0.05."""
    out = []
    for s in SEQ_SHARES:
        rest = round(1.0 - 0.10 - s, 10)
        n = int(round(rest / STEP))
        for i in range(n + 1):
            e = round(i * STEP, 10)
            out.append((0.10, e, round(rest - e, 10), s))
    return out


def fold_cal(Z: np.ndarray, ly: np.ndarray, masks, w) -> np.ndarray:
    z = np.average(Z, axis=0, weights=np.asarray(w, float))
    return np.array([shifted_rmsle(ly[m], z[m]) for m in masks])


def main() -> None:
    exps = sorted(set(REF_EXPS + [CAP, E02, DIST] + SEQ_MEMBERS))
    Z, y, cut = aligned(exps)
    idx = {e: i for i, e in enumerate(exps)}
    ly = np.log1p(y)
    folds, masks = fold_masks(cut)
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()
    print(f"n = {len(y):,} строк OOF, фолды {folds}\n")

    ref_fc = fold_cal(Z[[idx[e] for e in REF_EXPS]], ly, masks, REF_W)
    ref_wcv = float(w_f @ ref_fc)
    print(f"ОПОРА  SEQ-01-MIX {REF_W}  wCV={ref_wcv:.5f}   "
          + " ".join(f"{v:.5f}" for v in ref_fc))

    print("\nсоло SEQ-члена (калиброванный RMSLE пофолдово):")
    for m in SEQ_MEMBERS:
        fc = fold_cal(Z[[idx[m]]], ly, masks, [1.0])
        print(f"  {m:>20} wCV={float(w_f @ fc):.5f}   " + " ".join(f"{v:.5f}" for v in fc))

    # разнообразие: то, ради чего SEQ вообще стоит в смеси (exp_025, exp_032b)
    z_tab = np.average(Z[[idx[e] for e in (CAP, E02, DIST)]], axis=0,
                       weights=[0.10, 0.20, 0.25])
    print("\nразнообразие против табличной части (CAP+E02+DIST, веса 0.10/0.20/0.25):")
    for m in SEQ_MEMBERS:
        zm = Z[idx[m]]
        print(f"  {m:>20} Var(z - z_tab)={np.var(zm - z_tab):.5f}  "
              f"corr остатков={np.corrcoef(ly - zm, ly - z_tab)[0, 1]:.5f}")

    grid = narrow_grid()
    print(f"\nузкая сетка поиска: {len(grid)} комбинаций "
          f"(CAP=0.10, SEQ из {SEQ_SHARES}, шаг {STEP})")

    rows = []
    for m in SEQ_MEMBERS:
        sub = Z[[idx[CAP], idx[E02], idx[DIST], idx[m]]]
        FC = np.vstack([fold_cal(sub, ly, masks, w) for w in grid])
        sc = FC @ w_f

        fix_fc = fold_cal(sub, ly, masks, FIXED_W)
        fix_wcv = float(w_f @ fix_fc)
        d_fix = fix_fc - ref_fc
        wins_fix = int((d_fix < 0).sum())

        held = np.zeros(len(folds))
        chosen = []
        for h in range(len(folds)):
            keep = [i for i in range(len(folds)) if i != h]
            wh = w_f[keep] / w_f[keep].sum()
            b = int(np.argmin(FC[:, keep] @ wh))
            held[h] = FC[b, h]
            chosen.append(grid[b])
        d_lofo = held - ref_fc
        best = int(np.argmin(sc))

        print(f"\n===== SEQ-член: {m} =====")
        print(f"  фиксированные {FIXED_W}: wCV={fix_wcv:.5f}  "
              f"дельта к опоре {fix_wcv - ref_wcv:+.5f}  фолдов лучше {wins_fix}/4"
              f"  (10-16: {d_fix[3]:+.5f})")
        print("    пофолдово " + " ".join(f"{v:.5f}" for v in fix_fc)
              + "   дельты " + " ".join(f"{v:+.5f}" for v in d_fix))
        print(f"  {'отложенный фолд':<14}{'веса без него':<28}{'на нём':>10}"
              f"{'опора':>10}{'дельта':>10}")
        for h, f in enumerate(folds):
            print(f"  {f:<14}{str(list(chosen[h])):<28}{held[h]:>10.5f}"
                  f"{ref_fc[h]:>10.5f}{d_lofo[h]:>+10.5f}")
        print(f"  ЧЕСТНЫЙ LOFO по wCV: {float(w_f @ d_lofo):+.5f}"
              f"   фолдов лучше {int((d_lofo < 0).sum())}/4"
              f"   (в выборке оптимум {list(grid[best])}: {sc[best] - ref_wcv:+.5f})")

        rows.append(dict(member=m, ref_wcv=ref_wcv, fixed_wcv=fix_wcv,
                         fixed_delta=fix_wcv - ref_wcv, fixed_folds=wins_fix,
                         fixed_fold_cal=fix_fc.tolist(), fixed_delta_folds=d_fix.tolist(),
                         lofo_delta=float(w_f @ d_lofo),
                         lofo_folds=int((d_lofo < 0).sum()),
                         lofo_delta_folds=d_lofo.tolist(),
                         lofo_weights=[list(c) for c in chosen],
                         insample_best=list(grid[best]),
                         insample_delta=float(sc[best] - ref_wcv)))

    out = ARTIFACTS / "MIX9_lofo.json"
    out.write_text(json.dumps(dict(folds=folds, ref_exps=REF_EXPS, ref_w=REF_W,
                                   ref_fold_cal=ref_fc.tolist(), rows=rows),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаписано: {out}")


if __name__ == "__main__":
    main()
