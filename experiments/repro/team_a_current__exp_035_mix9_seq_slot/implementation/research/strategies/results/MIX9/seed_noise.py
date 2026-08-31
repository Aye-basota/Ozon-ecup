"""MIX9: разрешима ли разница между кандидатами в слоте SEQ.

`lofo_mix.py` даёт `D3A-AVG3` −0.00061 и `SEQ-AVG3` −0.00055 к чемпиону. Разница
0.00006 — это 4-й знак, который `MIX-E11` уже стоил проекту +0.00023 LB. Вопрос,
который решает выбор кандидата: какой шум у этой величины.

Здесь он меряется прямо, а не берётся из общей константы `seed std` = 0.00250:
каждый ОТДЕЛЬНЫЙ сид ставится в слот SEQ при боевых весах 0.10/0.20/0.25/0.45, и
считается разброс получившихся wCV смеси внутри семейства. Это ровно тот шум,
который усреднение трёх сидов делит на sqrt(3).

Запуск: PYTHONPATH=. python research/strategies/results/MIX9/seed_noise.py
"""
from __future__ import annotations

import numpy as np

from src.blend import aligned, fold_masks, shifted_rmsle
from src.config import FOLD_WEIGHTS_S1

CAP, E02, DIST = "S1-E03a", "S1-E02", "S1-DIST"
W = [0.10, 0.20, 0.25, 0.45]

FAMILIES = {
    "SEQ-01 (exp_026, локально)": ["SEQ-01-S42", "SEQ-01-S43", "SEQ-01-S44"],
    "SEQ-D3A (exp_030c)": ["SEQ-D3A-S42", "SEQ-D3A-G1-S43", "SEQ-D3A-G2-S44"],
    "SEQ-D3A-BASE (exp_030c)": ["SEQ-D3A-BASE-S42", "SEQ-D3A-G1-BASE-S43",
                                "SEQ-D3A-G2-BASE-S44"],
}
AVERAGES = {"SEQ-01": "SEQ-AVG3", "SEQ-D3A": "SEQ-D3A-AVG3",
            "SEQ-D3A-BASE": "SEQ-D3A-BASE-AVG3"}
REF_EXPS = ["S1-E10", E02, CAP, DIST, "SEQ-01-S42"]
REF_W = [0.15, 0.20, 0.10, 0.25, 0.30]


def main() -> None:
    members = sorted({m for v in FAMILIES.values() for m in v} | set(AVERAGES.values()))
    exps = sorted(set(REF_EXPS + [CAP, E02, DIST] + members))
    Z, y, cut = aligned(exps)
    idx = {e: i for i, e in enumerate(exps)}
    ly = np.log1p(y)
    folds, masks = fold_masks(cut)
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()

    def mix_wcv(seq: str) -> float:
        sub = Z[[idx[CAP], idx[E02], idx[DIST], idx[seq]]]
        z = np.average(sub, axis=0, weights=W)
        return float(w_f @ np.array([shifted_rmsle(ly[m], z[m]) for m in masks]))

    ref = np.average(Z[[idx[e] for e in REF_EXPS]], axis=0, weights=REF_W)
    ref_wcv = float(w_f @ np.array([shifted_rmsle(ly[m], ref[m]) for m in masks]))
    print(f"опора SEQ-01-MIX: wCV={ref_wcv:.5f}\n")
    print("wCV смеси CAP/E02/DIST/SEQ = 0.10/0.20/0.25/0.45 при ОДНОМ сиде в слоте SEQ:")

    for name, parts in FAMILIES.items():
        v = np.array([mix_wcv(p) for p in parts])
        d = v - ref_wcv
        print(f"\n  {name}")
        for p, x in zip(parts, d):
            print(f"    {p:>22} дельта к опоре {x:+.5f}")
        print(f"    среднее {d.mean():+.5f}   sd по сидам {d.std(ddof=1):.5f}"
              f"   ожидаемый sd среднего трёх {d.std(ddof=1) / np.sqrt(3):.5f}")

    print("\nфактическое усреднение (сначала z, потом метрика — не то же самое,"
          "\nчто среднее пофолдовых wCV, и обычно лучше него):")
    for name, avg in AVERAGES.items():
        print(f"    {avg:>22} дельта к опоре {mix_wcv(avg) - ref_wcv:+.5f}")

    sd = np.mean([np.std([mix_wcv(p) for p in parts], ddof=1) / np.sqrt(3)
                  for parts in FAMILIES.values()])
    print(f"\nсредний sd 3-сидового кандидата в слоте SEQ: {sd:.5f}")
    print(f"sd РАЗНОСТИ двух независимых 3-сидовых кандидатов: {sd * np.sqrt(2):.5f}")
    print("пол разрешения валидатора (STATE): 0.00050; парная SE public LB: 0.00025")


if __name__ == "__main__":
    main()
