"""ETX2 — вменяемость ТРЁХ тестовых моделей ETX до сборки сабмита (`EXP-037`).

Расширение `ETX1/check_test_etx.py` на три сида и на новую политику статика.
Метки теста недоступны, но проверяемо без них:

1. **обрезка глубины применена** — `Var(z_full − z_clip289)` у каждого сида
   заметно больше нуля (`exp_027`: у TCN 0.0899; у ETX опора длинной истории
   сильнее, значит не меньше);
2. **статик приведён в обученный диапазон** — прогноз `*-DCW` отличается от
   сырого, и уровень уехал ВВЕРХ (сырой занижал: `mean z` 2.10 против 2.39 у
   `SEQ-AVG3`, механизм — `depth_fix.py`);
3. **сиды согласованы между собой** — `Var(z_i − z_avg)` того же порядка, что у
   сидов TCN на тесте (`exp_035`: 0.0055..0.0059);
4. **режим пары ETX-AVG3 ↔ SEQ-AVG3 на тесте совпадает с OOF** — главный гейт
   `exp_036`; отношение вне 0.6–1.2x останавливает сборку;
5. **панель и порядок строк** совпадают у всех компонент.

Запуск: PYTHONPATH=. python research/strategies/results/ETX2/check_test_etx2.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "research/strategies/results/ETX2")
import common  # noqa: E402

from src.config import ARTIFACTS  # noqa: E402
from src.tracking import load_oof  # noqa: E402

ETX = [f"ETX-01-S4{i}-DCW" for i in (2, 3, 4)]
ETX_RAW = [f"ETX-01-S4{i}" for i in (2, 3, 4)]
SEQ = ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"]
TAB = ["S1-CAP", "S1-UNC", "S1-DIST"]
RATIO_BAND = (0.6, 1.2)          # полоса всех пар БЕЗ ETX (`exp_036`): 0.63..1.11


def main() -> None:
    uid = common.test_uid()
    print("5. панель и порядок строк:")
    for n in ETX + SEQ + TAB:
        u = np.load(ARTIFACTS / f"uid_{n}.npy")
        assert np.array_equal(u, uid), f"{n}: другой набор/порядок user_id"
    print(f"  [OK ] {len(uid):,} строк, все компоненты на одной панели")

    ze = [common.ztest(n) for n in ETX]
    zavg = np.mean(ze, axis=0)
    zs = np.mean([common.ztest(n) for n in SEQ], axis=0)

    print("\n2. статик приведён в обученный диапазон (глубина 289, dow = четверг):")
    for n, r, z in zip(ETX, ETX_RAW, ze):
        p = ARTIFACTS / f"ztest_{r}.npy"
        if not p.exists():
            print(f"  {n}: сырого прогноза нет, сравнение пропущено")
            continue
        zr = np.load(p)
        print(f"  {n}: mean {z.mean():.4f} против сырого {zr.mean():.4f} "
              f"({z.mean() - zr.mean():+.4f}), Var(разности) {np.var(z - zr):.5f}, "
              f"corr {np.corrcoef(z, zr)[0, 1]:.5f}")
        assert z.mean() > zr.mean(), "уровень не поднялся — статик не применён?"
    for n, z in zip(ETX, ze):
        assert np.isfinite(z).all() and (z >= 0).all(), f"{n}: NaN/inf/отрицательные"
        assert 1.5 < z.mean() < 4.0, f"{n}: уровень {z.mean():.4f} вне коридора"

    print("\n1. обрезка глубины применена (эталон exp_027 у TCN: Var = 0.0899):")
    for n, r in zip(ETX, ETX_RAW):
        p = ARTIFACTS / f"ztest_{r}-FULL.npy"
        if not p.exists():
            print(f"  {r}: полной глубины на диске нет, проверка пропущена")
            continue
        d = np.load(p) - common.ztest(n)
        print(f"  {r}: Var(full − DCW) = {np.var(d):.5f}, mean сдвиг {d.mean():+.5f}")
        assert np.var(d) > 0.01, "обрезка почти ничего не изменила — проверить --depth-clip"

    print("\n3. согласованность сидов ETX на тесте (у TCN 0.0055..0.0059, exp_035):")
    for n, z in zip(ETX, ze):
        print(f"  {n}: Var(z − z_avg) = {np.var(z - zavg):.5f}, mean {z.mean():.4f}")
    for i in range(len(ze)):
        for j in range(i + 1, len(ze)):
            print(f"  {ETX[i]} − {ETX[j]}: Var = {np.var(ze[i] - ze[j]):.5f}, "
                  f"corr = {np.corrcoef(ze[i], ze[j])[0, 1]:.5f}")

    print("\n4. режим пары ETX-AVG3 ↔ SEQ-AVG3: тест против OOF")
    de, ds = load_oof("ETX-AVG3"), load_oof("SEQ-AVG3")
    ke = np.char.add(np.asarray(de["cutoff"], "U10"), np.asarray(de["user_id"]).astype("U20"))
    ks = np.char.add(np.asarray(ds["cutoff"], "U10"), np.asarray(ds["user_id"]).astype("U20"))
    oe, os_ = np.argsort(ke), np.argsort(ks)
    assert np.array_equal(ke[oe], ks[os_]), "OOF на разных наборах строк"
    d_oof = np.asarray(de["z"], float)[oe] - np.asarray(ds["z"], float)[os_]
    v_oof, v_test = float(np.var(d_oof)), float(np.var(zavg - zs))
    print(f"  OOF {v_oof:.5f}   тест {v_test:.5f}   отношение {v_test / v_oof:.2f}x   "
          f"corr на тесте {np.corrcoef(zavg, zs)[0, 1]:.5f}")
    lo, hi = RATIO_BAND
    assert lo < v_test / v_oof < hi, (
        f"режим пары вне полосы {lo}..{hi}x — сборка остановлена (`exp_036`)")
    print("\nвсе проверки пройдены")


if __name__ == "__main__":
    main()
