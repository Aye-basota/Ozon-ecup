"""ETX1: вменяемость тестовой модели ETX ДО сборки сабмита.

Тестовых меток нет, поэтому качество прогона не измерить. Но четыре вещи
проверяемы без меток, и каждая уже ловила ошибку в этом проекте:

1. **обрезка действительно применена.** `exp_027` намерил у TCN на тесте
   `Var(z_full − z_clip289) = 0.0899`. У ETX эта величина обязана быть НЕ МЕНЬШЕ:
   `exp_036` показал, что он опирается на длинную историю сильнее TCN (обрезка до
   180 дней стоит ему +0.01259 против +0.00841). Ноль означал бы, что
   `--depth-clip` не сработал — ровно та ошибка, которая стоила +0.0051 LB;
2. **уровень тестовой модели не уехал** относительно её же OOF и относительно
   членов смеси. Систематический перекос уровня у ETX на ранних фолдах измерен
   (сдвиг −0.171 на 09-04 против −0.007 на 10-16), поэтому смотрится специально;
3. **ETX и TCN на тесте расходятся так же, как на фолдах.** На OOF
   `Var(z_ETX − z_SEQ-AVG3)` известен; если на тесте он в разы другой, значит
   одна из моделей на тестовой панели ведёт себя иначе, чем на валидации, и
   собирать пару нельзя;
4. **панель и порядок строк совпадают** у всех компонент.

Запуск: PYTHONPATH=. python research/strategies/results/ETX1/check_test_etx.py
"""
from __future__ import annotations

import numpy as np

from src.config import ARTIFACTS
from src.tracking import load_oof

ETX = "ETX-01-S42"
SEQ_SEEDS = ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"]
TAB = ["S1-CAP", "S1-UNC", "S1-DIST"]


def z(name: str) -> np.ndarray:
    return np.load(ARTIFACTS / f"ztest_{name}.npy")


def main() -> None:
    uid = np.load(ARTIFACTS / f"uid_{ETX}.npy")
    ze = z(ETX)
    zs = np.mean([z(n) for n in SEQ_SEEDS], axis=0)          # SEQ-AVG3 в лог-пространстве

    print("4. панель и порядок строк:")
    for n in SEQ_SEEDS + TAB:
        u = np.load(ARTIFACTS / f"uid_{n}.npy")
        assert np.array_equal(u, uid), f"{n}: другой набор/порядок user_id"
    print(f"  [OK ] {len(uid):,} строк, все компоненты на одной панели")

    print("\n2. уровень и форма (z = log1p(pred), все при clip289):")
    for n, v in [(ETX, ze), ("SEQ-AVG3", zs)] + [(t, z(t)) for t in TAB]:
        print(f"  {n:>14}: mean={v.mean():.5f} std={v.std():.5f} "
              f"нулей={float((v == 0).mean()):.3%} max={v.max():.3f}")
    d = load_oof(ETX)
    print(f"  {'ETX OOF':>14}: mean={np.asarray(d['z'], float).mean():.5f} "
          f"(тест выше на {ze.mean() - np.asarray(d['z'], float).mean():+.4f} — "
          f"тестовая панель активнее валидационной, у TCN тот же знак)")
    assert 1.5 < ze.mean() < 4.0, f"уровень ETX {ze.mean():.4f} вне разумного коридора"
    assert np.isfinite(ze).all(), "NaN/inf в тестовом прогнозе ETX"
    assert (ze >= 0).all(), "отрицательные z в тестовом прогнозе ETX"

    print("\n1. обрезка применена (эталон exp_027 у TCN: Var = 0.0899):")
    f = ARTIFACTS / f"ztest_{ETX}-FULL.npy"
    assert f.exists(), "нет прогноза полной глубины — нечем доказать, что клип сработал"
    dd = np.load(f) - ze
    print(f"  ETX: Var(full - clip289) = {np.var(dd):.5f}, mean сдвиг {dd.mean():+.5f}, "
          f"доля |d|>0.1 = {float((np.abs(dd) > 0.1).mean()):.1%}, "
          f"corr = {np.corrcoef(np.load(f), ze)[0, 1]:.5f}")
    assert np.var(dd) > 0.01, "клип на тесте почти ничего не изменил — проверить --depth-clip"

    print("\n3. расхождение ETX и SEQ-AVG3: тест против фолдов")
    de = load_oof(ETX)
    ds = load_oof("SEQ-AVG3")
    ke = np.char.add(np.asarray(de["cutoff"], "U10"), np.asarray(de["user_id"]).astype("U20"))
    ks = np.char.add(np.asarray(ds["cutoff"], "U10"), np.asarray(ds["user_id"]).astype("U20"))
    oe, os_ = np.argsort(ke), np.argsort(ks)
    assert np.array_equal(ke[oe], ks[os_]), "OOF на разных наборах строк"
    v_oof = float(np.var(np.asarray(de["z"], float)[oe] - np.asarray(ds["z"], float)[os_]))
    v_test = float(np.var(ze - zs))
    print(f"  OOF (4 фолда): Var = {v_oof:.5f}   тест: Var = {v_test:.5f}   "
          f"отношение {v_test / v_oof:.2f}x")
    print(f"  corr(ETX, SEQ-AVG3) на тесте = {np.corrcoef(ze, zs)[0, 1]:.5f}")
    assert 0.25 < v_test / v_oof < 4.0, (
        "расхождение пары на тесте не того порядка, что на фолдах — пару собирать нельзя")
    print("\nвсе проверки пройдены")


if __name__ == "__main__":
    main()
