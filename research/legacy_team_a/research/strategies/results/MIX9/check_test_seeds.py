"""MIX9: вменяемость новых тестовых моделей ДО сборки сабмита.

Тестовых меток нет, поэтому качество новых прогонов не измерить. Но три вещи
проверяемы без меток, и каждая уже один раз ловила ошибку в этом проекте:

1. **обрезка действительно применена.** `exp_027` намерил на тесте
   `Var(z_full − z_clip289) = 0.08990` у сида 42. У новых сидов величина обязана
   быть того же порядка; ноль означал бы, что `--depth-clip` не сработал;
2. **сиды согласованы между собой.** `exp_027` §1 мерил согласие сидов на
   `corr = 0.78..0.84` по оси глубины; здесь смотрится прямая корреляция
   прогнозов и `Var` попарных разностей — вырожденный или разъехавшийся сид
   виден сразу;
3. **разброс сидов на тесте сопоставим с разбросом на OOF.** На фолдах
   `Var(z_i − z_avg)` = 0.0063..0.0097 (`build_d3a_avg3.py`). Тестовая панель
   больше и глубина другая, поэтому точного равенства не ждём — но порядок
   величины обязан совпасть.

Запуск: PYTHONPATH=. python research/strategies/results/MIX9/check_test_seeds.py
"""
from __future__ import annotations

import itertools

import numpy as np

from src.config import ARTIFACTS

SEEDS = {42: "SEQ-01", 43: "SEQ-C289-S43", 44: "SEQ-C289-S44"}


def z(name: str) -> np.ndarray:
    return np.load(ARTIFACTS / f"ztest_{name}.npy")


def main() -> None:
    Z = {s: z(n) for s, n in SEEDS.items()}
    uid = np.load(ARTIFACTS / f"uid_{SEEDS[42]}.npy")
    print("уровень и форма прогноза (z = log1p(pred), обрезка 289):")
    for s, n in SEEDS.items():
        assert np.array_equal(np.load(ARTIFACTS / f"uid_{n}.npy"), uid), f"{n}: другой uid"
        v = Z[s]
        print(f"  сид {s} ({n:>14}): n={len(v):,} mean={v.mean():.5f} std={v.std():.5f} "
              f"нулей={float((v == 0).mean()):.3%} max={v.max():.3f}")

    print("\n1. обрезка применена (эталон exp_027, сид 42: Var = 0.08990):")
    for s, n in SEEDS.items():
        f = ARTIFACTS / f"ztest_{n}-FULL.npy"
        if not f.exists():
            print(f"  сид {s}: полной глубины на диске нет — пропуск")
            continue
        d = np.load(f) - Z[s]
        print(f"  сид {s}: Var(full - clip289) = {np.var(d):.5f}, "
              f"mean сдвиг {d.mean():+.5f}, доля |d|>0.1 = {float((np.abs(d) > 0.1).mean()):.1%}")

    print("\n2. согласие сидов между собой:")
    for a, b in itertools.combinations(SEEDS, 2):
        print(f"  {a} vs {b}: corr={np.corrcoef(Z[a], Z[b])[0, 1]:.5f}  "
              f"Var(разности)={np.var(Z[a] - Z[b]):.5f}")

    avg = np.mean(list(Z.values()), axis=0)
    print(f"\n3. разброс вокруг лог-среднего (на фолдах было 0.0063..0.0097):")
    for s in SEEDS:
        print(f"  сид {s}: Var(z - z_avg) = {np.var(Z[s] - avg):.5f}")
    print(f"  лог-среднее трёх сидов: mean z = {avg.mean():.5f}")

    # --- 4. цена ПЕРЕОБУЧЕНИЯ тестовой модели, которую проект ещё не мерил -------
    # У сидов 43/44 на диске лежат прогнозы `exp_026` на ПОЛНОЙ глубине. Новые
    # прогоны того же рецепта и того же сида дают свою полную глубину. Разность
    # двух — чистый шум прогона на тесте: та самая величина, которой рискует
    # сабмит, собранный из НОВЫХ моделей под OOF, снятый со СТАРЫХ.
    print("\n4. шум переобучения на тесте (тот же сид и рецепт, другой прогон, "
          "обе полной глубины):")
    for s, old in ((43, "SEQ-S43-FULL"), (44, "SEQ-S44-FULL")):
        f_new = ARTIFACTS / f"ztest_{SEEDS[s]}-FULL.npy"
        f_old = ARTIFACTS / f"ztest_{old}.npy"
        if not (f_new.exists() and f_old.exists()):
            print(f"  сид {s}: нет пары для сравнения — пропуск")
            continue
        a, b = np.load(f_new), np.load(f_old)
        print(f"  сид {s}: Var(новый - старый) = {np.var(a - b):.5f}, "
              f"corr = {np.corrcoef(a, b)[0, 1]:.5f}, "
              f"сдвиг уровня {a.mean() - b.mean():+.5f}")
    print("  ориентир: ось глубины у сида 42 стоила Var = 0.08990 (exp_027), "
          "\n  пол сидов на валидации Var(delta) = 0.00712 (exp_016)")


if __name__ == "__main__":
    main()
