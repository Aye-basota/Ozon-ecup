"""ETX2 — механизм блокера ETX и его проверка на РАЗМЕЧЕННЫХ данных (EXP-037).

Что найдено чтением кода, а не подбором. Query-токен ETX несёт статик-контекст
`[depth/365, log1p(depth), ...]`, где `depth = min(cut_day + 1, 365)` —
КАЛЕНДАРНАЯ глубина cutoff'а (`src/etx.py`, `Tokenizer.__call__`). `--depth-clip D`
режет ОКНО событий (`select`), но статик не трогает. Отсюда:

* обучающие cutoff'ы тестовой модели 2025-04-03..2025-10-16 дают `depth` 93..289;
* тестовый cutoff 2026-02-13 даёт `cut_day + 1 = 409` -> `depth = 365`,
  то есть значение, которого в обучении НЕ БЫЛО НИ РАЗУ, и пару «окно 289 дней +
  глубина 365», которой не существует по построению.

У TCN этой дыры нет: `seq.gather(depth_clip=D)` гасит канал `avail` на срезанных
днях, поэтому обрезка автоматически ставит модель в ВИДЕННЫЙ режим. Это ровно то,
что `exp_027` купил ценой +0.0051 LB, и для ETX оно оказалось выполнено лишь
наполовину.

Предсказание механизма, сделанное ДО замера: `exp_036` намерил, что калибровочный
сдвиг ETX растёт к МЕЛКИМ глубинам (−0.171 на 09-04 против −0.007 на 10-16), то
есть на малой глубине модель завышает уровень. Экстраполяция за 289 обязана дать
обратный знак — занижение. Факт на тесте: `mean(z_ETX) = 2.104` против 2.390 у
`SEQ-AVG3` и 2.476 у `S1-DIST`, при том что на фолде 10-16 ETX был ВЫШЕ TCN на
+0.025. Знак совпал.

Здесь механизм проверяется там, где ЕСТЬ МЕТКИ, и потому проверка честная:

  A. `--mode fold`: на фолде 10-16 окно режется до D, а статик-глубина либо
     остаётся календарной (289, как сейчас на тесте), либо тоже становится D.
     Если дело в статике, вторая версия обязана быть ЛУЧШЕ по RMSLE и ближе к TCN.
     Опора — кривая глубины `exp_036` (D=150: +0.02082, D=254: +0.00244).
  B. `--mode test`: тестовый прогноз тем же чекпойнтом при `depth_clip=289` и
     `depth_cap=289`. Метки не нужны — меряется режим (`regime.py`).

Запуск:
  PYTHONPATH=. python research/strategies/results/ETX2/depth_fix.py --mode fold \
      --ckpt ETX-01-S42-V1016 --depths 150 212 254
  PYTHONPATH=. python research/strategies/results/ETX2/depth_fix.py --mode test \
      --ckpt ETX-01-S42-TEST --exp ETX-01-S42-DC
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt

import numpy as np

from src import etx, seq
from src.config import ARTIFACTS, CUTOFF_TEST
from src.features import panel_users
from src.validation import calibrate, rmsle_z

OUT = "research/strategies/results/ETX2"


def predict_with(model, tk, T, rows, cfg, dev, depth_clip, depth_cap, dow_shift=0.0):
    tk.depth_cap, tk.cdow_shift = depth_cap, dow_shift
    try:
        return np.maximum(etx.predict(model, tk, T, rows, cfg, dev,
                                      depth_clip=depth_clip), 0.0)
    finally:
        tk.depth_cap, tk.cdow_shift = None, 0.0


def mode_fold(a) -> None:
    model, tk, cfg, V, dev = etx.load_ckpt(a.ckpt)
    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = seq.user_rows(uv)
    y = seq.target_at(V, rv)
    cal_day = seq.day_index(V) + 1
    print(f"чекпойнт {a.ckpt}, фолд {V}, {len(uv):,} строк, "
          f"календарная глубина cutoff'а {cal_day}")

    z0 = predict_with(model, tk, V, rv, cfg, dev, None, None)
    d0, _ = calibrate(y, z0)
    base = rmsle_z(y, np.maximum(z0 + d0, 0.0))
    print(f"  полная глубина: RMSLE_cal = {base:.5f}, mean z = {z0.mean():.4f}")

    rows = [dict(depth="full", static="кал.", rmsle_cal=base, d=0.0,
                 mean_z=float(z0.mean()), var_vs_full=0.0)]
    print(f"\n{'D':>6}{'статик':>10}{'RMSLE_cal':>12}{'Δ к полной':>12}"
          f"{'mean z':>10}{'Var(z−z_full)':>15}")
    for D in a.depths:
        for cap, tag in [(None, "кал."), (D, f"={D}")]:
            z = predict_with(model, tk, V, rv, cfg, dev, D, cap)
            d, _ = calibrate(y, z)
            r = rmsle_z(y, np.maximum(z + d, 0.0))
            print(f"{D:>6}{tag:>10}{r:>12.5f}{r - base:>+12.5f}{z.mean():>10.4f}"
                  f"{float(np.var(z - z0)):>15.5f}")
            rows.append(dict(depth=D, static=tag, rmsle_cal=r, d=r - base,
                             mean_z=float(z.mean()), var_vs_full=float(np.var(z - z0))))
    # чувствительность к дню недели cutoff'а: в обучении он КОНСТАНТА (четверг),
    # на тесте — пятница, то есть вход, вес которого ничем не закреплён
    print("")
    print(f"{'сдвиг dow':>10}{'RMSLE_cal':>12}{'Δ':>10}{'mean z':>10}{'Var(z−z0)':>12}")
    for sh in a.dow:
        z = predict_with(model, tk, V, rv, cfg, dev, None, None, dow_shift=sh)
        d, _ = calibrate(y, z)
        r = rmsle_z(y, np.maximum(z + d, 0.0))
        print(f"{sh:>10.0f}{r:>12.5f}{r - base:>+10.5f}{z.mean():>10.4f}"
              f"{float(np.var(z - z0)):>12.5f}")
        rows.append(dict(depth="full", static=f"dow{sh:+.0f}", rmsle_cal=r, d=r - base,
                         mean_z=float(z.mean()), var_vs_full=float(np.var(z - z0))))

    with open(f"{OUT}/depth_static_{a.ckpt}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nзаписано: {OUT}/depth_static_{a.ckpt}.csv")


def mode_test(a) -> None:
    model, tk, cfg, _, dev = etx.load_ckpt(a.ckpt)
    ut = panel_users(CUTOFF_TEST, 3)["user_id"].to_numpy()
    rt = seq.user_rows(ut)
    cal_day = seq.day_index(CUTOFF_TEST) + 1
    print(f"чекпойнт {a.ckpt}, тест {CUTOFF_TEST}, {len(ut):,} строк, "
          f"календарная глубина {cal_day} -> статик был min({cal_day}, 365) = "
          f"{min(cal_day, 365)}, станет {a.depth_clip}")
    if a.dow_shift:
        w = (CUTOFF_TEST.weekday() + int(a.dow_shift)) % 7
        print(f"  день недели cutoff'а: {CUTOFF_TEST.weekday()} -> {w} "
              f"(все обучающие cutoff'ы = 3, четверг; в обучении этот вход КОНСТАНТА)")
    z = predict_with(model, tk, CUTOFF_TEST, rt, cfg, dev, a.depth_clip, a.depth_clip,
                     dow_shift=a.dow_shift)
    np.save(ARTIFACTS / f"ztest_{a.exp}.npy", z.astype(np.float64))
    np.save(ARTIFACTS / f"uid_{a.exp}.npy", ut)
    old = ARTIFACTS / f"ztest_{a.ckpt.replace('-TEST', '')}.npy"
    print(f"  mean z = {z.mean():.5f}, std = {z.std():.5f}, "
          f"нулей {float((z == 0).mean()):.3%}")
    if old.exists():
        zo = np.load(old)
        print(f"  против прежнего (статик 365): Var(разности) = {np.var(z - zo):.5f}, "
              f"сдвиг уровня {z.mean() - zo.mean():+.5f}, "
              f"corr = {np.corrcoef(z, zo)[0, 1]:.5f}")
    print(f"  сохранено: artifacts/ztest_{a.exp}.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fold", "test"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--depths", type=int, nargs="*", default=[150, 212, 254])
    ap.add_argument("--depth-clip", type=int, default=289, dest="depth_clip")
    ap.add_argument("--dow", type=float, nargs="*", default=[1.0, -1.0],
                    help="сдвиги дня недели cutoff'а для диагностики (fold)")
    ap.add_argument("--dow-shift", type=float, default=0.0, dest="dow_shift",
                    help="сдвиг dow cutoff'а в статике query (test): -1 = вернуть четверг")
    ap.add_argument("--exp", default=None)
    a = ap.parse_args()
    if a.mode == "fold":
        mode_fold(a)
    else:
        a.exp = a.exp or (a.ckpt.replace("-TEST", "") + "-DC")
        mode_test(a)


if __name__ == "__main__":
    main()
