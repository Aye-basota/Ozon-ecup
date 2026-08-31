"""S3 — сезонная поправка: насколько целевое окно теста отличается от обучающих.

Тестовое target-окно 2026-02-14..03-15 календарно совпадает с 2025-02-14..03-15,
а обучающие cutoff'ы покрывают май–ноябрь 2025. Модель выучивает СРЕДНЕЕ по своим
целевым окнам, поэтому нужна поправка

    delta_season = dm(тестовое окно) - mean(dm по обучающим cutoff'ам),

где dm(T) = mean log1p(y_{T+1..T+30}) - mean log1p(x_{T-29..T}) — сдвиг уровня
за 30 дней вперёд относительно последних 30 дней, на ФИКСИРОВАННОЙ панели
(1-блочное правило — единственное, конструируемое с 2025-01-31).

Часть A считается без единой модели: это чистая статистика данных.
Часть B — модельная проверка: модель, обученная на «обычных» cutoff'ах,
применяется назад по времени к нескольким контрольным датам, включая
календарный аналог теста; сезонный фолд обязан выделяться на фоне остальных.

Запуск: python -m src.seasonal
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc

import numpy as np
import polars as pl

from src.config import CORRIDOR_END, DATA_START
from src.data import load
from src.features import feature_names, make_xy, panel_users, target, to_np
from src.train import Setup, assemble, fit_free, infer
from src.validation import bias_z, rmsle_z

YOY = dt.date(2025, 2, 13)          # календарный аналог тестового cutoff'а


def dm_stats(step: int = 7):
    """Часть A: dm(T) на 1-блочной панели.

    Сетка выровнена так, чтобы в неё попали И реальные обучающие cutoff'ы
    (та же сетка, что в cutoff_grid), И календарный аналог теста 2025-02-13.
    """
    from src.config import cutoff_grid
    df = load()
    grid = set(cutoff_grid(90, step))
    T = YOY
    while T >= dt.date(2025, 1, 31):                 # назад от YoY-аналога
        grid.add(T)
        T -= dt.timedelta(days=step)
    T = YOY
    while T <= CORRIDOR_END:                          # вперёд от YoY-аналога
        grid.add(T)
        T += dt.timedelta(days=step)
    out = []
    for T in sorted(grid):
        u = panel_users(T, 1)
        a30, b30 = T - dt.timedelta(days=29), T
        x = (df.lazy().filter((pl.col("event_date") >= a30) & (pl.col("event_date") <= b30))
             .group_by("user_id").agg(pl.col("gmv").sum().alias("x")).collect())
        xx = (u.join(x, on="user_id", how="left").with_columns(pl.col("x").fill_null(0.0))
              ["x"].to_numpy())
        yy = target(T, u)["y"].to_numpy()
        mx, my = float(np.log1p(xx).mean()), float(np.log1p(yy).mean())
        out.append((T, u.height, mx, my, my - mx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    load()

    print("=" * 96)
    print("A. dm(T) = mean log1p(y_next30) - mean log1p(x_last30), 1-блочная панель")
    print("=" * 96)
    rows = dm_stats(a.step)
    print(f"{'cutoff':>12} {'n':>9} {'m_x':>8} {'m_y':>8} {'dm':>8}")
    for T, n, mx, my, dm in rows:
        from src.config import cutoff_grid as _cg
        mark = "  <- YoY-аналог теста" if T == YOY else ("  <- коридор" if T in set(_cg(90, a.step)) else "")
        print(f"{str(T):>12} {n:>9,} {mx:>8.4f} {my:>8.4f} {dm:>+8.4f}{mark}")

    from src.config import cutoff_grid
    train_grid = set(cutoff_grid(90, a.step))
    corr = [dm for T, _, _, _, dm in rows if T in train_grid]
    yoy = [dm for T, _, _, _, dm in rows if T == YOY][0]
    print(f"\n  коридор обучения (T >= 2025-04-03): n={len(corr)}  mean dm={np.mean(corr):+.4f}  "
          f"median={np.median(corr):+.4f}  std={np.std(corr):.4f}")
    print(f"  YoY-аналог теста 2025-02-13:        dm={yoy:+.4f}")
    print(f"  => delta_season (полная)  = {yoy - np.mean(corr):+.4f}")
    print(f"  => delta_season (половина) = {(yoy - np.mean(corr)) / 2:+.4f}   "
          f"(n=1 год, поправка на неопределённость)")

    if a.skip_model:
        return

    print("\n" + "=" * 96)
    print("B. Модельная проверка: обучение на коридоре -> предсказание НАЗАД во времени")
    print("=" * 96)
    L = 43                       # на 2025-02-13 доступно ровно 43 дня истории
    s = Setup(L=None, min_history=90, step=14, panel_blocks=1, train_blocks=1)
    s.L = L
    from src.features import features_cached
    vals = [YOY, dt.date(2025, 3, 13), dt.date(2025, 4, 10), dt.date(2025, 5, 8),
            dt.date(2025, 6, 5)]
    train_cuts = [dt.date(2025, 7, 3) + dt.timedelta(days=14 * k) for k in range(8)]
    train_cuts = [T for T in train_cuts if T <= CORRIDOR_END]
    for T in vals + train_cuts:
        features_cached(T, L)
    Xr, _ = make_xy(train_cuts[0], L, 1)
    feats = feature_names(Xr)
    Xtr, ytr, _ = assemble(train_cuts, s, feats)
    print(f"  train: {len(train_cuts)} cutoff'ов {min(train_cuts)}..{max(train_cuts)}, "
          f"{Xtr.shape[0]:,} строк, {len(feats)} признаков (L={L})")
    box = [Xtr]
    del Xtr
    m = fit_free(s, box, ytr, None)
    gc.collect()
    print(f"\n{'val cutoff':>12} {'n':>9} {'RMSLE':>9} {'bias':>9}   комментарий")
    biases = {}
    for V in vals:
        Xv, yv = make_xy(V, L, 1)
        z = np.maximum(infer(s, m, to_np(Xv, feats)), 0.0)
        b = bias_z(yv, z)
        biases[V] = b
        note = "СЕЗОННЫЙ ФОЛД (аналог теста)" if V == YOY else "контроль"
        print(f"{str(V):>12} {len(yv):>9,} {rmsle_z(yv, z):>9.5f} {b:>+9.4f}   {note}")
    ctrl = np.mean([b for V, b in biases.items() if V != YOY])
    print(f"\n  средний bias по контрольным фолдам: {ctrl:+.4f}")
    print(f"  bias на сезонном фолде:              {biases[YOY]:+.4f}")
    print(f"  => разница (модельная оценка delta_season) = {biases[YOY] - ctrl:+.4f}")


if __name__ == "__main__":
    main()
