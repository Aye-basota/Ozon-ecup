"""Проверки корректности пайплайна перед экспериментами.

1. Правило панели воспроизводит тестовую выборку ровно: 250 000 пользователей.
2. sample_submit == сумма gmv за [2026-01-15..2026-02-13] (якорь уровня).
3. Leakage-тест: build_features(T) на полном датасете и на датасете, обрезанном
   по T, дают побитово одинаковый результат.
4. Таргет не пересекается с признаковым окном.

Запуск: python -m src.smoke
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import polars as pl

from src import data
from src.config import ANCHOR_M_X, CUTOFF_TEST, HISTORY_L
from src.features import build_features, make_xy, panel_users, target
from src.validation import get_folds, describe_folds, gap_curve_folds

t0 = time.time()


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}", flush=True)
    assert cond, msg


print("== 1. панель ==")
u = panel_users(CUTOFF_TEST, 3)
ok(u.height == 250_000, f"panel_users(2026-02-13, 3).height == 250_000 (получено {u.height:,})")
ss = data.sample_submit()
ok(set(u["user_id"].to_list()) == set(ss["user_id"].to_list()),
   "состав панели совпадает с sample_submit")

print("== 2. якорь уровня из sample_submit ==")
df = data.load()
w = (df.lazy().filter((pl.col("event_date") >= dt.date(2026, 1, 15))
                      & (pl.col("event_date") <= dt.date(2026, 2, 13)))
     .group_by("user_id").agg(pl.col("gmv").sum().alias("x")).collect())
j = ss.join(w, on="user_id", how="left").with_columns(pl.col("x").fill_null(0.0))
mx = float(np.log1p(j["x"].to_numpy()).mean())
exact = float(np.mean(np.abs(j["predict"].to_numpy() - j["x"].to_numpy()) < 1e-5))
ok(exact > 0.999, f"sample_submit == gmv 30d [2026-01-15..02-13]: совпадений {exact:.4%}")
ok(abs(mx - ANCHOR_M_X) < 1e-3, f"m_x = {mx:.4f} (ожидание {ANCHOR_M_X})")

print("== 3. leakage-тест build_features ==")
T = dt.date(2025, 9, 16)
full = build_features(T, HISTORY_L)
data._CACHE["df"] = df.filter(pl.col("event_date") <= T)      # обрезаем «будущее»
trunc = build_features(T, HISTORY_L)
data._CACHE["df"] = df
ok(full.equals(trunc), "build_features(T) не зависит от строк с event_date > T")

print("== 4. таргет и окна ==")
X, y = make_xy(T, HISTORY_L)
ok(X.height == 190_690, f"3-блочная панель на 2025-09-16: n={X.height:,} (ожидание 190 690)")
ok(abs(float(np.log1p(y).mean()) - 2.6431) < 5e-3,
   f"mean log1p(y) = {float(np.log1p(y).mean()):.4f} (ожидание 2.6431)")
ok(abs(float((y > 0).mean()) - 0.6106) < 5e-3, f"P(y>0) = {float((y > 0).mean()):.4f} (ожидание 0.6106)")
y1 = target(T, panel_users(T, 1))["y"].to_numpy()
ok(abs(float(np.log1p(y1).mean()) - 2.4450) < 5e-3,
   f"1-блочная панель: mean log1p(y) = {float(np.log1p(y1).mean()):.4f} (ожидание 2.4450)")

print("== 5. фолды ==")
folds = get_folds()
print(describe_folds(folds))
for tr2, v2, g in gap_curve_folds():
    print(f"  S2 gap={g:3d}d: val {v2}  train {min(tr2)}..{max(tr2)} ({len(tr2)} cutoffs)")
for tr, V in folds:
    ok(max(tr) + dt.timedelta(days=30) <= V, f"фолд {V}: train-таргеты не заходят за V")

print(f"\nвсе проверки пройдены за {time.time() - t0:.0f}s")
