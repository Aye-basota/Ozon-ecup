"""S2 — измерение out-of-time дрейфа как функции разрыва train->val.

Реальный разрыв «последний чистый cutoff (2025-10-16) -> тест (2026-02-13)» равен
120 дням и локально недостижим. Поэтому bias измеряется на нескольких доступных
разрывах и экстраполируется линейно (research/strategy_1.md §10, Эксперимент 5).

Одновременно скрипт печатает уровень mean(log1p(pred)) на ТЕСТОВОМ cutoff'е для
модели, обученной на всём чистом коридоре — это второй, независимый способ оценить
поправку (якорь из sample_submit, eda_findings §5.3).

Запуск: python -m src.drift --L 180 --min-history 90
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc

import numpy as np

from src.config import ANCHOR_BAND, CUTOFF_TEST, S2_VAL
from src.data import load
from src.features import feature_names, make_xy, to_np
from src.train import Setup, assemble, fit_free, infer
from src.validation import best_offset, bias_z, gap_curve_folds, rmsle_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=180)
    ap.add_argument("--min-history", type=int, default=90)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--train-blocks", type=int, default=1)
    ap.add_argument("--model", default="direct")
    ap.add_argument("--norm-long", action="store_true")
    a = ap.parse_args()
    s = Setup(L=a.L, min_history=a.min_history, step=a.step, model=a.model,
              train_blocks=a.train_blocks, norm_long=a.norm_long)
    load()
    Xv, yv = make_xy(S2_VAL, s.L, s.panel_blocks, norm_long=s.norm_long)
    feats = feature_names(Xv)
    Av = to_np(Xv, feats)
    ly = np.log1p(yv)

    print(f"S2 drift curve: val={S2_VAL}  L={s.L}  min_hist={s.min_history}  "
          f"train_blocks={s.train_blocks}  n_train_cutoffs={a.n_train}  feats={len(feats)}")
    # RMSLE после оптимального сдвига — главное число: показывает, остаётся ли
    # преимущество конфигурации после того, как ошибка УРОВНЯ убрана.
    print(f"{'gap,д':>6} {'глубина ист., дн':>17} {'train окно':>25} {'RMSLE':>9} {'bias':>9} "
          f"{'после сдвига':>13}")
    gaps, biases = [], []
    for tr, V, g in gap_curve_folds(s.min_history, s.step, S2_VAL, a.n_train):
        Xtr, ytr, _ = assemble(tr, s, feats)
        box = [Xtr]
        del Xtr
        m = fit_free(s, box, ytr, None)
        z = np.maximum(infer(s, m, Av), 0.0)
        b = bias_z(yv, z)
        _, sc_o = best_offset(yv, z)
        gaps.append(g); biases.append(b)
        depth = f"{(min(tr) - dt.date(2025, 1, 1)).days}..{(max(tr) - dt.date(2025, 1, 1)).days}"
        print(f"{g:>6} {depth:>17} {str(min(tr)) + '..' + str(max(tr)):>25} "
              f"{rmsle_z(yv, z):>9.5f} {b:>+9.4f} {sc_o:>13.5f}", flush=True)
        del ytr, m, box
        gc.collect()

    gaps_a, bias_a = np.array(gaps, float), np.array(biases)
    k, c = np.polyfit(gaps_a, bias_a, 1)
    test_gap = (CUTOFF_TEST - dt.date(2025, 10, 16)).days
    print(f"\nлинейная подгонка: bias = {k:+.5f} * gap {c:+.4f}   (R^2="
          f"{1 - np.var(bias_a - (k * gaps_a + c)) / max(np.var(bias_a), 1e-12):.3f})")
    print(f"экстраполяция на тестовый разрыв {test_gap} дн: bias = {k * test_gap + c:+.4f}")
    print(f"  => delta_drift = {k * test_gap + c:+.4f} (столько нужно прибавить к z)")

    # --- независимая проверка: уровень прогноза на тестовом cutoff'е -------------
    grid = [T for T in s.grid()]
    Xtr, ytr, _ = assemble(grid, s, feats)
    box = [Xtr]
    del Xtr
    m = fit_free(s, box, ytr, None)
    Xt, _ = make_xy(CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
    zt = np.maximum(infer(s, m, to_np(Xt, feats)), 0.0)
    lo, hi = ANCHOR_BAND
    print(f"\nмодель на всём коридоре ({len(grid)} cutoff'ов) -> ТЕСТ:")
    print(f"  mean(log1p(pred)) = {zt.mean():.4f}   якорный коридор {lo}..{hi}")
    print(f"  => delta_anchor = {lo - zt.mean():+.4f} .. {hi - zt.mean():+.4f}")
    np.save(f"artifacts/ztest_L{s.L}_norm{int(s.norm_long)}_tb{s.train_blocks}.npy", zt)


if __name__ == "__main__":
    main()
