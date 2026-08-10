"""Якорная калибровка уровня и её ЧЕСТНАЯ проверка на исторических фолдах.

Идея. Для RMSLE-оптимального прогноза выполняется тождество
`mean_i log1p(pred_i) = E[log1p(y)]` (eda_findings §2.3). На тестовом cutoff'е
величина `m_x = mean log1p(gmv за последние 30 дней) = 2.2421` известна ТОЧНО
(из `sample_submit`), поэтому

    целевой уровень = m_x(T) + dm_hat,     dm_hat — оценка сдвига уровня за 30 дней
    delta           = целевой уровень - mean(z модели)

Это одна поправка, которая автоматически поглощает и out-of-time дрейф модели,
и сезонную фазу — их не нужно (и нельзя) складывать по отдельности, иначе они
частично посчитаются дважды.

Проверка процедуры (главное здесь): на каждом историческом фолде `dm_hat` берётся
как медиана dm по ДРУГИМ cutoff'ам (leave-fold-out), после чего измеряется,
приближает ли такая калибровка к оракульному сдвигу. Ничего из фолда не
используется при выборе поправки.

Запуск: python -m src.anchor --exp S1-E03a
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from src.config import ANCHOR_M_X, CUTOFF_TEST, HISTORY_L, cutoff_grid
from src.data import load
from src.features import make_xy, panel_users, target
from src.tracking import load_oof
from src.validation import best_offset, rmsle_z


def level_stats(T: dt.date, blocks: int, L: int | None = HISTORY_L):
    """(m_x, m_y, dm) на панели `blocks` для cutoff'а T."""
    X, y = make_xy(T, L, blocks)
    mx = float(np.log1p(X["w30_gmv"].to_numpy()).mean())
    my = float(np.log1p(y).mean())
    return mx, my, my - mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="ID эксперимента с OOF для бэктеста калибровки")
    ap.add_argument("--L", type=int, default=HISTORY_L)
    ap.add_argument("--step", type=int, default=7)
    a = ap.parse_args()
    L = None if a.L <= 0 else a.L
    load()

    grid = cutoff_grid(90, a.step)
    print("=" * 100)
    print("A. dm на ТЕСТОВОЙ (3-блочной) панели против наивной 1-блочной")
    print("=" * 100)
    print(f"{'cutoff':>12} {'m_x(3бл)':>10} {'m_y(3бл)':>10} {'dm(3бл)':>9} {'dm(1бл)':>9} {'разница':>9}")
    d3, d1 = [], []
    for T in grid:
        mx3, my3, dm3 = level_stats(T, 3, L)
        mx1, my1, dm1 = level_stats(T, 1, L)
        d3.append(dm3); d1.append(dm1)
        print(f"{str(T):>12} {mx3:>10.4f} {my3:>10.4f} {dm3:>+9.4f} {dm1:>+9.4f} {dm3 - dm1:>+9.4f}")
    d3, d1 = np.array(d3), np.array(d1)
    print(f"\n  3-блочная: mean={d3.mean():+.4f} median={np.median(d3):+.4f} std={d3.std():.4f}")
    print(f"  1-блочная: mean={d1.mean():+.4f} median={np.median(d1):+.4f} std={d1.std():.4f}")
    print(f"  систематическая разница панелей: {(d3 - d1).mean():+.4f}  "
          f"(нужна, чтобы перенести YoY-оценку с 1-блочной панели на тестовую)")

    mxT = float(np.log1p(make_xy(CUTOFF_TEST, L, 3, with_target=False)[0]["w30_gmv"].to_numpy()).mean())
    print(f"\n  m_x на ТЕСТЕ = {mxT:.4f}  (якорь из sample_submit = {ANCHOR_M_X})")
    print(f"  сценарии целевого уровня E[log1p(y_test)]:")
    for nm, dmv in [("медианный дрейф коридора", float(np.median(d3))),
                    ("средний дрейф коридора", float(d3.mean())),
                    ("YoY-аналог (сезонный максимум)", 0.1659 + (d3 - d1).mean()),
                    ("половина сезонной поправки",
                     float(np.median(d3)) + (0.1659 + (d3 - d1).mean() - float(np.median(d3))) / 2)]:
        print(f"    {nm:32s} dm={dmv:+.4f}  ->  уровень {mxT + dmv:.4f}")

    if not a.exp:
        return

    print("\n" + "=" * 100)
    print(f"B. Бэктест якорной калибровки на OOF эксперимента {a.exp}")
    print("=" * 100)
    d = load_oof(a.exp)
    y, z, cut = d["y"], d["z"], d["cutoff"]
    dm_by_cut = {str(T): dm for T, dm in zip(grid, d3)}
    print(f"{'фолд':>12} {'m_x':>8} {'dm_hat':>8} {'цель':>8} {'mean z':>8} "
          f"{'delta':>8} {'RMSLE':>9} {'после':>9} {'оракул':>9}")
    tot_raw, tot_cal, tot_or, n = [], [], [], []
    for c in np.unique(cut):
        m = cut == c
        T = dt.date.fromisoformat(str(c))
        mx, _, _ = level_stats(T, 3, L)
        # leave-fold-out: dm_hat — медиана по ВСЕМ ОСТАЛЬНЫМ cutoff'ам коридора
        dm_hat = float(np.median([v for k, v in dm_by_cut.items() if k != str(c)]))
        tgt = mx + dm_hat
        delta = tgt - float(z[m].mean())
        zc = np.maximum(z[m] + delta, 0.0)
        o, sc_o = best_offset(y[m], z[m])
        r0, r1 = rmsle_z(y[m], z[m]), rmsle_z(y[m], zc)
        tot_raw.append(r0); tot_cal.append(r1); tot_or.append(sc_o); n.append(m.sum())
        print(f"{str(c):>12} {mx:>8.4f} {dm_hat:>+8.4f} {tgt:>8.4f} {float(z[m].mean()):>8.4f} "
              f"{delta:>+8.4f} {r0:>9.5f} {r1:>9.5f} {sc_o:>9.5f}")
    print(f"\n  CV без калибровки        {np.mean(tot_raw):.5f}")
    print(f"  CV с якорной калибровкой {np.mean(tot_cal):.5f}   "
          f"({np.mean(tot_cal) - np.mean(tot_raw):+.5f})")
    print(f"  CV с оракульным сдвигом  {np.mean(tot_or):.5f}   "
          f"(верхняя граница, недостижима)")
    got = (np.mean(tot_raw) - np.mean(tot_cal)) / max(np.mean(tot_raw) - np.mean(tot_or), 1e-9)
    print(f"  доля оракульного выигрыша, которую забирает якорь: {got:.1%}")


if __name__ == "__main__":
    main()
