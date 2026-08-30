"""Постпроцессинг: глобальная и посегментная лог-калибровка.

Метрика — MSE по z = log1p(y), поэтому сдвиг всех прогнозов на константу delta
меняет скор как sqrt(sigma^2 + (b - delta)^2). Оптимальный delta равен среднему
остатку. Локально выигрыш выглядит мизерным, потому что зависимость квадратичная:
при b = 0.09 прибавка 0.002, при b = 0.30 — уже 0.025 (eda_findings §7.4).

ВАЖНО: параметры калибровки подбираются ТОЛЬКО на OOF-предсказаниях,
никогда на тесте и никогда по leaderboard.

Запуск: python -m src.calibrate --exp S1-E02
"""
from __future__ import annotations

import argparse

import numpy as np

from src.tracking import load_oof
from src.validation import best_offset, rmsle_z


def global_offset(y, z):
    """Оптимальный глобальный сдвиг = средний остаток в лог-пространстве."""
    return float(np.log1p(y).mean() - z.mean())


def segment_offsets(y, z, seg, min_n=5000):
    """Оптимальный сдвиг по сегментам (усадка к глобальному при малом n)."""
    g = global_offset(y, z)
    out = {}
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < min_n:
            out[s] = g
            continue
        d = float(np.log1p(y[m]).mean() - z[m].mean())
        w = m.sum() / (m.sum() + min_n)          # усадка: маленькие сегменты тянутся к глобальному
        out[s] = w * d + (1 - w) * g
    return out


def apply_segment(z, seg, offs):
    out = z.copy()
    for s, d in offs.items():
        out[seg == s] = np.maximum(out[seg == s] + d, 0.0)
    return out


def buy_bucket(days_buy: np.ndarray) -> np.ndarray:
    """Бакеты по частоте покупок — сегментация из eda_findings §7.3."""
    b = np.zeros(len(days_buy), dtype=np.int8)
    b[days_buy == 1] = 1
    b[(days_buy >= 2) & (days_buy <= 3)] = 2
    b[(days_buy >= 4) & (days_buy <= 7)] = 3
    b[(days_buy >= 8) & (days_buy <= 15)] = 4
    b[days_buy >= 16] = 5
    return b


def report(exp_id: str):
    d = load_oof(exp_id)
    y, z, cut = d["y"], d["z"], d["cutoff"]
    print(f"OOF {exp_id}: n={len(y):,}  RMSLE={rmsle_z(y, z):.5f}")
    print(f"  mean log1p(y)={np.log1p(y).mean():.4f}  mean z={z.mean():.4f}  "
          f"bias={global_offset(y, z):+.4f}")
    o, sc = best_offset(y, z)
    print(f"  глобальный сдвиг {o:+.4f} -> RMSLE {sc:.5f}  (выигрыш {rmsle_z(y, z) - sc:+.5f})")
    print("\n  по фолдам:")
    for c in np.unique(cut):
        m = cut == c
        oo, ss = best_offset(y[m], z[m])
        print(f"    {c}: n={m.sum():>7,} RMSLE={rmsle_z(y[m], z[m]):.5f} "
              f"bias={global_offset(y[m], z[m]):+.4f} off*={oo:+.3f} -> {ss:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    report(ap.parse_args().exp)
