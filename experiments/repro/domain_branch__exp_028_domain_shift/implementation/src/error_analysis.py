"""Анализ ошибок: где именно решение выигрывает и где остаётся слабым.

Сравнивает OOF двух экспериментов по сегментам пользователей. Сегменты берутся
из признаков на том же cutoff'е, поэтому ничего из таргета в разбиение не попадает.

Запуск: python -m src.error_analysis --base S1-B0 --best S1-E02 --L-base 0 --L-best 0
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from src.data import load
from src.features import make_xy
from src.tracking import load_oof
from src.validation import rmsle_z


def seg_frame(exp: str, L: int | None, blocks: int = 3) -> pl.DataFrame:
    d = load_oof(exp)
    parts = []
    for c in np.unique(d["cutoff"]):
        m = d["cutoff"] == c
        X, _ = make_xy(dt.date.fromisoformat(str(c)), L, blocks)
        parts.append(pl.DataFrame({
            "user_id": d["user_id"][m], "cutoff": [str(c)] * int(m.sum()),
            "z": d["z"][m], "y": d["y"][m],
            "days_buy": X["w90_days_buy"].to_numpy().astype(np.float32),
            "rec_buy": X["rec_buy"].to_numpy().astype(np.float32),
            "w30_gmv": X["w30_gmv"].to_numpy().astype(np.float32),
            "days_present": X["w90_days_present"].to_numpy().astype(np.float32)}))
    return pl.concat(parts)


def buckets(v: np.ndarray, edges, labels):
    out = np.array([labels[-1]] * len(v), dtype=object)
    for i, e in enumerate(edges):
        out[v <= e] = labels[i]
        v = np.where(v <= e, np.inf, v)
    return out


def table(name, seg, y, zb, zg):
    eb = (np.log1p(y) - zb) ** 2
    eg = (np.log1p(y) - zg) ** 2
    tot_b, tot_g = eb.sum(), eg.sum()
    print(f"\n  {name}")
    print(f"    {'сегмент':>16} {'n':>8} {'доля':>6} {'RMSLE база':>11} {'RMSLE best':>11} "
          f"{'Δ':>9} {'доля MSE':>9} {'вклад в Δ':>10}")
    order = sorted(set(seg), key=lambda s: str(s))
    for s in order:
        m = seg == s
        rb, rg = np.sqrt(eb[m].mean()), np.sqrt(eg[m].mean())
        print(f"    {str(s):>16} {m.sum():>8,} {m.mean():>6.1%} {rb:>11.4f} {rg:>11.4f} "
              f"{rg - rb:>+9.4f} {eg[m].sum() / tot_g:>9.1%} "
              f"{(eg[m].sum() - eb[m].sum()) / (tot_g - tot_b) if tot_g != tot_b else 0:>10.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--best", required=True)
    ap.add_argument("--L-base", type=int, default=0)
    ap.add_argument("--L-best", type=int, default=180)
    a = ap.parse_args()
    load()
    fb = seg_frame(a.base, None if a.L_base <= 0 else a.L_base)
    fg = seg_frame(a.best, None if a.L_best <= 0 else a.L_best)
    j = fb.select(["user_id", "cutoff", "z", "y", "days_buy", "rec_buy", "w30_gmv"]).join(
        fg.select(["user_id", "cutoff", pl.col("z").alias("zg")]), on=["user_id", "cutoff"])
    print(f"сопоставлено {j.height:,} строк OOF")
    y = j["y"].to_numpy(); zb = j["z"].to_numpy(); zg = j["zg"].to_numpy()
    print(f"\n  RMSLE {a.base}={rmsle_z(y, zb):.5f}   {a.best}={rmsle_z(y, zg):.5f}   "
          f"Δ={rmsle_z(y, zg) - rmsle_z(y, zb):+.5f}")

    table("по исходу", np.where(y > 0, "y > 0", "y = 0"), y, zb, zg)
    db = j["days_buy"].to_numpy()
    table("по числу дней с покупкой за 90 дней",
          buckets(db.copy(), [0, 1, 3, 7, 15], ["0", "1", "2-3", "4-7", "8-15", "16+"]), y, zb, zg)
    rb = np.nan_to_num(j["rec_buy"].to_numpy(), nan=999)
    table("по свежести последней покупки, дней",
          buckets(rb.copy(), [7, 30, 90, 180], ["0-7", "8-30", "31-90", "91-180", "не покупал"]),
          y, zb, zg)
    table("по фолду", j["cutoff"].to_numpy(), y, zb, zg)

    print("\n  где решения расходятся сильнее всего (|zg - zb|):")
    d = np.abs(zg - zb)
    q = np.quantile(d, [0.5, 0.9, 0.99])
    for lab, lo, hi in [("медиана и ниже", -1, q[0]), ("p50-p90", q[0], q[1]),
                        ("p90-p99", q[1], q[2]), ("p99+", q[2], 1e9)]:
        m = (d > lo) & (d <= hi)
        print(f"    {lab:>16} n={m.sum():>8,}  RMSLE база={np.sqrt(((np.log1p(y[m]) - zb[m])**2).mean()):.4f}"
              f"  best={np.sqrt(((np.log1p(y[m]) - zg[m])**2).mean()):.4f}"
              f"  P(y>0)={float((y[m] > 0).mean()):.3f}")


if __name__ == "__main__":
    main()
