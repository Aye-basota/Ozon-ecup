"""Четыре графика диагностики. Запуск после `diagnose.py`.

    python research/rmsle_diagnostics/plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
DIAG, PLOTS = HERE / "diagnostics", HERE / "plots"
PLOTS.mkdir(exist_ok=True)
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=120)
    plt.close(fig)
    print(f"  -> plots/{name}")


def calibration():
    cur = pl.read_csv(DIAG / "calibration_curve.csv")
    cal = pl.read_csv(DIAG / "calibration_temporal_safe.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    p, a = cur["mean_pred_log"].to_numpy(), cur["mean_actual_log"].to_numpy()
    ax[0].plot([0, 6], [0, 6], "k--", lw=1, label="идеальная")
    ax[0].plot(p, a, "o-", color=C[0], label="смесь S1-DIST-MIX")
    for x, y, s in zip(p, a, cur["segment"]):
        ax[0].annotate(s, (x, y), fontsize=7, xytext=(3, -9), textcoords="offset points")
    ax[0].set(xlabel="mean предсказанного log1p", ylabel="mean фактического log1p",
              title="Калибровочная кривая по децилям прогноза\n(после пофолдового сдвига)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    b = cal.drop_nulls("d_affine")
    x = np.arange(b.height)
    ax[1].bar(x - .18, b["d_affine"], .36, label="аффинная a·z+b", color=C[0])
    ax[1].bar(x + .18, b["d_seg4"], .36, label="4 сегмента, свой сдвиг", color=C[1])
    ax[1].axhline(-0.0005, color="k", ls=":", lw=1)
    ax[1].annotate("порог различимости −0.0005 (exp_016 §6)", (-.4, -0.00052), fontsize=7,
                   va="top")
    ax[1].set_xticks(x, [f[5:] for f in b["fold"]])
    ax[1].set(ylabel="Δ RMSLE к текущей схеме (сдвиг)",
              title="Выигрыш перекалибровки, обученной\nТОЛЬКО на прошлых фолдах")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")
    save(fig, "calibration.png")


ORDER = {
    "lifecycle": ["0 дней покупок", "1", "2-3", "4-7", "8-15", "16-30", "31+"],
    "gmv_bucket": ["y=0", "0-500", "500-2k", "2k-5k", "5k-15k", "15k-50k", "50k+"],
    "pred_decile": [f"D{i:02d}" for i in range(1, 11)],
}


def ordered(d: pl.DataFrame, seg: str) -> pl.DataFrame:
    s = d.filter(pl.col("segmentation") == seg)
    k = {v: i for i, v in enumerate(ORDER[seg])}
    return s.with_columns(k=pl.col("segment").replace_strict(k, default=99)).sort("k")


def error_share():
    d = pl.read_csv(DIAG / "error_decomposition.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, (seg, title) in enumerate([
            ("lifecycle", "дней с покупкой за 180 дней"),
            ("gmv_bucket", "фактический GMV следующих 30 дней"),
            ("pred_decile", "дециль прогноза")]):
        s = ordered(d, seg)
        x = np.arange(s.height)
        ax[k].bar(x - .2, s["users_pct"], .4, label="доля пользователей", color=C[0])
        ax[k].bar(x + .2, s["error_share"], .4, label="доля ошибки", color=C[3])
        ax[k].set_xticks(x, s["segment"], rotation=45, ha="right", fontsize=8)
        ax[k].set(title=title)
        ax[k].grid(alpha=.3, axis="y")
        if k == 0:
            ax[k].set_ylabel("доля (веса фолдов 1:2:4:8)")
            ax[k].legend(fontsize=8)
    fig.suptitle("Кто создаёт квадратичную лог-ошибку: доля людей против доли ошибки",
                 fontsize=11)
    save(fig, "error_share.png")


def recency():
    d = pl.read_csv(DIAG / "error_decomposition.csv").filter(
        pl.col("segmentation") == "recency")
    order = ["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181+",
             "никогда не покупал"]
    d = d.with_columns(k=pl.col("segment").replace_strict(
        {s: i for i, s in enumerate(order)})).sort("k")
    x = np.arange(d.height)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(x - .2, d["users_pct"], .4, label="доля пользователей", color=C[0])
    ax[0].bar(x + .2, d["error_share"], .4, label="доля ошибки", color=C[3])
    ax[0].set_xticks(x, d["segment"], rotation=45, ha="right", fontsize=8)
    ax[0].set(ylabel="доля", title="Свежесть последней покупки: люди против ошибки")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")

    ax[1].plot(x, d["rmsle"], "o-", color=C[0], label="RMSLE сегмента")
    ax[1].set_xticks(x, d["segment"], rotation=45, ha="right", fontsize=8)
    ax[1].set(ylabel="RMSLE", title="RMSLE и остаточное смещение по свежести")
    a2 = ax[1].twinx()
    a2.bar(x, d["log_bias"], .5, alpha=.35, color=C[1], label="log bias")
    a2.axhline(0, color="k", lw=.8)
    a2.set_ylabel("log bias (факт − прогноз)")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8, loc="upper right")
    a2.legend(fontsize=8, loc="lower right")
    save(fig, "recency.png")


def fold_stability():
    s = pl.read_csv(DIAG / "fold_stability.csv")
    t = pl.read_csv(DIAG / "test_shift.csv")
    f = pl.read_parquet(HERE / "fold_predictions.parquet")
    te = pl.read_parquet(HERE / "test_predictions.parquet")
    x = np.arange(s.height)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    ax[0].plot(x, s["rmsle_cal"], "o-", color=C[0], label="RMSLE (калибр.)")
    ax[0].plot(x, s["rmsle_raw"], "s--", color=C[1], alpha=.7, label="RMSLE (сырой)")
    ax[0].set_xticks(x, [c[5:] for c in s["fold"]])
    ax[0].set(ylabel="RMSLE", title="Скор по фолдам:\nмонотонно лучше ближе к тесту")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].plot(x, s["zero_rate"], "o-", color=C[3], label="доля y = 0")
    ax[1].plot(x, s["auc_zero"], "s-", color=C[2], label="AUC(y > 0)")
    ax[1].plot(x, s["spread_ratio"], "^-", color=C[0], label="std(z) / std(log1p y)")
    ax[1].set_xticks(x, [c[5:] for c in s["fold"]])
    ax[1].set(title="Устойчивость популяции и модели")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    bins = np.linspace(0, 8, 81)
    for i, c in enumerate(FOLDS):
        z = f.filter(pl.col("cutoff") == c)["z_cal"].to_numpy()
        ax[2].hist(z, bins=bins, density=True, histtype="step", color=C[i], lw=1,
                   alpha=.8, label=f"OOF {c[5:]}")
    ax[2].hist(te["z_cal"].to_numpy(), bins=bins, density=True, histtype="stepfilled",
               color="k", alpha=.25, label="тест 2026-02-13")
    ax[2].set(xlabel="log1p(prediction)", ylabel="плотность",
              title="Распределение прогноза:\nOOF против теста (уровень 2.3293)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    save(fig, "fold_stability.png")


if __name__ == "__main__":
    calibration()
    error_share()
    recency()
    fold_stability()
