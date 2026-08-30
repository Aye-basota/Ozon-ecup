"""ETX2 — сводка соло-качества членов (EXP-037, этап 1).

Таблица в тех же координатах, что `exp_036` §«Полная схема S1»: пофолдовый
КАЛИБРОВАННЫЙ RMSLE, wCV с весами 1:2:4:8, AUC(1[y>0]) на склеенном OOF, и три
величины «на кого похож»: `Var(z - z_SEQ-AVG3)` (расстояние до того, КОГО
дополняем — по `exp_036` это и есть правильная ось), `Var(z - z_tab)` и
корреляция остатков с табличной частью.

Запуск: PYTHONPATH=. python research/strategies/results/ETX2/summary.py
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from src.blend import aligned, fold_masks
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1
from src.validation import calibrate, rmsle_z

OUT = "research/strategies/results/ETX2"
CAP, E02, DIST = "S1-E03a", "S1-E02", "S1-DIST"
WANT = ["SEQ-01-S42", "SEQ-AVG3", "SEQ-D3A-AVG3", "SEQ-D3A-S42",
        "ETX-01-S42", "ETX-01-S43", "ETX-01-S44", "ETX-AVG2", "ETX-AVG3"]
REF = "SEQ-AVG3"


def auc(pos: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos.astype(np.int8), score))


def main() -> None:
    have = [n for n in WANT if (ARTIFACTS / f"oof_{n}.npz").exists()]
    miss = [n for n in WANT if n not in have]
    if miss:
        print(f"нет OOF: {' '.join(miss)}")
    base = sorted(set(have + [CAP, E02, DIST]))
    Z, y, cut = aligned(base)
    idx = {e: i for i, e in enumerate(base)}
    ly, pos = np.log1p(y), y > 0
    folds, masks = fold_masks(cut)
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()
    z_tab = np.average(Z[[idx[CAP], idx[E02], idx[DIST]]], axis=0,
                       weights=[0.10, 0.20, 0.25])
    zref = Z[idx[REF]] if REF in idx else None

    rows = []
    print(f"\n{'модель':<16}" + "".join(f"{f[5:]:>10}" for f in folds)
          + f"{'wCV':>10}{'AUC':>9}{'V-SEQ':>9}{'V-tab':>9}{'cor-res':>9}")
    for n in have:
        z = Z[idx[n]]
        fc = []
        for m in masks:
            d, _ = calibrate(y[m], z[m])
            fc.append(rmsle_z(y[m], np.maximum(z[m] + d, 0.0)))
        fc = np.array(fc)
        d_all, _ = calibrate(y, z)
        a = auc(pos, np.maximum(z + d_all, 0.0))
        vseq = float(np.var(z - zref)) if zref is not None else float("nan")
        vtab = float(np.var(z - z_tab))
        cr = float(np.corrcoef(ly - z, ly - z_tab)[0, 1])
        print(f"{n:<16}" + "".join(f"{v:>10.5f}" for v in fc)
              + f"{float(w_f @ fc):>10.5f}{a:>9.5f}{vseq:>9.5f}{vtab:>9.5f}{cr:>9.5f}")
        rows.append(dict(model=n, wcv=float(w_f @ fc), auc=a,
                         folds={f: float(v) for f, v in zip(folds, fc)},
                         var_vs_seq=vseq, var_vs_tab=vtab, corr_resid_tab=cr))

    if {"ETX-01-S42", "ETX-01-S43", "ETX-01-S44"} <= set(have):
        print("\nпопарное расхождение сидов ETX (OOF, все фолды):")
        for i, a_ in enumerate(["ETX-01-S42", "ETX-01-S43", "ETX-01-S44"]):
            for b_ in ["ETX-01-S42", "ETX-01-S43", "ETX-01-S44"][i + 1:]:
                print(f"  {a_} - {b_}: Var = {np.var(Z[idx[a_]] - Z[idx[b_]]):.5f}, "
                      f"corr = {np.corrcoef(Z[idx[a_]], Z[idx[b_]])[0, 1]:.5f}")
        w = np.array([w_f @ np.array([rmsle_z(y[m], np.maximum(
            Z[idx[s]][m] + calibrate(y[m], Z[idx[s]][m])[0], 0.0)) for m in masks])
            for s in ["ETX-01-S42", "ETX-01-S43", "ETX-01-S44"]])
        print(f"  wCV по сидам: {np.round(w, 5).tolist()}  sd = {w.std(ddof=1):.5f}")

    with open(f"{OUT}/summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["model"] + folds + ["wcv", "auc", "var_vs_SEQ-AVG3", "var_vs_tab",
                                         "corr_resid_tab"])
        for r in rows:
            wr.writerow([r["model"]] + [f"{r['folds'][f]:.5f}" for f in folds]
                        + [f"{r['wcv']:.5f}", f"{r['auc']:.5f}", f"{r['var_vs_seq']:.5f}",
                           f"{r['var_vs_tab']:.5f}", f"{r['corr_resid_tab']:.5f}"])
    print(f"\nзаписано: {OUT}/summary.csv")


if __name__ == "__main__":
    main()
