"""SEQ-01 — диагностика по сохранённым OOF.

Отвечает на вопросы постановки, которые не покрывает `src/ptime_eval.py`:
пофолдовая таблица против табличных эталонов ТОЙ ЖЕ ёмкости, AUC активности
пофолдово (в первую очередь 2025-10-16), разнообразие относительно каждого члена
боевой смеси и относительно самой смеси, и всё это — против пола сидов
`Var(Δ)=0.00712`, `corr(остатки)=0.99885` (`exp_018`).

Табличный эталон здесь — `S1-ROUNDS` (те же 227 признаков, 300 раундов, argmin
кривой ёмкости `exp_017`), а не `S1-E10` (600 раундов = переобучение). Оба в
таблице, чтобы сравнение не зависело от выбора точки ёмкости.

Запуск: python research/strategies/results/SEQ/analyze.py [--exp SEQ-01-S42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from src.blend import aligned
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.validation import calibrate, rmsle_z

FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]
W = np.asarray(FOLD_WEIGHTS_S1, float)
W = W / W.sum()
MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}
SEED_FLOOR_VAR, SEED_FLOOR_CORR = 0.00712, 0.99885
OUT = Path(__file__).parent


def auc(pos, score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos.astype(np.int8), score))


def per_fold_cal(z, y, cut):
    """Пофолдовый оптимальный сдвиг — ровно то, что меряет wCV."""
    out = np.empty_like(z, dtype=float)
    for c in FOLDS:
        m = cut == c
        d, _ = calibrate(y[m], z[m])
        out[m] = np.maximum(z[m] + d, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="SEQ-01-S42")
    ap.add_argument("--tab", nargs="*", default=["S1-ROUNDS", "S1-E10", "S1-DIST"],
                    help="табличные эталоны на тех же фолдах")
    a = ap.parse_args()

    names = [a.exp] + a.tab + [e for e in MIX if e not in a.tab]
    names = list(dict.fromkeys(names))
    Z, y, cut = aligned(names)
    ly = np.log1p(y)
    pos = y > 0
    zmix = np.average(np.vstack([Z[names.index(e)] for e in MIX]), axis=0,
                      weights=[MIX[e] for e in MIX])
    Z = np.vstack([Z, zmix[None, :]])
    names = names + ["S1-DIST-MIX"]
    Zc = np.vstack([per_fold_cal(Z[i], y, cut) for i in range(len(names))])
    print(f"строк OOF {len(y):,}, моделей {len(names)}")

    # --- 1. пофолдовая таблица + wCV + AUC ------------------------------------
    rows = []
    base = names.index("S1-DIST-MIX")
    for i, e in enumerate(names):
        fc = [rmsle_z(y[cut == c], Zc[i, cut == c]) for c in FOLDS]
        au = [auc(pos[cut == c], Zc[i, cut == c]) for c in FOLDS]
        rows.append(dict(exp=e, wcv=float(np.dot(W, fc)), auc_w=float(np.dot(W, au)),
                         auc_pooled=auc(pos, Zc[i]),
                         **{f"rmsle_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, fc)},
                         **{f"auc_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, au)}))
    w0, a0 = rows[base]["wcv"], rows[base]["auc_w"]
    print(f"\n{'модель':>14} {'wCV':>9} {'Δ к смеси':>10} {'AUC(y>0)':>9} {'Δ':>9} "
          f"{'AUC 10-16':>10}   пофолдовый RMSLE_cal")
    for r in rows:
        print(f"{r['exp']:>14} {r['wcv']:>9.5f} {r['wcv'] - w0:>+10.5f} {r['auc_w']:>9.5f} "
              f"{r['auc_w'] - a0:>+9.5f} {r['auc_1016']:>10.5f}   "
              + " ".join(f"{r[f'rmsle_{c[5:7]}{c[8:10]}']:.5f}" for c in FOLDS))
    pl.DataFrame(rows).write_csv(OUT / "folds.csv")

    # --- 2. разнообразие ------------------------------------------------------
    i0 = names.index(a.exp)
    div = []
    for j, e in enumerate(names):
        if j == i0:
            continue
        d = Z[i0] - Z[j]
        div.append(dict(vs=e, var_delta=float(np.var(d)),
                        corr_pred=float(np.corrcoef(Z[i0], Z[j])[0, 1]),
                        corr_resid=float(np.corrcoef(ly - Z[i0], ly - Z[j])[0, 1]),
                        var_over_seed_floor=float(np.var(d) / SEED_FLOOR_VAR)))
    print(f"\nразнообразие {a.exp} против прочих "
          f"(пол сидов Var={SEED_FLOOR_VAR}, corr остатков={SEED_FLOOR_CORR}):")
    print(f"{'против':>14} {'Var(Δ)':>10} {'x пола':>8} {'corr предск.':>13} {'corr остатков':>14}")
    for d in div:
        print(f"{d['vs']:>14} {d['var_delta']:>10.5f} {d['var_over_seed_floor']:>8.2f} "
              f"{d['corr_pred']:>13.5f} {d['corr_resid']:>14.5f}")
    pl.DataFrame(div).write_csv(OUT / "diversity.csv")

    # --- 3. подстановка в слот S1-E10 при фиксированных весах -----------------
    print("\nподстановка в слот S1-E10 боевой смеси (веса 0.15/0.30/0.10/0.45 не подбираются):")
    sub = []
    for name, zc in [("S1-DIST-MIX (база)", None), (a.exp, Z[i0])]:
        Zx = np.vstack([Z[names.index(e)] for e in MIX])
        if zc is not None:
            Zx[list(MIX).index("S1-E10")] = zc
        z = np.average(Zx, axis=0, weights=[MIX[e] for e in MIX])
        fc = [calibrate(y[cut == c], z[cut == c])[1] for c in FOLDS]
        sub.append(dict(config=name, wcv=float(np.dot(W, fc)),
                        **{f"fold_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, fc)}))
    for r in sub:
        print(f"{r['config']:>20} wCV {r['wcv']:.6f} ({r['wcv'] - sub[0]['wcv']:+.6f})   "
              + " ".join(f"{v:.5f}" for k, v in r.items() if k.startswith("fold_")))
    pl.DataFrame(sub).write_csv(OUT / "fixed_mix.csv")

    # --- 4. простая аддитивная смесь TCN + смесь ------------------------------
    print("\nдоля TCN в смеси с S1-DIST-MIX (страховка S1-E03a внутри смеси сохранена):")
    add = []
    for w in np.arange(0.0, 0.55, 0.05):
        z = (1 - w) * zmix + w * Z[i0]
        fc = [calibrate(y[cut == c], z[cut == c])[1] for c in FOLDS]
        add.append(dict(w_seq=round(float(w), 2), wcv=float(np.dot(W, fc)),
                        better=int(sum(v < b - 1e-12 for v, b in
                                       zip(fc, [rows[base][f"rmsle_{c[5:7]}{c[8:10]}"]
                                                for c in FOLDS]))),
                        **{f"fold_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, fc)}))
    for r in add:
        print(f"  w={r['w_seq']:.2f}  wCV {r['wcv']:.6f} ({r['wcv'] - add[0]['wcv']:+.6f})  "
              f"фолдов лучше {r['better']}/4   "
              + " ".join(f"{v:.5f}" for k, v in r.items() if k.startswith("fold_")))
    pl.DataFrame(add).write_csv(OUT / "blend_curve.csv")
    print(f"\nзаписано: {OUT}/folds.csv, diversity.csv, fixed_mix.csv, blend_curve.csv")


if __name__ == "__main__":
    main()
