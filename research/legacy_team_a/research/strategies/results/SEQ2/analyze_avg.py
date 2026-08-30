"""SEQ-02 — что даёт усреднение сидов и сколько разнообразия после него остаётся.

`exp_025` намерил у TCN собственный шум сида `Var(z42 - z43) = 0.02270` — втрое
выше пола сидов GBDT. Отсюда главный вопрос эксперимента: рекордное разнообразие
`SEQ-01` (`Var(z - z_mix) = 0.03749`, 5.27x пола) — это НОВАЯ ФУНКЦИЯ или просто
шум обучения, который усреднением сидов исчезнет?

Вопрос решается разложением, а не мнением. Если сид даёт `z_i = f + e_i` с
независимыми `e_i` дисперсии `s2`, то для среднего `k` сидов

    V_k = Var(z_(k) - mix) = D + s2 / k,    D = Var(f - mix)

где `D` — устойчивая часть расхождения со смесью (новая функция), а `s2/k` —
стохастическая. Двух точек (k=1 и k=2) хватает, чтобы решить систему:

    s2 = 2 (V_1 - V_2),   D = 2 V_2 - V_1

а третья точка (k=3) эту модель ПРОВЕРЯЕТ: предсказание `D + s2/3` обязано
совпасть с замером. Независимая оценка того же `s2` берётся из попарного
`Var(z_i - z_j) = 2 s2` — две оценки обязаны сойтись.

Запуск: PYTHONPATH=. python research/strategies/results/SEQ2/analyze_avg.py
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

from src.blend import aligned
from src.config import FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.validation import calibrate, rmsle_z

FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]
W = np.asarray(FOLD_WEIGHTS_S1, float)
W = W / W.sum()
MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}
SEED_FLOOR_VAR, SEED_FLOOR_CORR = 0.00712, 0.99885     # exp_018, пол разнообразия GBDT
OUT = Path(__file__).parent


def auc(pos, score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos.astype(np.int8), score))


def per_fold_cal(z, y, cut):
    out = np.empty_like(z, dtype=float)
    for c in FOLDS:
        m = cut == c
        d, _ = calibrate(y[m], z[m])
        out[m] = np.maximum(z[m] + d, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["SEQ-01-S42", "SEQ-01-S43"])
    ap.add_argument("--avgs", nargs="*", default=["SEQ-AVG2"])
    a = ap.parse_args()

    names = list(dict.fromkeys(a.seeds + a.avgs + list(MIX)))
    Z, y, cut = aligned(names)
    ly, pos = np.log1p(y), y > 0
    zmix = np.average(np.vstack([Z[names.index(e)] for e in MIX]), axis=0,
                      weights=list(MIX.values()))
    Z = np.vstack([Z, zmix[None, :]])
    names = names + ["S1-DIST-MIX"]
    Zc = np.vstack([per_fold_cal(Z[i], y, cut) for i in range(len(names))])
    print(f"строк OOF {len(y):,}")

    # --- 1. standalone: сиды по отдельности, их средние, боевая смесь ---------
    shown = a.seeds + a.avgs + ["S1-DIST-MIX"]
    base = names.index("S1-DIST-MIX")
    rows = []
    for e in shown:
        i = names.index(e)
        fc = [rmsle_z(y[cut == c], Zc[i, cut == c]) for c in FOLDS]
        rows.append(dict(exp=e, wcv=float(np.dot(W, fc)), auc_w=float(np.dot(
            W, [auc(pos[cut == c], Zc[i, cut == c]) for c in FOLDS])),
            **{f"f_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, fc)}))
    w0 = [r for r in rows if r["exp"] == "S1-DIST-MIX"][0]
    print(f"\n{'модель':>14} {'wCV':>9} {'Δ к смеси':>10} {'AUC(y>0)':>9} {'Δ AUC':>9}   пофолдово")
    for r in rows:
        print(f"{r['exp']:>14} {r['wcv']:>9.5f} {r['wcv'] - w0['wcv']:>+10.5f} "
              f"{r['auc_w']:>9.5f} {r['auc_w'] - w0['auc_w']:>+9.5f}   "
              + " ".join(f"{r[f'f_{c[5:7]}{c[8:10]}']:.5f}" for c in FOLDS))
    pl.DataFrame(rows).write_csv(OUT / "standalone.csv")

    # --- 2. попарная дисперсия сидов: Var(z_i - z_j) = 2 s2 -------------------
    print("\nпопарное расхождение сидов (ожидается 2·s2):")
    pair = []
    for e1, e2 in itertools.combinations(a.seeds, 2):
        d = Z[names.index(e1)] - Z[names.index(e2)]
        per = {c: float(np.var(d[cut == c])) for c in FOLDS}
        pair.append(dict(a=e1, b=e2, var=float(np.var(d)), s2_from_pair=float(np.var(d)) / 2,
                         corr=float(np.corrcoef(Z[names.index(e1)], Z[names.index(e2)])[0, 1]),
                         **{f"f_{c[5:7]}{c[8:10]}": v for c, v in per.items()}))
        print(f"  {e1} vs {e2}: Var(Δ) {pair[-1]['var']:.5f} -> s2 = {pair[-1]['s2_from_pair']:.5f}"
              f", corr {pair[-1]['corr']:.5f}, пофолдово "
              + " ".join(f"{v:.5f}" for v in per.values()))
    pl.DataFrame(pair).write_csv(OUT / "seed_pairs.csv")

    # --- 3. разложение разнообразия: сколько шума, сколько новой функции ------
    print(f"\nразнообразие против боевой смеси (пол сидов GBDT Var={SEED_FLOOR_VAR}, "
          f"corr остатков={SEED_FLOOR_CORR}):")
    div, vk = [], {}
    for e in shown[:-1]:
        i = names.index(e)
        d = Z[i] - zmix
        k = int(e.replace("SEQ-AVG", "")) if e.startswith("SEQ-AVG") else 1
        v = float(np.var(d))
        vk.setdefault(k, []).append(v)      # k=1 усредняется по всем одиночным сидам
        div.append(dict(exp=e, k_seeds=k, var_delta=v, x_floor=v / SEED_FLOOR_VAR,
                        corr_pred=float(np.corrcoef(Z[i], zmix)[0, 1]),
                        corr_resid=float(np.corrcoef(ly - Z[i], ly - zmix)[0, 1])))
        print(f"  {e:>14} k={k}  Var(Δ) {v:.5f}  ({v / SEED_FLOOR_VAR:5.2f}x пола)  "
              f"corr предск. {div[-1]['corr_pred']:.5f}  corr остатков {div[-1]['corr_resid']:.5f}")

    vk = {k: float(np.mean(v)) for k, v in vk.items()}
    if 1 in vk and 2 in vk:
        v1, v2 = vk[1], vk[2]
        s2, D = 2 * (v1 - v2), 2 * v2 - v1
        print(f"\nразложение V_k = D + s2/k по точкам k=1,2:")
        print(f"  стохастическая часть s2 = {s2:.5f}  "
              f"(независимая оценка из пары сидов: {pair[0]['s2_from_pair']:.5f})")
        print(f"  устойчивая часть      D = {D:.5f}  = {D / SEED_FLOOR_VAR:.2f}x пола сидов GBDT")
        print(f"  доля шума в разнообразии SEQ-01: {s2 / v1:.1%}, "
              f"остаётся новой функцией: {D / v1:.1%}")
        for k in sorted(vk):
            print(f"  k={k}: замер {vk[k]:.5f}   модель {D + s2 / k:.5f}"
                  + ("   <- проверка модели" if k >= 3 else ""))
        pl.DataFrame([dict(s2=s2, D=D, s2_from_pair=pair[0]["s2_from_pair"],
                           D_over_floor=D / SEED_FLOOR_VAR, noise_share=s2 / v1,
                           **{f"V_{k}": vk[k] for k in sorted(vk)})]).write_csv(
            OUT / "decomposition.csv")
    pl.DataFrame(div).write_csv(OUT / "diversity_avg.csv")   # diversity.csv занят ptime_eval

    # --- 4. кривая доли SEQ в смеси, для каждого варианта ---------------------
    print("\nдоля SEQ в смеси с S1-DIST-MIX (страховка S1-E03a внутри смеси сохранена):")
    base_fc = [rmsle_z(y[cut == c], Zc[base, cut == c]) for c in FOLDS]
    curve = []
    for e in shown[:-1]:
        i = names.index(e)
        best = None
        for w in np.arange(0.0, 0.55, 0.05):
            z = (1 - w) * zmix + w * Z[i]
            fc = [calibrate(y[cut == c], z[cut == c])[1] for c in FOLDS]
            r = dict(exp=e, w_seq=round(float(w), 2), wcv=float(np.dot(W, fc)),
                     better=int(sum(v < b - 1e-12 for v, b in zip(fc, base_fc))),
                     **{f"f_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, fc)})
            curve.append(r)
            if best is None or r["wcv"] < best["wcv"]:
                best = r
        print(f"  {e:>14}: лучшая доля {best['w_seq']:.2f} -> wCV {best['wcv']:.6f} "
              f"({best['wcv'] - np.dot(W, base_fc):+.6f}), фолдов лучше {best['better']}/4")
    pl.DataFrame(curve).write_csv(OUT / "blend_curve.csv")
    print(f"\nзаписано: {OUT}/standalone.csv, seed_pairs.csv, diversity_avg.csv, "
          f"decomposition.csv, blend_curve.csv")


if __name__ == "__main__":
    main()
