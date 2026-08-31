"""EXP-030 `SEQ-D3A` — сводка по всем четырём фолдам одного сида.

Считает ровно те числа, которыми принимается решение по постановке EXP-030:

  1. калиброванный RMSLE по каждому фолду и Δ к BASE;
  2. `wCV` 1:2:4:8 и ΔwCV (главная метрика проекта, `exp_016`);
  3. `AUC(y>0)` пофолдово и на склеенном OOF;
  4. сегменты `rec_buy 15–60` и `w180_days_buy 2–15` (и их пересечение);
  5. `Var(z_D3A − z_BASE)` против пола сидов 0.00712 и пары сидов TCN;
  6. корреляция предсказаний и остатков.

Калибровка — пофолдовая, как в `wCV`; внутри сегмента НЕ пересчитывается
(`rmsle_diagnostics` §3: посегментная перекалибровка закрыта измерением).
Границы сегментов и источник признаков — те же, что в `SEQ4/diag.py` и
`src/ptime_eval.py`, чтобы числа сравнивались с уже опубликованными.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ5/analyze.py
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS, DATA_PROCESSED, VAL_FOLDS_S1
from src.merge_oof import auc_positive
from src.report import evaluate, format_report
from src.tracking import load_oof
from src.validation import calibrate, rmsle_z, wcv

SEG_COLS = ["rec_buy", "w180_days_buy"]
KEY_SEGMENTS = ["ВСЕ", "rec_buy 15-60", "w180_days_buy 2-15", "пересечение"]
SEED_FLOOR = 0.00712            # пол разнообразия проекта (exp_018)


def segments(df: pl.DataFrame) -> dict[str, np.ndarray]:
    rb, nb = df["rec_buy"].to_numpy(), df["w180_days_buy"].to_numpy()
    known = ~np.isnan(rb)
    rec = known & (rb >= 15) & (rb <= 60)
    freq = (nb >= 2) & (nb <= 15)
    return {
        "ВСЕ": np.ones(len(rb), bool),
        "rec_buy 15-60": rec,
        "w180_days_buy 2-15": freq,
        "пересечение": rec & freq,
        "rec_buy 0-14": known & (rb <= 14),
        "rec_buy 61-180": known & (rb >= 61) & (rb <= 180),
        "rec_buy 180+": known & (rb > 180),
        "никогда не покупал": ~known,
        "w180_days_buy 0-1": nb <= 1,
        "w180_days_buy 16+": nb >= 16,
    }


def auc(pos: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(pos):
        return float("nan")
    return float(roc_auc_score(pos.astype(np.int8), score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="SEQ-D3A-BASE-S42")
    ap.add_argument("--exp", default="SEQ-D3A-S42")
    ap.add_argument("--folds", nargs="*", default=[d.isoformat() for d in VAL_FOLDS_S1])
    ap.add_argument("--out", default="research/strategies/results/SEQ5")
    a = ap.parse_args()

    folds = [dt.date.fromisoformat(f) for f in a.folds]
    ready = [V for V in folds
             if all((ARTIFACTS / f"oof_{e}-V{V.strftime('%m%d')}.npz").exists()
                    for e in (a.base, a.exp))]
    if len(ready) < len(folds):
        print(f"пока посчитаны пары только на {len(ready)}/{len(folds)} фолдах: "
              f"{', '.join(V.isoformat() for V in ready)}")
    assert ready, "ни одной готовой пары BASE/вариант"

    rows, srows, pool = [], [], {a.base: [], a.exp: []}
    pool_y, pool_cut = [], []
    for V in ready:
        tag = f"V{V.strftime('%m%d')}"
        d = {e: load_oof(f"{e}-{tag}") for e in (a.base, a.exp)}
        uid = np.asarray(d[a.base]["user_id"])
        y = np.asarray(d[a.base]["y"], float)
        assert np.array_equal(np.asarray(d[a.exp]["user_id"]), uid), f"{tag}: другой набор строк"
        assert np.allclose(d[a.exp]["y"], y), f"{tag}: другой таргет"
        ly, pos = np.log1p(y), y > 0

        z_raw = {e: np.asarray(d[e]["z"], float) for e in (a.base, a.exp)}
        z_cal, off = {}, {}
        for e in (a.base, a.exp):
            k, _ = calibrate(y, z_raw[e])
            off[e], z_cal[e] = k, np.maximum(z_raw[e] + k, 0.0)
            pool[e].append(z_raw[e])
        pool_y.append(y)
        pool_cut.append(np.full(len(uid), V.isoformat()))

        f = pl.read_parquet(DATA_PROCESSED / f"feat_{V.strftime('%Y%m%d')}_LNone.parquet",
                            columns=["user_id"] + SEG_COLS)
        f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
        assert f.height == len(uid)
        segs = segments(f)

        rb = ly - z_cal[a.base]
        for e in (a.base, a.exp):
            r = ly - z_cal[e]
            rows.append(dict(
                fold=V.isoformat(), exp=e, n=len(uid), offset=off[e],
                rmsle=rmsle_z(y, z_raw[e]), rmsle_cal=rmsle_z(y, z_cal[e]),
                d_rmsle_cal=rmsle_z(y, z_cal[e]) - rmsle_z(y, z_cal[a.base]),
                auc=auc(pos, z_cal[e]), d_auc=auc(pos, z_cal[e]) - auc(pos, z_cal[a.base]),
                var_vs_base=float(np.var(z_raw[e] - z_raw[a.base])),
                var_cal_vs_base=float(np.var(z_cal[e] - z_cal[a.base])),
                corr_pred=1.0 if e == a.base else float(np.corrcoef(z_cal[e], z_cal[a.base])[0, 1]),
                corr_resid=1.0 if e == a.base else float(np.corrcoef(r, rb)[0, 1]),
                mean_z=float(z_raw[e].mean()), std_z=float(z_raw[e].std())))
        for name, m in segs.items():
            base_r = rmsle_z(y[m], z_cal[a.base][m])
            base_a = auc(pos[m], z_cal[a.base][m])
            for e in (a.base, a.exp):
                srows.append(dict(fold=V.isoformat(), segment=name, exp=e, n=int(m.sum()),
                                  share=float(m.mean()),
                                  rmsle=rmsle_z(y[m], z_cal[e][m]),
                                  d_rmsle=rmsle_z(y[m], z_cal[e][m]) - base_r,
                                  auc=auc(pos[m], z_cal[e][m]),
                                  d_auc=auc(pos[m], z_cal[e][m]) - base_a,
                                  mse_share=float(((ly - z_cal[e]) ** 2)[m].sum()
                                                  / ((ly - z_cal[e]) ** 2).sum())))

    fold_df = pl.DataFrame(rows)
    seg_df = pl.DataFrame(srows)

    # --- склеенный OOF: wCV и общий AUC ---------------------------------------
    y_all = np.concatenate(pool_y)
    cut_all = np.concatenate(pool_cut)
    rep = {e: evaluate(y_all, np.concatenate(pool[e]), cut_all) for e in (a.base, a.exp)}
    full = len(ready) == len(VAL_FOLDS_S1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fold_df.write_csv(out / "folds.csv")
    seg_df.write_csv(out / "segments.csv")

    w = max(len(a.base), len(a.exp)) + 1
    print(f"\n=== EXP-030 SEQ-D3A: {len(ready)} фолдов, опора {a.base} ===")
    print(f"{'фолд':>12} {'вариант':<{w}} {'RMSLE_cal':>10} {'Δ':>9} {'AUC':>8} {'ΔAUC':>9} "
          f"{'Var(Δ)':>8} {'corr_res':>9} {'сдвиг':>7}")
    for r in fold_df.to_dicts():
        print(f"{r['fold']:>12} {r['exp']:<{w}} {r['rmsle_cal']:>10.5f} "
              f"{r['d_rmsle_cal']:>+9.5f} {r['auc']:>8.5f} {r['d_auc']:>+9.5f} "
              f"{r['var_vs_base']:>8.5f} {r['corr_resid']:>9.5f} {r['offset']:>+7.3f}")

    print(format_report(rep[a.exp], rep[a.base]))
    if full:
        w0, w1 = rep[a.base]["wcv"], rep[a.exp]["wcv"]
        print(f"\n  wCV BASE {w0:.5f} -> D3A {w1:.5f}   ΔwCV = {w1 - w0:+.5f}")
        print(f"  порог отправки -0.0020 | порог разработки -0.0005 | пол разрешения 0.0005")
    else:
        sc = {r["fold"]: r for r in fold_df.filter(pl.col("exp") == a.exp).to_dicts()}
        print(f"\n  wCV не считается: посчитано {len(ready)}/4 фолдов "
              f"(`validation.wcv` определён только на полной схеме S1)")
        print("  частичная взвешенная сумма (НЕ wCV, только для слежения): "
              + " ".join(f"{k}:{v['d_rmsle_cal']:+.5f}" for k, v in sc.items()))

    for e in (a.base, a.exp):
        print(f"  AUC(y>0) склеенного OOF, {e}: "
              f"{auc_positive(y_all, np.concatenate(pool[e])):.5f}")

    wins = int((fold_df.filter(pl.col("exp") == a.exp)["d_rmsle_cal"] < 0).sum())
    print(f"  лучше BASE на {wins} фолдах из {len(ready)}")
    v = fold_df.filter(pl.col("exp") == a.exp)["var_vs_base"].to_numpy()
    print(f"  Var(z_D3A - z_BASE) = {v.mean():.5f} в среднем по фолдам "
          f"({v.mean() / SEED_FLOOR:.2f}x пола сидов {SEED_FLOOR})")

    print()
    for name in KEY_SEGMENTS:
        sub = seg_df.filter((pl.col("segment") == name) & (pl.col("exp") == a.exp))
        print(f"-- {name}")
        for r in sub.to_dicts():
            print(f"   {r['fold']}  n {r['n']:>7,}  RMSLE {r['rmsle']:.5f} ({r['d_rmsle']:+.5f})"
                  f"  AUC {r['auc']:.5f} ({r['d_auc']:+.5f})")
    print(f"\nзаписано: {out}/folds.csv, {out}/segments.csv")


if __name__ == "__main__":
    main()
