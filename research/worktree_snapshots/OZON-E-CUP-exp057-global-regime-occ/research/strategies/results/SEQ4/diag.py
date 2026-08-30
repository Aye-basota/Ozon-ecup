"""SEQ-03A этап 1 — сводка по диагностическому фолду.

Считает по одному фолду ровно те шесть чисел, которыми принимается решение
(постановка `SEQ-03A`), плюс `availprobe` из `exp_027`:

  1. калиброванный RMSLE;                4. полоса `w180_days_buy` 2–15;
  2. AUC(y>0);                           5. `Var(z_вариант − z_BASE)`;
  3. полоса `rec_buy` 15–60;             6. корреляция остатков.

Сегменты — те же границы и тот же источник (`feat_<T>_LNone.parquet`), что в
`src/ptime_eval.py` и `research/rmsle_diagnostics`, чтобы числа сравнивались с
уже опубликованными. Калибровка — пофолдовая, внутри сегмента НЕ пересчитывается
(`rmsle_diagnostics` §3: сегментная перекалибровка закрыта измерением).

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ4/diag.py \
      --exps SEQ-03A-BASE-S42 SEQ-03A-B-S42 --fold 2025-10-16
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS, DATA_PROCESSED
from src.tracking import load_oof
from src.validation import calibrate, rmsle_z

SEG_COLS = ["rec_buy", "w180_days_buy"]
KEY_SEGMENTS = ["ВСЕ", "rec_buy 15-60", "w180_days_buy 2-15", "пересечение"]


def auc(pos: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(pos):
        return float("nan")
    return float(roc_auc_score(pos.astype(np.int8), score))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True, help="без суффикса -V<MMDD>")
    ap.add_argument("--fold", default="2025-10-16")
    ap.add_argument("--out", default="research/strategies/results/SEQ4")
    a = ap.parse_args()

    V = dt.date.fromisoformat(a.fold)
    tag = f"V{V.strftime('%m%d')}"
    exps = [e for e in dict.fromkeys(a.exps)
            if (ARTIFACTS / f"oof_{e}-{tag}.npz").exists()]
    missing = [e for e in a.exps if e not in exps]
    if missing:
        print(f"нет OOF (ещё не посчитаны): {' '.join(missing)}")
    assert exps, "ни одного посчитанного варианта"
    base = exps[0]

    ds = {e: load_oof(f"{e}-{tag}") for e in exps}
    uid = np.asarray(ds[base]["user_id"])
    y = np.asarray(ds[base]["y"], float)
    for e in exps[1:]:
        assert np.array_equal(np.asarray(ds[e]["user_id"]), uid), f"{e}: другой набор строк"
        assert np.allclose(ds[e]["y"], y), f"{e}: другой таргет"
    ly = np.log1p(y)
    pos = y > 0

    # калибровка — один сдвиг на фолд, как в wCV
    z_raw = {e: np.asarray(ds[e]["z"], float) for e in exps}
    z_cal, off = {}, {}
    for e in exps:
        d, _ = calibrate(y, z_raw[e])
        off[e] = d
        z_cal[e] = np.maximum(z_raw[e] + d, 0.0)

    f = pl.read_parquet(DATA_PROCESSED / f"feat_{V.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id"] + SEG_COLS)
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    assert f.height == len(uid) and f["rec_buy"].len() == len(uid)
    segs = segments(f)

    # --- главная таблица -----------------------------------------------------
    rows = []
    rb = ly - z_cal[base]
    for e in exps:
        r = ly - z_cal[e]
        d_raw = z_raw[e] - z_raw[base]
        d_cal = z_cal[e] - z_cal[base]
        rows.append(dict(
            exp=e, fold=a.fold, n=len(uid), offset=off[e],
            rmsle=rmsle_z(y, z_raw[e]), rmsle_cal=rmsle_z(y, z_cal[e]),
            d_rmsle_cal=rmsle_z(y, z_cal[e]) - rmsle_z(y, z_cal[base]),
            auc=auc(pos, z_cal[e]), d_auc=auc(pos, z_cal[e]) - auc(pos, z_cal[base]),
            # Основная Var(Δ) сопоставима с exp_018/025/026: сохранённые
            # предсказания в log-space до пофолдового сдвига. Калиброванную
            # оставляем отдельно как проверку чувствительности к уровню.
            var_vs_base=float(np.var(d_raw)),
            var_cal_vs_base=float(np.var(d_cal)),
            corr_resid=1.0 if e == base else float(np.corrcoef(r, rb)[0, 1]),
            corr_pred=1.0 if e == base else float(np.corrcoef(z_cal[e], z_cal[base])[0, 1]),
            mean_z=float(z_raw[e].mean()), std_z=float(z_raw[e].std())))
    main_df = pl.DataFrame(rows)

    # --- сегменты ------------------------------------------------------------
    srows = []
    for name, m in segs.items():
        for e in exps:
            srows.append(dict(segment=name, exp=e, n=int(m.sum()), share=float(m.mean()),
                              rmsle=rmsle_z(y[m], z_cal[e][m]), auc=auc(pos[m], z_cal[e][m]),
                              mse_share=float(((ly - z_cal[e]) ** 2)[m].sum()
                                              / ((ly - z_cal[e]) ** 2).sum())))
    seg_df = pl.DataFrame(srows)
    base_seg = {(r["segment"]): r for r in srows if r["exp"] == base}
    seg_df = seg_df.with_columns(
        d_rmsle=pl.Series([r["rmsle"] - base_seg[r["segment"]]["rmsle"] for r in srows]),
        d_auc=pl.Series([r["auc"] - base_seg[r["segment"]]["auc"] for r in srows]))

    # --- availprobe: тот же файл, что пишет `src/seq.py availprobe` ----------
    prows = []
    for e in exps:
        p = ARTIFACTS / f"availprobe_{e}-{tag}_on{V.strftime('%m%d')}.csv"
        if not p.exists():
            continue
        t = pl.read_csv(p)
        r = t.filter(pl.col("mode").str.contains("≡ 1")).to_dicts()[0]
        prows.append(dict(exp=e, aug=r.get("aug", ""),
                          d_cal=r["d_cal"], var_dz=r["var_vs_base"],
                          corr=r["corr_vs_base"],
                          bias=r["mean_z"] - t.to_dicts()[0]["mean_z"],
                          rmsle_cal_normal=t.to_dicts()[0]["rmsle_cal"],
                          rmsle_cal_forced=r["rmsle_cal"]))
    probe_df = pl.DataFrame(prows) if prows else None

    # --- availcurve: штраф как функция ПОЛОЖЕНИЯ границы ---------------------
    # 76 нулей — край, реально прожитый самым глубоким обучающим cutoff'ом;
    # 0 нулей — режим теста на полной глубине. Плоская кривая = `full365`
    # перестал быть особой точкой, а не «стал чуть менее особой».
    curves = []
    for e in exps:
        p = ARTIFACTS / f"availcurve_{e}-{tag}_on{V.strftime('%m%d')}.csv"
        if p.exists():
            curves.append(pl.read_csv(p))
    curve_df = pl.concat(curves) if curves else None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if curve_df is not None:
        curve_df.write_csv(out / f"availcurve_{tag}.csv")
    main_df.write_csv(out / f"fold_{tag}.csv")
    seg_df.write_csv(out / f"segments_{tag}.csv")
    if probe_df is not None:
        probe_df.write_csv(out / f"availprobe_{tag}.csv")

    w = max(len(e) for e in exps) + 1
    print(f"\n=== фолд {a.fold}, {len(uid):,} строк, опора {base} ===")
    print(f"{'вариант':<{w}} {'RMSLE_cal':>10} {'Δ':>9} {'AUC':>8} {'ΔAUC':>9} "
          f"{'Var(Δ)':>8} {'Var_cal':>8} {'corr_res':>9} {'сдвиг':>7}")
    for r in main_df.to_dicts():
        print(f"{r['exp']:<{w}} {r['rmsle_cal']:>10.5f} {r['d_rmsle_cal']:>+9.5f} "
              f"{r['auc']:>8.5f} {r['d_auc']:>+9.5f} {r['var_vs_base']:>8.5f} "
              f"{r['var_cal_vs_base']:>8.5f} "
              f"{r['corr_resid']:>9.5f} {r['offset']:>+7.3f}")

    print(f"\n{'availprobe: normal -> avail ≡ 1':<{w}} {'Δ RMSLE_cal':>12} {'Var(Δz)':>9} "
          f"{'bias':>8} {'corr':>9}")
    for r in (probe_df.to_dicts() if probe_df is not None else []):
        print(f"{r['exp']:<{w}} {r['d_cal']:>+12.5f} {r['var_dz']:>9.5f} "
              f"{r['bias']:>+8.4f} {r['corr']:>9.5f}")

    if curve_df is not None:
        left = sorted(curve_df["n_zero_left"].unique().to_list(), reverse=True)
        print(f"\navailcurve: Δ калибр. RMSLE по числу ОСТАВШИХСЯ позиций avail=0 "
              f"(76 = край train support, 0 = тест на полной глубине)")
        print(f"{'вариант':<{w}} " + " ".join(f"{v:>8}" for v in left))
        for e in exps:
            sub = curve_df.filter(pl.col("ckpt") == f"{e}-{tag}")
            if not sub.height:
                continue
            m = dict(zip(sub["n_zero_left"].to_list(), sub["d_cal"].to_list()))
            print(f"{e:<{w}} " + " ".join(f"{m[v]:>+8.5f}" if v in m else f"{'—':>8}"
                                          for v in left))

    print()
    for name in KEY_SEGMENTS:
        print(f"-- {name} (доля {segs[name].mean():.3f})")
        for r in seg_df.filter(pl.col("segment") == name).to_dicts():
            print(f"   {r['exp']:<{w}} RMSLE {r['rmsle']:.5f} ({r['d_rmsle']:+.5f})  "
                  f"AUC {r['auc']:.5f} ({r['d_auc']:+.5f})  доля MSE {r['mse_share']:.3f}")
    print(f"\nзаписано: {out}/fold_{tag}.csv, segments_{tag}.csv, availprobe_{tag}.csv")


if __name__ == "__main__":
    main()
