"""EXP-030b — разделяющий замер: воспроизводится ли провал `SEQ-D3A` на 09-18?

На фолде 2025-09-18 есть четыре прогона одного и того же кода:

    BASE-S42, D3A-S42   (`exp_030`)
    BASE-S43, D3A-S43   (этот замер)

Скрипт считает ЧЕТЫРЕ контраста, а не один:

  1. `D3A - BASE` при сиде 42   — исходный провал +0.00355;
  2. `D3A - BASE` при сиде 43   — воспроизводится ли он;
  3. `BASE43 - BASE42`          — контраст ТОЛЬКО по сиду, шумовой ориентир;
  4. `D3A43 - D3A42`            — тот же шумовой ориентир внутри варианта.

Пункты 3-4 — главное, чего не было в `exp_030`: там `Var(Δ)` сравнивалась с
проектным полом 0.00712 и с парой сидов из `exp_029` на ДРУГОМ фолде. Здесь
шумовая полоса измерена на этом же фолде тем же кодом.

Метрики — те же, что в `SEQ5/analyze.py` (пофолдовая калибровка, сегменты по
`rec_buy`/`w180_days_buy`, внутри сегмента калибровка НЕ пересчитывается).

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ6/seed_pair.py
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
SEED_FLOOR = 0.00712                     # пол разнообразия проекта (exp_018)
TCN_SEED_STD_WCV = 0.00250               # seed std TCN по wCV (STATE.md)


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
    ap.add_argument("--val", default="2025-09-18")
    ap.add_argument("--seeds", type=int, nargs=2, default=[42, 43],
                    help="опорный и проверочный сид; --seeds 42 42 = самопроверка кода "
                         "(контрасты по сиду обязаны выйти РОВНО нулевыми)")
    ap.add_argument("--out", default="research/strategies/results/SEQ6")
    a = ap.parse_args()

    V = dt.date.fromisoformat(a.val)
    tag = f"V{V.strftime('%m%d')}"
    s0, s1 = a.seeds
    runs = {"BASE42": f"SEQ-D3A-BASE-S{s0}-{tag}", "D3A42": f"SEQ-D3A-S{s0}-{tag}",
            "BASE43": f"SEQ-D3A-BASE-S{s1}-{tag}", "D3A43": f"SEQ-D3A-S{s1}-{tag}"}
    miss = [k for k, p in runs.items() if not (ARTIFACTS / f"oof_{p}.npz").exists()]
    assert not miss, f"нет OOF: {miss}"

    d = {k: load_oof(p) for k, p in runs.items()}
    uid = np.asarray(d["BASE42"]["user_id"])
    y = np.asarray(d["BASE42"]["y"], float)
    for k in runs:
        assert np.array_equal(np.asarray(d[k]["user_id"]), uid), f"{k}: другой набор строк"
        assert np.allclose(d[k]["y"], y), f"{k}: другой таргет"
    ly, pos = np.log1p(y), y > 0

    z_raw, z_cal, off = {}, {}, {}
    for k in runs:
        z = np.asarray(d[k]["z"], float)
        o, _ = calibrate(y, z)
        z_raw[k], off[k], z_cal[k] = z, o, np.maximum(z + o, 0.0)

    f = pl.read_parquet(DATA_PROCESSED / f"feat_{V.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id"] + SEG_COLS)
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    assert f.height == len(uid)
    segs = segments(f)

    # --- абсолютные числа по каждому прогону ---------------------------------
    abs_rows = [dict(run=k, part=runs[k], n=len(uid), offset=off[k],
                     rmsle=rmsle_z(y, z_raw[k]), rmsle_cal=rmsle_z(y, z_cal[k]),
                     auc=auc(pos, z_cal[k]),
                     mean_z=float(z_raw[k].mean()), std_z=float(z_raw[k].std()))
                for k in runs]

    contrasts = [("приём, сид 42", "D3A42", "BASE42"),
                 ("приём, сид 43", "D3A43", "BASE43"),
                 ("сид, BASE 43-42", "BASE43", "BASE42"),
                 ("сид, D3A 43-42", "D3A43", "D3A42")]

    rows, srows = [], []
    for name, hi, lo in contrasts:
        r_hi, r_lo = ly - z_cal[hi], ly - z_cal[lo]
        rows.append(dict(
            contrast=name, hi=hi, lo=lo,
            rmsle_lo=rmsle_z(y, z_cal[lo]), rmsle_hi=rmsle_z(y, z_cal[hi]),
            d_rmsle_cal=rmsle_z(y, z_cal[hi]) - rmsle_z(y, z_cal[lo]),
            auc_lo=auc(pos, z_cal[lo]), d_auc=auc(pos, z_cal[hi]) - auc(pos, z_cal[lo]),
            var_raw=float(np.var(z_raw[hi] - z_raw[lo])),
            var_cal=float(np.var(z_cal[hi] - z_cal[lo])),
            corr_pred=float(np.corrcoef(z_cal[hi], z_cal[lo])[0, 1]),
            corr_resid=float(np.corrcoef(r_hi, r_lo)[0, 1]),
            d_offset=off[hi] - off[lo]))
        for sname, m in segs.items():
            # Разложение потери внутри сегмента на УРОВЕНЬ и РАНЖИРОВАНИЕ.
            # `*_seglocal` — гипотетический счёт, если бы каждому варианту дали
            # СВОЙ сдвиг внутри сегмента. Это ДИАГНОСТИКА, а не приём:
            # посегментная перекалибровка как постобработка закрыта
            # (`rmsle_diagnostics` §3, STATE.md «Не повторять»). Смысл такой:
            # если Δ после этого схлопывается — потеря чисто уровневая и
            # лечится одним числом; если остаётся — сломано ранжирование.
            loc = {}
            for k in (lo, hi):
                o_s, cal_s = calibrate(y[m], z_raw[k][m])
                loc[k] = (o_s, cal_s)
            srows.append(dict(contrast=name, segment=sname, n=int(m.sum()),
                              share=float(m.mean()),
                              rmsle_lo=rmsle_z(y[m], z_cal[lo][m]),
                              rmsle_hi=rmsle_z(y[m], z_cal[hi][m]),
                              d_rmsle=rmsle_z(y[m], z_cal[hi][m]) - rmsle_z(y[m], z_cal[lo][m]),
                              bias_lo=float((ly - z_cal[lo])[m].mean()),
                              bias_hi=float((ly - z_cal[hi])[m].mean()),
                              offset_loc_lo=loc[lo][0], offset_loc_hi=loc[hi][0],
                              d_rmsle_seglocal=loc[hi][1] - loc[lo][1],
                              auc_lo=auc(pos[m], z_cal[lo][m]),
                              d_auc=auc(pos[m], z_cal[hi][m]) - auc(pos[m], z_cal[lo][m]),
                              mse_share=float(((ly - z_cal[hi]) ** 2)[m].sum()
                                              / ((ly - z_cal[hi]) ** 2).sum())))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(abs_rows).write_csv(out / "runs.csv")
    con = pl.DataFrame(rows)
    seg = pl.DataFrame(srows)
    con.write_csv(out / "contrasts.csv")
    seg.write_csv(out / "seg_contrasts.csv")

    print(f"\n=== EXP-030b: фолд {V}, n = {len(uid):,}, доля y>0 = {pos.mean():.4f} ===\n")
    print(f"{'прогон':<8} {'RMSLE':>9} {'RMSLE_cal':>10} {'AUC':>8} {'сдвиг':>7} {'mean z':>8}")
    for r in abs_rows:
        print(f"{r['run']:<8} {r['rmsle']:>9.5f} {r['rmsle_cal']:>10.5f} {r['auc']:>8.5f} "
              f"{r['offset']:>+7.3f} {r['mean_z']:>8.4f}")

    print(f"\n{'контраст':<18} {'d RMSLE_cal':>12} {'dAUC':>9} {'Var(dz)':>9} "
          f"{'x пола':>7} {'corr_res':>9} {'d сдвига':>9}")
    for r in rows:
        print(f"{r['contrast']:<18} {r['d_rmsle_cal']:>+12.5f} {r['d_auc']:>+9.5f} "
              f"{r['var_raw']:>9.5f} {r['var_raw'] / SEED_FLOOR:>7.2f} "
              f"{r['corr_resid']:>9.5f} {r['d_offset']:>+9.3f}")

    print("\n--- сегменты: d RMSLE_cal (dAUC) по каждому контрасту ---")
    keys = ["rec_buy 15-60", "w180_days_buy 2-15", "пересечение",
            "никогда не покупал", "w180_days_buy 0-1", "w180_days_buy 16+"]
    hdr = "".join(f"{c[0]:>21}" for c in contrasts)
    print(f"{'сегмент':<22}{'доля':>7}{hdr}")
    for sname in keys:
        sh = seg.filter(pl.col("segment") == sname)["share"][0]
        line = f"{sname:<22}{sh:>7.3f}"
        for name, _, _ in contrasts:
            r = seg.filter((pl.col("segment") == sname)
                           & (pl.col("contrast") == name)).to_dicts()[0]
            line += f"{r['d_rmsle']:>+11.5f}({r['d_auc']:>+8.5f})"
        print(line)

    print("\n--- уровень или ранжирование: Δ при СВОЁМ сдвиге внутри сегмента ---")
    print("    (диагностика; посегментная перекалибровка как приём закрыта)")
    print(f"{'сегмент':<22}{'контраст':<18}{'Δ общий':>10}{'Δ свой сдвиг':>14}"
          f"{'bias lo':>9}{'bias hi':>9}{'ΔAUC':>10}")
    for sname in ["никогда не покупал", "w180_days_buy 0-1", "ВСЕ"]:
        for name, _, _ in contrasts:
            r = seg.filter((pl.col("segment") == sname)
                           & (pl.col("contrast") == name)).to_dicts()[0]
            print(f"{sname:<22}{name:<18}{r['d_rmsle']:>+10.5f}"
                  f"{r['d_rmsle_seglocal']:>+14.5f}{r['bias_lo']:>+9.4f}"
                  f"{r['bias_hi']:>+9.4f}{r['d_auc']:>+10.5f}")

    print(f"\nпол разнообразия проекта Var = {SEED_FLOOR}; "
          f"seed std TCN по wCV = {TCN_SEED_STD_WCV}")
    print(f"записано: {out}/runs.csv, {out}/contrasts.csv, {out}/seg_contrasts.csv")


if __name__ == "__main__":
    main()
