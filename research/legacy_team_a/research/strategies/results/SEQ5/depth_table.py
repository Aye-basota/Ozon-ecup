"""EXP-030 — сводка диагностики глубины: BASE против depth curriculum.

Собирает три уже посчитанные `src/seq.py` таблицы в одно сравнение:

  * `artifacts/depth_<ckpt>.csv`        — кривая по глубине на своём фолде;
  * `artifacts/xdepth_<ckpt>_on1016.csv` — ранняя модель на поздней панели (+77);
  * `artifacts/availprobe_<ckpt>_on<MMDD>.csv` — цена бита `avail ≡ 1`.

Главный вопрос, ради которого это считается: `exp_029` купил инвариантность
частичным отказом от длинной истории (gain +77 сжался на 33–47 %). Curriculum
обязан этого НЕ повторить. Поэтому центральные числа — `gain` от мелкой
опорной глубины к полной и положение оптимума.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ5/depth_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import ARTIFACTS

TAGS = ["V0904", "V0918", "V1002", "V1016"]


def _read(p: Path):
    return pl.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="SEQ-D3A-BASE-S42")
    ap.add_argument("--exp", default="SEQ-D3A-S42")
    ap.add_argument("--ref-depth", type=int, default=212,
                    help="мелкая опорная глубина для gain (212 = train max фолда 09-04)")
    ap.add_argument("--out", default="research/strategies/results/SEQ5")
    a = ap.parse_args()
    exps = [a.base, a.exp]

    rows = []
    for tag in TAGS:
        for e in exps:
            t = _read(ARTIFACTS / f"depth_{e}-{tag}.csv")
            if t is None:
                continue
            m = dict(zip(t["depth"].to_list(), t["rmsle_cal"].to_list()))
            full = int(t["depth"].max())
            best = t.filter(pl.col("rmsle_cal") == t["rmsle_cal"].min()).to_dicts()[0]
            ref = a.ref_depth if a.ref_depth in m else min(m)
            rows.append(dict(
                fold=tag, exp=e, train_max=int(t["train_max"][0]), full=full,
                cal_ref=m[ref], ref_depth=ref, cal_full=m[full],
                gain_ref_to_full=m[full] - m[ref],
                opt_depth=int(best["depth"]), cal_opt=best["rmsle_cal"],
                full_minus_opt=m[full] - best["rmsle_cal"],
                cal_90=m.get(90), cal_150=m.get(150), cal_180=m.get(180),
                cal_254=m.get(254)))
    depth_df = pl.DataFrame(rows) if rows else None

    xrows = []
    for e in exps:
        t = _read(ARTIFACTS / f"xdepth_{e}-V0904_on1016.csv")
        if t is None:
            continue
        m = dict(zip(t["depth"].to_list(), t["rmsle_cal"].to_list()))
        tm, av = int(t["train_max"][0]), int(t["avail"][0])
        best = t.filter(pl.col("rmsle_cal") == t["rmsle_cal"].min()).to_dicts()[0]
        xrows.append(dict(exp=e, train_max=tm, avail=av, extrap=av - tm,
                          cal_clip=m[tm], cal_full=m[av], gain=m[av] - m[tm],
                          opt_depth=int(best["depth"]), cal_opt=best["rmsle_cal"],
                          full_minus_opt=m[av] - best["rmsle_cal"]))
    x_df = pl.DataFrame(xrows) if xrows else None

    prows = []
    for e in exps:
        for tag in TAGS:
            t = _read(ARTIFACTS / f"availprobe_{e}-{tag}_on{tag[1:]}.csv")
            if t is None:
                continue
            r = t.to_dicts()[1]
            prows.append(dict(exp=e, fold=tag, d_cal=r["d_cal"],
                              var_dz=r["var_vs_base"], corr=r["corr_vs_base"],
                              rmsle_cal=t.to_dicts()[0]["rmsle_cal"]))
    p_df = pl.DataFrame(prows) if prows else None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = max(len(e) for e in exps) + 1
    if depth_df is not None:
        depth_df.write_csv(out / "depth.csv")
        print(f"\n=== кривая по глубине на своём фолде (калиброванный RMSLE) ===")
        print(f"{'фолд':>6} {'вариант':<{w}} {'train_max':>9} {'full':>5} {'cal@ref':>9} "
              f"{'cal@full':>9} {'gain':>9} {'опт':>5} {'full-опт':>9}")
        for r in depth_df.to_dicts():
            print(f"{r['fold']:>6} {r['exp']:<{w}} {r['train_max']:>9} {r['full']:>5} "
                  f"{r['cal_ref']:>9.5f} {r['cal_full']:>9.5f} {r['gain_ref_to_full']:>+9.5f} "
                  f"{r['opt_depth']:>5} {r['full_minus_opt']:>+9.5f}")
    if x_df is not None:
        x_df.write_csv(out / "crossdepth.csv")
        print(f"\n=== crossdepth: модель 09-04 на панели 10-16 (аналог теста, +77 дней) ===")
        print(f"{'вариант':<{w}} {'cal@212':>9} {'cal@289':>9} {'gain +77':>9} "
              f"{'опт':>5} {'full-опт':>9}")
        for r in x_df.to_dicts():
            print(f"{r['exp']:<{w}} {r['cal_clip']:>9.5f} {r['cal_full']:>9.5f} "
                  f"{r['gain']:>+9.5f} {r['opt_depth']:>5} {r['full_minus_opt']:>+9.5f}")
    if p_df is not None:
        p_df.write_csv(out / "availprobe.csv")
        print(f"\n=== availprobe: цена бита `avail ≡ 1` (у curriculum обязана ОСТАТЬСЯ) ===")
        print(f"{'вариант':<{w}} {'фолд':>6} {'Δ cal':>9} {'Var(Δz)':>9} {'corr':>9}")
        for r in p_df.to_dicts():
            print(f"{r['exp']:<{w}} {r['fold']:>6} {r['d_cal']:>+9.5f} "
                  f"{r['var_dz']:>9.5f} {r['corr']:>9.5f}")
    print(f"\nзаписано в {out}")


if __name__ == "__main__":
    main()
