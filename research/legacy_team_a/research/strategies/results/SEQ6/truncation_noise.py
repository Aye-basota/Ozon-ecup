"""EXP-030b — механизм: что depth curriculum делает со строками БЕЗ истории покупок.

Гипотеза механизма (проверяется здесь чистым замером по данным, без модели).

Обрезка до глубины `D` обнуляет ВСЕ поведенческие каналы старше `D` и ставит
там `avail = 0` (`src/seq.py`, `gather`). Значит пользователь, чья последняя
покупка была `rec_buy > D` дней назад, при обрезке подаётся на вход как строка,
в которой покупок нет ВООБЩЕ — то есть неотличимо от того, кто не покупал
никогда. Граница `avail` у обоих тоже одинаковая: она задаётся `D`, а не
историей пользователя.

Таргет при этом НЕ обрезается: он остаётся своим. Если у «спрятанных
покупателей» уровень таргета выше, чем у настоящих непокупателей, curriculum
подмешивает в область «пустой истории покупок» систематически ЗАВЫШЕННУЮ
разметку. Тогда модель обязана поднять прогноз на этой области и потерять
ранжирование внутри неё — ровно то, что наблюдалось на 09-18 (`exp_030`:
bias по сегменту «никогда не покупал» +0.078 -> -0.186, AUC -0.064).

Что считается:

  * `share_hidden(D)` — доля панели, у кого есть покупка, но `rec_buy > D`;
  * `E[log1p(y)]` у спрятанных покупателей против настоящих непокупателей;
  * то же с усреднением по сетке глубин и с учётом `p` (ожидаемая доля
    отравленных предъявлений на эпоху);
  * отдельно на валидационной панели фолда и на его обучающих cutoff'ах.

`rec_buy` берётся из `feat_*_LNone.parquet` (полная история, как в сегментации
`SEQ5/analyze.py`), таргет валидации — из сохранённого OOF, таргет обучающих
cutoff'ов — из `seq_gmv_v1.npy` тем же определением, что `seq.target_at`.
Большая панель `seq_panel_v1.npy` НЕ загружается.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ6/truncation_noise.py
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS, DATA_PROCESSED
from src.tracking import load_oof

DATA_START = dt.date(2025, 1, 1)
TARGET_DAYS = 30
SEQ_L = 365
DEPTH_GRID = (90, 120, 150, 180, 220, 254, 289)
P_AUG = 0.5


def rec_buy_at(T: dt.date, uid: np.ndarray) -> np.ndarray:
    f = pl.read_parquet(DATA_PROCESSED / f"feat_{T.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id", "rec_buy"])
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    assert f.height == len(uid)
    return f["rec_buy"].to_numpy().astype(float)


def real_depth(T: dt.date) -> int:
    """Сколько реальных дней истории доступно на cutoff'е T (как в `seq.gather`)."""
    d = (T - DATA_START).days
    return min(SEQ_L, d + 1)


def rows_for(T: dt.date, uid: np.ndarray, y: np.ndarray) -> list[dict]:
    rb = rec_buy_at(T, uid)
    R = real_depth(T)
    ly = np.log1p(y)
    never = np.isnan(rb)                       # не покупал никогда за всю историю
    bought = ~never
    out = []
    for D in DEPTH_GRID:
        eff = min(D, R)                        # глубины сверх реальной — no-op
        hidden = bought & (rb > eff)           # покупка есть, но при обрезке не видна
        n_h = int(hidden.sum())
        out.append(dict(
            cutoff=T.isoformat(), depth=D, real_depth=R, eff_depth=eff,
            n=len(uid), n_never=int(never.sum()), share_never=float(never.mean()),
            n_hidden=n_h, share_hidden=float(hidden.mean()),
            # во что превращается область «покупок не видно»
            share_of_empty=float(n_h / max(n_h + int(never.sum()), 1)),
            z_never=float(ly[never].mean()),
            z_hidden=float(ly[hidden].mean()) if n_h else float("nan"),
            d_z=float(ly[hidden].mean() - ly[never].mean()) if n_h else float("nan"),
            pos_never=float((y[never] > 0).mean()),
            pos_hidden=float((y[hidden] > 0).mean()) if n_h else float("nan"),
            # суммы — чтобы агрегировать по cutoff'ам точно, а не средним средних:
            # на мелких cutoff'ах часть сетки — no-op, и там среднее не определено
            sum_z_never=float(ly[never].sum()), sum_z_hidden=float(ly[hidden].sum()),
            sum_pos_never=float((y[never] > 0).sum()),
            sum_pos_hidden=float((y[hidden] > 0).sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="2025-09-18")
    ap.add_argument("--oof", default="SEQ-D3A-BASE-S42-V0918")
    ap.add_argument("--train-cutoffs", nargs="*", default=None,
                    help="по умолчанию — все обучающие cutoff'ы фолда, шаг 7")
    ap.add_argument("--out", default="research/strategies/results/SEQ6")
    a = ap.parse_args()

    V = dt.date.fromisoformat(a.val)
    d = load_oof(a.oof)
    uid_v = np.asarray(d["user_id"])
    y_v = np.asarray(d["y"], float)

    rows = rows_for(V, uid_v, y_v)
    for r in rows:
        r["panel"] = "валидация"

    # --- обучающие cutoff'ы того же фолда -------------------------------------
    if a.train_cutoffs:
        cuts = [dt.date.fromisoformat(c) for c in a.train_cutoffs]
    else:
        cuts, T = [], V - dt.timedelta(days=35)     # первый обучающий: T + 30 <= V
        while (T - DATA_START).days + 1 >= 90:
            if (DATA_PROCESSED / f"feat_{T.strftime('%Y%m%d')}_LNone.parquet").exists():
                cuts.append(T)
            T -= dt.timedelta(days=7)
        cuts = sorted(cuts)

    gmv = np.load(DATA_PROCESSED / "seq_gmv_v1.npy", mmap_mode="r")
    uid_all = np.load(DATA_PROCESSED / "seq_uid_v1.npy")
    for T in cuts:
        f = pl.read_parquet(DATA_PROCESSED / f"feat_{T.strftime('%Y%m%d')}_LNone.parquet",
                            columns=["user_id"])
        uid_t = f["user_id"].to_numpy()
        idx = np.searchsorted(uid_all, uid_t)
        assert np.array_equal(uid_all[idx], uid_t), f"{T}: неизвестный user_id"
        dT = (T - DATA_START).days
        y_t = np.asarray(gmv[idx, dT + 1:dT + 1 + TARGET_DAYS]).sum(axis=1)
        for r in rows_for(T, uid_t, y_t):
            r["panel"] = "обучение"
            rows.append(r)

    df = pl.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.write_csv(out / "truncation_noise.csv")

    print(f"\n=== Разметочный шум обрезки, фолд {V} ===")
    print(f"обучающих cutoff'ов: {len(cuts)} "
          f"({cuts[0]}..{cuts[-1]}), реальная глубина {real_depth(cuts[0])}..{real_depth(cuts[-1])}\n")

    for panel in ("валидация", "обучение"):
        sub = df.filter(pl.col("panel") == panel)
        if not sub.height:
            continue
        g = (sub.group_by("depth")
             .agg(pl.col("n").sum().alias("N"), pl.col("n_never").sum().alias("NV"),
                  pl.col("n_hidden").sum().alias("HD"),
                  pl.col("sum_z_never").sum(), pl.col("sum_z_hidden").sum(),
                  pl.col("sum_pos_never").sum(), pl.col("sum_pos_hidden").sum())
             .with_columns(
                 share_never=pl.col("NV") / pl.col("N"),
                 share_hidden=pl.col("HD") / pl.col("N"),
                 share_of_empty=pl.col("HD") / (pl.col("HD") + pl.col("NV")),
                 z_never=pl.col("sum_z_never") / pl.col("NV"),
                 z_hidden=pl.when(pl.col("HD") > 0)
                            .then(pl.col("sum_z_hidden") / pl.col("HD")),
                 pos_never=pl.col("sum_pos_never") / pl.col("NV"),
                 pos_hidden=pl.when(pl.col("HD") > 0)
                              .then(pl.col("sum_pos_hidden") / pl.col("HD")))
             .with_columns(d_z=pl.col("z_hidden") - pl.col("z_never"))
             .sort("depth"))
        print(f"-- панель: {panel}")
        print(f"{'D':>5}{'доля never':>12}{'доля hidden':>13}{'hidden/пусто':>14}"
              f"{'E z never':>11}{'E z hidden':>12}{'Δ z':>9}{'P(y>0) nv':>11}{'P(y>0) hd':>11}")
        for r in g.to_dicts():
            zh = float("nan") if r["z_hidden"] is None else r["z_hidden"]
            dzv = float("nan") if r["d_z"] is None else r["d_z"]
            ph = float("nan") if r["pos_hidden"] is None else r["pos_hidden"]
            print(f"{r['depth']:>5}{r['share_never']:>12.4f}{r['share_hidden']:>13.4f}"
                  f"{r['share_of_empty']:>14.4f}{r['z_never']:>11.4f}{zh:>12.4f}"
                  f"{dzv:>+9.4f}{r['pos_never']:>11.4f}{ph:>11.4f}")
        # Глубины сверх реальной — no-op: там hidden = 0, а Δz не определена.
        # Поэтому средняя тяжесть шума взвешивается долей, а не берётся как
        # арифметическое среднее по сетке.
        gd = g.to_dicts()
        sh = np.array([r["share_hidden"] for r in gd], float)
        dz = np.array([np.nan if r["d_z"] is None else r["d_z"] for r in gd], float)
        emp = np.array([r["share_of_empty"] for r in gd], float)
        ok = ~np.isnan(dz)
        dz_w = float((sh[ok] * dz[ok]).sum() / sh[ok].sum()) if sh[ok].sum() else float("nan")
        print(f"   по сетке: средняя доля hidden {sh.mean():.4f}, "
              f"взвешенная Δz {dz_w:+.4f}; при p={P_AUG} ожидаемая доля "
              f"отравленных предъявлений {P_AUG * sh.mean():.4f}")
        print(f"   в области «покупок не видно» доля спрятанных покупателей: "
              f"среднее по сетке {emp.mean():.4f}, максимум {emp.max():.4f} "
              f"(при D = {g.to_dicts()[int(emp.argmax())]['depth']})\n")

    print(f"записано: {out}/truncation_noise.csv")


if __name__ == "__main__":
    main()
