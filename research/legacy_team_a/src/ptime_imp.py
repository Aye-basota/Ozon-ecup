"""Перераспределение gain и СПЛИТОВ между признаками: база против личного времени.

Механизм `STRATEGY_08` считается подтверждённым не по скору, а по тому, **забрали
ли новые признаки сплиты у старых**. Исходное наблюдение (`artifacts/
importance_S1-BEST.md`): на структуру интервалов (`buygap_mean/std/cv`) уходит
**4 424 сплита ради 0.54% gain** — дерево тысячами осевых разрезов приближает
одну гладкую нормировку. Если явная нормировка на личный ритм — та самая
недостающая функция, эти сплиты обязаны освободиться.

`--imp` в `src/train.py` печатает только топ-15 по gain и не показывает число
сплитов, поэтому здесь обучается одна модель на обучающей выборке последнего
фолда и выгружаются обе величины по ВСЕМ признакам плюс mean|SHAP|.

Запуск:
  python -m src.ptime_imp --exp PT-IMP-BASE --ptime off
  python -m src.ptime_imp --exp PT-IMP-FULL --ptime full
  python -m src.ptime_imp --compare PT-IMP-BASE PT-IMP-FULL
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc

import numpy as np
import polars as pl

from src.config import ARTIFACTS, SEED
from src.data import load
from src.features import feature_names, to_np
from src.train import GROUPS, Setup, assemble, fit_free, select_features, xy

# признаки, которыми дерево сейчас приближает структуру интервалов
GAP_STRUCT = ["buygap_mean", "buygap_std", "buygap_cv", "gap_mean", "gap_std",
              "gap_max_frac", "gap_cv", "rec_over_buygap", "rec_over_gap"]

SHAP_N = 20_000                 # как в artifacts/importance_S1-BEST.md


def run(exp: str, ptime: str | None, source: str, V: dt.date, rounds: int, seed: int) -> None:
    load()
    s = Setup(L=0, min_history=90, panel_blocks=3, train_blocks=1, cutoffs="all",
              model="direct", rounds=rounds, params={"seed": seed}, norm_long=True,
              ptime=ptime, ptime_source=source, vals=[V])
    feats = select_features(feature_names(xy(V, s)[0]), [], None)
    cuts = s.train_cutoffs(V)
    print(f"{exp}: {len(feats)} признаков, {len(cuts)} cutoff'ов, val {V}", flush=True)
    Xtr, ytr, _ = assemble(cuts, s, feats, V)
    box = [Xtr]
    del Xtr
    gc.collect()
    m = fit_free(s, box, ytr, None)

    gain = np.asarray(m.feature_importance("gain"), float)
    split = np.asarray(m.feature_importance("split"), float)

    Xv, _ = xy(V, s)
    A = to_np(Xv, feats)
    idx = np.random.default_rng(SEED).choice(len(A), min(SHAP_N, len(A)), replace=False)
    sv = m.predict(A[idx], pred_contrib=True)          # точный TreeSHAP
    shap = np.abs(np.asarray(sv)[:, :-1]).mean(axis=0)

    pl.DataFrame({"feature": feats, "gain": gain, "split": split, "shap": shap,
                  "gain_share": gain / gain.sum(), "split_share": split / split.sum(),
                  "shap_share": shap / shap.sum()}
                 ).sort("gain", descending=True).write_csv(ARTIFACTS / f"imp_{exp}.csv")
    print(f"  записано artifacts/imp_{exp}.csv  (сплитов всего {split.sum():,.0f})")


def compare(base: str, new: str) -> None:
    b = pl.read_csv(ARTIFACTS / f"imp_{base}.csv")
    n = pl.read_csv(ARTIFACTS / f"imp_{new}.csv")
    bd = {r["feature"]: r for r in b.iter_rows(named=True)}
    nd = {r["feature"]: r for r in n.iter_rows(named=True)}

    print(f"\nвсего сплитов: {base} {b['split'].sum():,.0f} -> {new} {n['split'].sum():,.0f}")
    print(f"признаков: {b.height} -> {n.height}\n")

    print("структура интервалов — забрали ли новые признаки сплиты у старых:")
    print(f"{'признак':>20} {'сплитов до':>11} {'после':>9} {'дельта':>9} "
          f"{'gain% до':>9} {'после':>8}")
    tb = ts = ta = tg = tgn = 0.0
    for f in GAP_STRUCT:
        if f not in bd:
            continue
        r0, r1 = bd[f], nd.get(f)
        s1 = r1["split"] if r1 else 0.0
        g1 = r1["gain_share"] if r1 else 0.0
        print(f"{f:>20} {r0['split']:>11,.0f} {s1:>9,.0f} {s1 - r0['split']:>+9,.0f} "
              f"{r0['gain_share']:>8.2%} {g1:>8.2%}")
        tb += r0["split"]; ts += s1; tg += r0["gain_share"]; tgn += g1
    print(f"{'ИТОГО':>20} {tb:>11,.0f} {ts:>9,.0f} {ts - tb:>+9,.0f} {tg:>8.2%} {tgn:>8.2%}")

    pt = n.filter(pl.col("feature").str.starts_with("pt_"))
    print(f"\nпризнаки личного времени: {pt.height} колонок, "
          f"{pt['gain_share'].sum():.2%} gain, {pt['split'].sum():,.0f} сплитов "
          f"({pt['split_share'].sum():.2%}), {pt['shap_share'].sum():.2%} |SHAP|")
    print(f"\n{'#':>3} {'признак':>22} {'gain%':>8} {'сплитов':>9} {'gain/сплит':>11} "
          f"{'|SHAP|%':>9}")
    top = n.sort("shap", descending=True).head(40)
    for i, r in enumerate(top.iter_rows(named=True), 1):
        mark = " <-" if r["feature"].startswith("pt_") else ""
        print(f"{i:>3} {r['feature']:>22} {r['gain_share']:>8.2%} {r['split']:>9,.0f} "
              f"{r['gain'] / max(r['split'], 1):>11,.4f} {r['shap_share']:>9.2%}{mark}")

    ranks = {r["feature"]: i for i, r in enumerate(
        n.sort("shap", descending=True).iter_rows(named=True), 1)}
    pt_top = [(f, ranks[f]) for f in ranks if f.startswith("pt_") and ranks[f] <= 40]
    print(f"\nпризнаков личного времени в топ-40 по |SHAP|: {len(pt_top)}  "
          + ", ".join(f"{f}#{r}" for f, r in sorted(pt_top, key=lambda t: t[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--ptime", default="off", choices=["off", "od", "full"])
    ap.add_argument("--ptime-source", default="real", choices=["real", "shuf"])
    ap.add_argument("--val", default="2025-10-16")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("BASE", "NEW"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    assert a.exp, "нужен --exp или --compare"
    run(a.exp, None if a.ptime == "off" else a.ptime, a.ptime_source,
        dt.date.fromisoformat(a.val), a.rounds, a.seed)


if __name__ == "__main__":
    main()
