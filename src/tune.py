"""Подбор числа раундов и ключевых гиперпараметров.

Дешёвый трюк: одна модель на 2000 раундов даёт всю кривую качества сразу —
LightGBM умеет предсказывать на любом префиксе деревьев (`num_iteration`).
Поэтому кривая «RMSLE от числа раундов» стоит ровно одного обучения, а не десяти.

Порядок приоритетов (research/strategy_1.md §6): сначала rounds x learning_rate,
затем min_data_in_leaf, затем num_leaves + lambda_l2. Остальное — третий знак.

Запуск: python -m src.tune --mode rounds --L 180
        python -m src.tune --mode grid   --L 180
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from src.data import load
from src.features import feature_names, to_np
from src.train import Setup, assemble, xy
from src.validation import bias_z, get_folds, rmsle_z

T0 = time.time()


def curve(s: Setup, feats, folds, rounds, checkpoints):
    from src import models
    per_fold = {c: [] for c in checkpoints}
    for _, V in folds:
        tr = s.train_cutoffs(V)
        Xtr, ytr, _ = assemble(tr, s, feats, V)
        box = [Xtr]
        del Xtr
        dss = models.make_datasets("direct", box[0], ytr, None, s.params)
        box[0] = None
        gc.collect()
        m = models.train_direct_ds(dss[0], s.params, rounds)
        Xv, yv = xy(V, s)
        Av = to_np(Xv, feats)
        for c in checkpoints:
            z = np.maximum(m.predict(Av, num_iteration=c), 0.0)
            per_fold[c].append(rmsle_z(yv, z))
        print(f"  [{time.time() - T0:5.0f}s] фолд {V} готов", flush=True)
        del m, dss, Av
        gc.collect()
    return per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rounds", choices=["rounds", "grid"])
    ap.add_argument("--L", type=int, default=180)
    ap.add_argument("--min-history", type=int, default=90)
    ap.add_argument("--train-blocks", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=2)
    a = ap.parse_args()
    load()
    s = Setup(L=a.L, min_history=a.min_history, train_blocks=a.train_blocks,
              params={"learning_rate": a.lr})
    folds = get_folds(s.min_history, s.step)[-a.folds:]
    feats = feature_names(xy(folds[0][1], s)[0])

    if a.mode == "rounds":
        cps = [200, 400, 600, 900, 1200, 1600, 2000, 2500]
        cps = [c for c in cps if c <= a.rounds]
        print(f"кривая по раундам, lr={a.lr}, L={a.L}, фолдов={len(folds)}, feats={len(feats)}")
        pf = curve(s, feats, folds, a.rounds, cps)
        print(f"\n{'rounds':>7} {'CV mean':>9} {'CV std':>8}   по фолдам")
        best = min(cps, key=lambda c: np.mean(pf[c]))
        for c in cps:
            v = pf[c]
            print(f"{c:>7} {np.mean(v):>9.5f} {np.std(v):>8.5f}   "
                  f"{' '.join(f'{x:.5f}' for x in v)}{'   <- лучший' if c == best else ''}")
        return

    grid = [
        ("база", {}),
        ("min_data_in_leaf=50", {"min_data_in_leaf": 50}),
        ("min_data_in_leaf=500", {"min_data_in_leaf": 500}),
        ("num_leaves=63", {"num_leaves": 63}),
        ("num_leaves=255", {"num_leaves": 255}),
        ("lambda_l2=20", {"lambda_l2": 20.0}),
        ("feature_fraction=0.5", {"feature_fraction": 0.5}),
        ("feature_fraction=0.9", {"feature_fraction": 0.9}),
    ]
    print(f"{'конфигурация':>24} {'CV mean':>9} {'CV std':>8} {'bias':>8}")
    from src import models
    for name, over in grid:
        s.params = {"learning_rate": a.lr, **over}
        sc, bi = [], []
        for _, V in folds:
            Xtr, ytr, _ = assemble(s.train_cutoffs(V), s, feats, V)
            box = [Xtr]
            del Xtr
            dss = models.make_datasets("direct", box[0], ytr, None, s.params)
            box[0] = None
            gc.collect()
            m = models.train_direct_ds(dss[0], s.params, a.rounds)
            Xv, yv = xy(V, s)
            z = np.maximum(m.predict(to_np(Xv, feats)), 0.0)
            sc.append(rmsle_z(yv, z)); bi.append(bias_z(yv, z))
            del m, dss
            gc.collect()
        print(f"{name:>24} {np.mean(sc):>9.5f} {np.std(sc):>8.5f} {np.mean(bi):>+8.4f}", flush=True)


if __name__ == "__main__":
    main()
