"""Adversarial validation: насколько признаковое пространство cutoff'а отличается от тестового.

Высокий AUC означает, что модель может отличить исторический cutoff от тестового
по признакам — то есть на тесте она работает вне области обучения (eda_findings §7.4).
Основной драйвер здесь — не сдвиг поведения, а РАЗНАЯ ДОСТУПНАЯ ГЛУБИНА ИСТОРИИ,
и лечится он усечением до фиксированного L.

Запуск: python -m src.adversarial --L 180 --L 90 --L 0
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import CUTOFF_TEST, LGB_PARAMS, SEED
from src.data import load
from src.features import feature_names, make_xy, to_np

PAIRS_DEFAULT = [dt.date(2025, 9, 4), dt.date(2025, 10, 16)]


def adv_auc(A, B, feats, rounds=250):
    import lightgbm as lgb
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    idx = np.random.RandomState(SEED).permutation(len(X))
    X, y = X[idx], y[idx]
    c = int(0.7 * len(X))
    p = dict(LGB_PARAMS)
    p.update(objective="binary", metric="auc", num_leaves=63)
    m = lgb.train(p, lgb.Dataset(X[:c], y[:c]), num_boost_round=rounds)
    auc = float(roc_auc_score(y[c:], m.predict(X[c:])))
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda t: -t[1])[:8]
    return auc, [n for n, _ in imp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, action="append", default=None)
    ap.add_argument("--cuts", nargs="*", default=None)
    ap.add_argument("--norm-long", action="store_true")
    a = ap.parse_args()
    Ls = a.L or [0, 90, 180, 270]
    cuts = [dt.date.fromisoformat(s) for s in a.cuts] if a.cuts else PAIRS_DEFAULT
    load()
    print(f"{'L':>6} {'cutoff':>12} {'adv AUC vs TEST':>16}   драйверы разделения")
    print("-" * 110)
    for L in Ls:
        LL = None if L <= 0 else L
        Xt, _ = make_xy(CUTOFF_TEST, LL, with_target=False, norm_long=a.norm_long)
        feats = feature_names(Xt)
        At = to_np(Xt, feats)
        for V in cuts:
            Xv, _ = make_xy(V, LL, norm_long=a.norm_long)
            Av = to_np(Xv, feats)
            auc, drv = adv_auc(Av, At, feats)
            print(f"{L if L else 'нет':>6} {str(V):>12} {auc:>16.4f}   {', '.join(drv[:6])}")
        del Xt, At


if __name__ == "__main__":
    main()
