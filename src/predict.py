"""Inference: обучение на всём чистом коридоре и построение сабмита.

Обучающая выборка — все чистые cutoff'ы (T <= 2025-10-16), признаки на тестовом
cutoff'е 2026-02-13 строятся тем же кодом и с той же глубиной истории L.

Постпроцессинг:
  z  -> z + delta        (лог-калибровка, delta подобрана на OOF/S2, НЕ на LB)
  pred = expm1(z), клип снизу нулём.

Запуск:
  python -m src.predict --exp S1-BEST --L 180 --min-history 90 --blend direct two_part \
                        --delta -0.12 --out submission_strategy_1.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import time

import numpy as np
import polars as pl

from src.config import ANCHOR_BAND, ARTIFACTS, CUTOFF_TEST, SUBMISSIONS
from src.data import load, sample_submit
from src.features import feature_names, make_xy, to_np
from src.train import Setup, assemble, fit, infer

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def train_full(s: Setup, feats: list[str], models: list[str]):
    """Обучает по одной модели каждого типа на всех чистых cutoff'ах коридора."""
    cuts = s.grid()
    log(f"train cutoffs: {len(cuts)} шт, {min(cuts)}..{max(cuts)}")
    Xtr, ytr, wtr = assemble(cuts, s, feats)
    log(f"train matrix {Xtr.shape[0]:,} x {Xtr.shape[1]}  "
        f"mean log1p(y)={np.log1p(ytr).mean():.4f}")
    out = {}
    for mt in models:
        t = time.time()
        s_i = Setup(**{**s.as_dict(), "model": mt, "vals": None})
        s_i.params = s.params
        out[mt] = (s_i, fit(s_i, Xtr, ytr, wtr if s.weight_tau else None))
        log(f"  обучена {mt} за {time.time() - t:.0f}s")
    del Xtr, ytr, wtr
    gc.collect()
    return out


def check_submission(sub: pl.DataFrame) -> None:
    ss = sample_submit()
    p = sub["predict"].to_numpy()
    assert sub.height == 250_000, f"строк {sub.height}, ожидается 250 000"
    assert sub["user_id"].n_unique() == 250_000, "дубликаты user_id"
    assert set(sub["user_id"].to_list()) == set(ss["user_id"].to_list()), "состав user_id != sample_submit"
    assert sub["user_id"].to_list() == ss["user_id"].to_list(), "порядок user_id != sample_submit"
    assert np.isfinite(p).all(), "NaN/inf в predict"
    assert (p >= 0).all(), "отрицательные предсказания"
    assert list(sub.columns) == ["user_id", "predict"], f"колонки {sub.columns}"
    log(f"  проверки пройдены: n={sub.height:,}  min={p.min():.4f}  max={p.max():.1f}  "
        f"нулей={float((p == 0).mean()):.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="S1-BEST")
    ap.add_argument("--L", type=int, default=180)
    ap.add_argument("--min-history", type=int, default=90)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--train-blocks", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--blend", nargs="+", default=["direct"], help="direct two_part catboost")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--drop", nargs="*", default=None)
    ap.add_argument("--delta", type=float, default=0.0, help="глобальный лог-сдвиг")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--out", default="submission_strategy_1.csv")
    a = ap.parse_args()

    from src.train import select_features
    s = Setup(L=a.L, min_history=a.min_history, step=a.step, rounds=a.rounds,
              train_blocks=a.train_blocks, drop_groups=a.drop or [])
    load()
    Xt, _ = make_xy(CUTOFF_TEST, s.L, s.panel_blocks, with_target=False)
    feats = select_features(feature_names(Xt), s.drop_groups, None)
    At = to_np(Xt, feats)
    log(f"тестовая матрица {At.shape[0]:,} x {At.shape[1]}")

    zs, names = [], []
    for seed in a.seeds:
        s.params = dict(s.params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        trained = train_full(s, feats, a.blend)
        for mt, (s_i, m) in trained.items():
            z = np.maximum(infer(s_i, m, At), 0.0)
            zs.append(z); names.append(f"{mt}/seed{seed}")
            log(f"  {mt} seed={seed}: mean(log1p(pred))={z.mean():.4f}")
            del m
        gc.collect()

    w = np.array(a.weights, float) if a.weights else np.ones(len(zs))
    w = w / w.sum()
    assert len(w) == len(zs), f"весов {len(w)}, моделей {len(zs)}"
    z = np.average(np.vstack(zs), axis=0, weights=w)     # усреднение В ЛОГ-ПРОСТРАНСТВЕ
    log(f"ансамбль {dict(zip(names, np.round(w, 3)))}: mean(log1p(pred))={z.mean():.4f}")

    z_cal = np.maximum(z + a.delta, 0.0)
    lo, hi = ANCHOR_BAND
    log(f"после delta={a.delta:+.4f}: mean(log1p(pred))={z_cal.mean():.4f}  "
        f"якорь {lo}..{hi}  {'В КОРИДОРЕ' if lo <= z_cal.mean() <= hi else '!!! ВНЕ КОРИДОРА'}")

    pred = np.maximum(np.expm1(z_cal), 0.0)
    sub = pl.DataFrame({"user_id": Xt["user_id"], "predict": pred.astype(np.float64)})
    sub = sub.join(sample_submit().select("user_id").with_row_index("o"), on="user_id").sort("o").drop("o")
    check_submission(sub)
    out = SUBMISSIONS / a.out
    sub.write_csv(out, float_precision=6)
    np.save(ARTIFACTS / f"ztest_{a.exp}.npy", z_cal)
    log(f"сабмит записан: {out}")


if __name__ == "__main__":
    main()
