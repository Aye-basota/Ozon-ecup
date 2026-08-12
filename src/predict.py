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

from src.config import ANCHOR_BAND, ARTIFACTS, CUTOFF_TEST, SEED, SUBMISSIONS
from src.data import load, sample_submit
from src.features import feature_names, make_xy, to_np
from src.train import Setup, assemble, fit, infer

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def train_full(s: Setup, feats: list[str], models: list[str], for_fold: dt.date | None = None):
    """Обучает по одной модели каждого типа на всех чистых cutoff'ах коридора.

    `for_fold` — обучить ровно на обучающей выборке этого фолда (`T + 30 <= V`)
    вместо всего коридора. Нужно, чтобы перенести на тест ту самую модель,
    чей скор виден на фолде. Данных при этом МЕНЬШЕ: у фолда 2025-10-16 это
    24 cutoff'а из 29, без пяти самых свежих.
    """
    cuts = s.train_cutoffs(for_fold) if for_fold else s.grid()
    log(f"train cutoffs: {len(cuts)} шт, {min(cuts)}..{max(cuts)}"
        + (f"  (обучающая выборка фолда {for_fold})" if for_fold else "  (весь коридор)"))
    Xtr, ytr, wtr = assemble(cuts, s, feats)
    from src.train import _XY
    _XY.clear()          # кэш обучающих фреймов больше не нужен, это ~2.3 ГБ
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
    ap.add_argument("--norm-long", action="store_true")
    ap.add_argument("--delta", type=float, default=0.0, help="глобальный лог-сдвиг")
    ap.add_argument("--seeds", nargs="*", type=int, default=[SEED])
    ap.add_argument("--out", default=None, help="если задан — сразу пишет сабмит")
    ap.add_argument("--variant", default=None, help="имя, под которым сохранить z на тесте")
    ap.add_argument("--train-for-fold", default=None,
                    help="обучить на обучающей выборке этого фолда (YYYY-MM-DD), "
                         "а не на всём коридоре")
    a = ap.parse_args()
    for_fold = dt.date.fromisoformat(a.train_for_fold) if a.train_for_fold else None

    from src.train import select_features
    s = Setup(L=a.L, min_history=a.min_history, step=a.step, rounds=a.rounds,
              train_blocks=a.train_blocks, drop_groups=a.drop or [], norm_long=a.norm_long)
    load()
    Xt, _ = make_xy(CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
    feats = select_features(feature_names(Xt), s.drop_groups, None)
    At = to_np(Xt, feats)
    log(f"тестовая матрица {At.shape[0]:,} x {At.shape[1]}")

    zs, names = [], []
    for seed in a.seeds:
        # Match validation exactly: LightGBM derives all subordinate RNG streams
        # from the master seed unless they are explicitly overridden.
        s.params = dict(s.params, seed=seed)
        trained = train_full(s, feats, a.blend, for_fold)
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

    variant = a.variant or a.exp
    np.save(ARTIFACTS / f"ztest_{variant}.npy", z)
    np.save(ARTIFACTS / f"uid_{variant}.npy", Xt["user_id"].to_numpy())
    log(f"z на тесте сохранён: artifacts/ztest_{variant}.npy")
    if not a.out:
        return
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
    log(f"сабмит записан: {out}")


if __name__ == "__main__":
    main()
