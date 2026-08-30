"""Финальная сборка сабмита из сохранённых предсказаний на тесте.

Разделение с `predict.py` намеренное: обучение моделей дорогое, а подбор веса
смеси и калибровочного сдвига — нет. Поэтому z на тесте считаются один раз
(`predict.py --variant NAME`), а здесь только комбинируются.

Постпроцессинг (research/strategy_1.md §7):
  z_mix = Σ w_i z_i         усреднение В ЛОГ-ПРОСТРАНСТВЕ (в сыром GMV сместило бы
                            уровень вверх по неравенству Йенсена)
  z_cal = max(z_mix + δ, 0) δ — лог-калибровка уровня под якорь из sample_submit
  pred  = expm1(z_cal)      неотрицательно по построению

Запуск:
  python -m src.submit --z S1-NORM S1-CAP --weights 0.65 0.35 --level 2.33 \
                       --out submission_strategy_1.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from src.config import ANCHOR_BAND, ARTIFACTS, CUTOFF_TEST, SUBMISSIONS
from src.data import sample_submit
from src.features import panel_users


def check_submission(sub: pl.DataFrame) -> None:
    ss = sample_submit()
    p = sub["predict"].to_numpy()
    assert list(sub.columns) == ["user_id", "predict"], f"колонки {sub.columns}"
    assert sub.height == 250_000, f"строк {sub.height}, ожидается 250 000"
    assert sub["user_id"].n_unique() == 250_000, "дубликаты user_id"
    assert sub["user_id"].to_list() == ss["user_id"].to_list(), "порядок/состав user_id != sample_submit"
    assert np.isfinite(p).all(), "NaN/inf в predict"
    assert (p >= 0).all(), "отрицательные предсказания"
    print(f"  проверки: n={sub.height:,}  min={p.min():.6f}  max={p.max():,.1f}  "
          f"нулей={float((p == 0).mean()):.3%}  mean log1p={np.log1p(p).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", nargs="+", required=True, help="имена вариантов из artifacts/ztest_*.npy")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--level", type=float, default=None,
                    help="целевой mean(log1p(pred)); δ считается автоматически")
    ap.add_argument("--delta", type=float, default=None, help="или явный δ вместо --level")
    ap.add_argument("--out", default="submission_strategy_1.csv")
    a = ap.parse_args()

    Z = [np.load(ARTIFACTS / f"ztest_{n}.npy") for n in a.z]
    uid = np.load(ARTIFACTS / f"uid_{a.z[0]}.npy")
    for n, z in zip(a.z, Z):
        assert len(z) == len(uid), f"{n}: длина {len(z)} != {len(uid)}"
        print(f"  {n:>12}: mean(log1p(pred))={z.mean():.4f}")
    u_panel = panel_users(CUTOFF_TEST, 3)["user_id"].to_numpy()
    assert np.array_equal(uid, u_panel), "порядок user_id не совпадает с панелью теста"

    w = np.array(a.weights, float) if a.weights else np.ones(len(Z))
    w = w / w.sum()
    assert len(w) == len(Z), f"весов {len(w)}, вариантов {len(Z)}"
    z = np.average(np.vstack(Z), axis=0, weights=w)
    print(f"\n  смесь {dict(zip(a.z, np.round(w, 3)))}: mean(log1p)={z.mean():.4f}")

    if a.delta is not None:
        delta = a.delta
    elif a.level is not None:
        delta = a.level - float(z.mean())
    else:
        delta = 0.0
    z_cal = np.maximum(z + delta, 0.0)
    lo, hi = ANCHOR_BAND
    ok = lo <= z_cal.mean() <= hi
    print(f"  delta = {delta:+.4f}  ->  mean(log1p(pred)) = {z_cal.mean():.4f}")
    print(f"  якорный коридор {lo}..{hi}: {'В КОРИДОРЕ' if ok else '!!! ВНЕ КОРИДОРА'}")
    assert ok, "уровень вне якорного коридора — это ошибка, а не открытие (strategy_1 §7.2)"

    pred = np.maximum(np.expm1(z_cal), 0.0)
    sub = pl.DataFrame({"user_id": uid, "predict": pred.astype(np.float64)})
    order = sample_submit().select("user_id").with_row_index("o")
    sub = sub.join(order, on="user_id", how="inner").sort("o").drop("o")
    check_submission(sub)
    out = SUBMISSIONS / a.out
    sub.write_csv(out, float_precision=6)
    print(f"  сабмит записан: {out}")
    print(sub.head(3))


if __name__ == "__main__":
    main()
