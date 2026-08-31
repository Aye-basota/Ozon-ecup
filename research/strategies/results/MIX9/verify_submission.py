"""MIX9: независимая проверка готового сабмита — по файлу, а не по памяти процесса.

`src.submit` печатает свои проверки на лету, но их видит только тот, кто смотрел
в консоль. Здесь файл читается С ДИСКА и сверяется с заявленным рецептом:
состав и веса, лог-пространство, уровень, панель, отсутствие NaN/inf и
отрицательных. `exp_027` восстанавливал отправленные сабмиты ровно этим приёмом
(`max|log1p(файл) − реконструкция| = 5e-07`), и без такой сверки «что именно
отправлено» становится вопросом веры.

Запуск:
  PYTHONPATH=. python research/strategies/results/MIX9/verify_submission.py \
      --csv submission_SEQAVG3_clip289_mix.csv \
      --z S1-CAP S1-UNC S1-DIST SEQ-01 SEQ-C289-S43 SEQ-C289-S44 \
      --weights 0.10 0.20 0.25 0.15 0.15 0.15 --level 2.3293
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from src.config import ANCHOR_BAND, ARTIFACTS, CUTOFF_TEST, SUBMISSIONS
from src.data import sample_submit
from src.features import panel_users

TOL_Z = 1e-06          # запас над float32 -> csv float_precision=6 (exp_027: факт 5e-07)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--z", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--level", type=float, default=2.3293)
    a = ap.parse_args()
    assert len(a.z) == len(a.weights), "число весов не совпадает с числом компонент"

    sub = pl.read_csv(SUBMISSIONS / a.csv)
    ss = sample_submit()
    p = sub["predict"].to_numpy()
    ok = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        ok.append(cond)
        print(f"  [{'OK ' if cond else 'ПРОВАЛ'}] {name}{('  ' + detail) if detail else ''}")

    print(f"файл: submissions/{a.csv}")
    check("колонки == [user_id, predict]", list(sub.columns) == ["user_id", "predict"],
          str(sub.columns))
    check("строк ровно 250 000", sub.height == 250_000, f"{sub.height:,}")
    check("user_id уникальны", sub["user_id"].n_unique() == sub.height)
    check("порядок и состав user_id == sample_submit",
          sub["user_id"].to_list() == ss["user_id"].to_list())
    check("нет NaN/inf", bool(np.isfinite(p).all()))
    check("нет отрицательных", bool((p >= 0).all()), f"min={p.min():.6f}")

    lvl = float(np.log1p(p).mean())
    lo, hi = ANCHOR_BAND
    check(f"уровень mean(log1p(pred)) == {a.level}", abs(lvl - a.level) < 5e-5,
          f"факт {lvl:.6f}, отклонение {lvl - a.level:+.2e}")
    check(f"уровень в якорном коридоре {lo}..{hi}", lo <= lvl <= hi)

    # --- реконструкция рецепта: смесь В ЛОГ-ПРОСТРАНСТВЕ + один общий сдвиг ------
    uid = np.load(ARTIFACTS / f"uid_{a.z[0]}.npy")
    u_panel = panel_users(CUTOFF_TEST, 3)["user_id"].to_numpy()
    check("uid компонент == панель теста", np.array_equal(uid, u_panel))

    Z = []
    for n in a.z:
        z = np.load(ARTIFACTS / f"ztest_{n}.npy")
        u = np.load(ARTIFACTS / f"uid_{n}.npy")
        assert np.array_equal(u, uid), f"{n}: другой порядок user_id"
        Z.append(z)
        print(f"      {n:>16}: mean z = {z.mean():.5f}")
    w = np.asarray(a.weights, float)
    check("веса суммируются в 1", abs(w.sum() - 1) < 1e-9, f"{w.sum():.6f}")
    z_mix = np.average(np.vstack(Z), axis=0, weights=w)
    delta = a.level - float(z_mix.mean())
    z_rec = np.maximum(z_mix + delta, 0.0)

    order = {u: i for i, u in enumerate(sub["user_id"].to_numpy())}
    pos = np.array([order[u] for u in uid])
    z_file = np.log1p(p[pos])
    err = float(np.abs(z_file - z_rec).max())
    check("файл воспроизводится из заявленных компонент и весов", err < TOL_Z,
          f"max|log1p(файл) - реконструкция| = {err:.2e}, сдвиг delta = {delta:+.5f}")

    zero = float((p == 0).mean())
    print(f"\n  сводка: mean log1p = {lvl:.6f}, min = {p.min():.6f}, max = {p.max():,.1f}, "
          f"нулей = {zero:.3%}, delta = {delta:+.5f}")
    print(f"\n{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all(ok) else 'ЕСТЬ ПРОВАЛЫ — САБМИТ НЕ ГОТОВ'}")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
