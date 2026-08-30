"""SEQ-03A этап 2 — сводка кросс-фолдового стресса по глубине.

Читает `artifacts/xdepth_*.npz`, которые пишет `src/seq.py crossdepth`, и
отвечает на вопрос, который отличает успех от ложного успеха:

    инвариантность к `avail` могла быть куплена потерей способности
    пользоваться реальными дополнительными днями.

Поэтому по каждой модели считается ОБЕ величины отдельно:

  * `gain(+77)` — калиброванный RMSLE при полной глубине панели минус тот же
    RMSLE при обрезке на обученной глубине. У BASE это −0.0037 (exp_027);
    если у аугментированной модели он схлопнулся к нулю, дни перестали
    работать, и это не успех, а другой способ проиграть;
  * положение внутреннего оптимума и кривая усадки
    `z(α) = (1−α)·z_clip + α·z_full` — та же, что в exp_027 §Этап 3.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ4/crossdepth.py --variants BASE B
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS
from src.validation import calibrate, rmsle_z

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True, help="теги, напр. BASE B")
    ap.add_argument("--ckpt-fold", default="V0904")
    ap.add_argument("--panel", default="1016")
    ap.add_argument("--out", default="research/strategies/results/SEQ4")
    a = ap.parse_args()

    curve, alpha_rows = [], []
    for v in a.variants:
        f = ARTIFACTS / f"xdepth_SEQ-03A-{v}-S42-{a.ckpt_fold}_on{a.panel}.npz"
        if not f.exists():
            print(f"нет {f.name}, пропуск")
            continue
        d = np.load(f)
        y = d["y"].astype(float)
        depths, Z = d["depths"], d["z"].astype(float)
        d_train, avail = int(d["train_max"]), int(d["avail"])
        cal = {}
        for i, D in enumerate(depths):
            z = np.maximum(Z[i], 0.0)
            cal[int(D)] = calibrate(y, z)[1]
        ref = cal[d_train]
        for D in depths:
            curve.append(dict(variant=v, depth=int(D), over_train=int(D) - d_train,
                              rmsle_cal=cal[int(D)], d_vs_clip=cal[int(D)] - ref,
                              train_max=d_train, avail=avail))
        best = min(cal, key=cal.get)
        print(f"\n=== {v}: модель {a.ckpt_fold} (обучена до {d_train}) на панели "
              f"{a.panel}, доступно {avail} (+{avail - d_train}) ===")
        for D in depths:
            mark = " <- оптимум" if int(D) == best else ""
            print(f"  глубина {int(D):>3} ({int(D) - d_train:+4d}): "
                  f"{cal[int(D)]:.5f} ({cal[int(D)] - ref:+.5f}){mark}")

        zc, zf = np.maximum(Z[list(depths).index(d_train)], 0.0), np.maximum(Z[-1], 0.0)
        row = dict(variant=v)
        for al in ALPHAS:
            row[f"a{al}"] = calibrate(y, (1 - al) * zc + al * zf)[1]
        row["best_alpha"] = min(ALPHAS, key=lambda al: row[f"a{al}"])
        row["gain_full"] = row["a1.0"] - row["a0.0"]
        row["gain_best_alpha"] = row[f"a{row['best_alpha']}"] - row["a0.0"]
        row["var_full_vs_clip"] = float(np.var(zf - zc))
        alpha_rows.append(row)
        print(f"  усадка α: " + "  ".join(f"{al}={row[f'a{al}']:.5f}" for al in ALPHAS)
              + f"   -> α*={row['best_alpha']}")
        print(f"  польза дополнительных дней (полная глубина к обрезке): "
              f"{row['gain_full']:+.5f}; Var(z_full − z_clip) = {row['var_full_vs_clip']:.5f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(curve).write_csv(out / "crossdepth_curve.csv")
    pl.DataFrame(alpha_rows).write_csv(out / "crossdepth_alpha.csv")
    print(f"\nзаписано: {out}/crossdepth_curve.csv, crossdepth_alpha.csv")


if __name__ == "__main__":
    main()
