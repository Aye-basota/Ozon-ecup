"""Усреднение одной и той же конфигурации по нескольким сидам.

Зачем отдельный модуль, а не `blend.py`. Блендинг соединяет РАЗНЫЕ модели, и его
веса подбираются. Здесь модель одна, а сиды — повторные реализации одного и того
же оценщика, поэтому веса равные **по построению** и подбирать их нечего:

    E[(z̄_k − z*)²] = Bias² + Var/k,

то есть усреднение k сидов убирает долю (1 − 1/k) дисперсии оценщика и не может
увеличить ожидаемую ошибку. Смещение оно не трогает вообще.

Усреднение — в лог-пространстве z, как и в `blend.py`: метрика есть MSE по z,
среднее сырых предсказаний сместило бы уровень вверх (неравенство Йенсена).

Результат сохраняется как обычный эксперимент (`oof_<out>.npz` + отчёт), чтобы
усреднённая модель участвовала в смеси наравне с остальными.

Запуск:
  python -m src.seedavg --out S1-CAP-S3 --exps S1-CAP-S42 S1-CAP-S43 S1-CAP-S44 \
                        --ref S1-E10 --desc "3-сидовое среднее" --stats
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from src.blend import aligned
from src.report import evaluate, format_report, save_report
from src.tracking import log_from_report, save_oof


def average(Z: np.ndarray) -> np.ndarray:
    """Среднее по сидам в лог-пространстве. Веса равные и не обсуждаются."""
    return Z.mean(axis=0)


def wcv_of(z, y, cut) -> float | None:
    """wCV одного набора предсказаний — тем же кодом, что и у любого эксперимента."""
    return evaluate(y, z, cut)["wcv"]


def per_seed_table(exps, Z, y, cut) -> list[dict]:
    out = []
    for e, z in zip(exps, Z):
        r = evaluate(y, z, cut)
        out.append(dict(exp=e, wcv=r["wcv"], fold_cal=r["fold_cal"], mean_z=r["mean_z"]))
    return out


def pairwise(Z: np.ndarray, y) -> list[tuple[int, int, float, float]]:
    """(i, j, Var(z_i − z_j), corr остатков) — нижняя граница «разнообразия» проекта.

    Два сида одной конфигурации не несут никакой новой информации, поэтому их
    Var(Δ) — это чистый шум оценщика. Любой Var(Δ) другой стратегии, не
    превосходящий это число, разнообразием считать нельзя.
    """
    ly = np.log1p(y)
    out = []
    for i, j in itertools.combinations(range(len(Z)), 2):
        r_i, r_j = ly - Z[i], ly - Z[j]
        out.append((i, j, float(np.var(Z[i] - Z[j])), float(np.corrcoef(r_i, r_j)[0, 1])))
    return out


def averaging_curve(Z: np.ndarray, y, cut) -> list[dict]:
    """wCV среднего из k сидов, по ВСЕМ сочетаниям из k — кривая обязана быть
    монотонной по k и выходить на плато; немонотонность означала бы, что сиды
    различаются не только шумом."""
    rows = []
    for k in range(1, len(Z) + 1):
        vals = [wcv_of(average(Z[list(c)]), y, cut)
                for c in itertools.combinations(range(len(Z)), k)]
        v = np.asarray(vals, float)
        rows.append(dict(k=k, n_subsets=len(vals), mean=float(v.mean()), std=float(v.std()),
                         lo=float(v.min()), hi=float(v.max())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exps", nargs="+", required=True, help="OOF отдельных сидов")
    ap.add_argument("--ref", default=None, help="опорный эксперимент для дельт")
    ap.add_argument("--desc", default="")
    ap.add_argument("--model", default="direct")
    ap.add_argument("--params", default="")
    ap.add_argument("--n-features", type=int, default=0)
    ap.add_argument("--stats", action="store_true",
                    help="разброс между сидами и кривая «скор от числа усреднённых сидов»")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    Z, y, cut = aligned(a.exps)
    print(f"сидов {len(a.exps)}, строк OOF {len(y):,}")

    if a.stats:
        print(f"\n{'сид':>14} {'wCV':>9}   пофолдово (калибр.)          mean z")
        rows = per_seed_table(a.exps, Z, y, cut)
        for r in rows:
            w = f"{r['wcv']:.5f}" if r["wcv"] is not None else "     н/д"
            print(f"{r['exp']:>14} {w:>9}   " + " ".join(f"{v:.5f}" for v in r["fold_cal"])
                  + f"   {r['mean_z']:.4f}")
        ws = np.array([r["wcv"] for r in rows], float)
        if not np.any(np.equal(ws, None)):
            print(f"{'разброс':>14} mean={ws.mean():.5f} std={ws.std(ddof=1):.5f} "
                  f"range={ws.max() - ws.min():.5f}")
        F = np.array([r["fold_cal"] for r in rows], float)
        print("  std по сидам пофолдово: " + " ".join(f"{v:.5f}" for v in F.std(axis=0, ddof=1)))

        print("\nпопарно (сиды несут ноль информации друг о друге — это чистый шум):")
        for i, j, vd, cr in pairwise(Z, y):
            print(f"  {a.exps[i]:>14} vs {a.exps[j]:<14} Var(z_i - z_j) {vd:.5f}"
                  f"   corr остатков {cr:.5f}")

        print("\nкривая усреднения (по всем сочетаниям из k сидов):")
        print(f"{'k':>3} {'сочетаний':>10} {'wCV mean':>10} {'std':>9} {'мин':>10} {'макс':>10}")
        for r in averaging_curve(Z, y, cut):
            print(f"{r['k']:>3} {r['n_subsets']:>10} {r['mean']:>10.5f} {r['std']:>9.5f} "
                  f"{r['lo']:>10.5f} {r['hi']:>10.5f}")

    z = average(Z)
    rep = evaluate(y, z, cut)
    ref_rep = None
    if a.ref:
        from src.report import from_oof
        ref_rep = from_oof(a.ref)
    print(format_report(rep, ref_rep))

    save_oof(a.out, _uid(a.exps[0], cut, y), cut, z, y)
    save_report(a.out, rep, extra=dict(description=a.desc, seeds=a.exps, ref=a.ref,
                                       n_seeds=len(a.exps)))
    print(f"\nOOF сохранён: artifacts/oof_{a.out}.npz, отчёт: artifacts/report_{a.out}.json")
    if not a.no_log:
        log_from_report(a.out, a.desc, rep, scenario="S1", n_features=a.n_features,
                        model=a.model, params=a.params)
        print("строка записана в experiments/log.csv")


def _uid(exp: str, cut, y):
    """user_id в порядке `aligned` — он сортирует строки по ключу (cutoff, user_id)."""
    from src.tracking import load_oof
    d = load_oof(exp)
    k = np.char.add(np.asarray(d["cutoff"], dtype="U10"), np.asarray(d["user_id"]).astype("U20"))
    o = np.argsort(k)
    assert np.array_equal(np.asarray(d["cutoff"], dtype="U10")[o], cut), "порядок строк разошёлся"
    return d["user_id"][o]


if __name__ == "__main__":
    main()
