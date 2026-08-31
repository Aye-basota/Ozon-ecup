"""Блендинг OOF-предсказаний нескольких экспериментов.

Усреднение делается **в лог-пространстве** z, а не в сыром GMV: метрика — MSE по z,
а усреднение сырых предсказаний смещает уровень вверх (неравенство Йенсена).

Веса подбираются на OOF и только на OOF, по главной метрике **wCV**
(`src/report.py`). Дополнительно печатается `--lofo`: веса подбираются на трёх
фолдах и проверяются на четвёртом. Это единственная доступная проверка того, что
подбор весов не переобучается на те же фолды, по которым он и оптимизируется, —
без неё «улучшение смеси» неотличимо от подгонки.

Запуск: python -m src.blend --exps S1-E02 S1-E03a --lofo
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from src.config import FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.report import evaluate, format_report
from src.tracking import load_oof
from src.validation import calibrate, rmsle_z


def shifted_rmsle(ly: np.ndarray, z: np.ndarray) -> float:
    """RMSLE после ОПТИМАЛЬНОГО глобального сдвига, аналитически.

    MSE(delta) = mean((ly - z - delta)^2) минимизируется при delta = mean(ly - z),
    и тогда MSE = Var(ly - z). Сеточный поиск здесь не нужен и в 241 раз дороже —
    это важно при переборе тысяч комбинаций весов.
    """
    r = ly - z
    return float(np.sqrt(r.var()))


def aligned(exps: list[str]):
    """Выравнивает OOF по (cutoff, user_id) — порядок строк обязан совпадать."""
    ds = [load_oof(e) for e in exps]
    keys = [np.char.add(np.asarray(d["cutoff"], dtype="U10"),
                        np.asarray(d["user_id"]).astype("U20")) for d in ds]
    base = keys[0]
    order = np.argsort(base)
    y = ds[0]["y"][order]
    Z = []
    for d, k in zip(ds, keys):
        o = np.argsort(k)
        assert np.array_equal(k[o], base[order]), "OOF на разных наборах строк"
        Z.append(d["z"][o])
    return np.vstack(Z), y, np.asarray(ds[0]["cutoff"], dtype="U10")[order]


def fold_masks(cut: np.ndarray):
    folds = sorted(set(cut.tolist()))
    return folds, [cut == c for c in folds]


def fold_cal_matrix(Z, ly, masks, ws):
    """Пофолдовый калиброванный RMSLE для каждой комбинации весов: (комбинаций, фолдов)."""
    out = np.empty((len(ws), len(masks)))
    for i, w in enumerate(ws):
        z = np.average(Z, axis=0, weights=w)
        for j, m in enumerate(masks):
            out[i, j] = shifted_rmsle(ly[m], z[m])
    return out


def weight_grid(n: int, step: float, fixed: dict[int, float] | None = None):
    """Все веса на сетке `step`, суммирующиеся в 1.

    `fixed` — веса, которые НЕ подбираются (индекс модели -> значение). Нужен для
    страховки `S1-E03a`: перебор её всегда обнуляет, а `MIX-E11` показал, что
    обнуление стоит +0.00023 LB при локальном выигрыше −0.00038. Без `fixed`
    поведение прежнее.
    """
    if not fixed:
        g = np.arange(0, 1 + 1e-9, step)
        return [w for w in itertools.product(g, repeat=n) if abs(sum(w) - 1) < 1e-9]
    rest = 1.0 - sum(fixed.values())
    assert rest > -1e-9, "фиксированные веса дают больше единицы"
    free = [i for i in range(n) if i not in fixed]
    g = np.arange(0, rest + 1e-9, step)
    out = []
    for w in itertools.product(g, repeat=len(free)):
        if abs(sum(w) - rest) > 1e-9:
            continue
        full = [0.0] * n
        for i, v in fixed.items():
            full[i] = v
        for i, v in zip(free, w):
            full[i] = v
        out.append(tuple(full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--ref", nargs="*", type=float, default=None,
                    help="веса опорной смеси, чтобы печатать дельты к ней")
    ap.add_argument("--lofo", action="store_true",
                    help="подобрать веса без каждого фолда и проверить на нём")
    ap.add_argument("--fix", nargs="*", default=None, metavar="EXP=W",
                    help="не подбирать вес модели, а зафиксировать: --fix S1-E03a=0.10")
    a = ap.parse_args()
    Z, y, cut = aligned(a.exps)
    ly = np.log1p(y)
    folds, masks = fold_masks(cut)
    full = folds == [d.isoformat() for d in VAL_FOLDS_S1]
    w_f = np.asarray(FOLD_WEIGHTS_S1[:len(folds)], float)
    w_f = w_f / w_f.sum()
    print(f"n = {len(y):,} строк OOF, моделей {len(a.exps)}, фолдов {len(folds)}")
    if not full:
        print("  ВНИМАНИЕ: набор фолдов неполный, веса усечены — это НЕ wCV проекта\n")
    else:
        print()

    print(f"{'модель':>12} {'wCV':>9} {'OOF cal':>9}   пофолдово (калибр.)")
    for e, z in zip(a.exps, Z):
        fc = [shifted_rmsle(ly[m], z[m]) for m in masks]
        print(f"{e:>12} {float(np.dot(w_f, fc)):>9.5f} {calibrate(y, z)[1]:>9.5f}   "
              + " ".join(f"{v:.5f}" for v in fc))

    print("\nкорреляция остатков (в лог-пространстве):")
    R = np.vstack([ly - z for z in Z])
    for i, j in itertools.combinations(range(len(a.exps)), 2):
        print(f"  {a.exps[i]:>12} vs {a.exps[j]:<12} corr {np.corrcoef(R[i], R[j])[0, 1]:.4f}"
              f"   Var(z_i - z_j) {np.var(Z[i] - Z[j]):.5f}")

    fixed = {}
    for spec in (a.fix or []):
        name, val = spec.split("=")
        assert name in a.exps, f"--fix {name}: такой модели нет в --exps"
        fixed[a.exps.index(name)] = float(val)
    if fixed:
        print("\nфиксированные веса (не подбираются): "
              + ", ".join(f"{a.exps[i]}={v}" for i, v in fixed.items()))
    ws = weight_grid(len(a.exps), a.step, fixed)
    FC = fold_cal_matrix(Z, ly, masks, ws)
    sc = FC @ w_f
    o = np.argsort(sc)
    ref_fc = None
    if a.ref:
        assert abs(sum(a.ref) - 1) < 1e-9, "веса опорной смеси не суммируются в 1"
        ref_fc = fold_cal_matrix(Z, ly, masks, [a.ref])[0]
        print(f"\nопорная смесь {np.round(a.ref, 2)}: wCV={float(np.dot(w_f, ref_fc)):.5f}   "
              + " ".join(f"{v:.5f}" for v in ref_fc))

    print(f"\nлучшие смеси по wCV ({len(ws)} комбинаций, шаг {a.step}):")
    for i in o[:6]:
        d = ("   " + " ".join(f"{v - r:+.5f}" for v, r in zip(FC[i], ref_fc))
             + f"   {(FC[i] < ref_fc).sum()}/{len(folds)}") if ref_fc is not None else ""
        print(f"  w={np.round(ws[i], 2)}  wCV={sc[i]:.5f}" + d)
    plateau = int((sc <= sc[o[0]] + 5e-5).sum())
    print(f"  плато: {plateau} комбинаций в пределах 0.00005 от оптимума из {len(ws)}")

    if a.lofo and ref_fc is not None:
        print("\nLOFO: веса подобраны без фолда, проверены на нём (честная оценка выигрыша)")
        print(f"{'отложенный фолд':<16}{'веса без него':<34}{'на нём':>10}{'опорная':>10}{'дельта':>10}")
        held = np.zeros(len(folds))
        for h in range(len(folds)):
            idx = [i for i in range(len(folds)) if i != h]
            wh = w_f[idx] / w_f[idx].sum()
            b = int(np.argmin(FC[:, idx] @ wh))
            held[h] = FC[b, h]
            print(f"{folds[h]:<16}{str(np.round(ws[b], 2)):<34}{FC[b, h]:>10.5f}"
                  f"{ref_fc[h]:>10.5f}{FC[b, h] - ref_fc[h]:>+10.5f}")
        print(f"  честный выигрыш по wCV: {float(np.dot(w_f, held - ref_fc)):+.5f}"
              f"   (в выборке: {sc[o[0]] - float(np.dot(w_f, ref_fc)):+.5f})")

    best = ws[o[0]]
    zb = np.average(Z, axis=0, weights=best)
    print(format_report(evaluate(y, zb, cut)))


if __name__ == "__main__":
    main()
