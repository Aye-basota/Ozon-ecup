"""Калибровка валидатора по leaderboard: какая локальная метрика переносится на LB.

Читает реестр сабмитов `experiments/submissions.csv` и OOF из `artifacts/`,
пересчитывает локальные метрики каждого отправленного варианта и меряет
**коэффициент переноса** — во сколько раз локальная дельта превращается в дельту LB.
Порядок (Spearman) при пяти точках неинформативен: все разумные метрики дают 1.0,
потому что три из пяти разрывов больше 0.01. Информативен именно разброс переноса.

Здесь же оценивается **парный шум LB**. Все сабмиты считаются на одних и тех же
50 000 пользователей public, поэтому сравнение двух сабмитов — парное, и его
стандартная ошибка на порядок меньше непарной: 0.00025 против 0.0058. Именно
парная величина, а не непарная, задаёт порог «различимо на LB».

Запуск: python -m src.cv_lb
"""
from __future__ import annotations

import csv
import datetime as dt

import numpy as np

from src.config import ARTIFACTS, EXPERIMENTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.report import evaluate
from src.tracking import load_oof

REGISTRY = EXPERIMENTS / "submissions.csv"
LSTAR = 2.3293            # E[log1p(y_test)], восстановлен тремя уровневыми сабмитами
N_PUBLIC = 50_000         # размер public-части (strategy_1_results §5.5)
RMSLE_LB = 1.651          # масштаб LB, нужен для перевода MSE-дельты в RMSLE-дельту

# Точки, у которых локальные метрики есть, а OOF-массива нет (посчитаны другим кодом
# или на другой ветке). Пофолдовые калиброванные скоры восстановлены как
# sqrt(RMSLE^2 - bias^2): оптимальный сдвиг ровно снимает bias.
EXTERNAL = {
    "EXP-MIN": dict(fold_scores=[1.78521, 1.77760, 1.76795, 1.76262], bias=-0.09545,
                    oof_cal=1.77065, src="artifacts/final_experiments_results.json"),
    "S2-BEST": dict(fold_scores=[1.78113, 1.77351, 1.76246, 1.75612],
                    fold_bias=[-0.10385, -0.05499, 0.01729, -0.03527], bias=-0.04420,
                    oof_cal=None, src="ветка team-a-strategy-2-impl, s2_oof_best.json"),
    "EXP-SIM": dict(oof_cal=1.77210, src="artifacts/final_experiments_results.json"),
}


def level_free(lb: float, level: float | None) -> float:
    """LB без вклада ошибки уровня: MSE(L) = C + (L*-L)^2 (проверено до 7-го знака)."""
    if level is None:
        return lb
    return float(np.sqrt(max(lb ** 2 - (level - LSTAR) ** 2, 0.0)))


def read_registry() -> list[dict]:
    with REGISTRY.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["lb_public"]]


def blend_from_spec(spec: str):
    """'blend:A=0.4|B=0.6' -> (z смеси, y, cutoff) по сохранённым OOF."""
    parts = [p.split("=") for p in spec.split(":", 1)[1].split("|")]
    exps = [e for e, _ in parts]
    w = np.array([float(v) for _, v in parts])
    ds = [load_oof(e) for e in exps]
    keys = [np.char.add(np.asarray(d["cutoff"], dtype="U10"),
                        np.asarray(d["user_id"]).astype("U20")) for d in ds]
    o0 = np.argsort(keys[0])
    Z = []
    for d, k in zip(ds, keys):
        o = np.argsort(k)
        assert np.array_equal(k[o], keys[0][o0]), f"{exps}: OOF на разных наборах строк"
        Z.append(d["z"][o])
    return (np.average(np.vstack(Z), axis=0, weights=w), ds[0]["y"][o0],
            np.asarray(ds[0]["cutoff"], dtype="U10")[o0])


def external_metrics(name: str) -> dict | None:
    e = EXTERNAL.get(name)
    if not e or "fold_scores" not in e:
        return dict(oof_cal=e["oof_cal"]) if e else None
    fs = np.array(e["fold_scores"])
    fb = np.array(e.get("fold_bias", [e["bias"]] * len(fs)))
    fc = np.sqrt(fs ** 2 - fb ** 2)
    w = np.array(FOLD_WEIGHTS_S1, float); w /= w.sum()
    return dict(fold_cal=fc.tolist(), wcv=float(np.dot(w, fc)),
                cv_mean=float(fs.mean()), cv_cal_mean=float(fc.mean()),
                oof_cal=e["oof_cal"], approx=True)


def collect() -> dict[str, dict]:
    """exp_id -> {lb, lb_free, локальные метрики}. Уровневые дубли схлопываются."""
    out: dict[str, dict] = {}
    for r in read_registry():
        exp, lb = r["exp_id"], float(r["lb_public"])
        lvl = float(r["level"]) if r["level"] else None
        free = level_free(lb, lvl)
        if exp in out and out[exp]["lb_free"] <= free:
            continue                       # из уровневых сабмитов берём лучший
        rec = dict(lb=lb, level=lvl, lb_free=free, file=r["file"], note=r["note"])
        if r["oof_source"].startswith("blend:"):
            z, y, cut = blend_from_spec(r["oof_source"])
            rec.update({k: v for k, v in evaluate(y, z, cut).items()
                        if k in ("wcv", "cv_mean", "cv_cal_mean", "oof_rmsle", "oof_cal",
                                 "fold_cal", "oof_bias", "mean_z")})
        else:
            m = external_metrics(exp)
            if m:
                rec.update(m)
        out[exp] = rec
    return out


def transfer_table(pts: dict, base: str = "S1-BEST") -> None:
    """Коэффициент переноса локальной дельты в дельту LB для каждой схемы."""
    schemes = {
        "wCV 1:2:4:8": lambda m: m.get("wcv"),
        "CV cal (равные)": lambda m: m.get("cv_cal_mean"),
        "CV raw": lambda m: m.get("cv_mean"),
        "OOF cal": lambda m: m.get("oof_cal"),
        **{f"только фолд {d}": (lambda m, i=i: (m.get("fold_cal") or [None] * 4)[i])
           for i, d in enumerate(f.isoformat()[5:] for f in VAL_FOLDS_S1)},
    }
    b = pts[base]
    # точки без единой локальной метрики в переносе не участвуют: колонка была бы пустой
    others = [k for k, m in pts.items()
              if k != base and (m.get("wcv") is not None or m.get("oof_cal") is not None)]
    print(f"{'схема':<20}" + "".join(f"{k[:12]:>13}" for k in others)
          + f"{'средний':>10}{'разброс':>10}")
    for name, f in schemes.items():
        rats, cells = [], []
        for k in others:
            dl = None if f(pts[k]) is None or f(b) is None else f(pts[k]) - f(b)
            dlb = pts[k]["lb_free"] - b["lb_free"]
            if dl is None or abs(dl) < 1e-9:
                cells.append(f"{'—':>13}")
                continue
            rats.append(dlb / dl)
            cells.append(f"{dlb / dl:>13.3f}")
        if not rats:
            continue
        mu, sd = float(np.mean(rats)), float(np.std(rats))
        print(f"{name:<20}" + "".join(cells) + f"{mu:>10.3f}{sd / mu:>10.3f}")


def lb_noise(spec_a: str, spec_b: str) -> None:
    """Парный и непарный шум LB, оценённые по OOF-остаткам двух смесей."""
    zа, y, cut = blend_from_spec(spec_a)
    zb, _, _ = blend_from_spec(spec_b)
    ly = np.log1p(y)
    for label, m in [("OOF целиком", np.ones(len(y), bool))] + \
                    [(f"фолд {c}", cut == c) for c in [VAL_FOLDS_S1[-1].isoformat()]]:
        ra, rb = ly[m] - zа[m], ly[m] - zb[m]
        ra, rb = ra - ra.mean(), rb - rb.mean()          # уровень сабмита задан якорем
        se_un = float(np.std(ra ** 2) / np.sqrt(N_PUBLIC)) / (2 * RMSLE_LB)
        se_pa = float(np.std(ra ** 2 - rb ** 2) / np.sqrt(N_PUBLIC)) / (2 * RMSLE_LB)
        print(f"  {label:<18} непарная SE {se_un:.5f}   ПАРНАЯ SE {se_pa:.5f}"
              f"   -> различимо на LB от {2 * se_pa:.5f}")


def main():
    pts = collect()
    print("=" * 104)
    print("СВОДКА: сабмиты, у которых есть и локальная метрика, и LB")
    print("=" * 104)
    print(f"{'эксперимент':<15}{'LB public':>12}{'LB level-free':>14}{'wCV':>10}"
          f"{'OOF cal':>10}{'CV raw':>9}   фолды (калибр.)")
    for k, m in sorted(pts.items(), key=lambda t: t[1]["lb_free"]):
        fc = " ".join(f"{v:.5f}" for v in (m.get("fold_cal") or []))
        def g(x):
            return f"{m[x]:.5f}" if m.get(x) is not None else "—".rjust(7)
        print(f"{k:<15}{m['lb']:>12.7f}{m['lb_free']:>14.7f}{g('wcv'):>10}"
              f"{g('oof_cal'):>10}{g('cv_mean'):>9}   {fc}")

    print("\n" + "=" * 104)
    print("КОЭФФИЦИЕНТ ПЕРЕНОСА: дельта LB / дельта локальной метрики (база S1-BEST)")
    print("чем ближе разброс к нулю, тем предсказуемее схема; сам средний ~1.0")
    print("=" * 104)
    transfer_table(pts)

    print("\n" + "=" * 104)
    print(f"ШУМ LEADERBOARD (public = {N_PUBLIC:,} строк)")
    print("=" * 104)
    reg = {r["exp_id"]: r["oof_source"] for r in read_registry() if r["oof_source"]}
    lb_noise(reg["S1-DIST-MIX"], reg["S1-BEST"])
    print("\n  Прежний порог «не верить разнице < 0.01 на public» посчитан по НЕПАРНОЙ SE")
    print("  и к сравнению двух сабмитов неприменим: public-выборка у них одна и та же.")


if __name__ == "__main__":
    main()
