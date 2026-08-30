"""Этапы 2-3 — что происходит при экстраполяции глубины на +60..+80 дней и что с этим делать.

`exp_026` снял кривую глубины на СВОЁМ фолде у каждой модели. Там за границей
обученной глубины остаётся всего 35 дней, кривая на этом отрезке монотонна, и из
этого был сделан вывод «минимум всегда на максимуме доступного, обрезать не надо».
На тесте отрезок другой: обучено до 289, подаётся 365, то есть **+76**.

Здесь тот же режим воспроизводится без обучения: модель РАННЕГО фолда на ПОЗДНЕЙ
панели (`src/seq.py crossdepth`). Пара «09-04 -> 10-16» даёт +77 — почти точную
копию теста. Внутри пары фиксировано всё: веса, панель, таргет, пользователи;
меняется одна глубина входа.

Этап 3 — усадка вместо бинарного выбора:

    z(α) = (1 − α)·z_clip + α·z_full,   z_clip = обученная глубина, z_full = вся доступная

α подбирается ТОЛЬКО на исторических парах и честно (leave-one-pair-out), после
чего та же политика переносится на `clip289 / full365`. Отдельно проверяется
гейтинг: продлевать глубину лишь тем, у кого в добавляемом куске реально есть
активность.

Запуск: PYTHONIOENCODING=utf-8 PYTHONPATH=. python research/strategies/results/SEQ3/stress.py
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS
from src.features import features_cached
from src.validation import bias_z, calibrate, rmsle_z

OUT = Path(__file__).parent
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


def auc(pos, score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos.astype(np.int8), score))


def load_pairs():
    """Все кросс-фолдовые прогоны: (имя, сид, панель, глубины, Z, y, uid)."""
    out = []
    for f in sorted(ARTIFACTS.glob("xdepth_*.npz")):
        d = np.load(f)
        tag = f.stem.replace("xdepth_", "")
        ck, panel = tag.split("_on")
        seed = ck.split("-S")[1].split("-V")[0]
        out.append(dict(tag=tag, ckpt=ck, seed=seed, panel=panel,
                        depths=[int(x) for x in d["depths"]], train_max=int(d["train_max"]),
                        avail=int(d["avail"]), Z=d["z"].astype(np.float64),
                        y=d["y"].astype(np.float64), uid=d["user_id"]))
    return out


def seg_masks(panel_tag: str, uid: np.ndarray, train_max: int, avail: int):
    """Сегменты диагностики + «есть ли активность в добавляемом старом куске»."""
    from src.seq import CHANNELS, day_index, panel, user_rows
    V = dt.date(2025, int(panel_tag[:2]), int(panel_tag[2:]))
    f = features_cached(V, None, True)
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    rec, d180 = f["rec_buy"].to_numpy(), f["w180_days_buy"].to_numpy()
    p, _, _, _ = panel()
    dT = day_index(V)
    lo, hi = dT - avail + 1, dT - train_max + 1        # добавляемый кусок
    rows = user_rows(uid)
    add = p[rows, lo:hi, :][:, :, [CHANNELS.index("present"), CHANNELS.index("buy")]]
    n_days = add[:, :, 0].astype(np.float32).sum(1)
    n_buys = add[:, :, 1].astype(np.float32).sum(1)
    m = {"ВСЕ": np.ones(len(uid), bool),
         "rec_buy 15-60": (rec >= 15) & (rec <= 60),
         "полоса 2-15": (d180 >= 2) & (d180 <= 15),
         "добавка ПУСТА": n_days == 0,
         "в добавке есть дни": n_days > 0,
         "в добавке есть покупки": n_buys > 0}
    return m, n_days, n_buys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-seg", action="store_true")
    a = ap.parse_args()
    pairs = load_pairs()
    assert pairs, "нет ни одного artifacts/xdepth_*.npz — сначала src.seq crossdepth"

    # ---------------------------------------------------------------- этап 2: кривая
    rows = []
    print("ЭТАП 2. Кривая глубины при экстраполяции далеко за обученное\n")
    for p in pairs:
        y, Z, dep = p["y"], p["Z"], p["depths"]
        i_clip = dep.index(p["train_max"])
        base = rmsle_z(y, np.maximum(Z[i_clip] + calibrate(y, Z[i_clip])[0], 0))
        print(f"{p['ckpt']} -> панель {p['panel']}: обучено до {p['train_max']}, "
              f"доступно {p['avail']}, ЭКСТРАПОЛЯЦИЯ +{p['avail'] - p['train_max']}")
        print(f"  {'глубина':>8}{'+к обуч.':>10}{'RMSLE':>10}{'калибр.':>10}"
              f"{'Δ к clip':>10}{'AUC(y>0)':>10}{'bias':>9}{'Var(Δ)':>9}")
        for i, D in enumerate(dep):
            zc = np.maximum(Z[i] + calibrate(y, Z[i])[0], 0)
            r = dict(ckpt=p["ckpt"], seed=p["seed"], panel=p["panel"], depth=D,
                     train_max=p["train_max"], avail=p["avail"],
                     extrap=p["avail"] - p["train_max"], over_train=D - p["train_max"],
                     rmsle=rmsle_z(y, Z[i]), rmsle_cal=rmsle_z(y, zc),
                     d_cal=rmsle_z(y, zc) - base, auc=auc(y > 0, Z[i]),
                     bias=bias_z(y, Z[i]), mean_z=float(Z[i].mean()),
                     var_vs_clip=float(np.var(Z[i] - Z[i_clip])))
            rows.append(r)
            print(f"  {D:>8}{r['over_train']:>+10d}{r['rmsle']:>10.5f}{r['rmsle_cal']:>10.5f}"
                  f"{r['d_cal']:>+10.5f}{r['auc']:>10.5f}{r['bias']:>+9.4f}"
                  f"{r['var_vs_clip']:>9.5f}")
        best = min([r for r in rows if r["ckpt"] == p["ckpt"] and r["panel"] == p["panel"]],
                   key=lambda r: r["rmsle_cal"])
        full = [r for r in rows if r["ckpt"] == p["ckpt"] and r["panel"] == p["panel"]
                and r["depth"] == p["avail"]][0]
        print(f"  оптимум на {best['depth']} ({best['over_train']:+d} к обученной); "
              f"полная глубина хуже оптимума на {full['d_cal'] - best['d_cal']:+.5f}, "
              f"лучше обрезки на {full['d_cal']:+.5f}\n")
    pl.DataFrame(rows).write_csv(OUT / "stress_curve.csv")

    # сводка: выигрыш полной глубины как функция размера экстраполяции
    print("Сводка: что даёт ПОЛНАЯ глубина против обрезки на обученной, по размеру шага")
    print(f"  {'экстрап.':>9}{'пар':>5}{'Δ полн. к clip':>16}{'Δ оптимума':>13}"
          f"{'оптимум +к обуч.':>18}")
    summ = []
    for ex in sorted({r["extrap"] for r in rows}):
        sub = [r for r in rows if r["extrap"] == ex]
        keys = sorted({(r["ckpt"], r["panel"]) for r in sub})
        dfull = [[r for r in sub if (r["ckpt"], r["panel"]) == k and r["depth"] == r["avail"]][0]
                 ["d_cal"] for k in keys]
        bests = [min([r for r in sub if (r["ckpt"], r["panel"]) == k],
                     key=lambda r: r["rmsle_cal"]) for k in keys]
        summ.append(dict(extrap=ex, n=len(keys), d_full=float(np.mean(dfull)),
                         d_best=float(np.mean([b["d_cal"] for b in bests])),
                         best_over=float(np.mean([b["over_train"] for b in bests]))))
        print(f"  {ex:>+9d}{len(keys):>5}{np.mean(dfull):>+16.5f}"
              f"{np.mean([b['d_cal'] for b in bests]):>+13.5f}"
              f"{np.mean([b['over_train'] for b in bests]):>+18.1f}")
    pl.DataFrame(summ).write_csv(OUT / "stress_summary.csv")

    # ---------------------------------------------------------------- этап 3: усадка
    print("\nЭТАП 3. Усадка z(α) = (1−α)·z_clip + α·z_full вместо бинарного выбора\n")
    arows = []
    print(f"  {'пара':>34}{'экстр.':>7}" + "".join(f"{'α=' + str(x):>10}" for x in ALPHAS))
    for p in pairs:
        y, Z, dep = p["y"], p["Z"], p["depths"]
        zc, zf = Z[dep.index(p["train_max"])], Z[dep.index(p["avail"])]
        vals = []
        for al in ALPHAS:
            zz = (1 - al) * zc + al * zf
            vals.append(rmsle_z(y, np.maximum(zz + calibrate(y, zz)[0], 0)))
            arows.append(dict(ckpt=p["ckpt"], panel=p["panel"], seed=p["seed"],
                              extrap=p["avail"] - p["train_max"], alpha=al,
                              rmsle_cal=vals[-1]))
        b = int(np.argmin(vals))
        print(f"  {p['ckpt'] + '->' + p['panel']:>34}{p['avail'] - p['train_max']:>+7d}"
              + "".join(f"{v:>10.5f}" for v in vals) + f"   argmin α={ALPHAS[b]}")
    pl.DataFrame(arows).write_csv(OUT / "alpha_curve.csv")

    # честный выбор α: leave-one-pair-out
    keys = sorted({(r["ckpt"], r["panel"]) for r in arows})
    print(f"\n  честный выбор α (leave-one-pair-out, {len(keys)} пар):")
    print(f"  {'отложенная пара':>34}{'α без неё':>11}{'её RMSLE при этом α':>22}"
          f"{'при α=1':>10}{'при α=0':>10}")
    held = []
    for k in keys:
        rest = [r for r in arows if (r["ckpt"], r["panel"]) != k]
        # усреднять сырые RMSLE разных панелей нельзя — сравниваем Δ к α=0 внутри пары
        d = {al: [] for al in ALPHAS}
        for kk in keys:
            if kk == k:
                continue
            sub = {r["alpha"]: r["rmsle_cal"] for r in rest
                   if (r["ckpt"], r["panel"]) == kk}
            for al in ALPHAS:
                d[al].append(sub[al] - sub[0.0])
        a_star = ALPHAS[int(np.argmin([np.mean(d[al]) for al in ALPHAS]))]
        own = {r["alpha"]: r["rmsle_cal"] for r in arows if (r["ckpt"], r["panel"]) == k}
        held.append(dict(pair=f"{k[0]}->{k[1]}", alpha=a_star, score=own[a_star],
                         d_vs_a0=own[a_star] - own[0.0], d_a1=own[1.0] - own[0.0]))
        print(f"  {k[0] + '->' + k[1]:>34}{a_star:>11.2f}{own[a_star]:>22.5f}"
              f"{own[1.0]:>10.5f}{own[0.0]:>10.5f}")
    hd = float(np.mean([h["d_vs_a0"] for h in held]))
    a1 = float(np.mean([h["d_a1"] for h in held]))
    print(f"  честный α: средний Δ к обрезке {hd:+.5f}; у бинарного α=1 (полная) {a1:+.5f}; "
          f"выигрыш политики {hd - a1:+.5f}")
    pl.DataFrame(held).write_csv(OUT / "alpha_lopo.csv")

    # ---------------------------------------------------------------- гейтинг
    if a.no_seg:
        return
    print("\nГЕЙТИНГ: продлевать глубину только тем, у кого в добавке реально есть активность\n")
    grows = []
    for p in pairs:
        y, Z, dep = p["y"], p["Z"], p["depths"]
        zc, zf = Z[dep.index(p["train_max"])], Z[dep.index(p["avail"])]
        m, n_days, n_buys = seg_masks(p["panel"], p["uid"], p["train_max"], p["avail"])
        def cal(zz):
            return rmsle_z(y, np.maximum(zz + calibrate(y, zz)[0], 0))
        v0, v1 = cal(zc), cal(zf)
        gd = np.where(n_days > 0, zf, zc)
        gb = np.where(n_buys > 0, zf, zc)
        grows.append(dict(ckpt=p["ckpt"], panel=p["panel"],
                          extrap=p["avail"] - p["train_max"], clip=v0, full=v1,
                          gate_days=cal(gd), gate_buys=cal(gb),
                          share_days=float((n_days > 0).mean()),
                          share_buys=float((n_buys > 0).mean())))
        print(f"  {p['ckpt']}->{p['panel']}: clip {v0:.5f}  full {v1 - v0:+.5f}  "
              f"гейт по дням {cal(gd) - v0:+.5f} ({(n_days > 0).mean():.0%})  "
              f"гейт по покупкам {cal(gb) - v0:+.5f} ({(n_buys > 0).mean():.0%})")
        # посегментно: где именно полная глубина вредит
        print(f"    {'сегмент':<24}{'доля':>7}{'Δ full к clip':>15}{'Var(Δz)':>10}")
        for sname, mm in m.items():
            zc_m = np.maximum(zc[mm] + calibrate(y[mm], zc[mm])[0], 0)
            zf_m = np.maximum(zf[mm] + calibrate(y[mm], zf[mm])[0], 0)
            dd = rmsle_z(y[mm], zf_m) - rmsle_z(y[mm], zc_m)
            print(f"    {sname:<24}{mm.mean():>7.1%}{dd:>+15.5f}"
                  f"{np.var(zf[mm] - zc[mm]):>10.5f}")
            grows[-1][f"seg_{sname}"] = float(dd)
    pl.DataFrame(grows).write_csv(OUT / "gating.csv")
    print(f"\nзаписано: {OUT}/stress_curve.csv, stress_summary.csv, alpha_curve.csv, "
          f"alpha_lopo.csv, gating.csv")


if __name__ == "__main__":
    main()
