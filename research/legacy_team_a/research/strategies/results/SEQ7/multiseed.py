"""EXP-030 `SEQ-D3A` — сводка ПО СИДАМ: главный ответ даёт среднее Δ, а не один прогон.

Постановка (`exp_030` -> `exp_030b`, «что делать дальше», п. 1): решение по приёму
принимается по среднему пофолдовому Δ по сидам и по РАЗБРОСУ Δ, а не по отдельным
удачным прогонам. `exp_030b` намерил на одном фолде, что смена сида двигает
результат сильнее самого приёма, поэтому одиночный сид неразрешим в принципе
(`seed std` TCN по wCV = 0.00250 против наблюдённой дельты 0.00077).

## Что здесь считается и чего здесь СОЗНАТЕЛЬНО нет

Единица анализа — **пофолдовая парная дельта** `Δ(seed, fold) = RMSLE_cal(D3A) −
RMSLE_cal(BASE)`, снятая на одной и той же машине, одном коде и одном сиде.
Это единственная величина, которую можно складывать между сидами: абсолютные
уровни прогонов на РАЗНЫХ машинах несравнимы (`exp_030`: локальный eager
систематически горячее A10 на 0.0010–0.0015), а здесь сид 42 посчитан локально
в eager, сиды 43/44 — на двух арендованных A10 с `--compile`.

Отсюда прямое следствие, которое надо назвать честно: **сравнение «разброс BASE
по сидам против разброса D3A по сидам» (главный вопрос `exp_030b`, п. 2) на
таком наборе НЕ считается** — межсидовый разброс здесь смешан с межмашинным.
Считается то, что от среды защищено: разброс САМОЙ ДЕЛЬТЫ по сидам.

`ΔwCV` берётся с весами 1:2:4:8 и определён только когда у сида посчитаны все
4 фолда (`validation.wcv`).

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ7/multiseed.py
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.tracking import load_oof
from src.validation import calibrate, rmsle_z

SEED_FLOOR = 0.00712        # проектный пол Var(dz) (exp_018)
SEND, DEV = -0.0020, -0.0005

# сид -> (BASE, вариант, где считалось). Пары ОБЯЗАНЫ быть внутрисредовыми.
RUNS = {
    42: ("SEQ-D3A-BASE-S42", "SEQ-D3A-S42", "локальная 4060 Ti, eager (exp_030)"),
    43: ("SEQ-D3A-G1-BASE-S43", "SEQ-D3A-G1-S43", "A10 #1, compile"),
    44: ("SEQ-D3A-G2-BASE-S44", "SEQ-D3A-G2-S44", "A10 #2, compile"),
}
# внесидовые реплики того же контраста: другая среда, тот же сид и фолд
REPLICAS = {(43, "2025-09-18"): ("SEQ-D3A-BASE-S43", "SEQ-D3A-S43",
                                 "локальная 4060 Ti, eager (exp_030b)")}


def auc(pos, score) -> float:
    from sklearn.metrics import roc_auc_score
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(pos):
        return float("nan")
    return float(roc_auc_score(pos.astype(np.int8), score))


def pair(base: str, exp: str, V: dt.date):
    """Одна пара BASE/вариант на одном фолде. None, если пары нет на диске."""
    tag = f"V{V.strftime('%m%d')}"
    if not all((ARTIFACTS / f"oof_{e}-{tag}.npz").exists() for e in (base, exp)):
        return None
    d = {e: load_oof(f"{e}-{tag}") for e in (base, exp)}
    uid = np.asarray(d[base]["user_id"])
    assert np.array_equal(np.asarray(d[exp]["user_id"]), uid), f"{tag}: другой набор строк"
    y = np.asarray(d[base]["y"], float)
    assert np.allclose(d[exp]["y"], y), f"{tag}: другой таргет"
    pos = y > 0
    out = {}
    for e in (base, exp):
        z = np.asarray(d[e]["z"], float)
        k, _ = calibrate(y, z)
        out[e] = dict(z=z, zc=np.maximum(z + k, 0.0), off=k)
    zb, zv = out[base], out[exp]
    return dict(
        fold=V.isoformat(), n=len(uid),
        base_cal=rmsle_z(y, zb["zc"]), exp_cal=rmsle_z(y, zv["zc"]),
        delta=rmsle_z(y, zv["zc"]) - rmsle_z(y, zb["zc"]),
        base_auc=auc(pos, zb["zc"]), d_auc=auc(pos, zv["zc"]) - auc(pos, zb["zc"]),
        var_delta=float(np.var(zv["z"] - zb["z"])),
        corr_resid=float(np.corrcoef(np.log1p(y) - zv["zc"], np.log1p(y) - zb["zc"])[0, 1]),
        off_base=zb["off"], off_exp=zv["off"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/strategies/results/SEQ7")
    a = ap.parse_args()
    folds = list(VAL_FOLDS_S1)
    wmap = dict(zip([d.isoformat() for d in folds], FOLD_WEIGHTS_S1))

    rows = []
    for seed, (base, exp, where) in RUNS.items():
        for V in folds:
            r = pair(base, exp, V)
            if r is None:
                continue
            rows.append(dict(seed=seed, where=where, base=base, exp=exp, **r))
    assert rows, "ни одной готовой пары BASE/D3A"
    df = pl.DataFrame(rows)

    print("=== EXP-030 SEQ-D3A: парные дельты по сидам ===\n")
    print(f"{'сид':>4} {'фолд':>12} {'вес':>4} {'BASE_cal':>9} {'D3A_cal':>9} {'Δ':>9} "
          f"{'ΔAUC':>9} {'Var(Δz)':>8} {'x пол':>6} {'corr_res':>9}")
    for r in df.sort(["seed", "fold"]).to_dicts():
        print(f"{r['seed']:>4} {r['fold']:>12} {wmap[r['fold']]:>4.0f} {r['base_cal']:>9.5f} "
              f"{r['exp_cal']:>9.5f} {r['delta']:>+9.5f} {r['d_auc']:>+9.5f} "
              f"{r['var_delta']:>8.5f} {r['var_delta'] / SEED_FLOOR:>6.2f} "
              f"{r['corr_resid']:>9.5f}")

    # --- ΔwCV по сидам (только при полных 4 фолдах) --------------------------
    print("\n--- ΔwCV (веса 1:2:4:8; определён только на полных 4 фолдах) ---")
    dwcv = {}
    for seed in sorted(df["seed"].unique().to_list()):
        s = df.filter(pl.col("seed") == seed)
        got = set(s["fold"].to_list())
        if got != {d.isoformat() for d in folds}:
            miss = sorted({d.isoformat() for d in folds} - got)
            print(f"  сид {seed}: {len(got)}/4 фолдов, не хватает {', '.join(miss)} — ΔwCV не считается")
            continue
        w = np.array([wmap[f] for f in s["fold"].to_list()])
        b = float((np.array(s["base_cal"]) * w).sum() / w.sum())
        v = float((np.array(s["exp_cal"]) * w).sum() / w.sum())
        dwcv[seed] = v - b
        wins = int((np.array(s["delta"]) < 0).sum())
        print(f"  сид {seed}: wCV {b:.5f} -> {v:.5f}   ΔwCV = {v - b:+.5f}   "
              f"лучше BASE на {wins}/4 фолдах   [{s['where'][0]}]")
    if len(dwcv) >= 2:
        d = np.array(list(dwcv.values()))
        print(f"\n  СРЕДНЕЕ ΔwCV по {len(d)} сидам = {d.mean():+.5f} "
              f"(sd {d.std(ddof=1):.5f}, se {d.std(ddof=1) / np.sqrt(len(d)):.5f})")
        print(f"  порог отправки {SEND:+.4f} | порог разработки {DEV:+.4f} | "
              f"пол разрешения 0.0005")
        verdict = ("ОТПРАВЛЯТЬ" if d.mean() <= SEND else
                   "В РАЗРАБОТКУ" if d.mean() <= DEV else "ШУМ/ХУЖЕ")
        print(f"  по среднему ΔwCV: {verdict}")

    # --- пофолдовое среднее по сидам ----------------------------------------
    print("\n--- Δ по фолдам, усреднённое по сидам (главная таблица решения) ---")
    print(f"{'фолд':>12} {'вес':>4} {'сидов':>6} {'AVG Δ':>9} {'sd Δ':>8} {'min':>9} {'max':>9} "
          f"{'улучш.':>7}")
    agg = []
    for V in folds:
        f = V.isoformat()
        s = df.filter(pl.col("fold") == f)
        if s.height == 0:
            continue
        d = np.array(s["delta"])
        agg.append(dict(fold=f, w=wmap[f], n_seeds=s.height, avg=float(d.mean()),
                        sd=float(d.std(ddof=1)) if s.height > 1 else float("nan"),
                        lo=float(d.min()), hi=float(d.max()),
                        wins=int((d < 0).sum())))
        print(f"{f:>12} {wmap[f]:>4.0f} {s.height:>6} {d.mean():>+9.5f} "
              f"{(d.std(ddof=1) if s.height > 1 else float('nan')):>8.5f} "
              f"{d.min():>+9.5f} {d.max():>+9.5f} {int((d < 0).sum()):>4}/{s.height}")
    if agg and all(r["n_seeds"] == agg[0]["n_seeds"] for r in agg) and len(agg) == 4:
        w = np.array([r["w"] for r in agg])
        av = np.array([r["avg"] for r in agg])
        print(f"\n  ΔwCV по УСРЕДНЁННЫМ пофолдовым дельтам = {float((av * w).sum() / w.sum()):+.5f}")
        print(f"  фолдов с AVG Δ < 0: {int((av < 0).sum())}/4 "
              f"(сессионный гейт требует >=3/4, включая 10-16)")

    # --- разброс самой дельты: то, что от среды защищено ---------------------
    print("\n--- разброс приёма (sd Δ по сидам) против пола сидов ---")
    for V in folds:
        s = df.filter(pl.col("fold") == V.isoformat())
        if s.height < 2:
            continue
        d = np.array(s["delta"])
        print(f"  {V}: sd(Δ) = {d.std(ddof=1):.5f} по {s.height} сидам; "
              f"средний Var(Δz) = {float(s['var_delta'].mean()):.5f} "
              f"({float(s['var_delta'].mean()) / SEED_FLOOR:.2f}x пола)")

    # --- прямой вопрос exp_030b, п. 1: чей разброс по сидам больше -----------
    # Считается С ОГОВОРКОЙ и печатается вместе с ней. Уровень прогона зависит
    # от машины, поэтому в sd по сидам сидит и межмашинная компонента. Она ОБЩАЯ
    # у BASE и D3A внутри сида (пара всегда на одной машине), то есть раздувает
    # обе величины примерно одинаково и сдвигает их отношение К ЕДИНИЦЕ.
    # Значит превышение у D3A — сигнал (оценка консервативная), а паритет —
    # НЕ доказательство отсутствия эффекта.
    print("\n--- sd по сидам отдельно у BASE и у D3A (см. оговорку в docstring) ---")
    for V in folds:
        s = df.filter(pl.col("fold") == V.isoformat())
        if s.height < 2:
            continue
        sb = float(np.std(np.array(s["base_cal"]), ddof=1))
        sv = float(np.std(np.array(s["exp_cal"]), ddof=1))
        print(f"  {V}: sd(BASE) = {sb:.5f}, sd(D3A) = {sv:.5f}, "
              f"отношение {sv / sb if sb else float('nan'):.2f}x "
              f"(exp_030b на 09-18 намерил 5.1x на паре сидов)")

    # --- реплики того же контраста в другой среде ---------------------------
    rep = []
    for (seed, f), (base, exp, where) in REPLICAS.items():
        r = pair(base, exp, dt.date.fromisoformat(f))
        if r is None:
            continue
        rep.append(dict(seed=seed, where=where, **r))
    if rep:
        print("\n--- независимые реплики контраста (другая среда, тот же сид/фолд) ---")
        for r in rep:
            m = df.filter((pl.col("seed") == r["seed"]) & (pl.col("fold") == r["fold"]))
            same = f"{float(m['delta'][0]):+.5f}" if m.height else "нет"
            print(f"  сид {r['seed']} {r['fold']}: Δ = {r['delta']:+.5f} [{r['where']}] "
                  f"против Δ = {same} в основном наборе")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.write_csv(out / "seed_folds.csv")
    if agg:
        pl.DataFrame(agg).write_csv(out / "fold_avg.csv")
    print(f"\nзаписано: {out}/seed_folds.csv, {out}/fold_avg.csv")


if __name__ == "__main__":
    main()
