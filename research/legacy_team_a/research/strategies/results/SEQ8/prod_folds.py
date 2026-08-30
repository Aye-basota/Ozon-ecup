"""EXP-032B — сводка по фолдам: боевой экстенсив вместо переобученной головы `P(y>0)`.

`exp_032` оставил открытым ровно одно: приём (`FRESH − CLEAN` = −0.00128, 4/4)
подтверждён, но абсолютное число в смесь идти не может, потому что двухчастная
пересборка сама стоит **+0.00088 wCV(A)** — и вся эта цена сидит в экстенсиве,
переобученном поверх эмбеддингов. Здесь `μ̂` тот же, а `p̂` берётся из уже
существующей CLEAN-модели проекта (основной источник — `1 − p0` боевой головы
`S1-DIST`, `exp_014`).

Три числа, ради которых таблица считается:

1. `P_prod × μ_FRESH − P_prod × μ_CLEAN` — переживает ли эффект свежести смену
   экстенсива (гейт: ≥3/4 фолдов, включая 10-16; желательно ≤ −0.001);
2. `P_prod × μ_FRESH − BASE-1HEAD` — стало ли итоговое число лучше одноголовой
   базы (гейт: ≥3/4 фолдов ЛИБО заметное сокращение прежнего +0.00088);
3. `P_prod × μ_CLEAN − BASE-1HEAD` — сколько от прежнего налога осталось.

Оговорка, как и в `exp_032`: **`wCV` здесь на группе A**, то есть на половине
панели. С проектными 1.749…1.753 не сравнивать; сравнимы только контрасты
внутри этих таблиц.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ8/prod_folds.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1

MU = ("CLEAN", "VOL", "FRESH")
BASE = "BASE-1HEAD"


def wavg(x, w):
    return float((np.asarray(x, float) * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="S04PROD_S42")
    ap.add_argument("--out", default="research/strategies/results/SEQ8")
    a = ap.parse_args()

    folds, cal, per_seed_d, meta = [], {}, [], []
    for V, w in zip(VAL_FOLDS_S1, FOLD_WEIGHTS_S1):
        p = ARTIFACTS / f"{a.prefix}-V{V.strftime('%m%d')}.json"
        if not p.exists():
            print(f"{V}: нет {p.name} — фолд пропущен")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = {r["variant"]: r for r in d["per_seed"][0]["rows"]}
        folds.append(dict(fold=V.isoformat(), w=w, n=d["n_groupA"], primary=d["primary"]))
        meta.append(d)
        for v, r in rows.items():
            cal.setdefault(v, {})[V.isoformat()] = r["rmsle_cal"]
        # разброс по сидам головы считается по КАЖДОМУ варианту отдельно
        acc = {}
        for s in d["per_seed"]:
            for r in s["rows"]:
                acc.setdefault(r["variant"], []).append(r["rmsle_cal"])
        per_seed_d.append(acc)
    assert folds, f"не найдено ни одного {a.prefix}-V*.json"

    keys = [f["fold"] for f in folds]
    w = np.array([f["w"] for f in folds], float)
    full = len(folds) == len(VAL_FOLDS_S1)
    tag = "wCV(A)" if full else "частичная взвешенная сумма (НЕ wCV)"
    pp = folds[0]["primary"]
    src = sorted({v.split("x")[0] for v in cal if "x" in v})
    src = [pp] + [s for s in src if s != pp]

    def col(v):
        return [cal[v][k] for k in keys] if all(k in cal.get(v, {}) for k in keys) else None

    print(f"=== EXP-032B: {len(folds)}/4 фолдов, метрика на группе A, "
          f"основной экстенсив {pp} ===\n")
    variants = [BASE, "DIST-TAB"] + [f"{s}x{m}" for s in src for m in MU]
    variants = [v for v in variants if col(v) is not None]
    print(f"{'фолд':>12} {'вес':>4} {'n':>8} " + " ".join(f"{v:>12}" for v in variants))
    for i, f in enumerate(folds):
        print(f"{f['fold']:>12} {f['w']:>4.0f} {f['n']:>8,} "
              + " ".join(f"{cal[v][keys[i]]:>12.5f}" for v in variants))
    print(f"{tag:>12} {'':>4} {'':>8} "
          + " ".join(f"{wavg(col(v), w):>12.5f}" for v in variants))

    contrasts = [(f"{pp}xFRESH − {pp}xCLEAN", f"{pp}xFRESH", f"{pp}xCLEAN",
                  "главный контраст: свежая supervision при боевом экстенсиве"),
                 (f"{pp}xVOL   − {pp}xCLEAN", f"{pp}xVOL", f"{pp}xCLEAN",
                  "контроль объёма"),
                 (f"{pp}xFRESH − {BASE}", f"{pp}xFRESH", BASE,
                  "итог против одноголовой базы"),
                 (f"{pp}xCLEAN − {BASE}", f"{pp}xCLEAN", BASE,
                  "остаток налога двухчастной схемы"),
                 ("SEQxCLEAN − " + BASE, "SEQxCLEAN", BASE,
                  "прежний налог exp_032 (+0.00088)"),
                 ("SEQxFRESH − SEQxCLEAN", "SEQxFRESH", "SEQxCLEAN",
                  "прежний приём exp_032 (−0.00128)"),
                 (f"{pp}xFRESH − DIST-TAB", f"{pp}xFRESH", "DIST-TAB",
                  "добавляет ли интенсив SEQ что-то поверх модели-донора")]
    print("\n=== контрасты (RMSLE_cal, группа A) ===")
    print(f"{'контраст':>26} " + " ".join(f"{k[5:]:>10}" for k in keys)
          + f" {tag:>12} {'лучше':>7}")
    out_rows = []
    for name, hi, lo, why in contrasts:
        if col(hi) is None or col(lo) is None:
            continue
        d = np.array(col(hi)) - np.array(col(lo))
        print(f"{name:>26} " + " ".join(f"{x:>+10.5f}" for x in d)
              + f" {wavg(d, w):>+12.5f} {int((d < 0).sum())}/{len(d):<5}   {why}")
        out_rows.append(dict(contrast=name, **{k: float(x) for k, x in zip(keys, d)},
                             weighted=wavg(d, w), better=int((d < 0).sum())))

    # интенсив и AUC итогового `ẑ` — обе величины требует постановка EXP-032B
    mu_cal, auc_z = {}, {}
    for m, k in zip(meta, keys):
        for r in m["per_seed"][0]["rows"]:
            if "rmsle_mu" in r:
                mu_cal.setdefault(r["variant"], {})[k] = r["rmsle_mu"]
            auc_z.setdefault(r["variant"], {})[k] = r["auc"]
    print("\n=== интенсив на y>0 (RMSLE от μ̂, не зависит от выбора P) ===")
    for v in ("CLEAN", "VOL", "FRESH"):
        kk = f"{pp}x{v}"
        if all(k in mu_cal.get(kk, {}) for k in keys):
            x = [mu_cal[kk][k] for k in keys]
            print(f"  μ_{v:<6} " + " ".join(f"{y:.5f}" for y in x)
                  + f"   {tag} {wavg(x, w):.5f}")
    a_f = [mu_cal[f"{pp}xFRESH"][k] - mu_cal[f"{pp}xCLEAN"][k] for k in keys]
    print(f"  Δ FRESH−CLEAN " + " ".join(f"{x:+.5f}" for x in a_f)
          + f"   {tag} {wavg(a_f, w):+.5f}   лучше {sum(1 for x in a_f if x < 0)}/{len(a_f)}")

    print("\n=== AUC(y>0) итогового ẑ ===")
    for v in [BASE, "DIST-TAB"] + [f"{pp}x{m}" for m in MU]:
        if all(k in auc_z.get(v, {}) for k in keys):
            x = [auc_z[v][k] for k in keys]
            print(f"  {v:<14} " + " ".join(f"{y:.5f}" for y in x)
                  + f"   {tag} {wavg(x, w):.5f}")

    print("\n=== разброс по сидам головы (энкодер и эмбеддинги общие) ===")
    for i, f in enumerate(folds):
        acc = per_seed_d[i]
        line = []
        for hi, lo in ((f"{pp}xFRESH", f"{pp}xCLEAN"), (f"{pp}xFRESH", BASE)):
            if hi in acc and lo in acc and len(acc[hi]) == len(acc[lo]):
                dd = np.array(acc[hi]) - np.array(acc[lo])
                line.append(f"{hi.split('x')[-1]}−{lo.split('x')[-1]}: {dd.mean():+.5f} "
                            f"(sd {dd.std(ddof=1) if len(dd) > 1 else 0:.5f}, "
                            f"{int((dd < 0).sum())}/{len(dd)})")
        print(f"  {f['fold']}: " + " | ".join(line))

    print("\n=== AUC(y>0) источников экстенсива, группа A ===")
    names = sorted(meta[0]["auc_p"])
    print(f"{'источник':>10} " + " ".join(f"{k[5:]:>9}" for k in keys) + f" {tag:>10}")
    for n in names:
        v = [m["auc_p"][n] for m in meta]
        print(f"{n:>10} " + " ".join(f"{x:>9.5f}" for x in v) + f" {wavg(v, w):>10.5f}")

    # Почему налог двухчастной схемы вообще был: `ẑ = p̂·μ̂` — произведение, а
    # калибровка метрики — АДДИТИВНЫЙ сдвиг в логах. Ошибка УРОВНЯ у `p̂`
    # аддитивным сдвигом не снимается, поэтому смещённый экстенсив стоит денег
    # ровно там, где он смещён. Таблица показывает, что переобученная голова SEQ
    # смещена на ранних фолдах, а боевые табличные — нет.
    print("\n=== калибровка источников экстенсива (группа A) ===")
    cal_rows, seen = [], []
    for V, kk in zip(VAL_FOLDS_S1, keys):
        f = ARTIFACTS / f"{a.prefix}-V{V.strftime('%m%d')}_z.npz"
        if not f.exists():
            continue
        d = np.load(f)
        A = d["group"] == 0
        lab = (d["y"][A] > 0).astype(float)
        row = {"fold": kk, "rate": float(lab.mean())}
        for k in d.files:
            if not k.startswith("P_"):
                continue
            p = np.clip(d[k][A].astype(float), 1e-6, 1 - 1e-6)
            row[f"{k[2:]}_dlevel"] = float(p.mean() - lab.mean())
            row[f"{k[2:]}_brier"] = float(np.mean((p - lab) ** 2))
        # переобученная голова SEQ хранится в пилоте, но её `ẑ` уже есть здесь
        if "z_SEQ_X_CLEAN" in d.files and "mu_CLEAN" in d.files:
            p = np.clip(d["z_SEQ_X_CLEAN"][A] / np.maximum(d["mu_CLEAN"][A], 1e-9),
                        1e-6, 1 - 1e-6)
            row["SEQ_dlevel"] = float(p.mean() - lab.mean())
            row["SEQ_brier"] = float(np.mean((p - lab) ** 2))
        cal_rows.append(row)
        seen.append(kk)
    if cal_rows:
        srcs = sorted({k[:-7] for k in cal_rows[0] if k.endswith("_dlevel")})
        print(f"{'источник':>10} " + " ".join(f"{k[5:]:>9}" for k in seen)
              + f"   {'Brier ' + tag:>18}")
        for s in srcs:
            v = [r.get(f"{s}_dlevel") for r in cal_rows]
            b = [r.get(f"{s}_brier") for r in cal_rows]
            if any(x is None for x in v):
                continue
            print(f"{s:>10} " + " ".join(f"{x:>+9.4f}" for x in v)
                  + f"   {wavg(b, w[:len(b)]):>18.5f}")
        print("           (строки — сдвиг УРОВНЯ mean p − факт по фолдам)")

    print("\n=== устойчивость к выбору P: Δ(FRESH − CLEAN) по источникам ===")
    for s in src:
        if col(f"{s}xFRESH") is None:
            continue
        d = np.array(col(f"{s}xFRESH")) - np.array(col(f"{s}xCLEAN"))
        db = (np.array(col(f"{s}xFRESH")) - np.array(col(BASE))) if col(BASE) else None
        print(f"  {s:<8} FRESH−CLEAN {wavg(d, w):+.5f} ({int((d < 0).sum())}/{len(d)})"
              + (f"   FRESH−BASE {wavg(db, w):+.5f} ({int((db < 0).sum())}/{len(db)})"
                 if db is not None else ""))

    print("\n=== сегменты (RMSLE_cal, взвешено по фолдам) ===")
    segs = {}
    for m, f in zip(meta, folds):
        for r in m["segments"]:
            segs.setdefault(r["segment"], []).append((f["w"], r))
    print(f"{'сегмент':<24} {'доля':>6} {'Δ FRESH−CLEAN':>14} {'Δ FRESH−BASE':>14}")
    for nm, lst in segs.items():
        ww = np.array([x[0] for x in lst], float)
        sh = wavg([x[1]["share"] for x in lst], ww)
        d1 = wavg([x[1]["delta"] for x in lst], ww)
        d2 = ([x[1]["delta_base"] for x in lst] if all(x[1].get("delta_base") is not None
                                                       for x in lst) else None)
        print(f"{nm:<24} {sh:>6.3f} {d1:>+14.5f} "
              + (f"{wavg(d2, ww):>+14.5f}" if d2 else f"{'н/д':>14}"))

    print("\n=== диагностика точечности изменения ===")
    for m, f in zip(meta, folds):
        print(f"  {f['fold']}: Var(Δz) = {m['var_delta']:.5f} "
              f"({m['var_delta'] / 0.00712:.2f}x пола сидов), "
              f"corr остатков {m['corr_resid']:.5f}, "
              f"сверка с пилотом max|Δz| "
              + ", ".join(f"{k} {v:.1e}" for k, v in m.get("pilot_recheck", {}).items()))

    if full:
        print("\nпороги exp_016: отправка −0.0020 | разработка −0.0005 | "
              "гейт ≥3/4 фолдов, включая 10-16")

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    import polars as pl
    pl.DataFrame([dict(fold=k, w=float(w[i]), **{v: cal[v][k] for v in variants})
                  for i, k in enumerate(keys)]).write_csv(o / "prod_folds.csv")
    pl.DataFrame(out_rows).write_csv(o / "prod_contrasts.csv")
    print(f"\nзаписано: {o}/prod_folds.csv, {o}/prod_contrasts.csv")


if __name__ == "__main__":
    main()
