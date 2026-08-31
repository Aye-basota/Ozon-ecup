"""EXP-032 (S04-SEQ) — сводка пилота по фолдам: `ΔwCV` на группе A.

Пилот (`exp_032_s04_cond_fresh_pilot.md`) закрыл гейт на одном фолде 10-16.
Здесь те же три варианта считаются на всех четырёх фолдах, каждый — на СВОЁМ
замороженном энкодере `SEQ-D3A-BASE-S42-V*`, и складываются в `wCV` с весами
1:2:4:8.

Две оговорки, без которых числа читаются неправильно:

1. **Это `wCV` на группе A**, то есть на половине панели. Расщепление по
   пользователям обязательно (cutoff'ы EXTRA лежат в будущем относительно
   фолда), и метрика на половине людей — цена этой обязательности. Сравнивать
   с проектными 1.749…1.753 нельзя; сравнимы только контрасты внутри таблицы.
2. **Единица решения — контраст `FRESH − CLEAN`.** `COND-VOL` отделяет объём от
   свежести, `BASE-1HEAD` показывает цену самой двухчастной пересборки.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ7/s04_folds.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1

VARIANTS = ["BASE-1HEAD", "COND-CLEAN", "COND-VOL", "COND-FRESH"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="S04SEQ_PILOT-S42")
    ap.add_argument("--out", default="research/strategies/results/SEQ7")
    a = ap.parse_args()

    got, cal = [], {v: [] for v in VARIANTS}
    for V, w in zip(VAL_FOLDS_S1, FOLD_WEIGHTS_S1):
        p = ARTIFACTS / f"{a.prefix}-V{V.strftime('%m%d')}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = {r["variant"]: r for r in d["rows"]}
        if not all(v in rows for v in VARIANTS):
            print(f"{V}: нет всех вариантов ({sorted(rows)}) — фолд пропущен")
            continue
        got.append(dict(fold=V.isoformat(), w=w, n=rows["COND-CLEAN"]["n"],
                        poison=d.get("poison_resid_mean"),
                        n_extra=d.get("n_extra_pos"), sweep=d.get("sweep")))
        for v in VARIANTS:
            cal[v].append(rows[v]["rmsle_cal"])
    assert got, f"не найдено ни одного {a.prefix}-V*.json"

    print(f"=== EXP-032 S04-SEQ: {len(got)}/4 фолдов, метрика на группе A ===\n")
    hdr = f"{'фолд':>12} {'вес':>4} {'n':>8} " + " ".join(f"{v:>11}" for v in VARIANTS)
    print(hdr)
    for i, g in enumerate(got):
        print(f"{g['fold']:>12} {g['w']:>4.0f} {g['n']:>8,} "
              + " ".join(f"{cal[v][i]:>11.5f}" for v in VARIANTS))

    print(f"\n{'Δ к COND-CLEAN':>12}")
    print(f"{'фолд':>12} {'вес':>4} {'FRESH':>11} {'VOL':>11} {'1HEAD':>11}")
    for i, g in enumerate(got):
        print(f"{g['fold']:>12} {g['w']:>4.0f} "
              f"{cal['COND-FRESH'][i] - cal['COND-CLEAN'][i]:>+11.5f} "
              f"{cal['COND-VOL'][i] - cal['COND-CLEAN'][i]:>+11.5f} "
              f"{cal['BASE-1HEAD'][i] - cal['COND-CLEAN'][i]:>+11.5f}")

    w = np.array([g["w"] for g in got], float)
    full = len(got) == len(VAL_FOLDS_S1)
    tag = "wCV(группа A)" if full else "частичная взвешенная сумма (НЕ wCV)"
    print()
    for v in VARIANTS:
        print(f"  {tag}, {v:<11} = {float((np.array(cal[v]) * w).sum() / w.sum()):.5f}")
    dv = (np.array(cal["COND-FRESH"]) - np.array(cal["COND-CLEAN"]))
    dvol = (np.array(cal["COND-VOL"]) - np.array(cal["COND-CLEAN"]))
    print(f"\n  Δ{tag} FRESH − CLEAN = {float((dv * w).sum() / w.sum()):+.5f}   "
          f"лучше на {int((dv < 0).sum())}/{len(dv)} фолдах")
    print(f"  Δ{tag} VOL   − CLEAN = {float((dvol * w).sum() / w.sum()):+.5f}   "
          "(контроль объёма)")
    if full:
        print("  порог отправки -0.0020 | порог разработки -0.0005 | "
              "гейт STRATEGY_04: лучше A на >=3/4 фолдах И лучше C")

    print("\n  диагностика отравления интенсива (порог закрытия 0.03):")
    for g in got:
        print(f"    {g['fold']}: остаток {g['poison']:+.4f}, "
              f"EXTRA-строк {g['n_extra']:,}")

    sw = [g for g in got if g.get("sweep")]
    if sw:
        print("\n  развёртка по сидам головы:")
        for g in sw:
            d = np.array([r["d_cal"] for r in g["sweep"]])
            print(f"    {g['fold']}: FRESH−CLEAN {d.mean():+.5f} "
                  f"(sd {d.std(ddof=1):.5f}, {int((d < 0).sum())}/{len(d)} сидов)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import polars as pl
    pl.DataFrame([dict(fold=g["fold"], w=g["w"], n=g["n"],
                       **{v: cal[v][i] for v in VARIANTS},
                       d_fresh=cal["COND-FRESH"][i] - cal["COND-CLEAN"][i],
                       d_vol=cal["COND-VOL"][i] - cal["COND-CLEAN"][i],
                       poison=g["poison"])
                  for i, g in enumerate(got)]).write_csv(out / "s04_folds.csv")
    print(f"\nзаписано: {out}/s04_folds.csv")


if __name__ == "__main__":
    main()
