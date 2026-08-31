"""EXP-032B — цена гибридизации: остаётся ли SEQ-член ортогональным табличной смеси.

Замена переобученной головы `P(y>0)` на боевой экстенсив улучшает ОДИНОЧНЫЙ скор,
но она же вставляет в SEQ-член предсказание табличной модели. А ценность SEQ в
смеси — не его собственный RMSLE, а ортогональность ошибки: `exp_025` брал его
в смесь именно за это. Поэтому выигрыш в одиночном скоре сам по себе ничего не
решает, и этот скрипт меряет вторую половину.

Три величины на группе A (`exp_024` §Сравнение — тот же набор, что применялся к MHZ):

* `corr` остатков кандидата с остатком ТАБЛИЧНОЙ части смеси — прямая мера того,
  насколько член перестал быть «другой функцией»;
* `Var(z − z_tab)` — разнообразие; проектный порог полезного разнообразия 0.10,
  пол сидов `Var(Δ)` = 0.00712 (`exp_018`);
* подстановка в боевую смесь ПРИ ФИКСИРОВАННЫХ весах `0.15/0.20/0.10/0.25/0.30`.
  Это не LOFO и не подбор весов: веса боевые и не трогаются, меняется только
  пятый член. `exp_024` требует мерить стекинг от `SELF`, поэтому опорная точка
  здесь — смесь с `BASE-1HEAD` того же энкодера, а не с `SEQ-01`.

Запуск:
  PYTHONPATH=. python research/strategies/results/SEQ8/mix_diversity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.blend import aligned
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.validation import calibrate

# боевая смесь SEQ-01-MIX (`exp_025`): NORM / UNC / CAP / DIST / SEQ
MEMBERS = ["S1-E10", "S1-E02", "S1-E03a", "S1-DIST", "SEQ-01-S42"]
WEIGHTS = [0.15, 0.20, 0.10, 0.25, 0.30]
W_SEQ = WEIGHTS[-1]
W_DIST = WEIGHTS[3]
SEED_VAR_FLOOR = 0.00712


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="S04PROD_S42")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="ключи z_* из S04PROD_*_z.npz; по умолчанию — основные")
    ap.add_argument("--out", default="research/strategies/results/SEQ8")
    a = ap.parse_args()

    Z, y, cut = aligned(MEMBERS)
    z_mix = np.average(Z, axis=0, weights=WEIGHTS)
    z_tab = np.average(Z[:-1], axis=0, weights=WEIGHTS[:-1])
    ly = np.log1p(y)

    # строки OOF отсортированы по (cutoff, user_id); нам нужен доступ по user_id внутри фолда
    from src.tracking import load_oof
    d0 = load_oof(MEMBERS[0])
    k0 = np.char.add(np.asarray(d0["cutoff"], dtype="U10"),
                     np.asarray(d0["user_id"]).astype("U20"))
    order = np.argsort(k0)
    uid_sorted = np.asarray(d0["user_id"])[order]
    cut_sorted = np.asarray(d0["cutoff"], dtype="U10")[order]

    rows, W = [], []
    for V, w in zip(VAL_FOLDS_S1, FOLD_WEIGHTS_S1):
        f = ARTIFACTS / f"{a.prefix}-V{V.strftime('%m%d')}_z.npz"
        if not f.exists():
            print(f"{V}: нет {f.name} — пропуск")
            continue
        d = np.load(f)
        uid, grp = d["uid"], d["group"]
        A = grp == 0
        m = cut_sorted == V.isoformat()
        # выравнивание по user_id внутри фолда
        us = uid_sorted[m]
        o = np.argsort(us)
        pos = np.searchsorted(us[o], uid[A])
        assert np.array_equal(us[o][pos], uid[A]), f"{V}: панели не совпали"
        idx = np.flatnonzero(m)[o][pos]
        assert np.allclose(np.log1p(d["y"][A]), ly[idx], atol=1e-6), f"{V}: таргет разошёлся"

        names = a.variants or [k for k in d.files if k.startswith("z_")]
        r_tab = ly[idx] - z_tab[idx]
        base = None
        for n in sorted(names):
            zv = d[n][A].astype(float)
            zt = z_tab[idx]
            row = dict(fold=V.isoformat(), w=w, variant=n[2:],
                       corr_tab=float(np.corrcoef(r_tab, ly[idx] - zv)[0, 1]),
                       var_div=float(np.var(zv - zt)),
                       solo=calibrate(d["y"][A], zv)[1],
                       # слот SEQ: кандидат вместо `SEQ-01`, вес 0.30
                       mix=calibrate(d["y"][A], (1 - W_SEQ) * zt + W_SEQ * zv)[1],
                       # слот DIST: кандидат вместо `S1-DIST`, вес 0.25. Гибрид —
                       # это DIST с чужим интенсивом, поэтому осмысленно проверить
                       # его и как ЗАМЕНУ донора, а не только как замену SEQ-члена.
                       mix_d=calibrate(d["y"][A],
                                       z_mix[idx] - W_DIST * Z[3][idx] + W_DIST * zv)[1])
            if n == "z_BASE_1HEAD":
                base = row["mix"]
            rows.append(row)
        for row in rows:
            if row["fold"] == V.isoformat() and base is not None:
                row["mix_vs_base"] = row["mix"] - base
                row["mix_d_vs_prod"] = row["mix_d"] - calibrate(d["y"][A], z_mix[idx])[1]
        # смесь с боевым SEQ-01 — внешняя опора того же фолда
        mix01 = z_mix[idx]
        rows.append(dict(fold=V.isoformat(), w=w, variant="[смесь с SEQ-01]",
                         corr_tab=float(np.corrcoef(r_tab, ly[idx] - Z[-1][idx])[0, 1]),
                         var_div=float(np.var(Z[-1][idx] - z_tab[idx])),
                         solo=calibrate(d["y"][A], Z[-1][idx])[1],
                         mix=calibrate(d["y"][A], mix01)[1],
                         mix_d=calibrate(d["y"][A], mix01)[1],
                         mix_vs_base=(calibrate(d["y"][A], mix01)[1] - base
                                      if base is not None else None),
                         mix_d_vs_prod=0.0))
        W.append(w)
    assert rows, "не найдено ни одного S04PROD_*_z.npz"

    ww = np.array(W, float)
    vs = sorted({r["variant"] for r in rows})
    print("=== разнообразие и подстановка в боевую смесь (группа A, веса 1:2:4:8) ===")
    print(f"{'вариант':<22} {'corr(ост.,tab)':>15} {'Var(z-z_tab)':>13} "
          f"{'solo':>9} {'слот SEQ':>9} {'Δ к BASE':>10} {'слот DIST':>10} {'Δ к смеси':>10}")
    out = []
    for v in vs:
        sel = [r for r in rows if r["variant"] == v]
        if len(sel) != len(ww):
            continue
        g = lambda k: float(sum(r[k] * r["w"] for r in sel) / ww.sum())
        dm = ([r.get("mix_vs_base") for r in sel]
              if all(r.get("mix_vs_base") is not None for r in sel) else None)
        dmv = float(sum(x * r["w"] for x, r in zip(dm, sel)) / ww.sum()) if dm else None
        dd = ([r.get("mix_d_vs_prod") for r in sel]
              if all(r.get("mix_d_vs_prod") is not None for r in sel) else None)
        ddv = float(sum(x * r["w"] for x, r in zip(dd, sel)) / ww.sum()) if dd else None
        print(f"{v:<22} {g('corr_tab'):>15.5f} {g('var_div'):>13.5f} "
              f"{g('solo'):>9.5f} {g('mix'):>9.5f} "
              + (f"{dmv:>+10.5f}" if dmv is not None else f"{'—':>10}")
              + (f" ({sum(1 for x in dm if x < 0)}/{len(dm)})" if dm else "")
              + f" {g('mix_d'):>10.5f}"
              + (f" {ddv:>+10.5f} ({sum(1 for x in dd if x < 0)}/{len(dd)})"
                 if ddv is not None else ""))
        out.append(dict(variant=v, corr_tab=g("corr_tab"), var_div=g("var_div"),
                        solo=g("solo"), mix_seq_slot=g("mix"), mix_vs_base=dmv,
                        mix_dist_slot=g("mix_d"), mix_d_vs_prod=ddv))
    print(f"\nпорог полезного разнообразия 0.10; пол сидов Var(Δ) = {SEED_VAR_FLOOR}")
    print("подстановка — при БОЕВЫХ весах 0.15/0.20/0.10/0.25/0.30, это не LOFO "
          "и не подбор весов")

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    import polars as pl
    pl.DataFrame(out).write_csv(o / "mix_diversity.csv")
    pl.DataFrame([{k: v for k, v in r.items() if k != "w"} for r in rows]).write_csv(
        o / "mix_diversity_folds.csv")
    print(f"\nзаписано: {o}/mix_diversity.csv, {o}/mix_diversity_folds.csv")


if __name__ == "__main__":
    main()
