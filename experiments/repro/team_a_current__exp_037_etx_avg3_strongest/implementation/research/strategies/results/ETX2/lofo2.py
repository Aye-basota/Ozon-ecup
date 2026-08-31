"""ETX2 — честный LOFO слота SEQ: `ETX-AVG3` и его смеси с TCN (EXP-037, этап 2).

Протокол ДОСЛОВНО `MIX9/lofo_mix.py` (`exp_035`) и `ETX1/lofo_etx.py` (`exp_036`):
опора — отправленный `SEQ-01-MIX` (wCV 1.74834, LB 1.6501764), та же узкая сетка
весов (`CAP` фиксирован 0.10, доля слота SEQ из {0.40, 0.45, 0.50}, шаг 0.05),
веса подбираются на ТРЁХ фолдах и проверяются на ЧЕТВЁРТОМ. Совпадение протокола
обязательно: только тогда новое число сравнимо с уже принятыми −0.00055 (`SEQ-AVG3`)
и −0.00091 (`0.5·ETX-S42 + 0.5·SEQ-AVG3`), а не живёт в своей шкале.

Оба контроля из прошлых карточек считаются здесь же и обязаны воспроизвестись —
это проверка того, что пайплайн не поехал.

Запуск: PYTHONPATH=. python research/strategies/results/ETX2/lofo2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.blend import aligned, fold_masks, shifted_rmsle
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1

CAP, E02, DIST = "S1-E03a", "S1-E02", "S1-DIST"
REF_EXPS = ["S1-E10", E02, CAP, DIST, "SEQ-01-S42"]      # отправленный SEQ-01-MIX
REF_W = [0.15, 0.20, 0.10, 0.25, 0.30]
FIXED_W = [0.10, 0.20, 0.25, 0.45]                       # CAP, E02, DIST, слот SEQ
SEQ_SHARES = [0.40, 0.45, 0.50]
STEP = 0.05
ALPHAS = [0.25, 0.5, 0.75]
OUT = Path("research/strategies/results/ETX2")


def narrow_grid():
    out = []
    for s in SEQ_SHARES:
        rest = round(1.0 - 0.10 - s, 10)
        n = int(round(rest / STEP))
        for i in range(n + 1):
            e = round(i * STEP, 10)
            out.append((0.10, e, round(rest - e, 10), s))
    return out


def fold_cal(Z, ly, masks, w):
    z = np.average(Z, axis=0, weights=np.asarray(w, float))
    return np.array([shifted_rmsle(ly[m], z[m]) for m in masks])


def have(name: str) -> bool:
    return (ARTIFACTS / f"oof_{name}.npz").exists()


def build_members(Z, idx):
    """Кандидаты на слот SEQ. Пропускаются те, чьих OOF ещё нет на диске."""
    m, avail = {}, set(idx)

    def z(n):
        return Z[idx[n]]

    for solo in ["SEQ-AVG3", "SEQ-D3A-AVG3", "ETX-01-S42", "ETX-AVG2", "ETX-AVG3"]:
        if solo in avail:
            m[solo] = z(solo)
    # контроль exp_036: одиночный ETX в паре с TCN обязан дать −0.00091 4/4
    if {"ETX-01-S42", "SEQ-AVG3"} <= avail:
        m["0.5*ETX-S42+0.5*SEQ-AVG3"] = 0.5 * z("ETX-01-S42") + 0.5 * z("SEQ-AVG3")
    for tcn in ["SEQ-AVG3", "SEQ-D3A-AVG3"]:
        for e in ["ETX-AVG3", "ETX-AVG2"]:
            if {e, tcn} <= avail:
                for a in ALPHAS:
                    m[f"{a:g}*{e}+{1 - a:g}*{tcn}"] = a * z(e) + (1 - a) * z(tcn)
    return m


def lofo_one(zm, Z, idx, ly, masks, grid, w_f, ref_fc, folds):
    sub = np.vstack([Z[idx[CAP]], Z[idx[E02]], Z[idx[DIST]], zm])
    FC = np.vstack([fold_cal(sub, ly, masks, w) for w in grid])
    sc = FC @ w_f
    fix_fc = fold_cal(sub, ly, masks, FIXED_W)
    d_fix = fix_fc - ref_fc
    held, chosen = np.zeros(len(folds)), []
    for h in range(len(folds)):
        keep = [i for i in range(len(folds)) if i != h]
        wh = w_f[keep] / w_f[keep].sum()
        b = int(np.argmin(FC[:, keep] @ wh))
        held[h] = FC[b, h]
        chosen.append(list(grid[b]))
    d_lofo = held - ref_fc
    best = int(np.argmin(sc))
    return dict(fix_fc=fix_fc, d_fix=d_fix, held=held, d_lofo=d_lofo,
                chosen=chosen, insample=list(grid[best]), insample_sc=float(sc[best]))


def main() -> None:
    want = ["S1-E10", CAP, E02, DIST, "SEQ-01-S42", "SEQ-AVG3", "SEQ-D3A-AVG3",
            "ETX-01-S42", "ETX-AVG2", "ETX-AVG3"]
    base = sorted({n for n in want if have(n)})
    missing = [n for n in want if n not in base]
    if missing:
        print(f"нет OOF (пропущены): {' '.join(missing)}\n")
    Z, y, cut = aligned(base)
    idx = {e: i for i, e in enumerate(base)}
    ly = np.log1p(y)
    folds, masks = fold_masks(cut)
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()
    print(f"n = {len(y):,} строк OOF, фолды {folds}, веса {list(np.round(w_f, 4))}\n")

    ref_fc = fold_cal(Z[[idx[e] for e in REF_EXPS]], ly, masks, REF_W)
    ref_wcv = float(w_f @ ref_fc)
    print(f"ОПОРА SEQ-01-MIX wCV={ref_wcv:.5f}   " + " ".join(f"{v:.5f}" for v in ref_fc))

    members = build_members(Z, idx)
    z_tab = np.average(Z[[idx[CAP], idx[E02], idx[DIST]]], axis=0,
                       weights=[0.10, 0.20, 0.25])
    grid = narrow_grid()
    rows = []
    print(f"\n{'член слота SEQ':<30}{'соло wCV':>10}{'фикс Δ':>10}{'ф':>4}"
          f"{'ЧЕСТНЫЙ LOFO':>14}{'ф':>4}{'Var-tab':>10}{'corr-res':>10}")
    for m, zm in members.items():
        solo = fold_cal(zm[None, :], ly, masks, [1.0])
        r = lofo_one(zm, Z, idx, ly, masks, grid, w_f, ref_fc, folds)
        d_lofo = float(w_f @ r["d_lofo"])
        vt = float(np.var(zm - z_tab))
        cr = float(np.corrcoef(ly - zm, ly - z_tab)[0, 1])
        print(f"{m:<30}{float(w_f @ solo):>10.5f}"
              f"{float(w_f @ r['fix_fc']) - ref_wcv:>+10.5f}{int((r['d_fix'] < 0).sum()):>3}/4"
              f"{d_lofo:>+14.5f}{int((r['d_lofo'] < 0).sum()):>3}/4{vt:>10.5f}{cr:>10.5f}")
        rows.append(dict(member=m, solo_wcv=float(w_f @ solo), solo_fold=solo.tolist(),
                         fixed_wcv=float(w_f @ r["fix_fc"]),
                         fixed_delta=float(w_f @ r["fix_fc"]) - ref_wcv,
                         fixed_folds=int((r["d_fix"] < 0).sum()),
                         fixed_delta_folds=r["d_fix"].tolist(),
                         lofo_delta=d_lofo, lofo_folds=int((r["d_lofo"] < 0).sum()),
                         lofo_delta_folds=r["d_lofo"].tolist(),
                         lofo_weights=r["chosen"], insample=r["insample"],
                         insample_delta=r["insample_sc"] - ref_wcv,
                         var_vs_tab=vt, corr_resid_tab=cr))

    print("\nпофолдовые дельты честного LOFO (09-04 / 09-18 / 10-02 / 10-16):")
    for r in rows:
        print(f"  {r['member']:<30}" + " ".join(f"{v:+.5f}" for v in r["lofo_delta_folds"])
              + f"   веса 10-16 {r['lofo_weights'][3]}")

    (ARTIFACTS / "ETX2_lofo.json").write_text(
        json.dumps(dict(folds=folds, ref_w=REF_W, ref_exps=REF_EXPS, ref_wcv=ref_wcv,
                        ref_fold_cal=ref_fc.tolist(), rows=rows),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    import csv
    with open(OUT / "lofo_slot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["member", "solo_wcv", "fixed_delta", "fixed_folds", "lofo_delta",
                    "lofo_folds", "d0904", "d0918", "d1002", "d1016", "var_vs_tab",
                    "corr_resid_tab"])
        for r in rows:
            w.writerow([r["member"], f"{r['solo_wcv']:.5f}", f"{r['fixed_delta']:+.5f}",
                        f"{r['fixed_folds']}/4", f"{r['lofo_delta']:+.5f}",
                        f"{r['lofo_folds']}/4"]
                       + [f"{v:+.5f}" for v in r["lofo_delta_folds"]]
                       + [f"{r['var_vs_tab']:.5f}", f"{r['corr_resid_tab']:.5f}"])
    print(f"\nзаписано: artifacts/ETX2_lofo.json, {OUT}/lofo_slot.csv")


if __name__ == "__main__":
    main()
