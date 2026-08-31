"""ETX1 — сводка гейта Sparse Event Transformer на одном фолде.

Считает по фолду ровно те числа, которыми принимается решение по постановке
`EXP-036`, и ни одного лишнего:

  1. калиброванный RMSLE и Δ к базе;      5. `Var(z − z_SEQ)`, `Var(z − z_DIST_MIX)`;
  2. AUC(1[y>0]);                         6. corr прогнозов и corr остатков;
  3. полоса `rec_buy` 15–60;              7. стоимость прогона (из `curve_*.json`).
  4. 2–15 purchase days / 180д;

Отличие от `research/strategies/results/SEQ4/diag.py` ровно одно и оно
техническое: сюда допускаются не только пофолдовые части `<exp>-V<MMDD>`, но и
СМЕСИ, собранные из общих 4-фолдовых OOF табличных членов (`S1-DIST-MIX`,
`SEQ-01-MIX`) — их на фолде надо ещё построить, отдельного файла у них нет.
Границы сегментов, источник (`feat_<T>_LNone.parquet`) и правило калибровки
(один сдвиг на фолд, внутри сегмента НЕ пересчитывается) те же, поэтому числа
сравнимы с уже опубликованными в `exp_022`/`exp_024`/`SEQ4`.

Запуск:
  PYTHONPATH=. python research/strategies/results/ETX1/diag.py \
      --exps ETX-01-S42 --fold 2025-10-16
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS, DATA_PROCESSED
from src.tracking import load_oof
from src.validation import calibrate, rmsle_z

SEG_COLS = ["rec_buy", "w180_days_buy"]
KEY_SEGMENTS = ["ВСЕ", "rec_buy 15-60", "w180_days_buy 2-15", "пересечение"]

# Опоры сравнения. Пофолдовые части SEQ лежат готовыми; смеси собираются здесь
# из 4-фолдовых OOF табличных членов с боевыми весами (`experiments/submissions.csv`).
FOLD_PARTS = ["SEQ-D3A-S42", "SEQ-D3A-AVG3", "SEQ-AVG3", "SEQ-01-S42", "SEQ-D3A-BASE-S42"]
MIXES = {
    "S1-DIST-MIX": {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45},
    "SEQ-01-MIX": {"S1-E10": 0.15, "S1-E02": 0.20, "S1-E03a": 0.10, "S1-DIST": 0.25,
                   "@SEQ-01-S42": 0.30},
    "SEQAVG3-CLIP-MIX": {"S1-E03a": 0.10, "S1-E02": 0.20, "S1-DIST": 0.25,
                         "@SEQ-AVG3": 0.45},
}


def auc(pos: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(pos):
        return float("nan")
    return float(roc_auc_score(pos.astype(np.int8), score))


def segments(df: pl.DataFrame) -> dict[str, np.ndarray]:
    rb, nb = df["rec_buy"].to_numpy(), df["w180_days_buy"].to_numpy()
    known = ~np.isnan(rb)
    rec = known & (rb >= 15) & (rb <= 60)
    freq = (nb >= 2) & (nb <= 15)
    return {
        "ВСЕ": np.ones(len(rb), bool),
        "rec_buy 15-60": rec,
        "w180_days_buy 2-15": freq,
        "пересечение": rec & freq,
        "rec_buy 0-14": known & (rb <= 14),
        "rec_buy 61-180": known & (rb >= 61) & (rb <= 180),
        "rec_buy 180+": known & (rb > 180),
        "никогда не покупал": ~known,
        "w180_days_buy 0-1": nb <= 1,
        "w180_days_buy 16+": nb >= 16,
    }


def load_fold(name: str, tag: str, fold: str, uid: np.ndarray | None, y):
    """z эксперимента на одном фолде: пофолдовая часть или срез общего OOF."""
    p = ARTIFACTS / f"oof_{name}-{tag}.npz"
    if p.exists():
        d = load_oof(f"{name}-{tag}")
        u, z, yy = np.asarray(d["user_id"]), np.asarray(d["z"], float), np.asarray(d["y"], float)
    else:
        d = load_oof(name)
        m = np.asarray(d["cutoff"], dtype="U10") == fold
        u, z, yy = (np.asarray(d["user_id"])[m], np.asarray(d["z"], float)[m],
                    np.asarray(d["y"], float)[m])
    o = np.argsort(u)
    u, z, yy = u[o], z[o], yy[o]
    if uid is not None:
        assert np.array_equal(u, uid), f"{name}: другой набор строк"
        assert np.allclose(yy, y), f"{name}: другой таргет"
    return u, z, yy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True, help="кандидаты, без суффикса -V<MMDD>")
    ap.add_argument("--fold", default="2025-10-16")
    ap.add_argument("--base", default="SEQ-D3A-S42", help="опора дельт")
    ap.add_argument("--out", default="research/strategies/results/ETX1")
    a = ap.parse_args()

    V = dt.date.fromisoformat(a.fold)
    tag = f"V{V.strftime('%m%d')}"
    cand = [e for e in dict.fromkeys(a.exps) if (ARTIFACTS / f"oof_{e}-{tag}.npz").exists()]
    missing = [e for e in a.exps if e not in cand]
    if missing:
        print(f"нет OOF (ещё не посчитаны): {' '.join(missing)}")
    assert cand, "ни одного посчитанного кандидата"

    uid, z0, y = load_fold(cand[0], tag, a.fold, None, None)
    ly, pos = np.log1p(y), y > 0
    z_raw = {cand[0]: z0}
    for e in cand[1:] + FOLD_PARTS:
        _, z, _ = load_fold(e, tag, a.fold, uid, y)
        z_raw[e] = z
    parts = {}
    for mix, w in MIXES.items():
        acc = np.zeros(len(uid))
        for k, v in w.items():
            if k.startswith("@"):
                acc += v * z_raw[k[1:]]
            else:
                if k not in parts:
                    parts[k] = load_fold(k, tag, a.fold, uid, y)[1]
                acc += v * parts[k]
        z_raw[mix] = acc
    order = list(dict.fromkeys(cand + FOLD_PARTS + list(MIXES)))
    base = a.base
    assert base in z_raw, f"опора {base} не найдена"

    off, z_cal = {}, {}
    for e in order:
        d, _ = calibrate(y, z_raw[e])
        off[e], z_cal[e] = d, np.maximum(z_raw[e] + d, 0.0)

    # --- главная таблица -----------------------------------------------------
    rb_res = ly - z_cal[base]
    rows = []
    for e in order:
        r = ly - z_cal[e]
        rows.append(dict(
            exp=e, fold=a.fold, n=len(uid), offset=off[e], rmsle=rmsle_z(y, z_raw[e]),
            rmsle_cal=rmsle_z(y, z_cal[e]),
            d_rmsle_cal=rmsle_z(y, z_cal[e]) - rmsle_z(y, z_cal[base]),
            auc=auc(pos, z_cal[e]), d_auc=auc(pos, z_cal[e]) - auc(pos, z_cal[base]),
            # Var(Δ) — на сохранённых z ДО пофолдового сдвига, как в exp_018/025/026
            var_vs_base=float(np.var(z_raw[e] - z_raw[base])),
            var_vs_distmix=float(np.var(z_raw[e] - z_raw["S1-DIST-MIX"])),
            corr_pred=float(np.corrcoef(z_raw[e], z_raw[base])[0, 1]),
            corr_pred_distmix=float(np.corrcoef(z_raw[e], z_raw["S1-DIST-MIX"])[0, 1]),
            corr_resid=float(np.corrcoef(r, rb_res)[0, 1]),
            corr_resid_distmix=float(np.corrcoef(r, ly - z_cal["S1-DIST-MIX"])[0, 1]),
            mean_z=float(z_raw[e].mean()), std_z=float(z_raw[e].std())))
    main_df = pl.DataFrame(rows)

    # --- сегменты ------------------------------------------------------------
    f = pl.read_parquet(DATA_PROCESSED / f"feat_{V.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id"] + SEG_COLS)
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    assert f.height == len(uid)
    segs = segments(f)
    srows = []
    for name, m in segs.items():
        for e in order:
            srows.append(dict(segment=name, exp=e, n=int(m.sum()), share=float(m.mean()),
                              rmsle=rmsle_z(y[m], z_cal[e][m]), auc=auc(pos[m], z_cal[e][m]),
                              mse_share=float(((ly - z_cal[e]) ** 2)[m].sum()
                                              / ((ly - z_cal[e]) ** 2).sum())))
    bs = {r["segment"]: r for r in srows if r["exp"] == base}
    seg_df = pl.DataFrame(srows).with_columns(
        d_rmsle=pl.Series([r["rmsle"] - bs[r["segment"]]["rmsle"] for r in srows]),
        d_auc=pl.Series([r["auc"] - bs[r["segment"]]["auc"] for r in srows]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main_df.write_csv(out / f"fold_{tag}.csv")
    seg_df.write_csv(out / f"segments_{tag}.csv")

    w = max(len(e) for e in order) + 1
    print(f"\n=== фолд {a.fold}, {len(uid):,} строк, опора {base} ===")
    print(f"{'модель':<{w}} {'RMSLE_cal':>10} {'Δ база':>9} {'AUC':>8} {'ΔAUC':>9} "
          f"{'Var−SEQ':>8} {'Var−MIX':>8} {'corr_res':>9} {'сдвиг':>7}")
    for r in main_df.to_dicts():
        print(f"{r['exp']:<{w}} {r['rmsle_cal']:>10.5f} {r['d_rmsle_cal']:>+9.5f} "
              f"{r['auc']:>8.5f} {r['d_auc']:>+9.5f} {r['var_vs_base']:>8.5f} "
              f"{r['var_vs_distmix']:>8.5f} {r['corr_resid']:>9.5f} {r['offset']:>+7.3f}")

    print()
    for name in KEY_SEGMENTS:
        print(f"-- {name} (доля {segs[name].mean():.3f})")
        for r in seg_df.filter(pl.col("segment") == name).to_dicts():
            print(f"   {r['exp']:<{w}} RMSLE {r['rmsle']:.5f} ({r['d_rmsle']:+.5f})  "
                  f"AUC {r['auc']:.5f} ({r['d_auc']:+.5f})  доля MSE {r['mse_share']:.3f}")

    # --- стоимость прогона ---------------------------------------------------
    cost = []
    for e in cand:
        p = ARTIFACTS / f"curve_{e}-{tag}.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8"))
        cost.append(dict(exp=e, n_params=c.get("n_params"), runtime_min=c["runtime_s"] / 60,
                         peak_vram_gb=c.get("peak_vram_gb"),
                         epochs=c["cfg"].get("epochs"), batch=c["cfg"].get("batch"),
                         n_tok=c["cfg"].get("n_tok"),
                         tau_min=min(c.get("tau_final") or [float("nan")]),
                         tau_max=max(c.get("tau_final") or [float("nan")])))
    if cost:
        pl.DataFrame(cost).write_csv(out / f"cost_{tag}.csv")
        print("\nстоимость прогона:")
        for c in cost:
            print(f"   {c['exp']:<{w}} параметров {c['n_params']:,}  "
                  f"{c['runtime_min']:.1f} мин  пик VRAM {c['peak_vram_gb']:.2f} ГБ  "
                  f"tau {c['tau_min']:.1f}..{c['tau_max']:.1f}д")
    print(f"\nзаписано: {out}/fold_{tag}.csv, segments_{tag}.csv")


if __name__ == "__main__":
    main()
