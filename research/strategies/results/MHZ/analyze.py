"""Диагностика MHZ (exp_024) по СОХРАНЁННЫМ артефактам. Ничего не переобучает.

Читает `artifacts/mhz_val_<V>.npz` (aux-предсказания и разметка на валидации),
OOF четырёх членов боевой смеси `S1-DIST-MIX` и OOF арок стекинга. Пишет таблицы
в `research/strategies/results/MHZ/`.

Запуск: python research/strategies/results/MHZ/analyze.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.mhz import AUX_COLS, HAZ_EDGES, POISON_START, VARIANTS, mix_z
from src.validation import calibrate, rmsle_z

OUT = Path(__file__).resolve().parent
ARMS = list(VARIANTS)
IX = {c: i for i, c in enumerate(AUX_COLS)}
W = np.asarray(FOLD_WEIGHTS_S1, float)
W = W / W.sum()


def wavg(per_fold: dict) -> float:
    return float(sum(W[i] * per_fold[V.isoformat()] for i, V in enumerate(VAL_FOLDS_S1)))


def auc(lab, score) -> float:
    lab = np.asarray(lab).astype(np.int8)
    if lab.min() == lab.max() or len(lab) < 200:
        return float("nan")
    return float(roc_auc_score(lab, score))


def arm_z(arm: str, V) -> np.ndarray | None:
    p = ARTIFACTS / f"oof_MHZ-{arm}-V{V.strftime('%m%d')}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    o = np.argsort(np.asarray(d["user_id"]))
    return np.asarray(d["z"])[o]


def load_fold(V):
    p = ARTIFACTS / f"mhz_val_{V.isoformat()}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    o = np.argsort(np.asarray(d["user_id"]))
    out = {k: np.asarray(d[k])[o] for k in ("user_id", "y", "gap", "n30", "rec_buy",
                                            "w180_days_buy")}
    out["aux"] = np.asarray(d["aux"])[o]
    zm, um, ym = mix_z(V)
    assert np.array_equal(um, out["user_id"]), f"{V}: смесь на другом наборе строк"
    assert np.allclose(ym, out["y"], rtol=1e-4), f"{V}: таргет разошёлся со смесью"
    out["z_mix"] = zm
    out["arms"] = {a: arm_z(a, V) for a in ARMS}
    return out


# --------------------------------------------------------------- таблицы
def t_folds(F) -> pl.DataFrame:
    """wCV и пофолдовый калиброванный RMSLE каждой арки плюс боевая смесь."""
    rows = []
    for name, get in [("S1-DIST-MIX", lambda f: f["z_mix"])] + \
                     [(a, (lambda a: lambda f: f["arms"][a])(a)) for a in ARMS]:
        per, ok = {}, True
        for V in VAL_FOLDS_S1:
            f = F.get(V)
            z = get(f) if f else None
            if z is None:
                ok = False
                continue
            per[V.isoformat()] = calibrate(f["y"], z)[1]
        r = dict(model=name, **{k: round(v, 5) for k, v in per.items()})
        r["wcv"] = round(wavg(per), 5) if ok and len(per) == 4 else None
        rows.append(r)
    base = next(r for r in rows if r["model"] == "BASE")
    mix = next(r for r in rows if r["model"] == "S1-DIST-MIX")
    for r in rows:
        r["d_base"] = None if r["wcv"] is None or base["wcv"] is None else round(
            r["wcv"] - base["wcv"], 5)
        r["d_mix"] = None if r["wcv"] is None or mix["wcv"] is None else round(
            r["wcv"] - mix["wcv"], 5)
    return pl.DataFrame(rows)


SEGMENTS = {
    "all": lambda f: np.ones(len(f["y"]), bool),
    "rec_buy 15-60": lambda f: (f["rec_buy"] >= 15) & (f["rec_buy"] <= 60),
    "w180_days_buy 2-15": lambda f: (f["w180_days_buy"] >= 2) & (f["w180_days_buy"] <= 15),
}


def t_segments(F) -> pl.DataFrame:
    """AUC(y>0) и RMSLE в целевых сегментах диагностики — там, где сидит 70% ошибки."""
    rows = []
    for V in VAL_FOLDS_S1:
        f = F.get(V)
        if not f:
            continue
        for sname, sel in SEGMENTS.items():
            m = sel(f)
            lab = (f["y"][m] > 0)
            for name, z in [("S1-DIST-MIX", f["z_mix"])] + \
                           [(a, f["arms"][a]) for a in ARMS if f["arms"][a] is not None]:
                zz = z[m]
                rows.append(dict(fold=V.isoformat(), segment=sname, model=name,
                                 n=int(m.sum()),
                                 auc=round(auc(lab, zz), 5),
                                 rmsle=round(rmsle_z(f["y"][m], zz + calibrate(f["y"], z)[0]), 5)))
    return pl.DataFrame(rows)


def t_seg_auc(F) -> pl.DataFrame:
    """AUC(y>0) САМИХ сигналов активности в целевых сегментах.

    Главный вопрос эксперимента в чистом виде: даёт ли многогоризонтная разметка
    ранжирование активности лучше, чем то, что уже есть в боевой смеси. Ориентиры
    `exp_014`: `1 - p_0` головы распределения = 0.84689, специально обученный
    бинарный классификатор = 0.8457, `ẑ` смеси = 0.84234.
    """
    sig = {"haz_p30": lambda A: A[:, IX["haz_p30"]],
           "haz_p60": lambda A: A[:, IX["haz_p60"]],
           "cnt_act": lambda A: 1 - A[:, IX["cnt_p0"]],
           "cnt_en": lambda A: A[:, IX["cnt_en"]],
           "b30_p": lambda A: A[:, IX["b30_p"]]}
    rows = []
    for V in VAL_FOLDS_S1:
        f = F.get(V)
        if not f:
            continue
        for sname, sel in SEGMENTS.items():
            m = sel(f)
            lab = f["y"][m] > 0
            r = dict(fold=V.isoformat(), segment=sname, n=int(m.sum()),
                     mix_z=round(auc(lab, f["z_mix"][m]), 5))
            for k, fn in sig.items():
                r[k] = round(auc(lab, fn(f["aux"])[m]), 5)
            r["best_minus_mix"] = round(max(r[k] for k in sig) - r["mix_z"], 5)
            rows.append(r)
    return pl.DataFrame(rows)


def t_horizons(F) -> pl.DataFrame:
    """Качество кумулятивов hazard как предсказаний buy_h.

    `poison` = окно (V, V + h] задевает гарантированную область панели
    2025-11-16..2026-02-13: там метка смещена вверх отбором, и число справочное.
    """
    rows = []
    cols = ["haz_p7", "haz_p14", "haz_p21", "haz_p30", "haz_p45", "haz_p60"]
    for V in VAL_FOLDS_S1:
        f = F.get(V)
        if not f:
            continue
        for h, c in zip(HAZ_EDGES, cols):
            lab = (f["gap"] <= h).astype(np.int8)
            p = f["aux"][:, IX[c]]
            rows.append(dict(
                fold=V.isoformat(), h=h, col=c,
                poison=bool(V + dt.timedelta(days=h) >= POISON_START),
                rate=round(float(lab.mean()), 5), mean_p=round(float(p.mean()), 5),
                auc=round(auc(lab, p), 5),
                # чем уже располагает боевая смесь: её ẑ как ранжировщик того же события
                auc_mix_z=round(auc(lab, f["z_mix"]), 5),
                brier=round(float(np.mean((p - lab) ** 2)), 5),
                bias=round(float(p.mean() - lab.mean()), 5)))
    return pl.DataFrame(rows)


def t_heads(F) -> pl.DataFrame:
    """Качество самих голов: активность, счёт, условная величина."""
    rows = []
    for V in VAL_FOLDS_S1:
        f = F.get(V)
        if not f:
            continue
        y, n30, A = f["y"], f["n30"], f["aux"]
        pos = y > 0
        rows.append(dict(
            fold=V.isoformat(), n=len(y),
            auc_haz_p30=round(auc(pos, A[:, IX["haz_p30"]]), 5),
            auc_cnt_act=round(auc(pos, 1 - A[:, IX["cnt_p0"]]), 5),
            auc_b30=round(auc(pos, A[:, IX["b30_p"]]), 5),
            auc_mix_z=round(auc(pos, f["z_mix"]), 5),
            auc_selfz=round(auc(pos, A[:, IX["selfz"]]), 5),
            en_mean=round(float(A[:, IX["cnt_en"]].mean()), 4),
            n30_mean=round(float(n30.mean()), 4),
            en_corr=round(float(np.corrcoef(A[:, IX["cnt_en"]], n30)[0, 1]), 5),
            edays_corr=round(float(np.corrcoef(A[:, IX["haz_edays"]],
                                               np.minimum(f["gap"], 75))[0, 1]), 5),
            val_rmse_pos=round(float(np.sqrt(np.mean(
                (np.log1p(y[pos]) - A[pos, IX["val_mu"]]) ** 2))), 5),
            r2_n30=round(1 - float(np.mean((A[:, IX["cnt_en"]] - n30) ** 2) / n30.var()), 5)))
    return pl.DataFrame(rows)


def t_shape(F) -> pl.DataFrame:
    """Даёт ли форма кривой информацию СВЕРХ p30 — без единой подгонки.

    Внутри дециля `haz_p30` (то есть при почти одинаковой оценке P(buy30)) меряется
    AUC наклонов кривой как предсказателей `1[y>0]`. 0.5 = формы нет.
    """
    rows = []
    for V in VAL_FOLDS_S1:
        f = F.get(V)
        if not f:
            continue
        p30, A, y = f["aux"][:, IX["haz_p30"]], f["aux"], f["y"]
        q = np.quantile(p30, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(q[1:-1], p30, "right"), 0, 9)
        for c in ("haz_sl730", "haz_sl3060", "haz_edays", "cnt_en", "cnt_ge4"):
            a, n = [], []
            for d in range(10):
                m = dec == d
                v = auc(y[m] > 0, A[m, IX[c]])
                if not np.isnan(v):
                    a.append(v)
                    n.append(int(m.sum()))
            rows.append(dict(fold=V.isoformat(), col=c, n_dec=len(a),
                             auc_within_p30=round(float(np.average(a, weights=n)), 5),
                             auc_min=round(float(min(a)), 5), auc_max=round(float(max(a)), 5),
                             auc_marginal=round(auc(y > 0, A[:, IX[c]]), 5)))
    return pl.DataFrame(rows)


def t_diversity(F) -> pl.DataFrame:
    """Var(z_new - z_base) и корреляции с боевой смесью — критерии разнообразия."""
    rows = []
    for V in VAL_FOLDS_S1:
        f = F.get(V)
        if not f:
            continue
        ly = np.log1p(f["y"])
        rm = ly - f["z_mix"]
        for a in ARMS:
            z = f["arms"][a]
            if z is None:
                continue
            r = ly - z
            zb = f["arms"]["BASE"]
            rows.append(dict(
                fold=V.isoformat(), arm=a,
                var_vs_mix=round(float(np.var(z - f["z_mix"])), 5),
                var_vs_base=round(float(np.var(z - zb)) if zb is not None else float("nan"), 5),
                corr_pred_mix=round(float(np.corrcoef(z, f["z_mix"])[0, 1]), 5),
                corr_resid_mix=round(float(np.corrcoef(r, rm)[0, 1]), 5)))
    return pl.DataFrame(rows)


def main():
    F = {}
    for V in VAL_FOLDS_S1:
        f = load_fold(V)
        if f:
            F[V] = f
            print(f"  {V}: {len(f['y']):,} строк, арок {sum(v is not None for v in f['arms'].values())}")
    if not F:
        print("нет артефактов mhz_val_*.npz")
        return
    for name, fn in [("folds", t_folds), ("segments", t_segments), ("seg_auc", t_seg_auc),
                     ("horizons", t_horizons), ("heads", t_heads), ("shape", t_shape),
                     ("diversity", t_diversity)]:
        df = fn(F)
        df.write_csv(OUT / f"{name}.csv")
        print(f"\n=== {name} ===")
        with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200):
            print(df)


if __name__ == "__main__":
    main()
