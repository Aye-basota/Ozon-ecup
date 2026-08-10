"""Обучение и валидация одной конфигурации Strategy 1.

Один запуск = один эксперимент = одна строка в experiments/log.csv + OOF-предсказания.

Примеры:
  python -m src.train --exp S1-B0  --desc "naive baseline"  --L 0 --min-history 90 \
                      --panel-blocks 1 --cutoffs recent3
  python -m src.train --exp S1-E04 --desc "L=180 strict"     --L 180 --min-history 180
  python -m src.train --exp S1-E07 --desc "two-part"         --model two_part
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import time

import numpy as np
import polars as pl

from src.config import (ARTIFACTS, CUTOFF_STEP, HISTORY_L, LGB_ROUNDS, PANEL_BLOCKS,
                        VAL_FOLDS_S1, cutoff_grid)
from src.data import load
from src.features import feature_names, make_xy, to_np
from src.tracking import log_experiment, save_oof
from src.validation import best_offset, bias_z, get_folds, rmsle_z

T0 = time.time()

# --- именованные группы признаков для ablation --------------------------------
GROUPS = {
    "ponly": lambda c: "presence_only" in c or "ponly" in c,
    "trend": lambda c: c.startswith("trend_") or c.startswith("dlog_"),
    "conv": lambda c: c.endswith("_cart2ord") or c.endswith("_srch2cart") or c.endswith("_srch_per_day"),
    "recnorm": lambda c: c.startswith("rec_over_") or c.endswith("gap_cv"),
    "gapstruct": lambda c: c.startswith("gap_") or c.startswith("buygap_"),
    "lgmv": lambda c: "lgmv_mean" in c or "lgmv_std" in c or c.endswith("_gmv_max"),
    "cat": lambda c: "_cat" in c and not c.startswith("cutoff_"),
    "aov": lambda c: c.endswith("_aov") or c.endswith("_gmv_per_day"),
    "recency": lambda c: c in ("rec_any", "rec_search", "rec_cart", "rec_buy", "rec_cat"),
    "weekend": lambda c: c == "weekend_share",
    "cal": lambda c: c.startswith("cutoff_"),
    # признаки, глубина которых зависит от того, сколько истории доступно на cutoff'е:
    # на 2025-04-03 это 92 дня, на тесте — 409. Главные подозреваемые в отказе
    # экстраполяции (eda_findings §7.4). Работает только при L=None.
    "long": lambda c: (c.startswith("w365") or c.startswith("all_")
                       or c.startswith("lifetime_") or c in ("tenure", "first_buy_age")
                       or c.endswith("_365")),
}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def calendar_cols(T: dt.date, n: int) -> dict[str, np.ndarray]:
    doy = T.timetuple().tm_yday
    return {"cutoff_doy_sin": np.full(n, np.sin(2 * np.pi * doy / 365.25), np.float32),
            "cutoff_doy_cos": np.full(n, np.cos(2 * np.pi * doy / 365.25), np.float32),
            "cutoff_month": np.full(n, T.month, np.float32)}


def select_features(all_feats: list[str], drop_groups: list[str], keep_only: list[str] | None):
    feats = list(all_feats)
    for g in drop_groups:
        f = GROUPS[g]
        feats = [c for c in feats if not f(c)]
    if keep_only:
        keep = set()
        for g in keep_only:
            keep |= {c for c in all_feats if GROUPS[g](c)}
        feats = [c for c in feats if c in keep]
    return feats


class Setup:
    """Полная спецификация эксперимента; всё, что влияет на результат."""

    def __init__(self, L=HISTORY_L, min_history=None, step=CUTOFF_STEP, panel_blocks=PANEL_BLOCKS,
                 model="direct", rounds=LGB_ROUNDS, params=None, drop_groups=(), keep_only=None,
                 calendar=False, weight_tau=None, cutoffs=None, vals=None, train_blocks=None,
                 norm_long=False):
        self.L = None if (L is None or L <= 0) else L
        self.min_history = min_history if min_history is not None else (self.L or 90)
        self.step = step
        self.panel_blocks = panel_blocks              # панель ВАЛИДАЦИИ — всегда как в тесте
        self.train_blocks = panel_blocks if train_blocks is None else train_blocks
        self.model = model
        self.rounds = rounds
        self.params = params or {}
        self.drop_groups = list(drop_groups)
        self.keep_only = keep_only
        self.calendar = calendar
        self.weight_tau = weight_tau          # экспоненциальный вес cutoff'ов, дни
        self.norm_long = norm_long            # нормировка длинных окон на доступную историю
        self.cutoffs = cutoffs                # 'all' | 'recentN' | явный список
        self.vals = vals or VAL_FOLDS_S1

    def grid(self):
        return cutoff_grid(self.min_history, self.step)

    def train_cutoffs(self, V: dt.date):
        g = [T for T in self.grid() if T + dt.timedelta(days=30) <= V]
        c = self.cutoffs
        if c is None or c == "all":
            return g
        if isinstance(c, str) and c.startswith("recent"):
            return g[-int(c[6:]):]
        if isinstance(c, str) and c.startswith("every"):        # every30 -> шаг 30 дней
            k = int(c[5:]) // self.step
            return g[::-1][::max(k, 1)][::-1]
        return [T for T in c if T + dt.timedelta(days=30) <= V]

    def as_dict(self):
        return dict(L=self.L, min_history=self.min_history, step=self.step,
                    panel_blocks=self.panel_blocks, train_blocks=self.train_blocks,
                    model=self.model, rounds=self.rounds,
                    params=self.params, drop_groups=self.drop_groups, keep_only=self.keep_only,
                    calendar=self.calendar, weight_tau=self.weight_tau, cutoffs=self.cutoffs,
                    norm_long=self.norm_long)


_XY: dict = {}


def xy(T: dt.date, s: Setup, with_target=True, blocks=None):
    b = s.panel_blocks if blocks is None else blocks
    k = (T, s.L, b, with_target, s.norm_long)
    if len(_XY) > 6:
        _XY.clear()
    if k not in _XY:
        _XY[k] = make_xy(T, s.L, b, with_target=with_target, norm_long=s.norm_long)
    return _XY[k]


def assemble(cuts: list[dt.date], s: Setup, feats: list[str], V: dt.date | None = None):
    Xs, ys, ws = [], [], []
    for T in cuts:
        X, y = xy(T, s, blocks=s.train_blocks)
        A = to_np(X, feats)
        if s.calendar:
            cal = calendar_cols(T, A.shape[0])
            A = np.hstack([A] + [cal[c].reshape(-1, 1) for c in ("cutoff_doy_sin", "cutoff_doy_cos",
                                                                 "cutoff_month")])
        Xs.append(A)
        ys.append(y)
        w = 1.0
        if s.weight_tau and V is not None:
            w = float(np.exp(-((V - T).days) / s.weight_tau))
        ws.append(np.full(A.shape[0], w, np.float32))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def fit(s: Setup, Xtr, ytr, wtr):
    from src import models
    if s.model == "direct":
        return models.train_direct(Xtr, ytr, wtr, s.params, s.rounds)
    if s.model == "two_part":
        return models.train_two_part(Xtr, ytr, wtr, s.params, s.rounds)
    if s.model == "catboost":
        return models.train_catboost(Xtr, ytr, wtr, s.params, s.rounds)
    raise ValueError(s.model)


def fit_free(s: Setup, box: list, ytr, wtr):
    """box = [Xtr]. Матрица биннится и освобождается ДО бустинга (экономит 5+ ГБ)."""
    from src import models
    if s.model in ("direct", "two_part"):
        dss = models.make_datasets(s.model, box[0], ytr, wtr, s.params)
        box[0] = None
        gc.collect()
        if s.model == "direct":
            return models.train_direct_ds(dss[0], s.params, s.rounds)
        return models.train_two_part_ds(dss, s.params, s.rounds)
    m = fit(s, box[0], ytr, wtr)
    box[0] = None
    return m


def infer(s: Setup, m, A) -> np.ndarray:
    from src import models
    if s.model == "two_part":
        return models.predict_two_part(m, A)
    if s.model == "catboost":
        return models.predict_catboost(m, A)
    return m.predict(A)


def run(exp_id: str, desc: str, s: Setup, save_model_feats=False, verbose_imp=False):
    load()
    folds = get_folds(s.min_history, s.step, s.vals)
    base_feats = feature_names(xy(folds[0][1], s)[0])
    feats = select_features(base_feats, s.drop_groups, s.keep_only)
    nfeat = len(feats) + (3 if s.calendar else 0)
    log(f"{exp_id}: {desc}")
    log(f"  L={s.L} min_hist={s.min_history} step={s.step} blocks={s.panel_blocks} "
        f"train_blocks={s.train_blocks} model={s.model} rounds={s.rounds} feats={nfeat}")

    scores, biases, offs, sizes = [], [], [], []
    oof_u, oof_c, oof_z, oof_y = [], [], [], []
    for _, V in folds:
        tr_cuts = s.train_cutoffs(V)
        if not tr_cuts:
            log(f"  val {V}: нет train-cutoff'ов, пропуск")
            continue
        t = time.time()
        Xtr, ytr, wtr = assemble(tr_cuts, s, feats, V)
        n_tr = Xtr.shape[0]
        box = [Xtr]
        del Xtr
        m = fit_free(s, box, ytr, wtr if s.weight_tau else None)
        Xv, yv = xy(V, s)
        Av = to_np(Xv, feats)
        if s.calendar:
            cal = calendar_cols(V, Av.shape[0])
            Av = np.hstack([Av] + [cal[c].reshape(-1, 1) for c in ("cutoff_doy_sin", "cutoff_doy_cos",
                                                                   "cutoff_month")])
        z = np.maximum(infer(s, m, Av), 0.0)
        sc, b = rmsle_z(yv, z), bias_z(yv, z)
        o, sc_o = best_offset(yv, z)
        scores.append(sc); biases.append(b); offs.append(o); sizes.append(len(yv))
        oof_u.append(Xv["user_id"].to_numpy()); oof_c.append([V.isoformat()] * len(yv))
        oof_z.append(z); oof_y.append(yv)
        log(f"  val {V} (n={len(yv):,}, train {len(tr_cuts)} cuts / {n_tr:,} rows, "
            f"gap {(V - max(tr_cuts)).days}d): RMSLE={sc:.5f} bias={b:+.4f} "
            f"off*={o:+.3f}->{sc_o:.5f}  [{time.time() - t:.0f}s]")
        if verbose_imp and s.model == "direct":
            from src.models import importance
            names = feats + (["cutoff_doy_sin", "cutoff_doy_cos", "cutoff_month"] if s.calendar else [])
            for n, v in importance(m, names, 15):
                log(f"      {n:28s} {v:14,.0f}")
            verbose_imp = False
        del ytr, wtr, m, Av, box
        gc.collect()

    cv, sd = float(np.mean(scores)), float(np.std(scores))
    bm = float(np.mean(biases))
    z_all = np.concatenate(oof_z); y_all = np.concatenate(oof_y)
    oof = rmsle_z(y_all, z_all)
    o_all, oof_cal = best_offset(y_all, z_all)
    log(f"  CV mean={cv:.5f} std={sd:.5f} | OOF={oof:.5f} bias={bm:+.4f} "
        f"| после сдвига {o_all:+.3f}: {oof_cal:.5f}")
    save_oof(exp_id, np.concatenate(oof_u), np.concatenate(oof_c), z_all, y_all)
    log_experiment(exp_id=exp_id, description=desc, scenario="S1",
                   n_features=nfeat, model=s.model, params=s.as_dict(),
                   cutoffs=f"{len(s.train_cutoffs(s.vals[-1]))} @ step {s.step}", L=s.L,
                   panel_blocks=s.panel_blocks, fold_scores=scores, cv_mean=cv, cv_std=sd,
                   bias_mean=bm, best_offset=o_all, cv_mean_calib=oof_cal,
                   runtime_s=round(time.time() - T0, 1))
    if save_model_feats:
        (ARTIFACTS / f"feats_{exp_id}.txt").write_text("\n".join(feats), encoding="utf-8")
    return dict(exp_id=exp_id, scores=scores, cv=cv, std=sd, oof=oof, bias=bm,
                offset=o_all, oof_cal=oof_cal, feats=feats)


def build_setup(a) -> Setup:
    return Setup(L=a.L, min_history=a.min_history, step=a.step, panel_blocks=a.panel_blocks,
                 model=a.model, rounds=a.rounds,
                 params={k: v for k, v in [("learning_rate", a.lr), ("num_leaves", a.leaves),
                                           ("min_data_in_leaf", a.min_leaf)] if v is not None},
                 drop_groups=a.drop or [], keep_only=a.keep, calendar=a.calendar,
                 weight_tau=a.weight_tau, cutoffs=a.cutoffs, train_blocks=a.train_blocks,
                 vals=VAL_FOLDS_S1[-a.folds:] if getattr(a, "folds", 0) else None,
                 norm_long=getattr(a, "norm_long", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--L", type=int, default=HISTORY_L, help="глубина истории; 0 = без усечения")
    ap.add_argument("--min-history", type=int, default=None)
    ap.add_argument("--step", type=int, default=CUTOFF_STEP)
    ap.add_argument("--panel-blocks", type=int, default=PANEL_BLOCKS)
    ap.add_argument("--train-blocks", type=int, default=None)
    ap.add_argument("--model", default="direct", choices=["direct", "two_part", "catboost"])
    ap.add_argument("--rounds", type=int, default=LGB_ROUNDS)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--leaves", type=int, default=None)
    ap.add_argument("--min-leaf", type=int, default=None)
    ap.add_argument("--drop", nargs="*", default=None, choices=list(GROUPS))
    ap.add_argument("--keep", nargs="*", default=None, choices=list(GROUPS))
    ap.add_argument("--calendar", action="store_true")
    ap.add_argument("--norm-long", action="store_true",
                    help="нормировать w365_*/tenure на доступную глубину истории")
    ap.add_argument("--weight-tau", type=float, default=None)
    ap.add_argument("--cutoffs", default=None, help="all | recentN | everyN")
    ap.add_argument("--imp", action="store_true")
    ap.add_argument("--folds", type=int, default=0, help="взять только последние N фолдов (скрининг)")
    a = ap.parse_args()
    run(a.exp, a.desc, build_setup(a), save_model_feats=True, verbose_imp=a.imp)


if __name__ == "__main__":
    main()
