"""Evaluation and artifact writer for RENEWAL-01.

Kept separate from training so diagnostics can be rerun without retraining the
four folds.  Every ensemble/meta result is evaluated out of fold; blend weights
use leave-one-fold-out selection and the residual Ridge correction is trained on
the other three folds.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, SUBMISSIONS, VAL_FOLDS_S1)
from src.data import sample_submit
from src.renewal import (EPS, PRIMARY_SHRINKAGE, SEED_FLOOR, SHRINKAGES,
                         make_frame)
from src.report import evaluate
from src.validation import calibrate, rmsle_z

FOLDS = [x.isoformat() for x in VAL_FOLDS_S1]
FW = np.asarray(FOLD_WEIGHTS_S1, float) / np.sum(FOLD_WEIGHTS_S1)
BASE_COMPONENTS = {
    "S1-E10": 0.15, "S1-E02": 0.20, "S1-E03a": 0.10,
    "S1-DIST": 0.25, "SEQ-01-S42": 0.30,
}


def _write_csv(name: str, rows: list[dict]) -> Path:
    p = ARTIFACTS / name
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _keys(uid: np.ndarray, cut: np.ndarray) -> np.ndarray:
    return np.char.add(np.asarray(cut, dtype="U10"), np.asarray(uid).astype("U20"))


def aligned_oof(path: Path, uid: np.ndarray, cut: np.ndarray) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if (np.array_equal(d["user_id"], uid) and
            np.array_equal(np.asarray(d["cutoff"], dtype="U10"), cut)):
        return d["z"].astype(float)
    want, have = _keys(uid, cut), _keys(d["user_id"], d["cutoff"])
    order = np.argsort(have)
    pos = np.searchsorted(have[order], want)
    assert np.array_equal(have[order][pos], want), f"OOF keys differ: {path}"
    return d["z"][order[pos]].astype(float)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    ans = {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "precision_05": float(precision_score(y, p >= 0.5, zero_division=0)),
        "recall_05": float(recall_score(y, p >= 0.5, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        ans["roc_auc"] = float(roc_auc_score(y, p))
        ans["pr_auc"] = float(average_precision_score(y, p))
    else:
        ans["roc_auc"] = ans["pr_auc"] = float("nan")
    return ans


def weighted_metric(rows: list[dict], key: str) -> float:
    vals = np.asarray([r[key] for r in rows], float)
    return float(np.dot(FW, vals))


def calibration_rows(y: np.ndarray, p: np.ndarray, model: str,
                     cut: np.ndarray, bins: int = 10) -> list[dict]:
    rows = []
    for fold in ["OOF"] + FOLDS:
        m = np.ones(len(y), bool) if fold == "OOF" else cut == fold
        idx = np.flatnonzero(m)
        order = idx[np.argsort(p[idx])]
        for b, part in enumerate(np.array_split(order, bins), 1):
            rows.append({
                "model": model, "fold": fold, "bin": b, "n": len(part),
                "mean_pred": float(np.mean(p[part])),
                "positive_rate": float(np.mean(y[part])),
                "abs_error": float(abs(np.mean(p[part]) - np.mean(y[part]))),
                "p_min": float(np.min(p[part])), "p_max": float(np.max(p[part])),
            })
    return rows


def load_existing_heads(source: Path, uid: np.ndarray, cut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.empty(len(uid), float)
    mu = np.empty(len(uid), float)
    for fold in FOLDS:
        d = np.load(source / f"mhz_val_{fold}.npz", allow_pickle=False)
        cols = d["aux_cols"].astype(str).tolist()
        im = {int(u): i for i, u in enumerate(d["user_id"])}
        m = cut == fold
        pos = np.fromiter((im[int(u)] for u in uid[m]), dtype=np.int64, count=int(m.sum()))
        p[m] = d["aux"][pos, cols.index("b30_p")]
        mu[m] = d["aux"][pos, cols.index("val_mu")]
    return p, mu


def load_clock_columns(uid: np.ndarray, cut: np.ndarray) -> dict[str, np.ndarray]:
    names = ["clk_n_events", "clk_n_intervals", "clk_recency", "clk_gap_median",
             "clk_gap_cv", "clk_regularity", "clk_rec_over_median",
             "clk_share_near_7", "clk_share_near_14", "clk_share_near_30",
             "clk_share_near_60", "clk_share_near_90"]
    out = {x: np.empty(len(uid), np.float32) for x in names}
    for fold in FOLDS:
        f = make_frame(dt.date.fromisoformat(fold), 3, True)
        m = cut == fold
        assert np.array_equal(f["user_id"].to_numpy(), uid[m])
        for name in names:
            out[name][m] = f[name].to_numpy()
    return out


def per_fold_classification(y: np.ndarray, cut: np.ndarray,
                            models: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    rows, weighted = [], {}
    for name, p in models.items():
        fold_rows = []
        for fold in FOLDS:
            m = cut == fold
            row = {"model": name, "fold": fold, **classification_metrics(y[m], p[m])}
            rows.append(row)
            fold_rows.append(row)
        weighted[name] = {key: weighted_metric(fold_rows, key)
                          for key in ("roc_auc", "pr_auc", "logloss", "brier")}
        weighted[name]["oof_roc_auc"] = classification_metrics(y, p)["roc_auc"]
    return rows, weighted


def fold_cal_scores(y: np.ndarray, z: np.ndarray, cut: np.ndarray) -> np.ndarray:
    return np.asarray([calibrate(y[cut == fold], z[cut == fold])[1] for fold in FOLDS])


def replacement_diagnostics(y: np.ndarray, cut: np.ndarray, zbase: np.ndarray,
                            zclock: np.ndarray, components: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    base_fc = fold_cal_scores(y, zbase, cut)
    rows = []
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, zcomp in components.items():
        if name == "S1-E03a":
            continue  # safety component is never replaced
        max_w = min(BASE_COMPONENTS[name], 0.20)
        weights = np.round(np.arange(0, max_w + 1e-9, 0.025), 3)
        fc = np.empty((len(weights), 4), float)
        for i, weight in enumerate(weights):
            z = zbase + weight * (zclock - zcomp)
            fc[i] = fold_cal_scores(y, z, cut)
            rows.append({
                "component": name, "clock_weight": weight,
                "wcv": float(np.dot(FW, fc[i])),
                "delta_wcv": float(np.dot(FW, fc[i] - base_fc)),
                **{f"fold_{fold}": float(fc[i, j]) for j, fold in enumerate(FOLDS)},
                "fold_wins": int(np.sum(fc[i] < base_fc)),
            })
        grids[name] = (weights, fc)

    # Primary, semantically closest replacement: Clock takes only part of DIST.
    weights, fc = grids["S1-DIST"]
    held = np.empty(4, float)
    chosen = np.empty(4, float)
    lofo_rows = []
    for h in range(4):
        tr = np.arange(4) != h
        wf = FW[tr] / FW[tr].sum()
        best = int(np.argmin(fc[:, tr] @ wf))
        held[h] = fc[best, h]
        chosen[h] = weights[best]
        lofo_rows.append({
            "heldout_fold": FOLDS[h], "selected_weight": float(weights[best]),
            "base_rmsle": float(base_fc[h]), "clock_rmsle": float(held[h]),
            "delta": float(held[h] - base_fc[h]),
        })
    final_weight = float(np.median(chosen))
    zfixed = zbase + final_weight * (zclock - components["S1-DIST"])
    summary = {
        "lofo_delta_wcv": float(np.dot(FW, held - base_fc)),
        "lofo_fold_deltas": (held - base_fc).tolist(),
        "lofo_selected_weights": chosen.tolist(),
        "fixed_weight": final_weight,
        "fixed_report": evaluate(y, zfixed, cut),
        "lofo_rows": lofo_rows,
        "z_fixed": zfixed,
    }
    return rows, summary


def meta_features(zbase: np.ndarray, pclock: np.ndarray,
                  cols: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    conf = np.clip(cols["clk_n_intervals"] / 5.0, 0.0, 1.0)
    reg = np.nan_to_num(cols["clk_regularity"], nan=0.0)
    norm = np.nan_to_num(np.clip(cols["clk_rec_over_median"], 0.0, 4.0), nan=-1.0)
    self_x = np.column_stack([zbase])
    clock_x = np.column_stack([zbase, pclock, conf, reg, norm, pclock * conf])
    return self_x.astype(np.float64), clock_x.astype(np.float64)


def _row_weights(cut: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = np.zeros(mask.sum(), float)
    c = cut[mask]
    for i, fold in enumerate(FOLDS):
        m = c == fold
        if m.any():
            w[m] = FW[i] / m.sum()
    w *= len(w) / w.sum()
    return w


def crossfit_ridge(X: np.ndarray, residual: np.ndarray, cut: np.ndarray,
                   alpha: float) -> tuple[np.ndarray, list[dict]]:
    correction = np.empty(len(residual), float)
    coefs = []
    for fold in FOLDS:
        va = cut == fold
        tr = ~va
        sw = _row_weights(cut, tr)
        scaler = StandardScaler().fit(X[tr], sample_weight=sw)
        xt = scaler.transform(X[tr])
        model = Ridge(alpha=alpha).fit(xt, residual[tr], sample_weight=sw)
        correction[va] = np.clip(model.predict(scaler.transform(X[va])), -0.25, 0.25)
        coefs.append({"fold": fold, "intercept": float(model.intercept_),
                      "coef": model.coef_.tolist()})
    return correction, coefs


def meta_diagnostics(y: np.ndarray, cut: np.ndarray, zbase: np.ndarray,
                     pclock: np.ndarray, cols: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    ly = np.log1p(y)
    residual = ly - zbase
    self_x, clock_x = meta_features(zbase, pclock, cols)
    base_fc = fold_cal_scores(y, zbase, cut)
    rows = []
    primary = None
    for alpha in (1e4, 1e5, 1e6):
        self_corr, _ = crossfit_ridge(self_x, residual, cut, alpha)
        clock_corr, coef = crossfit_ridge(clock_x, residual, cut, alpha)
        for name, corr in (("SELF", self_corr), ("CLOCK", clock_corr)):
            z = zbase + corr
            fc = fold_cal_scores(y, z, cut)
            rows.append({
                "alpha": alpha, "model": name, "wcv": float(np.dot(FW, fc)),
                "delta_vs_base": float(np.dot(FW, fc-base_fc)),
                "corr_correction_residual": float(np.corrcoef(corr, residual)[0, 1]),
                "std_correction": float(np.std(corr)),
                "fold_wins": int(np.sum(fc < base_fc)),
                **{f"delta_{fold}": float(fc[i]-base_fc[i]) for i, fold in enumerate(FOLDS)},
            })
        if alpha == 1e5:
            primary = {
                "z_self": zbase + self_corr, "z_clock": zbase + clock_corr,
                "self_report": evaluate(y, zbase + self_corr, cut),
                "clock_report": evaluate(y, zbase + clock_corr, cut),
                "coefficients": coef, "alpha": alpha,
            }
    assert primary is not None
    return rows, primary


def global_calibrated(y: np.ndarray, z: np.ndarray, cut: np.ndarray) -> np.ndarray:
    out = np.empty(len(z), float)
    for fold in FOLDS:
        m = cut == fold
        d, _ = calibrate(y[m], z[m])
        out[m] = np.maximum(z[m] + d, 0.0)
    return out


def segment_masks(cols: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    n = cols["clk_n_events"]
    ni = cols["clk_n_intervals"]
    rec = cols["clk_recency"]
    med = cols["clk_gap_median"]
    cv = cols["clk_gap_cv"]
    norm = cols["clk_rec_over_median"]
    ans = []
    for value, mask in [("0", n == 0), ("1", n == 1), ("2", n == 2), ("3+", n >= 3)]:
        ans.append(("purchase_history", value, mask))
    ans += [
        ("buyer_type", "regular", (ni >= 3) & np.isfinite(cv) & (cv <= 0.5)),
        ("buyer_type", "irregular", (ni >= 3) & np.isfinite(cv) & (cv > 0.5)),
        ("buyer_type", "dormant", (n >= 1) & np.isfinite(med) &
         (rec > np.maximum(90.0, 1.5 * med))),
        ("buyer_type", "high_frequency", (ni >= 2) & np.isfinite(med) & (med <= 14)),
        ("buyer_type", "low_frequency", (ni >= 1) & np.isfinite(med) & (med > 60)),
    ]
    edges = [-np.inf, 7, 14, 30, 60, 90, np.inf]
    labels = ["0-7", "8-14", "15-30", "31-60", "61-90", "91+"]
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        ans.append(("recency", label, (rec > lo) & (rec <= hi)))
    ans.append(("normalized_recency", "unknown", ~np.isfinite(norm)))
    nedges = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    nlabels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2+"]
    for lo, hi, label in zip(nedges[:-1], nedges[1:], nlabels):
        ans.append(("normalized_recency", label, np.isfinite(norm) & (norm > lo) & (norm <= hi)))
    return ans


def segment_diagnostics(y: np.ndarray, yb: np.ndarray, cut: np.ndarray,
                        p_models: dict[str, np.ndarray], z_models: dict[str, np.ndarray],
                        cols: dict[str, np.ndarray]) -> list[dict]:
    zcal = {name: global_calibrated(y, z, cut) for name, z in z_models.items()}
    rows = []
    for group_type, segment, mask in segment_masks(cols):
        if mask.sum() < 100:
            continue
        for name, p in p_models.items():
            cm = classification_metrics(yb[mask], p[mask])
            rows.append({"group_type": group_type, "segment": segment,
                         "metric_type": "classification", "model": name, **cm})
        base_r = rmsle_z(y[mask], zcal["BASE"][mask])
        for name, z in zcal.items():
            score = rmsle_z(y[mask], z[mask])
            rows.append({
                "group_type": group_type, "segment": segment, "metric_type": "rmsle",
                "model": name, "n": int(mask.sum()), "positive_rate": float(yb[mask].mean()),
                "rmsle": score, "delta_vs_base": score-base_r,
            })
    return rows


def correlation_tables(y: np.ndarray, models: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    names = list(models)
    ly = np.log1p(y)
    pred_rows, resid_rows = [], []
    for a in names:
        for b in names:
            pred_rows.append({"model_a": a, "model_b": b,
                              "correlation": float(np.corrcoef(models[a], models[b])[0, 1])})
            resid_rows.append({"model_a": a, "model_b": b,
                               "correlation": float(np.corrcoef(ly-models[a], ly-models[b])[0, 1])})
    return pred_rows, resid_rows


def disagreement(y: np.ndarray, pclock: np.ndarray, pexisting: np.ndarray) -> dict:
    ch, ex = pclock >= 0.5, pexisting >= 0.5
    clock_only = ch & ~ex
    existing_only = ex & ~ch
    clock_correct = ch == y
    existing_correct = ex == y
    return {
        "pearson": float(pearsonr(pclock, pexisting).statistic),
        "spearman": float(spearmanr(pclock, pexisting).statistic),
        "mean_abs_difference": float(np.mean(np.abs(pclock-pexisting))),
        "threshold_disagreement_rate": float(np.mean(ch != ex)),
        "clock_only_high_n": int(clock_only.sum()),
        "clock_only_high_positive_rate": float(y[clock_only].mean()) if clock_only.any() else None,
        "existing_only_high_n": int(existing_only.sum()),
        "existing_only_high_positive_rate": float(y[existing_only].mean()) if existing_only.any() else None,
        "clock_right_existing_wrong": int(np.sum(clock_correct & ~existing_correct)),
        "existing_right_clock_wrong": int(np.sum(existing_correct & ~clock_correct)),
    }


def maybe_prepare_meta_submission(oof_y: np.ndarray, cut: np.ndarray, zbase: np.ndarray,
                                  pclock: np.ndarray, cols: dict[str, np.ndarray],
                                  meta: dict, source: Path) -> dict:
    report = meta["clock_report"]
    base_report = evaluate(oof_y, zbase, cut)
    deltas = np.asarray(report["fold_cal"]) - np.asarray(base_report["fold_cal"])
    promising = report["wcv"] - base_report["wcv"] <= -0.0005 and \
        int(np.sum(deltas < 0)) >= 3 and deltas[-1] < 0
    result = {"prepared": False, "promising": bool(promising)}
    if not promising:
        return result

    test = np.load(ARTIFACTS / "test_RENEWAL-01.npz", allow_pickle=False)
    uid = test["user_id"]
    z_parts = {}
    variants = {"S1-E10": "S1-NORM", "S1-E02": "S1-UNC", "S1-E03a": "S1-CAP",
                "S1-DIST": "S1-DIST", "SEQ-01-S42": "SEQ-01"}
    for name, variant in variants.items():
        z = np.load(source / f"ztest_{variant}.npy").astype(float)
        u = np.load(source / f"uid_{variant}.npy")
        if not np.array_equal(u, uid):
            pos = {int(x): i for i, x in enumerate(u)}
            z = z[np.fromiter((pos[int(x)] for x in uid), np.int64, len(uid))]
        z_parts[name] = z
    ztest_base = sum(BASE_COMPONENTS[name] * z_parts[name] for name in BASE_COMPONENTS)
    tcols = {
        "clk_n_intervals": test["clk_n_intervals"],
        "clk_regularity": test["clk_regularity"],
        "clk_rec_over_median": test["clk_rec_over_median"],
    }
    _, xtest = meta_features(ztest_base, test["p_clock_30"], tcols)
    _, xtrain = meta_features(zbase, pclock, cols)
    residual = np.log1p(oof_y) - zbase
    sw = _row_weights(cut, np.ones(len(cut), bool))
    scaler = StandardScaler().fit(xtrain, sample_weight=sw)
    model = Ridge(alpha=meta["alpha"]).fit(scaler.transform(xtrain), residual,
                                           sample_weight=sw)
    correction = np.clip(model.predict(scaler.transform(xtest)), -0.25, 0.25)
    ztest = ztest_base + correction
    np.save(ARTIFACTS / "ztest_RENEWAL-01-META.npy", ztest.astype(np.float32))
    np.save(ARTIFACTS / "uid_RENEWAL-01-META.npy", uid)
    delta = 2.3293 - float(ztest.mean())
    zcal = np.maximum(ztest + delta, 0.0)
    pred = np.maximum(np.expm1(zcal), 0.0)
    sub = pl.DataFrame({"user_id": uid, "predict": pred})
    assert np.array_equal(uid, sample_submit()["user_id"].to_numpy())
    path = SUBMISSIONS / "submission_RENEWAL01_meta.csv"
    sub.write_csv(path, float_precision=6)
    result.update(prepared=True, path=str(path), mean_z=float(zcal.mean()),
                  correction_std=float(np.std(correction)))
    return result


def evaluate_experiment(oof: dict[str, np.ndarray], source: Path) -> dict:
    uid = oof["user_id"]
    cut = np.asarray(oof["cutoff"], dtype="U10")
    y = oof["y"].astype(float)
    yb = oof["y_buy"].astype(int)
    pclock = oof["p_clock_30"].astype(float)
    pr0 = oof["p_r0"].astype(float)

    components = {name: aligned_oof(source / f"oof_{name}.npz", uid, cut)
                  for name in BASE_COMPONENTS}
    zbase = sum(BASE_COMPONENTS[name] * components[name] for name in BASE_COMPONENTS)
    base_report = evaluate(y, zbase, cut)
    pexisting, val_mu = load_existing_heads(source, uid, cut)
    zclock = pclock * val_mu  # E[z] = P(buy) * E[z | buy], mathematically in log target space
    zr0 = pr0 * val_mu
    clock_report = evaluate(y, zclock, cut)
    r0_report = evaluate(y, zr0, cut)
    cols = load_clock_columns(uid, cut)

    p_models = {"R0": pr0, "R1_RAW": oof["p_r1_raw"], "CLOCK": pclock,
                "EXISTING_B30": pexisting}
    fold_cls, weighted_cls = per_fold_classification(yb, cut, p_models)
    _write_csv("renewal_01_fold_classification.csv", fold_cls)
    cal_rows = []
    for name, p in p_models.items():
        cal_rows += calibration_rows(yb, p, name, cut)
    _write_csv("renewal_01_calibration_bins.csv", cal_rows)
    calibration = {}
    for name in p_models:
        rows = [r for r in cal_rows if r["model"] == name and r["fold"] == "OOF"]
        total = sum(r["n"] for r in rows)
        calibration[name] = {
            "ece10": float(sum(r["n"] * r["abs_error"] for r in rows) / total),
            "max_bin_error": float(max(r["abs_error"] for r in rows)),
            "mean_pred": float(sum(r["n"] * r["mean_pred"] for r in rows) / total),
            "positive_rate": float(sum(r["n"] * r["positive_rate"] for r in rows) / total),
        }

    sensitivity = []
    for alpha in SHRINKAGES:
        name = f"p_r0_a{int(alpha)}"
        rows = []
        for fold in FOLDS:
            m = cut == fold
            rows.append(classification_metrics(yb[m], oof[name][m]))
        sensitivity.append({"type": "R0_shrinkage", "value": alpha,
                            **{k: weighted_metric(rows, k)
                               for k in ("roc_auc", "pr_auc", "logloss", "brier")}})
    for i in range(oof["p_r1_seeds"].shape[0]):
        rows = []
        for fold in FOLDS:
            m = cut == fold
            rows.append(classification_metrics(yb[m], oof["p_r1_seeds"][i, m]))
        sensitivity.append({"type": "R1_seed", "value": i,
                            **{k: weighted_metric(rows, k)
                               for k in ("roc_auc", "pr_auc", "logloss", "brier")}})
    _write_csv("renewal_01_sensitivity.csv", sensitivity)
    seed_pairs = []
    for i in range(oof["p_r1_seeds"].shape[0]):
        for j in range(i + 1, oof["p_r1_seeds"].shape[0]):
            a, b = oof["p_r1_seeds"][i], oof["p_r1_seeds"][j]
            seed_pairs.append({"seed_i": i, "seed_j": j,
                               "correlation": float(np.corrcoef(a, b)[0, 1]),
                               "var_difference": float(np.var(a-b)),
                               "mean_abs_difference": float(np.mean(np.abs(a-b)))})
    _write_csv("renewal_01_seed_pairs.csv", seed_pairs)

    replacement_rows, replacement = replacement_diagnostics(
        y, cut, zbase, zclock, components)
    _write_csv("renewal_01_replacement_grid.csv", replacement_rows)
    _write_csv("renewal_01_lofo.csv", replacement["lofo_rows"])

    meta_rows, meta = meta_diagnostics(y, cut, zbase, pclock, cols)
    _write_csv("renewal_01_meta.csv", meta_rows)

    z_models = {"BASE": zbase, "CLOCK_TWO_PART": zclock, "R0_TWO_PART": zr0,
                "CLOCK_REPLACE": replacement["z_fixed"], "CLOCK_META": meta["z_clock"]}
    segment_rows = segment_diagnostics(y, yb, cut,
                                       {"R0": pr0, "CLOCK": pclock,
                                        "EXISTING_B30": pexisting}, z_models, cols)
    _write_csv("renewal_01_segments.csv", segment_rows)

    profile_rows = []
    for fold in ["OOF"] + FOLDS:
        m = np.ones(len(y), bool) if fold == "OOF" else cut == fold
        has = m & (cols["clk_n_intervals"] > 0)
        row = {"fold": fold, "n": int(m.sum()), "n_with_interval": int(has.sum())}
        for day in (7, 14, 30, 60, 90):
            row[f"mean_share_near_{day}"] = float(np.nanmean(cols[f"clk_share_near_{day}"][has]))
        profile_rows.append(row)
    _write_csv("renewal_01_clock_profile.csv", profile_rows)

    corr_models = {"CLOCK_TWO_PART": zclock, "BASE": zbase,
                   "E10": components["S1-E10"], "DIST": components["S1-DIST"],
                   "SEQ": components["SEQ-01-S42"]}
    for optional, filename in (("E11", "oof_S1-E11.npz"),
                               ("MHZ", "oof_MHZ-FULL.npz"),
                               ("PTIME", "oof_PT-FULL-AVG3.npz")):
        p = source / filename
        if p.exists():
            corr_models[optional] = aligned_oof(p, uid, cut)
    pred_corr, resid_corr = correlation_tables(y, corr_models)
    _write_csv("renewal_01_prediction_correlations.csv", pred_corr)
    _write_csv("renewal_01_residual_correlations.csv", resid_corr)

    var_delta = float(np.var(zclock-zbase))
    resid_clock_base = float(np.corrcoef(np.log1p(y)-zclock, np.log1p(y)-zbase)[0, 1])
    disagreement_stats = disagreement(yb, pclock, pexisting)
    submission = maybe_prepare_meta_submission(y, cut, zbase, pclock, cols, meta, source)

    fold_rmsle_rows = []
    reports = {"BASE": base_report, "R0_TWO_PART": r0_report,
               "CLOCK_TWO_PART": clock_report,
               "CLOCK_REPLACE": replacement["fixed_report"],
               "META_SELF": meta["self_report"], "CLOCK_META": meta["clock_report"]}
    for name, rep in reports.items():
        for i, fold in enumerate(FOLDS):
            fold_rmsle_rows.append({"model": name, "fold": fold,
                                    "rmsle_cal": rep["fold_cal"][i],
                                    "delta_vs_base": rep["fold_cal"][i]-base_report["fold_cal"][i]})
    _write_csv("renewal_01_fold_rmsle.csv", fold_rmsle_rows)

    n_events = cols["clk_n_events"]
    support = {"0": float(np.mean(n_events == 0)), "1": float(np.mean(n_events == 1)),
               "2": float(np.mean(n_events == 2)), "3plus": float(np.mean(n_events >= 3))}
    segment_auc_delta = []
    seg_index = {(r["group_type"], r["segment"], r["model"]): r for r in segment_rows
                 if r["metric_type"] == "classification"}
    for (group, segment, model), row in seg_index.items():
        if model != "CLOCK":
            continue
        other = seg_index.get((group, segment, "EXISTING_B30"))
        if other and np.isfinite(row["roc_auc"]) and np.isfinite(other["roc_auc"]):
            segment_auc_delta.append({"group": group, "segment": segment,
                                      "n": row["n"],
                                      "auc_delta": row["roc_auc"]-other["roc_auc"]})
    best_segment_auc_delta = max(segment_auc_delta, key=lambda r: r["auc_delta"])

    test_summary = {}
    tp = ARTIFACTS / "test_RENEWAL-01.npz"
    if tp.exists():
        td = np.load(tp, allow_pickle=False)
        for name in ("p_clock_30", "p_r1_raw", "p_r0"):
            x = td[name].astype(float)
            test_summary[name] = {"mean": float(x.mean()), "std": float(x.std()),
                                  "q01": float(np.quantile(x, .01)),
                                  "q50": float(np.quantile(x, .50)),
                                  "q99": float(np.quantile(x, .99))}

    # Persist all aligned specialist inputs required to reproduce diagnostics.
    extended = dict(oof)
    extended.update(p_existing_b30=pexisting.astype(np.float32),
                    conditional_mu=val_mu.astype(np.float32),
                    z_clock_two_part=zclock.astype(np.float32),
                    z_base=zbase.astype(np.float32),
                    z_clock_meta=meta["z_clock"].astype(np.float32))
    for name in ("clk_n_events", "clk_n_intervals", "clk_recency",
                 "clk_gap_median", "clk_gap_cv", "clk_regularity",
                 "clk_rec_over_median"):
        extended[name] = cols[name]
    np.savez_compressed(ARTIFACTS / "oof_RENEWAL-01.npz", **extended)

    primary_report = meta["clock_report"]
    primary_delta = primary_report["wcv"] - base_report["wcv"]
    fold_delta = np.asarray(primary_report["fold_cal"]) - np.asarray(base_report["fold_cal"])
    ensemble_candidate = primary_delta <= -0.0005 and np.sum(fold_delta < 0) >= 3 and fold_delta[-1] < 0
    verdict = "ENSEMBLE-CANDIDATE" if ensemble_candidate else "STOP"
    summary = {
        "experiment": "RENEWAL-01 / Next-Purchase Clock",
        "baseline": "SEQ-01-MIX",
        "n_features": 0,  # filled below from importance file
        "classification": weighted_cls,
        "calibration": calibration,
        "clock_support_share": support,
        "best_segment_auc_delta_vs_existing": best_segment_auc_delta,
        "seed_pair_sensitivity": seed_pairs,
        "disagreement_vs_existing_pbuy": disagreement_stats,
        "rmsle": {
            "base_wcv": base_report["wcv"],
            "r0_two_part_wcv": r0_report["wcv"],
            "clock_two_part_wcv": clock_report["wcv"],
            "replacement_fixed_wcv": replacement["fixed_report"]["wcv"],
            "replacement_lofo_delta": replacement["lofo_delta_wcv"],
            "replacement_lofo_fold_deltas": replacement["lofo_fold_deltas"],
            "meta_self_wcv": meta["self_report"]["wcv"],
            "meta_clock_wcv": primary_report["wcv"],
            "meta_clock_delta": primary_delta,
            "meta_increment_vs_self": primary_report["wcv"]-meta["self_report"]["wcv"],
            "meta_fold_deltas": fold_delta.tolist(),
        },
        "diversity": {
            "var_clock_minus_base": var_delta,
            "seed_floor": SEED_FLOOR,
            "ratio_to_seed_floor": var_delta/SEED_FLOOR,
            "residual_correlation_clock_base": resid_clock_base,
        },
        "submission": submission,
        "test_prediction_summary": test_summary,
        "verdict": verdict,
        "primary_shrinkage": PRIMARY_SHRINKAGE,
    }
    imp = ARTIFACTS / "renewal_01_importance.csv"
    if imp.exists():
        summary["n_features"] = max(0, len(imp.read_text(encoding="utf-8").splitlines())-1)
    (ARTIFACTS / "renewal_01_metrics.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=1, allow_nan=True),
        encoding="utf-8")
    return {"summary": summary, "rmsle_report": primary_report,
            "base_report": base_report, "clock_report": clock_report,
            "r0_report": r0_report}
