"""EXP-041: Ridge как фиксированный 15%-й член STRONGEST-CURRENT.

Запуск полного эксперимента:
    python src/ridge15.py

Матрица train целиком не материализуется. Для каждого fold два последовательных
прохода по штатным ``make_xy(cutoff)`` считают train-only mean/std, затем X'X/X'y.
В нормальных уравнениях используется formulation
``mean((y-Xb)^2) + lambda * ||b||^2``; intercept не штрафуется.
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import (ARTIFACTS, CUTOFF_TEST, FOLD_WEIGHTS_S1, SEED, SUBMISSIONS,
                        VAL_FOLDS_S1, cutoff_grid)
from src.features import feature_names, make_xy
from src.report import evaluate
from src.submit import check_submission
from src.tracking import load_oof, save_oof
from src.validation import calibrate, rmsle_z

EXP_ID = "RIDGE15"
RESULTS = Path(__file__).resolve().parent.parent / "research" / "strategies" / "results" / EXP_ID
LAMBDAS = np.asarray([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=np.float64)
BATCH_ROWS = 32768
LEVEL = 2.3293
SEED_VARIATION_VAR = 0.00712

CAP, UNC, DIST, ETX, SEQ = "S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3"
COMPONENTS = [CAP, UNC, DIST, ETX, SEQ]
BASE_WEIGHTS = {CAP: 0.10, UNC: 0.20, DIST: 0.25, ETX: 0.225, SEQ: 0.225}
MAIN_WEIGHTS = {CAP: 0.10, UNC: 0.05, DIST: 0.25, EXP_ID: 0.15, ETX: 0.225, SEQ: 0.225}
CONTROL_WEIGHTS = {CAP: 0.10, UNC: 0.125, DIST: 0.175, EXP_ID: 0.15, ETX: 0.225,
                   SEQ: 0.225}
PROD_TEST_NAMES = {
    CAP: ["S1-CAP"], UNC: ["S1-UNC"], DIST: ["S1-DIST"],
    ETX: ["ETX-01-S42-DCW", "ETX-01-S43-DCW", "ETX-01-S44-DCW"],
    SEQ: ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"],
}


def _log(message: str) -> None:
    print(message, flush=True)


def clean_train_cutoffs(val: dt.date | None) -> list[dt.date]:
    """Штатная clean grid; для fold исключает любой target, касающийся validation."""
    cuts = cutoff_grid(min_history=90)
    if val is not None:
        cuts = [cut for cut in cuts if cut + dt.timedelta(days=30) <= val]
        assert cuts and max(cuts) + dt.timedelta(days=30) <= val
        assert val not in cuts
    return cuts


def select_numeric_features(X: pl.DataFrame) -> list[str]:
    feats = feature_names(X)
    assert "user_id" not in feats
    assert feats, "пустой feature set"
    bad = [name for name in feats if not X.schema[name].is_numeric()]
    assert not bad, f"нечисловые признаки: {bad}"
    return feats


def _matrix(X: pl.DataFrame, feats: list[str]) -> np.ndarray:
    assert "user_id" not in feats
    return X.select(feats).to_numpy().astype(np.float64, copy=False)


def _batches(A: np.ndarray, y: np.ndarray | None = None, batch_rows: int = BATCH_ROWS):
    for start in range(0, len(A), batch_rows):
        stop = min(start + batch_rows, len(A))
        yield A[start:stop], None if y is None else y[start:stop]


def moments_from_arrays(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature-wise finite count/mean/std (ddof=0), suitable for tests and streaming."""
    p = arrays[0].shape[1]
    count = np.zeros(p, np.int64)
    total = np.zeros(p, np.float64)
    total2 = np.zeros(p, np.float64)
    for A in arrays:
        finite = np.isfinite(A)
        count += finite.sum(axis=0)
        clean = np.where(finite, A, 0.0)
        total += clean.sum(axis=0, dtype=np.float64)
        total2 += np.square(clean).sum(axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros(p), where=count > 0)
    var = np.divide(total2, count, out=np.zeros(p), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
    return count, mean, std


def standardize(A: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Train mean/std; любое исходное NaN/inf и non-finite после scale -> 0."""
    finite = np.isfinite(A)
    Z = np.zeros(A.shape, dtype=np.float64)
    np.subtract(A, mean, out=Z, where=finite)
    np.divide(Z, std, out=Z, where=finite)
    Z[~np.isfinite(Z)] = 0.0
    return Z


def solve_ridge_from_gram(gram: np.ndarray, rhs: np.ndarray, n: int,
                          lambdas: np.ndarray = LAMBDAS) -> np.ndarray:
    """Solve all lambdas; first coefficient is an unpenalized intercept."""
    assert gram.shape[0] == gram.shape[1] == len(rhs)
    assert n > 0
    base = gram / float(n)
    target = rhs / float(n)
    out = np.empty((len(rhs), len(lambdas)), np.float64)
    penalty = np.eye(len(rhs), dtype=np.float64)
    penalty[0, 0] = 0.0
    for j, lam in enumerate(np.asarray(lambdas, float)):
        assert lam >= 0
        system = base + float(lam) * penalty
        try:
            out[:, j] = np.linalg.solve(system, target)
        except np.linalg.LinAlgError:
            out[:, j] = np.linalg.lstsq(system, target, rcond=None)[0]
    return out


def fit_ridge_arrays(X: np.ndarray, y_z: np.ndarray, lam: float):
    """Reference in-memory fit used only by unit tests."""
    _, mean, std = moments_from_arrays([np.asarray(X, np.float64)])
    Z = standardize(np.asarray(X, np.float64), mean, std)
    aug = np.column_stack([np.ones(len(Z)), Z])
    coef = solve_ridge_from_gram(aug.T @ aug, aug.T @ y_z, len(Z), np.asarray([lam]))[:, 0]
    return mean, std, coef


def _fold_moments(cuts: list[dt.date], feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = len(feats)
    count = np.zeros(p, np.int64)
    total = np.zeros(p, np.float64)
    total2 = np.zeros(p, np.float64)
    for i, cut in enumerate(cuts, 1):
        X, _ = make_xy(cut, L=None, n_blocks=1, norm_long=True)
        A = _matrix(X, feats)
        for block, _ in _batches(A):
            finite = np.isfinite(block)
            count += finite.sum(axis=0)
            clean = np.where(finite, block, 0.0)
            total += clean.sum(axis=0, dtype=np.float64)
            total2 += np.square(clean).sum(axis=0, dtype=np.float64)
        del X, A
        gc.collect()
        _log(f"    moments {i:02d}/{len(cuts)} {cut}")
    mean = np.divide(total, count, out=np.zeros(p), where=count > 0)
    var = np.divide(total2, count, out=np.zeros(p), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
    return count, mean, std


def _fold_gram(cuts: list[dt.date], feats: list[str], mean: np.ndarray,
               std: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    p = len(feats)
    gram = np.zeros((p + 1, p + 1), np.float64)
    rhs = np.zeros(p + 1, np.float64)
    n = 0
    for i, cut in enumerate(cuts, 1):
        X, y = make_xy(cut, L=None, n_blocks=1, norm_long=True)
        A = _matrix(X, feats)
        yz = np.log1p(np.asarray(y, np.float64))
        for block, yb in _batches(A, yz):
            Z = standardize(block, mean, std)
            nb = len(Z)
            sx = Z.sum(axis=0, dtype=np.float64)
            gram[0, 0] += nb
            gram[0, 1:] += sx
            gram[1:, 0] += sx
            gram[1:, 1:] += Z.T @ Z
            rhs[0] += yb.sum(dtype=np.float64)
            rhs[1:] += Z.T @ yb
            n += nb
        del X, A, y, yz
        gc.collect()
        _log(f"    gram    {i:02d}/{len(cuts)} {cut}  rows={n:,}")
    return gram, rhs, n


def _predict_matrix(X: pl.DataFrame, feats: list[str], mean: np.ndarray, std: np.ndarray,
                    coefficients: np.ndarray) -> np.ndarray:
    A = _matrix(X, feats)
    pred = np.empty((len(A), coefficients.shape[1]), np.float64)
    for start in range(0, len(A), BATCH_ROWS):
        stop = min(start + BATCH_ROWS, len(A))
        Z = standardize(A[start:stop], mean, std)
        pred[start:stop] = coefficients[0] + Z @ coefficients[1:]
    return pred


def train_fold(val: dt.date, feats: list[str]) -> dict:
    cuts = clean_train_cutoffs(val)
    t0 = time.time()
    _log(f"\nfold {val}: {len(cuts)} train cutoffs {cuts[0]}..{cuts[-1]}")
    count, mean, std = _fold_moments(cuts, feats)
    gram, rhs, n = _fold_gram(cuts, feats, mean, std)
    coef = solve_ridge_from_gram(gram, rhs, n)
    Xv, yv = make_xy(val, L=None, n_blocks=3, norm_long=True)
    assert select_numeric_features(Xv) == feats
    pred = _predict_matrix(Xv, feats, mean, std, coef)
    scores = []
    raw = []
    for j in range(len(LAMBDAS)):
        raw.append(rmsle_z(yv, pred[:, j]))
        scores.append(calibrate(yv, pred[:, j])[1])
    _log("    lambda curve: " + " ".join(
        f"{lam:g}={sc:.6f}" for lam, sc in zip(LAMBDAS, scores)))
    return dict(val=val.isoformat(), cuts=[str(c) for c in cuts], n_train=n,
                finite_count=count, mean=mean, std=std, gram=gram, rhs=rhs, coef=coef,
                user_id=Xv["user_id"].to_numpy(), y=np.asarray(yv, np.float64), pred=pred,
                fold_cal=np.asarray(scores), fold_raw=np.asarray(raw), runtime_s=time.time() - t0)


def select_lofo_lambdas(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """curve shape=(lambda, fold); fixed grid, weighted other-three selection."""
    weights = np.asarray(FOLD_WEIGHTS_S1, np.float64)
    chosen = np.empty(curve.shape[1], np.int64)
    for held in range(curve.shape[1]):
        keep = np.arange(curve.shape[1]) != held
        scores = (curve[:, keep] @ weights[keep]) / weights[keep].sum()
        chosen[held] = int(np.argmin(scores))
    prod = int(np.argmin((curve @ weights) / weights.sum()))
    return chosen, LAMBDAS[chosen], prod


def _load_aligned(names: list[str]):
    ds = [load_oof(name) for name in names]
    keys = [np.rec.fromarrays([np.asarray(d["cutoff"], dtype="U10"), d["user_id"]],
                              names="cutoff,user_id") for d in ds]
    orders = [np.argsort(k, order=("cutoff", "user_id")) for k in keys]
    base = keys[0][orders[0]]
    for key, order, name in zip(keys, orders, names):
        assert np.array_equal(key[order], base), f"OOF mismatch: {name}"
    Z = np.vstack([np.asarray(d["z"], np.float64)[o] for d, o in zip(ds, orders)])
    y = np.asarray(ds[0]["y"], np.float64)[orders[0]]
    cut = np.asarray(ds[0]["cutoff"], dtype="U10")[orders[0]]
    uid = np.asarray(ds[0]["user_id"])[orders[0]]
    return Z, y, cut, uid


def _weighted_z(Z: np.ndarray, names: list[str], weights: dict[str, float]) -> np.ndarray:
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    idx = {name: i for i, name in enumerate(names)}
    return sum(float(weight) * Z[idx[name]] for name, weight in weights.items())


def _auc(y: np.ndarray, z: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y > 0, z))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def diversity_rows(y: np.ndarray, cut: np.ndarray, ridge: np.ndarray,
                   refs: dict[str, np.ndarray]) -> list[dict]:
    ly = np.log1p(y)
    rows = []
    scopes = [("ALL", np.ones(len(y), bool))] + [(fold, cut == fold) for fold in sorted(set(cut))]
    for name, z in refs.items():
        for scope, mask in scopes:
            rows.append(dict(component=name, scope=scope, n=int(mask.sum()),
                             var_diff=float(np.var(ridge[mask] - z[mask])),
                             pearson=float(np.corrcoef(ridge[mask], z[mask])[0, 1]),
                             spearman=_spearman(ridge[mask], z[mask]),
                             corr_residuals=float(np.corrcoef(
                                 ly[mask] - ridge[mask], ly[mask] - z[mask])[0, 1])))
    return rows


def _calibrated_z_by_fold(y: np.ndarray, z: np.ndarray, cut: np.ndarray) -> np.ndarray:
    out = z.copy()
    for fold in sorted(set(cut)):
        mask = cut == fold
        out[mask] = np.maximum(z[mask] + calibrate(y[mask], z[mask])[0], 0.0)
    return out


def blend_diagnostics(y: np.ndarray, cut: np.ndarray, uid: np.ndarray, z_base: np.ndarray,
                      candidates: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    base_rep = evaluate(y, z_base, cut)
    base_cal = _calibrated_z_by_fold(y, z_base, cut)
    ly = np.log1p(y)
    summary, segments = [], []
    fold_order = base_rep["folds"]
    for name, z in candidates.items():
        rep = evaluate(y, z, cut)
        cal = _calibrated_z_by_fold(y, z, cut)
        deltas = np.asarray(rep["fold_cal"]) - np.asarray(base_rep["fold_cal"])
        raw_deltas = np.asarray(rep["fold_scores"]) - np.asarray(base_rep["fold_scores"])
        summary.append(dict(
            blend=name, wcv=rep["wcv"], delta_wcv=rep["wcv"] - base_rep["wcv"],
            auc=_auc(y, z), base_auc=_auc(y, z_base), folds_better=int((deltas < 0).sum()),
            fold_delta={fold: float(v) for fold, v in zip(fold_order, deltas)},
            fold_raw_delta={fold: float(v) for fold, v in zip(fold_order, raw_deltas)},
            var_vs_strongest=float(np.var(z - z_base)),
            corr_residuals=float(np.corrcoef(ly - z, ly - z_base)[0, 1]),
            corr_residuals_cal=float(np.corrcoef(ly - cal, ly - base_cal)[0, 1])))

        for fold in fold_order:
            fm = cut == fold
            Xf, _ = make_xy(dt.date.fromisoformat(fold), L=None, n_blocks=3, norm_long=True)
            assert np.array_equal(Xf["user_id"].to_numpy(), uid[fm])
            rec = Xf["rec_buy"].to_numpy()
            buy = Xf["w180_days_buy"].to_numpy()
            seg_defs = {
                "rec_buy 15-60": np.isfinite(rec) & (rec >= 15) & (rec <= 60),
                "w180_days_buy 0-1": np.isfinite(buy) & (buy >= 0) & (buy <= 1),
                "w180_days_buy 2-15": np.isfinite(buy) & (buy >= 2) & (buy <= 15),
                "w180_days_buy >=16": np.isfinite(buy) & (buy >= 16),
                "never purchased": ~np.isfinite(rec),
            }
            zbf, zcf, yf = base_cal[fm], cal[fm], y[fm]
            for seg_name, sm in seg_defs.items():
                if not sm.any():
                    continue
                sb = rmsle_z(yf[sm], zbf[sm])
                sc = rmsle_z(yf[sm], zcf[sm])
                segments.append(dict(blend=name, fold=fold, segment=seg_name, n=int(sm.sum()),
                                     base_rmsle=sb, rmsle=sc, delta_rmsle=sc - sb))
    return summary, segments


def aggregate_segment_rows(rows: list[dict]) -> list[dict]:
    """1:2:4:8 aggregate of the requested per-fold segment deltas."""
    fold_weight = {val.isoformat(): float(weight)
                   for val, weight in zip(VAL_FOLDS_S1, FOLD_WEIGHTS_S1)}
    out = []
    for blend in sorted({row["blend"] for row in rows}):
        for segment in sorted({row["segment"] for row in rows if row["blend"] == blend}):
            selected = [row for row in rows
                        if row["blend"] == blend and row["segment"] == segment]
            den = sum(fold_weight[row["fold"]] for row in selected)
            delta = sum(fold_weight[row["fold"]] * row["delta_rmsle"]
                        for row in selected) / den
            out.append(dict(blend=blend, segment=segment, delta_rmsle_wcv=delta,
                            folds_better=sum(row["delta_rmsle"] < 0 for row in selected)))
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
                             for k, v in row.items()})


def _production(feats: list[str], lam: float, weights: dict[str, float]) -> dict:
    _log(f"\nproduction Ridge, lambda={lam:g}")
    cuts = clean_train_cutoffs(None)
    _, mean, std = _fold_moments(cuts, feats)
    gram, rhs, n = _fold_gram(cuts, feats, mean, std)
    coef = solve_ridge_from_gram(gram, rhs, n, np.asarray([lam]))
    Xt, _ = make_xy(CUTOFF_TEST, L=None, n_blocks=3, with_target=False, norm_long=True)
    uid = Xt["user_id"].to_numpy()
    zr = _predict_matrix(Xt, feats, mean, std, coef)[:, 0]
    np.save(ARTIFACTS / f"ztest_{EXP_ID}.npy", zr.astype(np.float64))
    np.save(ARTIFACTS / f"uid_{EXP_ID}.npy", uid)
    np.savez_compressed(ARTIFACTS / f"model_{EXP_ID}.npz", mean=mean, std=std,
                        coef=coef[:, 0], features=np.asarray(feats), lambda_=lam,
                        seed=SEED, n_train=n)

    zparts = {EXP_ID: zr}
    for component, names in PROD_TEST_NAMES.items():
        vals = []
        for name in names:
            zuid = np.load(ARTIFACTS / f"uid_{name}.npy")
            assert np.array_equal(zuid, uid), f"test uid mismatch: {name}"
            vals.append(np.load(ARTIFACTS / f"ztest_{name}.npy").astype(np.float64))
        zparts[component] = np.mean(vals, axis=0)
    z = sum(weights[name] * zparts[name] for name in weights)
    delta = LEVEL - float(z.mean())
    zcal = np.maximum(z + delta, 0.0)
    pred = np.maximum(np.expm1(zcal), 0.0)
    sub = pl.DataFrame({"user_id": uid, "predict": pred})
    from src.data import sample_submit
    order = sample_submit().select("user_id").with_row_index("o")
    sub = sub.join(order, on="user_id", how="inner").sort("o").drop("o")
    check_submission(sub)
    out = SUBMISSIONS / "submission_RIDGE15.csv"
    sub.write_csv(out, float_precision=6)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return dict(path=str(out), sha256=sha, n=sub.height, lambda_=lam, level=LEVEL,
                delta=delta, mean_log1p=float(np.log1p(sub["predict"].to_numpy()).mean()),
                min=float(pred.min()), max=float(pred.max()), negative=int((pred < 0).sum()),
                nonfinite=int((~np.isfinite(pred)).sum()))


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    X0, _ = make_xy(VAL_FOLDS_S1[0], L=None, n_blocks=3, norm_long=True)
    feats = select_numeric_features(X0)
    stored = ARTIFACTS / "feats_S1-E10.txt"
    if stored.exists():
        expected = stored.read_text(encoding="utf-8").splitlines()
        assert feats == expected, "feature set разошёлся с S1-E10"
    del X0
    gc.collect()
    _log(f"EXP-041 / {EXP_ID}: {len(feats)} S1-E10 features, seed={SEED}")

    folds = [train_fold(val, feats) for val in VAL_FOLDS_S1]
    curve = np.vstack([fold["fold_cal"] for fold in folds]).T
    raw_curve = np.vstack([fold["fold_raw"] for fold in folds]).T
    chosen_idx, chosen_lam, prod_idx = select_lofo_lambdas(curve)
    _log("\nLOFO lambda: " + " ".join(
        f"{val.isoformat()}={lam:g}" for val, lam in zip(VAL_FOLDS_S1, chosen_lam)))
    _log(f"production lambda from all four folds: {LAMBDAS[prod_idx]:g}")

    ridge_uid = np.concatenate([fold["user_id"] for fold in folds])
    ridge_y = np.concatenate([fold["y"] for fold in folds])
    ridge_cut = np.concatenate([[fold["val"]] * len(fold["y"]) for fold in folds])
    ridge_z = np.concatenate([fold["pred"][:, chosen_idx[i]] for i, fold in enumerate(folds)])
    save_oof(EXP_ID, ridge_uid, ridge_cut, ridge_z, ridge_y)
    ridge_rep = evaluate(ridge_y, ridge_z, ridge_cut)
    ridge_auc = _auc(ridge_y, ridge_z)

    names = COMPONENTS + [EXP_ID]
    Z, y, cut, uid = _load_aligned(names)
    idx = {name: i for i, name in enumerate(names)}
    assert np.allclose(y, ridge_y[np.lexsort((ridge_uid, ridge_cut))])
    z_base = _weighted_z(Z, names, BASE_WEIGHTS)
    z_main = _weighted_z(Z, names, MAIN_WEIGHTS)
    z_control = _weighted_z(Z, names, CONTROL_WEIGHTS)
    base_rep = evaluate(y, z_base, cut)
    refs = {UNC: Z[idx[UNC]], DIST: Z[idx[DIST]], "STRONGEST_CURRENT": z_base}
    diversity = diversity_rows(y, cut, Z[idx[EXP_ID]], refs)
    blend_summary, segments = blend_diagnostics(
        y, cut, uid, z_base, {"RIDGE15_MAIN": z_main, "RIDGE15_CONTROL": z_control})
    segment_summary = aggregate_segment_rows(segments)

    def accepted(row: dict) -> bool:
        all_diversity = {r["component"]: r for r in diversity if r["scope"] == "ALL"}
        return (row["delta_wcv"] <= -0.0005 and row["folds_better"] >= 3
                and row["fold_delta"]["2025-10-16"] < 0
                and all_diversity[UNC]["var_diff"] > SEED_VARIATION_VAR
                and all_diversity["STRONGEST_CURRENT"]["var_diff"] > SEED_VARIATION_VAR)

    accepted_rows = [row for row in blend_summary if accepted(row)]
    verdict = "REJECT"
    best = min(blend_summary, key=lambda row: row["delta_wcv"])
    if accepted_rows:
        best = min(accepted_rows, key=lambda row: row["delta_wcv"])
        verdict = "STRONG ACCEPT" if (best["delta_wcv"] <= -0.0010
                                      and best["folds_better"] == 4) else "ACCEPT"

    _write_csv(RESULTS / "lambda_curve.csv", [
        {"lambda": float(lam), **{val.isoformat(): float(curve[i, j])
                                  for j, val in enumerate(VAL_FOLDS_S1)},
         "wcv": float(curve[i] @ np.asarray(FOLD_WEIGHTS_S1) / sum(FOLD_WEIGHTS_S1)),
         "raw_wcv": float(raw_curve[i] @ np.asarray(FOLD_WEIGHTS_S1) / sum(FOLD_WEIGHTS_S1))}
        for i, lam in enumerate(LAMBDAS)])
    _write_csv(RESULTS / "diversity.csv", diversity)
    _write_csv(RESULTS / "blend_summary.csv", blend_summary)
    _write_csv(RESULTS / "segments.csv", segments)
    _write_csv(RESULTS / "segment_summary.csv", segment_summary)
    _write_csv(RESULTS / "folds.csv", [
        {"fold": fold["val"], "n_train": fold["n_train"], "n_val": len(fold["y"]),
         "lambda": float(chosen_lam[i]), "rmsle_cal": ridge_rep["fold_cal"][i],
         "rmsle_raw": ridge_rep["fold_scores"][i], "runtime_s": fold["runtime_s"]}
        for i, fold in enumerate(folds)])

    production = None
    if verdict != "REJECT":
        weights = MAIN_WEIGHTS if best["blend"] == "RIDGE15_MAIN" else CONTROL_WEIGHTS
        production = _production(feats, float(LAMBDAS[prod_idx]), weights)

    summary = dict(
        exp_id="EXP-041", seed=SEED, n_features=len(feats), lambdas=LAMBDAS.tolist(),
        lofo_lambda={val.isoformat(): float(lam) for val, lam in zip(VAL_FOLDS_S1, chosen_lam)},
        production_lambda=float(LAMBDAS[prod_idx]), ridge_wcv=ridge_rep["wcv"],
        ridge_fold_cal=ridge_rep["fold_cal"], ridge_fold_raw=ridge_rep["fold_scores"],
        ridge_auc=ridge_auc, strongest_wcv=base_rep["wcv"], diversity=diversity,
        blends=blend_summary, verdict=verdict, selected_blend=best["blend"],
        production=production, runtime_s=time.time() - started)
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (RESULTS / "config.json").write_text(json.dumps({
        "seed": SEED, "features": "S1-E10", "n_features": len(feats),
        "train_blocks": 1, "validation_blocks": 3, "L": None, "norm_long": True,
        "target": "log1p(GMV30)", "lambdas": LAMBDAS.tolist(),
        "base_weights": BASE_WEIGHTS, "main_weights": MAIN_WEIGHTS,
        "control_weights": CONTROL_WEIGHTS, "level": LEVEL,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    _log(f"\nRidge standalone wCV={ridge_rep['wcv']:.6f}, AUC={ridge_auc:.6f}")
    for row in blend_summary:
        _log(f"{row['blend']}: wCV={row['wcv']:.6f}, delta={row['delta_wcv']:+.6f}, "
             f"folds={row['folds_better']}/4, 10-16={row['fold_delta']['2025-10-16']:+.6f}")
    _log(f"VERDICT: {verdict}")
    return summary


if __name__ == "__main__":
    run()
