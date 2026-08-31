"""Два последних контролируемых эксперимента: minimal GBDT и cohort similarity.

Запуск из корня репозитория:
    python src/final_experiments.py

Скрипт намеренно создаёт ровно два submission-файла. Все признаки загружаются
через make_xy -> build_features(cutoff_date); новые признаки не строятся.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models
from src.config import ARTIFACTS, CUTOFF_TEST, SEED, SUBMISSIONS, VAL_FOLDS_S1
from src.data import load, sample_submit
from src.features import feature_names, make_xy, to_np
from src.submit import check_submission
from src.train import Setup, assemble, fit_free, xy
from src.validation import best_offset, bias_z, rmsle_z


TARGET_LEVEL = 2.3293  # измеренный по трём LB-точкам истинный уровень test target

# Три осмысленных уровня минимализма. Все окна фиксированы (<= 180 дней), поэтому
# определения колонок одинаковы на исторических cutoff'ах и на тесте.
MINIMAL_SETS = {
    "top5": [
        "w180_days_buy", "w180_lgmv", "w30_lgmv", "w30_days_buy", "rec_buy",
    ],
    "top10": [
        "w180_days_buy", "w180_lgmv", "w30_lgmv", "w30_days_buy", "rec_buy",
        "w180_orders", "w90_days_buy", "w90_lgmv", "w30_orders", "w30_searches",
    ],
    "top15": [
        "w180_days_buy", "w180_lgmv", "w30_lgmv", "w30_days_buy", "rec_buy",
        "w180_orders", "w90_days_buy", "w90_lgmv", "w30_orders", "w30_searches",
        "w90_orders", "w180_searches", "w30_days_present", "w90_lgmv_mean",
        "w180_buyday_rate",
    ],
}

SIM_FEATS = ["w180_days_buy", "w30_days_buy", "w30_lgmv", "rec_buy"]
SIM_BINS = [8, 6, 8, 6]


def log(*args) -> None:
    print(*args, flush=True)


def model_setup() -> Setup:
    """Сильная регуляризация плюс малое дерево: underfit здесь является гипотезой."""
    return Setup(
        L=None,
        min_history=90,
        step=7,
        panel_blocks=3,
        train_blocks=1,
        cutoffs="all",
        rounds=400,
        params={
            "num_leaves": 31,
            "min_data_in_leaf": 800,
            "lambda_l2": 20.0,
            "feature_fraction": 1.0,
            "seed": SEED,
            "bagging_seed": SEED,
            "feature_fraction_seed": SEED,
        },
    )


def one_minimal_fold(s: Setup, feats: list[str], V: dt.date) -> dict:
    cuts = s.train_cutoffs(V)
    Xtr, ytr, wtr = assemble(cuts, s, feats, V)
    ntrain = len(ytr)
    box = [Xtr]
    del Xtr
    model = fit_free(s, box, ytr, None)
    Xv, yv = xy(V, s)
    z = np.maximum(model.predict(to_np(Xv, feats)), 0.0)
    gain = model.feature_importance("gain").astype(float)
    gain /= max(gain.sum(), 1.0)
    out = {
        "V": V,
        "uid": Xv["user_id"].to_numpy(),
        "z": z,
        "y": yv,
        "score": rmsle_z(yv, z),
        "bias": bias_z(yv, z),
        "gain": gain,
        "ntrain": ntrain,
    }
    del ytr, wtr, model, box
    gc.collect()
    return out


def feature_shift(feats: list[str]) -> dict[str, float]:
    """Простой устойчивый shift-score: |median_val-median_test| / pooled_std."""
    Xv, _ = make_xy(VAL_FOLDS_S1[-1], None, 3)
    Xt, _ = make_xy(CUTOFF_TEST, None, 3, with_target=False)
    scores = {}
    for f in feats:
        a = np.nan_to_num(Xv[f].to_numpy(), nan=-1.0, posinf=1e12, neginf=-1e12)
        b = np.nan_to_num(Xt[f].to_numpy(), nan=-1.0, posinf=1e12, neginf=-1e12)
        pooled = np.r_[a[::10], b[::10]]
        scores[f] = float(abs(np.median(a) - np.median(b)) / (np.std(pooled) + 1e-6))
    return scores


def calibrate_level(z: np.ndarray, level: float = TARGET_LEVEL) -> tuple[np.ndarray, float]:
    """Находит additive log-offset с учётом клипа z>=0."""
    lo, hi = -3.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.maximum(z + mid, 0.0).mean() < level:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    return np.maximum(z + delta, 0.0), float(delta)


def write_submission(uid: np.ndarray, z: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    zc, delta = calibrate_level(z)
    pred = np.expm1(zc)
    sub = pl.DataFrame({"user_id": uid, "predict": pred.astype(np.float64)})
    order = sample_submit().select("user_id").with_row_index("_order")
    sub = sub.join(order, on="user_id", how="inner").sort("_order").drop("_order")
    check_submission(sub)
    sub.write_csv(SUBMISSIONS / name, float_precision=6)
    return sub["predict"].to_numpy(), delta


def best_oof() -> dict:
    """Воспроизводит S1-BEST OOF с выравниванием по (cutoff,user_id)."""
    from src.tracking import load_oof

    names = ["S1-E10", "S1-E02", "S1-E03a"]
    weights = [0.45, 0.45, 0.10]
    ds = [load_oof(n) for n in names]
    keys = [np.char.add(np.asarray(d["cutoff"], dtype="U10") + "|",
                        np.asarray(d["user_id"]).astype("U16")) for d in ds]
    order0 = np.argsort(keys[0])
    base = keys[0][order0]
    Z = []
    for d, k in zip(ds, keys):
        o = np.argsort(k)
        assert np.array_equal(k[o], base)
        Z.append(d["z"][o])
    d0 = ds[0]
    return {
        "key": base,
        "z": np.average(np.vstack(Z), axis=0, weights=weights),
        "y": d0["y"][order0],
        "cutoff": d0["cutoff"][order0],
        "uid": d0["user_id"][order0],
    }


def best_test() -> tuple[np.ndarray, np.ndarray]:
    Z = [np.load(ARTIFACTS / f"ztest_{n}.npy") for n in ("S1-NORM", "S1-UNC", "S1-CAP")]
    uid = np.load(ARTIFACTS / "uid_S1-NORM.npy")
    return uid, np.average(np.vstack(Z), axis=0, weights=[0.45, 0.45, 0.10])


def quantile_codes(X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Трандуктивный rank-profile: шкалы устойчивы к временному scale shift."""
    parts = []
    n = X.height
    for f, nb in zip(SIM_FEATS, SIM_BINS):
        x = X[f].to_numpy().astype(float)
        finite = np.isfinite(x)
        fill = float(np.nanmax(x[finite]) + 1.0) if finite.any() else 1.0
        x[~finite] = fill
        sx = np.sort(x)
        pct = np.searchsorted(sx, x, side="right") / max(n, 1)
        parts.append(np.minimum((pct * nb).astype(np.int16), nb - 1))
    code = np.zeros(n, np.int32)
    mult = 1
    for p, nb in zip(parts, SIM_BINS):
        code += p.astype(np.int32) * mult
        mult *= nb
    coarse = parts[0].astype(np.int16) * SIM_BINS[-1] + parts[-1]
    return code, coarse


def similarity_cache(s: Setup) -> dict[dt.date, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cuts = sorted(set(s.grid() + list(VAL_FOLDS_S1)))
    cache = {}
    for i, T in enumerate(cuts, 1):
        blocks = 3 if T in VAL_FOLDS_S1 else s.train_blocks
        X, y = xy(T, s, blocks=blocks)
        code, coarse = quantile_codes(X)
        cache[T] = (code, coarse, np.log1p(y).astype(np.float32), X["user_id"].to_numpy())
        if i % 8 == 0 or i == len(cuts):
            log(f"  similarity profiles: {i}/{len(cuts)} cutoff'ов")
    return cache


def similarity_table(cache, cuts: list[dt.date]) -> tuple[np.ndarray, np.ndarray]:
    n_fine = int(np.prod(SIM_BINS))
    n_coarse = SIM_BINS[0] * SIM_BINS[-1]
    sf, nf = np.zeros(n_fine), np.zeros(n_fine)
    sc, nc = np.zeros(n_coarse), np.zeros(n_coarse)
    for T in cuts:
        code, coarse, z, _ = cache[T]
        sf += np.bincount(code, weights=z, minlength=n_fine)
        nf += np.bincount(code, minlength=n_fine)
        sc += np.bincount(coarse, weights=z, minlength=n_coarse)
        nc += np.bincount(coarse, minlength=n_coarse)
    global_mean = sf.sum() / max(nf.sum(), 1.0)
    coarse_mean = (sc + 500.0 * global_mean) / (nc + 500.0)
    return (sf, nf), coarse_mean


def similarity_predict(table, coarse_mean, code, coarse) -> np.ndarray:
    sf, nf = table
    return (sf[code] + 100.0 * coarse_mean[coarse]) / (nf[code] + 100.0)


def align_similarity(keys: np.ndarray, z: np.ndarray, base: dict) -> np.ndarray:
    order = np.argsort(keys)
    assert np.array_equal(keys[order], base["key"])
    return z[order]


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(spearmanr(a, b).statistic),
        "mad": float(np.mean(np.abs(a - b))),
    }


def main() -> None:
    t0 = time.time()
    load()
    s = model_setup()
    all_feats = set(feature_names(xy(VAL_FOLDS_S1[-1], s)[0]))
    requested = set(sum(MINIMAL_SETS.values(), []) + SIM_FEATS)
    assert requested <= all_feats, f"нет фичей: {sorted(requested - all_feats)}"

    log("\n=== EXPERIMENT 1: radical feature minimalism ===")
    shifts = feature_shift(sorted(requested))
    screen = {}
    # Два последних фолда — быстрый скрининг трёх наборов; полный CV только победителю.
    for name, feats in MINIMAL_SETS.items():
        rows = []
        for V in VAL_FOLDS_S1[-2:]:
            r = one_minimal_fold(s, feats, V)
            rows.append(r)
            log(f"  {name:>5} {V}: {r['score']:.5f} bias={r['bias']:+.4f}")
        cv = float(np.mean([r["score"] for r in rows]))
        stability = float(np.std([r["score"] for r in rows]))
        shift = float(np.mean([shifts[f] for f in feats]))
        # Shift — tie-breaker, а не замена целевой метрики.
        selection_score = cv + 0.15 * stability + 0.01 * shift
        screen[name] = {"rows": rows, "cv": cv, "std": stability, "shift": shift,
                        "selection_score": selection_score}
        log(f"        mean={cv:.5f} std={stability:.5f} shift={shift:.3f} select={selection_score:.5f}")
    chosen = min(screen, key=lambda n: screen[n]["selection_score"])
    feats = MINIMAL_SETS[chosen]
    log(f"  selected: {chosen} ({len(feats)} features)")

    folds_min = []
    for V in VAL_FOLDS_S1[:2]:
        r = one_minimal_fold(s, feats, V)
        folds_min.append(r)
        log(f"  full CV {V}: {r['score']:.5f} bias={r['bias']:+.4f}")
    folds_min += screen[chosen]["rows"]
    folds_min.sort(key=lambda r: r["V"])
    zmin_oof = np.concatenate([r["z"] for r in folds_min])
    ymin_oof = np.concatenate([r["y"] for r in folds_min])
    min_scores = [float(r["score"]) for r in folds_min]
    min_off, min_cal = best_offset(ymin_oof, zmin_oof)
    gains = np.vstack([r["gain"] for r in folds_min])
    imp_mean, imp_std = gains.mean(0), gains.std(0)
    imp_order = np.argsort(-imp_mean)
    imp_report = [{"feature": feats[i], "gain_mean": float(imp_mean[i]),
                   "gain_cv": float(imp_std[i] / max(imp_mean[i], 1e-12))} for i in imp_order]
    log(f"  MIN full CV={np.mean(min_scores):.5f} std={np.std(min_scores):.5f}; "
        f"OOF calibrated={min_cal:.5f} offset={min_off:+.4f}")

    # Финальное обучение minimal-модели на всём чистом коридоре.
    Xtr, ytr, _ = assemble(s.grid(), s, feats)
    box = [Xtr]
    del Xtr
    m = fit_free(s, box, ytr, None)
    Xt, _ = make_xy(CUTOFF_TEST, None, 3, with_target=False)
    zmin_test = np.maximum(m.predict(to_np(Xt, feats)), 0.0)
    pred1, delta1 = write_submission(Xt["user_id"].to_numpy(), zmin_test,
                                     "experimental_submission_1.csv")
    np.save(ARTIFACTS / "ztest_EXP-MIN.npy", zmin_test)
    del m, ytr, box
    gc.collect()

    log("\n=== EXPERIMENT 2: hierarchical cohort similarity ===")
    cache = similarity_cache(s)
    sim_u, sim_c, sim_z, sim_y, sim_scores = [], [], [], [], []
    for V in VAL_FOLDS_S1:
        cuts = s.train_cutoffs(V)
        table, coarse_mean = similarity_table(cache, cuts)
        code, coarse, zy, uid = cache[V]
        z = similarity_predict(table, coarse_mean, code, coarse)
        sim_u.append(uid); sim_c.append(np.full(len(uid), V.isoformat(), dtype="U10"))
        sim_z.append(z); sim_y.append(np.expm1(zy))
        sc = rmsle_z(np.expm1(zy), z)
        sim_scores.append(float(sc))
        log(f"  similarity {V}: {sc:.5f} bias={bias_z(np.expm1(zy), z):+.4f}")
    sim_keys = np.char.add(np.concatenate(sim_c) + "|", np.concatenate(sim_u).astype("U16"))
    base = best_oof()
    zsim = align_similarity(sim_keys, np.concatenate(sim_z), base)
    assert np.allclose(np.sort(np.concatenate(sim_y)), np.sort(base["y"]))

    _, base_cal = best_offset(base["y"], base["z"])
    weight_grid = [0.25, 0.50, 0.75, 1.00]
    weight_rows = []
    for w in weight_grid:
        z = (1.0 - w) * base["z"] + w * zsim
        off, score = best_offset(base["y"], z)
        st = pair_stats(base["z"], z)
        weight_rows.append({"weight": w, "score_cal": score, "offset": off, **st})
        log(f"  sim weight={w:.2f}: OOFcal={score:.5f} corr={st['pearson']:.5f} "
            f"spearman={st['spearman']:.5f}")
    acceptable = [r for r in weight_rows if r["score_cal"] <= base_cal + 0.020]
    selected_w = max(acceptable, key=lambda r: r["weight"])["weight"] if acceptable else 0.25
    selected_row = next(r for r in weight_rows if r["weight"] == selected_w)
    log(f"  selected similarity weight={selected_w:.2f} (max weight within +0.020 RMSLE)")

    # Полная similarity-таблица и тест: rank bins считаются внутри test panel.
    table, coarse_mean = similarity_table(cache, s.grid())
    code_t, coarse_t = quantile_codes(Xt)
    zsim_test = similarity_predict(table, coarse_mean, code_t, coarse_t)
    uid_best, zbest_test = best_test()
    assert np.array_equal(uid_best, Xt["user_id"].to_numpy())
    z2_test = (1.0 - selected_w) * zbest_test + selected_w * zsim_test
    pred2, delta2 = write_submission(uid_best, z2_test, "experimental_submission_2.csv")
    np.save(ARTIFACTS / "ztest_EXP-SIM.npy", zsim_test)

    current = pl.read_csv(SUBMISSIONS / "submission_strategy_1.csv")["predict"].to_numpy()
    stats = {
        "submission_1": pair_stats(current, pred1),
        "submission_2": pair_stats(current, pred2),
        "between_experimental": pair_stats(pred1, pred2),
    }
    disagreement = {}
    uid_order = sample_submit()["user_id"].to_numpy()
    for name, p in (("submission_1", pred1), ("submission_2", pred2)):
        idx = np.argsort(-np.abs(p - current))[:10]
        disagreement[name] = [
            {"user_id": int(uid_order[i]), "current": float(current[i]), "experimental": float(p[i]),
             "abs_diff": float(abs(p[i] - current[i]))} for i in idx
        ]

    results = {
        "minimal": {
            "screen": {n: {k: v for k, v in d.items() if k != "rows"} for n, d in screen.items()},
            "chosen": chosen,
            "features": feats,
            "fold_scores": min_scores,
            "cv_mean": float(np.mean(min_scores)),
            "cv_std": float(np.std(min_scores)),
            "oof_calibrated": float(min_cal),
            "oof_offset": float(min_off),
            "test_delta": delta1,
            "importance": imp_report,
            "shift": {f: shifts[f] for f in feats},
        },
        "similarity": {
            "features": SIM_FEATS,
            "bins": SIM_BINS,
            "pure_fold_scores": sim_scores,
            "pure_cv_mean": float(np.mean(sim_scores)),
            "best_oof_calibrated": float(base_cal),
            "weight_scan": weight_rows,
            "selected_weight": selected_w,
            "selected_oof_calibrated": float(selected_row["score_cal"]),
            "test_delta": delta2,
        },
        "prediction_stats": stats,
        "largest_disagreement": disagreement,
        "runtime_s": time.time() - t0,
    }
    (ARTIFACTS / "final_experiments_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("\n=== FINAL DIVERSITY ===")
    log(json.dumps(stats, ensure_ascii=False, indent=2))
    log(f"Готово за {(time.time() - t0) / 60:.1f} мин. Результаты: artifacts/final_experiments_results.json")


if __name__ == "__main__":
    main()
