"""Honest S1 evaluation and production blend for the pinned team-b-B2 solution.

The teammate code is loaded verbatim from a pinned Git commit.  This keeps the
working ``src/features.py``/``src/train.py`` untouched and makes the comparison
reproducible even if the remote branch moves later.

Run:
    python src/team_b_b2_ensemble.py
"""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.blend import aligned
from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, RAW_PARQUET, ROOT, SEED,
                        SUBMISSIONS, VAL_FOLDS_S1)
from src.validation import calibrate


TEAM_SHA = "88dc69163b1f39aaac55ddfbfc9986e2203cfbdf"
TEAM_SOURCE_SHA256 = {
    "src/features.py": "491166bafef819c2e1b85af9a5cda4deebc43c16a9867f69ec00356d3b75894b",
    "src/train.py": "8f7ec4b191f0c9cd03cd9cea698b1bf0113ce88c02fc438c113aa19454b0898c",
    "src/predict.py": "e1cbf9ab0f36ce3c222323bdce1e213b763f5a0a3dd1363cc989bdd92c4ae7ec",
}
TEAM_COLUMNS = [
    "event_date", "user_id", "search", "cat", "to_cart", "to_ord", "gmv", "searches",
]
OUR_COMPONENTS = ["S1-E03a", "S1-E02", "S1-DIST", "SEQ-AVG3", "ETX-AVG3"]
OUR_WEIGHTS = np.array([0.10, 0.20, 0.25, 0.225, 0.225], dtype=np.float64)
PRODUCTION_LOG_LEVEL = 2.3293
FINAL_TEAM_TRAIN_CUTOFF = dt.date(2025, 10, 17)
TEST_FEATURE_CUTOFF = dt.date(2026, 2, 14)

RUN_DIR = ARTIFACTS / "TEAM_B_B2_EXP069"
RESULT_DIR = ROOT / "research" / "strategies" / "results" / "TEAM_B_B2_EXP069"
SUBMISSION_PATH = SUBMISSIONS / "submission_TEAM_B_B2_OPTIMAL_ENSEMBLE.csv"


T0 = time.time()


def log(*parts: object) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *parts, flush=True)


def git_blob(path: str) -> str:
    raw = subprocess.check_output(["git", "show", f"{TEAM_SHA}:{path}"], cwd=ROOT)
    actual = hashlib.sha256(raw).hexdigest()
    expected = TEAM_SOURCE_SHA256[path]
    if actual != expected:
        raise RuntimeError(f"source hash mismatch for {path}: {actual} != {expected}")
    return raw.decode("utf-8")


def load_team_modules() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    features = types.ModuleType("team_b_b2_features")
    exec(compile(git_blob("src/features.py"), f"{TEAM_SHA}:src/features.py", "exec"), features.__dict__)

    # The only import redirection: keep the exact pinned train source while
    # binding its build_df symbol to the pinned teammate features module.
    train_source = git_blob("src/train.py")
    needle = "from src.features import build_df\n"
    if train_source.count(needle) != 1:
        raise RuntimeError("unexpected team train.py import layout")
    train_source = train_source.replace(needle, "", 1)
    train = types.ModuleType("team_b_b2_train")
    train.__dict__["build_df"] = features.build_df
    exec(compile(train_source, f"{TEAM_SHA}:src/train.py", "exec"), train.__dict__)

    predict = types.ModuleType("team_b_b2_predict")
    exec(compile(git_blob("src/predict.py"), f"{TEAM_SHA}:src/predict.py", "exec"), predict.__dict__)
    return features, train, predict


def synthetic_leakage_audit(team_features: types.ModuleType) -> None:
    cutoff = pd.Timestamp("2025-02-01")
    rows = [
        ("2025-01-15", 1, 1, 0, 0, 0, 0.0, 2),
        ("2025-01-20", 2, 0, 1, 1, 1, 10.0, 0),
        ("2025-02-01", 1, 1, 1, 5, 3, 999.0, 50),
        ("2025-02-10", 2, 1, 1, 7, 4, 777.0, 60),
    ]
    tiny = pd.DataFrame(rows, columns=TEAM_COLUMNS)
    base = team_features.build_features(tiny, cutoff).sort_values("user_id").reset_index(drop=True)
    changed = tiny.copy()
    post = changed["event_date"] >= "2025-02-01"
    for column in ["search", "cat", "to_cart", "to_ord", "gmv", "searches"]:
        changed.loc[post, column] = changed.loc[post, column] * 1000 + 123
    probe = team_features.build_features(changed, cutoff).sort_values("user_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(base, probe, check_exact=True)
    log("synthetic cutoff/leakage audit PASS")


def load_raw() -> pd.DataFrame:
    log(f"loading {RAW_PARQUET.name}, columns={TEAM_COLUMNS}")
    df = pd.read_parquet(RAW_PARQUET, columns=TEAM_COLUMNS)
    df["event_date"] = pd.to_datetime(df["event_date"])
    if len(df) != 30_631_006 or df["user_id"].nunique() != 250_000:
        raise RuntimeError("unexpected raw data dimensions")
    log(f"raw loaded: {len(df):,} rows, {df['user_id'].nunique():,} users")
    return df


def ordered_values(frame: pd.DataFrame, users: np.ndarray, column: str) -> np.ndarray:
    series = frame.set_index("user_id")[column]
    out = series.reindex(users)
    if out.isna().any():
        raise RuntimeError(f"missing {column} for {int(out.isna().sum())} users")
    return out.to_numpy()


def save_models(models: dict[str, object], path: Path, features: list[str]) -> None:
    payload = {
        "team_sha": TEAM_SHA,
        "seed": SEED,
        "features": features,
        "models": models,
    }
    joblib.dump(payload, path, compress=3)


def load_models(path: Path) -> tuple[dict[str, object], list[str]]:
    payload = joblib.load(path)
    if payload["team_sha"] != TEAM_SHA or int(payload["seed"]) != int(SEED):
        raise RuntimeError(f"model provenance mismatch: {path}")
    return payload["models"], list(payload["features"])


def train_or_load(
    df_raw: pd.DataFrame,
    cutoff: dt.date,
    tag: str,
    team_features: types.ModuleType,
    team_train: types.ModuleType,
) -> tuple[dict[str, object], list[str]]:
    model_path = RUN_DIR / f"models_{tag}.joblib"
    if model_path.exists():
        log(f"reusing models: {model_path.name}")
        return load_models(model_path)

    t = time.time()
    log(f"building train frame {tag}: team cutoff={cutoff} (history < cutoff, target [cutoff, cutoff+30))")
    # Pinned build_features accepts date-like values, while its sibling
    # get_target compares without normalising first and therefore requires a
    # pandas Timestamp.  Normalise at this interface seam without changing the
    # pinned teammate source.
    df_train = team_features.build_df(df_raw, pd.Timestamp(cutoff), 30)
    features = team_train.get_feature_columns(df_train)
    if len(features) == 0 or not np.isfinite(df_train[features].to_numpy()).all():
        raise RuntimeError("empty or non-finite teammate feature matrix")
    log(f"train frame {tag}: {df_train.shape[0]:,} x {len(features)} built in {time.time() - t:.1f}s")

    t_fit = time.time()
    models = team_train.make_models()
    models = team_train.fit_models(models, df_train, features)
    log(f"four teammate models fitted for {tag} in {time.time() - t_fit:.1f}s")
    save_models(models, model_path, features)
    del df_train
    gc.collect()
    return models, features


def team_oof(
    df_raw: pd.DataFrame,
    our_y: np.ndarray,
    our_uid: np.ndarray,
    our_cut: np.ndarray,
    team_features: types.ModuleType,
    team_train: types.ModuleType,
    team_predict: types.ModuleType,
) -> None:
    for logical_val in VAL_FOLDS_S1:
        tag = logical_val.strftime("%Y%m%d")
        fold_path = RUN_DIR / f"oof_{tag}.npz"
        if fold_path.exists():
            log(f"reusing OOF: {fold_path.name}")
            continue

        # Teammate cutoff is the first excluded/history-future day.  Therefore
        # logical S1 V maps to V+1.  The latest valid single-snapshot training
        # target begins V-29 and ends V, fully before the validation target.
        val_feature_cutoff = logical_val + dt.timedelta(days=1)
        train_cutoff = logical_val - dt.timedelta(days=29)
        if train_cutoff + dt.timedelta(days=30) != val_feature_cutoff:
            raise AssertionError("cutoff seam arithmetic failed")

        mask = our_cut == logical_val.isoformat()
        fold_uid = our_uid[mask]
        fold_y = our_y[mask]
        models, features = train_or_load(
            df_raw, train_cutoff, f"fold_{tag}", team_features, team_train,
        )

        t = time.time()
        log(f"building validation features {tag}: history through {logical_val}")
        df_val = team_features.build_features(df_raw, val_feature_cutoff)
        if list(df_val[features].columns) != features:
            raise RuntimeError("teammate feature order mismatch")
        pred_log_all = np.asarray(team_predict.predict_log(models, df_val, features), dtype=np.float64)
        pred_gmv_all = np.asarray(team_predict.predict_gmv(models, df_val, features), dtype=np.float64)
        pred_log = ordered_values(
            pd.DataFrame({"user_id": df_val["user_id"], "pred": pred_log_all}), fold_uid, "pred",
        )
        pred_gmv = ordered_values(
            pd.DataFrame({"user_id": df_val["user_id"], "pred": pred_gmv_all}), fold_uid, "pred",
        )

        # Compare the teammate's own target implementation with the canonical
        # S1 OOF target.  This is also an executable audit of the +1 day seam.
        target = team_features.get_target(df_raw, pd.Timestamp(val_feature_cutoff), 30)
        team_y = ordered_values(target, fold_uid, "target_gmv")
        max_target_error = float(np.max(np.abs(team_y - fold_y)))
        if not np.allclose(team_y, fold_y, rtol=1e-7, atol=1e-6):
            raise RuntimeError(f"target mismatch on {tag}: max_abs={max_target_error}")
        if not np.isfinite(pred_log).all() or not np.isfinite(pred_gmv).all() or (pred_gmv < 0).any():
            raise RuntimeError(f"invalid teammate predictions on {tag}")
        if not np.allclose(np.log1p(pred_gmv), np.maximum(pred_log, 0.0), atol=1e-9, rtol=1e-9):
            raise RuntimeError("predict_log/predict_gmv contract mismatch")

        np.savez_compressed(
            fold_path,
            user_id=fold_uid,
            cutoff=np.full(len(fold_uid), logical_val.isoformat()),
            y=fold_y,
            z=np.log1p(pred_gmv),
            pred=pred_gmv,
            raw_pred_log=pred_log,
            train_cutoff=train_cutoff.isoformat(),
            feature_cutoff=val_feature_cutoff.isoformat(),
            target_max_abs_error=max_target_error,
        )
        log(
            f"OOF {tag}: n={len(fold_uid):,}, target audit max_abs={max_target_error:.3g}, "
            f"val features+predict={time.time() - t:.1f}s",
        )
        del models, df_val, target
        gc.collect()


def team_test(
    df_raw: pd.DataFrame,
    team_features: types.ModuleType,
    team_train: types.ModuleType,
    team_predict: types.ModuleType,
) -> None:
    out = RUN_DIR / "test_predictions.npz"
    if out.exists():
        log(f"reusing TEST predictions: {out.name}")
        return
    models, features = train_or_load(
        df_raw, FINAL_TEAM_TRAIN_CUTOFF, "production_clean", team_features, team_train,
    )
    t = time.time()
    log(f"building TEST features: history through {TEST_FEATURE_CUTOFF - dt.timedelta(days=1)}")
    df_test = team_features.build_features(df_raw, TEST_FEATURE_CUTOFF)
    pred_log = np.asarray(team_predict.predict_log(models, df_test, features), dtype=np.float64)
    pred_gmv = np.asarray(team_predict.predict_gmv(models, df_test, features), dtype=np.float64)
    if not np.isfinite(pred_gmv).all() or (pred_gmv < 0).any():
        raise RuntimeError("invalid teammate TEST predictions")
    np.savez_compressed(
        out,
        user_id=df_test["user_id"].to_numpy(),
        z=np.log1p(pred_gmv),
        pred=pred_gmv,
        raw_pred_log=pred_log,
        train_cutoff=FINAL_TEAM_TRAIN_CUTOFF.isoformat(),
        feature_cutoff=TEST_FEATURE_CUTOFF.isoformat(),
    )
    log(f"TEST predictions built in {time.time() - t:.1f}s")
    del models, df_test
    gc.collect()


def load_team_oof() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts = [np.load(RUN_DIR / f"oof_{v.strftime('%Y%m%d')}.npz") for v in VAL_FOLDS_S1]
    uid = np.concatenate([p["user_id"] for p in parts])
    cut = np.concatenate([p["cutoff"].astype("U10") for p in parts])
    y = np.concatenate([p["y"] for p in parts]).astype(np.float64)
    z = np.concatenate([p["z"] for p in parts]).astype(np.float64)
    return uid, cut, y, z


def align_team(
    our_uid: np.ndarray,
    our_cut: np.ndarray,
    our_y: np.ndarray,
    team_uid: np.ndarray,
    team_cut: np.ndarray,
    team_y: np.ndarray,
    team_z: np.ndarray,
) -> np.ndarray:
    our_key = np.char.add(our_cut.astype("U10"), our_uid.astype("U20"))
    team_key = np.char.add(team_cut.astype("U10"), team_uid.astype("U20"))
    team_order = np.argsort(team_key)
    if not np.array_equal(our_key, team_key[team_order]):
        raise RuntimeError("team OOF keys do not match canonical OOF keys")
    if not np.allclose(our_y, team_y[team_order], rtol=1e-7, atol=1e-6):
        raise RuntimeError("team and canonical OOF targets differ after alignment")
    return team_z[team_order]


def fold_scores(y: np.ndarray, z: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    return np.array([calibrate(y[m], z[m])[1] for m in masks], dtype=np.float64)


def optimise_weight(
    y: np.ndarray,
    pred_our: np.ndarray,
    pred_team: np.ndarray,
    masks: list[np.ndarray],
    fold_weights: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    weights = np.asarray(fold_weights, dtype=np.float64)
    weights = weights / weights.sum()

    def objective(w: float) -> float:
        pred = w * pred_our + (1.0 - w) * pred_team
        return float(weights @ fold_scores(y, np.log1p(pred), masks))

    coarse_w = np.linspace(0.0, 1.0, 101)
    coarse_score = np.array([objective(float(w)) for w in coarse_w])
    i = int(coarse_score.argmin())
    lo = float(coarse_w[max(0, i - 2)])
    hi = float(coarse_w[min(len(coarse_w) - 1, i + 2)])
    if lo == hi:
        best_w = lo
    else:
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
        best_w = float(result.x)
    candidates = [(0.0, objective(0.0)), (1.0, objective(1.0)), (best_w, objective(best_w))]
    best_w, best_score = min(candidates, key=lambda item: item[1])
    z_best = np.log1p(best_w * pred_our + (1.0 - best_w) * pred_team)
    return best_w, best_score, fold_scores(y, z_best, masks)


def analyse_and_submit(our_uid: np.ndarray, our_cut: np.ndarray, our_y: np.ndarray, z_our: np.ndarray) -> dict:
    team_uid, team_cut, team_y, team_z_raw = load_team_oof()
    z_team = align_team(our_uid, our_cut, our_y, team_uid, team_cut, team_y, team_z_raw)
    folds = [v.isoformat() for v in VAL_FOLDS_S1]
    masks = [our_cut == fold for fold in folds]
    fw = np.asarray(FOLD_WEIGHTS_S1, dtype=np.float64)
    fw = fw / fw.sum()

    our_fc = fold_scores(our_y, z_our, masks)
    team_fc = fold_scores(our_y, z_team, masks)
    our_wcv = float(fw @ our_fc)
    team_wcv = float(fw @ team_fc)
    pred_our = np.expm1(np.maximum(z_our, 0.0))
    pred_team = np.expm1(np.maximum(z_team, 0.0))
    best_w, blend_wcv, blend_fc = optimise_weight(our_y, pred_our, pred_team, masks, fw)

    lofo_weights: list[float] = []
    lofo_scores = np.zeros(len(folds), dtype=np.float64)
    for held in range(len(folds)):
        keep = [i for i in range(len(folds)) if i != held]
        w_held, _, _ = optimise_weight(
            our_y,
            pred_our,
            pred_team,
            [masks[i] for i in keep],
            fw[keep],
        )
        lofo_weights.append(w_held)
        z_held = np.log1p(w_held * pred_our[masks[held]] + (1.0 - w_held) * pred_team[masks[held]])
        lofo_scores[held] = calibrate(our_y[masks[held]], z_held)[1]
    lofo_wcv = float(fw @ lofo_scores)

    # Production regime audit in log space, aligned to the canonical submission.
    our_sub = pd.read_csv(SUBMISSIONS / "submission_STRONGEST_CURRENT.csv")
    team_test_npz = np.load(RUN_DIR / "test_predictions.npz")
    team_test_frame = pd.DataFrame({
        "user_id": team_test_npz["user_id"],
        "team_pred": team_test_npz["pred"],
    })
    test = our_sub.merge(team_test_frame, on="user_id", how="left", validate="one_to_one")
    if test["team_pred"].isna().any():
        raise RuntimeError("missing teammate TEST predictions after user alignment")
    z_our_test = np.log1p(test["predict"].to_numpy(dtype=np.float64))
    z_team_test = np.log1p(test["team_pred"].to_numpy(dtype=np.float64))
    var_diff_oof = float(np.var(z_our - z_team))
    var_diff_test = float(np.var(z_our_test - z_team_test))
    regime_ratio = var_diff_test / var_diff_oof

    # The OOF objective removes one global log-level per fold.  Apply the same
    # production policy to the raw test blend, using the frozen project level.
    raw_blend = best_w * test["predict"].to_numpy(dtype=np.float64) + (1.0 - best_w) * test[
        "team_pred"
    ].to_numpy(dtype=np.float64)
    raw_z = np.log1p(raw_blend)
    production_delta = PRODUCTION_LOG_LEVEL - float(raw_z.mean())
    final_pred = np.expm1(np.maximum(raw_z + production_delta, 0.0))

    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submit.csv")
    submission = pd.DataFrame({"user_id": test["user_id"], "predict": final_pred})
    submission = sample[["user_id"]].merge(submission, on="user_id", how="left", validate="one_to_one")
    if list(submission.columns) != ["user_id", "predict"]:
        raise RuntimeError("invalid submission columns")
    if len(submission) != 250_000 or submission["user_id"].nunique() != 250_000:
        raise RuntimeError("invalid submission row/user count")
    values = submission["predict"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("invalid final submission predictions")
    SUBMISSIONS.mkdir(exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False, float_format="%.6f")

    curve_rows = []
    for weight in np.linspace(0.0, 1.0, 101):
        z = np.log1p(weight * pred_our + (1.0 - weight) * pred_team)
        fc = fold_scores(our_y, z, masks)
        curve_rows.append({"our_weight": float(weight), "wcv": float(fw @ fc)})
    pd.DataFrame(curve_rows).to_csv(RESULT_DIR / "weight_curve.csv", index=False)

    metrics = {
        "experiment": "TEAM_B_B2_EXP069",
        "team_sha": TEAM_SHA,
        "seed": int(SEED),
        "folds": folds,
        "fold_weights": list(map(float, FOLD_WEIGHTS_S1)),
        "blend_space": "raw_gmv_then_per_fold_log_offset",
        "our": {"wcv": our_wcv, "fold_scores": our_fc.tolist()},
        "team": {"wcv": team_wcv, "fold_scores": team_fc.tolist()},
        "blend": {
            "our_weight": best_w,
            "team_weight": 1.0 - best_w,
            "wcv": blend_wcv,
            "fold_scores": blend_fc.tolist(),
            "delta_vs_our": blend_wcv - our_wcv,
            "lofo_weights": lofo_weights,
            "lofo_fold_scores": lofo_scores.tolist(),
            "lofo_wcv": lofo_wcv,
            "lofo_delta_vs_our": lofo_wcv - our_wcv,
        },
        "diversity": {
            "oof_log_prediction_correlation": float(np.corrcoef(z_our, z_team)[0, 1]),
            "oof_log_prediction_difference_variance": var_diff_oof,
            "test_log_prediction_correlation": float(np.corrcoef(z_our_test, z_team_test)[0, 1]),
            "test_log_prediction_difference_variance": var_diff_test,
            "test_oof_variance_ratio": regime_ratio,
        },
        "production": {
            "team_train_cutoff": FINAL_TEAM_TRAIN_CUTOFF.isoformat(),
            "team_feature_cutoff": TEST_FEATURE_CUTOFF.isoformat(),
            "target_log_level": PRODUCTION_LOG_LEVEL,
            "pre_level_blend_mean_log1p": float(raw_z.mean()),
            "global_log_delta": production_delta,
            "submission_mean_log1p": float(np.log1p(values).mean()),
            "submission_mean_predict": float(values.mean()),
            "submission_min": float(values.min()),
            "submission_max": float(values.max()),
            "submission_zero_fraction": float((values == 0).mean()),
            "submission_path": str(SUBMISSION_PATH.relative_to(ROOT)),
            "submission_sha256": hashlib.sha256(SUBMISSION_PATH.read_bytes()).hexdigest(),
        },
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(
        f"scores: our={our_wcv:.9f}, team={team_wcv:.9f}, w={best_w:.6f}, "
        f"blend={blend_wcv:.9f}, delta={blend_wcv - our_wcv:+.9f}",
    )
    log(
        f"LOFO={lofo_wcv:.9f} ({lofo_wcv - our_wcv:+.9f}), "
        f"regime test/oof={regime_ratio:.3f}, submission={SUBMISSION_PATH.name}",
    )
    return metrics


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    team_features, team_train, team_predict = load_team_modules()
    synthetic_leakage_audit(team_features)

    Z, our_y, our_cut = aligned(OUR_COMPONENTS)
    z_our = np.average(Z.astype(np.float64), axis=0, weights=OUR_WEIGHTS)
    # aligned() sorts by the same concatenated key; recover canonical user ids
    # from any component with the identical ordering.
    base = np.load(ARTIFACTS / f"oof_{OUR_COMPONENTS[0]}.npz")
    base_key = np.char.add(base["cutoff"].astype("U10"), base["user_id"].astype("U20"))
    order = np.argsort(base_key)
    our_uid = base["user_id"][order]
    if not np.array_equal(base["cutoff"].astype("U10")[order], our_cut):
        raise RuntimeError("canonical OOF cutoff recovery failed")

    expected_our = 1.747509862
    masks = [our_cut == v.isoformat() for v in VAL_FOLDS_S1]
    fw = np.asarray(FOLD_WEIGHTS_S1, dtype=np.float64)
    replay_our = float((fw / fw.sum()) @ fold_scores(our_y, z_our, masks))
    if abs(replay_our - expected_our) > 2e-6:
        raise RuntimeError(f"STRONGEST-CURRENT replay mismatch: {replay_our} vs {expected_our}")
    log(f"STRONGEST-CURRENT OOF replay PASS: {replay_our:.9f}")

    needed = [RUN_DIR / f"oof_{v.strftime('%Y%m%d')}.npz" for v in VAL_FOLDS_S1]
    needed.append(RUN_DIR / "test_predictions.npz")
    if not all(path.exists() for path in needed):
        df_raw = load_raw()
        team_oof(df_raw, our_y, our_uid, our_cut, team_features, team_train, team_predict)
        team_test(df_raw, team_features, team_train, team_predict)
        del df_raw
        gc.collect()
    analyse_and_submit(our_uid, our_cut, our_y, z_our)


if __name__ == "__main__":
    main()
