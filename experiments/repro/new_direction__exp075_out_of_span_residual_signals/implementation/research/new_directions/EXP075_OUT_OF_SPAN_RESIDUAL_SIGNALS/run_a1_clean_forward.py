from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
EXP = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
RAW = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw\train.parquet")
SAMPLE = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw\sample_submit.csv")
PANEL_PATH = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\processed\seq_panel_v1.npy")
GMV_PATH = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\processed\seq_gmv_v1.npy")
UID_PATH = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\processed\seq_uid_v1.npy")

DATA_START = dt.date(2025, 1, 1)
DATA_END = dt.date(2026, 2, 13)
N_DAYS = (DATA_END - DATA_START).days + 1
FOLDS = [dt.date(2025, 9, 4), dt.date(2025, 9, 18), dt.date(2025, 10, 2), dt.date(2025, 10, 16)]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
TRAIN_LAGS = [77, 63, 49, 35]
WINDOWS = [7, 14, 30, 60, 90, 180, 365]

PANEL_CHANNELS = [
    "present", "cat", "buy", "ponly", "searches", "search_to_cart",
    "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord",
    "gmv_search", "gmv_cat", "gmv",
]
RAW_CHANNELS = [
    "cat", "searches", "search_to_cart", "search_to_ord", "cat_to_cart",
    "cat_to_ord", "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv",
]
RAW_COLS = ["user_id", "event_date", *RAW_CHANNELS]
RAW_IDX = np.asarray([PANEL_CHANNELS.index(c) for c in RAW_CHANNELS], dtype=np.int64)
PRESENT_IDX = PANEL_CHANNELS.index("present")
BUY_IDX = PANEL_CHANNELS.index("buy")

SEED = 42
CHUNK = 16_000
T0 = time.time()


def log(*values: object) -> None:
    print(f"[{time.time() - T0:8.1f}s]", *values, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def day_index(value: dt.date) -> int:
    return (value - DATA_START).days


def stable_half(user_ids: np.ndarray) -> np.ndarray:
    x = np.asarray(user_ids, dtype=np.uint64)
    x = x ^ (x >> np.uint64(30))
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x *= np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(1)).astype(np.int8)


def lgb_params(kind: str) -> dict:
    common = dict(
        objective="regression_l2",
        metric="rmse",
        max_bin=63,
        bagging_fraction=0.8,
        bagging_freq=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        num_threads=max(1, min(10, (os.cpu_count() or 6) - 1)),
        seed=SEED,
        feature_fraction_seed=SEED,
        bagging_seed=SEED,
        data_random_seed=SEED,
    )
    if kind == "baseline":
        return dict(common, learning_rate=0.035, num_leaves=63, min_data_in_leaf=800,
                    feature_fraction=0.8, lambda_l2=20.0)
    return dict(common, learning_rate=0.03, num_leaves=31, min_data_in_leaf=1000,
                feature_fraction=0.65, lambda_l2=30.0)


def train_lgb(X: np.ndarray, y: np.ndarray, kind: str, rounds: int) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y.astype(np.float32, copy=False), free_raw_data=True)
    model = lgb.train(lgb_params(kind), dataset, num_boost_round=rounds)
    del dataset
    gc.collect()
    return model


class CleanData:
    def __init__(self) -> None:
        self.panel = np.load(PANEL_PATH, mmap_mode="r")
        self.gmv = np.load(GMV_PATH, mmap_mode="r")
        self.uid = np.load(UID_PATH, mmap_mode="r")
        assert self.panel.shape == (250_000, N_DAYS, len(PANEL_CHANNELS))
        assert self.gmv.shape == (250_000, N_DAYS)
        assert self.uid.shape == (250_000,)
        assert np.all(self.uid[1:] > self.uid[:-1])
        self._cutoff_cache: dict[dt.date, pd.DataFrame] = {}

    def rows(self, user_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(user_ids, dtype=np.int64)
        idx = np.searchsorted(self.uid, values)
        if idx.max(initial=0) >= len(self.uid) or not np.array_equal(self.uid[idx], values):
            raise AssertionError("Unknown or misaligned user_id")
        return idx.astype(np.int32)

    def raw_cutoff_frame(self, cutoff: dt.date) -> pd.DataFrame:
        if cutoff in self._cutoff_cache:
            return self._cutoff_cache[cutoff]
        feature_end = cutoff
        target_start = cutoff + dt.timedelta(days=1)
        target_end = cutoff + dt.timedelta(days=30)
        assert feature_end <= cutoff
        assert target_start > cutoff
        assert target_end <= DATA_END

        block_exprs = []
        for block in range(3):
            end = cutoff - dt.timedelta(days=30 * block)
            start = end - dt.timedelta(days=29)
            block_exprs.append(
                pl.col("event_date").is_between(start, end, closed="both").any().alias(f"b{block}")
            )
        eligible = (
            pl.scan_parquet(RAW)
            .filter(pl.col("event_date").is_between(cutoff - dt.timedelta(days=89), cutoff, closed="both"))
            .group_by("user_id")
            .agg(block_exprs)
            .filter(pl.all_horizontal([pl.col(f"b{i}") for i in range(3)]))
            .select("user_id")
            .collect()
            .sort("user_id")
        )
        future = (
            pl.scan_parquet(RAW)
            .filter(pl.col("event_date").is_between(target_start, target_end, closed="both"))
            .group_by("user_id")
            .agg([
                pl.col("gmv").sum().alias("target_y30"),
                pl.when(pl.col("event_date") <= cutoff + dt.timedelta(days=7))
                .then(pl.col("gmv")).otherwise(0.0).sum().alias("target_y7"),
                pl.when(pl.col("event_date") <= cutoff + dt.timedelta(days=14))
                .then(pl.col("gmv")).otherwise(0.0).sum().alias("target_y14"),
                (pl.col("gmv") > 0).sum().alias("target_purchase_days"),
            ])
            .collect()
        )
        frame = (
            eligible.join(future, on="user_id", how="left")
            .with_columns([
                pl.col("target_y30").fill_null(0.0),
                pl.col("target_y7").fill_null(0.0),
                pl.col("target_y14").fill_null(0.0),
                pl.col("target_purchase_days").fill_null(0),
            ])
            .sort("user_id")
            .to_pandas()
        )
        rows = self.rows(frame["user_id"].to_numpy())
        d = day_index(cutoff)
        mmap_target = self.gmv[rows, d + 1:d + 31].sum(axis=1)
        err = float(np.max(np.abs(mmap_target - frame["target_y30"].to_numpy())))
        if err > 1e-8:
            raise AssertionError(f"Raw/mmap target mismatch at {cutoff}: {err}")
        frame["cutoff"] = cutoff.isoformat()
        frame["target_log"] = np.log1p(frame["target_y30"].to_numpy(dtype=np.float64))
        self._cutoff_cache[cutoff] = frame
        log("raw cutoff", cutoff, "rows", len(frame), "target parity", err)
        return frame

    def audit_raw_parity(self) -> dict:
        # Full raw targets/panels are rebuilt above. This additional deterministic
        # audit verifies that the mmap used only for input slicing is a faithful
        # encoding of the raw 11-channel history.
        pick = self.uid[(np.arange(len(self.uid), dtype=np.uint64) * np.uint64(2654435761) % 997) < 16]
        pick = pick[:4096]
        rows = self.rows(pick)
        raw = (
            pl.scan_parquet(RAW)
            .filter(pl.col("user_id").is_in(pick.tolist()))
            .select(RAW_COLS)
            .collect()
            .sort(["user_id", "event_date"])
        )
        if raw.select(pl.struct(["user_id", "event_date"]).is_duplicated().sum()).item() != 0:
            raise AssertionError("Duplicate sampled raw user-day rows")
        expected = np.zeros((len(pick), N_DAYS, len(RAW_CHANNELS)), dtype=np.float16)
        expected_gmv = np.zeros((len(pick), N_DAYS), dtype=np.float64)
        u = raw["user_id"].to_numpy()
        ui = np.searchsorted(pick, u)
        di = (raw["event_date"].to_numpy() - np.datetime64(DATA_START)).astype("timedelta64[D]").astype(int)
        for j, name in enumerate(RAW_CHANNELS):
            values = raw[name].to_numpy()
            if name != "cat":
                # Exact historical builder arithmetic: cast raw values to fp32,
                # apply log1p in fp32, then store fp16. Using a float64 log here
                # creates harmless one-ulp fp16 differences and is not parity.
                values = np.log1p(values.astype(np.float32))
            else:
                values = values.astype(np.float32)
            expected[ui, di, j] = values.astype(np.float16)
        expected_gmv[ui, di] = raw["gmv"].to_numpy().astype(np.float64)
        actual = np.asarray(self.panel[rows][:, :, RAW_IDX])
        input_equal = bool(np.array_equal(expected, actual))
        gmv_err = float(np.max(np.abs(expected_gmv - np.asarray(self.gmv[rows]))))
        if not input_equal or gmv_err > 0.0:
            raise AssertionError(f"Raw mmap parity failed input_equal={input_equal} gmv_err={gmv_err}")
        return {
            "sample_users": int(len(pick)),
            "sample_raw_rows": int(raw.height),
            "input_fp16_bitwise_equal": input_equal,
            "gmv_max_abs_error": gmv_err,
            "raw_min_date": str(raw["event_date"].min()),
            "raw_max_date": str(raw["event_date"].max()),
            "raw_sha256": sha256(RAW),
            "panel_sha256": sha256(PANEL_PATH),
            "gmv_sha256": sha256(GMV_PATH),
            "uid_sha256": sha256(UID_PATH),
        }

    def _padded(self, rows: np.ndarray, cutoff: dt.date, history: int, channel_idx: np.ndarray) -> np.ndarray:
        d = day_index(cutoff)
        start = d - history + 1
        lo = max(0, start)
        out = np.zeros((len(rows), history, len(channel_idx)), dtype=np.float32)
        if lo <= d:
            offset = lo - start
            out[:, offset:, :] = np.asarray(self.panel[rows, lo:d + 1, :][:, :, channel_idx], dtype=np.float32)
        max_source_day = DATA_START + dt.timedelta(days=d)
        assert max_source_day <= cutoff
        return out

    def context_features(self, rows: np.ndarray, cutoff: dt.date) -> np.ndarray:
        n_features = len(WINDOWS) * len(RAW_CHANNELS) * 2 + len(RAW_CHANNELS) + 1
        out = np.empty((len(rows), n_features), dtype=np.float32)
        for start in range(0, len(rows), CHUNK):
            stop = min(start + CHUNK, len(rows))
            rr = rows[start:stop]
            seq = self._padded(rr, cutoff, 365, RAW_IDX)
            columns = []
            for window in WINDOWS:
                tail = seq[:, -window:, :]
                columns.extend([tail.sum(axis=1), (tail > 0).sum(axis=1, dtype=np.int32).astype(np.float32)])
            nz = seq > 0
            rec = np.full((len(rr), len(RAW_CHANNELS)), 366.0, dtype=np.float32)
            has = nz.any(axis=1)
            rev_arg = nz[:, ::-1, :].argmax(axis=1).astype(np.float32)
            rec[has] = rev_arg[has]
            available = np.full((len(rr), 1), min(day_index(cutoff) + 1, 365), dtype=np.float32)
            out[start:stop] = np.concatenate([*columns, rec, available], axis=1)
            del seq, nz, rec, columns
        return out

    def trajectory_features(self, rows: np.ndarray, cutoff: dt.date, history: int) -> np.ndarray:
        n_week = math.ceil(history / 7)
        trajectory_dim = (n_week + 28) * len(RAW_CHANNELS)
        out = np.empty((len(rows), trajectory_dim), dtype=np.float32)
        for start in range(0, len(rows), CHUNK):
            stop = min(start + CHUNK, len(rows))
            rr = rows[start:stop]
            seq = self._padded(rr, cutoff, history, RAW_IDX)
            pad = n_week * 7 - history
            if pad:
                seq_week = np.pad(seq, ((0, 0), (pad, 0), (0, 0)))
            else:
                seq_week = seq
            weekly = seq_week.reshape(len(rr), n_week, 7, len(RAW_CHANNELS)).sum(axis=2)
            daily = seq[:, -28:, :]
            traj = np.concatenate([weekly.reshape(len(rr), -1), daily.reshape(len(rr), -1)], axis=1)
            out[start:stop] = traj
            del seq, seq_week, weekly, daily, traj
        return out

    def candidate_features(
        self,
        rows: np.ndarray,
        cutoff: dt.date,
        history: int,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        traj = self.trajectory_features(rows, cutoff, history)
        ctx = self.context_features(rows, cutoff) if context is None else context
        if len(ctx) != len(traj):
            raise AssertionError("Context/trajectory row mismatch")
        out = np.empty((len(rows), traj.shape[1] + ctx.shape[1]), dtype=np.float32)
        out[:, :traj.shape[1]] = traj
        out[:, traj.shape[1]:] = ctx
        del traj
        return out


def concat_context(data: CleanData, frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = sum(len(f) for f in frames)
    dim = len(WINDOWS) * len(RAW_CHANNELS) * 2 + len(RAW_CHANNELS) + 1
    X = np.empty((total, dim), dtype=np.float32)
    y = np.empty(total, dtype=np.float64)
    uid = np.empty(total, dtype=np.int64)
    cutoff_index = np.empty(total, dtype=np.int16)
    pos = 0
    for ci, frame in enumerate(frames):
        n = len(frame)
        ids = frame["user_id"].to_numpy(dtype=np.int64)
        X[pos:pos + n] = data.context_features(data.rows(ids), dt.date.fromisoformat(frame["cutoff"].iloc[0]))
        y[pos:pos + n] = frame["target_log"].to_numpy(dtype=np.float64)
        uid[pos:pos + n] = ids
        cutoff_index[pos:pos + n] = ci
        pos += n
    return X, y, uid, cutoff_index


def concat_candidate(
    data: CleanData,
    frames: list[pd.DataFrame],
    history: int,
    context: np.ndarray | None = None,
) -> np.ndarray:
    total = sum(len(f) for f in frames)
    n_week = math.ceil(history / 7)
    context_dim = len(WINDOWS) * len(RAW_CHANNELS) * 2 + len(RAW_CHANNELS) + 1
    dim = (n_week + 28) * len(RAW_CHANNELS) + context_dim
    X = np.empty((total, dim), dtype=np.float32)
    pos = 0
    for frame in frames:
        n = len(frame)
        ids = frame["user_id"].to_numpy(dtype=np.int64)
        cutoff = dt.date.fromisoformat(frame["cutoff"].iloc[0])
        ctx = None if context is None else context[pos:pos + n]
        X[pos:pos + n] = data.candidate_features(data.rows(ids), cutoff, history, ctx)
        pos += n
    return X


def project_candidate(u: np.ndarray, baseline_z: np.ndarray) -> tuple[np.ndarray, dict]:
    u0 = np.asarray(u, dtype=np.float64) - float(np.mean(u))
    x = np.asarray(baseline_z, dtype=np.float64) - float(np.mean(baseline_z))
    denom = float(np.dot(x, x))
    beta = 0.0 if denom == 0 else float(np.dot(x, u0) / denom)
    perp = u0 - beta * x
    perp -= float(np.mean(perp))
    beta2 = 0.0 if denom == 0 else float(np.dot(x, perp) / denom)
    perp -= beta2 * x
    perp -= float(np.mean(perp))
    return perp, {
        "mean_removed": float(np.mean(u)),
        "baseline_beta_first": beta,
        "baseline_beta_second": beta2,
        "max_projection_after_second_pass": abs(float(np.dot(x, perp) / max(denom, 1e-300))),
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return 0.0 if denom == 0 else float(np.dot(x, y) / denom)


def fit_fold(data: CleanData, validation_cutoff: dt.date) -> tuple[pd.DataFrame, dict]:
    train_cutoffs = [validation_cutoff - dt.timedelta(days=lag) for lag in TRAIN_LAGS]
    assert all(t + dt.timedelta(days=30) <= validation_cutoff for t in train_cutoffs)
    assert validation_cutoff <= dt.date(2025, 10, 16)
    train_frames = [data.raw_cutoff_frame(t) for t in train_cutoffs]
    val_frame = data.raw_cutoff_frame(validation_cutoff).copy()
    log("fold", validation_cutoff, "building baseline context")
    Xb, y_train, uid_train, _ = concat_context(data, train_frames)
    halves = stable_half(uid_train)
    base_cf = np.empty(len(y_train), dtype=np.float64)
    for side in (0, 1):
        fit = halves != side
        pred = halves == side
        model = train_lgb(Xb[fit], y_train[fit], "baseline", 260)
        base_cf[pred] = model.predict(Xb[pred], num_iteration=model.best_iteration)
        del model
        gc.collect()
    base_offset = float(np.mean(y_train - base_cf))
    base_cf += base_offset
    residual_train = y_train - base_cf
    baseline_full = train_lgb(Xb, y_train, "baseline", 260)
    val_ids = val_frame["user_id"].to_numpy(dtype=np.int64)
    Xb_val = data.context_features(data.rows(val_ids), validation_cutoff)
    baseline_z = baseline_full.predict(Xb_val, num_iteration=baseline_full.best_iteration) + base_offset
    del baseline_full
    gc.collect()
    target_log = val_frame["target_log"].to_numpy(dtype=np.float64)
    val_frame["baseline_z"] = baseline_z
    val_frame["baseline_prediction"] = np.expm1(np.maximum(baseline_z, 0.0))
    val_frame["residual"] = target_log - baseline_z

    fold_meta = {
        "validation_cutoff": validation_cutoff.isoformat(),
        "train_cutoffs": [x.isoformat() for x in train_cutoffs],
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(val_frame)),
        "baseline_crossfit_offset": base_offset,
        "baseline_rmsle": float(np.sqrt(np.mean((target_log - baseline_z) ** 2))),
        "feature_source_max_date": validation_cutoff.isoformat(),
        "target_source_min_date": (validation_cutoff + dt.timedelta(days=1)).isoformat(),
        "target_source_max_date": (validation_cutoff + dt.timedelta(days=30)).isoformat(),
        "leakage_assertions": True,
    }

    for history in (180, 365):
        name = f"A1_TREE_TRAJ_{history}"
        log("fold", validation_cutoff, name, "building train matrix")
        Xa = concat_candidate(data, train_frames, history, Xb)
        model = train_lgb(Xa, residual_train, "candidate", 300)
        del Xa
        gc.collect()
        log("fold", validation_cutoff, name, "building validation matrix")
        Xa_val = data.candidate_features(data.rows(val_ids), validation_cutoff, history, Xb_val)
        u_raw = model.predict(Xa_val, num_iteration=model.best_iteration)
        del Xa_val, model
        gc.collect()
        u_perp, projection = project_candidate(u_raw, baseline_z)
        r = target_log - baseline_z
        b = float(np.mean(u_perp * r))
        G = float(np.mean(u_perp * u_perp))
        rho = correlation(u_perp, r)
        val_frame[f"u_raw_{history}"] = u_raw
        val_frame[f"u_perp_{history}"] = u_perp
        fold_meta[name] = {
            "rho": rho,
            "b": b,
            "G": G,
            "oracle_amplitude": b / G if G else 0.0,
            "oracle_MSE_gain": -(b * b / G) if G else 0.0,
            "u_raw_rms": float(np.sqrt(np.mean(u_raw * u_raw))),
            "u_perp_rms": float(np.sqrt(G)),
            "perp_fraction_vs_clean_baseline_span": float(np.sum(u_perp ** 2) / max(np.sum((u_raw - np.mean(u_raw)) ** 2), 1e-300)),
            "candidate_residual_vs_baseline_residual_corr": correlation(r - u_perp, r),
            "projection": projection,
        }
        log("fold", validation_cutoff, name, "rho", rho, "oracle amp", b / G if G else 0.0)
    del Xb, Xb_val
    gc.collect()
    return val_frame, fold_meta


def weighted_point(frame: pd.DataFrame, history: int) -> dict:
    fold_values = [d.isoformat() for d in FOLDS]
    amplitudes: dict[str, float] = {}
    past_u: list[np.ndarray] = []
    past_r: list[np.ndarray] = []
    past_w: list[np.ndarray] = []
    rows = []
    for fi, cutoff in enumerate(fold_values):
        part = frame[frame["cutoff"] == cutoff]
        u = part[f"u_perp_{history}"].to_numpy(dtype=np.float64)
        r = part["residual"].to_numpy(dtype=np.float64)
        if fi == 0:
            amplitude = 1.0
            source = "fixed_residual_objective"
        else:
            uu = np.concatenate(past_u)
            rr = np.concatenate(past_r)
            ww = np.concatenate(past_w)
            amplitude = float(np.sum(ww * uu * rr) / max(np.sum(ww * uu * uu), 1e-300))
            source = "strictly_earlier_heldout_folds"
        amplitudes[cutoff] = amplitude
        delta = (r - amplitude * u) ** 2 - r ** 2
        base_score = float(np.sqrt(np.mean(r ** 2)))
        corrected_score = float(np.sqrt(np.mean((r - amplitude * u) ** 2)))
        rows.append({
            "cutoff": cutoff,
            "rho": correlation(u, r),
            "b": float(np.mean(u * r)),
            "G": float(np.mean(u * u)),
            "deployable_amplitude": amplitude,
            "amplitude_source": source,
            "delta_MSE": float(np.mean(delta)),
            "baseline_RMSLE": base_score,
            "corrected_RMSLE": corrected_score,
            "delta_RMSLE": corrected_score - base_score,
        })
        frame.loc[part.index, f"amplitude_{history}"] = amplitude
        frame.loc[part.index, f"delta_mse_{history}"] = delta
        past_u.append(u)
        past_r.append(r)
        past_w.append(np.full(len(u), FOLD_WEIGHTS[fi] / len(u), dtype=np.float64))

    # Every fold contributes its preregistered fold weight independent of row count.
    u_all, r_all, w_all = [], [], []
    for fi, cutoff in enumerate(fold_values):
        part = frame[frame["cutoff"] == cutoff]
        u_all.append(part[f"u_perp_{history}"].to_numpy(dtype=np.float64))
        r_all.append(part["residual"].to_numpy(dtype=np.float64))
        w_all.append(np.full(len(part), FOLD_WEIGHTS[fi] / len(part), dtype=np.float64))
    u = np.concatenate(u_all)
    r = np.concatenate(r_all)
    w = np.concatenate(w_all)
    w /= w.sum()
    mu_u, mu_r = float(np.sum(w * u)), float(np.sum(w * r))
    cov = float(np.sum(w * (u - mu_u) * (r - mu_r)))
    var_u = float(np.sum(w * (u - mu_u) ** 2))
    var_r = float(np.sum(w * (r - mu_r) ** 2))
    weighted_rho = cov / math.sqrt(max(var_u * var_r, 1e-300))
    nested_delta_mse = float(sum(FOLD_WEIGHTS[i] * rows[i]["delta_MSE"] for i in range(4)) / FOLD_WEIGHTS.sum())
    nested_delta_rmsle = float(sum(FOLD_WEIGHTS[i] * rows[i]["delta_RMSLE"] for i in range(4)) / FOLD_WEIGHTS.sum())
    return {
        "history": history,
        "fold_rows": rows,
        "weighted_clean_forward_rho": weighted_rho,
        "latest_rho": rows[-1]["rho"],
        "nested_delta_MSE": nested_delta_mse,
        "nested_delta_RMSLE": nested_delta_rmsle,
        "amplitudes": amplitudes,
    }


def cluster_bootstrap(frame: pd.DataFrame, history: int, replicates: int = 1000) -> tuple[dict, np.ndarray, np.ndarray]:
    unique_uid, inv = np.unique(frame["user_id"].to_numpy(dtype=np.int64), return_inverse=True)
    n_users = len(unique_uid)
    fold_map = {d.isoformat(): i for i, d in enumerate(FOLDS)}
    fold_idx = np.asarray([fold_map[x] for x in frame["cutoff"]], dtype=np.int8)
    fold_counts = np.bincount(fold_idx, minlength=4).astype(np.float64)
    row_w = FOLD_WEIGHTS[fold_idx] / fold_counts[fold_idx]
    row_w /= row_w.sum()
    u = frame[f"u_perp_{history}"].to_numpy(dtype=np.float64)
    r = frame["residual"].to_numpy(dtype=np.float64)
    d = frame[f"delta_mse_{history}"].to_numpy(dtype=np.float64)
    stats = np.column_stack([row_w, row_w * u, row_w * r, row_w * u * u,
                             row_w * r * r, row_w * u * r, row_w * d])
    cluster = np.zeros((n_users, stats.shape[1]), dtype=np.float64)
    for j in range(stats.shape[1]):
        cluster[:, j] = np.bincount(inv, weights=stats[:, j], minlength=n_users)
    rng = np.random.default_rng(20260828 + history)
    rho_draws = np.empty(replicates, dtype=np.float64)
    delta_draws = np.empty(replicates, dtype=np.float64)
    batch = 20
    pos = 0
    while pos < replicates:
        m = min(batch, replicates - pos)
        counts = rng.poisson(1.0, size=(m, n_users)).astype(np.float64)
        sums = counts @ cluster
        sw = sums[:, 0]
        mu_u = sums[:, 1] / sw
        mu_r = sums[:, 2] / sw
        vu = sums[:, 3] / sw - mu_u ** 2
        vr = sums[:, 4] / sw - mu_r ** 2
        cv = sums[:, 5] / sw - mu_u * mu_r
        rho_draws[pos:pos + m] = cv / np.sqrt(np.maximum(vu * vr, 1e-300))
        delta_draws[pos:pos + m] = sums[:, 6] / sw
        pos += m
    point = weighted_point(frame.copy(), history)["weighted_clean_forward_rho"]
    se = float(np.std(rho_draws, ddof=1))
    result = {
        "method": "Poisson user-cluster bootstrap; all rows for a user share one resampling count",
        "replicates": replicates,
        "unique_users": n_users,
        "rho_point": point,
        "rho_ci_2_5": float(np.quantile(rho_draws, 0.025)),
        "rho_ci_97_5": float(np.quantile(rho_draws, 0.975)),
        "rho_bootstrap_se": se,
        "t_rho": point / se if se else float("inf"),
        "delta_MSE_ci_2_5": float(np.quantile(delta_draws, 0.025)),
        "delta_MSE_ci_97_5": float(np.quantile(delta_draws, 0.975)),
        "P_delta_MSE_lt_0": float(np.mean(delta_draws < 0)),
    }
    return result, rho_draws, delta_draws


def verdict(summary: dict, bootstrap: dict) -> str:
    rhos = [row["rho"] for row in summary["fold_rows"]]
    if summary["weighted_clean_forward_rho"] < 0.015:
        return "REJECT_RHO_LT_0.015"
    if summary["latest_rho"] <= 0:
        return "REJECT_LATEST_NONPOSITIVE"
    if min(rhos) < 0 < max(rhos):
        return "REJECT_SIGN_INSTABILITY"
    strong = (summary["weighted_clean_forward_rho"] >= 0.025
              and summary["latest_rho"] >= 0.020
              and bootstrap["t_rho"] >= 3.0
              and bootstrap["P_delta_MSE_lt_0"] >= 0.95)
    if strong:
        return "STRONG_SIGNAL"
    if summary["weighted_clean_forward_rho"] >= 0.020 and summary["nested_delta_MSE"] < 0:
        return "PROMISING"
    return "WEAK_SIGNAL"


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=False) if not EXP.exists() else None
    data = CleanData()
    audit = data.audit_raw_parity()
    audit.update({
        "raw_global_min_date": DATA_START.isoformat(),
        "raw_global_max_date": DATA_END.isoformat(),
        "clean_corridor_end": FOLDS[-1].isoformat(),
        "excluded_target_based_cutoffs_after": FOLDS[-1].isoformat(),
        "reason_excluded": "Historical panel contamination begins when target overlaps the organizer's final eligibility window.",
    })
    (EXP / "validation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log("raw parity audit PASS")

    all_frames = []
    fold_meta = []
    for fold in FOLDS:
        frame, meta = fit_fold(data, fold)
        all_frames.append(frame)
        fold_meta.append(meta)
        pd.DataFrame([meta]).to_json(EXP / f"fold_{fold.strftime('%Y%m%d')}_meta.json", orient="records", indent=2)
    oof = pd.concat(all_frames, ignore_index=True)
    summaries = {}
    boot_arrays = {}
    metric_rows = []
    for history in (180, 365):
        summary = weighted_point(oof, history)
        bootstrap, rho_draws, delta_draws = cluster_bootstrap(oof, history, 1000)
        summary["bootstrap"] = bootstrap
        summary["verdict"] = verdict(summary, bootstrap)
        summaries[f"A1_TREE_TRAJ_{history}"] = summary
        boot_arrays[f"rho_{history}"] = rho_draws
        boot_arrays[f"delta_mse_{history}"] = delta_draws
        for row in summary["fold_rows"]:
            metric_rows.append({"experiment": f"A1_TREE_TRAJ_{history}", **row})
        log("summary", history, summary["weighted_clean_forward_rho"], summary["latest_rho"], summary["verdict"])

    keep = [
        "user_id", "cutoff", "target_y30", "target_y7", "target_y14",
        "target_purchase_days", "target_log", "baseline_prediction", "baseline_z",
        "residual", "u_raw_180", "u_perp_180", "amplitude_180", "delta_mse_180",
        "u_raw_365", "u_perp_365", "amplitude_365", "delta_mse_365",
    ]
    oof[keep].to_parquet(EXP / "clean_forward_predictions.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(EXP / "fold_metrics.csv", index=False)
    (EXP / "rho_analysis.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    np.savez_compressed(EXP / "bootstrap_draws.npz", **boot_arrays)
    runtime = {
        "wall_seconds": time.time() - T0,
        "cpu_threads": lgb_params("candidate")["num_threads"],
        "gpu_used": False,
        "folds": [x.isoformat() for x in FOLDS],
        "completed": True,
    }
    (EXP / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    checks = []
    for path in sorted(EXP.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checks.append(f"{sha256(path)}  {path.name}")
    (EXP / "checksums.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
    log("A1 clean forward complete", runtime)


if __name__ == "__main__":
    main()
