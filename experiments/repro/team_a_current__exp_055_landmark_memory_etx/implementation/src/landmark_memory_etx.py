"""EXP-055: cutoff-safe retrospective landmark outcome memory.

The experiment is deliberately audit-gated.  The default command reconstructs
STRONGEST_CURRENT exactly, materialises REAL and matched SHUFFLED landmark
tokens for the four OOF folds, runs the fixed EXP-053 residual probe on CPU and
stops unless every preregistered pre-flight condition passes.

Run:
    python src/landmark_memory_etx.py
    python src/landmark_memory_etx.py --analysis-only

No test inference, submission or public-LB path exists in this module.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import residual_signal_discovery as exp053
from src import seq
from src.btyd_day_bgnbd import user_group
from src.config import DATA_START, FOLD_WEIGHTS_S1, SEED
from src.det_pair import optimizer_hash, state_dict_hash
from src.validation import calibrate


EXPERIMENT_ID = 55
PREFIX = "LANDMARK_MEMORY_EXP055"
BASE_HEAD = "a28a71fb2d0194052014c542f36d180dfe74bcf9"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = np.asarray(FOLD_WEIGHTS_S1, dtype=float)
EXPECTED_FOLD_SCORES = np.asarray(
    [1.766883357, 1.760509577, 1.748629224, 1.741278566], dtype=float
)
EXPECTED_WCV = 1.747509863

LAGS = np.asarray([30 + 15 * k for k in range(16)], dtype=np.int16)
N_LANDMARKS = 16
N_STATE = 18
N_NUMERIC = 20
QUERY_POS = 16
TOKEN_PAD = 0
TOKEN_LANDMARK = 1
TOKEN_QUERY = 2

ARTIFACT_DIR = ROOT / "artifacts" / PREFIX
RESULT_DIR = ROOT / "research" / "strategies" / "results" / PREFIX
REAL_TOKENS = ARTIFACT_DIR / "real_tokens.npy"
VALID_MASK = ARTIFACT_DIR / "valid_mask.npy"
TOKEN_TYPE = ARTIFACT_DIR / "token_type.npy"
OUTCOME_AVAILABLE = ARTIFACT_DIR / "outcome_available.npy"
PAST30_GMV = ARTIFACT_DIR / "landmark_past30_gmv.npy"
SHUF_TOKENS = {
    0: ARTIFACT_DIR / "shuffled_tokens_0to1.npy",
    1: ARTIFACT_DIR / "shuffled_tokens_1to0.npy",
}
SHUF_PERM = {
    0: ARTIFACT_DIR / "shuffle_permutation_0to1.npy",
    1: ARTIFACT_DIR / "shuffle_permutation_1to0.npy",
}

MEMORY_NAMES = (
    "last_safe_outcome",
    "mean_safe_outcome",
    "median_safe_outcome",
    "trend_safe_outcomes",
    "state_similarity_weighted_outcome",
    "nearest_state_outcome",
    "max_state_similarity",
    "number_valid_landmarks",
)

PROBE_PARAMS = dict(exp053.PROBE_PARAMS)
SCALES = np.asarray([0.0, 0.25, 0.50, 1.0], dtype=float)
FIXED_REPLACEMENT_WEIGHTS = {
    "CAP": 0.10, "UNC": 0.20, "DIST": 0.25,
    "LANDMARK_REAL": 0.225, "SEQ_AVG3": 0.225,
}
FIXED_COAUTHOR_WEIGHTS = {
    "CAP": 0.10, "UNC": 0.20, "DIST": 0.25,
    "ETX_AVG3": 0.1125, "LANDMARK_REAL": 0.1125, "SEQ_AVG3": 0.225,
}
FORBIDDEN_ACTIONS = {
    "test_inference": False,
    "submission": False,
    "public_lb": False,
    "full_folds": False,
}

NEURAL_CFG = {
    "d_model": 128, "blocks": 5, "heads": 8, "head_dim": 16,
    "ffn": 384, "dropout": 0.10, "epochs": 4, "lr": 1.5e-3,
    "wd": 1e-2, "warmup": 500, "batch": 512,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False, lineterminator="\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.view(np.uint8))
    return h.hexdigest()


def current_git_audit() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True,
                                       stderr=subprocess.STDOUT).strip()
    return {
        "head": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_porcelain_v1": run("git", "status", "--porcelain=v1", "--branch"),
        "required_base_head": BASE_HEAD,
    }


def landmark_schedule(T: dt.date) -> list[dict[str, Any]]:
    rows = []
    for lag in LAGS.tolist():
        t = T - dt.timedelta(days=int(lag))
        state_lo = t - dt.timedelta(days=30)
        outcome_hi = t + dt.timedelta(days=30)
        valid = state_lo >= DATA_START and outcome_hi <= T
        rows.append({
            "query_cutoff": T.isoformat(), "lag": int(lag),
            "landmark": t.isoformat(),
            "state_window": f"({state_lo.isoformat()},{t.isoformat()}]",
            "outcome_window": f"({t.isoformat()},{outcome_hi.isoformat()}]",
            "state_start_exclusive": state_lo.isoformat(),
            "state_end_inclusive": t.isoformat(),
            "outcome_start_exclusive": t.isoformat(),
            "outcome_end_inclusive": outcome_hi.isoformat(),
            "safe_outcome": bool(outcome_hi <= T), "valid": bool(valid),
        })
    return rows


def state_bounds(query_day: int, lag: int) -> tuple[int, int]:
    """Included integer days for (t-30,t]: [t-29, t+1)."""
    t = int(query_day) - int(lag)
    return t - 29, t + 1


def outcome_bounds(query_day: int, lag: int) -> tuple[int, int]:
    """Included integer days for (t,t+30]: [t+1, t+31)."""
    t = int(query_day) - int(lag)
    return t + 1, t + 31


def landmark_valid(query_day: int, lag: int) -> bool:
    t = int(query_day) - int(lag)
    return (t - 30 >= 0) and (t + 30 <= int(query_day))


def _last_recency(row: np.ndarray, channel: int, day: int) -> np.ndarray:
    seen = row[:, :day + 1, channel] > 0
    rev = seen[:, ::-1]
    any_seen = seen.any(axis=1)
    out = np.where(any_seen, rev.argmax(axis=1), day + 1)
    return out.astype(np.float32)


def build_tokens_from_arrays(panel: np.ndarray, gmv: np.ndarray, query_day: int,
                             lags: np.ndarray = LAGS,
                             scale: np.ndarray | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference token builder used by mutation/window tests.

    ``panel`` is (users, days, 14), ``gmv`` is raw GMV (users, days).  Numeric
    tokens are REAL and follow the raw-cache contract: 18 transformed state
    fields, log1p(lag), raw historical z30.  Query outcome is always zero.
    """
    n, _, c = panel.shape
    if c != 14:
        raise ValueError("expected the 14 stored ETX/SEQ behavioural channels")
    sc = np.ones(14, np.float32) if scale is None else np.asarray(scale, np.float32)
    out = np.zeros((n, len(lags) + 1, N_NUMERIC), np.float32)
    valid = np.zeros((n, len(lags)), bool)
    past30_gmv = np.zeros((n, len(lags)), np.float32)
    for j, lag0 in enumerate(np.asarray(lags, int)):
        lag = int(lag0)
        if not landmark_valid(query_day, lag):
            continue
        t = query_day - lag
        a, b = state_bounds(query_day, lag)
        oa, ob = outcome_bounds(query_day, lag)
        win = panel[:, a:b].astype(np.float32)
        sums = win.sum(axis=1)
        out[:, j, :14] = sums * sc
        out[:, j, 14] = np.log1p(_last_recency(panel, 0, t))
        out[:, j, 15] = np.log1p(_last_recency(panel, 2, t))
        out[:, j, 16] = np.log1p(sums[:, 0])
        out[:, j, 17] = np.log1p(sums[:, 2])
        out[:, j, 18] = np.log1p(float(lag))
        out[:, j, 19] = np.log1p(gmv[:, oa:ob].sum(axis=1))
        past30_gmv[:, j] = gmv[:, a:b].sum(axis=1).astype(np.float32)
        valid[:, j] = True

    qa, qb = query_day - 29, query_day + 1
    if qa < 0:
        raise ValueError("query state needs the full (T-30,T] window")
    qwin = panel[:, qa:qb].astype(np.float32)
    qs = qwin.sum(axis=1)
    out[:, -1, :14] = qs * sc
    out[:, -1, 14] = np.log1p(_last_recency(panel, 0, query_day))
    out[:, -1, 15] = np.log1p(_last_recency(panel, 2, query_day))
    out[:, -1, 16] = np.log1p(qs[:, 0])
    out[:, -1, 17] = np.log1p(qs[:, 2])
    out[:, -1, 18:] = 0.0
    return out, valid, past30_gmv, gmv[:, qa:qb].sum(axis=1).astype(np.float32)


def _needed_recency_days() -> set[int]:
    days: set[int] = set()
    for fold in FOLDS:
        q = seq.day_index(dt.date.fromisoformat(fold))
        days.add(q)
        for lag in LAGS:
            if landmark_valid(q, int(lag)):
                days.add(q - int(lag))
    return days


def _recency_snapshots(panel: np.ndarray, days: set[int]) -> tuple[dict[int, np.ndarray],
                                                                   dict[int, np.ndarray]]:
    last_any = np.full(panel.shape[0], -1, np.int16)
    last_buy = np.full(panel.shape[0], -1, np.int16)
    any_snap: dict[int, np.ndarray] = {}
    buy_snap: dict[int, np.ndarray] = {}
    for day in range(max(days) + 1):
        last_any[panel[:, day, 0] > 0] = day
        last_buy[panel[:, day, 2] > 0] = day
        if day in days:
            any_snap[day] = last_any.copy()
            buy_snap[day] = last_buy.copy()
    return any_snap, buy_snap


def _production_state(panel: np.ndarray, gmv: np.ndarray, rows: np.ndarray, t: int,
                      scale: np.ndarray, any_snap: dict[int, np.ndarray],
                      buy_snap: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a, b = t - 29, t + 1
    win = panel[rows, a:b].astype(np.float32)
    sums = win.sum(axis=1)
    state = np.empty((len(rows), N_STATE), np.float32)
    state[:, :14] = sums * scale
    la = any_snap[t][rows].astype(np.int32)
    lb = buy_snap[t][rows].astype(np.int32)
    state[:, 14] = np.log1p(np.where(la >= 0, t - la, t + 1))
    state[:, 15] = np.log1p(np.where(lb >= 0, t - lb, t + 1))
    state[:, 16] = np.log1p(sums[:, 0])
    state[:, 17] = np.log1p(sums[:, 2])
    past_gmv = gmv[rows, a:b].sum(axis=1).astype(np.float32)
    return state, past_gmv


def _open_tmp(path: Path, dtype: Any, shape: tuple[int, ...]) -> tuple[np.memmap, Path]:
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    if tmp.exists():
        tmp.unlink()
    return np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape), tmp


def _finish_tmp(mm: np.memmap, tmp: Path, path: Path) -> None:
    mm.flush()
    # Windows keeps the file locked while the mmap handle is alive; deleting a
    # local reference is insufficient because the caller still owns one.
    if getattr(mm, "_mmap", None) is not None:
        mm._mmap.close()
    os.replace(tmp, path)


def materialize_real_tokens(frame: dict[str, np.ndarray], force: bool = False) -> dict[str, Any]:
    required = (REAL_TOKENS, VALID_MASK, TOKEN_TYPE, OUTCOME_AVAILABLE, PAST30_GMV)
    manifest_path = ARTIFACT_DIR / "real_token_cache_manifest.json"
    if not force and manifest_path.exists() and all(path.exists() for path in required):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    panel, gmv, _, scale = seq.panel()
    n = len(frame["y"])
    real, real_tmp = _open_tmp(REAL_TOKENS, np.float16, (n, N_LANDMARKS + 1, N_NUMERIC))
    valid, valid_tmp = _open_tmp(VALID_MASK, np.bool_, (n, N_LANDMARKS))
    types, types_tmp = _open_tmp(TOKEN_TYPE, np.uint8, (n, N_LANDMARKS + 1))
    avail, avail_tmp = _open_tmp(OUTCOME_AVAILABLE, np.bool_, (n, N_LANDMARKS + 1))
    pgmv, pgmv_tmp = _open_tmp(PAST30_GMV, np.float32, (n, N_LANDMARKS))
    real[:] = 0
    valid[:] = False
    types[:] = TOKEN_PAD
    avail[:] = False
    pgmv[:] = 0

    any_snap, buy_snap = _recency_snapshots(panel, _needed_recency_days())
    schedule_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        mask = frame["cutoff"] == fold
        pos = np.flatnonzero(mask)
        rows = seq.user_rows(frame["user_id"][mask])
        q = seq.day_index(dt.date.fromisoformat(fold))
        for j, lag0 in enumerate(LAGS):
            lag = int(lag0)
            t = q - lag
            audit = landmark_schedule(dt.date.fromisoformat(fold))[j]
            schedule_rows.append(audit)
            if not landmark_valid(q, lag):
                continue
            st, past = _production_state(panel, gmv, rows, t, scale, any_snap, buy_snap)
            oa, ob = outcome_bounds(q, lag)
            hist = np.log1p(gmv[rows, oa:ob].sum(axis=1)).astype(np.float32)
            real[pos, j, :18] = st.astype(np.float16)
            real[pos, j, 18] = np.float16(np.log1p(float(lag)))
            real[pos, j, 19] = hist.astype(np.float16)
            valid[pos, j] = True
            types[pos, j] = TOKEN_LANDMARK
            avail[pos, j] = True
            pgmv[pos, j] = past

        qst, _ = _production_state(panel, gmv, rows, q, scale, any_snap, buy_snap)
        real[pos, QUERY_POS, :18] = qst.astype(np.float16)
        real[pos, QUERY_POS, 18:] = 0
        types[pos, QUERY_POS] = TOKEN_QUERY
        avail[pos, QUERY_POS] = False

    _finish_tmp(real, real_tmp, REAL_TOKENS)
    _finish_tmp(valid, valid_tmp, VALID_MASK)
    _finish_tmp(types, types_tmp, TOKEN_TYPE)
    _finish_tmp(avail, avail_tmp, OUTCOME_AVAILABLE)
    _finish_tmp(pgmv, pgmv_tmp, PAST30_GMV)

    real_r = np.load(REAL_TOKENS, mmap_mode="r")
    valid_r = np.load(VALID_MASK, mmap_mode="r")
    types_r = np.load(TOKEN_TYPE, mmap_mode="r")
    avail_r = np.load(OUTCOME_AVAILABLE, mmap_mode="r")
    if np.any(real_r[:, QUERY_POS, 19] != 0) or np.any(avail_r[:, QUERY_POS]):
        raise AssertionError("query outcome is not zero/masked")
    if not np.array_equal(types_r[:, :N_LANDMARKS] == TOKEN_LANDMARK, valid_r):
        raise AssertionError("PAD/token mask mismatch")
    if not np.array_equal(avail_r[:, :N_LANDMARKS], valid_r):
        raise AssertionError("outcome availability mismatch")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "cache_semantics": "REAL raw fold tokens; historical z is standardized later with donor-only statistics",
        "shape": list(real_r.shape), "dtype": str(real_r.dtype),
        "numeric_fields": list(seq.CHANNELS) + [
            "log1p_rec_any", "log1p_rec_buy", "log1p_days_present_30",
            "log1p_days_buy_30", "log1p_lag_to_query", "historical_z30_raw",
        ],
        "token_types": {"PAD": TOKEN_PAD, "LANDMARK": TOKEN_LANDMARK, "QUERY": TOKEN_QUERY},
        "lags": LAGS, "query_position": QUERY_POS,
        "paths": {name: path.resolve() for name, path in {
            "real_tokens": REAL_TOKENS, "valid_mask": VALID_MASK,
            "token_type": TOKEN_TYPE, "outcome_available": OUTCOME_AVAILABLE,
            "landmark_past30_gmv": PAST30_GMV,
        }.items()},
        "hashes": {path.name: file_sha256(path) for path in required},
        "row_key_sha256": array_sha256(frame["cutoff"], frame["user_id"]),
        "query_outcome_zero": True, "query_outcome_masked": True,
        "valid_counts_by_fold": {
            fold: int(valid_r[frame["cutoff"] == fold][0].sum()) for fold in FOLDS
        },
        "safe_window_audit": schedule_rows,
    }
    write_json(manifest_path, manifest)
    write_json(RESULT_DIR / "landmark_schedule_config.json", {
        "lags": LAGS, "stride_days": 15, "n_landmarks": 16,
        "state_window": "(t-30,t]", "outcome_window": "(t,t+30]",
        "validity": "t+30<=T and t-30>=DATA_START", "schedule": schedule_rows,
    })
    write_json(RESULT_DIR / "safe_window_audit.json", {
        "all_outcomes_end_by_query": all(row["safe_outcome"] for row in schedule_rows),
        "current_target_used": False, "rows": schedule_rows,
    })
    return manifest


def decile_edges(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, np.float64)
    if not len(v):
        raise ValueError("empty donor values for deciles")
    return np.quantile(v, np.arange(1, 10) / 10.0)


def bins_from_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), np.asarray(values), side="right").astype(np.int8)


def materialize_shuffle(values: np.ndarray, strata: np.ndarray, seed: int = SEED
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Stable permutation inside integer strata; returned indices are materialized."""
    v = np.asarray(values)
    s = np.asarray(strata)
    out = v.copy()
    perm = np.arange(len(v), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for key in np.unique(s):
        idx = np.flatnonzero(s == key)
        if len(idx) > 1:
            src = idx[rng.permutation(len(idx))]
            out[idx] = v[src]
            perm[idx] = src
    return out, perm


def shuffle_preserves_multiset(values: np.ndarray, shuffled: np.ndarray,
                               strata: np.ndarray) -> bool:
    for key in np.unique(strata):
        idx = strata == key
        if not np.array_equal(np.sort(values[idx]), np.sort(shuffled[idx])):
            return False
    return True


def materialize_shuffled_tokens(frame: dict[str, np.ndarray], donor_side: int,
                                force: bool = False) -> dict[str, Any]:
    path, perm_path = SHUF_TOKENS[donor_side], SHUF_PERM[donor_side]
    manifest_path = ARTIFACT_DIR / f"shuffle_manifest_{donor_side}to{1-donor_side}.json"
    if not force and path.exists() and perm_path.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    real = np.load(REAL_TOKENS, mmap_mode="r")
    valid = np.load(VALID_MASK, mmap_mode="r")
    pgmv = np.load(PAST30_GMV, mmap_mode="r")
    fi = np.asarray([FOLDS.index(v) for v in frame["cutoff"]], np.int8)
    side = user_group(frame["user_id"])
    donor = (fi < 3) & (side == donor_side)
    thresholds: list[np.ndarray] = []
    for j in range(N_LANDMARKS):
        thresholds.append(decile_edges(np.asarray(pgmv[donor, j])))

    shuf, shuf_tmp = _open_tmp(path, np.float16, real.shape)
    perm, perm_tmp = _open_tmp(perm_path, np.int32, (len(frame["y"]), N_LANDMARKS))
    for lo in range(0, len(frame["y"]), 20_000):
        hi = min(lo + 20_000, len(frame["y"]))
        shuf[lo:hi] = real[lo:hi]
        perm[lo:hi] = np.arange(lo, hi, dtype=np.int32)[:, None]

    rng = np.random.default_rng(SEED)
    audit_rows = []
    for fold_i, fold in enumerate(FOLDS):
        for j in range(N_LANDMARKS):
            if not valid[frame["cutoff"] == fold, j].any():
                continue
            for current_side in (0, 1):
                scope = (fi == fold_i) & (side == current_side) & valid[:, j]
                bins = bins_from_edges(np.asarray(pgmv[scope, j]), thresholds[j])
                scoped_idx = np.flatnonzero(scope)
                for decile in range(10):
                    idx = scoped_idx[bins == decile]
                    if len(idx) == 0:
                        continue
                    src = idx[rng.permutation(len(idx))]
                    before = np.asarray(real[idx, j, 19])
                    shuf[idx, j, 19] = np.asarray(real[src, j, 19])
                    perm[idx, j] = src.astype(np.int32)
                    after = np.asarray(shuf[idx, j, 19])
                    if not np.array_equal(np.sort(before), np.sort(after)):
                        raise AssertionError("shuffle changed a stratum outcome multiset")
                    audit_rows.append({
                        "query_cutoff": fold, "lag": int(LAGS[j]),
                        "user_side": current_side, "decile": decile, "n": len(idx),
                        "multiset_preserved": True,
                    })
    if np.any(np.asarray(shuf[:, QUERY_POS, 19]) != 0):
        raise AssertionError("shuffle changed query outcome")
    if not np.array_equal(np.asarray(shuf[:, :, :19]), np.asarray(real[:, :, :19])):
        raise AssertionError("REAL/SHUF state tokens differ")
    _finish_tmp(shuf, shuf_tmp, path)
    _finish_tmp(perm, perm_tmp, perm_path)
    manifest = {
        "direction": f"{donor_side}->{1-donor_side}", "seed": int(SEED),
        "decile_fit": "early-fold donor-side only, pooled by landmark lag",
        "permutation_strata": ["query_cutoff", "landmark_lag", "past30_gmv_decile", "user_side"],
        "thresholds_by_lag": {str(int(lag)): thresholds[j] for j, lag in enumerate(LAGS)},
        "state_tokens_identical_except_outcome": True,
        "query_outcome_zero_masked": True,
        "outcome_multiset_preserved_every_stratum": True,
        "changed_fraction": float(np.mean(
            np.asarray(np.load(path, mmap_mode="r")[:, :N_LANDMARKS, 19])
            != np.asarray(real[:, :N_LANDMARKS, 19]))),
        "paths": {"tokens": path.resolve(), "permutation": perm_path.resolve()},
        "hashes": {path.name: file_sha256(path), perm_path.name: file_sha256(perm_path)},
        "audit_rows": audit_rows,
    }
    write_json(manifest_path, manifest)
    return manifest


def donor_state_stats(tokens: np.ndarray, valid: np.ndarray,
                      donor_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(N_STATE, np.float64)
    sq = np.zeros(N_STATE, np.float64)
    count = 0
    idx = np.flatnonzero(donor_mask)
    for lo in range(0, len(idx), 10_000):
        part = idx[lo:lo + 10_000]
        x = np.asarray(tokens[part, :N_LANDMARKS, :N_STATE], np.float32)
        m = np.asarray(valid[part])
        flat = x[m]
        total += flat.sum(axis=0, dtype=np.float64)
        sq += np.square(flat, dtype=np.float64).sum(axis=0)
        count += len(flat)
    mean = total / count
    var = np.maximum(sq / count - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), count


def memory_summaries(tokens: np.ndarray, valid: np.ndarray, donor_mask: np.ndarray,
                     chunk: int = 10_000) -> tuple[np.ndarray, dict[str, Any]]:
    mean, std, n_fit = donor_state_stats(tokens, valid, donor_mask)
    out = np.zeros((tokens.shape[0], len(MEMORY_NAMES)), np.float32)
    lag_x = -LAGS.astype(np.float32)  # positive slope = outcome rises toward query
    for lo in range(0, len(out), chunk):
        hi = min(lo + chunk, len(out))
        lm = np.asarray(tokens[lo:hi, :N_LANDMARKS, :N_STATE], np.float32)
        q = np.asarray(tokens[lo:hi, QUERY_POS, :N_STATE], np.float32)
        z = np.asarray(tokens[lo:hi, :N_LANDMARKS, 19], np.float32)
        m = np.asarray(valid[lo:hi], bool)
        lmz = (lm - mean) / std
        qz = (q - mean) / std
        num = np.einsum("nkd,nd->nk", lmz, qz)
        den = np.linalg.norm(lmz, axis=2) * np.linalg.norm(qz, axis=1)[:, None]
        sim = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        sim[~m] = -np.inf
        count = m.sum(axis=1)
        safe_count = np.maximum(count, 1)
        zz = np.where(m, z, 0.0)
        out[lo:hi, 0] = np.where(m[:, 0], z[:, 0], 0.0)
        out[lo:hi, 1] = zz.sum(axis=1) / safe_count
        med = np.zeros(hi - lo, np.float32)
        trend = np.zeros(hi - lo, np.float32)
        for i in range(hi - lo):
            zi = z[i, m[i]]
            xi = lag_x[m[i]]
            if len(zi):
                med[i] = np.median(zi)
            if len(zi) >= 2:
                xc = xi - xi.mean()
                trend[i] = float(np.dot(xc, zi - zi.mean()) / max(np.dot(xc, xc), 1e-12))
        out[lo:hi, 2] = med
        out[lo:hi, 3] = trend
        weights = np.where(m, np.maximum(sim + 1.0, 0.0), 0.0)
        wsum = weights.sum(axis=1)
        out[lo:hi, 4] = np.divide((weights * z).sum(axis=1), wsum,
                                  out=out[lo:hi, 1].copy(), where=wsum > 1e-12)
        nearest = np.argmax(sim, axis=1)
        out[lo:hi, 5] = z[np.arange(hi - lo), nearest]
        out[lo:hi, 6] = np.max(sim, axis=1)
        out[lo:hi, 6][count == 0] = 0.0
        out[lo:hi, 7] = count.astype(np.float32)
    return out, {
        "donor_state_mean": mean, "donor_state_std": std,
        "n_valid_landmark_states_fit": n_fit,
        "similarity": "cosine after donor-only z-standardization",
        "similarity_weight": "max(cosine+1,0), normalized per row",
        "trend": "OLS z30 slope on -lag_days; positive means rising toward query",
    }


def _lgb_params() -> dict[str, Any]:
    params = {k: v for k, v in PROBE_PARAMS.items()
              if k not in {"num_boost_round", "early_stopping"}}
    params.update(objective="regression_l1", metric="l1", verbosity=-1,
                  num_threads=min(12, os.cpu_count() or 1))
    return params


def fit_probe(X: np.ndarray, target: np.ndarray, mask: np.ndarray) -> lgb.Booster:
    xx = np.asarray(X[mask], np.float32)
    yy = np.asarray(target[mask], np.float32)
    ds = lgb.Dataset(xx, label=yy, free_raw_data=True)
    return lgb.train(_lgb_params(), ds, num_boost_round=int(PROBE_PARAMS["num_boost_round"]))


def _score_scope(y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    offset, score = calibrate(y[mask], z[mask])
    return float(score), float(offset)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    if len(aa) < 2 or np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _select_scale(y: np.ndarray, z_base: np.ndarray, correction: np.ndarray,
                  fi: np.ndarray, donor: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for scale in SCALES:
        scores = []
        for fold in range(3):
            mask = donor & (fi == fold)
            scores.append(calibrate(y[mask], z_base[mask] + scale * correction[mask])[1])
        value = float(np.average(scores, weights=FOLD_WEIGHTS[:3]))
        rows.append({"scale": float(scale), "selection_wcv_3f": value,
                     "fold_scores": scores})
    best = min(row["selection_wcv_3f"] for row in rows)
    selected = min(row["scale"] for row in rows if row["selection_wcv_3f"] <= best + 1e-5)
    return selected, rows


def run_probe_variant(X: np.ndarray, frame: dict[str, np.ndarray], donor_side: int,
                      variant: str) -> dict[str, Any]:
    fi = frame["fold_index"]
    side = user_group(frame["user_id"])
    donor = (fi < 3) & (side == donor_side)
    recipient = (fi == 3) & (side == 1 - donor_side)
    if np.intersect1d(frame["user_id"][donor], frame["user_id"][recipient]).size:
        raise AssertionError("donor/recipient user halves overlap")
    centered = frame["r_strong"].copy()
    for fold in range(3):
        m = donor & (fi == fold)
        centered[m] -= centered[m].mean()

    donor_oof = np.full(len(centered), np.nan, np.float32)
    for held in range(3):
        train = donor & (fi != held)
        valid = donor & (fi == held)
        model = fit_probe(X, centered, train)
        donor_oof[valid] = model.predict(np.asarray(X[valid], np.float32)).astype(np.float32)
    if np.isnan(donor_oof[donor]).any():
        raise AssertionError("donor OOF correction incomplete")
    lo, hi = np.quantile(donor_oof[donor], [0.01, 0.99])
    processed = np.zeros(len(centered), np.float32)
    processed[donor] = np.clip(donor_oof[donor], lo, hi)
    for fold in range(3):
        m = donor & (fi == fold)
        processed[m] -= processed[m].mean()
    selected, curve = _select_scale(frame["y"], frame["z_strong_raw"], processed, fi, donor)

    model = fit_probe(X, centered, donor)
    pred = model.predict(np.asarray(X[recipient], np.float32)).astype(np.float32)
    pred = np.clip(pred, lo, hi)
    pred -= pred.mean()
    correction = np.zeros(len(centered), np.float32)
    correction[recipient] = selected * pred
    base_score, base_offset = _score_scope(frame["y"], frame["z_strong_raw"], recipient)
    cand_score, cand_offset = _score_scope(
        frame["y"], frame["z_strong_raw"] + correction, recipient)
    return {
        "variant": variant, "direction": f"{donor_side}->{1-donor_side}",
        "donor_side": donor_side, "recipient_side": 1 - donor_side,
        "n_donor": int(donor.sum()), "n_recipient": int(recipient.sum()),
        "recipient_mask": recipient, "correction": correction,
        "selected_scale": selected, "scale_curve": curve,
        "winsor_p01": float(lo), "winsor_p99": float(hi),
        "base_score": base_score, "candidate_score": cand_score,
        "delta": cand_score - base_score,
        "base_offset": base_offset, "candidate_offset": cand_offset,
        "residual_alignment": _safe_corr(pred, frame["r_strong"][recipient]),
        "recipient_labels_in_training": False, "recipient_user_overlap": 0,
    }


def _finite_state_matrix(state: Any) -> np.ndarray:
    x = state.to_numpy().astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return x


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(X, float)
    med = np.nanmedian(xx, axis=0)
    xx = np.where(np.isfinite(xx), xx, med)
    design = np.column_stack([np.ones(len(xx)), xx])
    return np.linalg.lstsq(design, np.asarray(y, float), rcond=None)[0], med


def _ols_predict(X: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    xx = np.asarray(X, float)
    coef, med = fit
    xx = np.where(np.isfinite(xx), xx, med)
    return np.column_stack([np.ones(len(xx)), xx]) @ coef


def control_axes(state: Any, state_names: list[str]) -> np.ndarray:
    wanted = ["w30_gmv", "w90_gmv", "w180_gmv", "rec_buy"]
    idx = [state_names.index(name) for name in wanted]
    raw = state.select([state_names[i] for i in idx]).to_numpy().astype(np.float64)
    return np.log1p(np.maximum(raw, 0.0))


def diagnostic_correlations(frame: dict[str, np.ndarray], state: Any,
                            state_names: list[str], summaries: dict[int, np.ndarray]
                            ) -> list[dict[str, Any]]:
    fi, side = frame["fold_index"], user_group(frame["user_id"])
    axes = control_axes(state, state_names)
    ly = np.log1p(frame["y"])
    rows = []
    for donor_side, memory in summaries.items():
        donor = (fi < 3) & (side == donor_side)
        recipient = (fi == 3) & (side == 1 - donor_side)
        nearest, last = memory[:, 5], memory[:, 0]
        cy = _ols_fit(axes[donor], ly[donor])
        cn = _ols_fit(axes[donor], nearest[donor])
        cl = _ols_fit(axes[donor], last[donor])
        y_partial = ly[recipient] - _ols_predict(axes[recipient], cy)
        n_innov = nearest[recipient] - _ols_predict(axes[recipient], cn)
        l_innov = last[recipient] - _ols_predict(axes[recipient], cl)
        rows.append({
            "direction": f"{donor_side}->{1-donor_side}",
            "corr_target_nearest": _safe_corr(ly[recipient], nearest[recipient]),
            "corr_target_last": _safe_corr(ly[recipient], last[recipient]),
            "corr_residual_nearest_innovation": _safe_corr(frame["r_strong"][recipient], n_innov),
            "corr_residual_last_innovation": _safe_corr(frame["r_strong"][recipient], l_innov),
            "partial_corr_target_nearest": _safe_corr(y_partial, n_innov),
            "partial_corr_target_last": _safe_corr(y_partial, l_innov),
            "recipient_n": int(recipient.sum()),
        })
    return rows


def pooled_late_score(frame: dict[str, np.ndarray], results: list[dict[str, Any]],
                      variant: str) -> dict[str, Any]:
    late = frame["fold_index"] == 3
    correction = np.zeros(len(frame["y"]), np.float32)
    chosen = [row for row in results if row["variant"] == variant]
    for row in chosen:
        correction += row["correction"]
    base_score, _ = _score_scope(frame["y"], frame["z_strong_raw"], late)
    score, _ = _score_scope(frame["y"], frame["z_strong_raw"] + correction, late)
    return {"variant": variant, "base_score": base_score, "score": score,
            "delta": score - base_score, "correction": correction,
            "selected_scales": [row["selected_scale"] for row in chosen],
            "half_deltas": [row["delta"] for row in chosen],
            "half_alignments": [row["residual_alignment"] for row in chosen]}


def build_baseline() -> tuple[dict[str, np.ndarray], Any, list[str], np.ndarray,
                              list[str], dict[str, Any]]:
    frame, manifest = exp053._load_core()
    exp053._audit_baseline(frame, manifest)
    # ``dist_p0``/``dist_p_act`` are two of the exact 34 disagreement columns
    # in EXP-053 COMBINED.  Load their already-saved, aligned OOF artifact; no
    # optional candidate prediction or test object is needed here.
    exp053._load_pact(frame, manifest)
    state, state_names = exp053._load_state_features(frame, manifest)
    disagreement, disagreement_names = exp053.build_disagreement_features(frame)
    manifest["git_audit"] = current_git_audit()
    manifest["base_head_matches_required"] = manifest["git_audit"]["head"] == BASE_HEAD
    manifest["exact_manifests"] = {
        "ETX_AVG3_report": ROOT / "artifacts" / "report_ETX-AVG3.json",
        "STRONGEST_CURRENT_report": ROOT / "research" / "strategies" / "results" / "ETX2"
                                   / "REPORT_STRONGEST_CURRENT.md",
        "EXP053_input_manifest": ROOT / "research" / "strategies" / "results"
                                / "RESIDUAL_SIGNAL_DISCOVERY" / "input_manifest.json",
    }
    manifest["source_correlation_60d"] = {
        "value": 0.4980,
        "fact_id": "N9",
        "primary_project_sources": [
            ROOT / "research" / "strategy_NN_report.md",
            ROOT / "research" / "strategy_NN_1.md",
            ROOT / "experiments" / "exp_024_multihorizon_hazard.md",
        ],
        "meaning": "raw same-user correlation of z30 targets at a 60-day cutoff shift",
        "incremental_utility_evidence": False,
    }
    for name, path in manifest["exact_manifests"].items():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing exact manifest {name}: {path}")
    write_json(ARTIFACT_DIR / "baseline_manifest.json", manifest)
    return frame, state, state_names, disagreement, disagreement_names, manifest


def build_memory_artifacts(frame: dict[str, np.ndarray], force: bool = False
                           ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict[str, Any]]]:
    real_tokens = np.load(REAL_TOKENS, mmap_mode="r")
    valid = np.load(VALID_MASK, mmap_mode="r")
    fi = frame["fold_index"]
    side = user_group(frame["user_id"])
    real_summary: dict[int, np.ndarray] = {}
    shuf_summary: dict[int, np.ndarray] = {}
    rows = []
    for donor_side in (0, 1):
        donor = (fi < 3) & (side == donor_side)
        real, meta_r = memory_summaries(real_tokens, valid, donor)
        shuf_tokens = np.load(SHUF_TOKENS[donor_side], mmap_mode="r")
        shuf, meta_s = memory_summaries(shuf_tokens, valid, donor)
        real_summary[donor_side] = real
        shuf_summary[donor_side] = shuf
        np.savez_compressed(ARTIFACT_DIR / f"memory_summaries_{donor_side}to{1-donor_side}.npz",
                            user_id=frame["user_id"], cutoff=frame["cutoff"],
                            names=np.asarray(MEMORY_NAMES, dtype="U"), real=real, shuffled=shuf)
        rows.append({"direction": f"{donor_side}->{1-donor_side}",
                     "real_meta": meta_r, "shuffled_meta": meta_s,
                     "real_sha256": array_sha256(real), "shuffled_sha256": array_sha256(shuf)})
    write_json(RESULT_DIR / "memory_summary_manifest.json", rows)
    return real_summary, shuf_summary, rows


def run_preflight(frame: dict[str, np.ndarray], state: Any, state_names: list[str],
                  disagreement: np.ndarray, disagreement_names: list[str],
                  real_summary: dict[int, np.ndarray], shuf_summary: dict[int, np.ndarray]
                  ) -> dict[str, Any]:
    state_x = _finite_state_matrix(state)
    base_x = np.column_stack([state_x, disagreement]).astype(np.float32)
    if base_x.shape[1] != 261:
        raise AssertionError(f"EXP-053 COMBINED must have 261 fields, got {base_x.shape[1]}")
    exp053.assert_no_future_feature_columns(state_names + disagreement_names)
    runs: list[dict[str, Any]] = []
    for donor_side in (0, 1):
        matrices = {
            "BASE": base_x,
            "REAL": np.column_stack([base_x, real_summary[donor_side]]).astype(np.float32),
            "SHUFFLED": np.column_stack([base_x, shuf_summary[donor_side]]).astype(np.float32),
            "NEAREST_ONLY": np.column_stack([base_x, real_summary[donor_side][:, 5]]).astype(np.float32),
            "LAST_ONLY": np.column_stack([base_x, real_summary[donor_side][:, 0]]).astype(np.float32),
        }
        if matrices["REAL"].shape != matrices["SHUFFLED"].shape:
            raise AssertionError("REAL/SHUFFLED probe capacity differs")
        for name, matrix in matrices.items():
            runs.append(run_probe_variant(matrix, frame, donor_side, name))

    pooled = {name: pooled_late_score(frame, runs, name)
              for name in ("BASE", "REAL", "SHUFFLED", "NEAREST_ONLY", "LAST_ONLY")}
    corr_rows = diagnostic_correlations(frame, state, state_names, real_summary)
    real_minus_shuffled = pooled["REAL"]["score"] - pooled["SHUFFLED"]["score"]
    partial_pooled = float(np.average(
        [row["corr_residual_nearest_innovation"] for row in corr_rows],
        weights=[row["recipient_n"] for row in corr_rows]))
    gates = {
        "real_late_delta_le_m0005": pooled["REAL"]["delta"] <= -0.0005,
        "real_minus_shuffled_le_m0004": real_minus_shuffled <= -0.0004,
        "both_recipient_halves_better": all(v < 0 for v in pooled["REAL"]["half_deltas"]),
        "residual_alignment_positive_both": all(v > 0 for v in pooled["REAL"]["half_alignments"]),
        "nearest_better_than_last_control": (
            pooled["NEAREST_ONLY"]["delta"] < pooled["LAST_ONLY"]["delta"]
            and all(row["corr_residual_nearest_innovation"]
                    > row["corr_residual_last_innovation"] for row in corr_rows)
        ),
        "partial_residual_corr_ge_002": partial_pooled >= 0.02,
    }
    verdict = "PASS" if all(gates.values()) else "NO_GO_PREFLIGHT"

    result_rows = []
    for row in runs:
        result_rows.append({k: v for k, v in row.items()
                            if k not in {"recipient_mask", "correction", "scale_curve"}})
    write_csv(RESULT_DIR / "preflight_direction_metrics.csv", result_rows)
    write_csv(RESULT_DIR / "memory_correlation_diagnostics.csv", corr_rows)
    write_csv(RESULT_DIR / "preflight_pooled_metrics.csv", [
        {k: v for k, v in row.items() if k != "correction"} for row in pooled.values()
    ])
    late = frame["fold_index"] == 3
    np.savez_compressed(
        ARTIFACT_DIR / "cpu_preflight_predictions.npz",
        user_id=frame["user_id"][late], y=frame["y"][late].astype(np.float32),
        side=user_group(frame["user_id"])[late],
        z_strong=frame["z_strong_raw"][late].astype(np.float32),
        **{f"correction_{name.lower()}": row["correction"][late].astype(np.float32)
           for name, row in pooled.items()},
    )
    payload = {
        "verdict": verdict, "promote_to_gpu": verdict == "PASS",
        "pooled": {name: {k: v for k, v in row.items() if k != "correction"}
                   for name, row in pooled.items()},
        "real_minus_shuffled": real_minus_shuffled,
        "partial_residual_correlation_pooled": partial_pooled,
        "gates": gates, "correlation_diagnostics": corr_rows,
        "probe_config": PROBE_PARAMS, "scales": SCALES,
        "base_feature_set": "exact EXP-053 COMBINED: 227 state + 34 disagreement",
        "target": "signed fold-calibrated STRONGEST_CURRENT residual",
        "forbidden_actions": FORBIDDEN_ACTIONS,
    }
    write_json(RESULT_DIR / "preflight_verdict.json", payload)
    return payload


def build_landmark_model(cfg: dict[str, Any] | None = None):
    """Paired-pilot architecture contract; instantiated on CPU by focused tests.

    It keeps the ETX dimensions and pre-LN causal attention backbone.  Landmark
    tokens add only the preregistered token-type and outcome-availability
    embeddings.  No auxiliary or control head exists.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    c = dict(NEURAL_CFG)
    if cfg:
        c.update(cfg)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d, h, dh, ffn = c["d_model"], c["heads"], c["head_dim"], c["ffn"]
            self.h, self.dh, self.scale = h, dh, dh ** -0.5
            self.n1 = nn.LayerNorm(d)
            self.q = nn.Linear(d, h * (dh - 1), bias=False)
            self.k = nn.Linear(d, h * (dh - 1), bias=False)
            self.v = nn.Linear(d, h * dh, bias=False)
            self.o = nn.Linear(h * dh, d, bias=False)
            tau = np.geomspace(4.0, 512.0, h)
            m0 = 64.0 / (self.scale * torch.as_tensor(tau, dtype=torch.float32))
            self.log_m = nn.Parameter(m0.log())
            self.n2 = nn.LayerNorm(d)
            self.w_in = nn.Linear(d, 2 * ffn)
            self.w_out = nn.Linear(ffn, d)
            self.drop = nn.Dropout(c["dropout"])
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.w_out.weight)
            nn.init.zeros_(self.w_out.bias)

        def forward(self, h, lag, valid):
            B, L, _ = h.shape
            x = self.n1(h)
            q = self.q(x).view(B, L, self.h, self.dh - 1).transpose(1, 2)
            k = self.k(x).view(B, L, self.h, self.dh - 1).transpose(1, 2)
            v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
            m = self.log_m.exp().view(1, self.h, 1, 1).expand(B, self.h, L, 1)
            age = (-lag / 64.0).view(B, 1, L, 1).expand(B, self.h, L, 1)
            causal = torch.ones(L, L, dtype=torch.bool, device=h.device).tril()
            key_ok = valid[:, None, None, :].expand(B, self.h, L, L)
            attn = causal.view(1, 1, L, L) & key_ok
            y = F.scaled_dot_product_attention(
                torch.cat([q, m], -1), torch.cat([k, age], -1), v,
                attn_mask=attn, scale=self.scale)
            h = h + self.drop(self.o(y.transpose(1, 2).reshape(B, L, self.h * self.dh)))
            g, u = self.w_in(self.n2(h)).chunk(2, -1)
            return h + self.drop(self.w_out(F.gelu(g) * u))

    class LandmarkETX(nn.Module):
        def __init__(self):
            super().__init__()
            d = c["d_model"]
            self.numeric = nn.Linear(N_NUMERIC, d)
            self.token_type = nn.Embedding(3, d)
            self.outcome_available = nn.Embedding(2, d)
            self.blocks = nn.ModuleList([Block() for _ in range(c["blocks"])])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(c["dropout"]),
                                      nn.Linear(d, 1))
            nn.init.zeros_(self.head[-1].weight)

        def forward(self, numeric, token_type, outcome_available, lag, valid):
            h = (self.numeric(numeric) + self.token_type(token_type)
                 + self.outcome_available(outcome_available.long()))
            for block in self.blocks:
                h = block(h, lag, valid)
            return self.head(self.norm(h[:, -1])).squeeze(-1)

    return LandmarkETX()


def build_neural_optimizer(model: Any, cfg: dict[str, Any] | None = None):
    import torch
    c = dict(NEURAL_CFG)
    if cfg:
        c.update(cfg)
    named = dict(model.named_parameters())
    decay = [p for n, p in named.items() if p.dim() > 1]
    nodecay = [p for n, p in named.items() if p.dim() <= 1 and "log_m" not in n]
    tau = [p for n, p in named.items() if "log_m" in n]
    return torch.optim.AdamW([
        dict(params=decay, weight_decay=c["wd"]),
        dict(params=nodecay, weight_decay=0.0),
        dict(params=tau, weight_decay=0.0, lr_mult=10.0),
    ], lr=c["lr"], betas=(0.9, 0.98))


def materialized_batch_plan(n_rows: int, epochs: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n_rows).astype(np.int32) for _ in range(epochs)])


def learning_rate_plan(total: int, cfg: dict[str, Any] | None = None) -> np.ndarray:
    c = dict(NEURAL_CFG)
    if cfg:
        c.update(cfg)
    step = np.arange(total, dtype=np.float64)
    return c["lr"] * (np.minimum(1.0, (step + 1.0) / c["warmup"])
                      * 0.5 * (1.0 + np.cos(np.pi * np.minimum(1.0, step / total))))


def paired_contract_hashes(n_rows: int = 1000) -> dict[str, Any]:
    import torch
    arms = []
    for _ in ("REAL", "SHUFFLED"):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = build_landmark_model()
        opt = build_neural_optimizer(model)
        arms.append({"model": state_dict_hash(model.state_dict()),
                     "optimizer": optimizer_hash(model, opt)})
    plan_a = materialized_batch_plan(n_rows, NEURAL_CFG["epochs"], SEED)
    plan_b = materialized_batch_plan(n_rows, NEURAL_CFG["epochs"], SEED)
    lr_a = learning_rate_plan(math.ceil(n_rows / NEURAL_CFG["batch"]) * NEURAL_CFG["epochs"])
    lr_b = learning_rate_plan(math.ceil(n_rows / NEURAL_CFG["batch"]) * NEURAL_CFG["epochs"])
    return {
        "arms": arms,
        "same_initial_model": arms[0]["model"] == arms[1]["model"],
        "same_initial_optimizer": arms[0]["optimizer"] == arms[1]["optimizer"],
        "batch_plan_sha256": [array_sha256(plan_a), array_sha256(plan_b)],
        "lr_plan_sha256": [array_sha256(lr_a), array_sha256(lr_b)],
        "same_batch_plan": np.array_equal(plan_a, plan_b),
        "same_lr_plan": np.array_equal(lr_a, lr_b),
        "dropout_rng_policy": "separate deterministic processes, torch/numpy seed reset to config.SEED",
        "architecture": NEURAL_CFG,
    }


def canonical_replay_hashes() -> dict[str, str]:
    paths = [
        ARTIFACT_DIR / "baseline_manifest.json",
        ARTIFACT_DIR / "real_token_cache_manifest.json",
        ARTIFACT_DIR / "cpu_preflight_predictions.npz",
        RESULT_DIR / "preflight_verdict.json",
        RESULT_DIR / "preflight_direction_metrics.csv",
        RESULT_DIR / "memory_correlation_diagnostics.csv",
    ]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def analysis_only() -> None:
    stored_path = RESULT_DIR / "reproducibility.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    current = canonical_replay_hashes()
    if current != stored["canonical_hashes"]:
        raise AssertionError(f"analysis-only replay hash mismatch: {current}")
    print("analysis-only replay PASS")


def write_summary(preflight: dict[str, Any], started: float) -> None:
    # This is a CPU-only construction/hash audit, not pilot training.  Persist
    # it even on NO_GO so the preregistered paired plan remains inspectable;
    # model fitting and CUDA stay prohibited unless the pre-flight passes.
    contract = paired_contract_hashes()
    write_json(ARTIFACT_DIR / "paired_neural_contract.json", contract)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "name": "RETROSPECTIVE LANDMARK OUTCOME MEMORY",
        "base_head": current_git_audit()["head"],
        "baseline_fold_2025_10_16": EXPECTED_FOLD_SCORES[-1],
        "baseline_wcv": EXPECTED_WCV,
        "preflight": preflight,
        "pilot_started": False,
        "pilot_status": "PROHIBITED_BY_PREFLIGHT" if preflight["verdict"] != "PASS"
                        else "READY_FOR_GPU",
        "promote_to_full_folds": "NO",
        "test_inference": False, "submission": False, "public_lb": False,
        "paired_contract": contract,
        "runtime_s": time.time() - started,
    }
    write_json(RESULT_DIR / "summary.json", summary)


def main(force: bool = False) -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame, state, state_names, disagreement, disagreement_names, _ = build_baseline()
    materialize_real_tokens(frame, force=force)
    for donor_side in (0, 1):
        materialize_shuffled_tokens(frame, donor_side, force=force)
    print("dataset built", flush=True)
    real_summary, shuf_summary, _ = build_memory_artifacts(frame, force=force)
    preflight = run_preflight(frame, state, state_names, disagreement, disagreement_names,
                              real_summary, shuf_summary)
    write_summary(preflight, started)
    write_json(RESULT_DIR / "reproducibility.json", {
        "canonical_hashes": canonical_replay_hashes(), "status": "BASELINE",
    })
    print("preflight complete", flush=True)
    if preflight["verdict"] == "PASS":
        print("READY_FOR_GPU", flush=True)
    else:
        print("VERDICT = NO_GO_PREFLIGHT", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.analysis_only:
        analysis_only()
    else:
        main(force=args.force)
