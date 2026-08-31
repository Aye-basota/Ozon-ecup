"""EXP-054: BURST-GAP-ETX structural representation audit.

The default command runs the exact baseline audit, builds cutoff-safe episode
summaries for the four S1 folds, and executes the fixed REAL-vs-SHUFFLED CPU
pre-flight.  Neural training is gated and is never reached after a failed
pre-flight.  There is deliberately no test-inference or submission command.

Run:
    python src/burst_gap_etx.py
    python src/burst_gap_etx.py audit
    python src/burst_gap_etx.py build
    python src/burst_gap_etx.py preflight
    python src/burst_gap_etx.py analysis-only
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
import polars as pl
from numba import njit, prange
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src import etx, residual_signal_discovery as exp053, seq
from src.btyd_day_bgnbd import user_group
from src.config import FOLD_WEIGHTS_S1, SEED
from src.validation import calibrate


EXPERIMENT_ID = 54
PREFIX = "BURST_GAP_EXP054"
FOLDS = tuple(exp053.FOLDS)
FOLD_WEIGHTS = np.asarray(FOLD_WEIGHTS_S1, float)
BASE_HEAD = "a28a71fb2d0194052014c542f36d180dfe74bcf9"

OUT_ARTIFACTS = ROOT / "artifacts" / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
CACHE = OUT_ARTIFACTS / "episode_cache.parquet"
INITIAL_STATUS = OUT_ARTIFACTS / "initial_git_status.txt"

BURST_THRESHOLD_DAYS = 3
MAX_HISTORY_TOKENS = 191
KEEP_RECENT_TOKENS = 190
N_TOKEN_NUMERIC = 22
LOG365 = math.log1p(365.0)

PAD, SUMMARY, BURST, GAP, QUERY = 0, 1, 2, 3, 4
TOKEN_TYPE_NAMES = {PAD: "PAD", SUMMARY: "SUMMARY", BURST: "BURST", GAP: "GAP", QUERY: "QUERY"}

EPISODE_NAMES = [
    "n_bursts", "median_burst_span", "mean_burst_span", "max_burst_span",
    "last_burst_span", "mean_burst_active_days", "last_burst_active_days",
    "last_burst_purchase_days", "last_burst_gmv", "last_burst_slope",
    "last_burst_intensity", "historical_burst_intensity",
    "last_to_historical_intensity", "current_open_gap", "median_closed_gap",
    "mean_closed_gap", "max_closed_gap", "current_gap_ratio",
    "reactivation_count", "burst_gap_token_count", "overflow", "singleton_burst_share",
    "last_burst_search_intensity", "last_burst_cart_intensity",
    "last_burst_order_intensity", "last_burst_peak_gmv",
    "previous_burst_purchase_presence", "mean_burst_gmv", "max_burst_gmv",
    "event_day_count", "gap_count", "available_depth",
]

PROBE_EPISODE_NAMES = [
    "n_bursts", "median_burst_span", "last_burst_span", "last_burst_active_days",
    "last_burst_purchase_days", "last_burst_gmv", "last_burst_slope",
    "current_open_gap", "median_closed_gap", "current_gap_ratio",
    "reactivation_count", "last_to_historical_intensity",
]

FIXED_ENSEMBLE_WEIGHTS = {
    "cap": 0.10, "unc": 0.20, "dist": 0.25, "etx": 0.225, "seq": 0.225,
}
REPLACEMENT_WEIGHTS = {
    "cap": 0.10, "unc": 0.20, "dist": 0.25, "burst": 0.225, "seq": 0.225,
}
COAUTHOR_WEIGHTS = {
    "cap": 0.10, "unc": 0.20, "dist": 0.25,
    "etx": 0.1125, "burst": 0.1125, "seq": 0.225,
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
    if isinstance(value, (tuple, list)):
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
    return exp053.array_sha256(*arrays)


def segmentation_config() -> dict[str, Any]:
    return {
        "event_day": "daily-log row exists",
        "burst_threshold_calendar_days": BURST_THRESHOLD_DAYS,
        "internal_gap_length": "next_event_day - previous_event_day - 1",
        "final_open_gap_length": "cutoff_day - last_event_day",
        "no_event_history": "one GAP spanning available history",
        "token_types": TOKEN_TYPE_NAMES,
        "numeric_dimension": N_TOKEN_NUMERIC,
        "history_token_cap": MAX_HISTORY_TOKENS,
        "overflow_policy": "SUMMARY of prefix + latest 190 tokens",
        "behavior_aggregation": "natural-domain sum -> log1p -> existing ETX inverse-RMS scale",
        "day_fields": "log1p(days)/log1p(365), fixed and validation-independent",
        "slope": "calendar-day least squares; signed log1p then existing GMV scale",
        "zero_gmv_search_share": 0.5,
        "seed": int(SEED),
    }


# --------------------------------------------------------------------------- pure segmentation/tokenizer
def segment_days(event_days: np.ndarray, cutoff_day: int, available_depth: int) -> list[dict[str, Any]]:
    """Fixed BURST/GAP segmentation. Future rows are ignored by construction."""
    days = np.asarray(event_days, np.int64)
    lo = cutoff_day - int(available_depth) + 1
    valid = days[(days >= lo) & (days <= cutoff_day)]
    valid = np.unique(valid)
    if not len(valid):
        return [{"type": GAP, "start": lo, "end": cutoff_day,
                 "length": int(available_depth), "event_days": np.empty(0, np.int64)}]
    out: list[dict[str, Any]] = []
    start = 0
    gap_before = 0
    for i in range(1, len(valid)):
        diff = int(valid[i] - valid[i - 1])
        if diff <= BURST_THRESHOLD_DAYS:
            continue
        burst_days = valid[start:i]
        out.append({"type": BURST, "start": int(burst_days[0]), "end": int(burst_days[-1]),
                    "length": int(burst_days[-1] - burst_days[0] + 1),
                    "gap_before": gap_before, "event_days": burst_days.copy()})
        gap = diff - 1
        out.append({"type": GAP, "start": int(valid[i - 1] + 1), "end": int(valid[i] - 1),
                    "length": gap, "event_days": np.empty(0, np.int64)})
        start = i
        gap_before = gap
    burst_days = valid[start:]
    out.append({"type": BURST, "start": int(burst_days[0]), "end": int(burst_days[-1]),
                "length": int(burst_days[-1] - burst_days[0] + 1),
                "gap_before": gap_before, "event_days": burst_days.copy()})
    open_gap = int(cutoff_day - valid[-1])
    out.append({"type": GAP, "start": int(valid[-1] + 1), "end": cutoff_day,
                "length": open_gap, "event_days": np.empty(0, np.int64)})
    return out


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value)) if value else 0.0


def _burst_numeric(days: np.ndarray, values: np.ndarray, cutoff_day: int,
                   gap_before: int, scale: np.ndarray) -> tuple[np.ndarray, float]:
    sums = np.asarray(values, float).sum(axis=0)
    out = np.zeros(N_TOKEN_NUMERIC, np.float32)
    out[:14] = np.log1p(np.maximum(sums, 0.0)) * np.asarray(scale, float)
    span = int(days[-1] - days[0] + 1)
    active = len(days)
    purchase = float(sums[seq.CHANNELS.index("buy")])
    gmv = np.asarray(values[:, seq.CHANNELS.index("gmv")], float)
    t = np.asarray(days - days[0], float)
    if len(days) <= 1 or float(np.var(t)) == 0.0:
        slope = 0.0
    else:
        slope = float(np.sum((t - t.mean()) * (gmv - gmv.mean())) / np.sum((t - t.mean()) ** 2))
    total_gmv = float(gmv.sum())
    search_gmv = float(values[:, seq.CHANNELS.index("gmv_search")].sum())
    out[14] = math.log1p(span) / LOG365
    out[15] = math.log1p(active) / LOG365
    out[16] = active / span
    out[17] = math.log1p(cutoff_day - int(days[-1])) / LOG365
    out[18] = math.log1p(gap_before) / LOG365
    out[19] = math.log1p(purchase) / LOG365
    out[20] = _signed_log1p(slope) * float(scale[seq.CHANNELS.index("gmv")])
    out[21] = search_gmv / total_gmv if total_gmv > 0 else 0.5
    return out, float(cutoff_day - int(days[-1]))


def _gap_numeric(length: int, age_end: int) -> np.ndarray:
    out = np.zeros(N_TOKEN_NUMERIC, np.float32)
    out[14] = math.log1p(max(length, 0)) / LOG365
    out[17] = math.log1p(max(age_end, 0)) / LOG365
    out[18] = math.log1p(max(length, 0)) / LOG365
    return out


def _summary_numeric(tokens: np.ndarray, count: int, scale: np.ndarray) -> np.ndarray:
    """Deterministic prefix SUMMARY; no fitted normalization or target input."""
    src = np.asarray(tokens[:count], float)
    out = np.zeros(N_TOKEN_NUMERIC, np.float32)
    for j in range(14):
        s = float(scale[j])
        natural = np.expm1(np.maximum(src[:, j] / s, 0.0)).sum()
        out[j] = math.log1p(max(natural, 0.0)) * s
    spans = np.expm1(src[:, 14] * LOG365)
    active = np.expm1(src[:, 15] * LOG365)
    purchase = np.expm1(src[:, 19] * LOG365)
    out[14] = math.log1p(float(spans.sum())) / LOG365
    out[15] = math.log1p(float(active.sum())) / LOG365
    out[16] = float(active.sum() / max(spans.sum(), 1.0))
    out[17] = float(src[-1, 17])
    out[18] = 0.0
    out[19] = math.log1p(float(purchase.sum())) / LOG365
    out[20] = 0.0
    gmv = np.expm1(np.maximum(out[seq.CHANNELS.index("gmv")] /
                                      float(scale[seq.CHANNELS.index("gmv")]), 0.0))
    sg = np.expm1(np.maximum(out[seq.CHANNELS.index("gmv_search")] /
                                    float(scale[seq.CHANNELS.index("gmv_search")]), 0.0))
    out[21] = float(sg / gmv) if gmv > 0 else 0.5
    return out


def build_history_tokens(event_days: np.ndarray, event_values: np.ndarray, cutoff_day: int,
                         available_depth: int, scale: np.ndarray) -> dict[str, Any]:
    """Build at most 191 history tokens; QUERY is appended by the model."""
    days = np.asarray(event_days, np.int64)
    values = np.asarray(event_values, float)
    if len(days) != len(values) or values.ndim != 2 or values.shape[1] != 14:
        raise ValueError("event arrays must be (n,) days and (n,14) natural values")
    lo = cutoff_day - int(available_depth) + 1
    mask = (days >= lo) & (days <= cutoff_day)
    d = days[mask]
    v = values[mask]
    order = np.argsort(d, kind="mergesort")
    d, v = d[order], v[order]
    segments = segment_days(d, cutoff_day, available_depth)
    tokens, types, ages = [], [], []
    for seg in segments:
        if seg["type"] == GAP:
            age = cutoff_day - int(seg["end"])
            tokens.append(_gap_numeric(int(seg["length"]), age))
            types.append(GAP); ages.append(float(age))
        else:
            md = np.isin(d, seg["event_days"])
            tok, age = _burst_numeric(d[md], v[md], cutoff_day,
                                      int(seg.get("gap_before", 0)), scale)
            tokens.append(tok); types.append(BURST); ages.append(age)
    token_array = np.asarray(tokens, np.float32)
    type_array = np.asarray(types, np.uint8)
    age_array = np.asarray(ages, np.float32)
    overflow = len(token_array) > MAX_HISTORY_TOKENS
    if overflow:
        old = len(token_array) - KEEP_RECENT_TOKENS
        token_array = np.vstack([_summary_numeric(token_array, old, scale), token_array[-KEEP_RECENT_TOKENS:]])
        type_array = np.concatenate([np.asarray([SUMMARY], np.uint8), type_array[-KEEP_RECENT_TOKENS:]])
        age_array = np.concatenate([np.asarray([age_array[old - 1]], np.float32),
                                    age_array[-KEEP_RECENT_TOKENS:]])
    assert token_array.shape == (len(type_array), N_TOKEN_NUMERIC)
    assert len(token_array) <= MAX_HISTORY_TOKENS
    return {"tokens": token_array, "types": type_array, "ages": age_array,
            "n_events": int(len(d)), "segments": segments, "overflow": bool(overflow),
            "covered_event_days": d.copy()}


# --------------------------------------------------------------------------- fast structural cache
@njit(cache=True)
def _median_prefix(values: np.ndarray, n: int) -> float:
    if n <= 0:
        return 0.0
    x = np.sort(values[:n])
    if n % 2:
        return float(x[n // 2])
    return float(0.5 * (x[n // 2 - 1] + x[n // 2]))


@njit(parallel=True, cache=True)
def _episode_kernel(ptr: np.ndarray, days: np.ndarray, raw: np.ndarray,
                    rows: np.ndarray, cut_days: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), 32), np.float32)
    for ii in prange(len(rows)):
        u = int(rows[ii]); cut = int(cut_days[ii]); lo = max(0, cut - 364)
        s = int(ptr[u]); e = int(ptr[u + 1])
        while s < e and int(days[s]) < lo:
            s += 1
        q = s
        while q < e and int(days[q]) <= cut:
            q += 1
        depth = min(365, cut + 1)
        out[ii, 31] = depth
        if s >= q:
            out[ii, 13] = depth; out[ii, 17] = depth
            out[ii, 19] = 1.0
            continue

        spans = np.empty(128, np.float32); active = np.empty(128, np.float32)
        purchases = np.empty(128, np.float32); gmvs = np.empty(128, np.float32)
        intens = np.empty(128, np.float32); slopes = np.empty(128, np.float32)
        peaks = np.empty(128, np.float32); sint = np.empty(128, np.float32)
        cint = np.empty(128, np.float32); oint = np.empty(128, np.float32)
        gaps = np.empty(128, np.float32)
        nb = 0; ng = 0
        bstart = int(days[s]); blast = bstart; nact = 0; npur = 0.0
        sumg = 0.0; sumsearch = 0.0; sumcat = 0.0; sumcart = 0.0; sumord = 0.0
        peak = 0.0; sumt = 0.0; sumt2 = 0.0; sumtg = 0.0
        last_event = bstart
        for k in range(s, q):
            day = int(days[k])
            if k > s and day - last_event > BURST_THRESHOLD_DAYS:
                span = blast - bstart + 1
                den = nact * sumt2 - sumt * sumt
                slope = 0.0 if nact <= 1 or den == 0 else (nact * sumtg - sumt * sumg) / den
                spans[nb] = span; active[nb] = nact; purchases[nb] = npur; gmvs[nb] = sumg
                intens[nb] = (sumsearch + sumcat + sumcart + sumord) / max(span, 1)
                slopes[nb] = slope; peaks[nb] = peak
                sint[nb] = sumsearch / max(span, 1); cint[nb] = sumcart / max(span, 1)
                oint[nb] = sumord / max(span, 1); nb += 1
                gaps[ng] = day - last_event - 1; ng += 1
                bstart = day; nact = 0; npur = 0.0; sumg = 0.0
                sumsearch = 0.0; sumcat = 0.0; sumcart = 0.0; sumord = 0.0
                peak = 0.0; sumt = 0.0; sumt2 = 0.0; sumtg = 0.0
            t = day - bstart
            g = float(raw[k, 13])
            nact += 1; npur += float(raw[k, 2]); sumg += g
            sumsearch += float(raw[k, 4]); sumcat += float(raw[k, 1])
            sumcart += float(raw[k, 9]); sumord += float(raw[k, 10])
            if g > peak: peak = g
            sumt += t; sumt2 += t * t; sumtg += t * g
            blast = day; last_event = day
        span = blast - bstart + 1
        den = nact * sumt2 - sumt * sumt
        slope = 0.0 if nact <= 1 or den == 0 else (nact * sumtg - sumt * sumg) / den
        spans[nb] = span; active[nb] = nact; purchases[nb] = npur; gmvs[nb] = sumg
        intens[nb] = (sumsearch + sumcat + sumcart + sumord) / max(span, 1)
        slopes[nb] = slope; peaks[nb] = peak
        sint[nb] = sumsearch / max(span, 1); cint[nb] = sumcart / max(span, 1)
        oint[nb] = sumord / max(span, 1); nb += 1

        mean_span = 0.0; mean_active = 0.0; mean_gmv = 0.0; max_span = 0.0; max_gmv = 0.0
        singleton = 0
        for j in range(nb):
            mean_span += spans[j]; mean_active += active[j]; mean_gmv += gmvs[j]
            if spans[j] > max_span: max_span = spans[j]
            if gmvs[j] > max_gmv: max_gmv = gmvs[j]
            if active[j] == 1: singleton += 1
        mean_span /= nb; mean_active /= nb; mean_gmv /= nb
        hist_int = intens[nb - 1]
        if nb > 1:
            hist_int = 0.0
            for j in range(nb - 1): hist_int += intens[j]
            hist_int /= nb - 1
        current = cut - last_event
        med_gap = _median_prefix(gaps, ng)
        mean_gap = 0.0; max_gap = 0.0
        for j in range(ng):
            mean_gap += gaps[j]
            if gaps[j] > max_gap: max_gap = gaps[j]
        if ng: mean_gap /= ng
        token_count = 2 * nb
        out[ii, 0] = nb; out[ii, 1] = _median_prefix(spans, nb)
        out[ii, 2] = mean_span; out[ii, 3] = max_span; out[ii, 4] = spans[nb - 1]
        out[ii, 5] = mean_active; out[ii, 6] = active[nb - 1]
        out[ii, 7] = purchases[nb - 1]; out[ii, 8] = gmvs[nb - 1]
        out[ii, 9] = slopes[nb - 1]; out[ii, 10] = intens[nb - 1]
        out[ii, 11] = hist_int; out[ii, 12] = (intens[nb - 1] + 1.0) / (hist_int + 1.0)
        out[ii, 13] = current; out[ii, 14] = med_gap; out[ii, 15] = mean_gap
        out[ii, 16] = max_gap; out[ii, 17] = current / max(med_gap, 1.0)
        out[ii, 18] = max(nb - 1, 0); out[ii, 19] = token_count
        out[ii, 20] = 1.0 if token_count > MAX_HISTORY_TOKENS else 0.0
        out[ii, 21] = singleton / nb; out[ii, 22] = sint[nb - 1]
        out[ii, 23] = cint[nb - 1]; out[ii, 24] = oint[nb - 1]
        out[ii, 25] = peaks[nb - 1]; out[ii, 26] = 1.0 if purchases[nb - 1] > 0 else 0.0
        out[ii, 27] = mean_gmv; out[ii, 28] = max_gmv
        out[ii, 29] = q - s; out[ii, 30] = ng
    return out


# --------------------------------------------------------------------------- candidate model / parameter audit
def build_candidate_model(cfg: dict[str, Any]):
    import torch
    from torch import nn

    class BurstGapETX(nn.Module):
        def __init__(self, c: dict[str, Any]):
            super().__init__()
            self.core = etx.build_model(c)
            self.type_embedding = nn.Embedding(5, c["d_model"])

        def forward(self, tok, token_type, static, age, n):
            base = self.core
            B, K, _ = tok.shape; d = base.cls.numel()
            ev = torch.arange(K, device=tok.device).unsqueeze(0) < n.unsqueeze(1)
            h = torch.zeros(B, K + 1, d, dtype=tok.dtype, device=tok.device)
            he = base.tok(tok) + self.type_embedding(token_type.long())
            h[:, :K] = he * ev.unsqueeze(-1)
            qtype = torch.full((B,), QUERY, dtype=torch.long, device=tok.device)
            qtok = (base.cls + base.static(static) + self.type_embedding(qtype)).unsqueeze(1)
            h = h.scatter(1, n.view(B, 1, 1).expand(B, 1, d), qtok.to(h.dtype))
            a = torch.zeros(B, K + 1, dtype=age.dtype, device=age.device)
            a[:, :K] = age * ev
            a = a / etx.TAU_UNIT
            for block in base.blocks:
                h = block(h, a)
            h = base.norm(h)
            zq = h.gather(1, n.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
            zl = h.gather(1, (n - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, d)).squeeze(1)
            w = ev.to(h.dtype).unsqueeze(-1)
            zm = (h[:, :K] * w).sum(1) / w.sum(1).clamp_min(1.0)
            return base.head(torch.cat([zq, zm, zl], dim=1)).squeeze(1)

    return BurstGapETX(cfg)


def parameter_audit() -> dict[str, Any]:
    cfg = dict(etx.DEFAULT_CFG, z0=0.0)
    baseline = etx.n_params()
    candidate = sum(p.numel() for p in build_candidate_model(cfg).parameters())
    delta = candidate - baseline
    return {"baseline_params": baseline, "candidate_params": candidate,
            "extra_params": delta, "relative_delta": delta / baseline,
            "within_2pct": abs(delta / baseline) <= 0.02,
            "extra_source": "5 x d_model learned token-type embedding"}


# --------------------------------------------------------------------------- exact baseline audit/context
def load_context(load_features: bool = True):
    frame, manifest = exp053._load_core()
    exp053._audit_baseline(frame, manifest)
    exp053._load_optional(frame, manifest)
    if not load_features:
        return frame, manifest, None, None, None, None
    state, state_names = exp053._load_state_features(frame, manifest)
    exp053._load_pact(frame, manifest)
    disagreement, disagreement_names = exp053.build_disagreement_features(frame)
    strata = exp053.build_strata(frame, state)
    return frame, manifest, state, state_names, disagreement, disagreement_names, strata


def audit_baseline() -> dict[str, Any]:
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    frame, manifest, *_ = load_context(False)
    late = frame["cutoff"] == FOLDS[-1]
    path = ROOT / "artifacts" / "oof_ETX-01-S42-V1016.npz"
    data = np.load(path, allow_pickle=False)
    order = exp053.canonical_order(data["user_id"], data["cutoff"])
    uid = np.asarray(data["user_id"], np.int64)[order]
    cut = np.asarray(data["cutoff"], dtype="U10")[order]
    y = np.asarray(data["y"], float)[order]
    z = np.asarray(data["z"], float)[order]
    if not np.array_equal(uid, frame["user_id"][late]) or not np.array_equal(cut, frame["cutoff"][late]):
        raise AssertionError("ETX-01-S42-V1016 row alignment failed")
    if not np.allclose(y, frame["y"][late], rtol=0.0, atol=1e-6):
        raise AssertionError("ETX-01-S42-V1016 target mismatch")
    if "etx_s42" not in frame or not np.allclose(z, frame["etx_s42"][late], atol=1e-7, rtol=0.0):
        raise AssertionError("ETX-01-S42-V1016 does not reconstruct full S42 OOF")

    import torch
    ckpt_path = ROOT / "artifacts" / "model_ETX-01-S42-V1016.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    strongest = manifest["calibration_audit"]
    report = {
        "experiment_id": EXPERIMENT_ID, "base_head": BASE_HEAD,
        "current_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
        "initial_git_status_path": INITIAL_STATUS,
        "isolation": "current tree; EXP-054-only files because required latest baseline code is uncommitted",
        "etx_baseline": {
            "name": "ETX-01-S42-V1016", "oof_path": path, "oof_file_sha256": file_sha256(path),
            "row_key_sha256": array_sha256(cut, uid), "target_sha256": array_sha256(y),
            "prediction_sha256": array_sha256(z), "checkpoint_path": ckpt_path,
            "checkpoint_sha256": file_sha256(ckpt_path), "checkpoint_val": ckpt["val"],
            "config": cfg, "target": "direct log1p(GMV30)", "depth_policy": 289,
            "numeric_input_dimension": etx.N_TOK_FEAT,
            "configured_n_tok": int(cfg["n_tok"]),
            "implemented_max_history_plus_query": int(cfg["n_tok"] + 1),
            "requested_description_said_including_query": True,
        },
        "strongest_current": strongest,
        "component_manifest": manifest["core_components"],
        "weights": FIXED_ENSEMBLE_WEIGHTS,
        "late_fold_expected": 1.741278566,
        "late_fold_observed": strongest["fold_scores"][-1],
        "parameter_audit": parameter_audit(),
        "status": "PASS_EXACT",
    }
    if abs(float(strongest["fold_scores"][-1]) - 1.741278566) > 5e-7:
        raise AssertionError("late-fold STRONGEST score mismatch")
    if cfg["d_model"] != 128 or cfg["blocks"] != 5 or cfg["heads"] != 8 or cfg["head_dim"] != 16:
        raise AssertionError("ETX backbone config mismatch")
    if cfg["ffn"] != 384 or cfg["dropout"] != 0.1 or cfg["epochs"] != 4:
        raise AssertionError("ETX training config mismatch")
    write_json(OUT_ARTIFACTS / "baseline_manifest.json", report)
    write_json(OUT_ARTIFACTS / "segmentation_config.json", segmentation_config())
    write_json(OUT_ARTIFACTS / "gpu_plan.json", {
        "gated": True, "fold": "2025-10-16", "seed": int(SEED),
        "baseline_retrain": False, "candidate_only_change": "tokenizer + <2% type embedding",
        "backbone_config": cfg, "history_tokens": MAX_HISTORY_TOKENS,
        "query_tokens": 1, "full_folds_automatic": False,
        "test_inference": False, "submission": False,
    })
    return report


def build_episode_cache(force: bool = False) -> pl.DataFrame:
    if CACHE.exists() and not force:
        return pl.read_parquet(CACHE)
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, _manifest, *_ = load_context(False)
    x, days, _key, ptr = etx.events()
    raw = np.asarray(x, dtype=np.float32)
    np.expm1(raw[:, 4:], out=raw[:, 4:])
    rows = seq.user_rows(frame["user_id"]).astype(np.int64)
    cut_days = np.asarray([seq.day_index(dt.date.fromisoformat(v)) for v in frame["cutoff"]],
                          np.int32)
    t0 = time.time()
    episode = _episode_kernel(np.asarray(ptr, np.int64), np.asarray(days, np.int16),
                              raw, rows, cut_days)
    del raw
    data = {"cutoff": frame["cutoff"], "user_id": frame["user_id"],
            "y": frame["y"].astype(np.float32)}
    data.update({name: episode[:, i] for i, name in enumerate(EPISODE_NAMES)})
    cache = pl.DataFrame(data)
    cache.write_parquet(CACHE, compression="zstd")
    write_json(OUT_ARTIFACTS / "cache_manifest.json", {
        "path": CACHE, "rows": cache.height, "columns": cache.columns,
        "row_key_sha256": array_sha256(frame["cutoff"], frame["user_id"]),
        "target_sha256": array_sha256(frame["y"]), "episode_sha256": array_sha256(episode),
        "runtime_s": time.time() - t0, "cutoff_safe": True,
    })
    return cache


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def _segment_masks(state: pl.DataFrame) -> dict[str, np.ndarray]:
    rec = state["rec_buy"].to_numpy().astype(float)
    buy = state["w180_days_buy"].to_numpy().astype(float)
    return {
        "all": np.ones(len(rec), bool),
        "rec_buy_15_60": (rec >= 15) & (rec <= 60),
        "w180_days_buy_2_15": (buy >= 2) & (buy <= 15),
        "intersection": (rec >= 15) & (rec <= 60) & (buy >= 2) & (buy <= 15),
        "w180_days_buy_0_1": buy <= 1,
        "w180_days_buy_16plus": buy >= 16,
        "never_purchased": np.nan_to_num(rec, nan=1e9) > 365,
    }


def structural_audit() -> dict[str, Any]:
    cache = build_episode_cache()
    frame, manifest, state, state_names, disagreement, disagreement_names, strata = load_context(True)
    assert state is not None and state_names is not None
    if not (np.array_equal(cache["user_id"].to_numpy(), frame["user_id"])
            and np.array_equal(cache["cutoff"].to_numpy().astype("U10"), frame["cutoff"])):
        raise AssertionError("episode cache alignment failed")
    ep = {name: cache[name].to_numpy().astype(float) for name in EPISODE_NAMES}
    masks = _segment_masks(state)

    # Structural statistics for every required activity segment and fold.
    stat_cols = [
        "n_bursts", "mean_burst_span", "mean_burst_active_days", "last_burst_purchase_days",
        "last_burst_gmv", "last_burst_search_intensity", "last_burst_cart_intensity",
        "last_burst_order_intensity", "last_burst_peak_gmv", "last_burst_slope",
        "gap_count", "median_closed_gap", "mean_closed_gap", "max_closed_gap",
        "current_open_gap", "current_gap_ratio", "last_burst_intensity",
        "previous_burst_purchase_presence", "reactivation_count", "burst_gap_token_count",
        "overflow", "singleton_burst_share",
    ]
    stat_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fm = frame["cutoff"] == fold
        for segment, sm in masks.items():
            m = fm & sm
            row: dict[str, Any] = {"fold": fold, "segment": segment, "n": int(m.sum())}
            for name in stat_cols:
                v = ep[name][m]
                row[name + "_mean"] = float(np.mean(v)) if len(v) else np.nan
                row[name + "_median"] = float(np.median(v)) if len(v) else np.nan
                row[name + "_max"] = float(np.max(v)) if len(v) else np.nan
            stat_rows.append(row)
    write_csv(RESULTS / "structural_statistics.csv", stat_rows)

    # Novelty versus the named existing state families.
    requested_existing = [
        "rec_buy", "rec_any", "buygap_median", "buygap_mean", "buygap_max",
        "gap_mean", "w30_days_buy", "w90_days_buy", "w180_days_buy",
        "w30_gmv", "w90_gmv", "w180_gmv",
    ]
    existing = [name for name in requested_existing if name in state_names]
    existing += [name for name in state_names if name.startswith("trend_") or name.startswith("dlog_")]
    existing = list(dict.fromkeys(existing))
    missing = [name for name in requested_existing if name not in state_names]
    key_episode = ["n_bursts", "median_burst_span", "last_burst_intensity",
                   "last_to_historical_intensity", "current_open_gap", "median_closed_gap",
                   "current_gap_ratio", "reactivation_count", "last_burst_slope"]
    novelty_rows = []
    for ename in key_episode:
        vals = ep[ename]
        best_name, best_corr = None, 0.0
        for sname in existing:
            corr = _safe_spearman(vals, state[sname].to_numpy().astype(float))
            novelty_rows.append({"episode": ename, "existing": sname, "spearman": corr})
            if abs(corr) > abs(best_corr): best_name, best_corr = sname, corr
        novelty_rows.append({"episode": ename, "existing": "__MAX_ABS__",
                             "spearman": best_corr, "best_existing": best_name})
    write_csv(RESULTS / "novelty_correlations.csv", novelty_rows)
    max_corrs = [abs(float(r["spearman"])) for r in novelty_rows if r["existing"] == "__MAX_ABS__"]
    novelty_pass = not all(value >= 0.95 for value in max_corrs)

    # Loss objects use the same per-fold calibration as EXP-053.
    z_etx_cal, _, _ = exp053.fold_calibrated(frame["y"], frame["etx"], frame["fold_index"])
    z_seq_cal, _, _ = exp053.fold_calibrated(frame["y"], frame["seq"], frame["fold_index"])
    ly = np.log1p(frame["y"])
    etx_minus_seq_loss = np.square(ly - z_etx_cal) - np.square(ly - z_seq_cal)

    quartile_vars = ["current_gap_ratio", "last_burst_intensity",
                     "last_to_historical_intensity", "reactivation_count"]
    qrows: list[dict[str, Any]] = []
    mid_spreads: dict[str, float] = {}
    for name in quartile_vars:
        series = pd.Series(ep[name])
        pct = series.groupby(pd.Series(strata)).rank(method="first", pct=True)
        quartile = np.minimum(np.ceil(pct.to_numpy() * 4).astype(np.int8), 4)
        for fold_i, fold in enumerate(FOLDS):
            for segment, sm in masks.items():
                for q in range(1, 5):
                    m = (frame["fold_index"] == fold_i) & sm & (quartile == q)
                    if not m.any(): continue
                    qrows.append({
                        "variable": name, "fold": fold, "segment": segment, "quartile": q,
                        "n": int(m.sum()), "p_y_positive": float(np.mean(frame["y"][m] > 0)),
                        "mean_log1p_y": float(np.mean(ly[m])),
                        "strongest_signed_residual": float(np.mean(frame["r_strong"][m])),
                        "etx_loss_minus_seq_loss": float(np.mean(etx_minus_seq_loss[m])),
                    })
        mid = masks["intersection"]
        rates = [float(np.mean(frame["y"][mid & (quartile == q)] > 0)) for q in range(1, 5)]
        mid_spreads[name] = float(max(rates) - min(rates))
    write_csv(RESULTS / "within_stratum_diagnostics.csv", qrows)
    best_mid_spread = max(mid_spreads.values())
    report = {
        "novelty_pass": novelty_pass, "max_abs_novelty_correlations": max_corrs,
        "missing_requested_state_columns": missing, "mid_activity_spreads": mid_spreads,
        "best_mid_activity_p_positive_spread": best_mid_spread,
        "overflow_share": float(np.mean(ep["overflow"])),
        "singleton_burst_share": float(np.mean(ep["singleton_burst_share"])),
        "cache_sha256": file_sha256(CACHE), "status": "PASS" if novelty_pass else "FAIL",
    }
    write_json(OUT_ARTIFACTS / "structural_audit.json", report)
    return report


# --------------------------------------------------------------------------- fixed CPU pre-flight
def protocol_masks(frame: dict[str, np.ndarray], donor_side: int) -> tuple[np.ndarray, np.ndarray]:
    side = user_group(frame["user_id"])
    donor = (frame["fold_index"] < 3) & (side == donor_side)
    recipient = (frame["fold_index"] == 3) & (side == 1 - donor_side)
    return donor, recipient


def _lgb_params(objective: str) -> dict[str, Any]:
    params = {k: v for k, v in exp053.PROBE_PARAMS.items()
              if k not in {"num_boost_round", "early_stopping"}}
    params.update(objective=objective, verbosity=-1,
                  num_threads=min(12, os.cpu_count() or 1),
                  metric="binary_logloss" if objective == "binary" else "l1")
    return params


def _fit(X: np.ndarray, target: np.ndarray, mask: np.ndarray, objective: str) -> lgb.Booster:
    data = lgb.Dataset(np.asarray(X[mask], np.float32), label=np.asarray(target[mask], np.float32),
                       free_raw_data=True)
    return lgb.train(_lgb_params(objective), data,
                     num_boost_round=int(exp053.PROBE_PARAMS["num_boost_round"]))


def _select_scale(y: np.ndarray, z: np.ndarray, correction: np.ndarray,
                  fold_index: np.ndarray, donor: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for scale in exp053.SCALES:
        scores = []
        for fold in range(3):
            m = donor & (fold_index == fold)
            scores.append(calibrate(y[m], z[m] + scale * correction[m])[1])
        value = float(np.average(scores, weights=FOLD_WEIGHTS[:3]))
        rows.append({"scale": float(scale), "selection_wcv_3f": value,
                     "fold_scores": scores})
    best = min(r["selection_wcv_3f"] for r in rows)
    selected = min(r["scale"] for r in rows if r["selection_wcv_3f"] <= best + 1e-5)
    return float(selected), rows


def run_residual_protocol(X: np.ndarray, frame: dict[str, np.ndarray], label: str) -> dict[str, Any]:
    residual = frame["r_strong"]; fi = frame["fold_index"]
    correction = np.full(len(residual), np.nan, float)
    halves, selection_rows = [], []
    for donor_side in (0, 1):
        donor, recipient = protocol_masks(frame, donor_side)
        if np.intersect1d(frame["user_id"][donor], frame["user_id"][recipient]).size:
            raise AssertionError("recipient user leaked into training")
        centered = residual.copy()
        for fold in range(3):
            m = donor & (fi == fold); centered[m] -= float(centered[m].mean())
        donor_oof = np.full(len(residual), np.nan, float)
        for held in range(3):
            train = donor & (fi != held); valid = donor & (fi == held)
            model = _fit(X, centered, train, "regression_l1")
            donor_oof[valid] = model.predict(np.asarray(X[valid], np.float32))
            del model
        lo, hi = np.quantile(donor_oof[donor], [0.01, 0.99])
        processed = np.zeros(len(residual), float)
        processed[donor] = np.clip(donor_oof[donor], lo, hi)
        for fold in range(3):
            m = donor & (fi == fold); processed[m] -= float(processed[m].mean())
        selected, curve = _select_scale(frame["y"], frame["z_strong_raw"], processed, fi, donor)
        for row in curve:
            selection_rows.append({"variant": label, "direction": f"{donor_side}->{1-donor_side}",
                                   "selected": row["scale"] == selected, **row})
        final = _fit(X, centered, donor, "regression_l1")
        pred = final.predict(np.asarray(X[recipient], np.float32)); del final
        pred = np.clip(pred, lo, hi); pred -= float(pred.mean())
        corr = selected * pred; correction[recipient] = corr
        base_score = calibrate(frame["y"][recipient], frame["z_strong_raw"][recipient])[1]
        cand_score = calibrate(frame["y"][recipient], frame["z_strong_raw"][recipient] + corr)[1]
        halves.append({
            "variant": label, "direction": f"{donor_side}->{1-donor_side}",
            "n_donor": int(donor.sum()), "n_recipient": int(recipient.sum()),
            "selected_scale": selected, "baseline_rmsle": base_score,
            "candidate_rmsle": cand_score, "delta": cand_score - base_score,
            "raw_probe_residual_corr": float(np.corrcoef(pred, residual[recipient])[0, 1]),
            "correction_residual_corr": (0.0 if selected == 0.0 else
                                           float(np.corrcoef(corr, residual[recipient])[0, 1])),
            "recipient_user_overlap": 0,
        })
    late = fi == 3
    if np.isnan(correction[late]).any():
        raise AssertionError("late correction incomplete")
    b = calibrate(frame["y"][late], frame["z_strong_raw"][late])[1]
    c = calibrate(frame["y"][late], frame["z_strong_raw"][late] + correction[late])[1]
    return {"variant": label, "correction": correction, "halves": halves,
            "selection_rows": selection_rows, "late_baseline": b,
            "late_candidate": c, "late_delta": c - b,
            "late_corr": float(np.corrcoef(correction[late], residual[late])[0, 1])}


def run_auc_protocol(X: np.ndarray, frame: dict[str, np.ndarray]) -> dict[str, Any]:
    probability = np.full(len(frame["y"]), np.nan, float)
    target = (frame["y"] > 0).astype(np.int8)
    halves = []
    for donor_side in (0, 1):
        donor, recipient = protocol_masks(frame, donor_side)
        model = _fit(X, target, donor, "binary")
        p = model.predict(np.asarray(X[recipient], np.float32)); del model
        probability[recipient] = p
        halves.append({"direction": f"{donor_side}->{1-donor_side}",
                       "auc": float(roc_auc_score(target[recipient], p)),
                       "n": int(recipient.sum())})
    late = frame["fold_index"] == 3
    return {"auc": float(roc_auc_score(target[late], probability[late])), "halves": halves}


def preflight() -> dict[str, Any]:
    structural = structural_audit()
    cache = pl.read_parquet(CACHE)
    frame, manifest, state, state_names, disagreement, disagreement_names, strata = load_context(True)
    assert state is not None and state_names is not None and disagreement is not None
    state_x = state.to_numpy().astype(np.float32); state_x[~np.isfinite(state_x)] = np.nan
    episode_x = cache.select(PROBE_EPISODE_NAMES).to_numpy().astype(np.float32)
    combined = np.column_stack([state_x, disagreement, episode_x]).astype(np.float32)
    real = run_residual_protocol(combined, frame, "REAL")
    auc_real = run_auc_protocol(combined, frame)
    del combined

    perm = exp053.permutation_within_strata(strata, np.ones(len(strata), bool), int(SEED))
    shuffled_episode = episode_x[perm]
    if not np.array_equal(strata, strata[perm]):
        raise AssertionError("joint episode shuffle changed strata")
    combined_shuf = np.column_stack([state_x, disagreement, shuffled_episode]).astype(np.float32)
    shuffled = run_residual_protocol(combined_shuf, frame, "SHUFFLED")
    auc_shuf = run_auc_protocol(combined_shuf, frame)
    del combined_shuf

    real_minus_shuf = float(real["late_delta"] - shuffled["late_delta"])
    both_halves_better = all(row["delta"] < 0 for row in real["halves"])
    both_corr_positive = all(row["correction_residual_corr"] > 0 for row in real["halves"])
    auc_gain = float(auc_real["auc"] - auc_shuf["auc"])
    additional = (structural["best_mid_activity_p_positive_spread"] >= 0.03
                  or auc_gain >= 0.002)
    passed = bool(
        real["late_delta"] <= -0.0005
        and real_minus_shuf <= -0.0004
        and both_halves_better
        and both_corr_positive
        and structural["novelty_pass"]
        and additional
    )
    verdict = "PASS_TO_GPU" if passed else "NO_GO_PREFLIGHT"
    half_rows = real["halves"] + shuffled["halves"]
    write_csv(RESULTS / "recipient_half_metrics.csv", half_rows)
    write_csv(RESULTS / "residual_scale_selection.csv",
              real["selection_rows"] + shuffled["selection_rows"])
    write_json(OUT_ARTIFACTS / "shuffle_audit.json", {
        "joint_rows": True, "strata_preserved": True,
        "strata_sha256": array_sha256(strata), "permutation_sha256": array_sha256(perm),
        "user_halves": "splitmix64(user_id)&1", "recipient_labels_in_training": False,
    })
    report = {
        "verdict": verdict, "gpu_allowed": passed,
        "real_late_delta": real["late_delta"], "shuffled_late_delta": shuffled["late_delta"],
        "real_minus_shuffled": real_minus_shuf, "both_recipient_halves_better": both_halves_better,
        "correction_residual_corr_positive_both": both_corr_positive,
        "real_half_metrics": real["halves"], "shuffled_half_metrics": shuffled["halves"],
        "episode_novelty_pass": structural["novelty_pass"],
        "mid_activity_p_positive_spread": structural["best_mid_activity_p_positive_spread"],
        "auc_real": auc_real, "auc_shuffled": auc_shuf, "auc_gain": auc_gain,
        "criteria": {
            "real_late_delta_le_-0.0005": real["late_delta"] <= -0.0005,
            "real_minus_shuffled_le_-0.0004": real_minus_shuf <= -0.0004,
            "both_recipient_halves_better": both_halves_better,
            "correlation_positive_both": both_corr_positive,
            "novelty": structural["novelty_pass"],
            "spread_ge_0.03_or_auc_gain_ge_0.002": additional,
        },
        "probe_params": exp053.PROBE_PARAMS,
        "base_feature_count": len(state_names) + len(disagreement_names),
        "episode_feature_count": len(PROBE_EPISODE_NAMES),
        "feature_names": state_names + disagreement_names + PROBE_EPISODE_NAMES,
        "fixed_scales": exp053.SCALES,
    }
    write_json(OUT_ARTIFACTS / "preflight_verdict.json", report)
    write_json(OUT_ARTIFACTS / "summary.json", report)
    return report


def analysis_only() -> dict[str, Any]:
    paths = [OUT_ARTIFACTS / "baseline_manifest.json", OUT_ARTIFACTS / "segmentation_config.json",
             OUT_ARTIFACTS / "cache_manifest.json", OUT_ARTIFACTS / "structural_audit.json",
             OUT_ARTIFACTS / "preflight_verdict.json", RESULTS / "structural_statistics.csv",
             RESULTS / "novelty_correlations.csv", RESULTS / "within_stratum_diagnostics.csv",
             RESULTS / "recipient_half_metrics.csv", RESULTS / "residual_scale_selection.csv"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"analysis-only artifacts missing: {missing}")
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    result = {"status": "PASS", "hashes": hashes, "forbidden_paths": {
        "test_inference": False, "submission": False, "leaderboard": False,
    }}
    write_json(OUT_ARTIFACTS / "analysis_hash_replay.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "audit", "build", "preflight", "analysis-only"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"all", "audit"}:
        audit_baseline()
    if args.command in {"all", "build"}:
        build_episode_cache(force=args.force)
        structural_audit()
    if args.command in {"all", "preflight"}:
        result = preflight()
        if result["gpu_allowed"]:
            print("READY_FOR_GPU")
    if args.command == "analysis-only" or args.command == "all":
        analysis_only()


if __name__ == "__main__":
    main()
