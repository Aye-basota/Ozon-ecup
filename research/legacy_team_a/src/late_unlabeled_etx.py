"""EXP-056: target-free late-history adaptation of the saved ETX V0904 checkpoint.

The primary contrast is LATE-SSL minus CONTROL-CLEANSSL.  Both arms load the
same saved checkpoint, replay the same clean direct batches, use the same
optimizer/LR/mask plan and differ only in the dates that supply unlabeled
histories.  No future target is built for either SSL corridor.

Commands are intentionally separated so the target-free audit and adaptation
can be inspected before validation targets are read::

    python src/late_unlabeled_etx.py audit
    python src/late_unlabeled_etx.py embed --model BASE
    python src/late_unlabeled_etx.py domain
    python src/late_unlabeled_etx.py replay
    python src/late_unlabeled_etx.py pilot
    python src/late_unlabeled_etx.py embed --model CONTROL
    python src/late_unlabeled_etx.py embed --model LATE
    python src/late_unlabeled_etx.py analyze

There is deliberately no test-prediction, leaderboard or submission command.
The production cutoff is admitted only by ``embed`` as an input-only domain
diagnostic; the direct head is never evaluated there.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import etx, seq
from src.config import ARTIFACTS, CUTOFF_TEST, DATA_PROCESSED, SEED
from src.features import panel_users
from src.validation import calibrate, rmsle_z

EXP = "LATE_SSL_EXP056"
OUT = ARTIFACTS / EXP
RESULTS = ROOT / "research" / "strategies" / "results" / EXP
CHECKPOINT = ARTIFACTS / "model_ETX-01-S42-V0904.pt"
PLAN_NPZ = OUT / "paired_plan.npz"
PLAN_JSON = OUT / "paired_plan.json"
AUDIT_NPZ = OUT / "domain_audit_cpu.npz"
AUDIT_JSON = RESULTS / "domain_audit_cpu.json"
DOMAIN_JSON = RESULTS / "domain_shift_summary.json"
REPLAY_JSON = RESULTS / "deterministic_replay_v2.json"
SUMMARY_JSON = RESULTS / "summary.json"
LIGHT_EVENT_KEY = OUT / "event_key_int32.npy"

START_VAL = dt.date(2025, 9, 4)
PRIMARY_VAL = dt.date(2025, 10, 16)
THURSDAY = 3
DEPTH_CAP = 212
N_TOK = 192
BATCH = 512
CHUNK = 128
MASK_RATE = 0.15
LAMBDA_SSL = 0.25
AUDIT_SAMPLE = 100_000
TOKEN_SAMPLE_HISTORIES = 8_000
EMBED_SAMPLE = 20_000
MMD_SAMPLE = 2_000
ACTIVITY_EDGES = np.array([0, 1, 2, 4, 8, 16, 10**9], np.int32)

CONTROL_CUTS = tuple(dt.date(2025, 5, 22) + dt.timedelta(days=7 * i) for i in range(11))
LATE_CUTS = tuple(dt.date(2025, 8, 7) + dt.timedelta(days=7 * i) for i in range(11))

FEATURE_NAMES = [
    "token_count", "event_density", "recency", "gap_mean", "gap_std", "gap_max",
    "mix_search", "mix_cart", "mix_order", "positive_gmv_day_rate",
] + [f"mean_{c}" for c in seq.CHANNELS]


def light_events():
    """Exact ETX event table with a memory-mapped int32 search key.

    The shared ETX loader materializes an int64 user id vector and int64 key in
    RAM.  EXP-056 needs the same rows but keeps the key in its isolated cache;
    ``N_USERS * DAY_STRIDE`` is safely below int32 max.  This changes neither
    selection nor tokenizer values and avoids a multi-hundred-MB transient.
    """
    if "x" not in etx._E:
        x = np.load(etx.EV_X, mmap_mode="r")
        day = np.load(etx.EV_DAY, mmap_mode="r")
        ptr = np.load(etx.EV_PTR, mmap_mode="r")
        assert int(ptr[-1]) == len(day) == len(x)
        assert int(seq.N_USERS) * int(etx.DAY_STRIDE) < np.iinfo(np.int32).max
        if not LIGHT_EVENT_KEY.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            tmp = LIGHT_EVENT_KEY.with_name(LIGHT_EVENT_KEY.name + ".tmp.npy")
            assert not tmp.exists(), f"partial isolated event key exists: {tmp}"
            key = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int32,
                                            shape=(len(day),))
            for u0 in range(0, seq.N_USERS, 20_000):
                u1 = min(u0 + 20_000, seq.N_USERS)
                lo, hi = int(ptr[u0]), int(ptr[u1])
                uid = np.repeat(np.arange(u0, u1, dtype=np.int32), np.diff(ptr[u0:u1 + 1]))
                key[lo:hi] = uid * np.int32(etx.DAY_STRIDE) + day[lo:hi].astype(np.int32)
            key.flush()
            del key
            os.replace(tmp, LIGHT_EVENT_KEY)
        key = np.load(LIGHT_EVENT_KEY, mmap_mode="r")
        assert key.dtype == np.int32 and len(key) == len(day)
        for lo in range(0, len(key), 1_000_000):
            hi = min(lo + 1_000_000, len(key))
            if hi - lo > 1:
                assert bool((np.diff(key[lo:hi]) > 0).all())
            if lo and hi > lo:
                assert int(key[lo]) > int(key[lo - 1])
        etx._E.update(x=x, day=day, ptr=ptr, key=key)
    return etx._E["x"], etx._E["day"], etx._E["key"], etx._E["ptr"]


def light_select(T: dt.date, rows: np.ndarray, n_tok: int,
                 depth_clip: int | None = None):
    """Bit-equivalent ``etx.select`` for the int32 memory-mapped key."""
    _, _, key, _ = light_events()
    d = seq.day_index(T)
    lo = max(0, d - seq.SEQ_L + 1)
    if depth_clip is not None:
        lo = max(lo, d + 1 - depth_clip)
    r = np.asarray(rows, np.int32)
    start_key = r * np.int32(etx.DAY_STRIDE) + np.int32(lo)
    end_key = r * np.int32(etx.DAY_STRIDE) + np.int32(d + 1)
    start = np.searchsorted(key, start_key, side="left")
    end = np.searchsorted(key, end_key, side="left")
    cnt = np.minimum(end - start, n_tok)
    j = np.arange(n_tok, dtype=np.int64)[None, :]
    idx = np.where(j < cnt[:, None], (end - cnt)[:, None] + j, 0)
    return idx.astype(np.int32), cnt.astype(np.int32)


# Isolated wrapper only: retain ETX semantics while keeping CPU audit/cache lean.
etx.events = light_events
etx.select = light_select

T0 = time.time()


def log(*parts: Any) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *parts, flush=True)


def iso_days(cuts: Iterable[dt.date]) -> list[str]:
    return [d.isoformat() for d in cuts]


def date_from_day(day: int) -> dt.date:
    return seq.DATA_START + dt.timedelta(days=int(day))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _hash_tensor(h: Any, name: str, value: Any) -> None:
    a = value.detach().cpu().contiguous().numpy()
    h.update(name.encode())
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, np.int64).tobytes())
    h.update(a.tobytes())


def state_dict_hash(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        _hash_tensor(h, name, state[name])
    return h.hexdigest()


def module_hash(module: Any) -> str:
    return state_dict_hash(module.state_dict())


def splitmix64(values: np.ndarray) -> np.ndarray:
    h = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        h += np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return h


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: dict[str, Any], reuse: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        if reuse and path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return sha256_file(path)


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "src" / "etx.py", ROOT / "src" / "seq.py",
             ROOT / "src" / "validation.py", ROOT / "src" / "config.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def load_checkpoint_cpu() -> dict[str, Any]:
    import torch
    assert CHECKPOINT.exists(), f"missing exact checkpoint {CHECKPOINT}"
    return torch.load(CHECKPOINT, map_location="cpu", weights_only=False)


def clean_cutoffs() -> tuple[dt.date, ...]:
    cuts = tuple(seq.fold_cutoffs(START_VAL))
    assert cuts[0] == dt.date(2025, 4, 3) and cuts[-1] == dt.date(2025, 7, 31)
    assert all(T + dt.timedelta(days=30) <= START_VAL for T in cuts)
    assert all(T.weekday() == THURSDAY for T in cuts)
    return cuts


def fixed_validation(with_target: bool = False):
    uid = panel_users(PRIMARY_VAL, 3)["user_id"].to_numpy().astype(np.int64)
    rows = seq.user_rows(uid)
    canonical = np.load(ARTIFACTS / "oof_ETX-01-S42-V1016.npz")
    assert np.array_equal(uid, canonical["user_id"]), "validation user order changed"
    assert np.all(np.asarray(canonical["cutoff"], dtype="U10") == PRIMARY_VAL.isoformat())
    if not with_target:
        return uid, rows
    y = seq.target_at(PRIMARY_VAL, rows)
    # Canonical OOF stores targets as float32; require exact equality in that
    # persisted representation (large GMV values differ by float32 rounding if
    # compared against the float64 panel sum directly).
    assert np.array_equal(y.astype(np.float32), canonical["y"])
    return uid, rows, y


def source_rows(T: dt.date) -> np.ndarray:
    uid = panel_users(T, 1)["user_id"].to_numpy()
    return seq.user_rows(uid)


def selected_counts_and_activity(T: dt.date, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Token count and buy-day activity using only events at or before T."""
    x, _, _, _ = etx.events()
    buy_i = seq.CHANNELS.index("buy")
    cnt = np.empty(len(rows), np.int16)
    buy = np.empty(len(rows), np.int16)
    for lo in range(0, len(rows), 20_000):
        hi = min(lo + 20_000, len(rows))
        idx, c = etx.select(T, rows[lo:hi], N_TOK, depth_clip=DEPTH_CAP)
        mask = np.arange(N_TOK)[None, :] < c[:, None]
        b = (x[idx, buy_i].astype(np.float32) * mask).sum(1)
        cnt[lo:hi] = c.astype(np.int16)
        buy[lo:hi] = np.rint(b).astype(np.int16)
    return cnt, buy


def activity_bin(buy_count: np.ndarray) -> np.ndarray:
    return np.digitize(buy_count, ACTIVITY_EDGES[1:-1], right=False).astype(np.int8)


def _chunks_from_pairs(parts: list[dict[str, np.ndarray]], rng: np.random.Generator):
    chunks: list[dict[str, np.ndarray]] = []
    for part in parts:
        order = rng.permutation(len(part["control_row"]))
        for lo in range(0, len(order), CHUNK):
            s = order[lo:lo + CHUNK]
            chunks.append({k: v[s] for k, v in part.items()})
    chunks = [chunks[i] for i in rng.permutation(len(chunks))]
    arrays = {k: [] for k in parts[0]}
    ptr = [0]
    for lo in range(0, len(chunks), BATCH // CHUNK):
        group = chunks[lo:lo + BATCH // CHUNK]
        for k in arrays:
            arrays[k].append(np.concatenate([g[k] for g in group]))
        ptr.append(ptr[-1] + len(arrays["control_row"][-1]))
    return {k: np.concatenate(v) for k, v in arrays.items()}, np.asarray(ptr, np.int64)


def build_matched_ssl(rng: np.random.Generator):
    parts: list[dict[str, np.ndarray]] = []
    audit = []
    for pair_i, (C, L) in enumerate(zip(CONTROL_CUTS, LATE_CUTS)):
        assert C.weekday() == L.weekday() == THURSDAY
        rc, rl = source_rows(C), source_rows(L)
        cc, bc = selected_counts_and_activity(C, rc)
        cl, bl = selected_counts_and_activity(L, rl)
        ac, al = activity_bin(bc), activity_bin(bl)
        key_c = cc.astype(np.int32) * 16 + ac.astype(np.int32)
        key_l = cl.astype(np.int32) * 16 + al.astype(np.int32)
        chosen_c, chosen_l = [], []
        for key in np.intersect1d(np.unique(key_c), np.unique(key_l)):
            ic = np.flatnonzero(key_c == key)
            il = np.flatnonzero(key_l == key)
            rng.shuffle(ic); rng.shuffle(il)
            n = min(len(ic), len(il))
            if n:
                chosen_c.append(ic[:n]); chosen_l.append(il[:n])
        ic = np.concatenate(chosen_c); il = np.concatenate(chosen_l)
        assert np.array_equal(cc[ic], cl[il])
        assert np.array_equal(ac[ic], al[il])
        part = dict(
            control_cut=np.full(len(ic), seq.day_index(C), np.int16),
            control_row=rc[ic].astype(np.int32),
            late_cut=np.full(len(il), seq.day_index(L), np.int16),
            late_row=rl[il].astype(np.int32),
            token_count=cc[ic].astype(np.int16),
            activity_bin=ac[ic].astype(np.int8),
        )
        parts.append(part)
        audit.append(dict(pair=pair_i, control=C, late=L, n_control=len(rc), n_late=len(rl),
                          n_matched=len(ic), matched_share_control=len(ic) / len(rc),
                          matched_share_late=len(il) / len(rl)))
        log(f"matched {C} -> {L}: {len(ic):,} exact token/activity pairs")
    arrays, ptr = _chunks_from_pairs(parts, rng)
    return arrays, ptr, audit


def _direct_chunks(cuts: tuple[dt.date, ...], ci: np.ndarray, ri: np.ndarray,
                   zy: np.ndarray, n_steps: int, rng: np.random.Generator):
    chunks: list[np.ndarray] = []
    for k in range(len(cuts)):
        idx = np.flatnonzero(ci == k)
        rng.shuffle(idx)
        chunks += [idx[i:i + CHUNK] for i in range(0, len(idx), CHUNK)]
    chunks = [chunks[i] for i in rng.permutation(len(chunks))]
    need = n_steps * (BATCH // CHUNK)
    assert len(chunks) > need + math.ceil(50_000 / CHUNK), "not enough clean rows"
    selected, ptr = [], [0]
    for lo in range(0, need, BATCH // CHUNK):
        s = np.concatenate(chunks[lo:lo + BATCH // CHUNK])
        selected.append(s); ptr.append(ptr[-1] + len(s))
    s = np.concatenate(selected)
    hold = np.concatenate(chunks[need:need + math.ceil(50_000 / CHUNK)])[:50_000]
    cut_day = np.asarray([seq.day_index(cuts[int(k)]) for k in ci[s]], np.int16)
    hold_cut = np.asarray([seq.day_index(cuts[int(k)]) for k in ci[hold]], np.int16)
    return dict(direct_cut=cut_day, direct_row=ri[s].astype(np.int32),
                direct_y=zy[s].astype(np.float32), direct_batch_ptr=np.asarray(ptr, np.int64),
                holdout_cut=hold_cut, holdout_row=ri[hold].astype(np.int32),
                holdout_y=zy[hold].astype(np.float32))


def lr_plan(cfg: dict[str, Any], total: int) -> np.ndarray:
    step = np.arange(total, dtype=np.float64)
    return cfg["lr"] * (np.minimum(1.0, (step + 1) / cfg["warmup"])
                        * 0.5 * (1.0 + np.cos(np.pi * np.minimum(1.0, step / total))))


def make_mask(cnt: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros((len(cnt), N_TOK), bool)
    for i, n0 in enumerate(cnt.astype(int)):
        if n0 <= 0:
            continue
        k = max(1, int(math.floor(MASK_RATE * n0 + 0.5)))
        out[i, rng.choice(n0, size=min(k, n0), replace=False)] = True
    return out


def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, size=min(n, k), replace=False)


def build_audit_samples(arrays: dict[str, np.ndarray], clean: tuple[dt.date, ...],
                        ci: np.ndarray, ri: np.ndarray, rng: np.random.Generator):
    out: dict[str, np.ndarray] = {}
    ia = _sample_indices(len(ri), AUDIT_SAMPLE, rng)
    out["A_cut"] = np.asarray([seq.day_index(clean[int(k)]) for k in ci[ia]], np.int16)
    out["A_row"] = ri[ia].astype(np.int32)
    for name, ck, rk in [("CONTROL", "control_cut", "control_row"),
                         ("LATE", "late_cut", "late_row")]:
        ix = _sample_indices(len(arrays[rk]), AUDIT_SAMPLE, rng)
        out[f"{name}_cut"] = arrays[ck][ix].astype(np.int16)
        out[f"{name}_row"] = arrays[rk][ix].astype(np.int32)
    rp = source_rows(CUTOFF_TEST)
    ip = _sample_indices(len(rp), AUDIT_SAMPLE, rng)
    out["PRODUCTION_cut"] = np.full(len(ip), seq.day_index(CUTOFF_TEST), np.int16)
    out["PRODUCTION_row"] = rp[ip].astype(np.int32)
    return out


def _history_features(cut_day: np.ndarray, rows: np.ndarray, rng: np.random.Generator):
    x, ev_day, _, _ = etx.events()
    scale = seq.panel()[3].astype(np.float32)
    out = np.empty((len(rows), len(FEATURE_NAMES)), np.float32)
    token_values, gap_values = [], []
    hist_token_sample = set(_sample_indices(len(rows), TOKEN_SAMPLE_HISTORIES, rng).tolist())
    for lo in range(0, len(rows), 4_000):
        hi = min(lo + 4_000, len(rows))
        idx_all = np.zeros((hi - lo, N_TOK), np.int32)
        cnt_all = np.zeros(hi - lo, np.int32)
        for d in np.unique(cut_day[lo:hi]):
            m = cut_day[lo:hi] == d
            idx, cnt = etx.select(date_from_day(int(d)), rows[lo:hi][m], N_TOK,
                                  depth_clip=DEPTH_CAP)
            idx_all[m], cnt_all[m] = idx, cnt
        pos = np.arange(N_TOK)[None, :]
        mask = pos < cnt_all[:, None]
        vals = x[idx_all].astype(np.float32)
        vals *= mask[:, :, None]
        norm = vals * scale[None, None, :]
        days = ev_day[idx_all].astype(np.int32)
        safe_last = np.maximum(cnt_all - 1, 0)
        last = days[np.arange(len(days)), safe_last]
        rec = cut_day[lo:hi].astype(np.int32) - last
        rec[cnt_all == 0] = DEPTH_CAP
        gaps = np.diff(days, axis=1).astype(np.float32)
        gmask = np.arange(N_TOK - 1)[None, :] < np.maximum(cnt_all - 1, 0)[:, None]
        gn = np.maximum(cnt_all - 1, 1).astype(np.float32)
        gsum = (gaps * gmask).sum(1)
        gmean = gsum / gn
        gvar = (((gaps - gmean[:, None]) ** 2) * gmask).sum(1) / gn
        gmax = np.where(gmask, gaps, 0).max(1)
        raw_search = np.expm1(vals[:, :, seq.CHANNELS.index("searches")]).sum(1)
        raw_cart = np.expm1(vals[:, :, seq.CHANNELS.index("to_cart")]).sum(1)
        raw_order = np.expm1(vals[:, :, seq.CHANNELS.index("to_ord")]).sum(1)
        mix_den = np.maximum(raw_search + raw_cart + raw_order, 1e-6)
        buy_rate = vals[:, :, seq.CHANNELS.index("buy")].sum(1) / np.maximum(cnt_all, 1)
        means = norm.sum(1) / np.maximum(cnt_all, 1)[:, None]
        base = np.column_stack([
            cnt_all, cnt_all / DEPTH_CAP, rec, gmean, np.sqrt(gvar), gmax,
            raw_search / mix_den, raw_cart / mix_den, raw_order / mix_den, buy_rate,
        ])
        out[lo:hi] = np.column_stack([base, means]).astype(np.float32)
        chosen = [j - lo for j in range(lo, hi) if j in hist_token_sample]
        if chosen:
            for j in chosen:
                token_values.append(norm[j][mask[j]])
                if cnt_all[j] > 1:
                    gap_values.append(gaps[j, :cnt_all[j] - 1])
    tokens = np.concatenate(token_values) if token_values else np.empty((0, seq.N_CH_STORED))
    gaps = np.concatenate(gap_values) if gap_values else np.empty(0)
    return out, tokens.astype(np.float32), gaps.astype(np.float32)


def _psi(ref: np.ndarray, other: np.ndarray) -> float:
    ref = np.asarray(ref, float); other = np.asarray(other, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, bins=edges)[0] / len(ref)
    b = np.histogram(other, bins=edges)[0] / len(other)
    a = np.maximum(a, 1e-6); b = np.maximum(b, 1e-6)
    return float(np.sum((b - a) * np.log(b / a)))


def _adversarial_auc(X0: np.ndarray, X1: np.ndarray, uid0: np.ndarray,
                     uid1: np.ndarray) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = np.vstack([X0, X1]); y = np.r_[np.zeros(len(X0), np.int8), np.ones(len(X1), np.int8)]
    uid = np.r_[uid0, uid1]
    test = (splitmix64(uid) & np.uint64(3)) == 0
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                           learning_rate=0.05, l2_regularization=5.0,
                                           random_state=SEED)
    model.fit(np.nan_to_num(X[~test], nan=-1.0), y[~test])
    return float(roc_auc_score(y[test], model.predict_proba(np.nan_to_num(X[test], nan=-1.0))[:, 1]))


def prepare_audit() -> dict[str, Any]:
    if PLAN_NPZ.exists() and PLAN_JSON.exists() and AUDIT_NPZ.exists() and AUDIT_JSON.exists():
        log("reuse completed CPU audit/cache")
        return json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    assert not any(p.exists() for p in (PLAN_NPZ, PLAN_JSON, AUDIT_NPZ, AUDIT_JSON)), (
        "partial EXP-056 audit exists; inspect it before retrying")
    OUT.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint_cpu()
    cfg = dict(ck["cfg"])
    assert ck["val"] == START_VAL.isoformat() and cfg["seed"] == SEED
    assert cfg["n_tok"] == N_TOK and cfg["batch"] == BATCH and not cfg.get("compile")
    clean = clean_cutoffs()
    assert seq.day_index(clean[-1]) + 1 == DEPTH_CAP
    rng = np.random.default_rng([SEED, 56])
    paired, ssl_ptr, match_audit = build_matched_ssl(rng)
    n_steps = len(ssl_ptr) - 1
    ci, ri, zy = seq.build_index(list(clean), blocks=1)
    direct = _direct_chunks(clean, ci, ri, zy, n_steps, rng)
    masks = np.asarray(rng.integers(0, 2**62, size=n_steps), np.int64)
    lr = lr_plan(cfg, n_steps)
    mask_hash = hashlib.sha256()
    masked = real = 0
    for i in range(n_steps):
        a, b = int(ssl_ptr[i]), int(ssl_ptr[i + 1])
        m = make_mask(paired["token_count"][a:b], int(masks[i]))
        mask_hash.update(m.tobytes()); masked += int(m.sum()); real += int(paired["token_count"][a:b].sum())
    audit_samples = build_audit_samples(paired, clean, ci, ri, rng)

    plan_arrays = {**paired, "ssl_batch_ptr": ssl_ptr, **direct,
                   "mask_seed": masks, "lr": lr}
    plan_sha = save_npz_new(PLAN_NPZ, plan_arrays)

    audit_arrays: dict[str, np.ndarray] = {**audit_samples}
    source_summaries = {}
    panel_uid = seq.panel()[2]
    for name in ("A", "CONTROL", "LATE", "PRODUCTION"):
        X, token, gaps = _history_features(audit_samples[f"{name}_cut"],
                                           audit_samples[f"{name}_row"], rng)
        audit_arrays[f"{name}_X"] = X
        audit_arrays[f"{name}_uid"] = panel_uid[audit_samples[f"{name}_row"]]
        q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        source_summaries[name] = dict(
            n=len(X), feature_mean=dict(zip(FEATURE_NAMES, X.mean(0).tolist())),
            feature_quantiles={f: np.quantile(X[:, j], q).tolist() for j, f in enumerate(FEATURE_NAMES)},
            normalized_channel_quantiles={c: np.quantile(token[:, j], q).tolist()
                                          for j, c in enumerate(seq.CHANNELS)},
            gap_quantiles=(np.quantile(gaps, q).tolist() if len(gaps) else [None] * len(q)),
            quantile_levels=q)
        log(f"CPU domain features {name}: {len(X):,} histories")
    # Zero is the fixed constant predictor.  It is target-free and gives a
    # non-zero normalizer even for the real-event `present` channel.
    _, exact_token_a, _ = _history_features(audit_samples["A_cut"][:TOKEN_SAMPLE_HISTORIES],
                                             audit_samples["A_row"][:TOKEN_SAMPLE_HISTORIES], rng)
    constant_arr = np.where(np.abs(exact_token_a) < 1,
                            0.5 * exact_token_a**2, np.abs(exact_token_a) - 0.5).mean(0)
    constant_arr = np.maximum(constant_arr, 1e-6).astype(np.float32)
    audit_arrays["constant_smoothl1"] = constant_arr
    audit_sha = save_npz_new(AUDIT_NPZ, audit_arrays)

    uid_v, _ = fixed_validation(False)
    tokenizer_policy = dict(depth_clip=DEPTH_CAP, query_depth_cap=DEPTH_CAP,
                            query_weekday="Thursday", n_tok=N_TOK,
                            normalized_behavior="seq_scale_v1 reciprocal RMS; no centering",
                            token_time_features="never masked")
    meta = dict(
        experiment=EXP, base_head=os.popen("git rev-parse HEAD").read().strip(),
        checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256_file(CHECKPOINT),
        checkpoint_state_sha256=state_dict_hash(ck["state"]), checkpoint_cfg=cfg,
        checkpoint_val=ck["val"], clean_cutoffs=iso_days(clean),
        clean_target_latest=(clean[-1] + dt.timedelta(days=30)).isoformat(),
        control_cutoffs=iso_days(CONTROL_CUTS), late_cutoffs=iso_days(LATE_CUTS),
        production_diagnostic_cutoff=CUTOFF_TEST.isoformat(),
        primary_validation=PRIMARY_VAL.isoformat(), validation_n=len(uid_v),
        validation_order_sha256=sha256_array(uid_v), tokenizer_policy=tokenizer_policy,
        matching=dict(exact_token_count=True, activity_definition="buy event-days in capped history",
                      activity_edges=ACTIVITY_EDGES.tolist(), paired_dates=True,
                      per_pair=match_audit),
        n_ssl_examples=len(paired["late_row"]), n_optimizer_steps=n_steps,
        ssl_batch_shapes=np.diff(ssl_ptr).tolist(), direct_batch_shapes=np.diff(direct["direct_batch_ptr"]).tolist(),
        masked_tokens=masked, real_tokens=real, realized_mask_rate=masked / real,
        mask_plan_sha256=mask_hash.hexdigest(), lr_plan_sha256=sha256_array(lr),
        direct_plan_sha256=sha256_json({k: sha256_array(direct[k]) for k in direct}),
        plan_file_sha256=plan_sha, audit_file_sha256=audit_sha,
        constant_normalization="per-channel SmoothL1 of fixed zero predictor on supervised-era real tokens",
        constant_smoothl1=constant_arr.tolist(), source_sha256=source_hashes(),
        forbidden=dict(ssl_targets=False, explicit_cutoff_classifier_feature=False,
                       query_weekday_classifier_feature=False, raw_depth_classifier_feature=False,
                       user_id_classifier_feature=False, submission_path=False),
    )
    write_json(PLAN_JSON, meta)
    domain_cpu = dict(experiment=EXP, sources=source_summaries,
                      classifier_features=FEATURE_NAMES,
                      classifier_forbidden=["cutoff", "weekday", "raw_depth", "user_id"])
    write_json(AUDIT_JSON, domain_cpu)
    log(f"CPU audit/cache complete: {len(paired['late_row']):,} paired examples, {n_steps:,} steps")
    return meta


# ============================================================================ GPU policy / model
def configure_determinism(seed: int = SEED) -> dict[str, Any]:
    import random
    import torch

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                bf16=True, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32, eager=True, workers=1,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"])


def rng_hash() -> str:
    import random
    import torch
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode())
    h.update(np.random.get_state()[1].tobytes())
    h.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for s in torch.cuda.get_rng_state_all():
            h.update(s.cpu().numpy().tobytes())
    return h.hexdigest()


def load_model(model_name: str):
    import torch
    if model_name == "BASE":
        data = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    else:
        p = OUT / "arms" / model_name / "model.pt"
        assert p.exists(), f"missing completed arm model {p}"
        data = torch.load(p, map_location="cpu", weights_only=False)
    cfg = dict(data["cfg"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = etx.build_model(cfg).to(dev)
    model.load_state_dict(data["state"])
    tk = etx.Tokenizer(dev)
    tk.depth_cap = DEPTH_CAP
    return model, tk, cfg, dev, data


def forward_parts(model: Any, tok: Any, static: Any, age: Any, n: Any,
                  direct: bool = True):
    """Exact ETX forward with event hidden states and the pooled 3*d embedding exposed."""
    import torch
    B, K, _ = tok.shape
    d = model.cls.numel()
    ev = torch.arange(K, device=tok.device).unsqueeze(0) < n.unsqueeze(1)
    h = torch.zeros(B, K + 1, d, dtype=tok.dtype, device=tok.device)
    h[:, :K] = model.tok(tok) * ev.unsqueeze(-1)
    qtok = (model.cls + model.static(static)).unsqueeze(1)
    h = h.scatter(1, n.view(B, 1, 1).expand(B, 1, d), qtok.to(h.dtype))
    a = torch.zeros(B, K + 1, dtype=age.dtype, device=age.device)
    a[:, :K] = age * ev
    a = a / etx.TAU_UNIT
    for block in model.blocks:
        h = block(h, a)
    h = model.norm(h)
    zq = h.gather(1, n.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
    zl = h.gather(1, (n - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, d)).squeeze(1)
    w = ev.to(h.dtype).unsqueeze(-1)
    zm = (h[:, :K] * w).sum(1) / w.sum(1).clamp_min(1.0)
    pooled = torch.cat([zq, zm, zl], dim=1)
    z = model.head(pooled).squeeze(1) if direct else None
    return z, pooled, h[:, :K], ev


def freeze_direct_head(model: Any) -> None:
    for p in model.head.parameters():
        p.requires_grad_(False)


def build_ssl_modules(model: Any, cfg: dict[str, Any], dev: Any):
    import torch
    from torch import nn
    # Re-seeding here makes the reconstruction initialization independent of
    # how many RNG draws model construction happened to consume.
    torch.manual_seed(SEED + 5600)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 5600)
    recon = nn.Sequential(nn.Linear(cfg["d_model"], cfg["d_model"]), nn.GELU(),
                          nn.Linear(cfg["d_model"], seq.N_CH_STORED)).to(dev)
    mask_value = nn.Parameter(torch.zeros(seq.N_CH_STORED, device=dev))
    return recon, mask_value


def trainable_named(model: Any, recon: Any, mask_value: Any) -> list[tuple[str, Any]]:
    out = [(f"encoder.{n}", p) for n, p in model.named_parameters() if p.requires_grad]
    out += [(f"reconstruction.{n}", p) for n, p in recon.named_parameters()]
    out.append(("mask_value", mask_value))
    return out


def build_optimizer(model: Any, recon: Any, mask_value: Any, cfg: dict[str, Any]):
    import torch
    named = trainable_named(model, recon, mask_value)
    decay = [(n, p) for n, p in named if p.dim() > 1]
    tau = [(n, p) for n, p in named if "log_m" in n]
    tau_ids = {id(p) for _, p in tau}
    decay_ids = {id(p) for _, p in decay}
    nodecay = [(n, p) for n, p in named if id(p) not in decay_ids and id(p) not in tau_ids]
    groups = [dict(params=[p for _, p in decay], weight_decay=cfg["wd"], lr_mult=1.0),
              dict(params=[p for _, p in nodecay], weight_decay=0.0, lr_mult=1.0),
              dict(params=[p for _, p in tau], weight_decay=0.0, lr_mult=10.0)]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.98))
    return opt, dict(named)


def optimizer_hash(opt: Any, named: dict[str, Any]) -> str:
    h = hashlib.sha256()
    reverse = {id(p): n for n, p in named.items()}
    for gi, group in enumerate(opt.param_groups):
        meta = {k: v for k, v in group.items() if k != "params"}
        h.update(json.dumps(jsonable(meta), sort_keys=True).encode())
        for p in group["params"]:
            name = reverse[id(p)]
            h.update(f"{gi}:{name}".encode())
            for key in sorted(opt.state[p]):
                value = opt.state[p][key]
                if hasattr(value, "detach"):
                    _hash_tensor(h, f"{name}:{key}", value)
                else:
                    h.update(repr(value).encode())
    return h.hexdigest()


def combined_state_hash(model: Any, recon: Any, mask_value: Any) -> str:
    h = hashlib.sha256()
    for prefix, state in [("model", model.state_dict()), ("recon", recon.state_dict())]:
        for name in sorted(state):
            _hash_tensor(h, f"{prefix}.{name}", state[name])
    _hash_tensor(h, "mask_value", mask_value)
    return h.hexdigest()


def _indices_for_batch(cut_day: np.ndarray, rows: np.ndarray):
    idx = np.zeros((len(rows), N_TOK), np.int32)
    cnt = np.zeros(len(rows), np.int32)
    for d in np.unique(cut_day):
        m = cut_day == d
        ii, cc = etx.select(date_from_day(int(d)), rows[m], N_TOK, depth_clip=DEPTH_CAP)
        idx[m], cnt[m] = ii, cc
    return idx, cnt, cut_day.astype(np.int32)


def _tokenize(tk: Any, cut_day: np.ndarray, rows: np.ndarray, dev: Any):
    import torch
    idx, cnt, cd = _indices_for_batch(cut_day, rows)
    ti = torch.from_numpy(idx).to(dev, non_blocking=True)
    tc = torch.from_numpy(cnt).to(dev, non_blocking=True)
    td = torch.from_numpy(cd).to(dev, non_blocking=True)
    tok, static, age, n = tk(ti, tc, td)
    return tok, static, age, n, cnt


def predict_direct(model: Any, tk: Any, cfg: dict[str, Any], cut_day: np.ndarray,
                   rows: np.ndarray, dev: Any, batch: int = BATCH) -> np.ndarray:
    import torch
    model.eval()
    out = np.empty(len(rows), np.float32)
    with torch.no_grad():
        for lo in range(0, len(rows), batch):
            hi = min(lo + batch, len(rows))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                tok, st, age, n, _ = _tokenize(tk, cut_day[lo:hi], rows[lo:hi], dev)
                z, _, _, _ = forward_parts(model, tok, st, age, n, direct=True)
            out[lo:hi] = z.float().cpu().numpy()
    model.train()
    return out


def _snapshot(model: Any, recon: Any, mask_value: Any, opt: Any,
              named: dict[str, Any], step: int) -> dict[str, Any]:
    return dict(step=step, combined_state_sha256=combined_state_hash(model, recon, mask_value),
                model_sha256=module_hash(model), direct_head_sha256=module_hash(model.head),
                reconstruction_sha256=module_hash(recon), optimizer_sha256=optimizer_hash(opt, named),
                rng_sha256=rng_hash())


def run_arm(arm: str, destination: Path, max_steps: int | None = None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    assert arm in ("CONTROL", "LATE")
    assert PLAN_NPZ.exists() and PLAN_JSON.exists() and AUDIT_NPZ.exists()
    if destination.exists():
        result = destination / "result.json"
        assert result.exists(), f"partial run directory {destination}"
        return json.loads(result.read_text(encoding="utf-8"))
    work = destination.with_name(destination.name + ".tmp")
    assert not work.exists(), f"partial temporary directory {work}"
    work.mkdir(parents=True)
    env = configure_determinism(SEED)
    assert torch.cuda.is_available(), "EXP-056 pilot requires CUDA"
    # npz members are otherwise decompressed afresh on every ``plan[key]``;
    # a 4k-step loop would repeatedly allocate/read the multi-million-row
    # arrays.  Materialize the frozen plan once per arm.
    with np.load(PLAN_NPZ) as frozen:
        plan = {k: frozen[k] for k in frozen.files}
    meta = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    assert sha256_file(PLAN_NPZ) == meta["plan_file_sha256"]
    model, tk, cfg, dev, data = load_model("BASE")
    assert state_dict_hash(data["state"]) == meta["checkpoint_state_sha256"]
    freeze_direct_head(model)
    recon, mask_value = build_ssl_modules(model, cfg, dev)
    opt, named = build_optimizer(model, recon, mask_value, cfg)
    direct_initial = module_hash(model.head)
    initial = _snapshot(model, recon, mask_value, opt, named, 0)
    n_total = len(plan["mask_seed"])
    n_steps = min(n_total, int(max_steps)) if max_steps else n_total
    snapshot_at = {0, 1, min(100, n_steps), max(1, n_steps // 2), n_steps}
    snapshots = [initial]
    constant = torch.from_numpy(np.load(AUDIT_NPZ)["constant_smoothl1"]).to(dev)
    ssl_row_key = "control_row" if arm == "CONTROL" else "late_row"
    ssl_cut_key = "control_cut" if arm == "CONTROL" else "late_cut"
    ssl_ptr, direct_ptr = plan["ssl_batch_ptr"], plan["direct_batch_ptr"]
    mask_digest = hashlib.sha256()
    run_direct = 0.0; run_ssl = 0.0; seen_d = 0; seen_m = 0
    first_ssl, last_ssl = [], []
    t_start = time.time()
    model.train(); recon.train()
    for step0 in range(n_steps):
        sa, sb = int(ssl_ptr[step0]), int(ssl_ptr[step0 + 1])
        da, db = int(direct_ptr[step0]), int(direct_ptr[step0 + 1])
        lr0 = float(plan["lr"][step0])
        for group in opt.param_groups:
            group["lr"] = lr0 * float(group.get("lr_mult", 1.0))
        opt.zero_grad(set_to_none=True)
        # Accumulate the two terms before the shared clip/step, but release the
        # direct graph before constructing the SSL graph.  On an 8 GB card this
        # preserves the exact batch/objective contract without paging two ETX
        # graphs through system memory at once.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            td, sd, ad, nd, _ = _tokenize(tk, plan["direct_cut"][da:db],
                                           plan["direct_row"][da:db], dev)
            z_direct, _, _, _ = forward_parts(model, td, sd, ad, nd, direct=True)
            y_direct = torch.from_numpy(plan["direct_y"][da:db]).to(dev)
            loss_direct = F.mse_loss(z_direct, y_direct)
        loss_direct.backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ts, ss, ass, ns, cnt = _tokenize(tk, plan[ssl_cut_key][sa:sb],
                                              plan[ssl_row_key][sa:sb], dev)
            expected = np.asarray(plan["token_count"][sa:sb], np.int32)
            assert np.array_equal(cnt, expected), "paired token counts drifted"
            mask_np = make_mask(cnt, int(plan["mask_seed"][step0]))
            mask_digest.update(mask_np.tobytes())
            mask = torch.from_numpy(mask_np).to(dev)
            target = ts[:, :, :seq.N_CH_STORED].detach().clone()
            masked = ts.clone()
            masked[:, :, :seq.N_CH_STORED] = torch.where(
                mask.unsqueeze(-1), mask_value.view(1, 1, -1),
                masked[:, :, :seq.N_CH_STORED])
            _, _, hidden, _ = forward_parts(model, masked, ss, ass, ns, direct=False)
            pred = recon(hidden)
            per = F.smooth_l1_loss(pred[mask], target[mask], reduction="none", beta=1.0)
            loss_ssl = (per / constant.view(1, -1)).mean()
        (LAMBDA_SSL * loss_ssl).backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in named.items()], 1.0)
        opt.step()
        ld, ls = float(loss_direct.detach()), float(loss_ssl.detach())
        run_direct += ld * (db - da); seen_d += db - da
        run_ssl += ls * int(mask_np.sum()); seen_m += int(mask_np.sum())
        if step0 < 20: first_ssl.append(ls)
        if step0 >= max(0, n_steps - 20): last_ssl.append(ls)
        step = step0 + 1
        if step in snapshot_at:
            snapshots.append(_snapshot(model, recon, mask_value, opt, named, step))
        if step % 500 == 0 or step == n_steps:
            log(f"{arm} step {step:,}/{n_steps:,}: direct {run_direct / seen_d:.5f}, "
                f"ssl {run_ssl / seen_m:.5f}")
    assert mask_digest.hexdigest() == hashlib.sha256(
        b"".join(make_mask(plan["token_count"][int(ssl_ptr[i]):int(ssl_ptr[i + 1])],
                            int(plan["mask_seed"][i])).tobytes() for i in range(n_steps))).hexdigest()
    if n_steps == n_total:
        assert mask_digest.hexdigest() == meta["mask_plan_sha256"], "materialized mask plan drifted"
    assert module_hash(model.head) == direct_initial, "frozen direct head changed"
    uid_v, rows_v = fixed_validation(False)
    probe_n = min(4096, len(rows_v))
    probe_cut = np.full(probe_n, seq.day_index(PRIMARY_VAL), np.int16)
    z_probe = predict_direct(model, tk, cfg, probe_cut, rows_v[:probe_n], dev)
    hcut, hrow, hy = plan["holdout_cut"], plan["holdout_row"], plan["holdout_y"]
    z_hold = predict_direct(model, tk, cfg, hcut, hrow, dev)
    hold_mse = float(np.mean((hy - np.maximum(z_hold, 0.0)) ** 2))
    payload = dict(state=model.state_dict(), cfg=cfg, reconstruction=recon.state_dict(),
                   mask_value=mask_value.detach().cpu(), arm=arm, steps=n_steps)
    model_tmp = work / "model.pt.tmp"
    torch.save(payload, model_tmp); os.replace(model_tmp, work / "model.pt")
    result = dict(
        experiment=EXP, arm=arm, n_steps=n_steps, planned_steps=n_total,
        full_epoch=n_steps == n_total, environment=env, runtime_s=time.time() - t_start,
        source_corridor=(iso_days(CONTROL_CUTS) if arm == "CONTROL" else iso_days(LATE_CUTS)),
        common=dict(checkpoint_sha256=meta["checkpoint_sha256"],
                    checkpoint_state_sha256=meta["checkpoint_state_sha256"],
                    plan_sha256=meta["plan_file_sha256"], direct_plan_sha256=meta["direct_plan_sha256"],
                    mask_plan_sha256=meta["mask_plan_sha256"], lr_plan_sha256=meta["lr_plan_sha256"],
                    initial_combined_state_sha256=initial["combined_state_sha256"],
                    initial_optimizer_sha256=initial["optimizer_sha256"],
                    initial_rng_sha256=initial["rng_sha256"],
                    direct_head_sha256=direct_initial, lambda_ssl=LAMBDA_SSL,
                    mask_rate=MASK_RATE, depth_cap=DEPTH_CAP, weekday="Thursday"),
        train=dict(direct_mse=run_direct / seen_d, normalized_ssl_loss=run_ssl / seen_m,
                   first20_ssl=float(np.mean(first_ssl)), last20_ssl=float(np.mean(last_ssl)),
                   masked_tokens=seen_m),
        clean_holdout_mse=hold_mse, probe_prediction_sha256=sha256_array(z_probe),
        probe_mean=float(z_probe.mean()), final_combined_state_sha256=combined_state_hash(model, recon, mask_value),
        final_model_sha256=module_hash(model), final_reconstruction_sha256=module_hash(recon),
        final_optimizer_sha256=optimizer_hash(opt, named), final_rng_sha256=rng_hash(),
        final_direct_head_sha256=module_hash(model.head), snapshots=snapshots,
        model_file_sha256=sha256_file(work / "model.pt"), runtime_source_sha256=source_hashes())
    write_json(work / "result.json", result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, destination)
    return result


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(SEED)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def launch_arm(arm: str, destination: Path, max_steps: int | None = None) -> dict[str, Any]:
    if (destination / "result.json").exists():
        return json.loads((destination / "result.json").read_text(encoding="utf-8"))
    args = [sys.executable, str(Path(__file__).resolve()), "arm", "--arm", arm,
            "--destination", str(destination)]
    if max_steps is not None:
        args += ["--max-steps", str(max_steps)]
    subprocess.run(args, cwd=ROOT, env=child_env(), check=True)
    return json.loads((destination / "result.json").read_text(encoding="utf-8"))


def run_replay() -> dict[str, Any]:
    if REPLAY_JSON.exists():
        return json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    r1 = launch_arm("CONTROL", OUT / "replay_v2" / "run1", 100)
    r2 = launch_arm("CONTROL", OUT / "replay_v2" / "run2", 100)
    keys = ["final_combined_state_sha256", "final_model_sha256",
            "final_reconstruction_sha256", "final_optimizer_sha256", "final_rng_sha256",
            "probe_prediction_sha256"]
    equal = {k: r1[k] == r2[k] for k in keys}
    snap_equal = r1["snapshots"] == r2["snapshots"]
    result = dict(experiment=EXP, steps=100, hashes_equal=equal,
                  snapshots_equal=snap_equal, pass_replay=all(equal.values()) and snap_equal,
                  run1=str(OUT / "replay_v2" / "run1"), run2=str(OUT / "replay_v2" / "run2"),
                  backward_policy="direct backward + 0.25*SSL backward; shared clip/optimizer step")
    assert result["pass_replay"], "deterministic 100-step replay failed"
    write_json(REPLAY_JSON, result)
    return result


def run_pilot() -> dict[str, Any]:
    replay = run_replay()
    assert replay["pass_replay"]
    control = launch_arm("CONTROL", OUT / "arms" / "CONTROL")
    late = launch_arm("LATE", OUT / "arms" / "LATE")
    common_keys = ["checkpoint_sha256", "checkpoint_state_sha256", "plan_sha256",
                   "direct_plan_sha256", "mask_plan_sha256", "lr_plan_sha256",
                   "initial_combined_state_sha256", "initial_optimizer_sha256",
                   "initial_rng_sha256", "direct_head_sha256", "lambda_ssl", "mask_rate",
                   "depth_cap", "weekday"]
    assert all(control["common"][k] == late["common"][k] for k in common_keys)
    assert control["n_steps"] == late["n_steps"]
    return dict(control=control, late=late)


# ============================================================================ embedding/domain audit
def compute_embeddings(model_name: str) -> dict[str, Any]:
    import torch
    out_path = OUT / f"embedding_{model_name}.npz"
    meta_path = RESULTS / f"embedding_{model_name}.json"
    if out_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    assert not out_path.exists() and not meta_path.exists()
    configure_determinism(SEED)
    assert torch.cuda.is_available(), "embedding audit requires CUDA"
    model, tk, cfg, dev, _ = load_model(model_name)
    model.eval()
    audit = np.load(AUDIT_NPZ)
    arrays: dict[str, np.ndarray] = {}
    source_meta = {}
    for source in ("A", "CONTROL", "LATE", "PRODUCTION"):
        # Starting-model support policy: all query context remains Thursday.
        # The production diagnostic cutoff is Friday, hence the fixed -1 shift.
        tk.cdow_shift = float(THURSDAY - CUTOFF_TEST.weekday()) if source == "PRODUCTION" else 0.0
        cut = audit[f"{source}_cut"][:EMBED_SAMPLE]
        rows = audit[f"{source}_row"][:EMBED_SAMPLE]
        emb = np.empty((len(rows), 3 * cfg["d_model"]), np.float32)
        with torch.no_grad():
            for lo in range(0, len(rows), BATCH):
                hi = min(lo + BATCH, len(rows))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok, st, age, n, _ = _tokenize(tk, cut[lo:hi], rows[lo:hi], dev)
                    _, pooled, _, _ = forward_parts(model, tok, st, age, n, direct=False)
                emb[lo:hi] = pooled.float().cpu().numpy()
        mean = emb.astype(np.float64).mean(0)
        cov = np.cov(emb.astype(np.float64), rowvar=False)
        arrays[f"{source}_mean"] = mean.astype(np.float32)
        arrays[f"{source}_cov"] = cov.astype(np.float32)
        arrays[f"{source}_sample"] = emb[:MMD_SAMPLE]
        source_meta[source] = dict(n=len(emb), mean_norm=float(np.linalg.norm(mean)),
                                   trace_cov=float(np.trace(cov)))
        log(f"embeddings {model_name}/{source}: {len(emb):,}")
    file_sha = save_npz_new(out_path, arrays)
    meta = dict(experiment=EXP, model=model_name, input_only=True, direct_head_called=False,
                depth_cap=DEPTH_CAP, query_weekday="Thursday", sources=source_meta,
                file_sha256=file_sha, model_state_sha256=module_hash(model))
    write_json(meta_path, meta)
    return meta


def rbf_mmd(X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import pairwise_distances
    X = np.asarray(X[:MMD_SAMPLE], np.float64); Y = np.asarray(Y[:MMD_SAMPLE], np.float64)
    Z = np.vstack([X[:500], Y[:500]])
    d2 = pairwise_distances(Z, metric="sqeuclidean")
    positive = d2[d2 > 0]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    Kxx = np.exp(-pairwise_distances(X, metric="sqeuclidean") / (2 * sigma2))
    Kyy = np.exp(-pairwise_distances(Y, metric="sqeuclidean") / (2 * sigma2))
    Kxy = np.exp(-pairwise_distances(X, Y, metric="sqeuclidean") / (2 * sigma2))
    return dict(mmd2=float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()), sigma2=sigma2)


def finish_domain_audit() -> dict[str, Any]:
    if DOMAIN_JSON.exists():
        return json.loads(DOMAIN_JSON.read_text(encoding="utf-8"))
    assert (OUT / "embedding_BASE.npz").exists(), "run BASE embedding audit first"
    audit = np.load(AUDIT_NPZ)
    pairs = [("A", "LATE"), ("A", "PRODUCTION"), ("CONTROL", "LATE")]
    pair_rows = []
    for a, b in pairs:
        X0, X1 = audit[f"{a}_X"], audit[f"{b}_X"]
        psi = {f: _psi(X0[:, j], X1[:, j]) for j, f in enumerate(FEATURE_NAMES)}
        auc = _adversarial_auc(X0, X1, audit[f"{a}_uid"], audit[f"{b}_uid"])
        pair_rows.append(dict(reference=a, other=b, adversarial_auc=auc,
                              psi_mean=float(np.mean(list(psi.values()))),
                              psi_max=float(np.max(list(psi.values()))), psi=psi))
    emb = np.load(OUT / "embedding_BASE.npz")
    mmd = {f"{a}_vs_{b}": rbf_mmd(emb[f"{a}_sample"], emb[f"{b}_sample"])
           for a, b in pairs}
    result = dict(experiment=EXP, target_free=True, plan_frozen_before_audit=True,
                  classifier_features=FEATURE_NAMES,
                  classifier_excluded=["explicit cutoff date", "query weekday", "raw depth", "user_id"],
                  pairs=pair_rows, frozen_etx_embedding_mmd=mmd,
                  frozen_embedding_file_sha256=sha256_file(OUT / "embedding_BASE.npz"))
    write_json(DOMAIN_JSON, result)
    return result


# ============================================================================ endpoint analysis
def auc_positive(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(y) > 0
    return float(roc_auc_score(pos.astype(np.int8), np.asarray(score)))


def metric_record(name: str, y: np.ndarray, z: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    offset, calibrated = calibrate(y, z)
    zc = np.maximum(np.asarray(z, float) + offset, 0.0)
    ly = np.log1p(y)
    pos = y > 0
    err = (ly - zc) ** 2
    record = dict(model=name, n=len(y), rmsle_raw=rmsle_z(y, z), rmsle_cal=float(np.sqrt(err.mean())),
                  offset=offset, auc=auc_positive(y, zc),
                  rmsle_zero=float(np.sqrt(err[~pos].mean())),
                  rmsle_positive=float(np.sqrt(err[pos].mean())),
                  mse_zero_contribution=float(err[~pos].sum() / len(err)),
                  mse_positive_contribution=float(err[pos].sum() / len(err)),
                  mean_z_raw=float(np.mean(z)), mean_z_cal=float(np.mean(zc)))
    return record, zc


def load_fold_prediction(name: str, uid: np.ndarray, y: np.ndarray) -> np.ndarray:
    exact = ARTIFACTS / f"oof_{name}-V1016.npz"
    p = exact if exact.exists() else ARTIFACTS / f"oof_{name}.npz"
    assert p.exists(), f"missing fixed-slot component {p}"
    d = np.load(p)
    u = np.asarray(d["user_id"])
    z = np.asarray(d["z"], float)
    yy = np.asarray(d["y"], float)
    if "cutoff" in d.files and len(u) != len(uid):
        m = np.asarray(d["cutoff"], dtype="U10") == PRIMARY_VAL.isoformat()
        u, z, yy = u[m], z[m], yy[m]
    assert len(u) == len(uid)
    if not np.array_equal(u, uid):
        order = np.argsort(u)
        u, z, yy = u[order], z[order], yy[order]
    assert np.array_equal(u, uid), f"{name}: validation order/rows differ"
    assert np.array_equal(yy.astype(np.float32), y.astype(np.float32)), (
        f"{name}: validation target differs")
    return z


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    assert rows, f"empty table {path}"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows([{k: jsonable(r[k]) for k in fields} for r in rows])


def _validation_predictions() -> dict[str, np.ndarray]:
    import torch
    path = OUT / "validation_predictions.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    uid, rows = fixed_validation(False)
    cut = np.full(len(rows), seq.day_index(PRIMARY_VAL), np.int16)
    predictions: dict[str, np.ndarray] = {"user_id": uid}
    plan = np.load(PLAN_NPZ)
    for name in ("BASE", "CONTROL", "LATE"):
        model, tk, cfg, dev, _ = load_model(name)
        predictions[name] = predict_direct(model, tk, cfg, cut, rows, dev)
        predictions[f"{name}_holdout"] = predict_direct(
            model, tk, cfg, plan["holdout_cut"], plan["holdout_row"], dev)
        del model, tk
        torch.cuda.empty_cache()
        log(f"validation prediction {name}: mean {predictions[name].mean():.5f}")
    save_npz_new(path, predictions)
    return predictions


def _segment_masks(uid: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    import polars as pl
    f = pl.read_parquet(DATA_PROCESSED / f"feat_{PRIMARY_VAL.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id", "rec_buy", "w180_days_buy"])
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    assert f.height == len(uid)
    rec = f["rec_buy"].to_numpy().astype(float)
    buy = f["w180_days_buy"].to_numpy().astype(float)
    known = np.isfinite(rec)
    return {
        "all": np.ones(len(uid), bool),
        "rec_buy_15_60": known & (rec >= 15) & (rec <= 60),
        "w180_days_buy_2_15": (buy >= 2) & (buy <= 15),
        "intersection": known & (rec >= 15) & (rec <= 60) & (buy >= 2) & (buy <= 15),
        "never_buyer": ~known,
        "frequent": buy >= 16,
        "y_zero": y == 0,
        "y_positive": y > 0,
    }


def _embedding_after_table() -> list[dict[str, Any]]:
    rows = []
    for name in ("BASE", "CONTROL", "LATE"):
        p = OUT / f"embedding_{name}.npz"
        assert p.exists(), f"missing {p}; run embed --model {name}"
        d = np.load(p)
        cl = rbf_mmd(d["CONTROL_sample"], d["LATE_sample"])
        lp = rbf_mmd(d["LATE_sample"], d["PRODUCTION_sample"])
        rows.append(dict(model=name, mmd2_control_late=cl["mmd2"],
                         sigma2_control_late=cl["sigma2"],
                         mmd2_late_production=lp["mmd2"],
                         sigma2_late_production=lp["sigma2"]))
    return rows


def analyze_results() -> dict[str, Any]:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    import torch
    configure_determinism(SEED)
    assert torch.cuda.is_available(), "endpoint analysis needs CUDA inference"
    replay = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    assert replay["pass_replay"]
    control_run = json.loads((OUT / "arms" / "CONTROL" / "result.json").read_text(encoding="utf-8"))
    late_run = json.loads((OUT / "arms" / "LATE" / "result.json").read_text(encoding="utf-8"))
    assert control_run["full_epoch"] and late_run["full_epoch"]
    pred = _validation_predictions()
    uid, rows, y = fixed_validation(True)
    assert np.array_equal(pred["user_id"], uid)
    standalone_rows: list[dict[str, Any]] = []
    zcal: dict[str, np.ndarray] = {}
    for name in ("BASE", "CONTROL", "LATE"):
        rec, zcal[name] = metric_record(name, y, pred[name])
        standalone_rows.append(rec)
    metrics = {r["model"]: r for r in standalone_rows}

    cap = load_fold_prediction("S1-E03a", uid, y)
    unc = load_fold_prediction("S1-E02", uid, y)
    dist = load_fold_prediction("S1-DIST", uid, y)
    seq_avg = load_fold_prediction("SEQ-AVG3", uid, y)
    fixed = 0.10 * cap + 0.20 * unc + 0.25 * dist + 0.225 * seq_avg
    slot_raw = {
        "NO_ADAPT_SLOT": fixed + 0.225 * pred["BASE"],
        "BASE_SLOT": fixed + 0.225 * pred["CONTROL"],
        "LATE_SLOT": fixed + 0.225 * pred["LATE"],
    }
    slot_rows: list[dict[str, Any]] = []
    slot_cal: dict[str, np.ndarray] = {}
    for name, z in slot_raw.items():
        rec, slot_cal[name] = metric_record(name, y, z)
        slot_rows.append(rec)
    slot_metrics = {r["model"]: r for r in slot_rows}

    ly = np.log1p(y)
    correction_raw = pred["LATE"].astype(float) - pred["CONTROL"].astype(float)
    correction_cal = zcal["LATE"] - zcal["CONTROL"]
    residual_control = ly - zcal["CONTROL"]
    residual_base = ly - zcal["BASE"]
    pair = dict(
        standalone_delta_raw=metrics["LATE"]["rmsle_raw"] - metrics["CONTROL"]["rmsle_raw"],
        standalone_delta_cal=metrics["LATE"]["rmsle_cal"] - metrics["CONTROL"]["rmsle_cal"],
        slot_delta_raw=slot_metrics["LATE_SLOT"]["rmsle_raw"] - slot_metrics["BASE_SLOT"]["rmsle_raw"],
        slot_delta_cal=slot_metrics["LATE_SLOT"]["rmsle_cal"] - slot_metrics["BASE_SLOT"]["rmsle_cal"],
        var_z_late_minus_control=float(np.var(correction_raw)),
        var_calibrated_correction=float(np.var(correction_cal)),
        mean_raw_correction=float(np.mean(correction_raw)),
        mean_calibrated_correction=float(np.mean(correction_cal)),
        corr_predictions=float(np.corrcoef(pred["LATE"], pred["CONTROL"])[0, 1]),
        corr_residuals=float(np.corrcoef(ly - zcal["LATE"], residual_control)[0, 1]),
        corr_correction_control_residual=float(np.corrcoef(correction_cal, residual_control)[0, 1]),
        corr_correction_baseline_residual=float(np.corrcoef(correction_cal, residual_base)[0, 1]))

    half_rows = []
    half = (splitmix64(uid) & np.uint64(1)).astype(np.int8)
    for h in (0, 1):
        m = half == h
        rc = float(np.sqrt(np.mean((ly[m] - zcal["CONTROL"][m]) ** 2)))
        rl = float(np.sqrt(np.mean((ly[m] - zcal["LATE"][m]) ** 2)))
        sb = float(np.sqrt(np.mean((ly[m] - slot_cal["BASE_SLOT"][m]) ** 2)))
        sl = float(np.sqrt(np.mean((ly[m] - slot_cal["LATE_SLOT"][m]) ** 2)))
        half_rows.append(dict(half=h, n=int(m.sum()), control=rc, late=rl,
                              standalone_delta=rl - rc, base_slot=sb, late_slot=sl,
                              slot_delta=sl - sb,
                              residual_alignment=float(np.corrcoef(correction_cal[m], residual_control[m])[0, 1])))

    segment_rows = []
    for name, m in _segment_masks(uid, y).items():
        rc = float(np.sqrt(np.mean((ly[m] - zcal["CONTROL"][m]) ** 2)))
        rl = float(np.sqrt(np.mean((ly[m] - zcal["LATE"][m]) ** 2)))
        sb = float(np.sqrt(np.mean((ly[m] - slot_cal["BASE_SLOT"][m]) ** 2)))
        sl = float(np.sqrt(np.mean((ly[m] - slot_cal["LATE_SLOT"][m]) ** 2)))
        segment_rows.append(dict(segment=name, n=int(m.sum()), share=float(m.mean()),
                                 control=rc, late=rl, standalone_delta=rl - rc,
                                 base_slot=sb, late_slot=sl, slot_delta=sl - sb))

    plan = np.load(PLAN_NPZ)
    hold_y = plan["holdout_y"]
    rehearsal = {}
    for name in ("BASE", "CONTROL", "LATE"):
        z = np.maximum(pred[f"{name}_holdout"], 0.0)
        rehearsal[name] = float(np.mean((hold_y - z) ** 2))
    rehearsal_gate = (rehearsal["CONTROL"] <= rehearsal["BASE"] + 1e-4
                       and rehearsal["LATE"] <= rehearsal["BASE"] + 1e-4)

    embedding_rows = _embedding_after_table()
    stand = pair["standalone_delta_cal"]; slot = pair["slot_delta_cal"]
    halves_gate = all(r["standalone_delta"] < 0 and r["slot_delta"] < 0 for r in half_rows)
    alignment_gate = pair["corr_correction_control_residual"] > 0
    shape_gate = pair["var_calibrated_correction"] > 1e-6 and abs(pair["mean_calibrated_correction"]) < 1e-4
    strong_effect = stand <= -0.001 or slot <= -0.0007
    borderline_effect = ((-0.001 < stand <= -0.0003) or (-0.0007 < slot <= -0.0003))
    reconstruction_improved = (control_run["train"]["last20_ssl"] < control_run["train"]["first20_ssl"]
                               and late_run["train"]["last20_ssl"] < late_run["train"]["first20_ssl"])
    common_gates = halves_gate and alignment_gate and rehearsal_gate and shape_gate
    if strong_effect and common_gates:
        verdict = "STRONG_PASS"
    elif borderline_effect and common_gates:
        verdict = "BORDERLINE"
    else:
        verdict = "REJECT"
    if reconstruction_improved and stand > -0.0003 and slot > -0.0003:
        verdict = "REJECT"
    promote = verdict == "STRONG_PASS"

    _csv(RESULTS / "standalone_metrics.csv", standalone_rows)
    _csv(RESULTS / "fixed_slot_metrics.csv", slot_rows)
    _csv(RESULTS / "user_half_metrics.csv", half_rows)
    _csv(RESULTS / "segments.csv", segment_rows)
    _csv(RESULTS / "embedding_mmd.csv", embedding_rows)
    result = dict(
        experiment=EXP, primary_validation=PRIMARY_VAL, n=len(uid),
        checkpoint_sha256=sha256_file(CHECKPOINT), validation_order_sha256=sha256_array(uid),
        standalone=metrics, fixed_slot=slot_metrics, paired=pair, user_halves=half_rows,
        clean_direct_rehearsal_mse=rehearsal, embedding_mmd=embedding_rows,
        training={"CONTROL": control_run["train"], "LATE": late_run["train"]},
        gates=dict(strong_effect=strong_effect, borderline_effect=borderline_effect,
                   both_halves_correct_sign=halves_gate, positive_residual_alignment=alignment_gate,
                   direct_clean_rehearsal_not_degraded=rehearsal_gate,
                   not_only_level_shift=shape_gate, reconstruction_improved=reconstruction_improved,
                   deterministic_replay=replay["pass_replay"]),
        verdict=verdict, PROMOTE_TO_FULL_FOLDS=("YES" if promote else "NO"),
        prohibited=dict(full_folds_not_run=True, production_prediction_not_run=True,
                        test_prediction_not_run=True, submission_not_created=True, lb_not_queried=True))
    write_json(SUMMARY_JSON, result)

    report = f"""# EXP-056 — LATE-UNLABELED-ETX-ADAPT

## Endpoint

- Verdict: **{verdict}**; `PROMOTE_TO_FULL_FOLDS={'YES' if promote else 'NO'}`.
- LATE−CONTROL standalone: raw **{pair['standalone_delta_raw']:+.6f}**, calibrated **{stand:+.6f}**.
- LATE_SLOT−BASE_SLOT: raw **{pair['slot_delta_raw']:+.6f}**, calibrated **{slot:+.6f}**.
- User halves, standalone: **{half_rows[0]['standalone_delta']:+.6f} / {half_rows[1]['standalone_delta']:+.6f}**; slot: **{half_rows[0]['slot_delta']:+.6f} / {half_rows[1]['slot_delta']:+.6f}**.
- `corr(correction, CONTROL residual)` = **{pair['corr_correction_control_residual']:+.6f}**; `Var(z_late-z_control)` = **{pair['var_z_late_minus_control']:.8f}**.
- Clean direct holdout MSE BASE/CONTROL/LATE = **{rehearsal['BASE']:.6f} / {rehearsal['CONTROL']:.6f} / {rehearsal['LATE']:.6f}**.
- Frozen/adapted embedding MMD is in `embedding_mmd.csv`; full domain audit is in `domain_shift_summary.json`.

## Contract

Both arms used the exact `ETX-01-S42-V0904` checkpoint, support depth **212**, Thursday query context, one materialized epoch, identical direct rows/LR/masks/RNG/head/optimizer initialization and deterministic eager bf16/TF32 CUDA. The SSL objective read only input histories and reconstructed the exact 14 normalized behavioral channels. Validation target was first read by this analysis stage.

No full folds, production/test prediction, submission or LB action was performed.
"""
    report_path = RESULTS / "REPORT.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(report, encoding="utf-8")
    return result


# ============================================================================ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-056 late unlabeled ETX adaptation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("embed")
    p.add_argument("--model", choices=["BASE", "CONTROL", "LATE"], required=True)
    sub.add_parser("domain")
    sub.add_parser("replay")
    sub.add_parser("pilot")
    p = sub.add_parser("arm")
    p.add_argument("--arm", choices=["CONTROL", "LATE"], required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None)
    sub.add_parser("analyze")
    a = ap.parse_args()
    if a.cmd == "audit":
        prepare_audit()
    elif a.cmd == "embed":
        compute_embeddings(a.model)
    elif a.cmd == "domain":
        finish_domain_audit()
    elif a.cmd == "replay":
        run_replay()
    elif a.cmd == "pilot":
        run_pilot()
    elif a.cmd == "arm":
        run_arm(a.arm, a.destination, a.max_steps)
    elif a.cmd == "analyze":
        analyze_results()


if __name__ == "__main__":
    main()
