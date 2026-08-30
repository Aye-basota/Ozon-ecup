"""FRESH-CONTRAST-MOE: residual correction from EXP-032 conditional heads.

The runner reuses the saved EXP-032B group-A predictions and embedding caches.
Only the missing symmetric half is computed: EXTRA targets from hash group A
train a frozen conditional head that predicts group B.  No TCN/ETX is trained.

Run the complete experiment with one command::

    python src/fresh_contrast.py

If validation is REJECT, production inference is deliberately not started.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.block4_saf import _strongest_fold, _strongest_test
from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, SEED, SUBMISSIONS,
                        VAL_FOLDS_S1)
from src.data import sample_submit
from src.features import features_cached, panel_users
from src.seq import fold_cutoffs, load_ckpt, target_at, user_rows
from src.seq_cond import (EXTRA_CUTOFFS, POS_ONLY, collect, fit_head, head_predict,
                          user_group)
from src.validation import calibrate

PREFIX = "FRESH_CONTRAST_MOE"
RESULTS = Path("research/strategies/results/FRESH_CONTRAST")
ALPHAS = np.asarray([0.00, 0.25, 0.50, 0.75, 1.00], dtype=float)
VARIANTS = ("GLOBAL", "HIGH16")
CONTRASTS = ("FRESH", "VOL")
L_STAR = 2.3293
HEAD_SEED = SEED
HEAD_EPOCHS = 4
HEAD_BATCH = 8192
HEAD_LR = 1e-3
HEAD_WD = 1e-2
HEAD_DROPOUT = 0.10
EXTRA_DEPTH_CLIP = 289
T0 = time.time()


def log(*parts) -> None:
    print(f"[{time.time() - T0:7.0f}s]", *parts, flush=True)


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def stable_group(user_ids) -> np.ndarray:
    """The project-wide ``splitmix64(user_id) & 1`` split."""
    return user_group(np.asarray(user_ids))


def validate_crossfit(recipient_uid: np.ndarray, donor_uid: np.ndarray,
                      recipient_group: int, donor_group: int) -> None:
    """Fail if an EXTRA donor can receive a prediction from its own head."""
    recipient_uid = np.asarray(recipient_uid)
    donor_uid = np.asarray(donor_uid)
    if recipient_group == donor_group:
        raise AssertionError("recipient and EXTRA donor groups must differ")
    if np.any(stable_group(recipient_uid) != recipient_group):
        raise AssertionError("recipient contains a user from the wrong hash group")
    if np.any(stable_group(donor_uid) != donor_group):
        raise AssertionError("EXTRA donors contain a user from the wrong hash group")
    overlap = np.intersect1d(np.unique(recipient_uid), np.unique(donor_uid)).size
    if overlap:
        raise AssertionError(f"{overlap} recipients occur among their EXTRA donors")


def merge_crossfit(group: np.ndarray, from_group_b: np.ndarray,
                   from_group_a: np.ndarray) -> np.ndarray:
    """Merge donor-B->recipient-A and donor-A->recipient-B predictions."""
    group = np.asarray(group)
    a = np.asarray(from_group_b)
    b = np.asarray(from_group_a)
    if group.shape != a.shape or group.shape != b.shape:
        raise AssertionError("cross-fit vectors have different shapes")
    if not np.isin(group, [0, 1]).all():
        raise AssertionError("hash group must be binary")
    return np.where(group == 0, a, b)


def conditional_contrasts(z_clean: np.ndarray, z_vol: np.ndarray,
                          z_fresh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the only two corrections allowed by the experiment."""
    z_clean = np.asarray(z_clean, float)
    z_vol = np.asarray(z_vol, float)
    z_fresh = np.asarray(z_fresh, float)
    if not (z_clean.shape == z_vol.shape == z_fresh.shape):
        raise AssertionError("conditional prediction vectors differ in shape")
    return z_fresh - z_clean, z_vol - z_clean


def winsor_bounds(raws: list[np.ndarray], donor_folds: list[int]) -> tuple[float, float]:
    """0.5/99.5% bounds fitted only on explicitly named donor folds."""
    if not donor_folds:
        raise AssertionError("winsorization needs at least one donor fold")
    x = np.concatenate([np.asarray(raws[i], float) for i in donor_folds])
    lo, hi = np.quantile(x, [0.005, 0.995])
    return float(lo), float(hi)


def process_correction(raw: np.ndarray, bounds: tuple[float, float], variant: str,
                       w180_days_buy: np.ndarray) -> tuple[np.ndarray, float]:
    """Winsorize, optionally gate by cutoff-safe HIGH16, then center.

    The bounds are always fitted on the ungated contrast.  HIGH16 is applied
    after clipping and before centering, matching the registered order:
    correction level cannot be changed through the specialist gate.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    raw = np.asarray(raw, float)
    w180 = np.asarray(w180_days_buy)
    if raw.shape != w180.shape:
        raise AssertionError("HIGH16 feature is not aligned with correction")
    lo, hi = bounds
    clipped = np.clip(raw, lo, hi)
    clipped_fraction = float(np.mean((raw < lo) | (raw > hi)))
    if variant == "HIGH16":
        clipped = clipped * (w180 >= 16)
    correction = clipped - clipped.mean()
    return correction, clipped_fraction


def add_log_correction(z_base: np.ndarray, correction: np.ndarray,
                       alpha: float) -> np.ndarray:
    """Apply the correction in log1p-space; alpha=0 is bitwise identity."""
    z_base = np.asarray(z_base)
    correction = np.asarray(correction)
    if z_base.shape != correction.shape:
        raise AssertionError("base and correction are not aligned")
    if float(alpha) == 0.0:
        return z_base.copy()
    return z_base + float(alpha) * correction


def fold_score(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    off, score = calibrate(np.asarray(y), np.asarray(z))
    return float(off), float(score)


def _candidate_order(variant: str) -> int:
    return 0 if variant == "GLOBAL" else 1


def fixed_curves(folds: list[dict], raw_key: str) -> tuple[list[dict], dict]:
    """Predefined GLOBAL/HIGH16 curves with fold-exclusive preprocessing."""
    rows, processed = [], {}
    n = len(folds)
    raws = [np.asarray(d[raw_key], float) for d in folds]
    base = np.asarray([fold_score(d["y"], d["z_base"])[1] for d in folds])
    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    for fi in range(n):
        donors = [j for j in range(n) if j != fi]
        bounds = winsor_bounds(raws, donors)
        for variant in VARIANTS:
            corr, clip = process_correction(raws[fi], bounds, variant, folds[fi]["w180"])
            processed[(fi, variant)] = corr
            for alpha in ALPHAS:
                off, score = fold_score(
                    folds[fi]["y"], add_log_correction(folds[fi]["z_base"], corr, alpha))
                rows.append({
                    "fold": str(VAL_FOLDS_S1[fi]), "fold_index": fi,
                    "variant": variant, "alpha": float(alpha), "score": score,
                    "baseline": float(base[fi]), "delta": score - base[fi],
                    "offset": off, "winsor_lo": bounds[0], "winsor_hi": bounds[1],
                    "clipped_fraction": clip, "bounds_folds": donors,
                })
    summary = []
    for variant in VARIANTS:
        for alpha in ALPHAS:
            rr = [r for r in rows if r["variant"] == variant and r["alpha"] == alpha]
            rr.sort(key=lambda x: x["fold_index"])
            delta = np.asarray([r["delta"] for r in rr])
            summary.append({
                "variant": variant, "alpha": float(alpha),
                "delta_wcv": float(weights @ delta / weights.sum()),
                "improved_folds": int((delta < 0).sum()),
                **{f"delta_{V:%Y%m%d}": float(v) for V, v in zip(VAL_FOLDS_S1, delta)},
            })
    return rows, {"summary": summary, "processed": processed}


def nested_lofo(folds: list[dict], raw_key: str) -> dict:
    """True outer LOFO for variant/alpha *and* winsor bounds.

    For outer fold ``h``, selection uses the remaining three folds.  Each of
    those training folds is itself preprocessed with bounds from the other two
    training folds.  Thus outer ``h`` cannot affect selection through labels,
    correction distribution, variant, alpha, or winsorization.
    """
    n = len(folds)
    if n != 4:
        raise AssertionError("project nested LOFO is defined on exactly four folds")
    raws = [np.asarray(d[raw_key], float) for d in folds]
    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    base_scores = np.asarray([fold_score(d["y"], d["z_base"])[1] for d in folds])
    selected, held_scores, held_offsets, held_corr = [], [], [], []
    for h in range(n):
        train = [i for i in range(n) if i != h]
        curves: dict[tuple[str, float], list[float]] = {
            (v, float(a)): [] for v in VARIANTS for a in ALPHAS
        }
        selection_bounds = {}
        for t in train:
            donors = [i for i in train if i != t]
            bounds = winsor_bounds(raws, donors)
            selection_bounds[t] = {"bounds": bounds, "donor_folds": donors}
            for variant in VARIANTS:
                corr, _ = process_correction(raws[t], bounds, variant, folds[t]["w180"])
                for alpha in ALPHAS:
                    score = fold_score(
                        folds[t]["y"], add_log_correction(folds[t]["z_base"], corr, alpha))[1]
                    curves[(variant, float(alpha))].append(score)
        wt = weights[train] / weights[train].sum()
        ranked = []
        for (variant, alpha), scores in curves.items():
            ranked.append((float(wt @ np.asarray(scores)), float(alpha),
                           _candidate_order(variant), variant))
        best_score, alpha, _, variant = min(ranked)
        held_bounds = winsor_bounds(raws, train)
        corr, clip = process_correction(raws[h], held_bounds, variant, folds[h]["w180"])
        off, score = fold_score(
            folds[h]["y"], add_log_correction(folds[h]["z_base"], corr, alpha))
        held_scores.append(score)
        held_offsets.append(off)
        held_corr.append(corr)
        selected.append({
            "fold": str(VAL_FOLDS_S1[h]), "heldout_fold": h,
            "selection_folds": train, "selection_bounds": selection_bounds,
            "heldout_bounds_folds": train, "heldout_bounds": held_bounds,
            "variant": variant, "alpha": alpha, "selection_score": best_score,
            "heldout_score": score, "baseline_score": float(base_scores[h]),
            "heldout_delta": score - base_scores[h], "heldout_offset": off,
            "clipped_fraction": clip,
        })
    held_scores = np.asarray(held_scores)
    delta = held_scores - base_scores
    w = weights / weights.sum()
    return {
        "base_scores": base_scores.tolist(), "base_wcv": float(w @ base_scores),
        "heldout_scores": held_scores.tolist(), "heldout_delta": delta.tolist(),
        "candidate_wcv": float(w @ held_scores), "delta_wcv": float(w @ delta),
        "improved_folds": int((delta < 0).sum()), "selected": selected,
        "processed_heldout": held_corr,
    }


def _source_paths(V: dt.date) -> tuple[Path, Path, Path]:
    tag = f"V{V:%m%d}"
    z = ARTIFACTS / f"S04PROD_S42-{tag}_z.npz"
    mu = ARTIFACTS / f"S04PROD_S42-{tag}_mu.npz"
    mirror = ARTIFACTS / f"{PREFIX}_mirror_A_{V:%Y%m%d}.npz"
    return z, mu, mirror


def _embedding_prefix(V: dt.date) -> str:
    return f"S04SEQ_emb_SEQ-D3A-BASE-S42-V{V:%m%d}"


def build_mirror_fold(V: dt.date, resume: bool = True) -> Path:
    """Train only the missing EXTRA-group-A heads and predict recipient group B."""
    zpath, mupath, out = _source_paths(V)
    if resume and out.exists():
        log(f"mirror {V}: loading {out.name}")
        return out
    if not zpath.exists() or not mupath.exists():
        raise FileNotFoundError(f"EXP-032B source artifacts missing for {V}")

    ckpt = f"SEQ-D3A-BASE-S42-V{V:%m%d}"
    model, cfg, Vc, dev = load_ckpt(ckpt)
    if Vc != V:
        raise AssertionError(f"checkpoint {ckpt} belongs to {Vc}, not {V}")
    for p in model.parameters():
        p.requires_grad_(False)
    checksum = float(sum(float(p.double().sum()) for p in model.parameters()))
    cpre = _embedding_prefix(V)

    clean_meta = np.load(ARTIFACTS / f"{cpre}_clean_meta.npz")
    Xc = np.load(ARTIFACTS / f"{cpre}_clean_X.npy")
    zc, cc = clean_meta["z"], clean_meta["c"]
    clean_cuts = fold_cutoffs(V)
    if [str(x) for x in clean_meta["cuts"]] != [x.isoformat() for x in clean_cuts]:
        raise AssertionError("CLEAN embedding cutoff grid changed since EXP-032")

    extra_cache = f"{PREFIX}_emb_{ckpt}_extraA"
    Xa, za, ua, ca = collect(
        model, cfg, dev, EXTRA_CUTOFFS, 1, keep=POS_ONLY, group_keep=0,
        depth_clip=EXTRA_DEPTH_CLIP, tag="EXTRA-A", cache=extra_cache)

    source = np.load(zpath)
    uv = np.asarray(source["uid"], np.int64)
    group = np.asarray(source["group"], np.int8)
    recipient = uv[group == 1]
    validate_crossfit(recipient, ua, recipient_group=1, donor_group=0)
    Xv = np.load(ARTIFACTS / f"{cpre}_val_X.npy")
    if len(Xv) != len(uv):
        raise AssertionError("validation embedding panel differs from EXP-032B")

    pos_c = zc > 0
    c_clean = np.asarray([zc[pos_c & (cc == k)].mean() for k in range(len(clean_cuts))])
    c_extra = np.asarray([za[ca == k].mean() for k in range(len(EXTRA_CUTOFFS))])
    c_hat = float(c_clean.mean())
    tcp = (zc[pos_c] - c_clean[cc[pos_c]]).astype(np.float32)
    tep = (za - c_extra[ca]).astype(np.float32)
    ci_pos = cc[pos_c]
    n_cp, n_extra = len(tcp), len(tep)
    Xp = np.empty((n_cp + n_extra, Xc.shape[1]), Xc.dtype)
    np.take(Xc, np.flatnonzero(pos_c), axis=0, out=Xp[:n_cp])
    Xp[n_cp:] = Xa
    tp = np.concatenate([tcp, tep])
    del Xc, Xa
    gc.collect()

    steps = int(np.ceil(n_cp / HEAD_BATCH)) * HEAD_EPOCHS
    early = np.flatnonzero(ci_pos < max(1, len(clean_cuts) // 3))
    rng = np.random.default_rng(HEAD_SEED)
    rows = {
        "VOL": np.concatenate([np.arange(n_cp),
                               rng.choice(early, size=n_extra, replace=True)]),
        "FRESH": np.concatenate([np.arange(n_cp),
                                 np.arange(n_cp, n_cp + n_extra)]),
    }
    pred = {}
    for name in ("VOL", "FRESH"):
        log(f"mirror {V}: training {name}, EXTRA-A={n_extra:,}, steps={steps:,}")
        net, loss = fit_head(
            Xp, tp, steps=steps, batch=HEAD_BATCH, lr=HEAD_LR, wd=HEAD_WD,
            hidden=cfg["hidden"], dropout=HEAD_DROPOUT, seed=HEAD_SEED,
            binary=False, dev=dev, out_bias=0.0, rows=rows[name])
        pred[f"mu_{name}"] = head_predict(net, Xv, dev).astype(np.float64) + c_hat
        pred[f"loss_{name}"] = np.asarray(loss)
        del net
        gc.collect()

    p = np.asarray(source["P_DIST"], float)
    for name in ("VOL", "FRESH"):
        pred[f"z_{name}"] = np.maximum(p * np.maximum(pred[f"mu_{name}"], 0.0), 0.0)
    np.savez_compressed(
        out, uid=uv, group=group, donor_uid=np.unique(ua), recipient_uid=recipient,
        n_extra=np.asarray(n_extra), c_hat=np.asarray(c_hat),
        encoder_checksum=np.asarray(checksum), **pred)
    after = float(sum(float(p.double().sum()) for p in model.parameters()))
    if after != checksum:
        raise AssertionError("frozen encoder changed while fitting mirror heads")
    log(f"mirror {V}: saved {out.name}")
    return out


def _segment_columns(V: dt.date, uid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = features_cached(V, None, False).select(
        ["user_id", "w180_days_buy", "rec_buy"])
    out = pl.DataFrame({"user_id": uid}).join(frame, on="user_id", how="left")
    if out.height != len(uid) or out["w180_days_buy"].null_count():
        raise AssertionError(f"segment feature alignment failed on {V}")
    return (out["w180_days_buy"].to_numpy().astype(np.float32),
            out["rec_buy"].to_numpy().astype(np.float32))


def build_full_fold(V: dt.date, resume: bool = True) -> Path:
    """Assemble full-panel cross-fit CLEAN/VOL/FRESH and STRONGEST_CURRENT."""
    out = ARTIFACTS / f"{PREFIX}_fold_{V:%Y%m%d}.npz"
    if resume and out.exists():
        log(f"full fold {V}: loading {out.name}")
        return out
    zpath, mupath, mirror_path = _source_paths(V)
    if not mirror_path.exists():
        build_mirror_fold(V, resume=resume)
    src = np.load(zpath)
    mirror = np.load(mirror_path)
    uid = np.asarray(src["uid"], np.int64)
    if not np.array_equal(uid, mirror["uid"]):
        raise AssertionError("mirror/source user order differs")
    group = np.asarray(src["group"], np.int8)
    if not np.array_equal(group, stable_group(uid)):
        raise AssertionError("saved EXP-032 hash groups do not match splitmix64")
    original_extra = np.load(
        ARTIFACTS / f"{_embedding_prefix(V)}_extra_meta.npz", allow_pickle=False)
    donor_b = np.unique(original_extra["u"])
    validate_crossfit(uid[group == 0], donor_b, 0, 1)
    validate_crossfit(mirror["recipient_uid"], mirror["donor_uid"], 1, 0)

    z_clean = np.asarray(src["z_DIST_X_CLEAN"], float)
    z_vol = merge_crossfit(group, src["z_DIST_X_VOL"], mirror["z_VOL"])
    z_fresh = merge_crossfit(group, src["z_DIST_X_FRESH"], mirror["z_FRESH"])
    d_fresh, d_vol = conditional_contrasts(z_clean, z_vol, z_fresh)

    p = np.asarray(src["P_DIST"], float)
    mu_clean = np.asarray(src["mu_CLEAN"], float)
    mu_vol = merge_crossfit(group, src["mu_VOL"], mirror["mu_VOL"])
    mu_fresh = merge_crossfit(group, src["mu_FRESH"], mirror["mu_FRESH"])
    alg_f = p * (mu_fresh - mu_clean)
    alg_v = p * (mu_vol - mu_clean)
    # All EXP-032 intensities are positive; this is the exact composition used.
    max_alg_f = float(np.max(np.abs(d_fresh - alg_f)))
    max_alg_v = float(np.max(np.abs(d_vol - alg_v)))
    if max_alg_f > 1e-10 or max_alg_v > 1e-10:
        raise AssertionError("z difference is not the conditional-intensity contribution")

    order = np.argsort(uid)
    uid, group = uid[order], group[order]
    y = np.asarray(src["y"], float)[order]
    panel_uid = panel_users(V, 3).sort("user_id")["user_id"].to_numpy()
    if not np.array_equal(uid, panel_uid):
        raise AssertionError(f"EXP-032 panel differs from project fold {V}")
    y_ref = target_at(V, user_rows(uid))
    if not np.allclose(y, y_ref, atol=1e-8):
        raise AssertionError(f"EXP-032 targets differ from project fold {V}")
    z_base = _strongest_fold(V, uid, y)
    w180, rec = _segment_columns(V, uid)
    np.savez_compressed(
        out, uid=uid, y=y.astype(np.float32), group=group, z_base=z_base.astype(np.float32),
        z_clean=z_clean[order].astype(np.float32), z_vol=z_vol[order].astype(np.float32),
        z_fresh=z_fresh[order].astype(np.float32),
        d_fresh=d_fresh[order].astype(np.float32), d_vol=d_vol[order].astype(np.float32),
        w180=w180, rec=rec, p_dist=p[order].astype(np.float32),
        algebra_max_fresh=np.asarray(max_alg_f), algebra_max_vol=np.asarray(max_alg_v))
    log(f"full fold {V}: saved {out.name}; algebra {max_alg_f:.2e}/{max_alg_v:.2e}")
    return out


def load_folds() -> list[dict]:
    folds = []
    for V in VAL_FOLDS_S1:
        p = ARTIFACTS / f"{PREFIX}_fold_{V:%Y%m%d}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        d = dict(np.load(p, allow_pickle=False))
        d["raw_fresh"], d["raw_vol"] = conditional_contrasts(
            d["z_clean"], d["z_vol"], d["z_fresh"])
        folds.append(d)
    return folds


def auc(y: np.ndarray, z: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y) > 0, np.asarray(z)))


def quantiles(x: np.ndarray) -> dict[str, float]:
    ps = [0.01, 0.05, 0.50, 0.95, 0.99]
    return {f"p{int(p * 100):02d}": float(v) for p, v in zip(ps, np.quantile(x, ps))}


def segment_masks(w180: np.ndarray, rec: np.ndarray) -> dict[str, np.ndarray]:
    known = np.isfinite(rec)
    return {
        "w180_days_buy >= 16": w180 >= 16,
        "w180_days_buy 2-15": (w180 >= 2) & (w180 <= 15),
        "w180_days_buy 0-1": w180 <= 1,
        "rec_buy 15-60": known & (rec >= 15) & (rec <= 60),
        "never purchased": ~known,
    }


def _rmse_on_mask(ly: np.ndarray, z: np.ndarray, off: float,
                  mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ly[mask] - np.maximum(z[mask] + off, 0.0)) ** 2)))


def diagnostics(folds: list[dict], nested: dict, raw_key: str) -> tuple[list[dict], list[dict], list[dict]]:
    rows, seg_rows, group_rows = [], [], []
    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    raw_other = "raw_vol" if raw_key == "raw_fresh" else "raw_fresh"
    for fi, d in enumerate(folds):
        sel = nested["selected"][fi]
        corr = np.asarray(nested["processed_heldout"][fi])
        z_base = np.asarray(d["z_base"], float)
        z_new = add_log_correction(z_base, corr, sel["alpha"])
        ly = np.log1p(d["y"])
        off_b, score_b = fold_score(d["y"], z_base)
        off_n, score_n = fold_score(d["y"], z_new)
        pos = d["y"] > 0
        residual = ly - z_base
        other = np.asarray(d[raw_other], float)
        raw = np.asarray(d[raw_key], float)
        rows.append({
            "fold": str(VAL_FOLDS_S1[fi]), "baseline_rmsle_cal": score_b,
            "candidate_rmsle_cal": score_n, "delta": score_n - score_b,
            "selected_variant": sel["variant"], "selected_alpha": sel["alpha"],
            "var_d_fresh": float(np.var(d["raw_fresh"])),
            "var_d_vol": float(np.var(d["raw_vol"])),
            "corr_d_fresh_residual": float(np.corrcoef(d["raw_fresh"], residual)[0, 1]),
            "corr_d_vol_residual": float(np.corrcoef(d["raw_vol"], residual)[0, 1]),
            "corr_selected_residual": float(np.corrcoef(corr, residual)[0, 1]),
            "corr_fresh_vol": float(np.corrcoef(raw, other)[0, 1]),
            "corr_residuals": float(np.corrcoef(residual, ly - z_new)[0, 1]),
            "auc_base": auc(d["y"], z_base), "auc_candidate": auc(d["y"], z_new),
            "positive_rmse_base": _rmse_on_mask(ly, z_base, off_b, pos),
            "positive_rmse_candidate": _rmse_on_mask(ly, z_new, off_n, pos),
            "correction_mean": float(corr.mean()), "correction_std": float(corr.std()),
            **{f"correction_{k}": v for k, v in quantiles(corr).items()},
            "clipped_fraction": sel["clipped_fraction"],
        })
        for name, mask in segment_masks(d["w180"], d["rec"]).items():
            seg_rows.append({
                "fold": str(VAL_FOLDS_S1[fi]), "segment": name, "n": int(mask.sum()),
                "share": float(mask.mean()),
                "baseline_rmsle": _rmse_on_mask(ly, z_base, off_b, mask),
                "candidate_rmsle": _rmse_on_mask(ly, z_new, off_n, mask),
            })
            seg_rows[-1]["delta"] = (seg_rows[-1]["candidate_rmsle"]
                                       - seg_rows[-1]["baseline_rmsle"])
        for g in (0, 1):
            mask = d["group"] == g
            gb = fold_score(d["y"][mask], z_base[mask])[1]
            gn = fold_score(d["y"][mask], z_new[mask])[1]
            group_rows.append({"fold": str(VAL_FOLDS_S1[fi]), "group": "A" if g == 0 else "B",
                               "n": int(mask.sum()), "baseline": gb, "candidate": gn,
                               "delta": gn - gb})
    # Append wCV rows without hiding the per-fold values.
    for out, keys in ((seg_rows, ("baseline_rmsle", "candidate_rmsle", "delta")),
                      (group_rows, ("baseline", "candidate", "delta"))):
        names = sorted({r.get("segment", r.get("group")) for r in out})
        for name in names:
            rr = [r for r in out if r.get("segment", r.get("group")) == name]
            rr.sort(key=lambda x: list(map(str, VAL_FOLDS_S1)).index(x["fold"]))
            row = {"fold": "wCV", ("segment" if "segment" in rr[0] else "group"): name,
                   "n": int(sum(r["n"] for r in rr))}
            for key in keys:
                row[key] = float(weights @ np.asarray([r[key] for r in rr]) / weights.sum())
            if "share" in rr[0]:
                row["share"] = float(weights @ np.asarray([r["share"] for r in rr]) / weights.sum())
            out.append(row)
    return rows, seg_rows, group_rows


def choose_all_oof(fixed_summary: list[dict]) -> dict:
    """Production candidate choice after nested LOFO validates the procedure."""
    ranked = sorted(fixed_summary, key=lambda r: (
        r["delta_wcv"], r["alpha"], _candidate_order(r["variant"])))
    return ranked[0]


def preliminary_decision(fresh: dict, vol: dict, group_rows: list[dict]) -> tuple[str, list[str]]:
    d = float(fresh["delta_wcv"])
    fd = np.asarray(fresh["heldout_delta"])
    causal = d - float(vol["delta_wcv"])
    group_wcv = {r["group"]: r["delta"] for r in group_rows if r["fold"] == "wCV"}
    same_sign = group_wcv.get("A", 1.0) < 0 and group_wcv.get("B", 1.0) < 0
    common = int((fd < 0).sum()) >= 3 and fd[-1] < 0 and same_sign
    if d <= -0.0010 and common and causal <= -0.0004:
        return "STRONG_ACCEPT_VALIDATION", ["validation gates passed; test audit required"]
    if -0.0010 < d <= -0.0005 and common and causal <= -0.0002:
        return "CONTINUE_VALIDATION", ["borderline validation gates passed; test audit required"]
    reasons = []
    if d > -0.0005:
        reasons.append(f"nested LOFO {d:+.6f} is above the -0.0005 floor")
    if int((fd < 0).sum()) < 3:
        reasons.append(f"only {int((fd < 0).sum())}/4 folds improve")
    if fd[-1] >= 0:
        reasons.append("2025-10-16 does not improve")
    if causal > -0.0002:
        reasons.append(f"FRESH is not materially better than VOL ({causal:+.6f})")
    if not same_sign:
        reasons.append(f"hash-group sign mismatch: {group_wcv}")
    return "REJECT", reasons or ["registered validation gate failed"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON-encode nested provenance so CSV remains rectangular and auditable.
    flat = []
    for row in rows:
        flat.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v)
                     for k, v in row.items()})
    if not flat:
        return
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def save_oof(folds: list[dict], fresh: dict, vol: dict) -> None:
    values = {
        "uid": np.concatenate([d["uid"] for d in folds]),
        "cutoff": np.concatenate([np.full(len(d["uid"]), str(V), dtype="U10")
                                  for V, d in zip(VAL_FOLDS_S1, folds)]),
        "y": np.concatenate([d["y"] for d in folds]),
        "group": np.concatenate([d["group"] for d in folds]),
        "z_base": np.concatenate([d["z_base"] for d in folds]),
        "z_clean": np.concatenate([d["z_clean"] for d in folds]),
        "z_vol": np.concatenate([d["z_vol"] for d in folds]),
        "z_fresh": np.concatenate([d["z_fresh"] for d in folds]),
        "d_fresh": np.concatenate([d["raw_fresh"] for d in folds]),
        "d_vol": np.concatenate([d["raw_vol"] for d in folds]),
        "w180": np.concatenate([d["w180"] for d in folds]),
        "rec": np.concatenate([d["rec"] for d in folds]),
        "fresh_processed_nested": np.concatenate(fresh["processed_heldout"]),
        "vol_processed_nested": np.concatenate(vol["processed_heldout"]),
    }
    np.savez_compressed(ARTIFACTS / f"oof_{PREFIX}.npz", **values)


def analyze() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    folds = load_folds()
    fresh_fixed_rows, fresh_fixed = fixed_curves(folds, "raw_fresh")
    vol_fixed_rows, vol_fixed = fixed_curves(folds, "raw_vol")
    fresh = nested_lofo(folds, "raw_fresh")
    vol = nested_lofo(folds, "raw_vol")
    diag, segments, groups = diagnostics(folds, fresh, "raw_fresh")
    decision, reasons = preliminary_decision(fresh, vol, groups)
    causal = float(fresh["delta_wcv"] - vol["delta_wcv"])
    # A four-fold production choice is authorized only after nested LOFO passes.
    choice = (choose_all_oof(fresh_fixed["summary"])
              if decision != "REJECT" else None)
    group_wcv = {r["group"]: r["delta"] for r in groups if r["fold"] == "wCV"}
    summary = {
        "experiment": "FRESH-CONTRAST-MOE", "two_sided_crossfit": True,
        "baseline_wcv": fresh["base_wcv"], "fresh_nested_lofo_delta": fresh["delta_wcv"],
        "vol_nested_lofo_delta": vol["delta_wcv"], "fresh_minus_vol": causal,
        "fresh_improved_folds": fresh["improved_folds"],
        "fold_20251016_delta": fresh["heldout_delta"][-1],
        "hash_group_delta": group_wcv,
        "hash_same_negative_sign": bool(group_wcv["A"] < 0 and group_wcv["B"] < 0),
        "validation_decision": decision, "reasons": reasons,
        "production_choice_if_eligible": choice,
        "test_regime": None, "submission": None,
    }
    json_dump(RESULTS / "validation.json", {
        "summary": summary, "fresh_nested": {k: v for k, v in fresh.items()
                                                if k != "processed_heldout"},
        "vol_nested": {k: v for k, v in vol.items() if k != "processed_heldout"},
        "fresh_fixed": fresh_fixed["summary"], "vol_fixed": vol_fixed["summary"],
    })
    json_dump(RESULTS / "summary.json", summary)
    _write_csv(RESULTS / "fixed_curves.csv", [
        {"contrast": "FRESH", **r} for r in fresh_fixed_rows] +
        [{"contrast": "VOL", **r} for r in vol_fixed_rows])
    _write_csv(RESULTS / "fixed_summary.csv", [
        {"contrast": "FRESH", **r} for r in fresh_fixed["summary"]] +
        [{"contrast": "VOL", **r} for r in vol_fixed["summary"]])
    _write_csv(RESULTS / "nested_lofo.csv", [
        {"contrast": "FRESH", **r} for r in fresh["selected"]] +
        [{"contrast": "VOL", **r} for r in vol["selected"]])
    _write_csv(RESULTS / "fold_diagnostics.csv", diag)
    _write_csv(RESULTS / "segments.csv", segments)
    _write_csv(RESULTS / "hash_groups.csv", groups)
    save_oof(folds, fresh, vol)
    json_dump(RESULTS / "config.json", {
        "prefix": PREFIX, "alphas": ALPHAS.tolist(), "variants": list(VARIANTS),
        "folds": VAL_FOLDS_S1, "fold_weights": FOLD_WEIGHTS_S1,
        "head_seed": HEAD_SEED, "head_epochs": HEAD_EPOCHS, "head_batch": HEAD_BATCH,
        "head_lr": HEAD_LR, "head_wd": HEAD_WD, "extra_depth_clip": EXTRA_DEPTH_CLIP,
        "preprocessing": "bounds on donor folds; clip d at 0.5/99.5%; HIGH16 gate; center",
        "base_oof": {"S1-E03a": .10, "S1-E02": .20, "S1-DIST": .25,
                     "ETX-AVG3": .225, "SEQ-AVG3": .225},
    })
    log(f"FRESH nested LOFO {fresh['delta_wcv']:+.6f}; "
        f"VOL {vol['delta_wcv']:+.6f}; contrast {causal:+.6f}; {decision}")
    return summary


def align_to_sample(uid: np.ndarray, values: np.ndarray,
                    sample_uid: np.ndarray) -> np.ndarray:
    """Return values in sample-submission order, with exact set validation."""
    uid = np.asarray(uid)
    values = np.asarray(values)
    sample_uid = np.asarray(sample_uid)
    if len(uid) != len(values) or len(np.unique(uid)) != len(uid):
        raise AssertionError("prediction user ids are invalid")
    order = np.argsort(uid)
    pos = np.searchsorted(uid[order], sample_uid)
    if np.any(pos >= len(uid)) or not np.array_equal(uid[order][pos], sample_uid):
        raise AssertionError("prediction users differ from sample_submission")
    return values[order][pos]


def level_shift(z: np.ndarray, target: float = L_STAR) -> float:
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(np.maximum(z + mid, 0.0).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(resume: bool = True) -> dict:
    for V in VAL_FOLDS_S1:
        build_mirror_fold(V, resume=resume)
        build_full_fold(V, resume=resume)
        gc.collect()
    summary = analyze()
    if summary["validation_decision"] == "REJECT":
        log("validation REJECT: production conditional inference and submission are skipped")
        return summary
    raise NotImplementedError(
        "validation passed, but production EXP-032-compatible frozen encoder artifact "
        "must be identified before test inference")


def main() -> None:
    ap = argparse.ArgumentParser(description="FRESH-CONTRAST-MOE")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "mirror", "assemble", "analyze"])
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    resume = not args.no_resume
    if args.command == "mirror":
        for V in VAL_FOLDS_S1:
            build_mirror_fold(V, resume=resume)
    elif args.command == "assemble":
        for V in VAL_FOLDS_S1:
            build_full_fold(V, resume=resume)
    elif args.command == "analyze":
        analyze()
    else:
        run(resume=resume)


if __name__ == "__main__":
    main()
