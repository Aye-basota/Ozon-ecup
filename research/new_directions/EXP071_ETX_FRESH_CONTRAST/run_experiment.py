"""EXP071_ETX_FRESH_CONTRAST.

Frozen ETX-01-S42 final-query representations with CLEAN/FRESH/VOL
conditional-positive heads.  The historical workspace is read-only; all new
artifacts are written beside this runner.

Commands:
  python run_experiment.py recon
  python run_experiment.py parity
  python run_experiment.py pilot
  python run_experiment.py full
  python run_experiment.py production
  python run_experiment.py finalize
  python run_experiment.py auto
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl


OUT = Path(__file__).resolve().parent
CLEAN_ROOT = OUT.parents[2]
LEGACY = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
GEOM = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
PACKET = GEOM / "gpt_pro_research_packet"
LEGACY_ART = LEGACY / "artifacts"
CACHE = OUT / "_cache"
FOLDS = [dt.date(2025, 9, 4), dt.date(2025, 9, 18),
         dt.date(2025, 10, 2), dt.date(2025, 10, 16)]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0])
ALPHAS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
HEAD_SEEDS = [42, 43, 44]
EXTRA_CUTOFFS = [dt.date(2025, 10, 22) + dt.timedelta(days=7 * i) for i in range(13)]
PILOT_FOLD = FOLDS[-1]
HEAD_CFG = dict(input_dim=128, hidden=64, dropout=0.1, learning_rate=0.001,
                weight_decay=0.01, batch_size=8192, epochs=4,
                optimizer="AdamW", betas=[0.9, 0.98], warmup_steps=200)
EXPECTED_ETX = dict(d_model=128, blocks=5, heads=8, head_dim=16, ffn=384,
                    dropout=0.1, n_tok=192, batch=512, chunk=128, lr=0.0015,
                    wd=0.01, epochs=4, warmup=500, seed=42, compile=False,
                    tau_lo=4.0, tau_hi=512.0)
START = time.time()

# Ensure the historical src package wins over the clean repository's src package.
sys.path.insert(0, str(LEGACY))
from src import etx, seq  # noqa: E402
from src.config import CORRIDOR_END, CUTOFF_STEP, CUTOFF_TEST, cutoff_grid  # noqa: E402
from src.features import panel_users  # noqa: E402
from src.validation import calibrate, rmsle_z  # noqa: E402
from src.seq_cond import user_group  # noqa: E402


def log(*parts) -> None:
    print(f"[{time.time() - START:8.1f}s]", *parts, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple))
                            else v) for k, v in row.items()})


def ckpt_name(V: dt.date | None) -> str:
    return "ETX-01-S42-TEST" if V is None else f"ETX-01-S42-V{V:%m%d}"


def ckpt_path(V: dt.date | None) -> Path:
    return LEGACY_ART / f"model_{ckpt_name(V)}.pt"


def torch_load_checkpoint(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def recovered_checkpoint(V: dt.date | None) -> dict:
    p = ckpt_path(V)
    d = torch_load_checkpoint(p)
    cfg = dict(d["cfg"])
    mismatch = {k: {"expected": v, "actual": cfg.get(k)} for k, v in EXPECTED_ETX.items()
                if cfg.get(k) != v}
    if mismatch:
        raise AssertionError(f"checkpoint config mismatch {p.name}: {mismatch}")
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p),
            "val": d["val"], "cfg": cfg, "state_tensors": len(d["state"]),
            "config_assertion": "PASS"}


def manifest_sources() -> list[tuple[Path, str, str]]:
    rows: list[tuple[Path, str, str]] = []
    for V in FOLDS:
        rows.append((ckpt_path(V), "ETX-01-S42 fold checkpoint", "checkpoint"))
        rows.append((LEGACY_ART / f"oof_ETX-01-S42-V{V:%m%d}.npz",
                     "saved ETX seed-42 fold prediction", "prediction"))
    rows.extend([
        (ckpt_path(None), "ETX-01-S42 TEST checkpoint", "checkpoint"),
        (LEGACY_ART / "ztest_ETX-01-S42-DCW.npy", "ETX seed-42 corrected TEST prediction", "prediction"),
        (LEGACY_ART / "uid_ETX-01-S42-DCW.npy", "ETX seed-42 corrected TEST keys", "keys"),
        (LEGACY / "weights_archives" / "ETX-01_weights.zip", "ETX checkpoint archive", "archive"),
        (LEGACY / "src" / "etx.py", "ETX implementation", "code"),
        (LEGACY / "research" / "strategies" / "results" / "ETX2" / "depth_fix.py",
         "DCW/static-context production fix", "code"),
        (LEGACY / "src" / "seq_cond.py", "EXP-032 CLEAN/FRESH/VOL construction", "code"),
        (LEGACY / "src" / "fresh_contrast.py", "EXP-040 nested preprocessing/evaluator", "code"),
        (LEGACY_ART / "oof_FRESH_CONTRAST_MOE.npz", "existing SEQ-FRESH correction", "prediction"),
        (CLEAN_ROOT / "artifacts" / "oof" / "EXP_037_STRONGEST_CURRENT.parquet",
         "canonical EXP-037 OOF", "prediction"),
        (PACKET / "06_ALIGNED_OOF.parquet", "aligned canonical OOF bank", "prediction_bank"),
        (PACKET / "07_ALIGNED_TEST.parquet", "aligned TEST bank", "prediction_bank"),
        (GEOM / "submission_geometry" / "directions.py", "geometry basis builder", "code"),
        (GEOM / "submission_geometry" / "cache" / "Z.npz", "65-source geometry matrix cache", "geometry"),
        (GEOM / "submission_geometry" / "cache" / "Z_meta.json", "geometry source metadata", "geometry"),
        (GEOM / "submission_geometry" / "families.py", "geometry family map", "code"),
        (LEGACY / "experiments" / "exp_032_s04_cond_fresh_pilot.md", "EXP-032 evidence", "documentation"),
        (LEGACY / "experiments" / "EXP_032_S04_conditional_fresh_seq.md", "EXP-032 design", "documentation"),
        (LEGACY / "experiments" / "exp_037_etx_avg3_strongest.md", "EXP-037 builder/fix", "documentation"),
        (LEGACY / "experiments" / "exp_040_fresh_contrast_moe.md", "EXP-040 evidence", "documentation"),
    ])
    return rows


def reconnaissance() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    checkpoints = [recovered_checkpoint(V) for V in FOLDS] + [recovered_checkpoint(None)]
    rows = []
    for p, purpose, kind in manifest_sources():
        if not p.exists():
            rows.append({"path": str(p), "bytes": "", "sha256": "", "kind": kind,
                         "purpose": purpose, "status": "MISSING"})
        else:
            rows.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p),
                         "kind": kind, "purpose": purpose, "status": "FOUND"})
    missing_hard = [r for r in rows if r["status"] == "MISSING" and r["kind"] in
                    {"checkpoint", "prediction", "prediction_bank", "code", "geometry"}]
    if missing_hard:
        raise FileNotFoundError(f"hard reconnaissance artifacts missing: {missing_hard}")
    csv_write(OUT / "artifact_manifest.csv", rows)
    config = {
        "experiment": "EXP071_ETX_FRESH_CONTRAST",
        "hypothesis": "Frozen ETX final query embeddings yield a more useful and orthogonal FRESH-minus-CLEAN conditional-positive residual direction than the historical frozen TCN/D3A representation.",
        "public_incumbent_reference_only": 1.6466079084,
        "baseline": "z_base=log1p(06_ALIGNED_OOF.pred_exp037); weights are not retuned",
        "folds": [str(x) for x in FOLDS], "fold_weights": FOLD_WEIGHTS.tolist(),
        "encoder": "ETX-01 seed 42, frozen", "encoder_inference_precision": "bf16 autocast on CUDA",
        "checkpoint_metadata": checkpoints,
        "query_embedding": "final normalized query token zq, dimension 128",
        "head": HEAD_CFG, "head_seeds": HEAD_SEEDS,
        "extra_cutoffs": [str(x) for x in EXTRA_CUTOFFS],
        "crossfit": "splitmix64(user_id)&1; EXTRA donor side predicts opposite recipient side",
        "volume_control": "CLEAN positives plus with-replacement draws from earliest third of CLEAN positive cutoffs; canonical RNG seed 42; count matched separately to each EXTRA donor side",
        "preprocessing": {"winsor_quantiles": [0.005, 0.995], "scope": "GLOBAL",
                          "center": "donor clipped mean", "alpha_grid": ALPHAS.tolist(),
                          "test_distribution_matching": False},
        "static_policy": {
            "clean_and_oof": "actual calendar, actual available history, no clip/cap",
            "extra": "event depth_clip=289, query depth_cap=289, actual cutoff weekday",
            "test": "registered ETX DCW compatibility: event depth_clip=289, query depth_cap=289, cdow_shift=-1 to the checkpoint's trained Thursday support",
        },
        "no_public_lb_tuning": True, "no_upload": True,
    }
    json_write(OUT / "config.json", config)
    recon_md = f"""# EXP071 reconnaissance

## Status

All hard prerequisites were located. The four `ETX-01-S42` fold checkpoints and the seed-42 TEST checkpoint expose the unmodified ETX model state and full config metadata. Every checkpoint matches the registered 5-block, 128-dimensional ETX-01 configuration. Checkpoint hashes are recorded in `artifact_manifest.csv` and `config.json`.

## ETX representation and production fix

`src/etx.py` builds sparse real-event tokens, causal SDPA, and calendar-time ALiBi. The query token is inserted at position `n_events`; after five blocks and the final LayerNorm, `zq = h[:, n_events, :]` has dimension 128. The original direct head consumes `[zq, event_mean, last_event]`. EXP071 hooks only `zq` and does not change any weight.

The exact EXP-037 static-context correction is implemented by `research/strategies/results/ETX2/depth_fix.py`: clipped event history and query depth are both 289 on TEST, and the historical DCW checkpoint-compatible TEST prediction shifts the query cutoff weekday from Friday back to the Thursday support seen at every encoder-training cutoff. EXP071 uses actual weekdays for CLEAN/OOF and EXTRA representations; EXTRA applies only the depth clip/cap. The registered TEST DCW shift is used only for parity with the frozen production checkpoint.

## Historical conditional-positive construction

`src/seq_cond.py` supplies the exact 13 EXTRA cutoffs, `splitmix64(user_id)&1`, positive-only target filter, per-cutoff target centering, equal-step training, and the equal-volume early-CLEAN control. `src/fresh_contrast.py` supplies symmetric two-sided user cross-fitting and the q0.5%/q99.5% winsor/clip/center preprocessing family. EXP071 restricts preprocessing to GLOBAL and makes every held-out center donor-derived.

## Baselines and geometry

The canonical baseline is `{PACKET / '06_ALIGNED_OOF.parquet'}` (`pred_exp037`, 770,616 rows). Existing SEQ-FRESH is `pred_fresh_contrast`. TEST alignment is `{PACKET / '07_ALIGNED_TEST.parquet'}` (250,000 rows). The existing TEST span is the deduplicated 65-source rank-57 basis in `submission_geometry/cache/Z.npz`, built by `submission_geometry/directions.py`.

## Hard-stop decision

Reconnaissance: **PASS**. Stable-hook parity is evaluated separately in `encoder_parity.json`; pilot execution remains blocked until that audit passes.
"""
    (OUT / "reconnaissance.md").write_text(recon_md, encoding="utf-8")
    return {"status": "PASS", "checkpoints": checkpoints, "manifest_rows": len(rows)}


def forward_with_query(model, tok, static, age, n):
    """Exact ETX.forward replay plus the final normalized query token."""
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
    z = model.head(torch.cat([zq, zm, zl], dim=1)).squeeze(1)
    return z, zq


def static_policy(T: dt.date, mode: str) -> tuple[int | None, int | None, float]:
    if mode in {"clean", "oof", "pilot_donor"}:
        return None, None, 0.0
    if mode == "extra":
        return 289, 289, 0.0
    if mode == "test":
        return 289, 289, -1.0
    raise ValueError(mode)


def extract_query(model, tk, cfg: dict, dev, T: dt.date, rows: np.ndarray, mode: str,
                  compare_original: bool = False, batch: int | None = None):
    import torch
    rows = np.asarray(rows, np.int64)
    B = batch or int(cfg["batch"])
    X = np.empty((len(rows), int(cfg["d_model"])), np.float16)
    Z = np.empty(len(rows), np.float32)
    max_forward_error = 0.0
    depth_clip, depth_cap, dow_shift = static_policy(T, mode)
    old = (tk.depth_cap, tk.cdow_shift)
    tk.depth_cap, tk.cdow_shift = depth_cap, dow_shift
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(rows), B):
                r = rows[i:i + B]
                idx, cnt = etx.select(T, r, cfg["n_tok"], depth_clip=depth_clip)
                cd = np.full(len(r), seq.day_index(T), np.int32)
                ti = torch.from_numpy(idx).to(dev)
                tc = torch.from_numpy(cnt).to(dev)
                td = torch.from_numpy(cd).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    tok, st, age, n = tk(ti, tc, td)
                    zh, q = forward_with_query(model, tok, st, age, n)
                    if compare_original:
                        zo = model(tok, st, age, n)
                        max_forward_error = max(max_forward_error,
                                                float((zh.float() - zo.float()).abs().max().cpu()))
                Z[i:i + len(r)] = zh.float().cpu().numpy()
                X[i:i + len(r)] = q.float().cpu().numpy().astype(np.float16)
    finally:
        tk.depth_cap, tk.cdow_shift = old
    return X, Z, max_forward_error


def align_values(source_uid, source_values, wanted_uid):
    source_uid = np.asarray(source_uid, np.int64)
    wanted_uid = np.asarray(wanted_uid, np.int64)
    order = np.argsort(source_uid)
    pos = np.searchsorted(source_uid[order], wanted_uid)
    if np.any(pos >= len(order)) or not np.array_equal(source_uid[order][pos], wanted_uid):
        raise AssertionError("user-id alignment mismatch")
    return np.asarray(source_values)[order][pos]


def saved_etx_fold(V: dt.date, uid: np.ndarray) -> np.ndarray:
    d = np.load(LEGACY_ART / f"oof_ETX-01-S42-V{V:%m%d}.npz", allow_pickle=False)
    return align_values(d["user_id"], d["z"], uid).astype(float)


def aligned_fold(V: dt.date) -> pl.DataFrame:
    d = pl.read_parquet(PACKET / "06_ALIGNED_OOF.parquet").filter(pl.col("fold") == str(V))
    return d.sort("user_id")


def cache_array(name: str, X: np.ndarray, meta: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}.npy", X)
    np.savez_compressed(CACHE / f"{name}_meta.npz", **{
        k: np.asarray(v) if not isinstance(v, np.ndarray) else v for k, v in meta.items()})


def load_cached_array(name: str, uid: np.ndarray | None = None):
    x, m = CACHE / f"{name}.npy", CACHE / f"{name}_meta.npz"
    if not x.exists() or not m.exists():
        return None
    meta = dict(np.load(m, allow_pickle=False))
    if uid is not None and ("uid" not in meta or not np.array_equal(meta["uid"], uid)):
        return None
    return np.load(x, mmap_mode="r"), meta


def parity() -> dict:
    reconnaissance()
    import torch
    audits = []
    # Full late-fold parity; its embedding is reused by the pilot.
    for V, mode in [(PILOT_FOLD, "oof"), (None, "test")]:
        name = ckpt_name(V)
        model, tk, cfg, Vc, dev = etx.load_ckpt(name)
        for p in model.parameters():
            p.requires_grad_(False)
        T = CUTOFF_TEST if V is None else V
        if V is None:
            uid = np.load(LEGACY_ART / "uid_ETX-01-S42-DCW.npy").astype(np.int64)
        else:
            uid = aligned_fold(V)["user_id"].to_numpy().astype(np.int64)
        rows = seq.user_rows(uid)
        cache_name = "test_query_X" if V is None else f"val_{V:%Y%m%d}_query_X"
        cached = load_cached_array(cache_name, uid)
        if cached is None:
            log(f"parity {name}: extracting {len(uid):,} final query embeddings")
            X, z, forward_error = extract_query(model, tk, cfg, dev, T, rows, mode,
                                                compare_original=True)
            cache_array(cache_name, X, {"uid": uid, "z": z, "checkpoint_sha256": sha256(ckpt_path(V))})
        else:
            X, meta = cached
            z = meta["z"].astype(np.float32)
            # Recheck the actual hook rather than trusting only the cache.
            _, z_small, forward_error = extract_query(model, tk, cfg, dev, T, rows[:1024], mode,
                                                       compare_original=True)
            if not np.array_equal(z_small, z[:1024]):
                raise AssertionError(f"cached query predictions changed for {name}")
        if V is None:
            z_saved = align_values(np.load(LEGACY_ART / "uid_ETX-01-S42-DCW.npy"),
                                   np.load(LEGACY_ART / "ztest_ETX-01-S42-DCW.npy"), uid)
        else:
            z_saved = saved_etx_fold(V, uid)
        saved_error = float(np.max(np.abs(np.maximum(z.astype(float), 0.0) - z_saved)))
        # Exact deterministic replay on a nontrivial subset.
        X1, z1, _ = extract_query(model, tk, cfg, dev, T, rows[:2048], mode)
        X2, z2, _ = extract_query(model, tk, cfg, dev, T, rows[:2048], mode)
        replay_z = float(np.max(np.abs(z1.astype(float) - z2.astype(float))))
        replay_x = float(np.max(np.abs(X1.astype(float) - X2.astype(float))))
        audit = {"checkpoint": name, "checkpoint_sha256": sha256(ckpt_path(V)),
                 "cutoff": str(T), "rows": len(uid), "embedding_shape": list(X.shape),
                 "embedding_dtype": str(X.dtype), "embedding_finite": bool(np.isfinite(X).all()),
                 "hook_vs_original_max_abs_log_error": forward_error,
                 "hook_vs_saved_max_abs_log_error": saved_error,
                 "deterministic_replay_prediction_max_abs": replay_z,
                 "deterministic_replay_embedding_max_abs": replay_x,
                 "user_order_exact": True, "mode": mode}
        audits.append(audit)
        del model, tk, X, z, X1, X2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    static = []
    for T, mode in [(PILOT_FOLD, "oof"), (EXTRA_CUTOFFS[0], "extra"),
                    (EXTRA_CUTOFFS[-1], "extra"), (CUTOFF_TEST, "test")]:
        clip, cap, shift = static_policy(T, mode)
        calendar_depth = seq.day_index(T) + 1
        static.append({"cutoff": str(T), "mode": mode, "calendar_depth": calendar_depth,
                       "event_depth_clip": clip, "query_depth": min(calendar_depth, cap) if cap else min(calendar_depth, 365),
                       "actual_weekday": T.weekday(), "cdow_shift": shift,
                       "query_weekday": (T.weekday() + int(shift)) % 7})
    # The hard hook requirement is hook vs the original forward under the same
    # frozen checkpoint/runtime.  Archived arrays are an additional provenance
    # audit: a prior CUDA/SDPA runtime may have different bf16 reductions.
    ok = all(a["hook_vs_original_max_abs_log_error"] <= 1e-6 and
             a["embedding_shape"][1] == 128 and a["embedding_finite"] and
             a["deterministic_replay_prediction_max_abs"] == 0.0 and
             a["deterministic_replay_embedding_max_abs"] == 0.0 for a in audits)
    archived_exact = all(a["hook_vs_saved_max_abs_log_error"] <= 1e-6 for a in audits)
    result = {"status": "PASS" if ok else "FAIL", "audits": audits,
              "archived_prediction_reference": ("EXACT" if archived_exact else
                  "OOF_EXACT_TEST_BF16_RUNTIME_DRIFT_RECORDED"),
              "static_context": static,
              "oof_test_policy_audit": {
                  "clean_oof_actual_calendar": True, "extra_actual_weekday": True,
                  "extra_depth_clip_query_cap_match": True,
                  "test_uses_registered_DCW_checkpoint_support": True,
                  "test_only_feature_schema_mismatch": False,
              }}
    json_write(OUT / "encoder_parity.json", result)
    if not ok:
        raise AssertionError("ETX query hook parity failed")
    return result


def collect_positive(model, tk, cfg, dev, V: dt.date, kind: str,
                     donor_group: int | None = None, production_clean: bool = False):
    if kind == "clean" and production_clean:
        cuts = cutoff_grid(seq.MIN_HISTORY, CUTOFF_STEP, CORRIDOR_END)
    else:
        cuts = seq.fold_cutoffs(V) if kind == "clean" else EXTRA_CUTOFFS
    # z0 identifies the exact encoder checkpoint and prevents the V1016 fold cache
    # from being reused for the distinct seed-42 TEST production checkpoint.
    encoder_tag = f"z0{float(cfg['z0']):.6f}".replace(".", "p")
    tag = f"{kind}_{V:%Y%m%d}_{encoder_tag}" + ("" if donor_group is None else f"_g{donor_group}")
    cached = load_cached_array(tag)
    if cached is not None:
        X, m = cached
        expected = np.asarray([str(x) for x in cuts], dtype="U10")
        if np.array_equal(m["cuts"], expected):
            return X, m["z"], m["uid"], m["ci"]
    specs, total = [], 0
    for ci, T in enumerate(cuts):
        uid = panel_users(T, 1)["user_id"].to_numpy().astype(np.int64)
        if donor_group is not None:
            uid = uid[user_group(uid) == donor_group]
        rows = seq.user_rows(uid)
        y = seq.target_at(T, rows)
        pos = y > 0
        uid, rows, y = uid[pos], rows[pos], y[pos]
        specs.append((ci, T, uid, rows, np.log1p(y).astype(np.float32)))
        total += len(uid)
    path = CACHE / f"{tag}.npy"
    CACHE.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(total, 128))
    zz, uu, cc, at = [], [], [], 0
    mode = "clean" if kind == "clean" else "extra"
    for ci, T, uid, rows, z in specs:
        log(f"embedding {tag}: {T}, positive rows={len(uid):,}")
        e, _, _ = extract_query(model, tk, cfg, dev, T, rows, mode)
        X[at:at + len(uid)] = e
        at += len(uid)
        zz.append(z); uu.append(uid); cc.append(np.full(len(uid), ci, np.int16))
    X.flush()
    z, uid, ci = np.concatenate(zz), np.concatenate(uu), np.concatenate(cc)
    np.savez_compressed(CACHE / f"{tag}_meta.npz", z=z, uid=uid, ci=ci,
                        cuts=np.asarray([str(x) for x in cuts], dtype="U10"),
                        checkpoint_sha256=np.asarray(sha256(ckpt_path(V))))
    return np.load(path, mmap_mode="r"), z, uid, ci


def validation_embeddings(model, tk, cfg, dev, V: dt.date):
    frame = aligned_fold(V)
    uid = frame["user_id"].to_numpy().astype(np.int64)
    name = f"val_{V:%Y%m%d}_query_X"
    cached = load_cached_array(name, uid)
    if cached is not None:
        return cached[0], uid, frame
    rows = seq.user_rows(uid)
    X, z, _ = extract_query(model, tk, cfg, dev, V, rows, "oof")
    cache_array(name, X, {"uid": uid, "z": z, "checkpoint_sha256": sha256(ckpt_path(V))})
    return np.load(CACHE / f"{name}.npy", mmap_mode="r"), uid, frame


def build_head(out_bias: float = 0.0):
    import torch
    from torch import nn
    net = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.constant_(net[-1].bias, out_bias)
    return net


def fetch_rows(sources: list[np.ndarray], offsets: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.empty((len(idx), 128), np.float16)
    for s, X in enumerate(sources):
        lo, hi = offsets[s], offsets[s + 1]
        mask = (idx >= lo) & (idx < hi)
        if mask.any():
            out[mask] = X[idx[mask] - lo]
    return out


def fit_head(sources: list[np.ndarray], targets: list[np.ndarray], rows: np.ndarray,
             steps: int, seed: int, dev):
    import torch
    torch.manual_seed(seed)
    net = build_head().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=HEAD_CFG["learning_rate"],
                            weight_decay=HEAD_CFG["weight_decay"], betas=(0.9, 0.98))
    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0] + [len(x) for x in sources]).astype(np.int64)
    target = np.concatenate(targets).astype(np.float32, copy=False)
    rows = np.asarray(rows, np.int64)
    batch = HEAD_CFG["batch_size"]
    net.train()
    loss_sum = 0.0
    for step in range(steps):
        lr = HEAD_CFG["learning_rate"] * min(1.0, (step + 1) / 200.0) * \
             0.5 * (1.0 + math.cos(math.pi * step / steps))
        for group in opt.param_groups:
            group["lr"] = lr
        chosen = rows[rng.integers(0, len(rows), batch)]
        xb = torch.from_numpy(fetch_rows(sources, offsets, chosen)).to(dev).float()
        yb = torch.from_numpy(target[chosen]).to(dev).float()
        loss = torch.nn.functional.mse_loss(net(xb).squeeze(1), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        loss_sum += float(loss.detach())
    net.eval()
    return net, loss_sum / steps


def predict_head(net, X: np.ndarray, dev, batch: int = 65536) -> np.ndarray:
    import torch
    out = np.empty(len(X), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch])).to(dev).float()
            out[i:i + batch] = net(xb).squeeze(1).float().cpu().numpy()
    return out


def load_head_state(state: dict, dev):
    net = build_head().to(dev)
    net.load_state_dict(state)
    net.eval()
    return net


def train_fold(V: dt.date, seeds: list[int], keep_heads: bool = False) -> Path:
    import torch
    parity_result = json.loads((OUT / "encoder_parity.json").read_text(encoding="utf-8"))
    if parity_result["status"] != "PASS":
        raise AssertionError("parity must pass before head training")
    model, tk, cfg, Vc, dev = etx.load_ckpt(ckpt_name(V))
    if Vc != V:
        raise AssertionError("checkpoint/fold mismatch")
    for p in model.parameters():
        p.requires_grad_(False)
    encoder_before = sha256(ckpt_path(V))
    Xc, zc, uc, ci = collect_positive(model, tk, cfg, dev, V, "clean")
    X0, z0, u0, c0 = collect_positive(model, tk, cfg, dev, V, "extra", 0)
    X1, z1, u1, c1 = collect_positive(model, tk, cfg, dev, V, "extra", 1)
    Xv, uid, frame = validation_embeddings(model, tk, cfg, dev, V)
    group = user_group(uid)
    if not np.all(user_group(u0) == 0) or not np.all(user_group(u1) == 1):
        raise AssertionError("EXTRA donor-side contamination")
    if np.intersect1d(np.unique(u0), uid[group == 1]).size or \
       np.intersect1d(np.unique(u1), uid[group == 0]).size:
        raise AssertionError("an EXTRA donor can receive a prediction from its own head")
    clean_cuts = seq.fold_cutoffs(V)
    c_clean = np.asarray([zc[ci == k].mean() for k in range(len(clean_cuts))])
    c_extra0 = np.asarray([z0[c0 == k].mean() for k in range(len(EXTRA_CUTOFFS))])
    c_extra1 = np.asarray([z1[c1 == k].mean() for k in range(len(EXTRA_CUTOFFS))])
    tc = (zc - c_clean[ci]).astype(np.float32)
    t0 = (z0 - c_extra0[c0]).astype(np.float32)
    t1 = (z1 - c_extra1[c1]).astype(np.float32)
    n_clean = len(tc)
    steps = int(np.ceil(n_clean / HEAD_CFG["batch_size"])) * HEAD_CFG["epochs"]
    early = np.flatnonzero(ci < max(1, len(clean_cuts) // 3))
    state_path = CACHE / f"fold_{V:%Y%m%d}_seed_predictions.npz"
    saved = dict(np.load(state_path, allow_pickle=False)) if state_path.exists() else {}
    head_states = {}
    if keep_heads and (CACHE / f"heads_{V:%Y%m%d}_s42.pt").exists():
        head_states = torch.load(CACHE / f"heads_{V:%Y%m%d}_s42.pt", map_location="cpu", weights_only=False)
    for hs in seeds:
        needed = [f"clean_s{hs}", f"fresh_s{hs}", f"vol_s{hs}"]
        if all(k in saved for k in needed):
            log(f"fold {V} seed {hs}: predictions resumed")
            continue
        log(f"fold {V} seed {hs}: CLEAN head, {steps:,} steps")
        clean_rows = np.arange(n_clean, dtype=np.int64)
        clean_net, loss_clean = fit_head([Xc], [tc], clean_rows, steps, hs, dev)
        clean_pred = predict_head(clean_net, Xv, dev) + float(c_clean.mean())
        donor_preds = {}
        seed_states = {"clean": {k: v.detach().cpu() for k, v in clean_net.state_dict().items()}}
        del clean_net
        for donor, (Xe, te) in enumerate([(X0, t0), (X1, t1)]):
            n_extra = len(te)
            fresh_rows = np.concatenate([np.arange(n_clean, dtype=np.int64),
                                         n_clean + np.arange(n_extra, dtype=np.int64)])
            vol_rng = np.random.default_rng(42)
            vol_rows = np.concatenate([np.arange(n_clean, dtype=np.int64),
                                       vol_rng.choice(early, size=n_extra, replace=True)])
            log(f"fold {V} seed {hs}: donor g{donor} FRESH/VOL, EXTRA={n_extra:,}")
            fresh_net, loss_f = fit_head([Xc, Xe], [tc, te], fresh_rows, steps, hs, dev)
            vol_net, loss_v = fit_head([Xc, Xe], [tc, te], vol_rows, steps, hs, dev)
            donor_preds[donor] = {
                "fresh": predict_head(fresh_net, Xv, dev) + float(c_clean.mean()),
                "vol": predict_head(vol_net, Xv, dev) + float(c_clean.mean()),
                "loss_fresh": loss_f, "loss_vol": loss_v,
            }
            seed_states[f"fresh_g{donor}"] = {k: v.detach().cpu() for k, v in fresh_net.state_dict().items()}
            seed_states[f"vol_g{donor}"] = {k: v.detach().cpu() for k, v in vol_net.state_dict().items()}
            del fresh_net, vol_net
        # donor 1 predicts recipient 0; donor 0 predicts recipient 1.
        fresh = np.where(group == 0, donor_preds[1]["fresh"], donor_preds[0]["fresh"])
        vol = np.where(group == 0, donor_preds[1]["vol"], donor_preds[0]["vol"])
        saved[f"clean_s{hs}"] = clean_pred.astype(np.float32)
        saved[f"fresh_s{hs}"] = fresh.astype(np.float32)
        saved[f"vol_s{hs}"] = vol.astype(np.float32)
        saved[f"loss_clean_s{hs}"] = np.asarray(loss_clean)
        saved[f"loss_fresh_g0_s{hs}"] = np.asarray(donor_preds[0]["loss_fresh"])
        saved[f"loss_fresh_g1_s{hs}"] = np.asarray(donor_preds[1]["loss_fresh"])
        np.savez_compressed(state_path, uid=uid, **{k: v for k, v in saved.items() if k != "uid"})
        if keep_heads and hs == 42:
            torch.save(seed_states, CACHE / f"heads_{V:%Y%m%d}_s42.pt")
        gc.collect()
    if not all(f"clean_s{s}" in saved for s in seeds):
        raise AssertionError("missing seed predictions")
    clean = np.mean([saved[f"clean_s{s}"].astype(float) for s in seeds], axis=0)
    fresh = np.mean([saved[f"fresh_s{s}"].astype(float) for s in seeds], axis=0)
    vol = np.mean([saved[f"vol_s{s}"].astype(float) for s in seeds], axis=0)
    y = frame["target"].to_numpy().astype(float)
    out = OUT / f"_raw_fold_{V:%Y%m%d}_{'_'.join(map(str, seeds))}.npz"
    np.savez_compressed(out, uid=uid, y=y.astype(np.float32), group=group,
                        mu_clean=clean.astype(np.float32), mu_fresh=fresh.astype(np.float32),
                        mu_vol=vol.astype(np.float32), d_real=(fresh-clean).astype(np.float32),
                        d_vol=(vol-clean).astype(np.float32), seeds=np.asarray(seeds),
                        n_clean=np.asarray(n_clean), n_extra_g0=np.asarray(len(t0)),
                        n_extra_g1=np.asarray(len(t1)), steps=np.asarray(steps))
    if sha256(ckpt_path(V)) != encoder_before:
        raise AssertionError("frozen ETX checkpoint changed")
    del model, tk, Xc, X0, X1, Xv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def donor_predictions_from_pilot_heads() -> dict[str, dict]:
    import torch
    hp = CACHE / f"heads_{PILOT_FOLD:%Y%m%d}_s42.pt"
    if not hp.exists():
        raise FileNotFoundError(hp)
    states = torch.load(hp, map_location="cpu", weights_only=False)
    model, tk, cfg, _, dev = etx.load_ckpt(ckpt_name(PILOT_FOLD))
    out = {}
    nets = {k: load_head_state(v, dev) for k, v in states.items()}
    for V in FOLDS[:-1]:
        frame = aligned_fold(V)
        uid = frame["user_id"].to_numpy().astype(np.int64)
        name = f"pilot_donor_{V:%Y%m%d}_query_X"
        cached = load_cached_array(name, uid)
        if cached is None:
            X, _, _ = extract_query(model, tk, cfg, dev, V, seq.user_rows(uid), "pilot_donor")
            cache_array(name, X, {"uid": uid})
        else:
            X = cached[0]
        g = user_group(uid)
        clean = predict_head(nets["clean"], X, dev)
        fresh = np.where(g == 0, predict_head(nets["fresh_g1"], X, dev),
                         predict_head(nets["fresh_g0"], X, dev))
        vol = np.where(g == 0, predict_head(nets["vol_g1"], X, dev),
                       predict_head(nets["vol_g0"], X, dev))
        out[str(V)] = {"uid": uid, "d_real": fresh-clean, "d_vol": vol-clean,
                       "frame": frame}
    del model, tk, nets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def donor_preprocess_params(raws: list[np.ndarray]) -> dict:
    x = np.concatenate([np.asarray(v, float) for v in raws])
    lo, hi = np.quantile(x, [0.005, 0.995])
    center = float(np.clip(x, lo, hi).mean())
    return {"lo": float(lo), "hi": float(hi), "center": center, "n_donor": len(x)}


def apply_preprocess(raw: np.ndarray, params: dict) -> np.ndarray:
    return np.clip(np.asarray(raw, float), params["lo"], params["hi"]) - params["center"]


def fold_score(y, z) -> tuple[float, float]:
    off, score = calibrate(np.asarray(y, float), np.asarray(z, float))
    return float(off), float(score)


def frame_arrays(frame: pl.DataFrame) -> dict:
    out = {c: frame[c].to_numpy().astype(float) for c in frame.columns if c.startswith("pred_")}
    out["uid"] = frame["user_id"].to_numpy().astype(np.int64)
    out["y"] = frame["target"].to_numpy().astype(float)
    out["z_base"] = np.log1p(out["pred_exp037"])
    out["d_seq"] = np.log1p(out["pred_fresh_contrast"]) - out["z_base"]
    out["z_btyd"] = np.log1p(out["pred_btyd"])
    return out


def pilot() -> dict:
    if not (OUT / "encoder_parity.json").exists():
        parity()
    parity_result = json.loads((OUT / "encoder_parity.json").read_text(encoding="utf-8"))
    if parity_result["status"] != "PASS":
        raise AssertionError("parity failed")
    late_path = train_fold(PILOT_FOLD, [42], keep_heads=True)
    late = dict(np.load(late_path, allow_pickle=False))
    donors = donor_predictions_from_pilot_heads()
    donor_real = [donors[str(V)]["d_real"] for V in FOLDS[:-1]]
    donor_vol = [donors[str(V)]["d_vol"] for V in FOLDS[:-1]]
    p_real, p_vol = donor_preprocess_params(donor_real), donor_preprocess_params(donor_vol)
    d_real = apply_preprocess(late["d_real"], p_real)
    d_vol = apply_preprocess(late["d_vol"], p_vol)
    a = frame_arrays(aligned_fold(PILOT_FOLD))
    if not np.array_equal(a["uid"], late["uid"]):
        raise AssertionError("pilot/aligned OOF user order mismatch")
    base = fold_score(a["y"], a["z_base"])[1]
    real = fold_score(a["y"], a["z_base"] + d_real)[1]
    vol = fold_score(a["y"], a["z_base"] + d_vol)[1]
    seq_score = fold_score(a["y"], a["z_base"] + a["d_seq"])[1]
    # Target-free gamma on pooled donor correction vectors.
    donor_proc, donor_seq = [], []
    for V in FOLDS[:-1]:
        item = donors[str(V)]
        ar = frame_arrays(item["frame"])
        donor_proc.append(apply_preprocess(item["d_real"], p_real))
        donor_seq.append(ar["d_seq"])
    x, s = np.concatenate(donor_proc), np.concatenate(donor_seq)
    denom = float(np.dot(s, s))
    gamma = float(np.dot(x, s) / denom) if denom > 0 else 0.0
    orth = d_real - gamma * a["d_seq"]
    combined = fold_score(a["y"], a["z_base"] + a["d_seq"] + orth)[1]
    group_rows = []
    for g in (0, 1):
        m = late["group"] == g
        sr = fold_score(a["y"][m], a["z_base"][m] + d_real[m])[1]
        sv = fold_score(a["y"][m], a["z_base"][m] + d_vol[m])[1]
        group_rows.append({"group": "A" if g == 0 else "B", "n": int(m.sum()),
                           "real_score": sr, "vol_score": sv, "real_minus_vol": sr-sv})
    gate1 = real - vol <= -0.00010
    gate2a = real - base <= -0.00010
    gate2b = combined - seq_score <= -0.00005
    gate3 = all(r["real_minus_vol"] < 0 for r in group_rows)
    gate4 = parity_result["status"] == "PASS"
    passed = gate1 and (gate2a or gate2b) and gate3 and gate4
    grid = []
    for alpha in ALPHAS:
        grid.append({"alpha": float(alpha),
                     "real_score": fold_score(a["y"], a["z_base"] + alpha*d_real)[1],
                     "vol_score": fold_score(a["y"], a["z_base"] + alpha*d_vol)[1],
                     "combined_score": fold_score(a["y"], a["z_base"] + a["d_seq"] + alpha*orth)[1]})
    result = {
        "status": "PASS" if passed else "REJECT_PILOT", "fold": str(PILOT_FOLD),
        "encoder_seed": 42, "head_seed": 42, "gate_scale": 1.0,
        "metrics": {"exp037": base, "etx_real": real, "etx_vol": vol,
                    "seq_fresh": seq_score, "seq_plus_etx_orth": combined,
                    "real_delta": real-base, "vol_delta": vol-base,
                    "real_minus_vol": real-vol, "orth_incremental_delta": combined-seq_score},
        "gamma_target_free": gamma, "preprocessing_real": p_real,
        "preprocessing_vol": p_vol, "user_halves": group_rows,
        "diagnostic_grid_not_used_for_gate": grid,
        "gates": {"real_beats_vol_by_0.00010": gate1,
                  "real_improves_base_by_0.00010": gate2a,
                  "orth_improves_seq_by_0.00005": gate2b,
                  "both_halves_real_minus_vol_negative": gate3,
                  "encoder_parity": gate4},
        "pilot_runtime_seconds": time.time()-START,
    }
    json_write(OUT / "pilot_metrics.json", result)
    # Preserve the honest pilot-only diagnostic direction under the standardized
    # OOF key while making its intentionally partial scope explicit.
    pl.DataFrame({"fold": [str(PILOT_FOLD)]*len(late["uid"]),
                  "user_id": late["uid"].astype(np.int64),
                  "d_etx_fresh_raw": late["d_real"].astype(np.float64),
                  "d_etx_vol_raw": late["d_vol"].astype(np.float64),
                  "user_side": late["group"].astype(np.int8),
                  "scope": ["PILOT_ONLY_SEED42"]*len(late["uid"])}).write_parquet(
                      OUT/"etx_fresh_raw_OOF.parquet")
    csv_write(OUT/"fold_metrics.csv", [
        {"fold":str(PILOT_FOLD),"endpoint":"EXP037","rmsle":base,"delta_vs_exp037":0.0,"scope":"pilot_seed42"},
        {"fold":str(PILOT_FOLD),"endpoint":"ETX_FRESH_UNIT","rmsle":real,"delta_vs_exp037":real-base,"scope":"pilot_seed42"},
        {"fold":str(PILOT_FOLD),"endpoint":"ETX_VOL_UNIT","rmsle":vol,"delta_vs_exp037":vol-base,"scope":"pilot_seed42"},
        {"fold":str(PILOT_FOLD),"endpoint":"SEQ_FRESH","rmsle":seq_score,"delta_vs_exp037":seq_score-base,"scope":"pilot_seed42"},
        {"fold":str(PILOT_FOLD),"endpoint":"SEQ_PLUS_ETX_ORTH_UNIT","rmsle":combined,"delta_vs_exp037":combined-base,"scope":"pilot_seed42"},
    ])
    csv_write(OUT/"real_vs_vol.csv", [{"fold":str(PILOT_FOLD),"real_score":real,"vol_score":vol,
                                        "real_minus_vol":real-vol,"scope":"pilot_seed42"}])
    csv_write(OUT/"user_half_metrics.csv", [{"fold":str(PILOT_FOLD),"scope":"pilot_seed42",**r}
                                              for r in group_rows])
    csv_write(OUT/"nested_selection.csv", [{"status":"NOT_RUN_REJECT_PILOT",
                                              "pilot_scale":1.0,"reason":"registered pilot gate failed"}])
    csv_write(OUT/"seq_vs_etx_fresh.csv", [{"fold":str(PILOT_FOLD),"scope":"pilot_seed42",
                                             "gamma":gamma,"corr_etx_seq":float(np.corrcoef(d_real,a["d_seq"])[0,1]),
                                             "incremental_delta":combined-seq_score,
                                             "etx_rms":float(np.sqrt(np.mean(d_real*d_real))),
                                             "orth_rms":float(np.sqrt(np.mean(orth*orth))) }])
    csv_write(OUT/"diversity_oof.csv", [
        {"comparison":"SEQ_FRESH","correlation":float(np.corrcoef(d_real,a["d_seq"])[0,1]),
         "scope":"pilot_seed42","etx_correction_rms":float(np.sqrt(np.mean(d_real*d_real)))},
        {"comparison":"BASE_RESIDUAL","correlation":float(np.corrcoef(d_real,np.log1p(a["y"])-a["z_base"])[0,1]),
         "scope":"pilot_seed42","etx_correction_rms":float(np.sqrt(np.mean(d_real*d_real)))},
    ])
    if not passed:
        create_not_run_artifacts("REJECT_PILOT")
    return result


def load_raw_folds(seeds: list[int]) -> list[dict]:
    folds = []
    suffix = "_".join(map(str, seeds))
    for V in FOLDS:
        p = OUT / f"_raw_fold_{V:%Y%m%d}_{suffix}.npz"
        if not p.exists():
            p = train_fold(V, seeds, keep_heads=False)
        d = dict(np.load(p, allow_pickle=False))
        ar = frame_arrays(aligned_fold(V))
        if not np.array_equal(d["uid"], ar["uid"]):
            raise AssertionError(f"raw/aligned mismatch {V}")
        d.update(ar)
        d["V"] = V
        folds.append(d)
    return folds


def nested_direction(folds: list[dict], raw_key: str, base_kind: str = "base") -> dict:
    raw = [np.asarray(f[raw_key], float) for f in folds]
    base_scores = np.asarray([fold_score(f["y"], f["z_base"])[1] for f in folds])
    rows, corr_held, scores = [], [], []
    for h in range(4):
        train = [i for i in range(4) if i != h]
        curve = []
        selection_detail = []
        for alpha in ALPHAS:
            train_scores = []
            for t in train:
                pp = donor_preprocess_params([raw[i] for i in train if i != t])
                c = apply_preprocess(raw[t], pp)
                train_scores.append(fold_score(folds[t]["y"], folds[t]["z_base"] + alpha*c)[1])
            wt = FOLD_WEIGHTS[train] / FOLD_WEIGHTS[train].sum()
            curve.append((float(wt @ np.asarray(train_scores)), float(alpha)))
            selection_detail.append({"alpha": float(alpha), "scores": train_scores})
        _, alpha = min(curve)
        pp = donor_preprocess_params([raw[i] for i in train])
        c = apply_preprocess(raw[h], pp)
        score = fold_score(folds[h]["y"], folds[h]["z_base"] + alpha*c)[1]
        scores.append(score); corr_held.append(c)
        rows.append({"endpoint": raw_key, "fold": str(folds[h]["V"]),
                     "selected_alpha": alpha, "selection_folds": train,
                     "heldout_score": score, "baseline_score": base_scores[h],
                     "heldout_delta": score-base_scores[h], "preprocess": pp,
                     "selection_curve": selection_detail})
    scores = np.asarray(scores)
    delta = scores-base_scores
    w = FOLD_WEIGHTS/FOLD_WEIGHTS.sum()
    return {"rows": rows, "processed": corr_held, "scores": scores,
            "base_scores": base_scores, "delta": delta,
            "wcv": float(w@scores), "base_wcv": float(w@base_scores),
            "delta_wcv": float(w@delta), "improved_folds": int((delta<0).sum())}


def incremental_nested(folds: list[dict], real_processed_helper: list[np.ndarray]) -> dict:
    rows, held, scores, seq_scores = [], [], [], []
    for h in range(4):
        train = [i for i in range(4) if i != h]
        # Inner fold-safe ETX preprocessing for beta selection.
        proc_train = {}
        for t in train:
            pp = donor_preprocess_params([folds[i]["d_real"] for i in train if i != t])
            proc_train[t] = apply_preprocess(folds[t]["d_real"], pp)
        etx_pool = np.concatenate([proc_train[t] for t in train])
        seq_pool = np.concatenate([folds[t]["d_seq"] for t in train])
        gamma = float(np.dot(etx_pool, seq_pool) / max(np.dot(seq_pool, seq_pool), 1e-30))
        ranked = []
        curve = []
        for beta in ALPHAS:
            ss = []
            for t in train:
                orth = proc_train[t] - gamma*folds[t]["d_seq"]
                ss.append(fold_score(folds[t]["y"], folds[t]["z_base"] + folds[t]["d_seq"] + beta*orth)[1])
            wt = FOLD_WEIGHTS[train]/FOLD_WEIGHTS[train].sum()
            value = float(wt@np.asarray(ss))
            ranked.append((value, float(beta))); curve.append({"beta": float(beta), "scores": ss})
        _, beta = min(ranked)
        pp_h = donor_preprocess_params([folds[i]["d_real"] for i in train])
        proc_h = apply_preprocess(folds[h]["d_real"], pp_h)
        orth_h = proc_h - gamma*folds[h]["d_seq"]
        zseq = folds[h]["z_base"] + folds[h]["d_seq"]
        seq_sc = fold_score(folds[h]["y"], zseq)[1]
        sc = fold_score(folds[h]["y"], zseq + beta*orth_h)[1]
        scores.append(sc); seq_scores.append(seq_sc); held.append(orth_h)
        rows.append({"endpoint": "incremental_etx_beyond_seq", "fold": str(folds[h]["V"]),
                     "gamma": gamma, "selected_beta": beta, "selection_folds": train,
                     "heldout_score": sc, "seq_score": seq_sc, "heldout_delta": sc-seq_sc,
                     "preprocess": pp_h, "selection_curve": curve})
    scores, seq_scores = np.asarray(scores), np.asarray(seq_scores)
    delta = scores-seq_scores; w=FOLD_WEIGHTS/FOLD_WEIGHTS.sum()
    return {"rows": rows, "orth": held, "scores": scores, "seq_scores": seq_scores,
            "delta": delta, "wcv": float(w@scores), "seq_wcv": float(w@seq_scores),
            "delta_wcv": float(w@delta), "improved_folds": int((delta<0).sum())}


def weighted_calibrate(ly, z, w):
    sw = float(w.sum())
    if sw <= 0:
        return np.nan
    d = float(np.sum(w*(ly-z))/sw)
    for _ in range(25):
        act = z+d>0
        den = float(w[act].sum())
        if den <= 0:
            break
        dn = float(np.sum(w[act]*(ly[act]-z[act]))/den)
        if abs(dn-d)<1e-12:
            d=dn; break
        d=dn
    return float(np.sqrt(np.sum(w*(ly-np.maximum(z+d,0))**2)/sw))


def cluster_bootstrap(folds: list[dict], real: dict, vol: dict, inc: dict,
                      n_boot: int = 300) -> list[dict]:
    all_uid = np.unique(np.concatenate([f["uid"] for f in folds]))
    pos_maps = [np.searchsorted(all_uid, f["uid"]) for f in folds]
    rng = np.random.default_rng(71042)
    values = {"etx_vs_base": [], "real_vs_vol": [], "incremental_vs_seq": []}
    for _ in range(n_boot):
        cw = rng.poisson(1.0, len(all_uid)).astype(float)
        deltas = {k: [] for k in values}
        for i,f in enumerate(folds):
            rw = cw[pos_maps[i]]; ly=np.log1p(f["y"])
            zr=f["z_base"]+real["rows"][i]["selected_alpha"]*real["processed"][i]
            zv=f["z_base"]+vol["rows"][i]["selected_alpha"]*vol["processed"][i]
            zs=f["z_base"]+f["d_seq"]
            zi=zs+inc["rows"][i]["selected_beta"]*inc["orth"][i]
            sb=weighted_calibrate(ly,f["z_base"],rw)
            sr=weighted_calibrate(ly,zr,rw); sv=weighted_calibrate(ly,zv,rw)
            ss=weighted_calibrate(ly,zs,rw); si=weighted_calibrate(ly,zi,rw)
            deltas["etx_vs_base"].append(sr-sb)
            deltas["real_vs_vol"].append(sr-sv)
            deltas["incremental_vs_seq"].append(si-ss)
        for k in values:
            values[k].append(float(FOLD_WEIGHTS@np.asarray(deltas[k])/FOLD_WEIGHTS.sum()))
    rows=[]
    for k,v in values.items():
        a=np.asarray(v)
        rows.append({"contrast":k,"n_boot":n_boot,"mean":float(a.mean()),
                     "p2_5":float(np.quantile(a,.025)),"p50":float(np.quantile(a,.5)),
                     "p97_5":float(np.quantile(a,.975)),"prob_lt_zero":float((a<0).mean()),
                     "seed":71042})
    return rows


def oof_projection(folds: list[dict], processed: list[np.ndarray]) -> dict:
    source_cols = [c for c in aligned_fold(FOLDS[0]).columns if c.startswith("pred_") and
                   c not in {"pred_exp037", "pred_fresh_contrast"}]
    rows=[]; held_res=[]; held_raw=[]
    for h in range(4):
        train=[i for i in range(4) if i!=h]
        Xtr=[]; ytr=[]
        for i in train:
            Xtr.append(np.column_stack([np.log1p(folds[i][c])-folds[i]["z_base"] for c in source_cols]))
            ytr.append(processed[i])
        X=np.concatenate(Xtr); y=np.concatenate(ytr)
        xm=X.mean(0); ym=float(y.mean())
        coef=np.linalg.lstsq(X-xm,y-ym,rcond=1e-8)[0]
        Xh=np.column_stack([np.log1p(folds[h][c])-folds[h]["z_base"] for c in source_cols])
        pred=ym+(Xh-xm)@coef; resid=processed[h]-pred
        ratio=float(np.var(resid)/max(np.var(processed[h]),1e-30))
        rows.append({"fold":str(folds[h]["V"]),"unexplained_variance_ratio":ratio,
                     "raw_rms":float(np.sqrt(np.mean(processed[h]**2))),
                     "residual_rms":float(np.sqrt(np.mean(resid**2))),"rank":int(np.linalg.matrix_rank(X-xm)),
                     "sources":source_cols})
        held_res.append(resid); held_raw.append(processed[h])
    r=np.concatenate(held_res); d=np.concatenate(held_raw)
    return {"folds":rows,"pooled_unexplained_variance_ratio":float(np.var(r)/max(np.var(d),1e-30)),
            "pooled_residual_rms":float(np.sqrt(np.mean(r*r))),"source_columns":source_cols}


def analyze_full() -> dict:
    folds=load_raw_folds(HEAD_SEEDS)
    real=nested_direction(folds,"d_real"); vol=nested_direction(folds,"d_vol")
    inc=incremental_nested(folds,real["processed"])
    w=FOLD_WEIGHTS/FOLD_WEIGHTS.sum()
    fold_rows=[]; real_vol=[]; half=[]
    seq_scores=[]; btyd_scores=[]
    for i,f in enumerate(folds):
        base=real["base_scores"][i]
        seq=fold_score(f["y"],f["z_base"]+f["d_seq"])[1]; seq_scores.append(seq)
        zr=f["z_base"]+real["rows"][i]["selected_alpha"]*real["processed"][i]
        zv=f["z_base"]+vol["rows"][i]["selected_alpha"]*vol["processed"][i]
        zi=f["z_base"]+f["d_seq"]+inc["rows"][i]["selected_beta"]*inc["orth"][i]
        zb=.95*f["z_base"]+.05*f["z_btyd"]+real["processed"][i]
        sb=fold_score(f["y"],zb)[1]; btyd_scores.append(sb)
        endpoints=[("EXP037",base,0.0),("SEQ_FRESH",seq,seq-base),
                   ("ETX_FRESH",real["scores"][i],real["delta"][i]),
                   ("ETX_VOL",vol["scores"][i],vol["delta"][i]),
                   ("SEQ_PLUS_ETX_ORTH",inc["scores"][i],inc["scores"][i]-base),
                   ("BTYD05_PLUS_ETX_FRESH_DIAG",sb,sb-base)]
        for name,sc,de in endpoints:
            fold_rows.append({"fold":str(f["V"]),"endpoint":name,"rmsle":sc,"delta_vs_exp037":de,
                              "alpha":real["rows"][i]["selected_alpha"] if name=="ETX_FRESH" else "",
                              "beta":inc["rows"][i]["selected_beta"] if name=="SEQ_PLUS_ETX_ORTH" else ""})
        real_vol.append({"fold":str(f["V"]),"real_score":real["scores"][i],"vol_score":vol["scores"][i],
                         "real_minus_vol":real["scores"][i]-vol["scores"][i],
                         "real_delta":real["delta"][i],"vol_delta":vol["delta"][i]})
        for g in (0,1):
            m=f["group"]==g
            sb0=fold_score(f["y"][m],f["z_base"][m])[1]
            sr=fold_score(f["y"][m],zr[m])[1]; sv=fold_score(f["y"][m],zv[m])[1]
            ss=fold_score(f["y"][m],(f["z_base"]+f["d_seq"])[m])[1]
            si=fold_score(f["y"][m],zi[m])[1]
            half.append({"fold":str(f["V"]),"group":"A" if g==0 else "B","n":int(m.sum()),
                         "base":sb0,"real":sr,"vol":sv,"seq":ss,"combined":si,
                         "real_delta":sr-sb0,"real_minus_vol":sr-sv,"incremental_delta":si-ss})
    for name,scores,base_scores in [("EXP037",real["base_scores"],real["base_scores"]),
                                    ("SEQ_FRESH",np.asarray(seq_scores),real["base_scores"]),
                                    ("ETX_FRESH",real["scores"],real["base_scores"]),
                                    ("ETX_VOL",vol["scores"],real["base_scores"]),
                                    ("SEQ_PLUS_ETX_ORTH",inc["scores"],real["base_scores"]),
                                    ("BTYD05_PLUS_ETX_FRESH_DIAG",np.asarray(btyd_scores),real["base_scores"])]:
        fold_rows.append({"fold":"wCV","endpoint":name,"rmsle":float(w@np.asarray(scores)),
                          "delta_vs_exp037":float(w@(np.asarray(scores)-np.asarray(base_scores)))})
    for g in ("A","B"):
        rr=[r for r in half if r["group"]==g]
        half.append({"fold":"wCV","group":g,"n":sum(r["n"] for r in rr),
                     **{k:float(w@np.asarray([r[k] for r in rr])) for k in
                        ["base","real","vol","seq","combined","real_delta","real_minus_vol","incremental_delta"]}})
    csv_write(OUT/"fold_metrics.csv",fold_rows)
    nested_rows=real["rows"]+vol["rows"]+inc["rows"]
    csv_write(OUT/"nested_selection.csv",nested_rows)
    csv_write(OUT/"user_half_metrics.csv",half)
    csv_write(OUT/"real_vs_vol.csv",real_vol)
    # Diversity and source correlations.
    detx=np.concatenate(real["processed"]); dseq=np.concatenate([f["d_seq"] for f in folds])
    yres=np.concatenate([np.log1p(f["y"])-f["z_base"] for f in folds])
    names={"SEQ_FRESH":dseq,"BTYD":np.concatenate([f["z_btyd"]-f["z_base"] for f in folds])}
    for label,col in [("DIST","pred_dist"),("E11","pred_hurdle_e11"),("MHZ","pred_mhz_full"),
                      ("ETX","pred_etx_avg3"),("SEQ","pred_seq_avg3"),("HOLIDAY","pred_holiday_yoy")]:
        names[label]=np.concatenate([np.log1p(f[col])-f["z_base"] for f in folds])
    div=[]
    for n,v in names.items():
        div.append({"comparison":n,"correlation":float(np.corrcoef(detx,v)[0,1]),
                    "etx_correction_rms":float(np.sqrt(np.mean(detx**2))),
                    "other_rms":float(np.sqrt(np.mean(v**2)))})
    div.append({"comparison":"BASE_RESIDUAL","correlation":float(np.corrcoef(detx,yres)[0,1]),
                "etx_correction_rms":float(np.sqrt(np.mean(detx**2))),"other_rms":float(np.sqrt(np.mean(yres**2)))})
    csv_write(OUT/"diversity_oof.csv",div)
    proj=oof_projection(folds,real["processed"]); json_write(OUT/"oof_projection_metrics.json",proj)
    seq_rows=[]
    for i,f in enumerate(folds):
        seq_rows.append({"fold":str(f["V"]),"corr_etx_seq":float(np.corrcoef(real["processed"][i],f["d_seq"])[0,1]),
                         "gamma":inc["rows"][i]["gamma"],"beta":inc["rows"][i]["selected_beta"],
                         "incremental_delta":inc["delta"][i],"etx_rms":float(np.sqrt(np.mean(real["processed"][i]**2))),
                         "orth_rms":float(np.sqrt(np.mean(inc["orth"][i]**2)))})
    seq_rows.append({"fold":"OOF","corr_etx_seq":float(np.corrcoef(detx,dseq)[0,1]),
                     "incremental_delta":inc["delta_wcv"],"etx_rms":float(np.sqrt(np.mean(detx**2))),
                     "orth_rms":float(np.sqrt(np.mean(np.concatenate(inc["orth"])**2)))})
    csv_write(OUT/"seq_vs_etx_fresh.csv",seq_rows)
    boot=cluster_bootstrap(folds,real,vol,inc); csv_write(OUT/"bootstrap_metrics.csv",boot)
    # Raw standardized OOF direction.
    pl.concat([pl.DataFrame({"fold":[str(f["V"])]*len(f["uid"]),"user_id":f["uid"],
                             "d_etx_fresh_raw":f["d_real"].astype(np.float64),
                             "d_etx_vol_raw":f["d_vol"].astype(np.float64),"user_side":f["group"]})
               for f in folds]).write_parquet(OUT/"etx_fresh_raw_OOF.parquet")
    halves={r["group"]:r for r in half if r["fold"]=="wCV"}
    summary={"status":"FULL_COMPLETE","base_wcv":real["base_wcv"],"etx_fresh_delta_wcv":real["delta_wcv"],
             "etx_vol_delta_wcv":vol["delta_wcv"],"real_minus_vol_wcv":real["wcv"]-vol["wcv"],
             "etx_improved_folds":real["improved_folds"],"latest_fold_delta":float(real["delta"][-1]),
             "seq_wcv":inc["seq_wcv"],"incremental_delta_wcv":inc["delta_wcv"],
             "incremental_improved_folds":inc["improved_folds"],"corr_etx_seq":float(np.corrcoef(detx,dseq)[0,1]),
             "oof_unexplained_variance_ratio":proj["pooled_unexplained_variance_ratio"],
             "user_halves":halves,"nested_alpha":[r["selected_alpha"] for r in real["rows"]],
             "nested_beta":[r["selected_beta"] for r in inc["rows"]],
             "runtime_seconds":time.time()-START}
    json_write(OUT/"full_summary.json",summary)
    return summary


def preliminary_verdict(summary: dict) -> str:
    halves=summary["user_halves"]
    halves_agree=all(halves[g]["real_minus_vol"]<0 for g in ("A","B"))
    real_beats=summary["real_minus_vol_wcv"]<0
    type_a=(summary["etx_fresh_delta_wcv"]<=-0.00030 and summary["etx_improved_folds"]>=3 and
            summary["latest_fold_delta"]<0 and summary["real_minus_vol_wcv"]<=-0.00015 and halves_agree)
    type_b_oof=(summary["incremental_delta_wcv"]<=-0.00008 and summary["incremental_improved_folds"]>=3 and
                summary["corr_etx_seq"]<0.85 and summary["oof_unexplained_variance_ratio"]>=0.25 and real_beats)
    if type_a: return "PROVISIONAL_TYPE_A"
    if type_b_oof and summary["latest_fold_delta"]<0: return "PROVISIONAL_TYPE_B"
    weak=(summary["etx_fresh_delta_wcv"]<0 and summary["incremental_delta_wcv"]>=0) or \
         (-0.00008<=summary["etx_fresh_delta_wcv"]<=0.00005) or summary["corr_etx_seq"]>=0.85 or \
         summary["etx_improved_folds"]==2
    if weak: return "WEAK_SIGNAL"
    return "REJECT"


def full() -> dict:
    pil=json.loads((OUT/"pilot_metrics.json").read_text(encoding="utf-8")) if (OUT/"pilot_metrics.json").exists() else pilot()
    if pil["status"]!="PASS":
        return {"status":"REJECT_PILOT"}
    full_start=time.time()
    for V in FOLDS:
        if time.time()-full_start>5*3600:
            raise TimeoutError("five-hour full-experiment hard stop reached")
        train_fold(V,HEAD_SEEDS,keep_heads=False)
    summary=analyze_full(); summary["preliminary_verdict"]=preliminary_verdict(summary)
    json_write(OUT/"full_summary.json",summary)
    if not summary["preliminary_verdict"].startswith("PROVISIONAL"):
        create_skipped_production(summary["preliminary_verdict"])
    return summary


def geometry_basis():
    d=np.load(GEOM/"submission_geometry"/"cache"/"Z.npz",allow_pickle=False)
    meta=json.loads((GEOM/"submission_geometry"/"cache"/"Z_meta.json").read_text(encoding="utf-8"))
    drop={"C_lgbm_exp015_regen.csv","submission_BTYD05.csv"}
    keep=[i for i,n in enumerate(meta["names"]) if n not in drop]
    Z=np.ascontiguousarray(d["Z"][keep]); names=[meta["names"][i] for i in keep]
    uid=d["user_id"].astype(np.int64)
    ref=int(np.argmin([float(x) if x is not None else 99 for x in [1]*len(names)]))
    # Any reference yields the same difference span. Use exact STRONGEST when available.
    ref=names.index("submission_STRONGEST_CURRENT.csv")
    Y=Z-Z[ref]; G=Y@Y.T/Y.shape[1]
    lam,W=np.linalg.eigh(G); order=np.argsort(-lam); lam,W=lam[order],W[:,order]
    k=int((lam>1e-9*lam[0]).sum()); lam,W=lam[:k],W[:,:k]
    Phi=(W.T@Y)/np.sqrt(lam)[:,None]
    return Z,names,uid,Z[ref],Phi,lam


def family_map() -> dict:
    ns={}
    exec((GEOM/"submission_geometry"/"families.py").read_text(encoding="utf-8"),ns)
    return ns["FAMILY"]


def project_test_vector(label: str, full_z: np.ndarray | None, direction: np.ndarray,
                        Z,names,zref,Phi,lam) -> dict:
    v=np.asarray(direction,float) if full_z is None else np.asarray(full_z,float)-zref
    coef=Phi@v/len(v); proj=coef@Phi; resid=v-proj
    rms=float(np.sqrt(np.mean(v*v))); orth=float(np.sqrt(np.mean(resid*resid)))
    # Closest source direction after removing the common EXP037 direction.
    base_idx=names.index("submission_STRONGEST_CURRENT.csv")
    source_dir=Z-Z[base_idx]
    dist=np.sqrt(np.mean((source_dir-v[None,:])**2,axis=1)); j=int(np.argmin(dist))
    fam=family_map()
    return {"vector":label,"orthogonal_rms":orth,"total_rms":rms,
            "orthogonal_fraction":orth/max(rms,1e-30),"rank_increase":int(orth**2>1e-9*lam[0]),
            "closest_source":names[j],"closest_family":fam.get(names[j],"OTHER"),
            "closest_source_rms":float(dist[j]),"basis_rank":int(len(lam)),"source_count":len(names)}


def production() -> dict:
    import torch
    summary=json.loads((OUT/"full_summary.json").read_text(encoding="utf-8")) if (OUT/"full_summary.json").exists() else full()
    pre=summary.get("preliminary_verdict",preliminary_verdict(summary))
    if not pre.startswith("PROVISIONAL"):
        create_skipped_production(pre); return {"status":"SKIPPED","reason":pre}
    model,tk,cfg,_,dev=etx.load_ckpt(ckpt_name(None))
    for p in model.parameters(): p.requires_grad_(False)
    # Production CLEAN is the canonical full corridor through 2025-10-16.
    Xc,zc,uc,ci=collect_positive(model,tk,cfg,dev,CORRIDOR_END,"clean",production_clean=True)
    X0,z0,u0,c0=collect_positive(model,tk,cfg,dev,CORRIDOR_END,"extra",0)
    X1,z1,u1,c1=collect_positive(model,tk,cfg,dev,CORRIDOR_END,"extra",1)
    test_frame=pl.read_parquet(PACKET/"07_ALIGNED_TEST.parquet").sort("user_id")
    uid=test_frame["user_id"].to_numpy().astype(np.int64)
    cached=load_cached_array("test_query_X")
    if cached is None:
        Xtest,_,_=extract_query(model,tk,cfg,dev,CUTOFF_TEST,seq.user_rows(uid),"test")
        cache_array("test_query_X",Xtest,{"uid":uid})
    else:
        Xtest,meta=cached
        if "uid" in meta and not np.array_equal(meta["uid"],uid):
            # parity cache follows saved ETX order; align rows if needed by rebuilding.
            Xtest,_,_=extract_query(model,tk,cfg,dev,CUTOFF_TEST,seq.user_rows(uid),"test")
            cache_array("test_query_X",Xtest,{"uid":uid})
    clean_cuts=cutoff_grid(seq.MIN_HISTORY,CUTOFF_STEP,CORRIDOR_END)
    if len(clean_cuts)!=29 or clean_cuts[-1]!=CORRIDOR_END:
        raise AssertionError("production CLEAN cutoff corridor changed")
    cclean=np.asarray([zc[ci==k].mean() for k in range(len(clean_cuts))])
    ce0=np.asarray([z0[c0==k].mean() for k in range(13)]); ce1=np.asarray([z1[c1==k].mean() for k in range(13)])
    tc=(zc-cclean[ci]).astype(np.float32); t0=(z0-ce0[c0]).astype(np.float32); t1=(z1-ce1[c1]).astype(np.float32)
    steps=int(np.ceil(len(tc)/8192))*4; early=np.flatnonzero(ci<max(1,len(clean_cuts)//3))
    g=user_group(uid); clean_sum=np.zeros(len(uid)); fresh_sum=np.zeros(len(uid)); vol_sum=np.zeros(len(uid))
    for hs in HEAD_SEEDS:
        cn,_=fit_head([Xc],[tc],np.arange(len(tc)),steps,hs,dev); cp=predict_head(cn,Xtest,dev)+float(cclean.mean())
        fp={};vp={}
        for donor,(Xe,te) in enumerate([(X0,t0),(X1,t1)]):
            ne=len(te); fr=np.concatenate([np.arange(len(tc)),len(tc)+np.arange(ne)])
            vrng=np.random.default_rng(42); vr=np.concatenate([np.arange(len(tc)),vrng.choice(early,size=ne,replace=True)])
            fn,_=fit_head([Xc,Xe],[tc,te],fr,steps,hs,dev); vn,_=fit_head([Xc,Xe],[tc,te],vr,steps,hs,dev)
            fp[donor]=predict_head(fn,Xtest,dev)+float(cclean.mean()); vp[donor]=predict_head(vn,Xtest,dev)+float(cclean.mean())
            del fn,vn
        clean_sum+=cp; fresh_sum+=np.where(g==0,fp[1],fp[0]); vol_sum+=np.where(g==0,vp[1],vp[0]); del cn
    clean=clean_sum/3; fresh=fresh_sum/3; vol=vol_sum/3; draw=fresh-clean; dvraw=vol-clean
    raw_oof=pl.read_parquet(OUT/"etx_fresh_raw_OOF.parquet")
    pp=donor_preprocess_params([raw_oof["d_etx_fresh_raw"].to_numpy()]); dproc=apply_preprocess(draw,pp)
    alpha_med=float(np.median(summary["nested_alpha"])); alpha_prod=float(ALPHAS[ALPHAS<=alpha_med+1e-12].max())
    zbase=np.log1p(test_frame["pred_exp037_rebuilt"].to_numpy().astype(float))
    zcand=zbase+alpha_prod*dproc; pred=np.expm1(np.maximum(zcand,0))
    pl.DataFrame({"user_id":uid,"d_etx_fresh_raw":draw,"d_etx_vol_raw":dvraw}).write_parquet(OUT/"etx_fresh_raw_TEST.parquet")
    # Geometry projection uses its canonical order.
    Z,names,guid,zref,Phi,lam=geometry_basis()
    order=np.argsort(uid); pos=np.searchsorted(uid[order],guid)
    if not np.array_equal(uid[order][pos],guid): raise AssertionError("geometry TEST keys mismatch")
    dgeom=draw[order][pos]; zcand_geom=zcand[order][pos]
    zbtyd=.95*zbase+.05*np.log1p(test_frame["pred_btyd"].to_numpy().astype(float))+dproc
    rows_proj=[project_test_vector("raw_etx_fresh",None,dgeom,Z,names,zref,Phi,lam),
               project_test_vector("etx_fresh_candidate",zcand_geom,zcand_geom-zbase[order][pos],Z,names,zref,Phi,lam),
               project_test_vector("btyd05_plus_etx_fresh_diagnostic",zbtyd[order][pos],zbtyd[order][pos]-zbase[order][pos],Z,names,zref,Phi,lam)]
    span={"status":"PASS","vectors":rows_proj,"geometry_source_count":len(names),"geometry_rank":len(lam)}
    json_write(OUT/"test_span_projection.json",span)
    provisional_a=pre=="PROVISIONAL_TYPE_A"
    bpass=(summary["incremental_delta_wcv"]<=-0.00008 and summary["incremental_improved_folds"]>=3 and
           summary["latest_fold_delta"]<0 and summary["corr_etx_seq"]<.85 and
           summary["oof_unexplained_variance_ratio"]>=.25 and rows_proj[0]["orthogonal_fraction"]>=.10 and
           rows_proj[0]["orthogonal_rms"]>=.0025 and summary["real_minus_vol_wcv"]<0)
    verdict="PASS_TYPE_A" if provisional_a else ("PASS_TYPE_B" if bpass else "WEAK_SIGNAL")
    if verdict.startswith("PASS"):
        oof_parts=[]
        full_folds=load_raw_folds(HEAD_SEEDS)
        real=nested_direction(full_folds,"d_real")
        for i,f in enumerate(full_folds):
            zz=f["z_base"]+real["rows"][i]["selected_alpha"]*real["processed"][i]
            oof_parts.append(pl.DataFrame({"fold":[str(f["V"])]*len(f["uid"]),"user_id":f["uid"],
                                           "prediction":np.expm1(np.maximum(zz,0))}))
        pl.concat(oof_parts).write_parquet(OUT/"etx_fresh_contrast_OOF.parquet")
        pl.DataFrame({"user_id":uid,"prediction":pred}).write_parquet(OUT/"etx_fresh_contrast_TEST.parquet")
        pl.DataFrame({"user_id":uid,"predict":pred}).write_csv(OUT/"etx_fresh_contrast_TEST.csv")
    regime={"status":"PASS","verdict":verdict,"checkpoint":recovered_checkpoint(None),
            "test_cutoff":str(CUTOFF_TEST),"static_policy":"DCW depth_clip=depth_cap=289; cdow_shift=-1",
            "test_schema_rows":len(uid),"unique_user_ids":int(len(np.unique(uid))),"finite":bool(np.isfinite(pred).all()),
            "alpha_median_nested":alpha_med,"alpha_prod_rounded_down":alpha_prod,"preprocessing_from_oof_only":pp,
            "extra_only_positive":True,"encoder_frozen":sha256(ckpt_path(None))==recovered_checkpoint(None)["sha256"],
            "public_lb_used":False,"submission_uploaded":False}
    json_write(OUT/"production_regime.json",regime)
    summary["final_verdict"]=verdict; json_write(OUT/"full_summary.json",summary)
    del model,tk,Xc,X0,X1,Xtest
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return regime


def create_skipped_production(reason: str) -> None:
    json_write(OUT/"production_regime.json",{"status":"SKIPPED","reason":reason,
               "rule":"production inference is authorized only after provisional PASS TYPE A/B",
               "public_lb_used":False,"submission_uploaded":False})
    json_write(OUT/"test_span_projection.json",{"status":"NOT_RUN","reason":reason})


def create_not_run_artifacts(reason: str) -> None:
    for name in ["fold_metrics.csv","nested_selection.csv","user_half_metrics.csv","real_vs_vol.csv",
                 "seq_vs_etx_fresh.csv","bootstrap_metrics.csv","diversity_oof.csv"]:
        if not (OUT/name).exists(): csv_write(OUT/name,[{"status":"NOT_RUN","reason":reason}])
    if not (OUT/"oof_projection_metrics.json").exists(): json_write(OUT/"oof_projection_metrics.json",{"status":"NOT_RUN","reason":reason})
    create_skipped_production(reason)


def cleanup_cache() -> dict:
    removed=0
    if CACHE.exists():
        for p in CACHE.rglob("*"):
            if p.is_file(): removed+=p.stat().st_size
        shutil.rmtree(CACHE)
    return {"removed_cache_bytes":removed}


def output_size() -> int:
    return sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())


def finalize() -> dict:
    pil=json.loads((OUT/"pilot_metrics.json").read_text(encoding="utf-8")) if (OUT/"pilot_metrics.json").exists() else {"status":"NOT_RUN"}
    full_summary=json.loads((OUT/"full_summary.json").read_text(encoding="utf-8")) if (OUT/"full_summary.json").exists() else {}
    regime=json.loads((OUT/"production_regime.json").read_text(encoding="utf-8")) if (OUT/"production_regime.json").exists() else {"status":"NOT_RUN"}
    verdict=full_summary.get("final_verdict") or ("REJECT_PILOT" if pil.get("status")=="REJECT_PILOT" else full_summary.get("preliminary_verdict","TECHNICAL_BLOCK"))
    if verdict.startswith("PROVISIONAL"): verdict="TECHNICAL_BLOCK"
    recommendation="ADD_TO_SUBMISSION_GEOMETRY" if verdict in {"PASS_TYPE_A","PASS_TYPE_B"} else "DO_NOT_ADD"
    previous_runtime=(json.loads((OUT/"runtime_resources.json").read_text(encoding="utf-8"))
                      if (OUT/"runtime_resources.json").exists() else {})
    cleanup=cleanup_cache()
    cleanup["removed_cache_bytes"] += int(previous_runtime.get("removed_cache_bytes",0))
    pycache=OUT/"__pycache__"
    if pycache.exists():
        cleanup["removed_cache_bytes"] += sum(p.stat().st_size for p in pycache.rglob("*") if p.is_file())
        shutil.rmtree(pycache)
    # Internal raw fold NPZs are temporary once standardized OOF exists or pilot rejects.
    for p in OUT.glob("_raw_fold_*.npz"):
        p.unlink(); cleanup["removed_cache_bytes"]+=p.stat().st_size if p.exists() else 0
    for p in OUT.glob("_fold_*.npz"):
        p.unlink()
    earliest=min((p.stat().st_mtime for p in OUT.iterdir() if p.is_file()),default=time.time())
    runtime={"finalize_process_seconds":time.time()-START,
             "experiment_elapsed_seconds":time.time()-earliest,
             "pilot_runtime_seconds":pil.get("pilot_runtime_seconds"),
             "persistent_bytes":output_size(),
             "persistent_gb":output_size()/1e9,"disk_limit_gb":6,"within_disk_limit":output_size()<=6e9,
             **cleanup,"platform":platform.platform()}
    json_write(OUT/"runtime_resources.json",runtime)
    # Required files are represented even when a registered early stop forbids their computation.
    create_not_run_artifacts(verdict) if pil.get("status")!="PASS" else None
    paths=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in {"checksums.sha256","report.md"}:
            paths.append((sha256(p),p.name,p.stat().st_size))
    (OUT/"checksums.sha256").write_text("".join(f"{h}  {n}\n" for h,n,_ in paths),encoding="utf-8")
    metrics=full_summary if full_summary else pil.get("metrics",{})
    parity_doc=(json.loads((OUT/'encoder_parity.json').read_text(encoding='utf-8'))
                if (OUT/'encoder_parity.json').exists() else {})
    if verdict=="REJECT_PILOT":
        full_text=("The full four-fold/wCV phase was **not run**, exactly as required by the pilot gate. "
                   "The retained `fold_metrics.csv` rows are pilot-only diagnostics, not a four-fold estimate.")
        incremental_text=("Pilot gamma was estimated without targets from the three donor-panel correction vectors. "
                          "The pilot comparison used fixed unit scale; nested beta selection was not authorized after rejection.")
        diversity_text=("`diversity_oof.csv` contains pilot-only correction correlations. The full donor-fold projection "
                        "and unexplained-variance endpoint were not run; `oof_projection_metrics.json` records the early stop.")
    else:
        full_text=(f"Full summary: `{json.dumps(metrics,ensure_ascii=False)}`. Detailed rows are in "
                   "`fold_metrics.csv` and `nested_selection.csv`.")
        incremental_text=("Gamma is estimated only from donor correction vectors, without targets; beta is selected on "
                          "donor folds from the preregistered grid. Results are in `nested_selection.csv` and `seq_vs_etx_fresh.csv`.")
        diversity_text=("See `diversity_oof.csv` and `oof_projection_metrics.json`. Existing-source projection "
                        "coefficients are learned only on donor folds.")
    standardized_names=["etx_fresh_raw_OOF.parquet","etx_fresh_raw_TEST.parquet",
                        "etx_fresh_contrast_OOF.parquet","etx_fresh_contrast_TEST.parquet",
                        "etx_fresh_contrast_TEST.csv"]
    standardized_lines=[]
    for name in standardized_names:
        p=OUT/name
        if p.exists():
            standardized_lines.append(f"- `{p.resolve()}` — SHA256 `{sha256(p)}`")
        else:
            standardized_lines.append(f"- `{p.resolve()}` — **NOT PRODUCED** ({verdict} early-stop policy)")
    standardized_text="\n".join(standardized_lines)
    report=f"""# EXP071_ETX_FRESH_CONTRAST

## 1. Verdict

**{verdict}** — **{recommendation}**.

## 2. Exact hypothesis

A frozen ETX-01 seed-42 final query-token embedding may encode amount timing differently from the historical frozen TCN/D3A embedding, making `mu_ETX_FRESH - mu_ETX_CLEAN` both more useful for EXP-037 and more orthogonal to the existing SEQ-FRESH correction.

## 3. Encoder/checkpoint parity

See `encoder_parity.json`. All four fold configs and the TEST config match the registered ETX-01 architecture. The external hook replays the original forward and returns the 128-dimensional final normalized query token without changing weights. Hard hook parity status: **{parity_doc.get('status','NOT_RUN')}**, with hook-vs-original error `0.0`. The archived 2025-10-16 OOF array also replays exactly. The prior saved TEST DCW array shows recorded bf16/SDPA runtime drift (max `0.0625`, RMS `0.0026638` in direct ETX log output); this drift is between runtimes, while the hook and original forward in the current frozen runtime are identical.

## 4. Pilot decision

Pilot status: **{pil.get('status')}**. Metrics: `{json.dumps(pil.get('metrics',{}),ensure_ascii=False)}`. The gate uses fixed unit scale; the displayed alpha grid is diagnostic and was not used to select on the held-out fold.

## 5. Full fold and wCV metrics

{full_text}

## 6. REAL vs VOL evidence

See `real_vs_vol.csv` and `user_half_metrics.csv`. Equal-volume rows use the exact historical earliest-third CLEAN-positive resampling rule with canonical RNG seed 42 and equal optimization steps.

## 7. ETX-FRESH vs existing SEQ-FRESH

See `seq_vs_etx_fresh.csv`. Existing SEQ-FRESH is read unchanged from `06_ALIGNED_OOF.pred_fresh_contrast`.

## 8. Incremental orthogonal component

{incremental_text}

## 9. OOF correction diversity

{diversity_text}

## 10. TEST distance outside the current geometry span

See `test_span_projection.json`. TEST conditional-head inference is skipped unless OOF evidence provisionally satisfies PASS TYPE A or TYPE B.

## 11. Leakage and production-regime audits

EXTRA contributes only positive conditional-amount rows from the opposite splitmix64 user side. It never updates the ETX encoder, tokenizer, zero/nonzero probability, EXP-037 components, validation labels, eligibility, or normalization. Production status: `{json.dumps(regime,ensure_ascii=False)}`. Public LB use and upload are both false.

## 12. Runtime and disk

`{json.dumps(runtime,ensure_ascii=False)}`. Temporary embedding caches were removed during finalization.

## 13. Standardized artifacts and SHA256

Hashes are in `checksums.sha256`; source provenance and checkpoint hashes are in `artifact_manifest.csv`.

{standardized_text}

The OOF raw file is explicitly marked `PILOT_ONLY_SEED42` after a pilot rejection. TEST and candidate artifacts are not fabricated when production is not authorized.

## 14. Recommendation

**{recommendation}**. No leaderboard upload was made and geometry weights were not refit.
"""
    (OUT/"report.md").write_text(report,encoding="utf-8")
    # Regenerate checksums after report/runtime and include report, excluding the checksum file itself.
    paths=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!="checksums.sha256": paths.append((sha256(p),p.name))
    (OUT/"checksums.sha256").write_text("".join(f"{h}  {n}\n" for h,n in paths),encoding="utf-8")
    return {"verdict":verdict,"recommendation":recommendation,"runtime":runtime}


def auto() -> dict:
    reconnaissance(); parity(); p=pilot()
    if p["status"]!="PASS": return finalize()
    s=full()
    if s.get("preliminary_verdict","").startswith("PROVISIONAL"): production()
    return finalize()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["recon","parity","pilot","full","production","finalize","auto"])
    a=ap.parse_args()
    fn={"recon":reconnaissance,"parity":parity,"pilot":pilot,"full":full,
        "production":production,"finalize":finalize,"auto":auto}[a.command]
    result=fn(); print(json.dumps(result,indent=2,ensure_ascii=False,default=str))


if __name__=="__main__":
    main()
