from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
OUT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP069_BTYD05_FRESH1_PROD")
sys.path.insert(0, str(OLD))

from src.config import CORRIDOR_END, CUTOFF_STEP, CUTOFF_TEST  # noqa: E402
from src.data import sample_submit  # noqa: E402
from src.seq import MIN_HISTORY, cutoff_grid, load_ckpt, user_rows  # noqa: E402
from src.seq_cond import EXTRA_CUTOFFS, POS_ONLY, collect, embed, fit_head, head_predict, user_group  # noqa: E402


CKPT_NAME = "SEQ-D3A-BASE-S42-TEST"
HEAD_SEEDS = [42, 43, 44]
HEAD_BATCH = 8192
HEAD_EPOCHS = 4
HEAD_LR = 0.001
HEAD_WD = 0.01
HEAD_DROPOUT = 0.10
TEST_DEPTH_CLIP = 289
ARMS = ("CLEAN", "VOL", "FRESH")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def distribution(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    qs = [0, 0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999, 1]
    return {
        "n": len(x), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "max": float(x.max()),
        "quantiles": {str(q): float(np.quantile(x, q)) for q in qs},
    }


def main() -> None:
    import torch

    started = time.time()
    checkpoint = OLD / "artifacts" / f"model_{CKPT_NAME}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"exactly-one production encoder has not completed: {checkpoint}")
    model, cfg, checkpoint_cutoff, dev = load_ckpt(CKPT_NAME)
    if checkpoint_cutoff != CUTOFF_TEST:
        raise AssertionError(f"production checkpoint cutoff is {checkpoint_cutoff}, expected {CUTOFF_TEST}")
    expected_cfg = {
        "hidden": 64, "blocks": 8, "kernel": 3, "dropout": 0.1,
        "batch": 1024, "chunk": 256, "lr": 0.003, "wd": 0.01,
        "epochs": 4, "warmup": 300, "seed": 42, "workers": 3,
        "compile": False, "aug": "none", "depth_aug": 0.0,
    }
    mismatch = {k: {"expected": v, "actual": cfg.get(k)} for k, v in expected_cfg.items() if cfg.get(k) != v}
    if mismatch:
        raise AssertionError(f"production encoder config mismatch: {mismatch}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder_checksum_before = float(sum(float(p.double().sum()) for p in model.parameters()))

    clean_cutoffs = cutoff_grid(MIN_HISTORY, CUTOFF_STEP)
    if len(clean_cutoffs) != 29 or clean_cutoffs[-1] != CORRIDOR_END:
        raise AssertionError(f"CLEAN cutoff corridor changed: {len(clean_cutoffs)}, {clean_cutoffs[-1]}")
    if [d.isoformat() for d in EXTRA_CUTOFFS] != [
        "2025-10-22", "2025-10-29", "2025-11-05", "2025-11-12", "2025-11-19",
        "2025-11-26", "2025-12-03", "2025-12-10", "2025-12-17", "2025-12-24",
        "2025-12-31", "2026-01-07", "2026-01-14",
    ]:
        raise AssertionError("EXTRA cutoff grid changed")

    print("collecting CLEAN positive embeddings", flush=True)
    Xc, zc, uc, cc = collect(model, cfg, dev, clean_cutoffs, 1, keep=POS_ONLY,
                             group_keep=None, depth_clip=None, tag="EXP069-CLEAN+", cache=None)
    print("collecting EXTRA positive embeddings", flush=True)
    Xe, ze, ue, ce = collect(model, cfg, dev, EXTRA_CUTOFFS, 1, keep=POS_ONLY,
                             group_keep=None, depth_clip=TEST_DEPTH_CLIP, tag="EXP069-EXTRA+", cache=None)
    if not np.all(zc > 0) or not np.all(ze > 0):
        raise AssertionError("nonpositive target entered conditional head pool")
    clean_side = user_group(uc)
    extra_side = user_group(ue)

    sample_uid = sample_submit()["user_id"].to_numpy().astype(np.int64)
    if len(sample_uid) != 250_000 or len(np.unique(sample_uid)) != 250_000:
        raise AssertionError("canonical sample order is invalid")
    print("embedding canonical TEST panel at depth clip 289", flush=True)
    Xtest = embed(model, cfg, dev, CUTOFF_TEST, user_rows(sample_uid),
                  depth_clip=TEST_DEPTH_CLIP, batch=1024)

    extensive_path = OLD / "artifacts" / "ZERO2D_DIST_test.npz"
    extensive = np.load(extensive_path, allow_pickle=False)
    extensive_uid = extensive["user_id"].astype(np.int64)
    order = np.argsort(extensive_uid)
    pos = np.searchsorted(extensive_uid[order], sample_uid)
    if pos.max() >= len(order) or not np.array_equal(extensive_uid[order][pos], sample_uid):
        raise AssertionError("reconstructed CLEAN-only DIST extensive rows do not align")
    p_dist = extensive["p_act"].astype(float)[order][pos]
    if not np.all(np.isfinite(p_dist)) or not np.all((p_dist >= 0) & (p_dist <= 1)):
        raise AssertionError("DIST extensive probability outside [0,1]")

    Xall = np.concatenate([Xc, Xe], axis=0)
    del Xc, Xe
    gc.collect()
    n_clean = len(zc)
    predictions: dict[str, np.ndarray] = {}
    head_states: dict[str, object] = {}
    side_metadata: dict[str, object] = {}

    for side in (0, 1):
        clean_idx = np.flatnonzero(clean_side == side)
        extra_local = np.flatnonzero(extra_side == side)
        extra_idx = n_clean + extra_local
        c_clean = np.asarray([zc[clean_idx[cc[clean_idx] == k]].mean() for k in range(len(clean_cutoffs))])
        c_extra = np.asarray([ze[extra_local[ce[extra_local] == k]].mean() for k in range(len(EXTRA_CUTOFFS))])
        if not np.all(np.isfinite(c_clean)) or not np.all(np.isfinite(c_extra)):
            raise AssertionError(f"side {side} has an empty conditional cutoff")
        c_hat = float(c_clean.mean())
        targets = np.zeros(len(Xall), np.float32)
        targets[clean_idx] = (zc[clean_idx] - c_clean[cc[clean_idx]]).astype(np.float32)
        targets[extra_idx] = (ze[extra_local] - c_extra[ce[extra_local]]).astype(np.float32)
        early = clean_idx[cc[clean_idx] < max(1, len(clean_cutoffs) // 3)]
        steps = int(np.ceil(len(clean_idx) / HEAD_BATCH)) * HEAD_EPOCHS
        arm_seed_predictions = {arm: [] for arm in ARMS}
        side_metadata[str(side)] = {
            "clean_positive_rows": int(len(clean_idx)),
            "extra_positive_rows": int(len(extra_idx)),
            "steps_per_head": steps,
            "clean_level_by_cutoff": c_clean.tolist(),
            "extra_centering_by_cutoff": c_extra.tolist(),
            "restored_clean_level": c_hat,
            "clean_users_sha256": hashlib.sha256(np.unique(uc[clean_idx]).tobytes()).hexdigest(),
            "extra_users_sha256": hashlib.sha256(np.unique(ue[extra_local]).tobytes()).hexdigest(),
        }
        for seed in HEAD_SEEDS:
            rng = np.random.default_rng(seed)
            rows = {
                "CLEAN": clean_idx,
                "VOL": np.concatenate([clean_idx, rng.choice(early, size=len(extra_idx), replace=True)]),
                "FRESH": np.concatenate([clean_idx, extra_idx]),
            }
            for arm in ARMS:
                print(f"side={side} seed={seed} arm={arm} rows={len(rows[arm]):,} steps={steps:,}", flush=True)
                net, loss = fit_head(
                    Xall, targets, steps=steps, batch=HEAD_BATCH, lr=HEAD_LR, wd=HEAD_WD,
                    hidden=cfg["hidden"], dropout=HEAD_DROPOUT, seed=seed,
                    binary=False, dev=dev, out_bias=0.0, rows=rows[arm])
                mu = head_predict(net, Xtest, dev).astype(np.float64) + c_hat
                z_cond = np.maximum(p_dist * np.maximum(mu, 0.0), 0.0)
                arm_seed_predictions[arm].append(z_cond)
                key = f"side{side}_seed{seed}_{arm}"
                head_states[key] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                side_metadata[str(side)].setdefault("loss", {})[f"seed{seed}_{arm}"] = float(loss)
                del net, mu, z_cond
                gc.collect()
        for arm in ARMS:
            predictions[f"z_{arm.lower()}_side{side}"] = np.mean(np.vstack(arm_seed_predictions[arm]), axis=0)

    for arm in ARMS:
        predictions[f"z_{arm.lower()}"] = 0.5 * (
            predictions[f"z_{arm.lower()}_side0"] + predictions[f"z_{arm.lower()}_side1"])
    d_raw_fresh = predictions["z_fresh"] - predictions["z_clean"]
    d_raw_vol = predictions["z_vol"] - predictions["z_clean"]
    if not all(np.isfinite(v).all() and np.all(v >= 0) for v in predictions.values()):
        raise AssertionError("conditional TEST predictions are invalid")
    if not np.isfinite(d_raw_fresh).all() or not np.isfinite(d_raw_vol).all():
        raise AssertionError("conditional TEST contrasts are invalid")

    encoder_checksum_after = float(sum(float(p.double().sum()) for p in model.parameters()))
    if encoder_checksum_after != encoder_checksum_before:
        raise AssertionError("frozen encoder changed during conditional-head training")

    raw_path = OUT / "fresh_production_raw.npz"
    np.savez_compressed(
        raw_path, user_id=sample_uid, p_dist=p_dist, d_raw_fresh=d_raw_fresh,
        d_raw_vol=d_raw_vol, **predictions)
    heads_path = OUT / "fresh_production_heads.pt"
    torch.save({
        "checkpoint": CKPT_NAME, "encoder_cfg": cfg, "head_seeds": HEAD_SEEDS,
        "head_recipe": {"batch": HEAD_BATCH, "epochs": HEAD_EPOCHS, "lr": HEAD_LR,
                        "wd": HEAD_WD, "dropout": HEAD_DROPOUT},
        "states": head_states,
    }, heads_path)
    audit = {
        "status": "PASS",
        "runtime_seconds": time.time() - started,
        "production_encoder": str(checkpoint),
        "production_encoder_sha256": sha256(checkpoint),
        "production_encoder_cfg": cfg,
        "encoder_checksum_before": encoder_checksum_before,
        "encoder_checksum_after": encoder_checksum_after,
        "clean_cutoffs": [d.isoformat() for d in clean_cutoffs],
        "extra_cutoffs": [d.isoformat() for d in EXTRA_CUTOFFS],
        "test_depth_clip": TEST_DEPTH_CLIP,
        "head_seeds": HEAD_SEEDS,
        "head_averaging": "three seeds in z/log space within each donor side; two donor sides averaged in z/log space",
        "side_metadata": side_metadata,
        "extensive_component": {
            "path": str(extensive_path),
            "sha256": sha256(extensive_path),
            "status": "same-recipe CLEAN-only S1-DIST reconstruction; not byte-exact to historical ztest_S1-DIST",
            "reference_reproduced": bool(extensive["reference_reproduced"]),
            "max_abs_z_rebuild_error": float(extensive["max_abs_dz"]),
            "mean_abs_z_rebuild_error": float(extensive["mean_abs_dz"]),
            "z_rebuild_correlation": float(extensive["correlation_z"]),
            "extra_rows_used": False,
        },
        "raw_fresh": distribution(d_raw_fresh),
        "raw_vol": distribution(d_raw_vol),
        "raw_path": str(raw_path), "raw_sha256": sha256(raw_path),
        "heads_path": str(heads_path), "heads_sha256": sha256(heads_path),
        "persistent_embedding_caches": [],
    }
    (OUT / "production_training_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
