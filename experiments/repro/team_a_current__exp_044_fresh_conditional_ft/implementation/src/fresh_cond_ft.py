"""EXP-044 — deterministic paired fresh conditional supervision for plain SEQ-01.

The experiment intentionally has one causal contrast only::

    FT-FRESH - FT-VOL

Three new deterministic plain SEQ-01 baselines are trained once (seeds derived
from ``src.config.SEED``).  Each seed then supplies exactly one checkpoint to a
paired one-CLEAN-epoch continuation.  The two arms share the direct CLEAN plan,
the common CLEAN-positive conditional stream, optimizer construction, initial
conditional head, RNG policy, shapes, step count, LR multipliers and snapshots.
Only the added positive donor row differs: early CLEAN group B for FT-VOL,
EXTRA group B for FT-FRESH.

No test prediction, blend, LOFO, submission or full-fold path exists here.

One-command run from the repository root::

    python src/fresh_cond_ft.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Iterable

# Required before any CUDA context is created, including in child processes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import ARTIFACTS, SEED, TARGET_DAYS
from src.features import panel_users
from src.seq import (Batcher, DEPTH_GRID, N_CH_STORED, build_index, build_model,
                     fold_cutoffs, gather, panel, predict, target_at, user_rows)
from src.seq_cond import EXTRA_CUTOFFS, _auc, _pool, build_head, seg_masks, segments, user_group
from src.validation import bias_z, calibrate, rmsle_z


EXP_NUM = 44
EXP_ID = "FRESH-COND-FT"
VAL = dt.date(2025, 10, 16)
SEEDS = tuple(SEED + i for i in range(3))
OUT = ARTIFACTS / "FRESH_COND_FT_EXP044"
PLANS = OUT / "plans"
BASELINES = OUT / "baselines"
ARMS = OUT / "arms"
REPLAY = OUT / "integration_replay"

BASELINE_EPOCHS = 4
PAIR_EPOCHS = 1
COND_HALF_BATCH = 128
LAMBDA_COND = 0.25
ENCODER_LR = 3e-5
COND_LR = 1e-3
WARMUP = 300
WEIGHT_DECAY = 1e-2
SNAPSHOT_FIXED = (0, 1, 100, 1000)

ARM_VOL = "VOL"
ARM_FRESH = "FRESH"
ARMS_ALLOWED = (ARM_VOL, ARM_FRESH)


def log(*parts: Any) -> None:
    print(f"[EXP-044 {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def baseline_name(seed: int) -> str:
    return f"DETSEQ01-S{seed}-V1016"


def arm_name(seed: int, arm: str) -> str:
    assert arm in ARMS_ALLOWED
    return f"FT-{arm}-S{seed}-V1016"


def baseline_checkpoint(seed: int) -> Path:
    return ARTIFACTS / f"model_{baseline_name(seed)}.pt"


def baseline_plan_paths(seed: int) -> tuple[Path, Path]:
    return PLANS / f"baseline_s{seed}.npz", PLANS / f"baseline_s{seed}.json"


def pair_plan_paths(seed: int) -> tuple[Path, Path]:
    return PLANS / f"pair_s{seed}.npz", PLANS / f"pair_s{seed}.json"


def pair_cond_init_path(seed: int) -> Path:
    return PLANS / f"pair_s{seed}_conditional_init.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hash_tensor(h: Any, name: str, value: torch.Tensor) -> None:
    a = value.detach().cpu().contiguous().numpy()
    h.update(name.encode("utf-8"))
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes())


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        _hash_tensor(h, name, value)
    return h.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    return state_dict_hash(module.state_dict())


def direct_head_hash(model: torch.nn.Module) -> str:
    return module_hash(model.head)


def optimizer_hash(model: torch.nn.Module, conditional: torch.nn.Module | None,
                   opt: torch.optim.Optimizer) -> str:
    names = {id(p): f"model.{n}" for n, p in model.named_parameters()}
    if conditional is not None:
        names.update({id(p): f"conditional.{n}" for n, p in conditional.named_parameters()})
    h = hashlib.sha256()
    for group_i, group in enumerate(opt.param_groups):
        spec = {k: v for k, v in group.items() if k != "params"}
        h.update(json.dumps(spec, sort_keys=True, default=str).encode("utf-8"))
        for p in group["params"]:
            name = names[id(p)]
            h.update(f"group={group_i}:{name}".encode("utf-8"))
            for key, value in sorted(opt.state.get(p, {}).items()):
                if torch.is_tensor(value):
                    _hash_tensor(h, f"{name}:{key}", value)
                else:
                    h.update(f"{name}:{key}:{value!r}".encode("utf-8"))
    return h.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state().cpu(),
        cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )


def rng_state_hash(state: dict[str, Any] | None = None) -> str:
    state = capture_rng_state() if state is None else state
    h = hashlib.sha256()
    h.update(pickle.dumps(state["python"], protocol=5))
    h.update(pickle.dumps(state["numpy"], protocol=5))
    h.update(state["torch"].numpy().tobytes())
    for value in state["cuda"]:
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [jsonable(x) for x in value]
    if isinstance(value, list):
        return [jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    assert not path.exists(), f"refusing to overwrite existing artifact: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True,
                               ensure_ascii=False), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    paths = {
        "fresh_cond_ft.py": Path(__file__),
        "seq.py": ROOT / "src" / "seq.py",
        "seq_cond.py": ROOT / "src" / "seq_cond.py",
        "validation.py": ROOT / "src" / "validation.py",
        "config.py": ROOT / "src" / "config.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def environment_record() -> dict[str, Any]:
    return dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        numpy=np.__version__,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_debug_mode=str(torch.get_deterministic_debug_mode()),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
    )


def configure_determinism(seed: int) -> dict[str, Any]:
    assert seed in SEEDS, f"seed {seed} must be derived from src.config.SEED"
    assert os.environ.get("PYTHONHASHSEED") == str(seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    return environment_record()


def baseline_cfg(seed: int) -> dict[str, Any]:
    assert seed in SEEDS, "experiment seeds must be derived from src.config.SEED"
    return dict(
        hidden=64, blocks=8, kernel=3, dropout=0.10,
        batch=1024, chunk=256,
        lr=3e-3, wd=1e-2, epochs=BASELINE_EPOCHS, warmup=WARMUP,
        seed=seed, workers=1, compile=False,
        aug="none", aug_p=0.0, aug_full=0.0,
        depth_aug=0.0, depth_grid=tuple(DEPTH_GRID),
    )


def validate_plain_baseline_cfg(cfg: dict[str, Any], name: str = "") -> None:
    assert "D3A" not in name.upper(), "SEQ-D3A artifacts are forbidden in EXP-044"
    expected = baseline_cfg(int(cfg["seed"]))
    for key, value in expected.items():
        actual = tuple(cfg[key]) if key == "depth_grid" else cfg[key]
        assert actual == value, f"plain SEQ-01 cfg mismatch at {key}: {actual!r} != {value!r}"
    assert float(cfg["depth_aug"]) == 0.0
    assert cfg["aug"] == "none"
    assert not cfg["compile"] and int(cfg["workers"]) == 1


def validate_baseline_checkpoint(checkpoint: dict[str, Any], expected_name: str) -> None:
    validate_plain_baseline_cfg(checkpoint["cfg"], expected_name)
    assert checkpoint.get("experiment_name") == expected_name
    assert dt.date.fromisoformat(checkpoint["val"]) == VAL


def lr_multiplier_plan(total: int) -> np.ndarray:
    out = np.empty(total, np.float64)
    for step in range(total):
        out[step] = (min(1.0, (step + 1) / WARMUP)
                     * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
    return out


def baseline_lr_plan(cfg: dict[str, Any], total: int) -> np.ndarray:
    return float(cfg["lr"]) * lr_multiplier_plan(total)


def pack_groups(groups: list[list[tuple[int, np.ndarray]]]) -> dict[str, np.ndarray]:
    group_chunk_offsets = [0]
    chunk_cutoff: list[int] = []
    chunk_sample_offsets = [0]
    sample_indices: list[np.ndarray] = []
    for group in groups:
        for cutoff_i, idx in group:
            idx = np.asarray(idx)
            assert idx.ndim == 1 and (idx >= 0).all()
            chunk_cutoff.append(int(cutoff_i))
            sample_indices.append(idx.astype(np.int32, copy=False))
            chunk_sample_offsets.append(chunk_sample_offsets[-1] + len(idx))
        group_chunk_offsets.append(len(chunk_cutoff))
    return dict(
        group_chunk_offsets=np.asarray(group_chunk_offsets, np.int32),
        chunk_cutoff=np.asarray(chunk_cutoff, np.int16),
        chunk_sample_offsets=np.asarray(chunk_sample_offsets, np.int64),
        sample_indices=(np.concatenate(sample_indices) if sample_indices
                        else np.empty(0, np.int32)),
    )


def unpack_group(plan: Any, batch_i: int) -> list[tuple[int, np.ndarray]]:
    c0, c1 = plan["group_chunk_offsets"][batch_i:batch_i + 2]
    out = []
    for chunk_i in range(int(c0), int(c1)):
        i0, i1 = plan["chunk_sample_offsets"][chunk_i:chunk_i + 2]
        out.append((int(plan["chunk_cutoff"][chunk_i]),
                    plan["sample_indices"][int(i0):int(i1)]))
    return out


def data_hashes(cuts: list[dt.date], ci: np.ndarray, ri: np.ndarray,
                zy: np.ndarray) -> dict[str, str]:
    return dict(
        cuts=sha256_array(np.asarray([x.isoformat() for x in cuts], dtype="U10")),
        ci=sha256_array(ci.astype(np.int16, copy=False)),
        ri=sha256_array(ri.astype(np.int32, copy=False)),
        zy=sha256_array(zy.astype(np.float32, copy=False)),
    )


def fixed_validation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = panel_users(VAL, 3)["user_id"].to_numpy().astype(np.int64, copy=False)
    users = users[user_group(users) == 0]
    rows = user_rows(users).astype(np.int32, copy=False)
    y = target_at(VAL, rows).astype(np.float64, copy=False)
    return users, rows, y


def snapshot_steps(n_steps: int) -> list[int]:
    half = n_steps // 2
    return sorted({x for x in (*SNAPSHOT_FIXED, half, n_steps) if 0 <= x <= n_steps})


def _save_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> str:
    assert not path.exists(), f"refusing to overwrite existing artifact: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)
    return sha256_file(path)


def _load_plan(npz_path: Path, json_path: Path) -> tuple[Any, dict[str, Any]]:
    assert npz_path.exists() and json_path.exists()
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert sha256_file(npz_path) == meta["plan_file_sha256"]
    plan = np.load(npz_path, allow_pickle=False)
    for name, expected in meta["array_sha256"].items():
        assert sha256_array(plan[name]) == expected, f"plan array changed: {name}"
    return plan, meta


def _reuse_plan(npz_path: Path, json_path: Path, kind: str, seed: int) -> dict[str, Any] | None:
    if not (npz_path.exists() or json_path.exists()):
        return None
    assert npz_path.exists() and json_path.exists(), "partial plan artifacts found; refusing overwrite"
    _, meta = _load_plan(npz_path, json_path)
    assert meta["kind"] == kind and meta["seed"] == seed and meta["val"] == VAL.isoformat()
    current = source_hashes()
    for name in ("seq.py", "seq_cond.py", "validation.py", "config.py", "fresh_cond_ft.py"):
        assert meta["source_sha256"][name] == current[name], f"source drift under plan: {name}"
    return meta


def prepare_baseline_plan(seed: int) -> dict[str, Any]:
    npz_path, json_path = baseline_plan_paths(seed)
    reuse = _reuse_plan(npz_path, json_path, "baseline", seed)
    if reuse is not None:
        log(f"reuse baseline plan seed {seed}: {reuse['n_steps']:,} steps")
        return reuse

    cfg = baseline_cfg(seed)
    cuts = fold_cutoffs(VAL)
    assert cuts[0] == dt.date(2025, 4, 3) and cuts[-1] == dt.date(2025, 9, 11)
    assert all(T + dt.timedelta(days=TARGET_DAYS) <= VAL for T in cuts)
    ci, ri, zy = build_index(cuts, blocks=1)
    cfg = dict(cfg, z0=float(zy.mean()))
    validate_plain_baseline_cfg(cfg, baseline_name(seed))

    batcher = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"],
                      np.random.default_rng(seed), workers=1,
                      aug=dict(mode="none", p=0.0, full=0.0),
                      aug_seed=[seed, 0xA7A1],
                      depth=dict(p=0.0, grid=DEPTH_GRID))
    all_groups: list[list[tuple[int, np.ndarray]]] = []
    epoch_step_offsets = [0]
    batch_seeds: list[np.ndarray] = []
    epoch_plan_hashes = []
    for epoch in range(BASELINE_EPOCHS):
        groups = batcher._plan()
        packed_epoch = pack_groups(groups)
        epoch_plan_hashes.append(hashlib.sha256(b"".join(
            np.ascontiguousarray(v).tobytes() for v in packed_epoch.values())).hexdigest())
        all_groups.extend(groups)
        batch_seeds.append(batcher.arng.integers(0, 2 ** 62, size=len(groups),
                                                 dtype=np.int64))
        epoch_step_offsets.append(len(all_groups))
    packed = pack_groups(all_groups)
    n_steps = len(all_groups)
    val_user, val_rows, val_y = fixed_validation()
    arrays = dict(
        cuts=np.asarray([x.isoformat() for x in cuts], dtype="U10"),
        epoch_step_offsets=np.asarray(epoch_step_offsets, np.int64),
        batch_seed=np.concatenate(batch_seeds),
        lr_plan=baseline_lr_plan(cfg, n_steps),
        val_user_id=val_user,
        val_rows=val_rows,
        val_y=val_y,
        snapshot_steps=np.asarray(sorted(set(SNAPSHOT_FIXED)
                                         | set(epoch_step_offsets)), np.int64),
        **packed,
    )
    array_hashes = {name: sha256_array(value) for name, value in arrays.items()}
    plan_sha = _save_npz_new(npz_path, arrays)
    core = dict(
        format=1, experiment=EXP_ID, kind="baseline", seed=seed, val=VAL,
        name=baseline_name(seed), cfg=cfg, n_cutoffs=len(cuts), n_examples=len(zy),
        n_steps=n_steps, steps_per_epoch=len(all_groups) // BASELINE_EPOCHS,
        epoch_step_offsets=epoch_step_offsets,
        epoch_plan_sha256=epoch_plan_hashes,
        clean_data_sha256=data_hashes(cuts, ci, ri, zy),
        validation_order_sha256=sha256_array(val_user),
        array_sha256=array_hashes, source_sha256=source_hashes(),
        execution_policy=dict(
            workers=1, materialized_plans=True, pythonhashseed=seed,
            rng="Python/NumPy/Torch/CUDA fixed", cudnn_benchmark=False,
            cudnn_deterministic=True, deterministic_algorithms=True,
            cublas_workspace_config=":4096:8", tf32=True, compile=False, bf16=True,
            separate_process=True,
        ),
    )
    plan_id = hashlib.sha256(json.dumps(jsonable(core), sort_keys=True).encode()).hexdigest()
    meta = dict(core, plan_id=plan_id, plan_file_sha256=plan_sha,
                created_at=dt.datetime.now().isoformat(timespec="seconds"))
    write_json_new(json_path, meta)
    log(f"baseline plan seed {seed}: {n_steps:,} steps, 4 epoch hashes saved")
    return meta


def collect_extra_positive_group_b() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows_all, z_all, ci_all, uid_all = [], [], [], []
    for cutoff_i, T in enumerate(EXTRA_CUTOFFS):
        users = panel_users(T, 1)["user_id"].to_numpy().astype(np.int64, copy=False)
        users = users[user_group(users) == 1]
        rows = user_rows(users)
        y = target_at(T, rows)
        keep = y > 0
        rows_all.append(rows[keep].astype(np.int32, copy=False))
        z_all.append(np.log1p(y[keep]).astype(np.float32, copy=False))
        ci_all.append(np.full(int(keep.sum()), cutoff_i, np.int16))
        uid_all.append(users[keep].astype(np.int64, copy=False))
    return (np.concatenate(rows_all), np.concatenate(z_all), np.concatenate(ci_all),
            np.concatenate(uid_all))


def prepare_conditional_initialization(seed: int, dim: int = 192) -> dict[str, Any]:
    path = pair_cond_init_path(seed)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        return dict(path=str(path.resolve()), file_sha256=sha256_file(path),
                    state_sha256=state_dict_hash(state))
    torch.manual_seed(seed)
    conditional = build_head(dim, 64, 0.10, 0.0).cpu()
    state = {k: v.detach().cpu().clone() for k, v in conditional.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(state, tmp)
    tmp.rename(path)
    return dict(path=str(path.resolve()), file_sha256=sha256_file(path),
                state_sha256=state_dict_hash(state))


def _checkpoint_result(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = BASELINES / baseline_name(seed) / "result.json"
    ckpt_path = baseline_checkpoint(seed)
    assert result_path.exists() and ckpt_path.exists(), f"baseline seed {seed} is incomplete"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    validate_baseline_checkpoint(checkpoint, baseline_name(seed))
    assert sha256_file(ckpt_path) == result["checkpoint_file_sha256"]
    assert state_dict_hash(checkpoint["state"]) == result["final_model_sha256"]
    return checkpoint, result


def prepare_pair_plan(seed: int) -> dict[str, Any]:
    npz_path, json_path = pair_plan_paths(seed)
    reuse = _reuse_plan(npz_path, json_path, "pair", seed)
    if reuse is not None:
        log(f"reuse paired plan seed {seed}: {reuse['n_steps']:,} steps")
        return reuse

    checkpoint, baseline_result = _checkpoint_result(seed)
    cfg = checkpoint["cfg"]
    validate_plain_baseline_cfg(cfg, baseline_name(seed))
    cuts = fold_cutoffs(VAL)
    ci, ri, zy = build_index(cuts, blocks=1)
    assert data_hashes(cuts, ci, ri, zy) == baseline_result["clean_data_sha256"]

    batcher = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"],
                      np.random.default_rng(seed), workers=1,
                      aug=dict(mode="none", p=0.0, full=0.0),
                      aug_seed=[seed, 0xA7A1], depth=dict(p=0.0, grid=DEPTH_GRID))
    groups = batcher._plan()
    packed = pack_groups(groups)
    batch_seed = batcher.arng.integers(0, 2 ** 62, size=len(groups), dtype=np.int64)
    n_steps = len(groups)

    pos = np.flatnonzero(zy > 0).astype(np.int32)
    clean_center = np.array([zy[(ci == k) & (zy > 0)].mean()
                             for k in range(len(cuts))], np.float32)
    extra_rows, extra_z, extra_ci, extra_uid = collect_extra_positive_group_b()
    assert len(extra_rows) and bool((extra_z > 0).all())
    assert bool((user_group(extra_uid) == 1).all())
    extra_center = np.array([extra_z[extra_ci == k].mean()
                             for k in range(len(EXTRA_CUTOFFS))], np.float32)

    early_limit = max(1, len(cuts) // 3)  # exact COND-VOL rule from src.seq_cond
    clean_uid = panel()[2][ri]
    early_b = np.flatnonzero((zy > 0) & (ci < early_limit)
                             & (user_group(clean_uid) == 1)).astype(np.int32)
    assert len(early_b)
    rng = np.random.default_rng([seed, 0xC04D, EXP_NUM])
    vol_added_index = rng.choice(early_b, size=len(extra_rows), replace=True).astype(np.int32)
    common_index = rng.choice(pos, size=n_steps * COND_HALF_BATCH,
                              replace=True).astype(np.int32)
    added_slot = rng.integers(0, len(extra_rows), size=n_steps * COND_HALF_BATCH,
                              dtype=np.int32)
    val_user, val_rows, val_y = fixed_validation()
    lr_mult = lr_multiplier_plan(n_steps)
    cond_init = prepare_conditional_initialization(seed, dim=3 * int(cfg["hidden"]))
    snaps = snapshot_steps(n_steps)

    arrays = dict(
        cuts=np.asarray([x.isoformat() for x in cuts], dtype="U10"),
        extra_cuts=np.asarray([x.isoformat() for x in EXTRA_CUTOFFS], dtype="U10"),
        batch_seed=batch_seed,
        common_index=common_index.reshape(n_steps, COND_HALF_BATCH),
        added_slot=added_slot.reshape(n_steps, COND_HALF_BATCH),
        vol_added_index=vol_added_index,
        extra_rows=extra_rows,
        extra_z=extra_z,
        extra_ci=extra_ci,
        clean_center=clean_center,
        extra_center=extra_center,
        lr_multiplier=lr_mult,
        encoder_lr=ENCODER_LR * lr_mult,
        conditional_lr=COND_LR * lr_mult,
        val_user_id=val_user,
        val_rows=val_rows,
        val_y=val_y,
        snapshot_steps=np.asarray(snaps, np.int64),
        **packed,
    )
    array_hashes = {name: sha256_array(value) for name, value in arrays.items()}
    plan_sha = _save_npz_new(npz_path, arrays)
    core = dict(
        format=1, experiment=EXP_ID, kind="pair", seed=seed, val=VAL,
        arms=[arm_name(seed, ARM_VOL), arm_name(seed, ARM_FRESH)],
        baseline_name=baseline_name(seed),
        baseline_checkpoint=str(baseline_checkpoint(seed).resolve()),
        baseline_checkpoint_sha256=baseline_result["checkpoint_file_sha256"],
        baseline_model_sha256=baseline_result["final_model_sha256"],
        baseline_prediction_sha256=baseline_result["prediction"]["sha256"],
        conditional_initialization=cond_init,
        n_cutoffs=len(cuts), n_examples=len(zy), n_steps=n_steps,
        direct_batch=int(cfg["batch"]), common_conditional_batch=COND_HALF_BATCH,
        added_conditional_batch=COND_HALF_BATCH,
        n_common_slots=n_steps * COND_HALF_BATCH,
        n_added_slots=n_steps * COND_HALF_BATCH,
        n_extra_positive=len(extra_rows), n_vol_donor_pool=len(vol_added_index),
        vol_semantics="early CLEAN-positive group B; first third of CLEAN cutoffs; replacement",
        fresh_semantics="all EXTRA-positive group B; depth_clip=289; conditional loss only",
        common_semantics="all CLEAN-positive training rows; common materialized index plan",
        extra_depth_clip=289, clean_data_sha256=data_hashes(cuts, ci, ri, zy),
        validation_order_sha256=sha256_array(val_user),
        direct_plan_sha256=hashlib.sha256(b"".join(
            np.ascontiguousarray(packed[k]).tobytes() for k in sorted(packed))).hexdigest(),
        common_plan_sha256=sha256_array(arrays["common_index"]),
        added_slot_plan_sha256=sha256_array(arrays["added_slot"]),
        lr_sha256=dict(encoder=sha256_array(arrays["encoder_lr"]),
                       conditional=sha256_array(arrays["conditional_lr"])),
        snapshots=snaps,
        fine_tune_cfg=dict(
            encoder_lr=ENCODER_LR, conditional_head_lr=COND_LR,
            lambda_cond=LAMBDA_COND, optimizer="AdamW", betas=[0.9, 0.98],
            weight_decay=WEIGHT_DECAY, grad_clip=1.0, clean_epochs=PAIR_EPOCHS,
            warmup=WARMUP, schedule="same cosine multiplier formula as exp_043",
            conditional_head="Linear(192,64)-GELU-Dropout(0.1)-Linear(64,1)",
            direct_head="frozen; used for direct prediction only",
        ),
        centering=dict(
            clean="positive CLEAN training rows grouped by their cutoff",
            extra="positive EXTRA group-B training rows grouped by their cutoff",
            validation_used=False, test_used=False, other_arm_used=False,
        ),
        array_sha256=array_hashes, source_sha256=source_hashes(),
    )
    plan_id = hashlib.sha256(json.dumps(jsonable(core), sort_keys=True).encode()).hexdigest()
    meta = dict(core, plan_id=plan_id, plan_file_sha256=plan_sha,
                created_at=dt.datetime.now().isoformat(timespec="seconds"))
    write_json_new(json_path, meta)
    log(f"paired plan seed {seed}: {n_steps:,} steps, {len(extra_rows):,} EXTRA-B positives")
    return meta


def build_baseline_optimizer(model: torch.nn.Module, cfg: dict[str, Any]):
    decay = [p for _, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for _, p in model.named_parameters() if p.dim() <= 1]
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=cfg["wd"]),
         dict(params=nodecay, weight_decay=0.0)],
        lr=cfg["lr"], betas=(0.9, 0.98),
    )


def freeze_direct_head(model: torch.nn.Module) -> None:
    for p in model.head.parameters():
        p.requires_grad_(False)


def encoder_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for name, p in model.named_parameters()
            if not name.startswith("head.") and p.requires_grad]


def build_pair_optimizer(model: torch.nn.Module, conditional: torch.nn.Module):
    enc = encoder_parameters(model)
    cond = [p for p in conditional.parameters() if p.requires_grad]
    assert enc and cond and all(not p.requires_grad for p in model.head.parameters())
    return torch.optim.AdamW(
        [dict(params=enc, lr=ENCODER_LR, weight_decay=WEIGHT_DECAY,
              role="encoder"),
         dict(params=cond, lr=COND_LR, weight_decay=WEIGHT_DECAY,
              role="conditional")],
        betas=(0.9, 0.98),
    )


def iter_materialized_batches(batcher: Batcher, plan: Any, n_steps: int | None = None):
    total = len(plan["batch_seed"]) if n_steps is None else n_steps
    q: Queue = Queue(maxsize=4)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for batch_i in range(total):
                group = unpack_group(plan, batch_i)
                q.put((batch_i, batcher._make(group, int(plan["batch_seed"][batch_i]))))
        except BaseException as exc:
            errors.append(exc)
            q.put((-1, None))

    thread = threading.Thread(target=work, daemon=True, name="exp044-worker-1")
    thread.start()
    for expected in range(total):
        batch_i, value = q.get()
        if errors:
            raise errors[0]
        assert batch_i == expected, f"materialized order changed: {batch_i} != {expected}"
        yield value
    thread.join()
    if errors:
        raise errors[0]


def prediction_record(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    offset, calibrated = calibrate(y, z)
    positive = y > 0
    positive_error = float(np.sqrt(np.mean(
        (np.log1p(y[positive]) - (z[positive] + offset)) ** 2)))
    return dict(
        n=len(z), dtype=str(z.dtype), rmsle_raw=float(rmsle_z(y, z)),
        rmsle_cal=float(calibrated), offset=float(offset), bias=float(bias_z(y, z)),
        auc=float(_auc(y > 0, z)), positive_only_error=positive_error,
        mean_z=float(z.mean()), std_z=float(z.std()), sha256=sha256_array(z),
    )


def _temporary_run_dir(parent: Path, tag: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"_work_{tag}_{os.getpid()}"
    assert not path.exists()
    path.mkdir()
    return path


def _finish_run_dir(work: Path, final: Path) -> None:
    assert not final.exists(), f"refusing to overwrite completed run: {final}"
    work.rename(final)


def save_training_snapshot(run_dir: Path, step: int, model: torch.nn.Module,
                           conditional: torch.nn.Module | None,
                           opt: torch.optim.Optimizer,
                           gradient_norms: dict[str, float] | None = None) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rng_state = capture_rng_state()
    record = dict(
        step=step, model_sha256=module_hash(model),
        direct_head_sha256=direct_head_hash(model),
        conditional_head_sha256=(module_hash(conditional) if conditional is not None else None),
        optimizer_sha256=optimizer_hash(model, conditional, opt),
        rng_sha256=rng_state_hash(rng_state), gradient_norms=gradient_norms,
    )
    path = run_dir / f"snapshot_step_{step:05d}.pt"
    assert not path.exists()
    torch.save(dict(
        step=step,
        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        conditional=(None if conditional is None else
                     {k: v.detach().cpu().clone() for k, v in conditional.state_dict().items()}),
        optimizer=opt.state_dict(), rng=rng_state,
    ), path)
    record.update(path=str(path.resolve()), file_sha256=sha256_file(path))
    return record


def _rebuild_clean(meta: dict[str, Any]) -> tuple[list[dt.date], np.ndarray, np.ndarray,
                                                    np.ndarray]:
    cuts = fold_cutoffs(VAL)
    ci, ri, zy = build_index(cuts, blocks=1)
    assert data_hashes(cuts, ci, ri, zy) == meta["clean_data_sha256"]
    return cuts, ci, ri, zy


def run_baseline(seed: int) -> dict[str, Any]:
    name = baseline_name(seed)
    final_dir = BASELINES / name
    assert not final_dir.exists(), f"baseline already exists: {final_dir}"
    assert not baseline_checkpoint(seed).exists(), (
        f"checkpoint already exists; refusing overwrite: {baseline_checkpoint(seed)}")
    plan, meta = _load_plan(*baseline_plan_paths(seed))
    cfg = dict(meta["cfg"])
    cfg["depth_grid"] = tuple(cfg["depth_grid"])
    validate_plain_baseline_cfg(cfg, name)
    env = configure_determinism(seed)
    assert torch.cuda.is_available(), "EXP-044 must exercise the CUDA bf16 path"
    dev = torch.device("cuda")
    cuts, ci, ri, zy = _rebuild_clean(meta)
    batcher = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"],
                      np.random.default_rng(seed), workers=1,
                      aug=dict(mode="none", p=0.0, full=0.0),
                      aug_seed=[seed, 0xA7A1], depth=dict(p=0.0, grid=DEPTH_GRID))
    work = _temporary_run_dir(BASELINES, name)
    started = time.time()
    model = build_model(cfg).to(dev)
    model.train()
    opt = build_baseline_optimizer(model, cfg)
    initial_model_hash = module_hash(model)
    initial_optimizer_hash = optimizer_hash(model, None, opt)
    snapshots = [save_training_snapshot(work, 0, model, None, opt)]
    requested = set(int(x) for x in plan["snapshot_steps"])
    scale = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    run_loss = None
    seen = 0
    epoch_losses = []
    epoch_offsets = set(int(x) for x in plan["epoch_step_offsets"][1:])
    epoch_run = None
    epoch_seen = 0
    for step0, (x, yb) in enumerate(iter_materialized_batches(batcher, plan)):
        lr = float(plan["lr_plan"][step0])
        for group in opt.param_groups:
            group["lr"] = lr
        n = len(yb)
        t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
        t[:, :N_CH_STORED] *= scale
        target = torch.from_numpy(yb).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss = torch.nn.functional.mse_loss(model(t), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss = loss.detach() * n if run_loss is None else run_loss + loss.detach() * n
        epoch_run = loss.detach() * n if epoch_run is None else epoch_run + loss.detach() * n
        seen += n
        epoch_seen += n
        completed = step0 + 1
        if completed in requested:
            snapshots.append(save_training_snapshot(work, completed, model, None, opt))
        if completed in epoch_offsets:
            epoch_losses.append(float(epoch_run) / epoch_seen)
            epoch_run, epoch_seen = None, 0
            log(f"{name}: epoch {len(epoch_losses)}/4, step {completed:,}, "
                f"MSE={epoch_losses[-1]:.6f}")
    assert seen == BASELINE_EPOCHS * len(zy)

    val_user = plan["val_user_id"]
    val_rows = plan["val_rows"]
    val_y = plan["val_y"]
    z = predict(model, VAL, val_rows, cfg, dev).astype(np.float32, copy=False)
    assert np.array_equal(val_user, fixed_validation()[0])
    z_path = work / "z_raw.npy"
    np.save(z_path, z)
    pred = prediction_record(val_y, z)
    pred.update(file=str(z_path.resolve()), file_sha256=sha256_file(z_path))

    checkpoint_tmp = work / f"model_{name}.pt"
    checkpoint = dict(
        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        cfg=cfg, val=VAL.isoformat(), experiment_name=name, experiment=EXP_ID,
        plan_id=meta["plan_id"],
    )
    torch.save(checkpoint, checkpoint_tmp)
    final_model_hash = state_dict_hash(checkpoint["state"])
    assert final_model_hash == module_hash(model)
    result = dict(
        experiment=EXP_ID, kind="baseline", name=name, seed=seed, val=VAL,
        cfg=cfg, plan_id=meta["plan_id"], plan_file_sha256=meta["plan_file_sha256"],
        epoch_plan_sha256=meta["epoch_plan_sha256"],
        clean_data_sha256=meta["clean_data_sha256"],
        validation_order_sha256=meta["validation_order_sha256"],
        initial_model_sha256=initial_model_hash,
        initial_optimizer_sha256=initial_optimizer_hash,
        final_model_sha256=final_model_hash,
        final_direct_head_sha256=direct_head_hash(model),
        final_optimizer_sha256=optimizer_hash(model, None, opt),
        final_rng_sha256=rng_state_hash(), snapshots=snapshots,
        n_steps=int(meta["n_steps"]), n_examples_per_epoch=len(zy),
        train_mse=float(run_loss) / seen, epoch_train_mse=epoch_losses,
        prediction=pred, environment=env, duration_seconds=time.time() - started,
    )
    cfg_path = work / "cfg.json"
    cfg_path.write_text(json.dumps(jsonable(cfg), indent=2, sort_keys=True), encoding="utf-8")
    result["cfg_file"] = str(cfg_path.resolve())
    result_path = work / "result.json"
    result_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True),
                           encoding="utf-8")
    _finish_run_dir(work, final_dir)

    # Move the checkpoint only after the run directory is complete.  There is no
    # code path that overwrites a prior checkpoint.
    checkpoint_tmp_final = final_dir / checkpoint_tmp.name
    checkpoint_tmp_final.rename(baseline_checkpoint(seed))
    result_path = final_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["checkpoint_path"] = str(baseline_checkpoint(seed).resolve())
    result["checkpoint_file_sha256"] = sha256_file(baseline_checkpoint(seed))
    # Paths inside the temp directory changed after rename; report canonical ones.
    result["prediction"]["file"] = str((final_dir / "z_raw.npy").resolve())
    result["cfg_file"] = str((final_dir / "cfg.json").resolve())
    for snap in result["snapshots"]:
        snap["path"] = str((final_dir / Path(snap["path"]).name).resolve())
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log(f"{name}: RMSLE_cal={pred['rmsle_cal']:.9f}, z={pred['sha256']}")
    return result


def gather_index_rows(cuts: list[dt.date], ci: np.ndarray, ri: np.ndarray,
                      indices: np.ndarray, *, depth_clip: int | None = None) -> np.ndarray:
    indices = np.asarray(indices, np.int64)
    out = np.empty((len(indices), 365, 17), np.float16)
    for cutoff_i in np.unique(ci[indices]):
        where = np.flatnonzero(ci[indices] == cutoff_i)
        out[where] = gather(cuts[int(cutoff_i)], ri[indices[where]],
                            depth_clip=depth_clip)
    return out


def gather_extra_rows(extra_cuts: list[dt.date], extra_ci: np.ndarray,
                      extra_rows: np.ndarray, indices: np.ndarray,
                      *, depth_clip: int) -> np.ndarray:
    indices = np.asarray(indices, np.int64)
    out = np.empty((len(indices), 365, 17), np.float16)
    for cutoff_i in np.unique(extra_ci[indices]):
        where = np.flatnonzero(extra_ci[indices] == cutoff_i)
        out[where] = gather(extra_cuts[int(cutoff_i)], extra_rows[indices[where]],
                            depth_clip=depth_clip)
    return out


def gradient_l2(grads: Iterable[torch.Tensor | None]) -> float:
    total = None
    for grad in grads:
        if grad is None:
            continue
        value = grad.detach().float().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else float(torch.sqrt(total))


def predict_pair_outputs(model: torch.nn.Module, conditional: torch.nn.Module,
                         cfg: dict[str, Any], rows: np.ndarray, dev: torch.device,
                         batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    conditional.eval()
    scale = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    z = np.empty(len(rows), np.float32)
    r = np.empty(len(rows), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = gather(VAL, rows[i:i + batch])
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= scale
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pooled = _pool(model.encode(t))
                # Primary inference is exactly the original direct head.  The
                # conditional output is retained only as an auxiliary diagnostic.
                zo = model.head(pooled).squeeze(1)
                ro = conditional(pooled).squeeze(1)
            z[i:i + len(x)] = zo.float().cpu().numpy()
            r[i:i + len(x)] = ro.float().cpu().numpy()
    model.train()
    conditional.train()
    return z, r


def auxiliary_conditional_rmse(y: np.ndarray, r: np.ndarray,
                               train_center: float) -> float:
    positive = y > 0
    target = np.log1p(y[positive]) - train_center
    return float(np.sqrt(np.mean((target - r[positive]) ** 2)))


def _snapshot_prediction(run_dir: Path, step: int, model: torch.nn.Module,
                         conditional: torch.nn.Module, cfg: dict[str, Any],
                         val_rows: np.ndarray, val_y: np.ndarray,
                         train_center: float, dev: torch.device) -> dict[str, Any]:
    z, r = predict_pair_outputs(model, conditional, cfg, val_rows, dev)
    z_path = run_dir / f"z_raw_step_{step:05d}.npy"
    r_path = run_dir / f"conditional_raw_step_{step:05d}.npy"
    np.save(z_path, z)
    np.save(r_path, r)
    record = prediction_record(val_y, z)
    record.update(
        file=str(z_path.resolve()), file_sha256=sha256_file(z_path),
        conditional_file=str(r_path.resolve()),
        conditional_file_sha256=sha256_file(r_path),
        auxiliary_conditional_rmse=auxiliary_conditional_rmse(val_y, r, train_center),
        inference_uses_conditional_head=False,
    )
    return record


def _fix_paths_after_rename(result: dict[str, Any], final_dir: Path) -> None:
    for snap in result["snapshots"]:
        snap["path"] = str((final_dir / Path(snap["path"]).name).resolve())
    for record in result["predictions"].values():
        record["file"] = str((final_dir / Path(record["file"]).name).resolve())
        record["conditional_file"] = str(
            (final_dir / Path(record["conditional_file"]).name).resolve())


def run_pair_arm(seed: int, arm: str, destination: Path,
                 max_steps: int | None = None) -> dict[str, Any]:
    assert arm in ARMS_ALLOWED
    assert not destination.exists(), f"refusing to overwrite completed run: {destination}"
    plan, meta = _load_plan(*pair_plan_paths(seed))
    checkpoint = torch.load(Path(meta["baseline_checkpoint"]), map_location="cpu",
                            weights_only=False)
    validate_baseline_checkpoint(checkpoint, baseline_name(seed))
    assert sha256_file(Path(meta["baseline_checkpoint"])) == meta["baseline_checkpoint_sha256"]
    cfg = dict(checkpoint["cfg"])
    cfg["depth_grid"] = tuple(cfg["depth_grid"])
    validate_plain_baseline_cfg(cfg, baseline_name(seed))
    env = configure_determinism(seed)
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    cuts, ci, ri, zy = _rebuild_clean(meta)
    extra_cuts = [dt.date.fromisoformat(str(x)) for x in plan["extra_cuts"]]
    assert extra_cuts == list(EXTRA_CUTOFFS)
    assert meta["extra_depth_clip"] == 289
    n_steps = int(meta["n_steps"] if max_steps is None else min(max_steps, meta["n_steps"]))
    tag = destination.name
    work = _temporary_run_dir(destination.parent, tag)
    started = time.time()

    model = build_model(cfg).to(dev)
    model.load_state_dict(checkpoint["state"])
    freeze_direct_head(model)
    model.train()
    conditional = build_head(3 * int(cfg["hidden"]), 64, 0.10, 0.0).to(dev)
    cond_state = torch.load(pair_cond_init_path(seed), map_location="cpu", weights_only=False)
    assert state_dict_hash(cond_state) == meta["conditional_initialization"]["state_sha256"]
    conditional.load_state_dict(cond_state)
    conditional.train()
    opt = build_pair_optimizer(model, conditional)

    initial = dict(
        model_sha256=module_hash(model), direct_head_sha256=direct_head_hash(model),
        conditional_head_sha256=module_hash(conditional),
        optimizer_sha256=optimizer_hash(model, conditional, opt),
        rng_sha256=rng_state_hash(),
    )
    assert initial["model_sha256"] == meta["baseline_model_sha256"]
    assert initial["conditional_head_sha256"] == meta["conditional_initialization"]["state_sha256"]
    direct_initial = initial["direct_head_sha256"]
    snapshots = [save_training_snapshot(work, 0, model, conditional, opt)]
    wanted = {x for x in (int(v) for v in plan["snapshot_steps"]) if x <= n_steps}
    wanted.add(n_steps)

    batcher = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"],
                      np.random.default_rng(seed), workers=1,
                      aug=dict(mode="none", p=0.0, full=0.0),
                      aug_seed=[seed, 0xA7A1], depth=dict(p=0.0, grid=DEPTH_GRID))
    scale = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    enc_params = encoder_parameters(model)
    trainable = enc_params + list(conditional.parameters())
    run_direct = None
    run_cond = None
    seen_direct = 0
    seen_cond = 0
    gradients: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    train_center = float(np.asarray(plan["clean_center"], np.float64).mean())
    extra_rows = plan["extra_rows"]
    extra_z = plan["extra_z"]
    extra_ci = plan["extra_ci"]
    clean_center = plan["clean_center"]
    extra_center = plan["extra_center"]

    for step0, (x_direct, y_direct) in enumerate(
            iter_materialized_batches(batcher, plan, n_steps=n_steps)):
        common_idx = plan["common_index"][step0]
        slot = plan["added_slot"][step0]
        x_common = gather_index_rows(cuts, ci, ri, common_idx)
        r_common = zy[common_idx] - clean_center[ci[common_idx]]
        if arm == ARM_VOL:
            added_idx = plan["vol_added_index"][slot]
            x_added = gather_index_rows(cuts, ci, ri, added_idx)
            r_added = zy[added_idx] - clean_center[ci[added_idx]]
        else:
            added_idx = slot
            x_added = gather_extra_rows(extra_cuts, extra_ci, extra_rows, added_idx,
                                        depth_clip=289)
            r_added = extra_z[added_idx] - extra_center[extra_ci[added_idx]]
            assert bool((extra_z[added_idx] > 0).all())
        assert x_common.shape == x_added.shape == (COND_HALF_BATCH, 365, 17)
        x_cond = np.concatenate([x_common, x_added])
        r_target = np.concatenate([r_common, r_added]).astype(np.float32, copy=False)
        x_all = np.concatenate([x_direct, x_cond])
        n_direct = len(x_direct)

        t = torch.from_numpy(x_all).to(dev).permute(0, 2, 1).contiguous().float()
        t[:, :N_CH_STORED] *= scale
        target_direct = torch.from_numpy(y_direct).to(dev)
        target_cond = torch.from_numpy(r_target).to(dev)
        mult = float(plan["lr_multiplier"][step0])
        opt.param_groups[0]["lr"] = ENCODER_LR * mult
        opt.param_groups[1]["lr"] = COND_LR * mult
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            pooled = _pool(model.encode(t))
            z_direct = model.head(pooled[:n_direct]).squeeze(1)
            r_cond = conditional(pooled[n_direct:]).squeeze(1)
            loss_direct = torch.nn.functional.mse_loss(z_direct, target_direct)
            loss_cond = torch.nn.functional.mse_loss(r_cond, target_cond)
            loss = loss_direct + LAMBDA_COND * loss_cond
        opt.zero_grad(set_to_none=True)
        completed = step0 + 1
        grad_record = None
        if completed in wanted:
            gd = torch.autograd.grad(loss_direct, enc_params, retain_graph=True,
                                     allow_unused=False)
            gc = torch.autograd.grad(loss_cond, enc_params, retain_graph=True,
                                     allow_unused=False)
            grad_record = dict(
                encoder_from_direct=gradient_l2(gd),
                encoder_from_conditional=gradient_l2(gc),
                encoder_from_weighted_conditional=LAMBDA_COND * gradient_l2(gc),
            )
            gradients[str(completed)] = grad_record
        loss.backward()
        assert all(p.grad is None for p in model.head.parameters()), (
            "frozen direct head received a gradient")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        run_direct = (loss_direct.detach() * n_direct if run_direct is None
                      else run_direct + loss_direct.detach() * n_direct)
        run_cond = (loss_cond.detach() * len(r_target) if run_cond is None
                    else run_cond + loss_cond.detach() * len(r_target))
        seen_direct += n_direct
        seen_cond += len(r_target)
        if completed in wanted:
            snap = save_training_snapshot(work, completed, model, conditional, opt,
                                          gradient_norms=grad_record)
            snapshots.append(snap)
            if completed in {int(meta["n_steps"]) // 2, n_steps}:
                predictions[str(completed)] = _snapshot_prediction(
                    work, completed, model, conditional, cfg,
                    plan["val_rows"], plan["val_y"], train_center, dev)
        if completed % 500 == 0 or completed == n_steps:
            log(f"{tag}: step {completed:,}/{n_steps:,}, lrE={ENCODER_LR * mult:.3g}, "
                f"lrC={COND_LR * mult:.3g}")

    assert direct_head_hash(model) == direct_initial, "direct head changed while frozen"
    assert all(p.grad is None for p in model.head.parameters())
    final = dict(
        model_sha256=module_hash(model), direct_head_sha256=direct_head_hash(model),
        conditional_head_sha256=module_hash(conditional),
        optimizer_sha256=optimizer_hash(model, conditional, opt),
        rng_sha256=rng_state_hash(),
    )
    # For truncated replay, the final prediction is always required even though
    # step 100 is not a half/full endpoint of the complete plan.
    if str(n_steps) not in predictions:
        predictions[str(n_steps)] = _snapshot_prediction(
            work, n_steps, model, conditional, cfg,
            plan["val_rows"], plan["val_y"], train_center, dev)
    result = dict(
        experiment=EXP_ID, kind=("integration_replay" if max_steps is not None else "arm"),
        name=tag, seed=seed, arm=arm, val=VAL, plan_id=meta["plan_id"],
        paired_plan_path=str(pair_plan_paths(seed)[0].resolve()),
        paired_plan_sha256=meta["plan_file_sha256"],
        baseline_checkpoint_path=meta["baseline_checkpoint"],
        baseline_checkpoint_sha256=meta["baseline_checkpoint_sha256"],
        baseline_prediction_sha256=meta["baseline_prediction_sha256"],
        initial=initial, final=final, snapshots=snapshots, gradients=gradients,
        predictions=predictions, validation_order_sha256=meta["validation_order_sha256"],
        direct_plan_sha256=meta["direct_plan_sha256"],
        common_plan_sha256=meta["common_plan_sha256"],
        added_slot_plan_sha256=meta["added_slot_plan_sha256"],
        lr_sha256=meta["lr_sha256"],
        n_steps=n_steps, full_plan_steps=int(meta["n_steps"]),
        n_direct_examples=seen_direct, n_conditional_examples=seen_cond,
        n_common_slots=n_steps * COND_HALF_BATCH,
        n_added_slots=n_steps * COND_HALF_BATCH,
        conditional_batch_shape=[2 * COND_HALF_BATCH, 365, 17],
        direct_train_loss=float(run_direct) / seen_direct,
        conditional_train_loss=float(run_cond) / seen_cond,
        direct_head_frozen=True, direct_head_received_gradients=False,
        extra_in_direct_loss=False, extra_positive_only=True,
        extra_donor_group="B", extra_depth_clip=289,
        inference_uses_conditional_head=False,
        environment=env, duration_seconds=time.time() - started,
    )
    result_path = work / "result.json"
    result_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True),
                           encoding="utf-8")
    _finish_run_dir(work, destination)
    result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
    _fix_paths_after_rename(result, destination)
    (destination / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True),
                                              encoding="utf-8")
    endpoint = result["predictions"][str(n_steps)]
    log(f"{tag}: RMSLE_cal={endpoint['rmsle_cal']:.9f}, z={endpoint['sha256']}")
    return result


def compare_replay(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    step = str(a["n_steps"])
    za = np.load(a["predictions"][step]["file"])
    zb = np.load(b["predictions"][step]["file"])
    delta = za.astype(np.float64) - zb.astype(np.float64)
    checks = dict(
        predictions=np.array_equal(za, zb),
        model_state=a["final"]["model_sha256"] == b["final"]["model_sha256"],
        optimizer_state=a["final"]["optimizer_sha256"] == b["final"]["optimizer_sha256"],
        conditional_head=a["final"]["conditional_head_sha256"] == b["final"]["conditional_head_sha256"],
        python_numpy_torch_cuda_rng=a["final"]["rng_sha256"] == b["final"]["rng_sha256"],
    )
    snapshot_checks = []
    for sa, sb in zip(a["snapshots"], b["snapshots"], strict=True):
        assert sa["step"] == sb["step"]
        snapshot_checks.append(dict(
            step=sa["step"],
            model=sa["model_sha256"] == sb["model_sha256"],
            direct_head=sa["direct_head_sha256"] == sb["direct_head_sha256"],
            conditional=sa["conditional_head_sha256"] == sb["conditional_head_sha256"],
            optimizer=sa["optimizer_sha256"] == sb["optimizer_sha256"],
            rng=sa["rng_sha256"] == sb["rng_sha256"],
        ))
    passed = all(checks.values()) and all(all(v for k, v in row.items() if k != "step")
                                           for row in snapshot_checks)
    return dict(
        technical_pass=passed, checks=checks, snapshot_checks=snapshot_checks,
        prediction_sha256=[a["predictions"][step]["sha256"],
                           b["predictions"][step]["sha256"]],
        var_delta_z=float(np.var(delta)), max_abs_delta_z=float(np.max(np.abs(delta))),
        mean_delta_z=float(delta.mean()),
    )


def child_env(seed: int) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _launch(args: list[str], seed: int) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    subprocess.run(cmd, cwd=ROOT, env=child_env(seed), check=True)


def launch_baseline(seed: int) -> None:
    result = BASELINES / baseline_name(seed) / "result.json"
    if result.exists() and baseline_checkpoint(seed).exists():
        _checkpoint_result(seed)
        log(f"reuse completed baseline {baseline_name(seed)}")
        return
    assert not result.exists() and not baseline_checkpoint(seed).exists(), (
        "partial baseline artifact found; refusing to overwrite")
    _launch(["baseline", "--seed", str(seed)], seed)


def launch_arm(seed: int, arm: str) -> None:
    destination = ARMS / arm_name(seed, arm)
    result = destination / "result.json"
    if result.exists():
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["plan_id"] == _load_plan(*pair_plan_paths(seed))[1]["plan_id"]
        log(f"reuse completed arm {arm_name(seed, arm)}")
        return
    assert not destination.exists(), "partial arm artifact found; refusing to overwrite"
    _launch(["arm", "--seed", str(seed), "--arm", arm], seed)


def run_integration_replay() -> dict[str, Any]:
    comparison_path = REPLAY / "comparison.json"
    if comparison_path.exists():
        result = json.loads(comparison_path.read_text(encoding="utf-8"))
        assert result["technical_pass"], "stored integration replay failed"
        log("reuse strict 100-step integration PASS")
        return result
    for repeat in (1, 2):
        destination = REPLAY / f"FT-VOL-S42-run{repeat}"
        if not (destination / "result.json").exists():
            assert not destination.exists(), "partial replay artifact found; refusing overwrite"
            _launch(["replay", "--run", str(repeat)], SEED)
    a = json.loads((REPLAY / "FT-VOL-S42-run1" / "result.json").read_text(encoding="utf-8"))
    b = json.loads((REPLAY / "FT-VOL-S42-run2" / "result.json").read_text(encoding="utf-8"))
    result = compare_replay(a, b)
    REPLAY.mkdir(parents=True, exist_ok=True)
    write_json_new(comparison_path, result)
    assert result["technical_pass"], (
        "100-step integration replay is not bitwise deterministic; full pilot is blocked")
    log("strict 100-step integration replay: PASS (all requested states bitwise equal)")
    return result


def load_completed_arm(seed: int, arm: str) -> dict[str, Any]:
    path = ARMS / arm_name(seed, arm) / "result.json"
    assert path.exists()
    return json.loads(path.read_text(encoding="utf-8"))


def apply_global_offset_error(y: np.ndarray, z: np.ndarray, offset: float,
                              mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y[mask]) - (z[mask] + offset)) ** 2)))


def decision_from_metrics(mean_delta: float, negative_seeds: int,
                          mean_positive_delta: float, positive_better_seeds: int,
                          mean_auc_delta: float, pooled_var: float,
                          mean_aux_delta: float, max_abs_delta: float,
                          technical_pass: bool,
                          mean_fresh_vs_baseline: float) -> tuple[str, bool, list[str]]:
    reasons = []
    signal = bool(
        technical_pass
        and mean_delta <= -0.0007
        and negative_seeds >= 2
        and mean_positive_delta < 0
        and positive_better_seeds >= 2
        and mean_auc_delta >= -0.0002
        and pooled_var < 0.05
    )
    if signal:
        decision = "SIGNAL PASS"
    else:
        hard_fail = (
            not technical_pass
            or mean_delta > -0.0003
            or negative_seeds <= 1
            or mean_positive_delta >= 0
            or mean_auc_delta < -0.0002
            or pooled_var >= 0.05
            or (mean_aux_delta < 0 and max_abs_delta == 0.0)
        )
        decision = "REJECT" if hard_fail else "INCONCLUSIVE"
    if not technical_pass:
        reasons.append("technical invariant/replay failed")
    if mean_delta > -0.0003:
        reasons.append("mean delta is above the rejection boundary -0.0003")
    if negative_seeds <= 1:
        reasons.append("FRESH is better on at most 1/3 seeds")
    if mean_positive_delta >= 0:
        reasons.append("mean positive-only error did not improve")
    if mean_auc_delta < -0.0002:
        reasons.append("mean AUC delta is below -0.0002")
    if pooled_var >= 0.05:
        reasons.append("Var(z_fresh-z_vol) is at least 0.05")
    if mean_aux_delta < 0 and max_abs_delta == 0.0:
        reasons.append("conditional metric improved but direct output did not change")
    promote = bool(signal and mean_fresh_vs_baseline <= 0.0005)
    return decision, promote, reasons


def analyze_results(technical: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    segment_rows = []
    pooled_deltas = []
    for seed in SEEDS:
        baseline = json.loads((BASELINES / baseline_name(seed) / "result.json").read_text(
            encoding="utf-8"))
        vol = load_completed_arm(seed, ARM_VOL)
        fresh = load_completed_arm(seed, ARM_FRESH)
        assert vol["baseline_checkpoint_sha256"] == fresh["baseline_checkpoint_sha256"]
        assert vol["initial"] == fresh["initial"]
        for key in ("direct_plan_sha256", "common_plan_sha256", "added_slot_plan_sha256",
                    "lr_sha256", "validation_order_sha256"):
            assert vol[key] == fresh[key]
        assert vol["n_steps"] == fresh["n_steps"]
        assert vol["n_added_slots"] == fresh["n_added_slots"]
        assert vol["conditional_batch_shape"] == fresh["conditional_batch_shape"]
        assert vol["initial"]["direct_head_sha256"] == vol["final"]["direct_head_sha256"]
        assert fresh["initial"]["direct_head_sha256"] == fresh["final"]["direct_head_sha256"]

        endpoint = str(vol["n_steps"])
        zv = np.load(vol["predictions"][endpoint]["file"])
        zf = np.load(fresh["predictions"][endpoint]["file"])
        zb = np.load(baseline["prediction"]["file"])
        y = np.load(pair_plan_paths(seed)[0])["val_y"]
        uid = np.load(pair_plan_paths(seed)[0])["val_user_id"]
        assert sha256_array(uid) == vol["validation_order_sha256"]
        assert len(zv) == len(zf) == len(zb) == len(y)
        pv = vol["predictions"][endpoint]
        pf = fresh["predictions"][endpoint]
        pb = baseline["prediction"]
        delta = zf.astype(np.float64) - zv.astype(np.float64)
        pooled_deltas.append(delta)
        row = dict(
            seed=seed,
            baseline_rmsle_cal=pb["rmsle_cal"],
            vol_rmsle_cal=pv["rmsle_cal"], fresh_rmsle_cal=pf["rmsle_cal"],
            delta=pf["rmsle_cal"] - pv["rmsle_cal"],
            vol_vs_baseline=pv["rmsle_cal"] - pb["rmsle_cal"],
            fresh_vs_baseline=pf["rmsle_cal"] - pb["rmsle_cal"],
            auc_vol=pv["auc"], auc_fresh=pf["auc"], auc_delta=pf["auc"] - pv["auc"],
            positive_vol=pv["positive_only_error"],
            positive_fresh=pf["positive_only_error"],
            positive_delta=pf["positive_only_error"] - pv["positive_only_error"],
            auxiliary_vol=pv["auxiliary_conditional_rmse"],
            auxiliary_fresh=pf["auxiliary_conditional_rmse"],
            auxiliary_delta=(pf["auxiliary_conditional_rmse"]
                             - pv["auxiliary_conditional_rmse"]),
            var_delta_z=float(np.var(delta)), max_abs_delta_z=float(np.max(np.abs(delta))),
            pearson=float(np.corrcoef(zf, zv)[0, 1]), mean_delta_z=float(delta.mean()),
            offset_difference=pf["offset"] - pv["offset"],
            baseline_prediction_sha256=pb["sha256"],
            vol_prediction_sha256=pv["sha256"], fresh_prediction_sha256=pf["sha256"],
        )
        rows.append(row)

        rv = np.load(pair_plan_paths(seed)[0])["val_rows"]
        masks = seg_masks(segments(VAL, rv))
        for name, mask in masks.items():
            if int(mask.sum()) < 100:
                continue
            ev = apply_global_offset_error(y, zv, pv["offset"], mask)
            ef = apply_global_offset_error(y, zf, pf["offset"], mask)
            segment_rows.append(dict(seed=seed, segment=name, n=int(mask.sum()),
                                     vol_error=ev, fresh_error=ef, delta=ef - ev))

    deltas = np.asarray([r["delta"] for r in rows])
    positive = np.asarray([r["positive_delta"] for r in rows])
    auc = np.asarray([r["auc_delta"] for r in rows])
    auxiliary = np.asarray([r["auxiliary_delta"] for r in rows])
    fresh_base = np.asarray([r["fresh_vs_baseline"] for r in rows])
    pooled = np.concatenate(pooled_deltas)
    summary = dict(
        per_seed=rows,
        mean_delta=float(deltas.mean()), median_delta=float(np.median(deltas)),
        sd_delta=float(deltas.std(ddof=1)), negative_deltas=int((deltas < 0).sum()),
        mean_positive_delta=float(positive.mean()),
        positive_better_seeds=int((positive < 0).sum()),
        mean_auc_delta=float(auc.mean()), mean_auxiliary_delta=float(auxiliary.mean()),
        pooled_var_delta_z=float(np.var(pooled)),
        pooled_max_abs_delta_z=float(np.max(np.abs(pooled))),
        mean_fresh_vs_baseline=float(fresh_base.mean()),
        integration_replay=technical,
    )
    decision, promote, reasons = decision_from_metrics(
        summary["mean_delta"], summary["negative_deltas"],
        summary["mean_positive_delta"], summary["positive_better_seeds"],
        summary["mean_auc_delta"], summary["pooled_var_delta_z"],
        summary["mean_auxiliary_delta"], summary["pooled_max_abs_delta_z"],
        bool(technical["technical_pass"]), summary["mean_fresh_vs_baseline"],
    )
    summary.update(decision=decision, promote_to_full_folds=promote,
                   decision_reasons=reasons,
                   interpretation=(
                       "fresh signal exists, current continuation recipe is not a viable member"
                       if decision == "SIGNAL PASS" and not promote else None),
                   scope=dict(
                       proven="paired contrast on newly trained deterministic plain SEQ-01 baselines",
                       not_transferable="historical SEQ-AVG3",
                       not_evaluated=["STRONGEST_CURRENT", "LOFO", "test", "LB", "submission"],
                   ))
    analysis_path = OUT / "analysis.json"
    if analysis_path.exists():
        stored = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert stored == jsonable(summary), "existing analysis differs; refusing overwrite"
    else:
        write_json_new(analysis_path, summary)
    for path, data in ((OUT / "seed_summary.csv", rows),
                       (OUT / "segment_summary.csv", segment_rows)):
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    log(f"decision={decision}; PROMOTE TO FULL FOLDS={'YES' if promote else 'NO'}")
    return summary


def run_all() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        prepare_baseline_plan(seed)
    for seed in SEEDS:
        launch_baseline(seed)
    for seed in SEEDS:
        prepare_pair_plan(seed)
    technical = run_integration_replay()
    assert technical["technical_pass"]
    for seed in SEEDS:
        launch_arm(seed, ARM_VOL)
        launch_arm(seed, ARM_FRESH)
    return analyze_results(technical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("all")
    sub.add_parser("prepare")
    b = sub.add_parser("baseline")
    b.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a = sub.add_parser("arm")
    a.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a.add_argument("--arm", required=True, choices=ARMS_ALLOWED)
    r = sub.add_parser("replay")
    r.add_argument("--run", type=int, required=True, choices=(1, 2))
    sub.add_parser("analyze")
    args = parser.parse_args()
    command = args.cmd or "all"
    if command == "all":
        run_all()
    elif command == "prepare":
        for seed in SEEDS:
            prepare_baseline_plan(seed)
    elif command == "baseline":
        run_baseline(args.seed)
    elif command == "arm":
        run_pair_arm(args.seed, args.arm, ARMS / arm_name(args.seed, args.arm))
    elif command == "replay":
        run_pair_arm(SEED, ARM_VOL, REPLAY / f"FT-VOL-S42-run{args.run}", max_steps=100)
    elif command == "analyze":
        technical = json.loads((REPLAY / "comparison.json").read_text(encoding="utf-8"))
        analyze_results(technical)


if __name__ == "__main__":
    main()
