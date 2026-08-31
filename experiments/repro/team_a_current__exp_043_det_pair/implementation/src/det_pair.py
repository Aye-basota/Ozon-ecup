"""DET-PAIR — deterministic continuation of the production-compatible SEQ checkpoint.

This is deliberately not FT-FRESH-ENC.  The script loads the confirmed
``SEQ-D3A`` fold checkpoint, creates a fresh production AdamW optimizer, and
continues the existing main MSE objective on the existing clean fold data.  It
changes execution policy only: a single materialized batch/index plan,
``workers=1`` and deterministic CUDA settings.  Two repeats run in separate
processes and their raw validation logits are compared.

One-command run from the repository root::

    python src/det_pair.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any

# Must be present before the first CUDA context is created.  Child repeat
# processes receive the same value explicitly from ``run_all`` as well.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import ARTIFACTS, SEED
from src.features import panel_users
from src.seq import (Batcher, N_CH_STORED, aug_spec, build_index, build_model, depth_spec,
                     fold_cutoffs, panel, predict, target_at, user_rows)
from src.validation import calibrate, rmsle_z


EXP_ID = "DET-PAIR"
VAL = dt.date(2025, 10, 16)
STARTING_CKPT = "SEQ-D3A-S42-V1016"
OUT = ARTIFACTS / "DET_PAIR"
PLAN_NPZ = OUT / "materialized_plan.npz"
PLAN_JSON = OUT / "materialized_plan.json"
CONTINUATION_EPOCHS = 1
SNAPSHOT_STEPS = (0, 1, 100, 1000)

THRESHOLDS = dict(
    abs_delta_rmsle=1e-4,
    var_delta_z=1e-5,
    max_abs_delta_z=1e-3,
    corr=0.999999,
)


def log(*parts: Any) -> None:
    print(f"[DET-PAIR {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    """SHA256 of the contiguous array value bytes (the reported prediction hash)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _hash_tensor(h: Any, name: str, value: torch.Tensor) -> None:
    a = value.detach().cpu().contiguous().numpy()
    h.update(name.encode("utf-8"))
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes())


def model_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _hash_tensor(h, name, value)
    return h.hexdigest()


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        _hash_tensor(h, name, value)
    return h.hexdigest()


def optimizer_hash(model: torch.nn.Module, opt: torch.optim.Optimizer) -> str:
    """Canonical optimizer hash independent of process-local parameter ids."""
    names = {id(p): n for n, p in model.named_parameters()}
    h = hashlib.sha256()
    for gi, group in enumerate(opt.param_groups):
        spec = {k: v for k, v in group.items() if k != "params"}
        h.update(json.dumps(spec, sort_keys=True, default=str).encode("utf-8"))
        for p in group["params"]:
            name = names[id(p)]
            h.update(f"group={gi}:{name}".encode("utf-8"))
            for key, value in sorted(opt.state.get(p, {}).items()):
                if torch.is_tensor(value):
                    _hash_tensor(h, f"{name}:{key}", value)
                else:
                    h.update(f"{name}:{key}:{value!r}".encode("utf-8"))
    return h.hexdigest()


def rng_hash() -> str:
    h = hashlib.sha256()
    h.update(pickle.dumps(random.getstate(), protocol=5))
    h.update(pickle.dumps(np.random.get_state(), protocol=5))
    h.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def source_hash(path: Path) -> str:
    return sha256_file(path)


def pack_plan(groups: list[list[tuple[int, np.ndarray]]]) -> dict[str, np.ndarray]:
    """Pack Batcher groups without pickle; every row index is stored exactly once."""
    group_chunk_offsets = [0]
    chunk_cutoff: list[int] = []
    chunk_sample_offsets = [0]
    samples: list[np.ndarray] = []
    for group in groups:
        for cutoff_i, idx in group:
            idx = np.asarray(idx)
            assert idx.ndim == 1 and (idx >= 0).all() and idx.max(initial=0) < 2 ** 31
            chunk_cutoff.append(int(cutoff_i))
            samples.append(idx.astype(np.int32, copy=False))
            chunk_sample_offsets.append(chunk_sample_offsets[-1] + len(idx))
        group_chunk_offsets.append(len(chunk_cutoff))
    return dict(
        group_chunk_offsets=np.asarray(group_chunk_offsets, np.int32),
        chunk_cutoff=np.asarray(chunk_cutoff, np.int16),
        chunk_sample_offsets=np.asarray(chunk_sample_offsets, np.int64),
        sample_indices=(np.concatenate(samples) if samples else np.empty(0, np.int32)),
    )


def unpack_group(plan: Any, batch_i: int) -> list[tuple[int, np.ndarray]]:
    c0, c1 = plan["group_chunk_offsets"][batch_i:batch_i + 2]
    out = []
    for c in range(int(c0), int(c1)):
        i0, i1 = plan["chunk_sample_offsets"][c:c + 2]
        out.append((int(plan["chunk_cutoff"][c]),
                    plan["sample_indices"][int(i0):int(i1)]))
    return out


def learning_rate_plan(cfg: dict[str, Any], total: int) -> np.ndarray:
    out = np.empty(total, np.float64)
    for step in range(total):
        out[step] = cfg["lr"] * (
            min(1.0, (step + 1) / cfg["warmup"])
            * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
        )
    return out


def continuation_cfg(checkpoint_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(checkpoint_cfg)
    assert cfg["seed"] == SEED, "starting checkpoint seed must come from src.config"
    cfg.update(epochs=CONTINUATION_EPOCHS, workers=1, compile=False)
    return cfg


def prepare_plan(force: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS / f"model_{STARTING_CKPT}.pt"
    assert ckpt_path.exists(), f"missing starting checkpoint: {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert dt.date.fromisoformat(checkpoint["val"]) == VAL
    cfg = continuation_cfg(checkpoint["cfg"])
    expected_sources = {
        "det_pair.py": source_hash(Path(__file__)),
        "seq.py": source_hash(ROOT / "src" / "seq.py"),
        "validation.py": source_hash(ROOT / "src" / "validation.py"),
        "config.py": source_hash(ROOT / "src" / "config.py"),
    }
    ckpt_sha = sha256_file(ckpt_path)
    ckpt_state_sha = state_dict_hash(checkpoint["state"])

    if PLAN_NPZ.exists() and PLAN_JSON.exists() and not force:
        meta = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
        assert meta["checkpoint_file_sha256"] == ckpt_sha
        assert meta["checkpoint_state_sha256"] == ckpt_state_sha
        # The runner itself may receive reporting-only fixes after a long plan is
        # materialized.  Training semantics are guarded by the three imported
        # project sources; their hashes must never drift under an existing plan.
        for name in ("seq.py", "validation.py", "config.py"):
            assert meta["source_sha256"][name] == expected_sources[name]
        assert meta["seed"] == SEED and meta["val"] == VAL.isoformat()
        assert sha256_file(PLAN_NPZ) == meta["plan_file_sha256"]
        log(f"reuse materialized plan {PLAN_NPZ} ({meta['n_steps']:,} steps)")
        return meta

    cuts = fold_cutoffs(VAL)
    ci, ri, zy = build_index(cuts, blocks=1)
    batcher = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"],
                      np.random.default_rng(SEED), workers=1,
                      aug=aug_spec(cfg), aug_seed=[SEED, 0xA7A1],
                      depth=depth_spec(cfg))
    groups = batcher._plan()
    packed = pack_plan(groups)
    batch_seed = batcher.arng.integers(0, 2 ** 62, size=len(groups), dtype=np.int64)
    n_steps = len(groups)
    lr_plan = learning_rate_plan(cfg, n_steps)

    uv = panel_users(VAL, 3)["user_id"].to_numpy().astype(np.int64, copy=False)
    rv = user_rows(uv)
    yv = target_at(VAL, rv)
    snapshots = sorted(set(SNAPSHOT_STEPS) | {n_steps})
    snapshots = [s for s in snapshots if 0 <= s <= n_steps]

    arrays: dict[str, np.ndarray] = dict(
        cuts=np.asarray([d.isoformat() for d in cuts], dtype="U10"),
        ci=ci.astype(np.int16, copy=False),
        ri=ri.astype(np.int32, copy=False),
        zy=zy.astype(np.float32, copy=False),
        batch_seed=batch_seed,
        lr_plan=lr_plan,
        val_user_id=uv,
        val_rows=rv.astype(np.int32, copy=False),
        val_y=yv.astype(np.float64, copy=False),
        snapshot_steps=np.asarray(snapshots, np.int64),
        **packed,
    )
    array_hashes = {name: sha256_array(value) for name, value in arrays.items()}
    np.savez(PLAN_NPZ, **arrays)
    meta_core = dict(
        format=1,
        experiment=EXP_ID,
        val=VAL.isoformat(),
        seed=SEED,
        starting_artifact=str(ckpt_path.resolve()),
        checkpoint_file_sha256=ckpt_sha,
        checkpoint_state_sha256=ckpt_state_sha,
        checkpoint_cfg=checkpoint["cfg"],
        continuation_cfg=cfg,
        n_cutoffs=len(cuts),
        n_examples=len(zy),
        n_steps=n_steps,
        batch=cfg["batch"],
        chunk=cfg["chunk"],
        workers=1,
        snapshots=snapshots,
        array_sha256=array_hashes,
        source_sha256=expected_sources,
        execution_policy=dict(
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            deterministic_algorithms=True,
            cublas_workspace_config=":4096:8",
            amp="torch.autocast(cuda, bfloat16)",
            tf32=True,
            compile=False,
            pythonhashseed=SEED,
            validation_order="materialized val_rows",
        ),
        optimizer=dict(name="AdamW", betas=[0.9, 0.98], lr=cfg["lr"], wd=cfg["wd"],
                       state="fresh/empty in each repeat"),
        scheduler="production cosine formula reset at step 0; lr_plan materialized once",
        objective="existing MSE(z30); no FT-FRESH conditional loss or EXTRA data",
    )
    plan_id = hashlib.sha256(json.dumps(meta_core, sort_keys=True, default=str).encode()).hexdigest()
    meta = dict(meta_core, plan_id=plan_id, plan_file_sha256=sha256_file(PLAN_NPZ),
                created_at=dt.datetime.now().isoformat(timespec="seconds"))
    PLAN_JSON.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log(f"materialized {n_steps:,} steps / {len(zy):,} examples once -> {PLAN_NPZ}")
    log(f"plan_id={plan_id}, file_sha256={meta['plan_file_sha256']}")
    return meta


def configure_determinism() -> dict[str, Any]:
    assert os.environ.get("PYTHONHASHSEED") == str(SEED), (
        "repeat process must start with PYTHONHASHSEED from src.config")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    return dict(
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_debug_mode=str(torch.get_deterministic_debug_mode()),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        pythonhashseed=os.environ["PYTHONHASHSEED"],
    )


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    decay = [p for _, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for _, p in model.named_parameters() if p.dim() <= 1]
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=cfg["wd"]),
         dict(params=nodecay, weight_decay=0.0)],
        lr=cfg["lr"], betas=(0.9, 0.98),
    )


def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return value


def save_snapshot(run_dir: Path, step: int, model: torch.nn.Module,
                  opt: torch.optim.Optimizer) -> dict[str, Any]:
    torch.cuda.synchronize()
    entry = dict(step=step, model_sha256=model_hash(model),
                 optimizer_sha256=optimizer_hash(model, opt), rng_sha256=rng_hash())
    path = run_dir / f"snapshot_step_{step:05d}.pt"
    torch.save(dict(step=step, model=_cpu_copy(model.state_dict()),
                    optimizer=_cpu_copy(opt.state_dict())), path)
    entry["path"] = str(path.resolve())
    entry["file_sha256"] = sha256_file(path)
    return entry


def iter_materialized_batches(batcher: Batcher, plan: Any):
    """One producer thread (workers=1), preserving the materialized order."""
    n = len(plan["batch_seed"])
    q: Queue = Queue(maxsize=4)
    error: list[BaseException] = []

    def work() -> None:
        try:
            for batch_i in range(n):
                group = unpack_group(plan, batch_i)
                value = batcher._make(group, int(plan["batch_seed"][batch_i]))
                q.put((batch_i, value))
        except BaseException as exc:  # propagate worker failures to the training process
            error.append(exc)
            q.put((-1, None))

    thread = threading.Thread(target=work, daemon=True, name="det-pair-worker-1")
    thread.start()
    for expected in range(n):
        batch_i, value = q.get()
        if error:
            raise error[0]
        assert batch_i == expected, f"batch order changed: {batch_i} != {expected}"
        yield value
    thread.join()
    if error:
        raise error[0]


def prediction_record(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    offset, rmsle_cal = calibrate(y, z)
    return dict(
        n=len(z),
        dtype=str(z.dtype),
        rmsle=float(rmsle_z(y, z)),
        rmsle_cal=float(rmsle_cal),
        calibration_offset=float(offset),
        mean_z=float(z.mean()),
        std_z=float(z.std()),
        sha256=sha256_array(z),
    )


def run_repeat(run_id: int) -> dict[str, Any]:
    assert run_id in (1, 2, 3)
    assert PLAN_NPZ.exists() and PLAN_JSON.exists(), "prepare the plan first"
    meta = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    assert sha256_file(PLAN_NPZ) == meta["plan_file_sha256"]
    run_dir = OUT / f"run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = configure_determinism()
    assert torch.cuda.is_available(), "DET-PAIR must exercise the CUDA production path"
    dev = torch.device("cuda")
    start_time = time.time()
    plan_data = np.load(PLAN_NPZ, allow_pickle=False)
    for name, expected in meta["array_sha256"].items():
        assert sha256_array(plan_data[name]) == expected, f"plan array changed: {name}"

    cfg = dict(meta["continuation_cfg"])
    cfg["depth_grid"] = tuple(cfg["depth_grid"])
    ckpt_path = Path(meta["starting_artifact"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert state_dict_hash(checkpoint["state"]) == meta["checkpoint_state_sha256"]
    model = build_model(cfg).to(dev)
    model.load_state_dict(checkpoint["state"])
    model.train()
    opt = build_optimizer(model, cfg)
    start_model_hash = model_hash(model)
    start_optimizer_hash = optimizer_hash(model, opt)
    assert start_model_hash == meta["checkpoint_state_sha256"]

    cuts = [dt.date.fromisoformat(str(x)) for x in plan_data["cuts"]]
    batcher = Batcher(cuts, plan_data["ci"], plan_data["ri"], plan_data["zy"],
                      cfg["batch"], cfg["chunk"], np.random.default_rng(SEED), workers=1,
                      aug=aug_spec(cfg), aug_seed=[SEED, 0xA7A1], depth=depth_spec(cfg))
    scale = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    snapshot_steps = set(int(x) for x in plan_data["snapshot_steps"])
    snapshots = []
    if 0 in snapshot_steps:
        snapshots.append(save_snapshot(run_dir, 0, model, opt))

    run_loss = None
    seen = 0
    n_steps = int(meta["n_steps"])
    log(f"run{run_id}: start {n_steps:,} optimizer steps, workers=1, fresh AdamW")
    for step0, (x, yb) in enumerate(iter_materialized_batches(batcher, plan_data)):
        lr = float(plan_data["lr_plan"][step0])
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
        seen += n
        completed = step0 + 1
        if completed in snapshot_steps:
            snapshots.append(save_snapshot(run_dir, completed, model, opt))
        if completed % 250 == 0 or completed == n_steps:
            log(f"run{run_id}: step {completed:,}/{n_steps:,}, lr={lr:.8g}")

    assert completed == n_steps and seen == len(plan_data["zy"])
    rv = plan_data["val_rows"]
    yv = plan_data["val_y"]
    z = predict(model, VAL, rv, cfg, dev).astype(np.float32, copy=False)
    z_path = run_dir / "z_raw.npy"
    np.save(z_path, z)
    record = prediction_record(yv, z)
    record["file"] = str(z_path.resolve())
    record["file_sha256"] = sha256_file(z_path)
    result = dict(
        experiment=EXP_ID,
        run=run_id,
        plan_id=meta["plan_id"],
        plan_file_sha256=meta["plan_file_sha256"],
        starting_artifact=meta["starting_artifact"],
        checkpoint_file_sha256=meta["checkpoint_file_sha256"],
        start_model_sha256=start_model_hash,
        start_optimizer_sha256=start_optimizer_hash,
        steps=n_steps,
        examples=seen,
        train_mse=float(run_loss) / seen,
        snapshots=snapshots,
        prediction=record,
        environment=env,
        duration_seconds=time.time() - start_time,
    )
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log(f"run{run_id}: RMSLE_cal={record['rmsle_cal']:.9f}, z_sha256={record['sha256']}")
    return result


def compare_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    za = np.load(a["prediction"]["file"])
    zb = np.load(b["prediction"]["file"])
    assert za.dtype == zb.dtype == np.float32 and za.shape == zb.shape
    delta = za.astype(np.float64) - zb.astype(np.float64)
    same = bool(np.array_equal(za, zb))
    corr = 1.0 if same else float(np.corrcoef(za, zb)[0, 1])
    snapshots = []
    for sa, sb in zip(a["snapshots"], b["snapshots"], strict=True):
        assert sa["step"] == sb["step"]
        snapshots.append(dict(
            step=sa["step"],
            model_equal=sa["model_sha256"] == sb["model_sha256"],
            optimizer_equal=sa["optimizer_sha256"] == sb["optimizer_sha256"],
            rng_equal=sa["rng_sha256"] == sb["rng_sha256"],
            model_sha256=[sa["model_sha256"], sb["model_sha256"]],
            optimizer_sha256=[sa["optimizer_sha256"], sb["optimizer_sha256"]],
        ))
    out = dict(
        runs=[a["run"], b["run"]],
        rmsle_cal=[a["prediction"]["rmsle_cal"], b["prediction"]["rmsle_cal"]],
        rmsle_raw=[a["prediction"]["rmsle"], b["prediction"]["rmsle"]],
        abs_delta_rmsle=abs(a["prediction"]["rmsle_cal"] - b["prediction"]["rmsle_cal"]),
        var_delta_z=float(np.var(delta)),
        max_abs_delta_z=float(np.max(np.abs(delta))),
        mean_delta_z=float(delta.mean()),
        corr=corr,
        prediction_sha256=[a["prediction"]["sha256"], b["prediction"]["sha256"]],
        prediction_hashes_equal=a["prediction"]["sha256"] == b["prediction"]["sha256"],
        arrays_equal=same,
        snapshots=snapshots,
    )
    out["within_tolerance"] = bool(
        out["abs_delta_rmsle"] <= THRESHOLDS["abs_delta_rmsle"]
        and out["var_delta_z"] <= THRESHOLDS["var_delta_z"]
        and out["max_abs_delta_z"] <= THRESHOLDS["max_abs_delta_z"]
        and out["corr"] >= THRESHOLDS["corr"]
    )
    return out


def load_result(run_id: int) -> dict[str, Any]:
    path = OUT / f"run{run_id}" / "result.json"
    assert path.exists(), f"missing run result: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_results(run_ids: list[int]) -> dict[str, Any]:
    runs = [load_result(i) for i in run_ids]
    assert len({r["plan_id"] for r in runs}) == 1
    pairs = [compare_pair(runs[i], runs[j])
             for i in range(len(runs)) for j in range(i + 1, len(runs))]
    exact = all(p["arrays_equal"] for p in pairs)
    passed = exact or all(p["within_tolerance"] for p in pairs)
    result = dict(
        experiment=EXP_ID,
        thresholds=THRESHOLDS,
        runs=run_ids,
        pairs=pairs,
        exact_predictions=exact,
        pass_det_pair=passed,
        third_repeat_required=(len(run_ids) == 2 and not exact and pairs[0]["within_tolerance"]),
        reliable_neural_effect_rmsle=(1e-4 if passed else None),
        interpretation=(
            "bitwise prediction floor is zero; declare <=1e-4 RMSLE as conservative execution floor"
            if exact else
            "numerical floor is the maximum pairwise discrepancy; effects must exceed it"
        ),
    )
    path = OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    p = pairs[0]
    log(f"compare {p['runs']}: exact={p['arrays_equal']} |delta RMSLE|={p['abs_delta_rmsle']:.3g} "
        f"Var(delta z)={p['var_delta_z']:.3g} max|delta z|={p['max_abs_delta_z']:.3g} "
        f"corr={p['corr']:.9f}")
    return result


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(SEED)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def launch_repeat(run_id: int, force: bool = False) -> None:
    result = OUT / f"run{run_id}" / "result.json"
    if result.exists() and not force:
        r = json.loads(result.read_text(encoding="utf-8"))
        meta = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
        if r.get("plan_id") == meta["plan_id"]:
            log(f"reuse complete run{run_id}: {result}")
            return
    cmd = [sys.executable, str(Path(__file__).resolve()), "repeat", "--run-id", str(run_id)]
    subprocess.run(cmd, cwd=ROOT, env=child_env(), check=True)


def run_all(force_plan: bool = False, force_runs: bool = False) -> dict[str, Any]:
    prepare_plan(force=force_plan)
    launch_repeat(1, force=force_runs)
    launch_repeat(2, force=force_runs)
    comparison = compare_results([1, 2])
    if comparison["third_repeat_required"]:
        log("hashes differ within the requested tolerance; launching mandatory third repeat")
        launch_repeat(3, force=force_runs)
        comparison = compare_results([1, 2, 3])
    if not comparison["pass_det_pair"]:
        log("FAIL: deterministic floor is above the gate; FT-FRESH remains blocked")
    else:
        log("PASS: DET-PAIR complete; FT-FRESH was not started")
    return comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="DET-PAIR deterministic SEQ continuation")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("all")
    a.add_argument("--force-plan", action="store_true")
    a.add_argument("--force-runs", action="store_true")
    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("repeat")
    r.add_argument("--run-id", required=True, type=int, choices=(1, 2, 3))
    c = sub.add_parser("compare")
    c.add_argument("--runs", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    cmd = args.cmd or "all"
    if cmd == "all":
        run_all(getattr(args, "force_plan", False), getattr(args, "force_runs", False))
    elif cmd == "prepare":
        prepare_plan(args.force)
    elif cmd == "repeat":
        run_repeat(args.run_id)
    elif cmd == "compare":
        compare_results(args.runs)


if __name__ == "__main__":
    main()
