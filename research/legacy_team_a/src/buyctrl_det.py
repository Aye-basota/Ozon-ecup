"""EXP-045 BUYCTRL-DET: true purchase supervision versus cutoff-wise shuffle.

The only causal contrast is BUYTRUE - BUYSHUF.  Both auxiliary arms train a
plain SEQ-01 from scratch for four epochs with the exact materialized plans
used by the deterministic EXP-044 baselines.  BUYTRUE uses ``1[y30 > 0]``;
BUYSHUF uses the same labels permuted independently inside every cutoff.

The already completed EXP-044 deterministic plain-SEQ baselines are the BASE
arms.  Their training plans/checkpoints are artifact-verified here and their
fixed fourth-epoch checkpoints are evaluated on the standard full validation
panel.  No test, blend, LOFO, leaderboard or submission path exists.

One-command execution from the repository root::

    python src/buyctrl_det.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any

# Must be set before CUDA is initialized, including in child processes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from src.config import ARTIFACTS, SEED, TARGET_DAYS
from src.features import panel_users
from src.fresh_cond_ft import (
    BASELINES as EXP044_BASELINES,
    _checkpoint_result,
    _finish_run_dir,
    _load_plan,
    _temporary_run_dir,
    baseline_name,
    baseline_plan_paths,
    build_baseline_optimizer,
    capture_rng_state,
    configure_determinism,
    data_hashes,
    direct_head_hash,
    jsonable,
    module_hash,
    optimizer_hash,
    prepare_baseline_plan,
    rng_state_hash,
    save_training_snapshot,
    sha256_array,
    sha256_file,
    state_dict_hash,
    write_json_new,
)
from src.seq import (Batcher, DEPTH_GRID, N_CH_STORED, build_index, build_model,
                     fold_cutoffs, gather, panel, user_rows)
from src.seq_cond import _auc, _pool, seg_masks, segments
from src.validation import bias_z, calibrate, rmsle_z


EXP_NUM = 45
EXP_ID = "BUYCTRL-DET"
VAL = dt.date(2025, 10, 16)
SEEDS = tuple(SEED + i for i in range(3))
OUT = ARTIFACTS / "BUYCTRL_DET_EXP045"
PLANS = OUT / "plans"
ARMS_DIR = OUT / "arms"
BASE_DIR = OUT / "base"

EPOCHS = 4
LAMBDA_AUX = 0.1
ARM_TRUE = "BUYTRUE"
ARM_SHUF = "BUYSHUF"
ARMS = (ARM_TRUE, ARM_SHUF)
SNAPSHOT_FIXED = (0, 1, 100, 1000)


def log(*parts: Any) -> None:
    print(f"[EXP-045 {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def arm_name(seed: int, arm: str) -> str:
    assert arm in ARMS
    return f"{arm}-S{seed}-V1016"


def base_name(seed: int) -> str:
    return f"BASE-S{seed}-V1016"


def label_plan_paths(seed: int) -> tuple[Path, Path]:
    return PLANS / f"buyctrl_s{seed}.npz", PLANS / f"buyctrl_s{seed}.json"


def aux_init_path(seed: int) -> Path:
    return PLANS / f"buyctrl_s{seed}_aux_init.pt"


def source_hashes() -> dict[str, str]:
    paths = {
        "buyctrl_det.py": Path(__file__),
        "fresh_cond_ft.py": ROOT / "src" / "fresh_cond_ft.py",
        "seq.py": ROOT / "src" / "seq.py",
        "seq_cond.py": ROOT / "src" / "seq_cond.py",
        "validation.py": ROOT / "src" / "validation.py",
        "config.py": ROOT / "src" / "config.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def full_validation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard three-block fold panel, in one fixed user/row order."""
    users = panel_users(VAL, 3)["user_id"].to_numpy().astype(np.int64, copy=False)
    rows = user_rows(users).astype(np.int32, copy=False)
    from src.seq import target_at
    y = target_at(VAL, rows).astype(np.float64, copy=False)
    return users, rows, y


def shuffle_within_cutoff(labels: np.ndarray, cutoff_index: np.ndarray,
                          seed: int) -> np.ndarray:
    """Permute labels independently per cutoff, preserving every prevalence."""
    labels = np.asarray(labels, np.uint8)
    cutoff_index = np.asarray(cutoff_index)
    assert labels.ndim == cutoff_index.ndim == 1 and len(labels) == len(cutoff_index)
    out = np.empty_like(labels)
    rng = np.random.default_rng([seed, EXP_NUM, 0xB0])
    for cutoff_i in np.unique(cutoff_index):
        idx = np.flatnonzero(cutoff_index == cutoff_i)
        out[idx] = labels[idx][rng.permutation(len(idx))]
        assert int(out[idx].sum()) == int(labels[idx].sum())
    assert np.array_equal(np.sort(out), np.sort(labels))
    return out


def build_aux_head(dim: int, prevalence: float) -> nn.Linear:
    assert 0.0 < prevalence < 1.0
    head = nn.Linear(dim, 1)
    with torch.no_grad():
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, math.log(prevalence / (1.0 - prevalence)))
    return head


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def prepare_aux_initialization(seed: int, prevalence: float, dim: int) -> dict[str, Any]:
    path = aux_init_path(seed)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        return dict(path=str(path.resolve()), file_sha256=sha256_file(path),
                    state_sha256=state_dict_hash(state))
    head = build_aux_head(dim, prevalence)
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(state, tmp)
    tmp.rename(path)
    return dict(path=str(path.resolve()), file_sha256=sha256_file(path),
                state_sha256=state_dict_hash(state))


def _write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> str:
    assert not path.exists(), f"refusing to overwrite existing artifact: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)
    return sha256_file(path)


def _load_label_plan(seed: int) -> tuple[Any, dict[str, Any]]:
    npz_path, json_path = label_plan_paths(seed)
    assert npz_path.exists() and json_path.exists()
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["seed"] == seed and meta["val"] == VAL.isoformat()
    assert sha256_file(npz_path) == meta["plan_file_sha256"]
    plan = np.load(npz_path, allow_pickle=False)
    for name, expected in meta["array_sha256"].items():
        assert sha256_array(plan[name]) == expected, f"label plan drift: {name}"
    current = source_hashes()
    for name, expected in meta["source_sha256"].items():
        assert current[name] == expected, f"source drift under EXP-045 plan: {name}"
    return plan, meta


def prepare_label_plan(seed: int) -> dict[str, Any]:
    assert seed in SEEDS, "seeds must be derived from src.config.SEED"
    npz_path, json_path = label_plan_paths(seed)
    if npz_path.exists() or json_path.exists():
        assert npz_path.exists() and json_path.exists(), "partial label plan"
        _, meta = _load_label_plan(seed)
        log(f"reuse label plan seed {seed}: {meta['n_steps']:,} steps")
        return meta

    base_meta = prepare_baseline_plan(seed)
    base_plan, loaded_base_meta = _load_plan(*baseline_plan_paths(seed))
    assert base_meta == loaded_base_meta
    checkpoint, baseline_result = _checkpoint_result(seed)
    cfg = dict(checkpoint["cfg"])
    assert int(cfg["epochs"]) == EPOCHS and float(cfg["depth_aug"]) == 0.0
    assert int(cfg["workers"]) == 1 and not cfg["compile"]

    cuts = fold_cutoffs(VAL)
    assert cuts[0] == dt.date(2025, 4, 3) and cuts[-1] == dt.date(2025, 9, 11)
    assert all(T + dt.timedelta(days=TARGET_DAYS) <= VAL for T in cuts)
    ci, ri, zy = build_index(cuts, blocks=1)
    assert data_hashes(cuts, ci, ri, zy) == base_meta["clean_data_sha256"]
    labels_true = (zy > 0).astype(np.uint8)
    labels_shuf = shuffle_within_cutoff(labels_true, ci, seed)
    prevalence = float(labels_true.mean())
    cutoff_prevalence = []
    for cutoff_i, cutoff in enumerate(cuts):
        mask = ci == cutoff_i
        pt = float(labels_true[mask].mean())
        ps = float(labels_shuf[mask].mean())
        assert pt == ps
        cutoff_prevalence.append(dict(cutoff=cutoff, n=int(mask.sum()), prevalence=pt))
    mismatch = float(np.mean(labels_true != labels_shuf))
    assert mismatch > 0.1, "shuffle unexpectedly retained the user-label connection"

    val_user, val_rows, val_y = full_validation()
    arrays = dict(
        labels_true=labels_true,
        labels_shuf=labels_shuf,
        val_user_id=val_user,
        val_rows=val_rows,
        val_y=val_y,
    )
    init = prepare_aux_initialization(seed, prevalence, 3 * int(cfg["hidden"]))
    array_hashes = {name: sha256_array(value) for name, value in arrays.items()}
    file_sha = _write_npz_new(npz_path, arrays)
    core = dict(
        format=1, experiment=EXP_ID, seed=seed, val=VAL,
        arms=["BASE", ARM_TRUE, ARM_SHUF], lambda_aux=LAMBDA_AUX,
        loss="MSE(z30) + 0.1 * BCEWithLogits(buy30)",
        labels_true="1[y30 > 0]",
        labels_shuf="labels_true permuted independently inside every train cutoff",
        prevalence=prevalence, cutoff_prevalence=cutoff_prevalence,
        shuffle_mismatch_fraction=mismatch,
        aux_head="Linear(3*hidden, 1), zero weight, global-prevalence logit bias",
        auxiliary_initialization=init,
        cfg=cfg, n_cutoffs=len(cuts), n_examples=len(zy),
        n_steps=int(base_meta["n_steps"]), epochs=EPOCHS,
        baseline_artifact=baseline_name(seed),
        baseline_result=str((EXP044_BASELINES / baseline_name(seed) / "result.json").resolve()),
        baseline_checkpoint=baseline_result["checkpoint_path"],
        baseline_checkpoint_sha256=baseline_result["checkpoint_file_sha256"],
        baseline_initial_model_sha256=baseline_result["initial_model_sha256"],
        baseline_initial_optimizer_sha256=baseline_result["initial_optimizer_sha256"],
        baseline_final_model_sha256=baseline_result["final_model_sha256"],
        baseline_plan_path=str(baseline_plan_paths(seed)[0].resolve()),
        baseline_plan_id=base_meta["plan_id"],
        baseline_plan_file_sha256=base_meta["plan_file_sha256"],
        epoch_plan_sha256=base_meta["epoch_plan_sha256"],
        batch_index_arrays={name: base_meta["array_sha256"][name] for name in (
            "group_chunk_offsets", "chunk_cutoff", "chunk_sample_offsets",
            "sample_indices", "batch_seed", "lr_plan", "epoch_step_offsets")},
        clean_data_sha256=base_meta["clean_data_sha256"],
        validation_order_sha256=sha256_array(val_user),
        validation_rows_sha256=sha256_array(val_rows),
        array_sha256=array_hashes,
        source_sha256=source_hashes(),
        execution_policy=dict(
            workers=1, materialized_index_batch_lr_and_label_plans=True,
            python_numpy_torch_cuda_rng="fixed per seed",
            cudnn_benchmark=False, cudnn_deterministic=True,
            deterministic_algorithms=True, cublas_workspace_config=":4096:8",
            tf32=True, bf16=True, compile=False, separate_process=True,
            endpoint="fixed end of epoch 4; no validation selection",
        ),
    )
    plan_id = hashlib.sha256(json.dumps(jsonable(core), sort_keys=True).encode()).hexdigest()
    meta = dict(core, plan_id=plan_id, plan_file_sha256=file_sha,
                created_at=dt.datetime.now().isoformat(timespec="seconds"))
    write_json_new(json_path, meta)
    log(f"label plan seed {seed}: prevalence={prevalence:.6f}, "
        f"shuffle mismatch={mismatch:.3f}, full val={len(val_y):,}")
    return meta


def environment_record() -> dict[str, Any]:
    return dict(
        python=sys.version, platform=platform.platform(), torch=torch.__version__,
        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
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


def build_joint_optimizer(model: nn.Module, auxiliary: nn.Module,
                          cfg: dict[str, Any]) -> torch.optim.Optimizer:
    named = [(f"model.{n}", p) for n, p in model.named_parameters()]
    named += [(f"auxiliary.{n}", p) for n, p in auxiliary.named_parameters()]
    decay = [p for _, p in named if p.dim() > 1]
    nodecay = [p for _, p in named if p.dim() <= 1]
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=float(cfg["wd"])),
         dict(params=nodecay, weight_decay=0.0)],
        lr=float(cfg["lr"]), betas=(0.9, 0.98),
    )


def iter_materialized_labeled_batches(batcher: Batcher, base_plan: Any,
                                      labels: np.ndarray):
    total = len(base_plan["batch_seed"])
    q: Queue = Queue(maxsize=4)
    errors: list[BaseException] = []

    def unpack(batch_i: int) -> list[tuple[int, np.ndarray]]:
        c0, c1 = base_plan["group_chunk_offsets"][batch_i:batch_i + 2]
        group = []
        for chunk_i in range(int(c0), int(c1)):
            i0, i1 = base_plan["chunk_sample_offsets"][chunk_i:chunk_i + 2]
            group.append((int(base_plan["chunk_cutoff"][chunk_i]),
                          base_plan["sample_indices"][int(i0):int(i1)]))
        return group

    def work() -> None:
        try:
            for batch_i in range(total):
                group = unpack(batch_i)
                x, y = batcher._make(group, int(base_plan["batch_seed"][batch_i]))
                selected = np.concatenate([idx for _, idx in group])
                q.put((batch_i, x, y, labels[selected]))
        except BaseException as exc:
            errors.append(exc)
            q.put((-1, None, None, None))

    thread = threading.Thread(target=work, daemon=True, name="exp045-worker-1")
    thread.start()
    for expected in range(total):
        batch_i, x, y, a = q.get()
        if errors:
            raise errors[0]
        assert batch_i == expected, f"materialized order changed: {batch_i} != {expected}"
        yield x, y, a
    thread.join()
    if errors:
        raise errors[0]


def forward_outputs(model: nn.Module, auxiliary: nn.Module,
                    x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = _pool(model.encode(x))
    z = model.head(pooled).squeeze(1)
    logits = auxiliary(pooled).squeeze(1)
    return z, logits


def error_decomposition(y: np.ndarray, z: np.ndarray,
                        offset: float) -> dict[str, float | int]:
    target = np.log1p(y)
    error2 = (target - (z.astype(np.float64) + offset)) ** 2
    positive = y > 0
    zero = ~positive
    return dict(
        n_zero=int(zero.sum()), n_positive=int(positive.sum()),
        zero_rmse=float(np.sqrt(error2[zero].mean())),
        positive_rmse=float(np.sqrt(error2[positive].mean())),
        zero_mse_contribution=float(error2[zero].sum() / len(y)),
        positive_mse_contribution=float(error2[positive].sum() / len(y)),
    )


def prediction_record(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    offset, calibrated = calibrate(y, z)
    record = dict(
        n=len(z), dtype=str(z.dtype), rmsle_raw=float(rmsle_z(y, z)),
        rmsle_cal=float(calibrated), offset=float(offset), bias=float(bias_z(y, z)),
        activity_auc=float(_auc(y > 0, z)), mean_z=float(z.mean()),
        std_z=float(z.std()), sha256=sha256_array(z),
    )
    record["error_decomposition"] = error_decomposition(y, z, float(offset))
    return record


def auxiliary_record(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    label = (y > 0).astype(np.float64)
    score = logits.astype(np.float64)
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))
    return dict(
        bce=float(np.mean(np.logaddexp(0.0, score) - label * score)),
        auc=float(_auc(label, score)), prevalence=float(label.mean()),
        mean_probability=float(prob.mean()), mean_logit=float(score.mean()),
        sha256=sha256_array(logits),
    )


def predict_outputs(model: nn.Module, auxiliary: nn.Module, cfg: dict[str, Any],
                    rows: np.ndarray, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    auxiliary.eval()
    scale = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    z = np.empty(len(rows), np.float32)
    logits = np.empty(len(rows), np.float32)
    batch = int(cfg["batch"])
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            x = gather(VAL, rows[start:start + batch])
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= scale
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                zo, ao = forward_outputs(model, auxiliary, t)
            z[start:start + len(x)] = zo.float().cpu().numpy()
            logits[start:start + len(x)] = ao.float().cpu().numpy()
    model.train()
    auxiliary.train()
    return z, logits


def _fix_result_paths(result: dict[str, Any], final_dir: Path) -> None:
    result["prediction"]["file"] = str((final_dir / "z_raw.npy").resolve())
    result["auxiliary_prediction"]["file"] = str((final_dir / "aux_logits.npy").resolve())
    for snap in result["snapshots"]:
        snap["path"] = str((final_dir / Path(snap["path"]).name).resolve())


def run_arm(seed: int, arm: str) -> dict[str, Any]:
    assert seed in SEEDS and arm in ARMS
    destination = ARMS_DIR / arm_name(seed, arm)
    assert not destination.exists(), f"refusing to overwrite completed run: {destination}"
    label_plan, meta = _load_label_plan(seed)
    base_plan, base_meta = _load_plan(*baseline_plan_paths(seed))
    assert base_meta["plan_id"] == meta["baseline_plan_id"]
    assert base_meta["plan_file_sha256"] == meta["baseline_plan_file_sha256"]
    checkpoint, baseline_result = _checkpoint_result(seed)
    cfg = dict(meta["cfg"])
    cfg["depth_grid"] = tuple(cfg["depth_grid"])
    assert checkpoint["cfg"] == cfg
    env = configure_determinism(seed)
    assert torch.cuda.is_available(), "EXP-045 must exercise CUDA bf16"
    dev = torch.device("cuda")

    cuts = fold_cutoffs(VAL)
    ci, ri, zy = build_index(cuts, blocks=1)
    assert data_hashes(cuts, ci, ri, zy) == meta["clean_data_sha256"]
    labels = label_plan["labels_true"] if arm == ARM_TRUE else label_plan["labels_shuf"]
    assert len(labels) == len(zy)
    batcher = Batcher(
        cuts, ci, ri, zy, int(cfg["batch"]), int(cfg["chunk"]),
        np.random.default_rng(seed), workers=1,
        aug=dict(mode="none", p=0.0, full=0.0), aug_seed=[seed, 0xA7A1],
        depth=dict(p=0.0, grid=DEPTH_GRID),
    )

    work = _temporary_run_dir(ARMS_DIR, arm_name(seed, arm))
    started = time.time()
    model = build_model(cfg).to(dev)
    assert module_hash(model) == baseline_result["initial_model_sha256"]
    post_model_rng = capture_rng_state()
    auxiliary = build_aux_head(3 * int(cfg["hidden"]), float(meta["prevalence"])).to(dev)
    aux_state = torch.load(aux_init_path(seed), map_location="cpu", weights_only=False)
    assert state_dict_hash(aux_state) == meta["auxiliary_initialization"]["state_sha256"]
    auxiliary.load_state_dict(aux_state)
    # Auxiliary construction must not perturb the dropout stream relative to BASE.
    restore_rng_state(post_model_rng)
    assert rng_state_hash() == baseline_result["snapshots"][0]["rng_sha256"]
    base_opt = build_baseline_optimizer(model, cfg)
    assert optimizer_hash(model, None, base_opt) == baseline_result["initial_optimizer_sha256"]
    del base_opt
    opt = build_joint_optimizer(model, auxiliary, cfg)
    model.train()
    auxiliary.train()

    initial = dict(
        model_sha256=module_hash(model), direct_head_sha256=direct_head_hash(model),
        auxiliary_head_sha256=module_hash(auxiliary),
        optimizer_sha256=optimizer_hash(model, auxiliary, opt),
        rng_sha256=rng_state_hash(),
        baseline_model_optimizer_sha256=baseline_result["initial_optimizer_sha256"],
    )
    snapshots = [save_training_snapshot(work, 0, model, auxiliary, opt)]
    requested = set(int(x) for x in base_plan["snapshot_steps"])
    requested.update(int(x) for x in base_plan["epoch_step_offsets"])
    scale = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    trainable = list(model.parameters()) + list(auxiliary.parameters())
    epoch_offsets = set(int(x) for x in base_plan["epoch_step_offsets"][1:])
    epoch_direct: list[float] = []
    epoch_aux: list[float] = []
    run_direct = run_aux = None
    seen = 0

    for step0, (x, y_direct, a_label) in enumerate(
            iter_materialized_labeled_batches(batcher, base_plan, labels)):
        lr = float(base_plan["lr_plan"][step0])
        for group in opt.param_groups:
            group["lr"] = lr
        n = len(y_direct)
        t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
        t[:, :N_CH_STORED] *= scale
        target = torch.from_numpy(y_direct).to(dev)
        aux_target = torch.from_numpy(a_label.astype(np.float32, copy=False)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            z, logits = forward_outputs(model, auxiliary, t)
            direct_loss = torch.nn.functional.mse_loss(z, target)
            aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), aux_target)
            loss = direct_loss + LAMBDA_AUX * aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        run_direct = (direct_loss.detach() * n if run_direct is None
                      else run_direct + direct_loss.detach() * n)
        run_aux = (aux_loss.detach() * n if run_aux is None
                   else run_aux + aux_loss.detach() * n)
        seen += n
        completed = step0 + 1
        if completed in requested:
            snapshots.append(save_training_snapshot(work, completed, model, auxiliary, opt))
        if completed in epoch_offsets:
            epoch_direct.append(float(run_direct) / seen)
            epoch_aux.append(float(run_aux) / seen)
            log(f"{arm_name(seed, arm)} epoch {len(epoch_direct)}/4 complete")
            run_direct = run_aux = None
            seen = 0

    assert len(epoch_direct) == EPOCHS
    assert int(base_meta["n_steps"]) == int(base_plan["epoch_step_offsets"][-1])
    z, logits = predict_outputs(model, auxiliary, cfg, label_plan["val_rows"], dev)
    z_path = work / "z_raw.npy"
    aux_path = work / "aux_logits.npy"
    np.save(z_path, z)
    np.save(aux_path, logits)
    pred = prediction_record(label_plan["val_y"], z)
    pred.update(file=str(z_path.resolve()), file_sha256=sha256_file(z_path),
                inference_uses_auxiliary_head=False)
    aux_pred = auxiliary_record(label_plan["val_y"], logits)
    aux_pred.update(file=str(aux_path.resolve()), file_sha256=sha256_file(aux_path),
                    evaluated_against="true y30>0 on the validation fold")
    result = dict(
        experiment=EXP_ID, kind="auxiliary_arm", name=arm_name(seed, arm),
        seed=seed, arm=arm, val=VAL, lambda_aux=LAMBDA_AUX, cfg=cfg,
        endpoint_epoch=EPOCHS, endpoint_selected_by_validation=False,
        label_semantics=("true 1[y30>0]" if arm == ARM_TRUE
                         else "cutoff-wise shuffled 1[y30>0]"),
        label_plan_id=meta["plan_id"], label_plan_sha256=meta["plan_file_sha256"],
        baseline_plan_id=base_meta["plan_id"],
        baseline_plan_sha256=base_meta["plan_file_sha256"],
        batch_index_arrays=meta["batch_index_arrays"],
        validation_order_sha256=meta["validation_order_sha256"],
        initial=initial,
        final=dict(model_sha256=module_hash(model),
                   direct_head_sha256=direct_head_hash(model),
                   auxiliary_head_sha256=module_hash(auxiliary),
                   optimizer_sha256=optimizer_hash(model, auxiliary, opt),
                   rng_sha256=rng_state_hash()),
        snapshots=snapshots, n_steps=int(base_meta["n_steps"]),
        n_examples_per_epoch=len(zy), epoch_direct_mse=epoch_direct,
        epoch_auxiliary_bce=epoch_aux, prediction=pred,
        auxiliary_prediction=aux_pred,
        auxiliary_head_used_for_final_prediction=False,
        environment=env, source_sha256=source_hashes(),
        duration_seconds=time.time() - started,
    )
    (work / "result.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    _finish_run_dir(work, destination)
    result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
    _fix_result_paths(result, destination)
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log(f"{arm_name(seed, arm)} complete: RMSLE_cal={pred['rmsle_cal']:.9f}")
    return result


def run_base(seed: int) -> dict[str, Any]:
    """Evaluate the exact EXP-044 fourth-epoch BASE checkpoint on full fold."""
    destination = BASE_DIR / base_name(seed)
    assert not destination.exists(), f"refusing to overwrite completed base: {destination}"
    label_plan, meta = _load_label_plan(seed)
    checkpoint, old = _checkpoint_result(seed)
    assert old["n_steps"] == meta["n_steps"] and len(old["epoch_train_mse"]) == EPOCHS
    env = configure_determinism(seed)
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    cfg = dict(checkpoint["cfg"])
    cfg["depth_grid"] = tuple(cfg["depth_grid"])
    model = build_model(cfg).to(dev)
    model.load_state_dict(checkpoint["state"])
    assert module_hash(model) == old["final_model_sha256"]
    started = time.time()
    # Direct inference path only: no auxiliary head exists or is consulted.
    from src.seq import predict
    z = predict(model, VAL, label_plan["val_rows"], cfg, dev).astype(np.float32, copy=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = _temporary_run_dir(BASE_DIR, base_name(seed))
    z_path = work / "z_raw.npy"
    np.save(z_path, z)
    pred = prediction_record(label_plan["val_y"], z)
    pred.update(file=str(z_path.resolve()), file_sha256=sha256_file(z_path),
                inference_uses_auxiliary_head=False)
    result = dict(
        experiment=EXP_ID, kind="reused_deterministic_base", name=base_name(seed),
        seed=seed, arm="BASE", val=VAL, cfg=cfg, endpoint_epoch=EPOCHS,
        endpoint_selected_by_validation=False, training_reused_from="EXP-044",
        training_artifact=baseline_name(seed),
        training_result=str((EXP044_BASELINES / baseline_name(seed) / "result.json").resolve()),
        training_result_sha256=sha256_file(
            EXP044_BASELINES / baseline_name(seed) / "result.json"),
        checkpoint_path=old["checkpoint_path"],
        checkpoint_sha256=old["checkpoint_file_sha256"],
        initial_model_sha256=old["initial_model_sha256"],
        initial_optimizer_sha256=old["initial_optimizer_sha256"],
        final_model_sha256=old["final_model_sha256"],
        final_optimizer_sha256=old["final_optimizer_sha256"],
        epoch_direct_mse=old["epoch_train_mse"], n_steps=old["n_steps"],
        baseline_plan_id=meta["baseline_plan_id"],
        baseline_plan_sha256=meta["baseline_plan_file_sha256"],
        batch_index_arrays=meta["batch_index_arrays"],
        validation_order_sha256=meta["validation_order_sha256"],
        prediction=pred, auxiliary_prediction=None,
        auxiliary_head_used_for_final_prediction=False,
        environment=env, duration_seconds=time.time() - started,
    )
    (work / "result.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    _finish_run_dir(work, destination)
    result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
    result["prediction"]["file"] = str((destination / "z_raw.npy").resolve())
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log(f"{base_name(seed)} complete: RMSLE_cal={pred['rmsle_cal']:.9f}")
    return result


def child_env(seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _launch(args: list[str], seed: int) -> None:
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args],
                   cwd=ROOT, env=child_env(seed), check=True)


def launch_base(seed: int) -> None:
    path = BASE_DIR / base_name(seed) / "result.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["baseline_plan_id"] == _load_label_plan(seed)[1]["baseline_plan_id"]
        log(f"reuse completed {base_name(seed)}")
        return
    assert not path.parent.exists(), "partial BASE artifact; refusing overwrite"
    _launch(["base", "--seed", str(seed)], seed)


def launch_arm(seed: int, arm: str) -> None:
    path = ARMS_DIR / arm_name(seed, arm) / "result.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["label_plan_id"] == _load_label_plan(seed)[1]["plan_id"]
        log(f"reuse completed {arm_name(seed, arm)}")
        return
    assert not path.parent.exists(), "partial arm artifact; refusing overwrite"
    _launch(["arm", "--seed", str(seed), "--arm", arm], seed)


def load_result(seed: int, arm: str) -> dict[str, Any]:
    path = ((BASE_DIR / base_name(seed)) if arm == "BASE"
            else (ARMS_DIR / arm_name(seed, arm))) / "result.json"
    assert path.exists(), f"missing completed result: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_global_offset_error(y: np.ndarray, z: np.ndarray, offset: float,
                              mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y[mask]) - (z[mask] + offset)) ** 2)))


def pair_diagnostics(left: np.ndarray, right: np.ndarray,
                     left_offset: float, right_offset: float) -> dict[str, float]:
    delta = left.astype(np.float64) - right.astype(np.float64)
    return dict(
        var_delta_z=float(np.var(delta)), mean_delta_z=float(delta.mean()),
        max_abs_delta_z=float(np.max(np.abs(delta))),
        correlation=float(np.corrcoef(left, right)[0, 1]),
        calibration_offset_difference=float(left_offset - right_offset),
    )


def paired_stats(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                sd=float(a.std(ddof=1)), negative_seeds=int((a < 0).sum()),
                values=a.tolist())


def decision_rule(primary: float, primary_negative: int,
                  secondary: float) -> tuple[str, list[str]]:
    reasons = []
    if primary <= -0.0007 and primary_negative >= 2 and secondary <= -0.0003:
        return "PASS", ["primary reached -0.0007 with >=2/3 signs and secondary was meaningful"]
    if primary > -0.0003:
        return "FAIL", ["mean BUYTRUE-BUYSHUF is above the preregistered -0.0003 FAIL boundary"]
    if primary <= -0.0007 and primary_negative < 2:
        reasons.append("mean reached the PASS scale but fewer than 2/3 seeds had the right sign")
    if primary <= -0.0007 and secondary > -0.0003:
        reasons.append("primary reached the PASS scale but BUYTRUE-BASE was not meaningful")
    if -0.0007 < primary <= -0.0003:
        reasons.append("primary lies in the preregistered inconclusive interval")
    return "INCONCLUSIVE", reasons


def analyze() -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    pooled_primary = []
    plan_records = []
    for seed in SEEDS:
        base = load_result(seed, "BASE")
        true = load_result(seed, ARM_TRUE)
        shuf = load_result(seed, ARM_SHUF)
        assert true["initial"] == shuf["initial"]
        for key in ("label_plan_id", "label_plan_sha256", "baseline_plan_id",
                    "baseline_plan_sha256", "batch_index_arrays",
                    "validation_order_sha256", "n_steps", "cfg"):
            assert true[key] == shuf[key], f"unpaired {key} for seed {seed}"
        assert base["baseline_plan_id"] == true["baseline_plan_id"]
        assert base["initial_model_sha256"] == true["initial"]["model_sha256"]
        assert base["initial_optimizer_sha256"] == true["initial"][
            "baseline_model_optimizer_sha256"]
        plan, meta = _load_label_plan(seed)
        y = plan["val_y"]
        rows = plan["val_rows"]
        zb = np.load(base["prediction"]["file"])
        zt = np.load(true["prediction"]["file"])
        zs = np.load(shuf["prediction"]["file"])
        assert len(y) == len(zb) == len(zt) == len(zs)
        assert sha256_array(zb) == base["prediction"]["sha256"]
        assert sha256_array(zt) == true["prediction"]["sha256"]
        assert sha256_array(zs) == shuf["prediction"]["sha256"]
        dt_s = pair_diagnostics(zt, zs, true["prediction"]["offset"],
                                shuf["prediction"]["offset"])
        dt_b = pair_diagnostics(zt, zb, true["prediction"]["offset"],
                                base["prediction"]["offset"])
        pooled_primary.append(zt.astype(np.float64) - zs.astype(np.float64))
        seed_rows.append(dict(
            seed=seed,
            rmsle_cal_base=base["prediction"]["rmsle_cal"],
            rmsle_cal_buytrue=true["prediction"]["rmsle_cal"],
            rmsle_cal_buyshuf=shuf["prediction"]["rmsle_cal"],
            buytrue_minus_buyshuf=(true["prediction"]["rmsle_cal"]
                                   - shuf["prediction"]["rmsle_cal"]),
            buytrue_minus_base=(true["prediction"]["rmsle_cal"]
                                - base["prediction"]["rmsle_cal"]),
            activity_auc_base=base["prediction"]["activity_auc"],
            activity_auc_buytrue=true["prediction"]["activity_auc"],
            activity_auc_buyshuf=shuf["prediction"]["activity_auc"],
            auxiliary_bce_buytrue=true["auxiliary_prediction"]["bce"],
            auxiliary_bce_buyshuf=shuf["auxiliary_prediction"]["bce"],
            auxiliary_auc_buytrue=true["auxiliary_prediction"]["auc"],
            auxiliary_auc_buyshuf=shuf["auxiliary_prediction"]["auc"],
            true_shuf_var_delta_z=dt_s["var_delta_z"],
            true_shuf_correlation=dt_s["correlation"],
            true_shuf_mean_delta_z=dt_s["mean_delta_z"],
            true_shuf_offset_difference=dt_s["calibration_offset_difference"],
            true_base_var_delta_z=dt_b["var_delta_z"],
            true_base_correlation=dt_b["correlation"],
            true_base_mean_delta_z=dt_b["mean_delta_z"],
            true_base_offset_difference=dt_b["calibration_offset_difference"],
            base_prediction_sha256=base["prediction"]["sha256"],
            buytrue_prediction_sha256=true["prediction"]["sha256"],
            buyshuf_prediction_sha256=shuf["prediction"]["sha256"],
        ))
        masks = seg_masks(segments(VAL, rows))
        for name, mask in masks.items():
            if int(mask.sum()) < 100:
                continue
            eb = apply_global_offset_error(y, zb, base["prediction"]["offset"], mask)
            et = apply_global_offset_error(y, zt, true["prediction"]["offset"], mask)
            es = apply_global_offset_error(y, zs, shuf["prediction"]["offset"], mask)
            segment_rows.append(dict(
                seed=seed, segment=name, n=int(mask.sum()), base_error=eb,
                buytrue_error=et, buyshuf_error=es,
                buytrue_minus_buyshuf=et - es, buytrue_minus_base=et - eb,
            ))
        for arm, result in (("BASE", base), (ARM_TRUE, true), (ARM_SHUF, shuf)):
            zero_rows.append(dict(seed=seed, arm=arm,
                                  **result["prediction"]["error_decomposition"]))
        plan_records.append(dict(
            seed=seed, plan_id=meta["plan_id"], plan_file_sha256=meta["plan_file_sha256"],
            baseline_plan_id=meta["baseline_plan_id"],
            baseline_plan_file_sha256=meta["baseline_plan_file_sha256"],
            auxiliary_initialization_sha256=meta["auxiliary_initialization"]["state_sha256"],
            labels_true_sha256=meta["array_sha256"]["labels_true"],
            labels_shuf_sha256=meta["array_sha256"]["labels_shuf"],
            shuffle_mismatch_fraction=meta["shuffle_mismatch_fraction"],
        ))

    primary_values = [r["buytrue_minus_buyshuf"] for r in seed_rows]
    secondary_values = [r["buytrue_minus_base"] for r in seed_rows]
    primary = paired_stats(primary_values)
    secondary = paired_stats(secondary_values)
    decision, reasons = decision_rule(primary["mean"], primary["negative_seeds"],
                                      secondary["mean"])
    mean_aux_auc_gain = float(np.mean([
        r["auxiliary_auc_buytrue"] - r["auxiliary_auc_buyshuf"] for r in seed_rows]))
    mean_aux_bce_gain = float(np.mean([
        r["auxiliary_bce_buytrue"] - r["auxiliary_bce_buyshuf"] for r in seed_rows]))
    if mean_aux_auc_gain > 0.01 and primary["mean"] <= -0.0007:
        mechanism = "auxiliary task learns and direct RMSLE improves: potential shared signal"
    elif mean_aux_auc_gain > 0.01 and primary["mean"] > -0.0003:
        mechanism = ("auxiliary task learns, but BUYTRUE approximately matches BUYSHUF on "
                     "direct RMSLE: negative result for auxiliary supervision")
    else:
        mechanism = "mechanism diagnostics are mixed; rely on the preregistered RMSLE endpoint"
    pooled = np.concatenate(pooled_primary)
    summary = dict(
        experiment=EXP_ID, fold=VAL, seeds=SEEDS,
        exact_config=dict(
            model="plain SEQ-01", epochs=EPOCHS, lambda_aux=LAMBDA_AUX,
            main_loss="MSE(log1p(y30))", auxiliary_loss="BCEWithLogits(1[y30>0])",
            auxiliary_head="Linear(192,1), training only",
            shuffle="within each cutoff, exact prevalence preserved",
            endpoint="end epoch 4, no validation selection",
            validation="standard full three-block panel",
            depth_aug=0.0, d3a=False, fresh=False,
        ),
        per_seed=seed_rows, primary=primary, secondary=secondary,
        mean_auxiliary_auc_true_minus_shuf=mean_aux_auc_gain,
        mean_auxiliary_bce_true_minus_shuf=mean_aux_bce_gain,
        pooled_primary_var_delta_z=float(np.var(pooled)),
        pooled_primary_mean_delta_z=float(pooled.mean()),
        decision=decision, decision_reasons=reasons, mechanism_interpretation=mechanism,
        promote_other_folds=False,
        plans=plan_records,
        scope=dict(
            proven=("causal contrast BUYTRUE-BUYSHUF on deterministic plain SEQ-01, "
                    "fold 2025-10-16, seeds 42/43/44"),
            not_proven=["other folds", "historical SEQ-AVG3", "STRONGEST_CURRENT",
                        "test/LB transfer", "other lambda or epoch choices"],
            leaderboard_submission_created=False,
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    analysis_path = OUT / "analysis.json"
    if analysis_path.exists():
        stored = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert stored == jsonable(summary), "existing analysis differs"
    else:
        write_json_new(analysis_path, summary)
    for path, rows in ((OUT / "seed_summary.csv", seed_rows),
                       (OUT / "segment_summary.csv", segment_rows),
                       (OUT / "zero_positive_summary.csv", zero_rows)):
        if path.exists():
            continue
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    log(f"decision={decision}; mean BUYTRUE-BUYSHUF={primary['mean']:+.9f}")
    return summary


def run_all() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        prepare_label_plan(seed)
    for seed in SEEDS:
        launch_base(seed)
        launch_arm(seed, ARM_TRUE)
        launch_arm(seed, ARM_SHUF)
    return analyze()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("all")
    sub.add_parser("prepare")
    b = sub.add_parser("base")
    b.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a = sub.add_parser("arm")
    a.add_argument("--seed", type=int, required=True, choices=SEEDS)
    a.add_argument("--arm", required=True, choices=ARMS)
    sub.add_parser("analyze")
    args = parser.parse_args()
    command = args.command or "all"
    if command == "all":
        run_all()
    elif command == "prepare":
        for seed in SEEDS:
            prepare_label_plan(seed)
    elif command == "base":
        run_base(args.seed)
    elif command == "arm":
        run_arm(args.seed, args.arm)
    elif command == "analyze":
        analyze()


if __name__ == "__main__":
    main()
