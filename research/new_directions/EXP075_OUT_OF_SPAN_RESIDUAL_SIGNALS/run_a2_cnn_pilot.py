from __future__ import annotations

import gc
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


EXP = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp075_a1", EXP / "run_a1_clean_forward.py")
A1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(A1)

SEED = 42
VAL = A1.FOLDS[-1]
MAX_EPOCHS = 10
PATIENCE = 2
BATCH = 2048
WEEK_BINS = math.ceil(365 / 7)
WEEK_DIM = WEEK_BINS * len(A1.RAW_CHANNELS)
T0 = time.time()


def log(*x: object) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *x, flush=True)


def hash64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64)
    x = x ^ (x >> np.uint64(30))
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x *= np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x


def build_weekly(data: A1.CleanData, frames: list[pd.DataFrame]) -> np.ndarray:
    total = sum(len(f) for f in frames)
    X = np.empty((total, WEEK_BINS, len(A1.RAW_CHANNELS)), dtype=np.float32)
    pos = 0
    for frame in frames:
        ids = frame.user_id.to_numpy(np.int64)
        cutoff = A1.dt.date.fromisoformat(frame.cutoff.iloc[0])
        tr = data.trajectory_features(data.rows(ids), cutoff, 365)
        n = len(frame)
        X[pos:pos + n] = tr[:, :WEEK_DIM].reshape(n, WEEK_BINS, len(A1.RAW_CHANNELS))
        pos += n
        del tr
        gc.collect()
    return X


class ResidualCNN(nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        width = 32
        self.stem = nn.Conv1d(len(A1.RAW_CHANNELS), width, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(4, width),
                nn.GELU(),
                nn.Conv1d(width, width, 3, padding=d, dilation=d),
                nn.GroupNorm(4, width),
                nn.GELU(),
                nn.Conv1d(width, width, 1),
            )
            for d in (1, 2, 4)
        ])
        self.context = nn.Sequential(nn.Linear(context_dim, 48), nn.GELU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(width * 3 + 48, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.stem(seq.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        pooled = torch.cat([h[:, :, -1], h.mean(dim=2), h.amax(dim=2), self.context(context)], dim=1)
        return self.head(pooled).squeeze(1)


def batches(indices: np.ndarray, shuffle: bool, rng: np.random.Generator):
    order = indices.copy()
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), BATCH):
        yield order[start:start + BATCH]


def evaluate(model, Xs, Xc, y, indices, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for idx in batches(indices, False, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            target = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            pred = model(s, c)
            total += float(torch.sum((pred - target) ** 2).cpu())
            count += len(idx)
    return total / count


def train_model(Xs, Xc, y, train_idx, valid_idx, device) -> tuple[dict, int, list[dict]]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = ResidualCNN(Xc.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    rng = np.random.default_rng(SEED)
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    curve = []
    first_steps_start = None
    first_steps = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for idx in batches(train_idx, True, rng):
            if first_steps_start is None:
                first_steps_start = time.time()
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            target = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(s, c)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(idx)
            train_n += len(idx)
            first_steps += 1
            if first_steps == 50:
                elapsed = time.time() - first_steps_start
                projected = elapsed / 50 * math.ceil(len(train_idx) / BATCH) * MAX_EPOCHS
                log("timing", {"seconds_50_steps": elapsed, "projected_max_seconds": projected})
                if projected > 6 * 3600:
                    raise RuntimeError("Projected GPU runtime exceeds six-hour hard limit")
        val_mse = evaluate(model, Xs, Xc, y, valid_idx, device)
        row = {"epoch": epoch, "train_mse": train_sum / train_n, "internal_valid_mse": val_mse}
        curve.append(row)
        log("epoch", row)
        if val_mse < best - 1e-5:
            best = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    return best_state, best_epoch, curve


def train_full_epochs(Xs, Xc, y, epochs, device) -> nn.Module:
    torch.manual_seed(SEED + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED + 1)
    model = ResidualCNN(Xc.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    rng = np.random.default_rng(SEED + 1)
    all_idx = np.arange(len(y), dtype=np.int64)
    for epoch in range(epochs):
        model.train()
        for idx in batches(all_idx, True, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            target = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(s, c) - target) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        log("full epoch", epoch + 1, "/", epochs)
    return model


def predict(model, Xs, Xc, device) -> np.ndarray:
    model.eval()
    out = np.empty(len(Xs), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for idx in batches(np.arange(len(Xs)), False, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            out[idx] = model(s, c).detach().cpu().numpy()
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A2 GPU pilot requires CUDA; CPU fallback would invalidate timing policy")
    log("device", torch.cuda.get_device_name(0))
    data = A1.CleanData()
    train_cutoffs = [VAL - A1.dt.timedelta(days=lag) for lag in A1.TRAIN_LAGS]
    frames = [data.raw_cutoff_frame(x) for x in train_cutoffs]
    val = data.raw_cutoff_frame(VAL).copy()

    log("baseline cross-fit")
    Xc, y, uid, _ = A1.concat_context(data, frames)
    halves = A1.stable_half(uid)
    base_cf = np.empty(len(y), dtype=float)
    for side in (0, 1):
        fit, pred_idx = halves != side, halves == side
        model = A1.train_lgb(Xc[fit], y[fit], "baseline", 260)
        base_cf[pred_idx] = model.predict(Xc[pred_idx])
        del model
    offset = float(np.mean(y - base_cf))
    residual = y - (base_cf + offset)
    full_base = A1.train_lgb(Xc, y, "baseline", 260)
    val_ids = val.user_id.to_numpy(np.int64)
    Xc_val = data.context_features(data.rows(val_ids), VAL)
    baseline_z = full_base.predict(Xc_val) + offset
    del full_base

    log("weekly tensors")
    Xs = build_weekly(data, frames)
    Xs_val = build_weekly(data, [val])
    channel_rms = np.sqrt(np.mean(Xs.astype(np.float64) ** 2, axis=(0, 1)))
    channel_rms = np.maximum(channel_rms, 1e-3)
    Xs = (Xs / channel_rms).astype(np.float16)
    Xs_val = (Xs_val / channel_rms).astype(np.float16)
    context_mean = Xc.mean(axis=0, dtype=np.float64)
    context_std = Xc.std(axis=0, dtype=np.float64)
    context_std = np.maximum(context_std, 1e-3)
    Xc = ((Xc - context_mean) / context_std).astype(np.float16)
    Xc_val = ((Xc_val - context_mean) / context_std).astype(np.float16)
    bucket = hash64(uid) % np.uint64(10)
    internal_valid = np.flatnonzero(bucket == 0)
    internal_train = np.flatnonzero(bucket != 0)

    _, best_epoch, curve = train_model(Xs, Xc, residual, internal_train, internal_valid, device)
    log("best epoch", best_epoch, "retraining all rows")
    final_model = train_full_epochs(Xs, Xc, residual, best_epoch, device)
    u_raw = predict(final_model, Xs_val, Xc_val, device)
    u_perp, projection = A1.project_candidate(u_raw, baseline_z)
    r = val.target_log.to_numpy(float) - baseline_z
    b = float(np.mean(u_perp * r))
    G = float(np.mean(u_perp * u_perp))
    rho = A1.correlation(u_perp, r)
    result = {
        "experiment": "A2_WEEKLY_RESIDUAL_CNN_PILOT",
        "validation_cutoff": VAL.isoformat(),
        "train_cutoffs": [x.isoformat() for x in train_cutoffs],
        "train_rows": int(len(y)),
        "validation_rows": int(len(val)),
        "device": torch.cuda.get_device_name(0),
        "best_epoch_internal": int(best_epoch),
        "curve": curve,
        "rho_latest": rho,
        "b": b,
        "G": G,
        "oracle_amplitude": b / G if G else 0.0,
        "oracle_MSE_gain": -(b * b / G) if G else 0.0,
        "u_raw_rms": float(np.sqrt(np.mean(u_raw ** 2))),
        "u_perp_rms": float(np.sqrt(G)),
        "corr_with_A1_365_latest": None,
        "projection": projection,
        "runtime_seconds": time.time() - T0,
        "verdict": "PROMOTE_TO_FULL_FOLDS" if rho >= 0.015 else "REJECT_LATEST_RHO_LT_0.015",
    }
    a1_oof = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    a1_latest = a1_oof[a1_oof.cutoff == VAL.isoformat()].sort_values("user_id")
    if np.array_equal(a1_latest.user_id.to_numpy(), val.user_id.to_numpy()):
        result["corr_with_A1_365_latest"] = A1.correlation(u_perp, a1_latest.u_perp_365.to_numpy(float))
    out = pd.DataFrame({
        "user_id": val.user_id,
        "cutoff": VAL.isoformat(),
        "target_log": val.target_log,
        "baseline_z": baseline_z,
        "residual": r,
        "u_raw_A2": u_raw,
        "u_perp_A2": u_perp,
    })
    out.to_parquet(EXP / "a2_latest_pilot_predictions.parquet", index=False)
    (EXP / "a2_pilot_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": final_model.state_dict(), "channel_rms": channel_rms,
                "context_mean": context_mean, "context_std": context_std,
                "best_epoch": best_epoch}, EXP / "a2_latest_pilot_model.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
