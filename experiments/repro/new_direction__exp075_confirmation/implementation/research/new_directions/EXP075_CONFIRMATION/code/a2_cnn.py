from __future__ import annotations
import math, time, numpy as np, torch
from torch import nn
import frozen_pipeline as A1

SEED = 42; MAX_EPOCHS = 10; PATIENCE = 2; BATCH = 2048
WEEK_BINS = math.ceil(365/7)

class ResidualCNN(nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        width = 32
        self.stem = nn.Conv1d(A1.NCH, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(4, width), nn.GELU(),
                          nn.Conv1d(width, width, 3, padding=d, dilation=d),
                          nn.GroupNorm(4, width), nn.GELU(),
                          nn.Conv1d(width, width, 1))
            for d in (1, 2, 4)])
        self.context = nn.Sequential(nn.Linear(context_dim, 48), nn.GELU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(width*3 + 48, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, 1))
    def forward(self, seq, context):
        h = self.stem(seq.transpose(1, 2))
        for b in self.blocks: h = h + b(h)
        pooled = torch.cat([h[:,:,-1], h.mean(dim=2), h.amax(dim=2), self.context(context)], dim=1)
        return self.head(pooled).squeeze(1)

def batches(indices, shuffle, rng):
    order = indices.copy()
    if shuffle: rng.shuffle(order)
    for s in range(0, len(order), BATCH): yield order[s:s+BATCH]

def evaluate(model, Xs, Xc, y, idxs, device):
    model.eval(); tot = 0.0; cnt = 0; rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for idx in batches(idxs, False, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            t = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            tot += float(torch.sum((model(s, c) - t)**2).cpu()); cnt += len(idx)
    return tot/cnt

def train_model(Xs, Xc, y, tr, va, device, log):
    torch.manual_seed(SEED)
    model = ResidualCNN(Xc.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    rng = np.random.default_rng(SEED)
    best = float("inf"); best_epoch = 0; stale = 0; curve = []
    for epoch in range(1, MAX_EPOCHS+1):
        model.train(); ts = 0.0; tn = 0; t0 = time.time()
        for idx in batches(tr, True, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            t = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(s, c) - t)**2)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            ts += float(loss.detach().cpu())*len(idx); tn += len(idx)
        vm = evaluate(model, Xs, Xc, y, va, device)
        curve.append({"epoch": epoch, "train_mse": ts/tn, "internal_valid_mse": vm,
                      "seconds": time.time()-t0})
        log("A2 epoch", curve[-1])
        if vm < best - 1e-5: best = vm; best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= PATIENCE: break
    return best_epoch, curve

def train_full_epochs(Xs, Xc, y, epochs, device, log):
    torch.manual_seed(SEED+1)
    model = ResidualCNN(Xc.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    rng = np.random.default_rng(SEED+1); all_idx = np.arange(len(y), dtype=np.int64)
    for epoch in range(epochs):
        model.train()
        for idx in batches(all_idx, True, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            t = torch.from_numpy(y[idx].astype(np.float32)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(s, c) - t)**2)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        log("A2 full epoch", epoch+1, "/", epochs)
    return model

def predict(model, Xs, Xc, device):
    model.eval(); out = np.empty(len(Xs), dtype=np.float64); rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for idx in batches(np.arange(len(Xs)), False, rng):
            s = torch.from_numpy(Xs[idx].astype(np.float32)).to(device)
            c = torch.from_numpy(Xc[idx].astype(np.float32)).to(device)
            out[idx] = model(s, c).detach().cpu().numpy()
    return out
