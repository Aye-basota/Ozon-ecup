"""Run one frozen-pipeline fold (A1-365 + A2) at an arbitrary validation cutoff."""
from __future__ import annotations
import datetime as dt, gc, json, math, os, sys, time
import numpy as np, pandas as pd, torch
import frozen_pipeline as A1
import a2_cnn as A2

WORK = "/home/claude/work"; OUTDIR = f"{WORK}/folds"
COEF = np.array([0.7462560853, 0.6466415685])

def main(cutoff_str: str):
    cutoff = dt.date.fromisoformat(cutoff_str)
    log = A1.log
    os.makedirs(OUTDIR, exist_ok=True)
    data = A1.CleanData()
    train_cutoffs = [cutoff - dt.timedelta(days=l) for l in A1.TRAIN_LAGS]
    assert all(t + dt.timedelta(days=30) <= cutoff for t in train_cutoffs)
    log("FOLD", cutoff, "train cutoffs", [t.isoformat() for t in train_cutoffs])
    frames = [data.raw_cutoff_frame(t) for t in train_cutoffs]
    val = data.raw_cutoff_frame(cutoff).copy()

    ntr = sum(len(f) for f in frames)
    Xb = np.lib.format.open_memmap(f"{WORK}/Xb.npy", mode="w+", dtype=np.float32, shape=(ntr, A1.CTX_DIM))
    y = np.empty(ntr); uid = np.empty(ntr, dtype=np.int64); pos = 0
    for f in frames:
        ids = f["user_id"].to_numpy(np.int64); n = len(f)
        data.context_features(data.rows(ids), dt.date.fromisoformat(f["cutoff"].iloc[0]), out=Xb[pos:pos+n])
        y[pos:pos+n] = f["target_log"].to_numpy(float); uid[pos:pos+n] = ids; pos += n
    Xb.flush(); log("context built", Xb.shape)

    halves = A1.stable_half(uid)
    base_cf = np.empty(ntr)
    for side in (0, 1):
        fit = np.flatnonzero(halves != side); pr = np.flatnonzero(halves == side)
        m = A1.train_lgb(np.ascontiguousarray(Xb[fit]), y[fit], "baseline", 260)
        base_cf[pr] = m.predict(np.ascontiguousarray(Xb[pr]))
        del m; gc.collect(); log("baseline half", side, "done")
    base_offset = float(np.mean(y - base_cf)); base_cf += base_offset
    residual = y - base_cf
    full = A1.train_lgb(Xb, y, "baseline", 260)
    val_ids = val["user_id"].to_numpy(np.int64); val_rows = data.rows(val_ids)
    Xb_val = data.context_features(val_rows, cutoff)
    baseline_z = full.predict(Xb_val) + base_offset
    del full; gc.collect(); log("baseline full done")
    target_log = val["target_log"].to_numpy(float)
    val["baseline_z"] = baseline_z
    val["residual"] = target_log - baseline_z
    r = val["residual"].to_numpy(float)
    meta = {"validation_cutoff": cutoff.isoformat(),
            "train_cutoffs": [t.isoformat() for t in train_cutoffs],
            "train_rows": int(ntr), "validation_rows": int(len(val)),
            "baseline_crossfit_offset": base_offset,
            "baseline_rmsle": float(np.sqrt(np.mean(r*r))),
            "feature_source_max_date": cutoff.isoformat(),
            "target_source_min_date": (cutoff+dt.timedelta(days=1)).isoformat(),
            "target_source_max_date": (cutoff+dt.timedelta(days=30)).isoformat(),
            "train_target_max_date": (max(train_cutoffs)+dt.timedelta(days=30)).isoformat(),
            "validation_inactive_fraction": float(np.mean(val.target_events.to_numpy()==0))}

    # ---- A1-365 ----
    nw = math.ceil(365/7); dim = (nw+28)*A1.NCH + A1.CTX_DIM
    Xa = np.lib.format.open_memmap(f"{WORK}/Xa.npy", mode="w+", dtype=np.float32, shape=(ntr, dim))
    pos = 0
    for f in frames:
        ids = f["user_id"].to_numpy(np.int64); n = len(f)
        data.candidate_into(Xa, pos, data.rows(ids), dt.date.fromisoformat(f["cutoff"].iloc[0]), 365, Xb[pos:pos+n])
        pos += n
    Xa.flush(); log("A1 train matrix built", Xa.shape)
    m1 = A1.train_lgb(Xa, residual, "candidate", 300)
    del Xa; gc.collect(); os.remove(f"{WORK}/Xa.npy")
    Xa_val = np.empty((len(val_rows), dim), dtype=np.float32)
    data.candidate_into(Xa_val, 0, val_rows, cutoff, 365, Xb_val)
    u1_raw = m1.predict(Xa_val)
    m1.save_model(f"{OUTDIR}/A1_365_{cutoff.isoformat()}.txt")
    del Xa_val, m1; gc.collect(); log("A1 predicted")
    u1, proj1 = A1.project_candidate(u1_raw, baseline_z)
    meta["A1_TREE_TRAJ_365"] = {"rho": A1.correlation(u1, r), "b": float(np.mean(u1*r)),
        "G": float(np.mean(u1*u1)), "u_raw_rms": float(np.sqrt(np.mean(u1_raw**2))),
        "u_perp_rms": float(np.sqrt(np.mean(u1*u1))), "projection": proj1}
    log("A1 rho", meta["A1_TREE_TRAJ_365"]["rho"])

    # ---- A2 ----
    Xs = np.lib.format.open_memmap(f"{WORK}/Xs.npy", mode="w+", dtype=np.float32, shape=(ntr, A2.WEEK_BINS, A1.NCH))
    pos = 0
    for f in frames:
        ids = f["user_id"].to_numpy(np.int64); n = len(f)
        data.weekly_only(data.rows(ids), dt.date.fromisoformat(f["cutoff"].iloc[0]), 365, out=Xs[pos:pos+n])
        pos += n
    Xs.flush(); log("A2 weekly built")
    Xs_val = data.weekly_only(val_rows, cutoff, 365)
    ss = np.zeros(A1.NCH, dtype=np.float64); cnt = 0
    for s in range(0, ntr, 100_000):
        e = min(s+100_000, ntr); blk = np.asarray(Xs[s:e], dtype=np.float64)
        ss += np.sum(blk**2, axis=(0,1)); cnt += blk.shape[0]*blk.shape[1]
    channel_rms = np.maximum(np.sqrt(ss/cnt), 1e-3)
    Xs16 = np.lib.format.open_memmap(f"{WORK}/Xs16.npy", mode="w+", dtype=np.float16, shape=Xs.shape)
    for s in range(0, ntr, 100_000):
        e = min(s+100_000, ntr); Xs16[s:e] = (np.asarray(Xs[s:e])/channel_rms).astype(np.float16)
    Xs16.flush(); del Xs; gc.collect(); os.remove(f"{WORK}/Xs.npy")
    Xs_val16 = (Xs_val/channel_rms).astype(np.float16); del Xs_val
    cm = np.empty(A1.CTX_DIM); cs = np.empty(A1.CTX_DIM)
    blk = np.asarray(Xb, dtype=np.float64)
    cm = blk.mean(axis=0); cs = np.maximum(blk.std(axis=0), 1e-3); del blk; gc.collect()
    Xc16 = np.lib.format.open_memmap(f"{WORK}/Xc16.npy", mode="w+", dtype=np.float16, shape=(ntr, A1.CTX_DIM))
    for s in range(0, ntr, 100_000):
        e = min(s+100_000, ntr); Xc16[s:e] = ((np.asarray(Xb[s:e], dtype=np.float64)-cm)/cs).astype(np.float16)
    Xc16.flush()
    Xc_val16 = ((Xb_val - cm)/cs).astype(np.float16)
    bucket = A1.hash64(uid) % np.uint64(10)
    itr = np.flatnonzero(bucket != 0); iva = np.flatnonzero(bucket == 0)
    device = torch.device("cpu"); torch.set_num_threads(2)
    best_epoch, curve = A2.train_model(Xs16, Xc16, residual, itr, iva, device, log)
    log("A2 best epoch", best_epoch)
    fm = A2.train_full_epochs(Xs16, Xc16, residual, best_epoch, device, log)
    u2_raw = A2.predict(fm, Xs_val16, Xc_val16, device)
    torch.save({"state_dict": fm.state_dict(), "channel_rms": channel_rms,
                "context_mean": cm, "context_std": cs, "best_epoch": best_epoch},
               f"{OUTDIR}/A2_{cutoff.isoformat()}.pt")
    del fm; gc.collect()
    u2, proj2 = A1.project_candidate(u2_raw, baseline_z)
    meta["A2_WEEKLY_RESIDUAL_CNN"] = {"rho": A1.correlation(u2, r), "b": float(np.mean(u2*r)),
        "G": float(np.mean(u2*u2)), "u_raw_rms": float(np.sqrt(np.mean(u2_raw**2))),
        "u_perp_rms": float(np.sqrt(np.mean(u2*u2))), "best_epoch": int(best_epoch),
        "curve": curve, "projection": proj2}
    log("A2 rho", meta["A2_WEEKLY_RESIDUAL_CNN"]["rho"])

    D = COEF[0]*u1 + COEF[1]*u2
    base = math.sqrt(np.mean(r*r)); cor = math.sqrt(np.mean((r-D)**2))
    meta["JOINT_FROZEN"] = {"coefficients": COEF.tolist(), "rho": A1.correlation(D, r),
        "delta_MSE": float(np.mean((r-D)**2 - r**2)), "delta_RMSLE": cor-base,
        "baseline_RMSLE": base, "rms_D": float(np.sqrt(np.mean(D*D))),
        "corr_A1_A2": A1.correlation(u1, u2)}
    log("JOINT rho", meta["JOINT_FROZEN"]["rho"], "dMSE", meta["JOINT_FROZEN"]["delta_MSE"])

    out = pd.DataFrame({"user_id": val_ids, "cutoff": cutoff.isoformat(),
        "target_y30": val.target_y30.to_numpy(), "target_events": val.target_events.to_numpy(),
        "target_log": target_log, "baseline_z": baseline_z, "residual": r,
        "u_raw_365": u1_raw, "u_perp_365": u1, "u_raw_A2": u2_raw, "u_perp_A2": u2})
    out.to_parquet(f"{OUTDIR}/fold_{cutoff.isoformat()}.parquet", index=False)
    meta["runtime_seconds"] = time.time() - A1.T0
    json.dump(meta, open(f"{OUTDIR}/meta_{cutoff.isoformat()}.json", "w"), indent=2, default=float)
    for p in ("Xb.npy", "Xs16.npy", "Xc16.npy"):
        try: os.remove(f"{WORK}/{p}")
        except OSError: pass
    print(json.dumps({k: v for k, v in meta.items() if k != "A2_WEEKLY_RESIDUAL_CNN"}, indent=2, default=float))

if __name__ == "__main__":
    main(sys.argv[1])
