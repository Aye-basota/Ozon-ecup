"""Independently rebuild the EXP075 TEST correction from the frozen final models."""
from __future__ import annotations
import datetime as dt, json, math, os, time
import numpy as np, pandas as pd, lightgbm as lgb, torch
import frozen_pipeline as A1
import a2_cnn as A2

UP = "/mnt/user-data/uploads/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
WORK = "/home/claude/work"
TEST_CUTOFF = dt.date(2026, 2, 13)
COEF = np.array([0.7462560853, 0.6466415685])

def main():
    log = A1.log
    data = A1.CleanData()
    sample = pd.read_csv(A1.SAMPLE)
    ids = sample.user_id.to_numpy(np.int64)
    rows = data.rows(ids)
    log("test rows", len(rows))
    Xc = data.context_features(rows, TEST_CUTOFF)
    log("context done")
    nw = math.ceil(365/7); dim = (nw+28)*A1.NCH + A1.CTX_DIM
    Xa = np.lib.format.open_memmap(f"{WORK}/Xa_test.npy", mode="w+", dtype=np.float32, shape=(len(rows), dim))
    data.candidate_into(Xa, 0, rows, TEST_CUTOFF, 365, Xc)
    Xa.flush(); log("A1 test matrix done")
    m1 = lgb.Booster(model_file=f"{UP}/final_A1_TREE_TRAJ_365.txt")
    u1 = m1.predict(Xa)
    del Xa, m1; os.remove(f"{WORK}/Xa_test.npy"); log("A1 predicted")
    ck = torch.load(f"{UP}/final_A2_WEEKLY_RESIDUAL_CNN.pt", map_location="cpu", weights_only=False)
    Xs = data.weekly_only(rows, TEST_CUTOFF, 365)
    Xs16 = (Xs / ck["channel_rms"]).astype(np.float16); del Xs
    Xc16 = ((Xc - ck["context_mean"]) / ck["context_std"]).astype(np.float16)
    torch.set_num_threads(2)
    model = A2.ResidualCNN(Xc16.shape[1])
    model.load_state_dict(ck["state_dict"])
    u2 = A2.predict(model, Xs16, Xc16, torch.device("cpu"))
    log("A2 predicted")
    D = COEF[0]*u1 + COEF[1]*u2
    np.save(f"{WORK}/TEST_u1_raw.npy", u1); np.save(f"{WORK}/TEST_u2_raw.npy", u2)
    np.save(f"{WORK}/TEST_D_raw.npy", D); np.save(f"{WORK}/TEST_ids.npy", ids)
    ref = np.load(f"{UP}/JOINT_A1_365_A2_TEST_raw_correction.npy").astype(np.float64)
    a1ref = np.load(f"{UP}/A1_TREE_TRAJ_365_TEST_raw_correction.npy").astype(np.float64)
    a2ref = np.load(f"{UP}/A2_WEEKLY_RESIDUAL_CNN_TEST_raw_correction.npy").astype(np.float64)
    out = {"n": int(len(ids)),
      "rms_D_rebuilt": float(np.sqrt(np.mean(D*D))),
      "rms_D_stored": float(np.sqrt(np.mean(ref*ref))),
      "max_abs_diff_joint_vs_stored_fp32": float(np.max(np.abs(D-ref))),
      "corr_joint_vs_stored": A1.correlation(D, ref),
      "A1_rebuilt_vs_stored_max_abs": float(np.max(np.abs(1.012306043162683*u1-a1ref))),
      "A1_corr": A1.correlation(u1, a1ref),
      "A2_rebuilt_vs_stored_max_abs": float(np.max(np.abs(0.9642014960450844*u2-a2ref))),
      "A2_corr": A1.correlation(u2, a2ref),
      "corr_u1_u2_test": A1.correlation(u1, u2)}
    json.dump(out, open(f"{WORK}/test_correction_rebuild.json","w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
