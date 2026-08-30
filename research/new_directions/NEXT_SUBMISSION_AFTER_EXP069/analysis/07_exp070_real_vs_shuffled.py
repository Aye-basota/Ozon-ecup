"""Phase 3b - independent EXP070 REAL vs SHUFFLED verification from _fold_*.npz caches."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pandas as pd
REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean"); sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
ND = REPO/"research"/"new_directions"; OUT = ND/"NEXT_SUBMISSION_AFTER_EXP069"
E70 = ND/"EXP070_COUNT_VALUE_MOE"
CF = ["2025-09-04","2025-09-18","2025-10-16"]; W3 = np.array([1.,2.,8.])

ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); uid = ao["user_id"].to_numpy()
y = ao["target"].to_numpy().astype(float)
z037 = np.log1p(ao["pred_exp037"].to_numpy().astype(float))
zdist = np.log1p(ao["pred_dist"].to_numpy().astype(float))
fm = pd.read_csv(E70/"fold_metrics.csv")
def rep(c,f):
    r = fm[(fm.candidate==c)&(fm.fold==f)]; return float(r.rmsle_cal.iloc[0]) if len(r) else float('nan')

res = {"per_fold": [], "artifact_note": "z_shuffled recovered from _fold_*.npz caches"}
scores = {}
for f in CF:
    z = np.load(E70/("_fold_"+f.replace("-","")+".npz"), allow_pickle=False)
    m = fold == f
    cu = uid[m]; fu = z["user_id"]
    o = np.argsort(fu); pos = np.searchsorted(fu[o], cu)
    assert np.array_equal(fu[o][pos], cu)
    zr = z["z_real"].astype(float)[o][pos]; zs = z["z_shuffled"].astype(float)[o][pos]
    tgt_ok = float(np.max(np.abs(z["target"].astype(float)[o][pos] - y[m])))
    b = z037[m]; d = zdist[m]
    cands = {
      "COUNT_REAL": zr, "COUNT_SHUFFLED": zs,
      "REPLACE_REAL_BETA1": b + 0.25*(zr-d), "REPLACE_SHUFFLED_BETA1": b + 0.25*(zs-d),
      "ADD10_REAL": 0.90*b + 0.10*zr, "ADD10_SHUFFLED": 0.90*b + 0.10*zs,
      "EXP037": b}
    row = {"fold": f, "target_max_err": tgt_ok}
    for k, v in cands.items():
        s = calibrate_log_offset(y[m], v)[1]
        scores.setdefault(k, {})[f] = s
        row[k] = s; row[k+"_reported"] = rep(k, f); row[k+"_diff"] = s - rep(k, f)
    res["per_fold"].append(row)
    print(f"fold {f} target_err={tgt_ok}")
    for k in cands:
        print(f"   {k:24s} rec {row[k]:.12f} rep {row[k+'_reported']:.12f} diff {row[k+'_diff']:+.3e}")

print("\nREAL - SHUFFLED per fold:")
rms_rows = []
for path, rk, sk in (("standalone","COUNT_REAL","COUNT_SHUFFLED"),
                     ("replacement_beta1","REPLACE_REAL_BETA1","REPLACE_SHUFFLED_BETA1"),
                     ("add10","ADD10_REAL","ADD10_SHUFFLED")):
    for f in CF:
        g = scores[rk][f]-scores[sk][f]
        rms_rows.append(dict(path=path, fold=f, real_minus_shuffled=g))
        print(f"   {path:20s} {f} {g:+.9f}")
    a = weighted_cv([scores[rk][f] for f in CF], W3)
    s_ = weighted_cv([scores[sk][f] for f in CF], W3)
    print(f"   {path:20s} 1:2:8 REAL-SHUFFLED {a-s_:+.9f}")
    rms_rows.append(dict(path=path, fold="1:2:8", real_minus_shuffled=float(a-s_)))
res["real_minus_shuffled"] = rms_rows
b128 = weighted_cv([scores["EXP037"][f] for f in CF], W3)
res["deltas_128"] = {k: float(weighted_cv([scores[k][f] for f in CF], W3) - b128) for k in scores}
print("\n1:2:8 deltas vs EXP037:", json.dumps({k: round(v,9) for k,v in res["deltas_128"].items()}, indent=1))
(OUT/"_p3b_exp070_shuffled.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("WROTE _p3b_exp070_shuffled.json")
