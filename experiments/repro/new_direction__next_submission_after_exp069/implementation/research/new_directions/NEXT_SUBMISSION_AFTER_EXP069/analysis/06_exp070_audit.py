"""Phase 3 - independent audit of EXP070 (reconstruction from the single saved
raw count-value MoE OOF vector)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pandas as pd

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
ND = REPO/"research"/"new_directions"; OUT = ND/"NEXT_SUBMISSION_AFTER_EXP069"
E70 = ND/"EXP070_COUNT_VALUE_MOE"
CF = ["2025-09-04","2025-09-18","2025-10-16"]
W3 = np.array([1.,2.,8.])

ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); uid = ao["user_id"].to_numpy()
y = ao["target"].to_numpy().astype(float)
z037 = np.log1p(ao["pred_exp037"].to_numpy().astype(float))
zdist = np.log1p(ao["pred_dist"].to_numpy().astype(float))

raw = pq.read_table(E70/"count_value_moe_raw_OOF.parquet").to_pandas()
key = {(f,u): i for i,(f,u) in enumerate(zip(fold, uid))}
idx = np.array([key[(f,u)] for f,u in zip(raw["fold"].astype(str), raw["user_id"])])
zcount_sub = raw["z_predict"].to_numpy().astype(float)
zbase_sub  = raw["z_base"].to_numpy().astype(float)
corr_sub   = raw["correction"].to_numpy().astype(float)
covered = np.zeros(len(fold), bool); covered[idx] = True
zcount = np.full(len(fold), np.nan); zcount[idx] = zcount_sub

R = {}
R["target_max_err"] = float(np.max(np.abs(raw["target"].to_numpy()-y[idx])))
R["zbase_equals_exp037_maxerr"] = float(np.max(np.abs(zbase_sub - z037[idx])))
R["identity_maxerr"] = float(np.max(np.abs(zcount_sub - (zbase_sub + corr_sub))))
print("target max err", R["target_max_err"])
print("z_base == z_exp037 max err", R["zbase_equals_exp037_maxerr"])
print("z_predict == z_base + correction max err", R["identity_maxerr"])
print("saved candidates:", sorted(raw['candidate_name'].unique().tolist()))

def sc(f, zv):
    m = (fold==f)
    return calibrate_log_offset(y[m], zv[m])[1]

fm = pd.read_csv(E70/"fold_metrics.csv")
def rep(cand, f):
    r = fm[(fm.candidate==cand)&(fm.fold==f)]
    return float(r.rmsle_cal.iloc[0]) if len(r) else float("nan")

Z_REPLACE = z037 + 0.25*(zcount - zdist)
Z_ADD10   = 0.90*z037 + 0.10*zcount
rows=[]
print("\n%-24s %-12s %-20s %-20s %s"%("candidate","fold","recomputed","reported","diff"))
for cand, zv in (("COUNT_REAL", zcount), ("REPLACE_REAL_BETA1", Z_REPLACE), ("ADD10_REAL", Z_ADD10)):
    for f in CF:
        v = sc(f, zv); r = rep(cand, f)
        rows.append(dict(candidate=cand, fold=f, recomputed=v, reported=r, diff=v-r))
        print("%-24s %-12s %-20.12f %-20.12f %+.3e"%(cand,f,v,r,v-r))
for f in CF:
    v = sc(f, z037); r = rep("EXP037", f)
    rows.append(dict(candidate="EXP037", fold=f, recomputed=v, reported=r, diff=v-r))
    print("%-24s %-12s %-20.12f %-20.12f %+.3e"%("EXP037",f,v,r,v-r))
R["recomputed"] = rows

base128 = float(weighted_cv([sc(f,z037) for f in CF], W3))
rep128 = float(fm[(fm.candidate=="REPLACE_REAL_BETA1")&(fm.fold.str.startswith("partial"))].rmsle_cal.iloc[0])
my128 = float(weighted_cv([sc(f,Z_REPLACE) for f in CF], W3))
print("\n1:2:8 diagnostic: base %.12f  replace %.12f (reported %.12f)  delta %+.9f (reported -0.000086647)"
      % (base128, my128, rep128, my128-base128))
R["base_128"]=base128; R["replace_128"]=my128; R["replace_128_delta"]=my128-base128

# canonical 4-fold wCV is impossible: 2025-10-02 rows are absent
print("\nfolds present in EXP070 OOF:", sorted(set(raw['fold'].astype(str))))
print("canonical wCV computable:", set(CF) == {"2025-09-04","2025-09-18","2025-10-02","2025-10-16"})
R["missing_fold"]="2025-10-02"
R["shuffled_control_vector_saved"]=False

# fold decay of REAL vs EXP-037 and (from report) REAL-SHUFFLED
per = {f: sc(f,Z_REPLACE)-sc(f,z037) for f in CF}
print("\nREPLACE_REAL_BETA1 delta vs EXP-037 per fold:", {k: round(v,9) for k,v in per.items()})
rvs = pd.read_csv(E70/"real_vs_shuffled.csv")
rvs = rvs[rvs.path=="replacement_beta1"]
print("reported real-minus-shuffled per fold:")
for _,r in rvs.iterrows(): print("   ", r.fold, round(float(r.real_minus_shuffled),9), r.status)
R["per_fold_delta"]=per

# sanity: is the replacement gain even distinguishable from noise?
#   user-cluster bootstrap of the 1:2:8 diagnostic on the three completed folds
def boot(zv, n=300, seed=42):
    rng=np.random.default_rng(seed); users=np.unique(uid)
    o=np.argsort(uid,kind="stable"); us=uid[o]
    st=np.searchsorted(us,users,"left"); en=np.searchsorted(us,users,"right")
    out=np.empty(n)
    for b in range(n):
        pick=rng.integers(0,len(users),len(users))
        rr=np.concatenate([o[st[p]:en[p]] for p in pick])
        yb,fb=y[rr],fold[rr]; s1=[];s2=[]
        for f in CF:
            m=fb==f
            s1.append(calibrate_log_offset(yb[m], z037[rr][m])[1])
            s2.append(calibrate_log_offset(yb[m], zv[rr][m])[1])
        out[b]=weighted_cv(s2,W3)-weighted_cv(s1,W3)
    return out
print("\nbootstrapping the 1:2:8 replacement delta (300 reps)...", flush=True)
bs = boot(Z_REPLACE)
R["boot_128"]=dict(point=my128-base128, p02_5=float(np.quantile(bs,.025)),
                   p97_5=float(np.quantile(bs,.975)), p_lt_0=float((bs<0).mean()),
                   sd=float(bs.std(ddof=1)))
print("  ", json.dumps(R["boot_128"], indent=1))
(OUT/"_p3_exp070.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
print("WROTE _p3_exp070.json")
