"""Phases 2 and 3 - independent audits of EXP071 and EXP070."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pandas as pd

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv

GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
ND = REPO/"research"/"new_directions"
OUT = ND/"NEXT_SUBMISSION_AFTER_EXP069"
FOLDS = ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]
W = np.array([1.,2.,4.,8.])

ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); uid = ao["user_id"].to_numpy()
y = ao["target"].to_numpy().astype(float)
zc = {c: np.log1p(ao[c].to_numpy().astype(float)) for c in ao.columns if c.startswith("pred_")}
z037 = zc["pred_exp037"]

def score(mask, zv):
    return calibrate_log_offset(y[mask], zv[mask])[1]

R = {}
# =====================================================================  EXP071
print("="*70); print("EXP071 INDEPENDENT AUDIT"); print("="*70)
E71 = ND/"EXP071_ETX_FRESH_CONTRAST"
e71 = pq.read_table(E71/"etx_fresh_raw_OOF.parquet").to_pandas()
pm = json.loads((E71/"pilot_metrics.json").read_text(encoding="utf-8"))
F = pm["fold"]
print("rows", len(e71), "folds", e71["fold"].unique().tolist(), "scope", e71["scope"].unique().tolist())
mask = fold == F
sub = e71[e71["fold"].astype(str) == F]
# align to canonical order
o = np.argsort(sub["user_id"].to_numpy()); su = sub["user_id"].to_numpy()[o]
cu = uid[mask]
aligned = bool(np.array_equal(np.sort(cu), su))
pos = np.searchsorted(su, cu)
assert np.array_equal(su[pos], cu)
d_raw = sub["d_etx_fresh_raw"].to_numpy()[o][pos].astype(float)
d_raw_vol = sub["d_etx_vol_raw"].to_numpy()[o][pos].astype(float)
uside = sub["user_side"].to_numpy()[o][pos].astype(int)

pr, pv = pm["preprocessing_real"], pm["preprocessing_vol"]
d_real = np.clip(d_raw, pr["lo"], pr["hi"]) - pr["center"]
d_vol  = np.clip(d_raw_vol, pv["lo"], pv["hi"]) - pv["center"]
# independent recomputation of donor-fold bounds (0.5/99.5 pct + winsorized mean)
donor = ~mask
# the donor raw vectors are NOT saved for EXP071 (pilot only saved the pilot fold)
z_seq_fresh_col = zc["pred_fresh_contrast"]

def zfull(delta):
    v = z037.copy(); v[mask] = z037[mask] + delta; return v

rec = {}
rec["exp037"] = score(mask, z037)
rec["etx_real"] = score(mask, zfull(d_real))
rec["etx_vol"] = score(mask, zfull(d_vol))
rec["seq_fresh"] = score(mask, z_seq_fresh_col)
gamma = json.loads((E71/"seq_vs_etx_fresh.csv").read_text(encoding="utf-8").splitlines()[1].split(",")[2]) \
        if False else float(pd.read_csv(E71/"seq_vs_etx_fresh.csv")["gamma"].iloc[0])
seq_corr = z_seq_fresh_col[mask] - z037[mask]
orth = d_real - gamma*seq_corr
zz = z037.copy(); zz[mask] = z_seq_fresh_col[mask] + orth
rec["seq_plus_etx_orth"] = score(mask, zz)
rec["real_delta"] = rec["etx_real"] - rec["exp037"]
rec["vol_delta"] = rec["etx_vol"] - rec["exp037"]
rec["real_minus_vol"] = rec["etx_real"] - rec["etx_vol"]
rec["orth_incremental_delta"] = rec["seq_plus_etx_orth"] - rec["seq_fresh"]
rep = pm["metrics"]
print("metric".ljust(24), "recomputed".ljust(22), "reported".ljust(22), "diff")
for k in ["exp037","etx_real","etx_vol","seq_fresh","seq_plus_etx_orth",
          "real_delta","vol_delta","real_minus_vol","orth_incremental_delta"]:
    print(f"  {k:22s} {rec[k]:< 22.12f} {rep[k]:< 22.12f} {rec[k]-rep[k]:+.3e}")
# correlation / rms diagnostics
corr_es = float(np.corrcoef(d_real, seq_corr)[0,1])
print("corr(etx_real, seq_corr) =", corr_es, " reported", 0.2402781055495691)
print("etx rms", float(np.sqrt(np.mean(d_real**2))), "orth rms", float(np.sqrt(np.mean(orth**2))))
print("gamma (target-free) =", gamma, " OLS gamma =", float(np.dot(d_real, seq_corr)/np.dot(seq_corr, seq_corr)))
# user halves
def splitmix64(x):
    x = np.asarray(x, dtype=np.uint64).copy()
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15); z1 = x
        z1 = (z1 ^ (z1 >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z1 = (z1 ^ (z1 >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z1 = z1 ^ (z1 >> np.uint64(31))
    return z1
side = (splitmix64(uid) & np.uint64(1)).astype(int)
print("saved user_side matches splitmix on pilot fold:", bool(np.array_equal(uside, side[mask])))
halves = {}
for g, sv in (("A",0), ("B",1)):
    m2 = mask & (side == sv)
    zr = z037.copy(); zr[mask] = z037[mask] + d_real
    zv = z037.copy(); zv[mask] = z037[mask] + d_vol
    r_ = score(m2, zr); v_ = score(m2, zv)
    halves[g] = dict(n=int(m2.sum()), real=r_, vol=v_, real_minus_vol=r_-v_)
    print(f"  half {g}: n={m2.sum()} real={r_:.12f} vol={v_:.12f} real-vol={r_-v_:+.9f}")
R["EXP071"] = dict(recomputed=rec, reported=rep, halves=halves,
                   corr_etx_seq=corr_es, gamma=gamma, rows=len(e71),
                   pilot_fold=F, aligned=aligned)

# ---------- alpha grid replay (diagnostic, as recorded) -----------------
grid = []
for a in [0.0,0.25,0.5,0.75,1.0]:
    zr = z037.copy(); zr[mask] = z037[mask] + a*d_real
    zv = z037.copy(); zv[mask] = z037[mask] + a*d_vol
    zcb = z037.copy(); zcb[mask] = z_seq_fresh_col[mask] + a*orth
    grid.append(dict(alpha=a, real=score(mask, zr), vol=score(mask, zv), combined=score(mask, zcb)))
print("\nalpha grid replay (real / vol / seq+orth):")
for g_, rrec in zip(pm["diagnostic_grid_not_used_for_gate"], grid):
    print(f"  a={rrec['alpha']:.2f} real {rrec['real']:.12f} (rep {g_['real_score']:.12f}) "
          f"vol {rrec['vol']:.12f} (rep {g_['vol_score']:.12f}) "
          f"comb {rrec['combined']:.12f} (rep {g_['combined_score']:.12f})")
R["EXP071"]["alpha_grid"] = grid

# =====================================================================  EXP070
print()
print("="*70); print("EXP070 INDEPENDENT AUDIT"); print("="*70)
E70 = ND/"EXP070_COUNT_VALUE_MOE"
raw = pq.read_table(E70/"count_value_moe_raw_OOF.parquet").to_pandas()
prob = pq.read_table(E70/"count_probabilities_OOF.parquet").to_pandas()
print("raw rows", len(raw), "candidates", sorted(raw["candidate_name"].unique().tolist()))
print("fold sizes", raw.groupby(raw["fold"].astype(str)).size().to_dict())
print("prob rows", len(prob), "fold sizes", prob.groupby(prob["fold"].astype(str)).size().to_dict())
ps = prob[["p0","p1","p2","p3","p4"]].to_numpy()
rowsum_err = float(np.max(np.abs(ps.sum(axis=1) - 1)))
print("max |row-sum - 1| =", rowsum_err, "(reported 5.16e-8)")
print("probabilities in [0,1]:", bool(((ps >= 0) & (ps <= 1)).all()))
cls = prob["count_class"].to_numpy()
print("class distribution (OOF, 3 folds):", {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))})
n30 = prob["N30"].to_numpy()
bins_ok = bool(np.array_equal(cls, np.digitize(n30, [1,2,4,8])))
print("bins C0=0,C1=1,C2=2-3,C3=4-7,C4>=8 reproduce from N30:", bins_ok,
      " C4 share in OOF =", float((cls == 4).mean()))

# alignment + target parity to canonical bank
key_ao = {(f, u): i for i, (f, u) in enumerate(zip(fold, uid))}
tgt_err, missing = 0.0, 0
sub_idx = {}
for cand in sorted(raw["candidate_name"].unique()):
    s = raw[raw["candidate_name"] == cand]
    idx = np.array([key_ao.get((f, u), -1) for f, u in zip(s["fold"].astype(str), s["user_id"])])
    missing += int((idx < 0).sum())
    sub_idx[cand] = (s, idx)
    tgt_err = max(tgt_err, float(np.max(np.abs(s["target"].to_numpy() - y[idx]))))
print("rows not found in canonical bank:", missing, " max target error:", tgt_err)

# recompute scores per candidate/fold
CF = ["2025-09-04","2025-09-18","2025-10-16"]
rows70 = []
for cand, (s, idx) in sub_idx.items():
    zp = s["z_predict"].to_numpy().astype(float)
    for f in CF:
        m = s["fold"].astype(str).to_numpy() == f
        sc = calibrate_log_offset(y[idx][m], zp[m])[1]
        b = score(fold == f, z037)
        rows70.append(dict(candidate=cand, fold=f, n=int(m.sum()), rmsle_cal=sc,
                           delta_vs_exp037=sc-b))
df70 = pd.DataFrame(rows70)
fm = pd.read_csv(E70/"fold_metrics.csv")
print("\nrecomputed vs reported (calibrated RMSLE):")
for _, r in df70.iterrows():
    rep_row = fm[(fm.candidate == r.candidate) & (fm.fold == r.fold)]
    rep_v = float(rep_row.rmsle_cal.iloc[0]) if len(rep_row) else float("nan")
    print(f"  {r.candidate:24s} {r.fold} recomputed {r.rmsle_cal:.12f} reported {rep_v:.12f} "
          f"diff {r.rmsle_cal-rep_v:+.3e}")
# formula check for replacement
s, idx = sub_idx.get("REPLACE_REAL_BETA1", (None, None))
if s is not None:
    zb = s["z_base"].to_numpy().astype(float); co = s["correction"].to_numpy().astype(float)
    print("\nREPLACE_REAL_BETA1: max|z_predict-(z_base+correction)| =",
          float(np.max(np.abs(s["z_predict"].to_numpy()-(zb+co)))))
    print("  max|z_base - z_exp037| =", float(np.max(np.abs(zb - z037[idx]))))
    # replacement of 0.25*DIST by 0.25*count -> correction should equal 0.25*(z_count - z_dist)
    st, it = sub_idx["COUNT_REAL"]
    # align COUNT_REAL rows to REPLACE rows
    kmap = {(f, u): i for i, (f, u) in enumerate(zip(st["fold"].astype(str), st["user_id"]))}
    j = np.array([kmap[(f, u)] for f, u in zip(s["fold"].astype(str), s["user_id"])])
    zcount = st["z_predict"].to_numpy().astype(float)[j]
    zdist = zc["pred_dist"][idx]
    pred_corr = 0.25*(zcount - zdist)
    print("  max|correction - 0.25*(z_count - z_dist)| =", float(np.max(np.abs(co - pred_corr))))

# 1:2:8 diagnostic and per-fold decay
W3 = np.array([1.,2.,8.])
def w128(cand):
    sc = [float(df70[(df70.candidate == cand) & (df70.fold == f)].rmsle_cal.iloc[0]) for f in CF]
    return float(weighted_cv(sc, W3))
base128 = float(weighted_cv([score(fold == f, z037) for f in CF], W3))
print("\n1:2:8 diagnostic (NOT canonical wCV):")
for cand in ["REPLACE_REAL_BETA1","REPLACE_SHUFFLED_BETA1","ADD10_REAL","ADD10_SHUFFLED",
             "COUNT_REAL","COUNT_SHUFFLED"]:
    if cand in sub_idx:
        v = w128(cand)
        print(f"  {cand:24s} {v:.12f} delta {v-base128:+.9f}")
rr = w128("REPLACE_REAL_BETA1") - base128
rs = w128("REPLACE_SHUFFLED_BETA1") - base128
print(f"  replacement REAL-SHUFFLED (1:2:8) = {rr-rs:+.9f}  (reported -0.000052609)")
print("  per-fold REAL-SHUFFLED (replacement):")
for f in CF:
    a = float(df70[(df70.candidate=='REPLACE_REAL_BETA1')&(df70.fold==f)].rmsle_cal.iloc[0])
    b = float(df70[(df70.candidate=='REPLACE_SHUFFLED_BETA1')&(df70.fold==f)].rmsle_cal.iloc[0])
    print(f"    {f}: {a-b:+.9f}")
R["EXP070"] = dict(rowsum_err=rowsum_err, bins_reproduce=bins_ok,
                   c4_share_oof=float((cls == 4).mean()),
                   target_max_err=tgt_err, rows_missing=missing,
                   recomputed=df70.to_dict("records"), base_128=base128,
                   replace_real_128_delta=rr, replace_shuffled_128_delta=rs,
                   real_minus_shuffled_128=rr-rs)

(OUT/"_p23.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
print("\nWROTE _p23.json")
