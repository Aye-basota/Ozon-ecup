"""Phases 4.3 / 4.4 / 4.5 / 5 - TEST artifact audit, component decomposition,
baseline-drift isolation and span-expansion reproduction."""
from __future__ import annotations
import json, sys, hashlib, csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
E69 = REPO/"research"/"new_directions"/"EXP069_BTYD05_FRESH1_PROD"
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
GEOM = GEO/"submission_geometry"
sys.path.insert(0, str(GEOM))
from core import load_unique            # noqa: E402
from directions import build_basis      # noqa: E402

def sha256f(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()

res = {}

# ------------------------------------------------------------------ 4.3 TEST audits
at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
ft = pq.read_table(E69/"fresh_conditional_TEST.parquet").to_pandas()
ct = pq.read_table(E69/"btyd05_fresh1_TEST.parquet").to_pandas()
import pandas as pd
cc = pd.read_csv(E69/"btyd05_fresh1_TEST.csv")
sample = pd.read_csv(OLD/"data"/"raw"/"sample_submit.csv")
suid = sample["user_id"].to_numpy().astype(np.int64)

uid = at["user_id"].to_numpy().astype(np.int64)
audit = dict(
  aligned_test_rows=len(at), fresh_rows=len(ft), comb_rows=len(ct), csv_rows=len(cc),
  aligned_order_is_sample=bool(np.array_equal(uid, suid)),
  fresh_order_is_sample=bool(np.array_equal(ft["user_id"].to_numpy().astype(np.int64), suid)),
  comb_order_is_sample=bool(np.array_equal(ct["user_id"].to_numpy().astype(np.int64), suid)),
  csv_order_is_sample=bool(np.array_equal(cc["user_id"].to_numpy().astype(np.int64), suid)),
  csv_columns=list(cc.columns),
  unique_users=int(len(np.unique(uid))),
  comb_predict_finite=bool(np.isfinite(ct["predict"].to_numpy()).all()),
  comb_predict_nonneg=bool((ct["predict"].to_numpy() >= 0).all()),
  csv_vs_parquet_max_abs=float(np.max(np.abs(cc["predict"].to_numpy() - ct["predict"].to_numpy()))),
  sha_fresh_test=sha256f(E69/"fresh_conditional_TEST.parquet"),
  sha_comb_test=sha256f(E69/"btyd05_fresh1_TEST.parquet"),
  sha_comb_csv=sha256f(E69/"btyd05_fresh1_TEST.csv"),
)
res["test_schema_audit"] = audit
print(json.dumps(audit, indent=1))

# ------------------------------------------------------ 4.4 component decomposition
z_base_used = np.log1p(at["pred_exp037_rebuilt"].to_numpy().astype(float))
z_btyd_t    = np.log1p(at["pred_btyd"].to_numpy().astype(float))
d_fresh_t   = ft["correction"].to_numpy().astype(float)
d_vol_t     = ft["vol_correction"].to_numpy().astype(float)
z_comb_t    = ct["z_predict"].to_numpy().astype(float)
z_base_ct   = ct["z_base"].to_numpy().astype(float)
corr_ct     = ct["correction"].to_numpy().astype(float)

d_combined_t = z_comb_t - z_base_used
d_btyd_t     = 0.05*(z_btyd_t - z_base_used)
d_fresh_chk  = d_combined_t - d_btyd_t
dec = dict(
  identity_max_abs=float(np.max(np.abs(d_combined_t - (d_btyd_t + d_fresh_t)))),
  saved_fresh_vs_derived_max_abs=float(np.max(np.abs(d_fresh_chk - d_fresh_t))),
  comb_zbase_equals_exp037_rebuilt_max=float(np.max(np.abs(z_base_ct - z_base_used))),
  comb_corr_equals_btyd_plus_fresh_max=float(np.max(np.abs(corr_ct - (d_btyd_t + d_fresh_t)))),
  ft_zbase_equals_exp037_rebuilt_max=float(np.max(np.abs(ft["z_base"].to_numpy().astype(float) - z_base_used))),
)
res["test_decomposition"] = dec
print("decomposition:", json.dumps(dec, indent=1))

# baseline reconstruction drift: exact EXP037 formula from aligned TEST arrays
zt = {c: np.log1p(at[c].to_numpy().astype(float)) for c in
      ["pred_cap","pred_unc","pred_dist","pred_etx_avg3","pred_seq_avg3"]}
z_exp037_exact = (0.10*zt["pred_cap"] + 0.20*zt["pred_unc"] + 0.25*zt["pred_dist"]
                  + 0.225*zt["pred_seq_avg3"] + 0.225*zt["pred_etx_avg3"])
drift = z_base_used - z_exp037_exact
lvl = float(np.mean(z_exp037_exact))
res["baseline_drift"] = dict(
  rms=float(np.sqrt(np.mean(drift**2))), mean=float(drift.mean()),
  max_abs=float(np.max(np.abs(drift))), std=float(drift.std()),
  quantiles={str(q): float(np.quantile(drift, q)) for q in [0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1]},
  rms_after_removing_mean=float(np.sqrt(np.mean((drift-drift.mean())**2))),
  exact_mean_z=lvl, used_mean_z=float(np.mean(z_base_used)),
  n_floored_at_zero=int(np.sum(z_base_used <= 0)),
  n_exact_below_zero_before_shift=int(np.sum(z_exp037_exact + (float(np.mean(z_base_used))-lvl) < 0)),
)
print("baseline drift:", json.dumps({k:v for k,v in res['baseline_drift'].items() if k!='quantiles'}, indent=1))
print("  quantiles:", res['baseline_drift']['quantiles'])

# ------------------------------------------------------------ geometry span
Z, names, lb, guid = load_unique()
assert Z.shape == (65, 250_000)
order = np.argsort(uid); pos = np.searchsorted(uid[order], guid)
assert np.array_equal(uid[order][pos], guid)
def to_geo(v):  return v[order][pos]

z_ref, Phi, C, lam, W_ = build_basis(Z, 0, tol=1e-12)
N = Z.shape[1]
z_center = Z.mean(axis=0)
res["geometry"] = dict(n_sources=int(Z.shape[0]), rank=int(Phi.shape[0]),
                       lam_max=float(lam[0]), lam_min=float(lam[-1]),
                       basis_sha256=sha256f(GEOM/"cache"/"Z.npz"))
print("geometry rank", Phi.shape[0], "lam", lam[0], lam[-1])

def proj_lin(d):
    """linear projection of a DIRECTION onto span(Phi) in the mean_N metric."""
    c = Phi @ d / N
    par = c @ Phi
    return par, d - par

ones = np.ones(N)
par_one, perp_one = proj_lin(ones)
res["constant_direction_in_span"] = dict(
  rms_residual=float(np.sqrt(np.mean(perp_one**2))),
  fraction_of_unit=float(np.sqrt(np.mean(perp_one**2))/1.0))
print("constant direction residual RMS (1.0 = fully outside span):",
      res["constant_direction_in_span"]["rms_residual"])

def point_span_metrics(z_point, tag):
    d = z_point - z_center
    c = Phi @ d / N
    residual = d - c @ Phi
    rms = float(np.sqrt(np.mean(residual**2)))
    drms = float(np.sqrt(np.mean(d**2)))
    dist = np.sqrt(np.mean((Z - z_point)**2, axis=1))
    i = int(np.argmin(dist))
    Y = Z - Z[0]; dref = z_point - Z[0]
    e = np.linalg.eigvalsh(Y@Y.T/N); r0 = int(np.sum(e > 1e-12*e[-1]))
    Ya = np.vstack([Y, dref]); ea = np.linalg.eigvalsh(Ya@Ya.T/N)
    r1 = int(np.sum(ea > 1e-12*ea[-1]))
    return dict(tag=tag, orthogonal_RMS=rms, centroid_RMS=drms, orth_fraction=rms/drms,
                nearest_source=names[i], nearest_source_rms=float(dist[i]),
                rank_before=r0, rank_after=r1)

pm = point_span_metrics(to_geo(z_comb_t), "EXP069_combined_point")
res["exp069_point_span"] = pm
print("EXP069 point span reproduction:", json.dumps(pm, indent=1))

# ------------------------------------------------- Phase 5: component directions
comp = {"d_combined": d_combined_t, "d_btyd": d_btyd_t, "d_fresh": d_fresh_t,
        "d_vol_placebo": d_vol_t, "baseline_drift": drift}
rows = []
perp_store = {}
for k, v in comp.items():
    g = to_geo(v)
    par, perp = proj_lin(g)
    perp_store[k] = perp
    rms = float(np.sqrt(np.mean(g**2)))
    rp = float(np.sqrt(np.mean(perp**2)))
    rpar = float(np.sqrt(np.mean(par**2)))
    # nearest existing source direction (as difference from centroid) correlation
    Ycent = Z - z_center
    corrs = []
    for i in range(Z.shape[0]):
        a = Ycent[i]; b = g
        corrs.append(float(np.dot(a-a.mean(), b-b.mean())/(np.linalg.norm(a-a.mean())*np.linalg.norm(b-b.mean())+1e-30)))
    corrs = np.array(corrs); j = int(np.argmax(np.abs(corrs)))
    rows.append(dict(component=k, total_RMS=rms, mean=float(g.mean()),
                     parallel_RMS=rpar, orthogonal_RMS=rp,
                     orthogonal_fraction=rp/rms if rms>0 else 0.0,
                     perp_mean=float(perp.mean()),
                     max_abs_corr_with_source_direction=float(corrs[j]),
                     nearest_source_direction=names[j]))
    print(f"{k:16s} RMS={rms:.6f} par={rpar:.6f} perp={rp:.6f} frac={rp/rms:.4f} "
          f"mean={g.mean():+.6f} perp_mean={perp.mean():+.3e} best|corr|={corrs[j]:+.3f} ({names[j]})")

# cross relations of orthogonal parts
pf, pc, pb = perp_store["d_fresh"], perp_store["d_combined"], perp_store["d_btyd"]
def cosang(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
res["perp_relations"] = dict(
  cos_fresh_combined=cosang(pf,pc), cos_fresh_btyd=cosang(pf,pb), cos_btyd_combined=cosang(pb,pc),
  rms_fresh=float(np.sqrt(np.mean(pf**2))), rms_combined=float(np.sqrt(np.mean(pc**2))),
  rms_btyd=float(np.sqrt(np.mean(pb**2))),
  rms_diff_fresh_minus_combined=float(np.sqrt(np.mean((pf-pc)**2))))
print("perp relations:", json.dumps(res["perp_relations"], indent=1))

# rank after adding each component direction
Y = Z - Z[0]; e = np.linalg.eigvalsh(Y@Y.T/N); r0 = int(np.sum(e > 1e-12*e[-1]))
rank_rows = {}
for k, v in comp.items():
    g = to_geo(v)
    Ya = np.vstack([Y, g])          # adding the DIRECTION itself
    ea = np.linalg.eigvalsh(Ya@Ya.T/N)
    rank_rows[k] = dict(rank_before=r0, rank_after=int(np.sum(ea > 1e-12*ea[-1])))
res["direction_rank"] = rank_rows
print("direction rank:", json.dumps(rank_rows, indent=1))

with open(OUT/"exp069_component_decomposition.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

np.savez_compressed(OUT/"_test_components.npz",
                    user_id=uid, d_fresh=d_fresh_t, d_btyd=d_btyd_t, d_combined=d_combined_t,
                    d_vol=d_vol_t, baseline_drift=drift, z_base_used=z_base_used,
                    z_exp037_exact=z_exp037_exact)
(OUT/"_p4345.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\nWROTE component decomposition + _p4345.json")
