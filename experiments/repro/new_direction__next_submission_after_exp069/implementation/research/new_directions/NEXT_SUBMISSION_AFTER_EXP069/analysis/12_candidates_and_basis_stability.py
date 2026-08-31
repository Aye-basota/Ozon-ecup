"""Phases 7 and 9 - three internal candidate families, basis stability, TEST regime."""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean"); sys.path.insert(0, str(REPO))
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research"); GEOM = GEO/"submission_geometry"
sys.path.insert(0, str(GEOM))
from core import load_unique
from directions import build_basis
E69 = REPO/"research"/"new_directions"/"EXP069_BTYD05_FRESH1_PROD"
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
ALPHAS = [0.25, 0.50, 0.75, 1.00]

at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
uid = at["user_id"].to_numpy().astype(np.int64)
Z, names, lb, guid = load_unique(); N = Z.shape[1]
print("aligned TEST order == geometry Z order:", bool(np.array_equal(uid, guid)))
order = np.argsort(uid); pos = np.searchsorted(uid[order], guid)
assert np.array_equal(uid[order][pos], guid)
def to_geo(v): return np.asarray(v)[order][pos]

comp = np.load(OUT/"_test_components.npz")
d_fresh = to_geo(comp["d_fresh"]); d_btyd = to_geo(comp["d_btyd"])
d_comb = to_geo(comp["d_combined"]); d_vol = to_geo(comp["d_vol"])
z_inc = np.log1p(to_geo(at["pred_current_1_6466079084"].to_numpy().astype(float)))
z_prev = np.log1p(to_geo(at["pred_previous_1_6467120249"].to_numpy().astype(float)))
z_037t = np.log1p(to_geo(at["pred_exp037_rebuilt"].to_numpy().astype(float)))

z_ref, Phi, C, lam, W_ = build_basis(Z, 0, tol=1e-12)
def perp(d, P=Phi):
    return d - (P @ d / N) @ P

R = {}
# ---- where does the incumbent sit relative to the span? -------------------
zc = Z.mean(axis=0)
res_inc = perp(z_inc - zc)
R["incumbent_span_residual_rms"] = float(np.sqrt(np.mean(res_inc**2)))
R["incumbent_span_residual_max"] = float(np.max(np.abs(res_inc)))
R["incumbent_zero_predictions"] = int(np.sum(z_inc <= 0))
print("incumbent residual from the 65-source affine span: RMS %.3e  max %.3e  zeros %d"
      % (R["incumbent_span_residual_rms"], R["incumbent_span_residual_max"], R["incumbent_zero_predictions"]))
print("  (a perp correction is therefore orthogonal to the incumbent's own span position "
      "to within this residual)")

d_fresh_perp = perp(d_fresh); d_comb_perp = perp(d_comb)
R["equivalence"] = dict(
  cos=float(np.dot(d_fresh_perp, d_comb_perp)/np.linalg.norm(d_fresh_perp)/np.linalg.norm(d_comb_perp)),
  rms_fresh=float(np.sqrt(np.mean(d_fresh_perp**2))),
  rms_comb=float(np.sqrt(np.mean(d_comb_perp**2))),
  rms_difference=float(np.sqrt(np.mean((d_fresh_perp-d_comb_perp)**2))),
  relative_difference=float(np.sqrt(np.mean((d_fresh_perp-d_comb_perp)**2))/np.sqrt(np.mean(d_fresh_perp**2))))
print("A vs B equivalence:", json.dumps(R["equivalence"], indent=1))

# ------------------------------------------------ candidate diagnostics ----
CANDS = {"A_fresh_perp": d_fresh_perp, "B_combined_perp": d_comb_perp, "C_combined_full": d_comb}
p_inc = np.expm1(z_inc)
rows = []
for cname, dv in CANDS.items():
    for a in ALPHAS:
        z = z_inc + a*dv
        p = np.maximum(np.expm1(z), 0.0)
        dz = z - z_inc
        rows.append(dict(candidate=cname, alpha=a,
            rms_move=float(np.sqrt(np.mean(dz**2))), mean_log_shift=float(dz.mean()),
            max_abs_move=float(np.max(np.abs(dz))),
            zeros_before=int((p_inc<=0).sum()), zeros_after=int((p<=0).sum()),
            n_floored_by_expm1=int((np.expm1(z)<0).sum()),
            q001=float(np.quantile(p,.001)), q25=float(np.quantile(p,.25)),
            q50=float(np.quantile(p,.5)), q75=float(np.quantile(p,.75)),
            q99=float(np.quantile(p,.99)), q999=float(np.quantile(p,.999)),
            max_pred=float(p.max()), mean_pred=float(p.mean()),
            top1pct_mean_change=float(np.mean(dz[p_inc>=np.quantile(p_inc,.99)])),
            orth_rms=float(np.sqrt(np.mean(perp(dz)**2))),
            orth_fraction=float(np.sqrt(np.mean(perp(dz)**2))/np.sqrt(np.mean(dz**2))),
            corr_with_incumbent=float(np.corrcoef(np.log1p(p), z_inc)[0,1]),
            corr_with_exp069_abs=float(np.corrcoef(np.log1p(p),
                np.log1p(to_geo(pq.read_table(E69/'btyd05_fresh1_TEST.parquet').to_pandas()['predict'].to_numpy())))[0,1]),
            second_order_lb_penalty=float(np.mean(dz**2)/(2*1.6466079084)),
            public_noise_sd=float(np.sqrt(np.mean(dz**2))*np.sqrt(0.8/50000)),
        ))
cand = pd.DataFrame(rows)
pd.set_option("display.width", 300)
print("\n=== candidate diagnostics ===")
print(cand[["candidate","alpha","rms_move","mean_log_shift","max_abs_move","zeros_after",
            "orth_fraction","second_order_lb_penalty","public_noise_sd","max_pred"]]
      .to_string(index=False, float_format=lambda v: f"{v: .6g}"))
cand.to_csv(OUT/"candidate_comparison.csv", index=False)

# ------------------------------------------------------- rank of candidate -
Y = Z - Z[0]; e = np.linalg.eigvalsh(Y@Y.T/N); r0 = int((e > 1e-12*e[-1]).sum())
rk = {}
for cname, dv in CANDS.items():
    Ya = np.vstack([Y, dv]); ea = np.linalg.eigvalsh(Ya@Ya.T/N)
    rk[cname] = dict(rank_before=r0, rank_after=int((ea > 1e-12*ea[-1]).sum()))
R["candidate_rank"] = rk
print("\ncandidate rank:", json.dumps(rk))

# ------------------------------------------------------- basis stability ---
print("\n=== basis stability ===", flush=True)
rng = np.random.default_rng(20260826)
stab_rows = []
def drift_for(sub_idx, tag):
    Zs = Z[sub_idx]
    _, Ph, _, _, _ = build_basis(Zs, 0, tol=1e-12)
    out = {}
    for cname, dv in (("A_fresh_perp", d_fresh), ("B_combined_perp", d_comb)):
        dp = perp(dv, Ph)
        base = d_fresh_perp if cname == "A_fresh_perp" else d_comb_perp
        out[cname] = dict(rms=float(np.sqrt(np.mean(dp**2))),
                          drift=float(np.sqrt(np.mean((dp-base)**2))),
                          drift_frac=float(np.sqrt(np.mean((dp-base)**2))/np.sqrt(np.mean(base**2))),
                          cos=float(np.dot(dp,base)/np.linalg.norm(dp)/np.linalg.norm(base)))
    return dict(tag=tag, rank=int(Ph.shape[0]), **{f"{k}_{m}": v for k, d_ in out.items() for m, v in d_.items()})

# (1) leave-one-source-out
for i in range(len(names)):
    keep = [j for j in range(len(names)) if j != i]
    stab_rows.append(drift_for(keep, "LOSO:"+names[i]))
loso = pd.DataFrame([r for r in stab_rows if r["tag"].startswith("LOSO")])
print("LOSO (65 refits): A drift_frac median %.4f  p90 %.4f  max %.4f | cos min %.5f"
      % (loso.A_fresh_perp_drift_frac.median(), loso.A_fresh_perp_drift_frac.quantile(.9),
         loso.A_fresh_perp_drift_frac.max(), loso.A_fresh_perp_cos.min()))
worst = loso.reindex(loso.A_fresh_perp_drift_frac.sort_values(ascending=False).index)[:6]
for _, r in worst.iterrows():
    print("   %-60s drift_frac %.4f" % (r.tag[5:], r.A_fresh_perp_drift_frac))

# (2) drop each of the 10 largest-|weight| sources (geometry recovered weights)
rw = np.load(GEOM/"cache"/"recovered_w.npz", allow_pickle=True)
w = rw["w"]; print("\nrecovered weight vector:", w.shape, "L1", float(np.abs(w).sum()))
top10 = np.argsort(-np.abs(w))[:10]
for i in top10:
    keep = [j for j in range(len(names)) if j != i]
    stab_rows.append(drift_for(keep, "TOPW:"+names[i]))
tw = pd.DataFrame([r for r in stab_rows if r["tag"].startswith("TOPW")])
print("top-10-weight drops: A drift_frac median %.4f max %.4f" %
      (tw.A_fresh_perp_drift_frac.median(), tw.A_fresh_perp_drift_frac.max()))
for _, r in tw.iterrows():
    print("   %-60s drift_frac %.4f" % (r.tag[5:], r.A_fresh_perp_drift_frac))

# (3) repeated 80% source subsamples, fixed seeds
for b in range(40):
    keep = np.sort(np.random.default_rng(1000+b).choice(len(names), int(0.8*len(names)), replace=False))
    stab_rows.append(drift_for(list(keep), f"SUB80:seed{1000+b}"))
sb = pd.DataFrame([r for r in stab_rows if r["tag"].startswith("SUB80")])
print("\n80%% subsample (40 fixed seeds): A drift_frac median %.4f  p90 %.4f  max %.4f | rank median %d"
      % (sb.A_fresh_perp_drift_frac.median(), sb.A_fresh_perp_drift_frac.quantile(.9),
         sb.A_fresh_perp_drift_frac.max(), int(sb["rank"].median())))
print("   cos with full-basis direction: min %.5f median %.5f"
      % (sb.A_fresh_perp_cos.min(), sb.A_fresh_perp_cos.median()))
S = pd.DataFrame(stab_rows)
S.to_csv(OUT/"basis_stability.csv", index=False)
R["basis_stability_summary"] = dict(
  loso=dict(median=float(loso.A_fresh_perp_drift_frac.median()), p90=float(loso.A_fresh_perp_drift_frac.quantile(.9)),
            max=float(loso.A_fresh_perp_drift_frac.max()), min_cos=float(loso.A_fresh_perp_cos.min())),
  top_weight=dict(median=float(tw.A_fresh_perp_drift_frac.median()), max=float(tw.A_fresh_perp_drift_frac.max())),
  sub80=dict(median=float(sb.A_fresh_perp_drift_frac.median()), p90=float(sb.A_fresh_perp_drift_frac.quantile(.9)),
             max=float(sb.A_fresh_perp_drift_frac.max()), min_cos=float(sb.A_fresh_perp_cos.min()),
             rank_min=int(sb["rank"].min()), rank_max=int(sb["rank"].max())))

# ------------------------------------------------------- TEST regime -------
ft = pq.read_table(E69/"fresh_conditional_TEST.parquet").to_pandas()
fo = pq.read_table(E69/"fresh_conditional_OOF.parquet").to_pandas()
oof_corr = fo["correction"].to_numpy().astype(float)
oof_raw = fo["raw_correction"].to_numpy().astype(float)
tst_corr = ft["correction"].to_numpy().astype(float)
tst_raw = ft["raw_correction"].to_numpy().astype(float)
p_dist_t = ft["p_dist"].to_numpy().astype(float)
def st(x):
    return dict(mean=float(x.mean()), std=float(x.std()), rms=float(np.sqrt(np.mean(x**2))),
                q=[float(np.quantile(x,q)) for q in (0.001,.01,.05,.25,.5,.75,.95,.99,.999)],
                min=float(x.min()), max=float(x.max()))
oof_perp_npz = np.load(OUT/"_oof_perp.npz")
regime = dict(
  oof_raw=st(oof_raw), test_raw=st(tst_raw),
  oof_processed=st(oof_corr), test_processed=st(tst_corr),
  oof_perp=st(oof_perp_npz["perp_fresh"]), test_perp=st(d_fresh_perp),
  raw_std_ratio_test_over_oof=float(tst_raw.std()/oof_raw.std()),
  processed_std_ratio=float(tst_corr.std()/oof_corr.std()),
  perp_rms_ratio=float(np.sqrt(np.mean(d_fresh_perp**2))/np.sqrt(np.mean(oof_perp_npz["perp_fresh"]**2))),
  test_clipped_fraction=float(np.mean((tst_raw < -0.12031936645507812)|(tst_raw > 0.12181227207183842))),
  p_dist=st(p_dist_t),
  p_dist_zero_fraction=float(np.mean(p_dist_t <= 0)),
  correction_zero_fraction_test=float(np.mean(np.abs(tst_raw) < 1e-12)),
  note=("TEST conditional heads average two donor sides x three seeds; the historical OOF "
        "vector is a single-seed cross-fit, so the TEST correction is a lower-variance "
        "estimate of the same contrast.  No TEST target and no rescaling were used."))
print("\n=== TEST regime ===")
for k in ("raw_std_ratio_test_over_oof","processed_std_ratio","perp_rms_ratio",
          "test_clipped_fraction","p_dist_zero_fraction","correction_zero_fraction_test"):
    print(f"  {k:34s} {regime[k]:.6f}")
print("  oof_perp rms %.6f   test_perp rms %.6f" % (regime["oof_perp"]["rms"], regime["test_perp"]["rms"]))
print("  p_dist mean %.4f std %.4f min %.4f max %.4f" %
      (regime["p_dist"]["mean"], regime["p_dist"]["std"], regime["p_dist"]["min"], regime["p_dist"]["max"]))
(OUT/"test_regime.json").write_text(json.dumps(regime, indent=2), encoding="utf-8")
(OUT/"_p79.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
np.savez_compressed(OUT/"_perp_directions.npz", user_id=guid,
                    d_fresh_perp=d_fresh_perp, d_comb_perp=d_comb_perp, d_comb=d_comb,
                    z_inc=z_inc)
print("\nWROTE candidate_comparison.csv, basis_stability.csv, test_regime.json")
