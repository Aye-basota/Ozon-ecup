"""Phase 6 - honest OOF analysis of the novel (out-of-span) component.

Baseline is ALWAYS pred_exp037 from 06_ALIGNED_OOF. The public incumbent is never
used as an OOF baseline, never as a residual target and never for selection.
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv

GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
E69 = REPO/"research"/"new_directions"/"EXP069_BTYD05_FRESH1_PROD"
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
FOLDS = ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]
W = np.array([1.,2.,4.,8.])
ALPHAS = np.array([0.0,0.25,0.50,0.75,1.00])

ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
f69 = pq.read_table(E69/"fresh_conditional_OOF.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); uid = ao["user_id"].to_numpy()
y = ao["target"].to_numpy().astype(float)
zc = {c: np.log1p(ao[c].to_numpy().astype(float)) for c in ao.columns
      if c.startswith("pred_")}
z037 = zc["pred_exp037"]
corr = f69["correction"].to_numpy().astype(float)
d_btyd = 0.05*(zc["pred_btyd"] - z037)
d_comb = d_btyd + corr

# ------------------------------------------------------------- eligible basis
# Rule: source has BOTH a canonical aligned OOF column AND a column in the aligned
# TEST bank (i.e. its TEST vector existed and its family is inside the 65-source
# geometry).  pred_fresh_contrast (the candidate's own historical direction),
# pred_seq_d3a_avg3, pred_mhz_full and pred_block4_saf have no aligned TEST column
# and therefore cannot be shown to be inside the current geometry: excluded.
BASIS_MAIN = ["pred_exp037","pred_cap","pred_unc","pred_dist","pred_etx_avg3",
              "pred_seq_avg3","pred_ridge15","pred_hurdle_e11","pred_holiday_yoy",
              "pred_btyd","pred_btyd05"]
BASIS_STRICT = ["pred_exp037","pred_holiday_yoy","pred_btyd05"]
BASIS_WIDE = BASIS_MAIN + ["pred_seq_d3a_avg3","pred_mhz_full","pred_block4_saf"]

def lofo_affine_residual(d, basis):
    """Per-fold: fit affine projection of d on `basis` using ONLY donor folds, apply
    to the held-out fold.  Target-free.  Returns residual + retained-variance stats."""
    Xall = np.column_stack([zc[b] for b in basis] + [np.ones(len(d))])
    perp = np.empty_like(d); par = np.empty_like(d)
    for f in FOLDS:
        te = fold == f; tr = ~te
        beta, *_ = np.linalg.lstsq(Xall[tr], d[tr], rcond=None)
        par[te] = Xall[te] @ beta
        perp[te] = d[te] - par[te]
    return par, perp

def wcv_of(zv, mask=None):
    sc = []
    for f in FOLDS:
        m = (fold == f) if mask is None else ((fold == f) & mask)
        _, s = calibrate_log_offset(y[m], zv[m]); sc.append(s)
    return float(weighted_cv(sc, W)), np.array(sc)

base_wcv, base_sc = wcv_of(z037)
print("EXP037 baseline wCV", repr(base_wcv))

def splitmix64(x):
    x = np.asarray(x, dtype=np.uint64).copy()
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        z1 = x
        z1 = (z1 ^ (z1 >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z1 = (z1 ^ (z1 >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z1 = z1 ^ (z1 >> np.uint64(31))
    return z1
side = (splitmix64(uid) & np.uint64(1)).astype(int)

def nested_alpha(dvec):
    """Honest LOFO alpha selection; ties -> smaller alpha."""
    # precompute per-fold score for every alpha
    S = np.zeros((len(ALPHAS), len(FOLDS)))
    for ia, a in enumerate(ALPHAS):
        zv = z037 + a*dvec
        for ifd, f in enumerate(FOLDS):
            m = fold == f
            _, s = calibrate_log_offset(y[m], zv[m]); S[ia, ifd] = s
    rows, held = [], np.zeros(len(FOLDS))
    for ifd, f in enumerate(FOLDS):
        donor = [j for j in range(len(FOLDS)) if j != ifd]
        dw = W[donor]
        dscore = (S[:, donor] * dw).sum(axis=1)/dw.sum()
        best = int(np.argmin(np.round(dscore, 12)))
        # ties -> smaller alpha (argmin already returns first/smallest index)
        held[ifd] = S[best, ifd]
        rows.append(dict(heldout_fold=f, selected_alpha=float(ALPHAS[best]),
                         donor_wcv=float(dscore[best]), heldout_score=float(S[best, ifd]),
                         baseline_heldout=float(base_sc[ifd]),
                         heldout_delta=float(S[best, ifd]-base_sc[ifd])))
    nested_wcv = float(weighted_cv(held, W))
    return rows, nested_wcv, S

TARGETS = {}
par_c, perp_c = lofo_affine_residual(d_comb, BASIS_MAIN)
par_f, perp_f = lofo_affine_residual(corr,   BASIS_MAIN)
par_b, perp_b = lofo_affine_residual(d_btyd, BASIS_MAIN)
TARGETS["combined_full"]   = d_comb
TARGETS["combined_perp"]   = perp_c
TARGETS["fresh_full"]      = corr
TARGETS["fresh_perp"]      = perp_f
TARGETS["btyd_full"]       = d_btyd
TARGETS["btyd_perp"]       = perp_b

# controls: VOL placebo, same pipeline
hist = np.load(Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")/"artifacts"/"oof_FRESH_CONTRAST_MOE.npz", allow_pickle=False)
order = {c+"|"+str(u): i for i,(c,u) in enumerate(zip(hist["cutoff"].astype(str), hist["uid"]))}
perm = np.array([order[f+"|"+str(u)] for f,u in zip(fold, uid)])
vol = hist["vol_processed_nested"][perm].astype(float)
par_v, perp_v = lofo_affine_residual(vol, BASIS_MAIN)
TARGETS["vol_full_control"] = vol
TARGETS["vol_perp_control"] = perp_v

rows_out, nested_out = [], []
for name, dv in TARGETS.items():
    fixed = {}
    for a in ALPHAS:
        w, sc = wcv_of(z037 + a*dv)
        fixed[float(a)] = dict(wcv=w, delta=w-base_wcv, fold_deltas=(sc-base_sc).tolist(),
                               improved=int((sc < base_sc).sum()))
    nrows, nwcv, S = nested_alpha(dv)
    for r in nrows: r["target"] = name
    nested_out.extend(nrows)
    var_ret = float(np.var(dv)/np.var(TARGETS["fresh_full"] if "fresh" in name else
                                      (TARGETS["combined_full"] if "combined" in name else
                                       (TARGETS["btyd_full"] if "btyd" in name else TARGETS["vol_full_control"]))))
    hA = wcv_of(z037 + dv, side==0)[0] - wcv_of(z037, side==0)[0]
    hB = wcv_of(z037 + dv, side==1)[0] - wcv_of(z037, side==1)[0]
    r = dict(target=name, rms=float(np.sqrt(np.mean(dv**2))), mean=float(dv.mean()),
             variance_retained_vs_full=var_ret,
             nested_wcv=nwcv, nested_delta=nwcv-base_wcv,
             nested_alphas=[x["selected_alpha"] for x in nrows],
             delta_a1=fixed[1.0]["delta"], improved_a1=fixed[1.0]["improved"],
             latest_fold_delta_a1=fixed[1.0]["fold_deltas"][-1],
             fold_deltas_a1=fixed[1.0]["fold_deltas"],
             delta_a075=fixed[0.75]["delta"], delta_a05=fixed[0.5]["delta"],
             delta_a025=fixed[0.25]["delta"],
             half_A_delta_a1=hA, half_B_delta_a1=hB)
    rows_out.append(r)
    print(f"{name:20s} rms={r['rms']:.6f} nested_a={r['nested_alphas']} nested_delta={r['nested_delta']:+.9f} "
          f"a1_delta={r['delta_a1']:+.9f} folds={np.round(r['fold_deltas_a1'],9).tolist()} "
          f"halves=({hA:+.9f},{hB:+.9f})")

# ---------------------------------------------- basis sensitivity for FRESH perp
sens = {}
for tag, B in (("main", BASIS_MAIN), ("strict", BASIS_STRICT), ("wide", BASIS_WIDE)):
    _, pp = lofo_affine_residual(corr, B)
    w1, sc1 = wcv_of(z037 + pp)
    sens[tag] = dict(n_basis=len(B), rms=float(np.sqrt(np.mean(pp**2))),
                     variance_retained=float(np.var(pp)/np.var(corr)),
                     delta_a1=w1-base_wcv, folds=(sc1-base_sc).tolist(),
                     corr_with_main_perp=float(np.corrcoef(pp, perp_f)[0,1]))
    print(f"basis {tag:7s} n={len(B):2d} rms={sens[tag]['rms']:.6f} var_ret={sens[tag]['variance_retained']:.4f} "
          f"delta_a1={sens[tag]['delta_a1']:+.9f} corr_main={sens[tag]['corr_with_main_perp']:.4f}")

# ---------------------------------------------------- bootstrap for fresh_perp a=1
def bootstrap_delta(zv, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    users = np.unique(uid)
    o = np.argsort(uid, kind="stable"); us = uid[o]
    st = np.searchsorted(us, users, "left"); en = np.searchsorted(us, users, "right")
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(users), len(users))
        rows = np.concatenate([o[st[p]:en[p]] for p in pick])
        yb, fb = y[rows], fold[rows]
        s1, s2 = [], []
        for f in FOLDS:
            m = fb == f
            s1.append(calibrate_log_offset(yb[m], z037[rows][m])[1])
            s2.append(calibrate_log_offset(yb[m], zv[rows][m])[1])
        out[b] = weighted_cv(s2, W) - weighted_cv(s1, W)
    return out

print("bootstrapping fresh_perp (alpha=1) and combined_perp (alpha=1) ...", flush=True)
bs_f = bootstrap_delta(z037 + perp_f, 500, 42)
bs_c = bootstrap_delta(z037 + perp_c, 500, 42)
boot = {
 "fresh_perp_a1": dict(point=float(wcv_of(z037+perp_f)[0]-base_wcv), p02_5=float(np.quantile(bs_f,.025)),
                       p10=float(np.quantile(bs_f,.10)), p90=float(np.quantile(bs_f,.90)),
                       p97_5=float(np.quantile(bs_f,.975)), p_lt_0=float((bs_f<0).mean()),
                       sd=float(bs_f.std(ddof=1))),
 "combined_perp_a1": dict(point=float(wcv_of(z037+perp_c)[0]-base_wcv), p02_5=float(np.quantile(bs_c,.025)),
                       p10=float(np.quantile(bs_c,.10)), p90=float(np.quantile(bs_c,.90)),
                       p97_5=float(np.quantile(bs_c,.975)), p_lt_0=float((bs_c<0).mean()),
                       sd=float(bs_c.std(ddof=1))),
}
print(json.dumps(boot, indent=1))

with open(OUT/"oof_component_metrics.csv","w",newline="",encoding="utf-8") as f:
    cols = ["target","rms","mean","variance_retained_vs_full","nested_wcv","nested_delta",
            "nested_alphas","delta_a1","improved_a1","latest_fold_delta_a1","fold_deltas_a1",
            "delta_a075","delta_a05","delta_a025","half_A_delta_a1","half_B_delta_a1"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows_out: w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in cols})
with open(OUT/"oof_nested_alpha.csv","w",newline="",encoding="utf-8") as f:
    cols = ["target","heldout_fold","selected_alpha","donor_wcv","heldout_score","baseline_heldout","heldout_delta"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in nested_out: w.writerow({k: r[k] for k in cols})
np.savez_compressed(OUT/"_oof_perp.npz", perp_fresh=perp_f, perp_comb=perp_c, perp_btyd=perp_b,
                    perp_vol=perp_v, bs_fresh=bs_f, bs_comb=bs_c)
(OUT/"_p6.json").write_text(json.dumps(dict(basis_main=BASIS_MAIN, basis_sensitivity=sens,
                                            bootstrap=boot, baseline_wcv=base_wcv), indent=2), encoding="utf-8")
print("WROTE oof_component_metrics.csv, oof_nested_alpha.csv")
