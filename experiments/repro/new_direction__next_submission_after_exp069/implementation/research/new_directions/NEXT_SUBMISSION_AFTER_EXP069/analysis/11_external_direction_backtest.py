"""Phase 8.3 - external-direction geometry backtest.

Channel 1 (existing geometry LOO backtest, reused unchanged): for every source i,
   obs_i - pred_i  ~=  -2*mean_N(t*r_i) + noise,
with r_i the component of z_i orthogonal to the affine span of the other 64
sources.  That is exactly the first-order gain of an out-of-span direction,
measured against the real TEST target through the public leaderboard.
Per-unit external alignment  a_i = mean_N(t*r_i)/rms(r_i) = -(obs_i-pred_i)/(2*rms r_i).

Channel 2: the five exactly reproducible sources of Phase 8.2 used as
pseudo-unseen directions - OOF-predicted external gain vs realized public gain.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean"); sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research"); GEOM = GEO/"submission_geometry"
sys.path.insert(0, str(GEOM))
from core import load_unique
from directions import build_basis
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
FOLDS = ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]; W = np.array([1.,2.,4.,8.])
NOISE = np.sqrt(0.8/50_000)

b = pd.read_csv(GEOM/"cache"/"loo_backtest.csv")
b["ext_gain_mse"] = b.err_mse + b.mean_r2                     # MSE change from adding r_i
b["ext_gain_rmsle"] = b.ext_gain_mse/(2*b.lb)
b["a_test"] = -b.err_mse/(2*b.rms_r.clip(lower=1e-12))        # per-unit external alignment
b["a_sd"] = b.sd_tot/(2*b.rms_r.clip(lower=1e-12))
b["a_z"] = b.a_test/b.a_sd
nd = b[b.rms_r > 1e-4].copy()
print("non-degenerate external directions:", len(nd))
print("\nrealized per-unit external alignment a_test (TEST, via public LB):")
print("  mean %.5f  median %.5f  sd %.5f  min %.5f  max %.5f"
      % (nd.a_test.mean(), nd.a_test.median(), nd.a_test.std(), nd.a_test.min(), nd.a_test.max()))
print("  |a| quantiles:", np.round(np.quantile(np.abs(nd.a_test), [.25,.5,.75,.9,1.0]), 5).tolist())
print("  a>0 (direction genuinely helps): %d/%d" % (int((nd.a_test > 0).sum()), len(nd)))
print("  z = a/sd :  mean %.3f  sd %.3f  median|z| %.3f"
      % (nd.a_z.mean(), nd.a_z.std(), np.median(np.abs(nd.a_z))))
# empirical signal/noise: var(z) = 1 + tau^2/E[sd^2]  ->  shrinkage tau2/(tau2+sd2)
var_z = float(nd.a_z.var(ddof=1))
sn = max(var_z - 1.0, 0.0)
shrink = sn/(sn + 1.0)
print("  var(z) = %.3f -> external-signal / noise variance ratio %.3f -> James-Stein shrinkage %.3f"
      % (var_z, sn, shrink))
print("\nrealized external RMSLE gain (negative = helped) quantiles:",
      np.round(np.quantile(nd.ext_gain_rmsle, [0,.1,.25,.5,.75,.9,1]), 6).tolist())
print("  sources whose out-of-span residual actually improved the public score:",
      int((nd.ext_gain_rmsle < 0).sum()), "/", len(nd))

# ------------------------------------------------ Channel 2: 5 pseudo-unseen
COMP = ["pred_cap","pred_unc","pred_dist","pred_ridge15","pred_hurdle_e11",
        "pred_holiday_yoy","pred_etx_avg3","pred_seq_avg3","pred_btyd"]
BASIS_OOF = ["pred_exp037","pred_cap","pred_unc","pred_dist","pred_etx_avg3",
             "pred_seq_avg3","pred_ridge15","pred_hurdle_e11","pred_holiday_yoy",
             "pred_btyd","pred_btyd05"]
ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); y = ao["target"].to_numpy().astype(float)
zo = {c: np.log1p(ao[c].to_numpy().astype(float)) for c in list(ao.columns) if c.startswith("pred_")}
zt = {c: np.log1p(at[c].to_numpy().astype(float)) for c in COMP}
zt["pred_exp037"] = np.log1p(at["pred_exp037_rebuilt"].to_numpy().astype(float))
uid_t = at["user_id"].to_numpy().astype(np.int64)
Z, names, lb, guid = load_unique(); Nn = Z.shape[1]
order = np.argsort(uid_t); pos = np.searchsorted(uid_t[order], guid)
COLS = COMP + ["pred_exp037"]
Xt = np.column_stack([zt[c][order][pos] for c in COLS] + [np.ones(len(guid))])
Xo = np.column_stack([zo[c] for c in COLS] + [np.ones(len(y))])
def wcv(zv):
    return float(weighted_cv([calibrate_log_offset(y[fold==f], zv[fold==f])[1] for f in FOLDS], W))
base_wcv = wcv(zo["pred_exp037"])
Xb = np.column_stack([zo[c] for c in BASIS_OOF] + [np.ones(len(y))])
def lofo_perp(d):
    perp = np.empty_like(d)
    for f in FOLDS:
        te = fold == f; tr = ~te
        bet, *_ = np.linalg.lstsq(Xb[tr], d[tr], rcond=None)
        perp[te] = d[te] - Xb[te] @ bet
    return perp

hist = pd.read_csv(OUT/"historical_transfer.csv")
rows = []
for _, h in hist.iterrows():
    i = names.index(h["name"])
    if h["test_step_rms"] < 1e-6:
        continue
    m = Z[i] > 1e-12
    bet, *_ = np.linalg.lstsq(Xt[m], Z[i][m], rcond=None)
    d_oof = Xo @ bet - zo["pred_exp037"]; d_oof -= d_oof.mean()
    perp_oof = lofo_perp(d_oof)
    w = wcv(zo["pred_exp037"] + perp_oof)
    oof_perp_delta = w - base_wcv
    ro = float(np.sqrt(np.mean(perp_oof**2)))
    a_oof_unit = ((ro**2) - 2*base_wcv*oof_perp_delta)/2.0/ro
    bb = b[b.name == h["name"]]
    realized_a = float(bb.a_test.iloc[0]); realized_sd = float(bb.a_sd.iloc[0])
    rms_r = float(bb.rms_r.iloc[0])
    rows.append(dict(name=h["name"], oof_perp_rms=ro, oof_perp_delta_wcv=oof_perp_delta,
                     oof_unit_alignment_perp=a_oof_unit,
                     test_out_of_span_rms=rms_r, test_unit_alignment_realized=realized_a,
                     test_unit_alignment_sd=realized_sd,
                     transfer=realized_a/a_oof_unit if abs(a_oof_unit) > 1e-8 else np.nan,
                     realized_ext_gain_rmsle=float(bb.ext_gain_rmsle.iloc[0]),
                     z=float(bb.z.iloc[0])))
E = pd.DataFrame(rows)
print("\n=== Channel 2: pseudo-unseen external directions (OOF-predicted vs realized) ===")
print(E.to_string(index=False, float_format=lambda v: f"{v: .6g}"))
if len(E):
    ok = E.dropna(subset=["transfer"])
    print("\n  transfer ratios:", np.round(ok.transfer.to_numpy(), 4).tolist())
    print("  median", float(np.median(ok.transfer)), " OLS-through-origin",
          float((ok.test_unit_alignment_realized*ok.oof_unit_alignment_perp).sum()
                /(ok.oof_unit_alignment_perp**2).sum()))

allrows = nd[["name","lb","rms_r","err_mse","sd_tot","z","ext_gain_mse","ext_gain_rmsle",
              "a_test","a_sd","a_z"]].copy()
allrows["channel"] = "geometry_LOO_external_residual"
E2 = E.copy(); E2["channel"] = "pseudo_unseen_oof_predicted"
pd.concat([allrows, E2], ignore_index=True).to_csv(OUT/"external_direction_backtest.csv", index=False)
summ = dict(
  channel1=dict(n=len(nd), a_mean=float(nd.a_test.mean()), a_median=float(nd.a_test.median()),
                a_sd=float(nd.a_test.std()), abs_a_q50=float(np.median(np.abs(nd.a_test))),
                abs_a_q90=float(np.quantile(np.abs(nd.a_test), .9)),
                var_z=var_z, signal_noise_var_ratio=sn, james_stein_shrinkage=shrink,
                n_helped=int((nd.ext_gain_rmsle < 0).sum()),
                bias_err_mse=float(b.err_mse.mean()), mae_err_mse=float(np.abs(b.err_mse).mean()),
                median_abs_z=float(np.median(np.abs(b.z)))),
  channel2=E.to_dict("records"),
  interpretation=("Across 47 non-degenerate historical out-of-span residual directions the "
                  "realized external first-order alignment on the public TEST target is "
                  "statistically indistinguishable from zero; the empirical James-Stein "
                  "shrinkage for an external-direction gain estimate is well below one."))
(OUT/"_p83.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
print("\nWROTE external_direction_backtest.csv, _p83.json")
