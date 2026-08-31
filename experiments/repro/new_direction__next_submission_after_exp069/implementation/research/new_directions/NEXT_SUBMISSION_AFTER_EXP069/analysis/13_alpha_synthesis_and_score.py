"""Phases 8.4 and 11 - alpha synthesis and public-LB score estimate.

Exact local model.  For an anchor z0 with residual r = t - z0 and an added
direction D = alpha*d:
    MSE(z0+D) - MSE(z0) = -2*mean(r*D) + mean(D^2)
    dRMSLE              ~= [ -2*A + Q ] / (2*S0)
A = mean(r*D) is the only unknown; A = alpha * rms(d) * a_test and
a_test = tau * a_oof, with a_oof measured honestly on canonical OOF against
pred_exp037.  Everything else is exactly computable.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean"); sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
S0 = 1.6466079084
S_OOF = 1.747509862493216
FOLDS = ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]; W = np.array([1.,2.,4.,8.])
NOISE = np.sqrt(0.8/50_000)

# ---- OOF alignment of the orthogonal FRESH direction ---------------------
p6 = json.loads((OUT/"_p6.json").read_text(encoding="utf-8"))
oofm = pd.read_csv(OUT/"oof_component_metrics.csv")
row = oofm[oofm.target == "fresh_perp"].iloc[0]
rms_oof = float(row["rms"]); dwcv = float(row["delta_a1"])
Q_oof = rms_oof**2
A_oof = (Q_oof - 2*S_OOF*dwcv)/2.0
a_oof = A_oof/rms_oof
print("OOF orthogonal FRESH: rms %.6f  dwCV(a=1) %+.9f  A_oof %.6e  a_oof %.6f"
      % (rms_oof, dwcv, A_oof, a_oof))
bs = np.load(OUT/"_oof_perp.npz")["bs_fresh"]
a_boot = ((Q_oof - 2*S_OOF*bs)/2.0)/rms_oof
print("  a_oof bootstrap: mean %.6f sd %.6f  2.5%% %.6f  97.5%% %.6f"
      % (a_boot.mean(), a_boot.std(ddof=1), np.quantile(a_boot,.025), np.quantile(a_boot,.975)))

# ---- exact TEST-side constants -------------------------------------------
pz = np.load(OUT/"_perp_directions.npz")
d = pz["d_fresh_perp"]; z_inc = pz["z_inc"]
at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
z037t = np.log1p(at["pred_exp037_rebuilt"].to_numpy().astype(float))   # same order (verified)
anchor_term = float(np.mean(d*(z_inc - z037t)))
print("\nanchor-invariance check  mean(d_perp*(z_inc - z_exp037_test)) = %.3e  "
      "(exactly 0 would mean the first-order gain is identical for both anchors)" % anchor_term)
print("  bound rms(d)*rms(z_inc-z037) = %.3e"
      % (np.sqrt(np.mean(d**2))*np.sqrt(np.mean((z_inc-z037t)**2))))

rows = []
for alpha in (0.25, 0.50, 0.75, 1.00):
    z = z_inc + alpha*d
    pred = np.maximum(np.expm1(z), 0.0)
    dz_eff = np.log1p(pred) - z_inc                     # after the non-negativity floor
    Qe = float(np.mean(dz_eff**2)); rms_e = float(np.sqrt(Qe))
    lin = float(np.mean(dz_eff*d))                      # effective projection onto d
    rows.append(dict(alpha=alpha, rms_effective=rms_e, Q_eff=Qe,
                     mean_shift=float(dz_eff.mean()),
                     lin_coeff=lin,                     # diagnostic: mean(dz_eff*d)
                     second_order=Qe/(2*S0),
                     public_noise_sd=rms_e*NOISE))
E = pd.DataFrame(rows)
print("\neffective step after the zero floor:")
print(E.to_string(index=False, float_format=lambda v: f"{v: .6g}"))

# ---- tau prior -----------------------------------------------------------
TAU_MU, TAU_SD = 0.20, 0.30
BASIS_REL_SD = 0.14        # 80% source-subsample median drift fraction (basis_stability.csv)
rng = np.random.default_rng(20260826)
NS = 400_000
tau = rng.normal(TAU_MU, TAU_SD, NS)
a_o = rng.choice(a_boot, NS, replace=True)              # OOF estimation uncertainty
basis = rng.normal(1.0, BASIS_REL_SD, NS)

out = {}
for _, r in E.iterrows():
    A = tau*a_o*basis*r.rms_effective   # A = a_test * rms(effective step)
    delta = (-2*A + r.Q_eff)/(2*S0) + rng.normal(0.0, r.public_noise_sd, NS)
    lb = S0 + delta
    out[float(r.alpha)] = dict(
        expected_gain=float(delta.mean()), sd=float(delta.std()),
        expected_lb=float(lb.mean()),
        p50=[float(np.quantile(lb,.25)), float(np.quantile(lb,.75))],
        p80=[float(np.quantile(lb,.10)), float(np.quantile(lb,.90))],
        p_beat=float((delta < 0).mean()),
        p_gain_ge_1e4=float((delta <= -1e-4).mean()),
        p_loss_ge_1e4=float((delta >= 1e-4).mean()),
        second_order_floor=float(r.Q_eff/(2*S0)),
        public_noise_sd=float(r.public_noise_sd),
        deterministic_if_tau=dict(
            tau_minus_0344=float((-2*(-0.344)*a_oof*r.rms_effective + r.Q_eff)/(2*S0)),
            tau_0=float(r.Q_eff/(2*S0)),
            tau_0_20=float((-2*0.20*a_oof*r.rms_effective + r.Q_eff)/(2*S0)),
            tau_0_356=float((-2*0.356*a_oof*r.rms_effective + r.Q_eff)/(2*S0)),
            tau_1=float((-2*1.0*a_oof*r.rms_effective + r.Q_eff)/(2*S0))))
print("\n=== alpha synthesis (tau ~ N(%.2f, %.2f), basis rel sd %.2f) ===" % (TAU_MU, TAU_SD, BASIS_REL_SD))
print("%6s %13s %11s %13s %9s %9s %9s" % ("alpha","E[gain]","sd","E[LB]","P(beat)","P(+1e-4)","P(-1e-4)"))
for a, v in out.items():
    print("%6.2f %+13.3e %11.3e %13.7f %9.3f %9.3f %9.3f"
          % (a, v["expected_gain"], v["sd"], v["expected_lb"], v["p_beat"],
             v["p_gain_ge_1e4"], v["p_loss_ge_1e4"]))
print("\ndeterministic LB delta under fixed tau scenarios:")
print("%6s %12s %12s %12s %12s %12s" % ("alpha","tau=-0.344","tau=0","tau=0.20","tau=0.356","tau=1.0"))
for a, v in out.items():
    dd = v["deterministic_if_tau"]
    print("%6.2f %+12.3e %+12.3e %+12.3e %+12.3e %+12.3e"
          % (a, dd["tau_minus_0344"], dd["tau_0"], dd["tau_0_20"], dd["tau_0_356"], dd["tau_1"]))
best = max(out, key=lambda a: -out[a]["expected_gain"])
print("\nalpha maximising expected gain:", best,
      " alpha maximising P(beat incumbent):", max(out, key=lambda a: out[a]["p_beat"]))
r1 = E[E.alpha==1].iloc[0]
k1 = a_oof*float(r1.rms_effective)/S0          # first-order gain per unit alpha per unit tau
k2 = float(r1.Q_eff)/(2*S0)                    # second-order penalty at alpha=1
print("k1 (first-order per alpha per tau) = %.6e ; k2 (second order at alpha=1) = %.6e" % (k1, k2))
print("continuous optimum alpha* = tau*k1/(2*k2):")
for t in (0.0, 0.10, 0.20, 0.30, 0.356, 0.50, 1.0):
    print("   tau=%.3f -> alpha* = %.3f" % (t, t*k1/(2*k2)))
print("break-even tau at each alpha (dLB = 0):")
for a in (0.25,0.5,0.75,1.0):
    print("   alpha=%.2f -> tau_breakeven = %.4f" % (a, a*k2/k1))
payload_extra = dict(k1=k1, k2=k2)

payload = dict(model=("dRMSLE = (-2*A + Q)/(2*S0); A = tau * a_oof * basis * mean(dz_eff*d); "
                      "a_oof from canonical OOF vs pred_exp037 only"),
               S0=S0, a_oof=a_oof, a_oof_boot_sd=float(a_boot.std(ddof=1)),
               tau_prior=dict(mean=TAU_MU, sd=TAU_SD,
                              justification=("channel B (5 exact CV->LB pairs) OLS -0.344 but confounded "
                                             "by winner's curse on the anchor and all in-span; after a 1-sigma "
                                             "anchor correction the 5 cases are 2 up / 2 down, i.e. tau~0. "
                                             "channel C (47 out-of-span residuals) gives an empirical "
                                             "James-Stein shrinkage 0.356 for an external gain estimate. "
                                             "Mechanistic prior is positive: the FRESH contrast is trained on "
                                             "13 EXTRA cutoffs running to 2026-01-14 and its OOF orthogonal "
                                             "gain increases monotonically with fold recency.")),
               basis_relative_sd=BASIS_REL_SD,
               effective_step=E.to_dict("records"), per_alpha=out,
               anchor_invariance_term=anchor_term, k1=k1, k2=k2)
(OUT/"score_estimate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("\nWROTE score_estimate.json")
