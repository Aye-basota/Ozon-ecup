"""Phase 8.2 (final) - historical CV -> public-LB transfer with a first-order
alignment decomposition.

For a direction d added to an anchor z0 with residual r = t - z0:
    MSE(z0+d) - MSE(z0) = -2*mean(r*d) + mean(d^2)
    RMSLE change        ~= [ -2*A + Q ] / (2*score),   A = mean(r*d), Q = mean(d^2)
so  A = ( Q - 2*score*dRMSLE ) / 2 .
Per-unit alignment a = A / rms(d) is scale free and is the quantity that must
transfer from OOF to TEST for a direction to keep helping.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pandas as pd

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean"); sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research"); GEOM = GEO/"submission_geometry"
sys.path.insert(0, str(GEOM))
from core import load_unique
from directions import build_basis
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
FOLDS = ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]; W = np.array([1.,2.,4.,8.])
NOISE = np.sqrt(0.8/50_000)

COMP = ["pred_cap","pred_unc","pred_dist","pred_ridge15","pred_hurdle_e11",
        "pred_holiday_yoy","pred_etx_avg3","pred_seq_avg3","pred_btyd"]
ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); y = ao["target"].to_numpy().astype(float)
zo = {c: np.log1p(ao[c].to_numpy().astype(float)) for c in COMP}
zo["pred_exp037"] = np.log1p(ao["pred_exp037"].to_numpy().astype(float))
zt = {c: np.log1p(at[c].to_numpy().astype(float)) for c in COMP}
zt["pred_exp037"] = np.log1p(at["pred_exp037_rebuilt"].to_numpy().astype(float))
uid_t = at["user_id"].to_numpy().astype(np.int64)
Z, names, lb, guid = load_unique()
order = np.argsort(uid_t); pos = np.searchsorted(uid_t[order], guid)
COLS = COMP + ["pred_exp037"]
Xt = np.column_stack([zt[c][order][pos] for c in COLS] + [np.ones(len(guid))])
Xo = np.column_stack([zo[c] for c in COLS] + [np.ones(len(y))])
sv = np.linalg.svd(Xt - Xt.mean(0), compute_uv=False)
print("TEST design singular values:", np.round(sv, 3), " cond =", float(sv[0]/sv[-1]))

def wcv(zv):
    sc = [calibrate_log_offset(y[fold==f], zv[fold==f])[1] for f in FOLDS]
    return float(weighted_cv(sc, W)), np.array(sc)
base_wcv, base_sc = wcv(zo["pred_exp037"])
z0t = zt["pred_exp037"][order][pos]
LB0 = float(lb[names.index("submission_STRONGEST_CURRENT.csv")])
Nn = Z.shape[1]

rows = []
for i, nm in enumerate(names):
    zs = Z[i]; m = zs > 1e-12
    beta, *_ = np.linalg.lstsq(Xt[m], zs[m], rcond=None)
    resid = float(np.sqrt(np.mean((zs[m]-(Xt@beta)[m])**2)))
    if resid > 2e-4:
        continue
    zoof = Xo @ beta
    w, sc = wcv(zoof)
    d_oof = zoof - zo["pred_exp037"]; d_oof = d_oof - d_oof.mean()   # level absorbed by calibrator
    d_tst = zs - z0t
    Qo, Qt = float(np.mean(d_oof**2)), float(np.mean(d_tst**2))
    dO, dL = w - base_wcv, float(lb[i]) - LB0
    Ao = (Qo - 2*base_wcv*dO)/2.0
    At = (Qt - 2*LB0*dL)/2.0
    ro, rt = float(np.sqrt(Qo)), float(np.sqrt(Qt))
    keep = [j for j in range(len(names)) if j != i]
    Zk = Z[keep]; zc_ = Zk.mean(axis=0)
    _, Phi_k, _, _, _ = build_basis(Zk, 0, tol=1e-12)
    dd = Z[i] - zc_; res = dd - (Phi_k @ dd/Nn) @ Phi_k
    ofrac = float(np.sqrt(np.mean(res**2))/np.sqrt(np.mean(dd**2)))
    rows.append(dict(name=nm, public_lb=float(lb[i]), affine_resid_rms=resid,
        oof_wcv=w, oof_delta_wcv=dO, lb_delta=dL,
        oof_step_rms_centered=ro, test_step_rms=rt,
        oof_alignment_A=Ao, test_alignment_A=At,
        oof_unit_alignment=Ao/ro if ro>0 else np.nan,
        test_unit_alignment=At/rt if rt>0 else np.nan,
        alignment_transfer=(At/rt)/(Ao/ro) if (ro>0 and rt>0 and abs(Ao/ro)>1e-6) else np.nan,
        second_order_lb_penalty=Qt/(2*LB0),
        first_order_lb_gain=-2*At/(2*LB0),
        lb_noise_sd=rt*NOISE, lb_delta_over_noise=dL/(rt*NOISE) if rt>0 else np.nan,
        orth_fraction_vs_other64=ofrac))
P = pd.DataFrame(rows).sort_values("oof_delta_wcv")
pd.set_option("display.width", 260)
print("\n=== exact CV->LB pairs (anchor EXP-037 / STRONGEST_CURRENT public 1.6496572) ===")
print(P[["name","public_lb","oof_delta_wcv","lb_delta","oof_step_rms_centered","test_step_rms",
         "oof_unit_alignment","test_unit_alignment","alignment_transfer",
         "second_order_lb_penalty","first_order_lb_gain","lb_noise_sd","orth_fraction_vs_other64"]]
      .to_string(index=False, float_format=lambda v: f"{v: .6g}"))

ok = P[np.abs(P.oof_unit_alignment) > 1e-4]
print("\nunit-alignment transfer ratios:", np.round(ok.alignment_transfer.to_numpy(), 4).tolist())
print("  median", float(np.median(ok.alignment_transfer)), " mean", float(ok.alignment_transfer.mean()),
      " min", float(ok.alignment_transfer.min()), " max", float(ok.alignment_transfer.max()))
print("  n negative (sign inversion):", int((ok.alignment_transfer < 0).sum()), "of", len(ok))
num = float((ok.test_unit_alignment*ok.oof_unit_alignment).sum())
den = float((ok.oof_unit_alignment**2).sum())
print("  OLS-through-origin (unit alignment):", num/den)
# leave-one-out robustness of that OLS estimate
loo = []
for k in range(len(ok)):
    s = ok.drop(ok.index[k])
    loo.append(float((s.test_unit_alignment*s.oof_unit_alignment).sum()/(s.oof_unit_alignment**2).sum()))
print("  leave-one-case-out OLS estimates:", np.round(loo, 4).tolist())
print("  wCV-delta OLS-through-origin:",
      float((ok.lb_delta*ok.oof_delta_wcv).sum()/(ok.oof_delta_wcv**2).sum()))

P.to_csv(OUT/"historical_transfer.csv", index=False)
summ = dict(pairs=int(len(P)), estimable=int(len(ok)),
            unit_alignment_transfer=dict(
                values=ok.alignment_transfer.tolist(), names=ok.name.tolist(),
                median=float(np.median(ok.alignment_transfer)),
                mean=float(ok.alignment_transfer.mean()),
                ols_through_origin=num/den, loo_ols=loo,
                n_sign_inversions=int((ok.alignment_transfer < 0).sum())),
            note="LEVEL_MINUS_006 has zero OOF delta by construction (fold calibrator removes level) "
                 "and its entire +0.001083 public loss equals its own second-order penalty "
                 "Q/(2*score); this confirms the TEST level of EXP-037/STRONGEST_CURRENT is "
                 "essentially optimal and validates the decomposition.")
(OUT/"_p82.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
print("\nLEVEL_MINUS_006 check: lb_delta", float(P[P.name=='submission_LEVEL_MINUS_006.csv'].lb_delta.iloc[0]),
      " pure second-order penalty", float(P[P.name=='submission_LEVEL_MINUS_006.csv'].second_order_lb_penalty.iloc[0]))
print("WROTE historical_transfer.csv, _p82.json")
