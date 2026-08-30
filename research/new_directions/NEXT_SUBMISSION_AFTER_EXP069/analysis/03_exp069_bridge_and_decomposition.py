"""Phase 4.2b - decomposition + preprocessing-bridge independent checks."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, weighted_cv

GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
E69 = REPO/"research"/"new_directions"/"EXP069_BTYD05_FRESH1_PROD"
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
FOLDS = ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]
W = np.array([1.,2.,4.,8.])

ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
c69 = pq.read_table(E69/"btyd05_fresh1_OOF.parquet").to_pandas()
f69 = pq.read_table(E69/"fresh_conditional_OOF.parquet").to_pandas()
fold = ao["fold"].astype(str).to_numpy(); uid = ao["user_id"].to_numpy()
y = ao["target"].to_numpy().astype(float)
z037 = np.log1p(ao["pred_exp037"].to_numpy().astype(float))
zbtyd = np.log1p(ao["pred_btyd"].to_numpy().astype(float))
zbtyd05 = np.log1p(ao["pred_btyd05"].to_numpy().astype(float))
corr_f = f69["correction"].to_numpy().astype(float)
zc = c69["z_predict"].to_numpy().astype(float)
zbase_c = c69["z_base"].to_numpy().astype(float)
corr_c = c69["correction"].to_numpy().astype(float)

out = {}
print("== combined OOF decomposition ==")
print("  max|z_predict_comb - (z_btyd05 + corr_fresh)| =",
      float(np.max(np.abs(zc - (zbtyd05 + corr_f)))))
print("  max|z_base_comb - z_exp037| =", float(np.max(np.abs(zbase_c - z037))))
print("  max|corr_comb - (0.05*(z_btyd - z_exp037) + corr_fresh)| =",
      float(np.max(np.abs(corr_c - (0.05*(zbtyd - z037) + corr_f)))))
out["combined_equals_btyd05_plus_fresh_maxerr"] = float(np.max(np.abs(zc - (zbtyd05 + corr_f))))
out["combined_zbase_is_exp037_maxerr"] = float(np.max(np.abs(zbase_c - z037)))
out["combined_corr_is_btyd_plus_fresh_maxerr"] = float(np.max(np.abs(corr_c - (0.05*(zbtyd - z037) + corr_f))))

hist = np.load(OLD/"artifacts"/"oof_FRESH_CONTRAST_MOE.npz", allow_pickle=False)
h_key = {c + "|" + str(u): i for c, u in zip(hist["cutoff"].astype(str), hist["uid"]) for i in [0]}
order = {c + "|" + str(u): i for i, (c, u) in enumerate(zip(hist["cutoff"].astype(str), hist["uid"]))}
perm = np.array([order[f + "|" + str(u)] for f, u in zip(fold, uid)])
d_raw = hist["d_fresh"][perm].astype(float)
d_raw_vol = hist["d_vol"][perm].astype(float)

prep = json.loads((E69/"preprocessing_parameters.json").read_text(encoding="utf-8"))
print("\n== per-fold LOFO clip parity (donor params from registered config) ==")
lofo_rows = []
for e in prep["lofo_emulation"]:
    f = e["fold"]; m = fold == f
    emu = np.clip(d_raw[m], e["q005"], e["q995"]) - e["center"]
    diff = emu - corr_f[m]
    row = dict(fold=f, rms=float(np.sqrt(np.mean(diff**2))), mean=float(diff.mean()),
               std=float(diff.std()), reported_rms=e["rms_difference_vs_saved"],
               reported_std=e["std_difference_vs_saved"])
    lofo_rows.append(row); print("  ", {k: (round(v,12) if isinstance(v,float) else v) for k,v in row.items()})
out["lofo_emulation_check"] = lofo_rows

def wcv_of(zv):
    sc=[]
    for f in FOLDS:
        m = fold==f
        _, s = calibrate_log_offset(y[m], zv[m]); sc.append(s)
    return float(weighted_cv(sc, W)), np.array(sc)

# bridge: apply per-fold donor params (= production bridge emulation)
corr_bridge = np.empty_like(corr_f)
for e in prep["lofo_emulation"]:
    m = fold == e["fold"]
    corr_bridge[m] = np.clip(d_raw[m], e["q005"], e["q995"]) - e["center"]
w_hist, s_hist = wcv_of(z037 + corr_f)
w_brid, s_brid = wcv_of(z037 + corr_bridge)
print("\n== bridge parity ==")
print("  historical FRESH wCV", repr(w_hist))
print("  bridge     FRESH wCV", repr(w_brid))
print("  bridge - historical  ", w_brid - w_hist)
out["bridge_wcv"] = w_brid; out["historical_wcv"] = w_hist
out["bridge_minus_historical"] = w_brid - w_hist

# global (production/TEST) preprocessing applied to OOF
corr_global = np.clip(d_raw, prep["q005"], prep["q995"]) - prep["center"]
w_glob, s_glob = wcv_of(z037 + corr_global)
print("  GLOBAL-params FRESH wCV (leaky diagnostic only)", repr(w_glob), "delta vs hist", w_glob-w_hist)
out["global_params_wcv_diagnostic"] = w_glob

# level statistics of corrections
print("\n== correction level statistics ==")
for nm, v in (("oof_nested_corr", corr_f), ("oof_bridge_corr", corr_bridge), ("oof_global_corr", corr_global)):
    print(f"  {nm:18s} mean={v.mean():+.6f} std={v.std():.6f} min={v.min():+.4f} max={v.max():+.4f}")
    out[nm+"_stats"] = dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()))
for f in FOLDS:
    m = fold==f
    print(f"    fold {f}: nested mean {corr_f[m].mean():+.6f} std {corr_f[m].std():.6f}")

(OUT/"_p42b_bridge.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWROTE", OUT/"_p42b_bridge.json")
