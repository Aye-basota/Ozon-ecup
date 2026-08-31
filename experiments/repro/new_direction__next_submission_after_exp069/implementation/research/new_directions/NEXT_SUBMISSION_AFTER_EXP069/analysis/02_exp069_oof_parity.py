"""Phase 4.2 - independent canonical OOF verification of EXP069.

Uses the clean repo's canonical src.metrics evaluator (NOT the EXP069 private copy)
and recomputes every reported quantity from the aligned banks + saved vectors.
"""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
sys.path.insert(0, str(REPO))
from src.metrics import calibrate_log_offset, rmsle_z, weighted_cv  # noqa: E402

GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
E69 = REPO / "research" / "new_directions" / "EXP069_BTYD05_FRESH1_PROD"
OUT = REPO / "research" / "new_directions" / "NEXT_SUBMISSION_AFTER_EXP069"
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
W = np.array([1.0, 2.0, 4.0, 8.0])

res = {}

# ---------------------------------------------------------------- load banks
ao = pq.read_table(GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet").to_pandas()
c69 = pq.read_table(E69/"btyd05_fresh1_OOF.parquet").to_pandas()
f69 = pq.read_table(E69/"fresh_conditional_OOF.parquet").to_pandas()

print("aligned OOF rows", len(ao), "combined", len(c69), "fresh", len(f69))

# ---------------------------------------------------------------- key checks
def keycheck(df, tag):
    k = list(zip(df["fold"].astype(str).to_numpy(), df["user_id"].to_numpy()))
    out = dict(rows=len(df), unique_keys=len(set(k)))
    sizes = df.groupby(df["fold"].astype(str)).size().to_dict()
    out["fold_sizes"] = {f: int(sizes.get(f, 0)) for f in FOLDS}
    print(tag, out)
    return out

res["keys_aligned_oof"] = keycheck(ao, "ALIGNED_OOF")
res["keys_exp069_combined"] = keycheck(c69, "EXP069_COMBINED_OOF")
res["keys_exp069_fresh"] = keycheck(f69, "EXP069_FRESH_OOF")

# row-order identity
same_order_comb = bool(np.array_equal(ao["user_id"].to_numpy(), c69["user_id"].to_numpy())
                       and np.array_equal(ao["fold"].astype(str).to_numpy(), c69["fold"].astype(str).to_numpy()))
same_order_fresh = bool(np.array_equal(ao["user_id"].to_numpy(), f69["user_id"].to_numpy())
                        and np.array_equal(ao["fold"].astype(str).to_numpy(), f69["fold"].astype(str).to_numpy()))
res["row_order_identical_combined"] = same_order_comb
res["row_order_identical_fresh"] = same_order_fresh
print("row order identical:", same_order_comb, same_order_fresh)
assert same_order_comb and same_order_fresh, "row order mismatch - would need reindexing"

fold = ao["fold"].astype(str).to_numpy()
uid = ao["user_id"].to_numpy()
y = ao["target"].to_numpy().astype(np.float64)

# targets
res["target_max_abs_diff_combined"] = float(np.max(np.abs(y - c69["target"].to_numpy().astype(np.float64))))
res["target_max_abs_diff_fresh"] = float(np.max(np.abs(y - f69["target"].to_numpy().astype(np.float64))))
print("target diffs", res["target_max_abs_diff_combined"], res["target_max_abs_diff_fresh"])

# ---------------------------------------------------------- EXP037 rebuild
z = {c: np.log1p(ao[c].to_numpy().astype(np.float64)) for c in
     ["pred_exp037","pred_cap","pred_unc","pred_dist","pred_etx_avg3","pred_seq_avg3",
      "pred_seq_d3a_avg3","pred_ridge15","pred_hurdle_e11","pred_mhz_full","pred_holiday_yoy",
      "pred_block4_saf","pred_fresh_contrast","pred_btyd","pred_btyd05"]}

z_exp037 = z["pred_exp037"]
z_rebuild = (0.10*z["pred_cap"] + 0.20*z["pred_unc"] + 0.25*z["pred_dist"]
             + 0.225*z["pred_seq_avg3"] + 0.225*z["pred_etx_avg3"])
res["exp037_rebuild_max_log_error"] = float(np.max(np.abs(z_rebuild - z_exp037)))
print("EXP037 rebuild max log error", res["exp037_rebuild_max_log_error"])

z_btyd05_rebuild = 0.95*z_exp037 + 0.05*z["pred_btyd"]
res["btyd05_rebuild_max_log_error"] = float(np.max(np.abs(z_btyd05_rebuild - z["pred_btyd05"])))
print("BTYD05 rebuild max log error", res["btyd05_rebuild_max_log_error"])

# ----------------------------------------------------- EXP069 saved vectors
z_base_c = c69["z_base"].to_numpy().astype(np.float64)
corr_c   = c69["correction"].to_numpy().astype(np.float64)
z_pred_c = c69["z_predict"].to_numpy().astype(np.float64)
z_base_f = f69["z_base"].to_numpy().astype(np.float64)
corr_f   = f69["correction"].to_numpy().astype(np.float64)
z_pred_f = f69["z_predict"].to_numpy().astype(np.float64)

res["combined_zbase_vs_btyd05_max_err"] = float(np.max(np.abs(z_base_c - z["pred_btyd05"])))
res["fresh_zbase_vs_exp037_max_err"]    = float(np.max(np.abs(z_base_f - z_exp037)))
res["combined_identity_max_err"]        = float(np.max(np.abs(z_pred_c - (z_base_c + corr_c))))
res["fresh_identity_max_err"]           = float(np.max(np.abs(z_pred_f - (z_base_f + corr_f))))
res["correction_same_in_both_max_err"]  = float(np.max(np.abs(corr_c - corr_f)))
print({k: res[k] for k in ["combined_zbase_vs_btyd05_max_err","fresh_zbase_vs_exp037_max_err",
                           "combined_identity_max_err","fresh_identity_max_err",
                           "correction_same_in_both_max_err"]})

# saved historical fresh vector
hist = np.load(OLD/"artifacts"/"oof_FRESH_CONTRAST_MOE.npz", allow_pickle=False)
h_uid, h_cut = hist["uid"], hist["cutoff"].astype(str)
key_ao = np.array([f + "|" + str(u) for f, u in zip(fold, uid)])
key_h  = np.array([c + "|" + str(u) for c, u in zip(h_cut, h_uid)])
order_h = {k: i for i, k in enumerate(key_h)}
perm = np.array([order_h[k] for k in key_ao])
res["historical_npz_covers_all_rows"] = bool(len(order_h) == len(key_ao))
fresh_proc_hist = hist["fresh_processed_nested"][perm].astype(np.float64)
vol_proc_hist   = hist["vol_processed_nested"][perm].astype(np.float64)
d_fresh_raw_hist = hist["d_fresh"][perm].astype(np.float64)
d_vol_raw_hist   = hist["d_vol"][perm].astype(np.float64)
res["saved_correction_vs_historical_max_err"] = float(np.max(np.abs(corr_f - fresh_proc_hist)))
print("saved correction vs historical fresh_processed_nested max err",
      res["saved_correction_vs_historical_max_err"])

# reconstruct the correction from the registered preprocessing recipe
prep = json.loads((E69/"preprocessing_parameters.json").read_text(encoding="utf-8"))
q005, q995, center = prep["q005"], prep["q995"], prep["center"]
corr_recipe = np.clip(d_fresh_raw_hist, q005, q995) - center
vol_recipe  = np.clip(d_vol_raw_hist,  q005, q995) - center
res["correction_recipe_vs_saved_max_err"] = float(np.max(np.abs(corr_recipe - corr_f)))
print("recipe-reconstructed correction vs saved (bridge, donor-frozen params):",
      res["correction_recipe_vs_saved_max_err"])

# --------------------------------------------------------------- evaluator
def wcv_of(zv, tag=None, gridcheck=False):
    scores, offs = [], []
    for f in FOLDS:
        m = fold == f
        off, sc = calibrate_log_offset(y[m], zv[m])
        if gridcheck:
            grid = np.linspace(off-0.5, off+0.5, 2001)
            best = min(rmsle_z(y[m], zv[m]+g) for g in grid)
            assert sc <= best + 1e-9, (tag, f, sc, best)
        scores.append(sc); offs.append(off)
    return float(weighted_cv(scores, W)), np.array(scores), np.array(offs)

CAND = {
 "EXP037":            z_exp037,
 "FRESH":             z_exp037 + corr_f,
 "BTYD05":            z["pred_btyd05"],
 "BTYD05_FRESH1":     z["pred_btyd05"] + corr_f,
 "VOL":               z_exp037 + vol_proc_hist,
}
base_wcv, base_scores, base_offs = wcv_of(z_exp037, "EXP037", gridcheck=True)
res["exp037_wcv"] = base_wcv
res["exp037_fold_scores"] = base_scores.tolist()
res["exp037_fold_offsets"] = base_offs.tolist()
print("EXP037 wCV", repr(base_wcv), "target 1.7475098625201952")

fixed = {}
for name, zv in CAND.items():
    w, sc, off = wcv_of(zv, name)
    fixed[name] = dict(wcv=w, delta=w-base_wcv, fold_scores=sc.tolist(),
                       fold_deltas=(sc-base_scores).tolist(),
                       improved_folds=int((sc < base_scores).sum()))
    print(f"{name:16s} wCV={w:.12f} delta={w-base_wcv:+.9f} folds={(sc-base_scores).round(9).tolist()}")
res["fixed"] = fixed
res["real_fresh_minus_vol_wcv"] = fixed["FRESH"]["delta"] - fixed["VOL"]["delta"]
print("REAL - VOL:", res["real_fresh_minus_vol_wcv"])

# ------------------------------------------------------------- user halves
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
res["half_sizes"] = {"A": int((side == 0).sum()), "B": int((side == 1).sum())}
# cross-check against saved user_side in fresh OOF
saved_side = f69["user_side"].to_numpy().astype(int)
res["splitmix_matches_saved_user_side"] = bool(np.array_equal(side, saved_side))
print("splitmix half sizes", res["half_sizes"], "matches saved user_side:",
      res["splitmix_matches_saved_user_side"])

def wcv_sub(zv, mask):
    scores = []
    for f in FOLDS:
        m = (fold == f) & mask
        _, sc = calibrate_log_offset(y[m], zv[m])
        scores.append(sc)
    return float(weighted_cv(scores, W))

halves = {}
for hname, hmask in (("A", side == 0), ("B", side == 1)):
    hb = wcv_sub(z_exp037, hmask)
    halves[hname] = {}
    for name, zv in CAND.items():
        halves[hname][name] = wcv_sub(zv, hmask) - hb
    halves[hname]["_baseline_wcv"] = hb
    print("half", hname, {k: round(v, 9) for k, v in halves[hname].items()})
res["user_halves"] = halves

# --------------------------------------------------------------- bootstrap
def bootstrap_delta(zv, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    users = np.unique(uid)
    # user-cluster bootstrap: resample users, evaluate wCV delta
    idx_by_user = {}
    order = np.argsort(uid, kind="stable")
    uid_s = uid[order]
    starts = np.searchsorted(uid_s, users, side="left")
    ends = np.searchsorted(uid_s, users, side="right")
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(users), len(users))
        rows = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        yb, fb = y[rows], fold[rows]
        sc_b, sc_c = [], []
        for f in FOLDS:
            m = fb == f
            _, s1 = calibrate_log_offset(yb[m], z_exp037[rows][m])
            _, s2 = calibrate_log_offset(yb[m], zv[rows][m])
            sc_b.append(s1); sc_c.append(s2)
        out[b] = weighted_cv(sc_c, W) - weighted_cv(sc_b, W)
    return out

print("running user-cluster bootstrap (500 reps) for BTYD05_FRESH1 ...", flush=True)
bs = bootstrap_delta(CAND["BTYD05_FRESH1"], 500, 42)
res["bootstrap_btyd05_fresh1"] = dict(
    point=fixed["BTYD05_FRESH1"]["delta"],
    p02_5=float(np.quantile(bs, 0.025)), p10=float(np.quantile(bs, 0.10)),
    p90=float(np.quantile(bs, 0.90)), p97_5=float(np.quantile(bs, 0.975)),
    p_lt_0=float(np.mean(bs < 0)), mean=float(bs.mean()), sd=float(bs.std(ddof=1)))
print("bootstrap BTYD05_FRESH1", res["bootstrap_btyd05_fresh1"])

np.savez_compressed(OUT/"_p42_bootstrap.npz", btyd05_fresh1=bs)
(OUT/"exp069_parity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("WROTE", OUT/"exp069_parity.json")
