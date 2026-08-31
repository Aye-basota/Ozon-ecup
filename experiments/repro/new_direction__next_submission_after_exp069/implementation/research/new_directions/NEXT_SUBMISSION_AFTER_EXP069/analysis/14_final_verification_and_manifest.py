"""Phase 12 - independent verification of the produced CSV + final manifest."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research"); GEOM = GEO/"submission_geometry"
sys.path.insert(0, str(GEOM))
from core import load_unique
from directions import build_basis
E69 = REPO/"research"/"new_directions"/"EXP069_BTYD05_FRESH1_PROD"
OUT = REPO/"research"/"new_directions"/"NEXT_SUBMISSION_AFTER_EXP069"
CSV = REPO/"submissions"/"SUBMIT_NEXT_AFTER_EXP069.csv"
BUILDER = REPO/"submission_geometry"/"build_NEXT_AFTER_EXP069.py"
ALPHA = 0.50; S0 = 1.6466079084

def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()

# ---------- independent reconstruction (does NOT import the builder) --------
inc = pd.read_csv(GEOM/"SUBMIT_NEXT_BEST.csv")
fresh = pq.read_table(E69/"fresh_conditional_TEST.parquet").to_pandas()
Z, names, lb, guid = load_unique(); N = Z.shape[1]
_zr, Phi, _C, lam, _W = build_basis(Z, 0, tol=1e-12)
assert np.array_equal(inc["user_id"].to_numpy().astype(np.int64), guid)
assert np.array_equal(fresh["user_id"].to_numpy().astype(np.int64), guid)
d = fresh["correction"].to_numpy().astype(float)
dp = d - (Phi @ d / N) @ Phi
dp = dp - (Phi @ dp / N) @ Phi
z_inc = np.log1p(np.clip(inc["predict"].to_numpy(float), 0, None))
pred_ref = np.maximum(np.expm1(z_inc + ALPHA*dp), 0.0)

got = pd.read_csv(CSV)
pv = got["predict"].to_numpy(float)
err = float(np.max(np.abs(pv - pred_ref)))
print("independent reconstruction max abs difference vs shipped CSV:", err)
assert err < 1e-8, err

# ---------- schema audits ---------------------------------------------------
sample = pd.read_csv(Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")/"data"/"raw"/"sample_submit.csv")
uid_s = sample["user_id"].to_numpy().astype(np.int64)
uid_g = got["user_id"].to_numpy().astype(np.int64)
dz = np.log1p(pv) - z_inc
audit = dict(
  rows=int(len(got)), unique_user_id=int(got.user_id.nunique()),
  columns=list(got.columns),
  canonical_sample_order=bool(np.array_equal(uid_g, uid_s)),
  missing_users=int(len(np.setdiff1d(uid_s, uid_g))),
  duplicate_users=int(len(got)-got.user_id.nunique()),
  nan=int(np.isnan(pv).sum()), inf=int(np.isinf(pv).sum()),
  negative=int((pv < 0).sum()), zeros=int((pv == 0).sum()),
  min=float(pv.min()), max=float(pv.max()), mean=float(pv.mean()),
  mean_log1p=float(np.log1p(pv).mean()),
  incumbent_mean_log1p=float(z_inc.mean()),
  quantiles={str(q): float(np.quantile(pv, q)) for q in (0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1)},
  effective_step_rms=float(np.sqrt(np.mean(dz**2))),
  effective_step_mean=float(dz.mean()),
  effective_step_max_abs=float(np.max(np.abs(dz))),
  rms_vs_incumbent_prediction_space=float(np.sqrt(np.mean((pv-inc["predict"].to_numpy(float))**2))),
  corr_log_with_incumbent=float(np.corrcoef(np.log1p(pv), z_inc)[0,1]),
  second_order_lb_penalty=float(np.mean(dz**2)/(2*S0)),
  public_sampling_noise_sd=float(np.sqrt(np.mean(dz**2))*np.sqrt(0.8/50000)),
)
audit["all_ok"] = bool(audit["rows"]==250000 and audit["unique_user_id"]==250000
                       and audit["columns"]==["user_id","predict"] and audit["canonical_sample_order"]
                       and audit["missing_users"]==0 and audit["duplicate_users"]==0
                       and audit["nan"]==0 and audit["inf"]==0 and audit["negative"]==0)
print(json.dumps({k:v for k,v in audit.items() if k!="quantiles"}, indent=1))
assert audit["all_ok"]

# distances to reference submissions
at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
zc = np.log1p(pv)
ref = {}
for col in ("pred_current_1_6466079084","pred_previous_1_6467120249","pred_latest_1_6492175622",
            "pred_strongest_1_6496571902","pred_btyd05_submission"):
    ref[col] = float(np.sqrt(np.mean((zc - np.log1p(at[col].to_numpy().astype(float)))**2)))
e69 = np.log1p(pq.read_table(E69/"btyd05_fresh1_TEST.parquet").to_pandas()["predict"].to_numpy())
ref["exp069_absolute_candidate"] = float(np.sqrt(np.mean((zc-e69)**2)))
ref["corr_log_with_exp069_absolute"] = float(np.corrcoef(zc, e69)[0,1])
print("log-space RMS distance to references:", json.dumps(ref, indent=1))
d_src = np.sqrt(np.mean((Z - zc)**2, axis=1)); j = int(np.argmin(d_src))
print("nearest geometry source:", names[j], float(d_src[j]))

git = subprocess.run(["git","-C",str(REPO),"status","--porcelain"], capture_output=True, text=True).stdout
head = subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"], capture_output=True, text=True).stdout.strip()

se = json.loads((OUT/"score_estimate.json").read_text(encoding="utf-8"))
man = {
  "artifact": "SUBMIT_NEXT_AFTER_EXP069.csv",
  "path": str(CSV), "sha256": sha(CSV), "bytes": CSV.stat().st_size,
  "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
  "upload_status": "DO_NOT_UPLOAD_WITHOUT_USER_ACTION; not uploaded by this task",
  "formula": ("z_inc = log1p(SUBMIT_NEXT_BEST.predict); "
              "d_fresh = fresh_conditional_TEST.correction; "
              "Phi = mean_N orthonormal basis of span{z_i - z_ref} over the 65 unique scored "
              "submissions (rank 57, eigenvalue tol 1e-12); "
              "d_perp = d_fresh - (Phi @ d_fresh / N) @ Phi (applied twice for re-orthogonalisation); "
              "z = z_inc + ALPHA * d_perp ; predict = max(expm1(z), 0)"),
  "projection_definition": ("linear projection of the correction DIRECTION onto the row space of "
                            "Phi in the mean_N inner product; the constant vector lies inside that "
                            "space (residual RMS 7.89e-09) so d_perp is level-neutral and the "
                            "affine residual is orthogonal to the intercept direction"),
  "alpha": ALPHA,
  "alpha_selection": ("conservative synthesis of (1) honest LOFO OOF nested alpha on canonical OOF "
                      "against pred_exp037, (2) a historical CV->LB transfer study over the exactly "
                      "reproducible scored submissions, (3) an external-direction geometry backtest. "
                      "No public leaderboard score of this candidate was used, produced or assumed."),
  "inputs": {},
  "row_count": int(len(got)),
  "prediction_statistics": audit,
  "reference_distances_log_space": ref,
  "nearest_geometry_source": {"name": names[j], "rms": float(d_src[j])},
  "geometry": json.loads((OUT/"_builder_result.json").read_text(encoding="utf-8"))["geometry"],
  "expected_public_lb": se["per_alpha"][str(ALPHA)] if str(ALPHA) in se["per_alpha"] else se["per_alpha"]["0.5"],
  "code_state": {"git_head": head, "git_status_porcelain": git.strip().splitlines()},
}
for p in (GEOM/"SUBMIT_NEXT_BEST.csv", E69/"fresh_conditional_TEST.parquet",
          E69/"btyd05_fresh1_TEST.parquet", GEOM/"cache"/"Z.npz",
          GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet",
          GEO/"gpt_pro_research_packet"/"06_ALIGNED_OOF.parquet"):
    man["inputs"][str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
(OUT/"final_submission_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
print("\nFINAL SHA256:", man["sha256"])
print("WROTE final_submission_manifest.json")
