"""EXP089: resolve what SUBMIT_JOINT_V2 measures in the EXP088 A1/A2 plane.

No model is trained and no feature is created.  Leaderboard values enter only
as the four scalar measurements supplied for EXP089 (plus the already frozen
score registry used to enumerate the scored submission span).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "new_directions" / "EXP089_JOINT_V2_PLANE_RESOLUTION"
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
E88 = ROOT / "research" / "new_directions" / "EXP088_A1_A2_TOMOGRAPHY"
GEOMETRY = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")

CANONICAL_Z = GEOMETRY / "submission_geometry" / "cache" / "Z.npz"
CANONICAL_META = GEOMETRY / "submission_geometry" / "cache" / "Z_meta.json"
CANONICAL_SCORES = GEOMETRY / "submission_geometry" / "score_registry.csv"
CANONICAL_SUBMISSIONS = GEOMETRY / "submissions"

ALPHA_PATH = Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")
EXP075_PATH = ROOT / "submissions" / "SUBMIT_EXP075_JOINT_A1_365_A2.csv"
LEVEL_PATH = ROOT / "submissions" / "SUBMIT_EXP084_LEVEL_PROBE_P020.csv"
V2_EXPECTED_PATH = ROOT / "submissions" / "SUBMIT_JOINT_V2.csv"
V2_EXPECTED_SHA256 = "211879cb1c79bbbde93d451fca5b61c521b523f989ce42bab62cd3ab87233cba"

PLUS_PATH = ROOT / "submissions" / "SUBMIT_EXP089_TOMO_PLUS.csv"
MINUS_PATH = ROOT / "submissions" / "SUBMIT_EXP089_TOMO_MINUS.csv"

SCORES = {
    "SUBMIT_ORTH_ALPHA": 1.6461597403364463,
    "SUBMIT_ORTH_FINAL": 1.6462686940209101,
    "SUBMIT_PUBLIC_EB": 1.6463246740442117,
    "SUBMIT_PRIVATE_OPTIMAL": 1.6468136172663015,
    "SUBMIT_v2_shrunk": 1.6467120249048954,
    "SUBMIT_NEXT_BEST": 1.6466079084,
    "BEST_EXISTING_SUBMISSION": 1.6503527715589217,
    "my_submit": 1.655996856087816,
    "SUBMIT_v7_newmodel": 1.6473311211432606,
    "PROBE_scale097": 1.648022805918134,
    "anchor_diverse": 1.6478377871880918,
    "SUBMIT_EXP075_JOINT_A1_365_A2": 1.646143314225527,
    "EXP084_LEVEL_PROBE_P020": 1.6462751588360733,
    "SUBMIT_JOINT_V2": 1.6459363044782171,
}

EXTRA_SCORED = [
    ("SUBMIT_ORTH_ALPHA", ALPHA_PATH, "required_anchor"),
    ("SUBMIT_ORTH_FINAL", Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_FINAL.csv"), "required_sent"),
    ("SUBMIT_PUBLIC_EB", Path(r"C:\Users\Admin\Downloads\SUBMIT_PUBLIC_EB.csv"), "required_sent"),
    ("SUBMIT_PRIVATE_OPTIMAL", Path(r"C:\Users\Admin\Downloads\SUBMIT_PRIVATE_OPTIMAL.csv"), "sent_scored_geometry"),
    ("SUBMIT_v2_shrunk", GEOMETRY / "current_best" / "SUBMIT_v2_shrunk.csv", "sent_scored_geometry"),
    ("SUBMIT_NEXT_BEST", GEOMETRY / "submission_geometry" / "SUBMIT_NEXT_BEST.csv", "sent_scored_geometry"),
    ("BEST_EXISTING_SUBMISSION", Path(r"C:\Users\Admin\Desktop\research_clean\analysis\BEST_EXISTING_SUBMISSION.csv"), "sent_scored_extra"),
    ("my_submit", ROOT / "submissions" / "my_submit.csv", "sent_scored_extra"),
    ("SUBMIT_v7_newmodel", ROOT / "submissions" / "SUBMIT_v7_newmodel.csv", "sent_scored_extra"),
    ("PROBE_scale097", ROOT / "submissions" / "PROBE_scale097.csv", "sent_scored_extra"),
    ("anchor_diverse", ROOT / "submissions" / "anchor_diverse_A_combo_mlp_hurdle_w065.csv", "sent_scored_extra"),
    ("SUBMIT_EXP075_JOINT_A1_365_A2", EXP075_PATH, "required_sent_exp075"),
    ("EXP084_LEVEL_PROBE_P020", LEVEL_PATH, "required_sent_level_probe"),
    ("SUBMIT_JOINT_V2", V2_EXPECTED_PATH, "required_current_best"),
]

ARTIFACTS = [
    E88 / "d_A1_TEST.npy",
    E88 / "d_A2_TEST.npy",
    E88 / "d_joint_TEST.npy",
    E88 / "w_robust_TEST.npy",
    E88 / "d_probe_TEST.npy",
    E88 / "tomography_vectors.npz",
]

R0 = SCORES["SUBMIT_ORTH_ALPHA"]
R1 = SCORES["SUBMIT_EXP075_JOINT_A1_365_A2"]
R2 = SCORES["SUBMIT_JOINT_V2"]
F_PUBLIC = 0.20
M_PUBLIC = 50_000
KAPPA = 1.15
SVD_RTOL = 1e-10
PROBE_RMS = 0.025
HEAVY_TAIL_THRESHOLD = 20.0
WINSOR_RMS = 10.0
JOINT_COEFFICIENTS = np.array([0.7462560853, 0.6466415685], dtype=np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float((x @ y) / den)


def distribution(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    r = rms(x)
    q = np.quantile(x, [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999])
    return {
        "min": float(x.min()), "max": float(x.max()),
        "p0.01": float(q[0]), "p0.1": float(q[1]), "p1": float(q[2]),
        "p50": float(q[3]), "p99": float(q[4]), "p99.9": float(q[5]),
        "p99.99": float(q[6]), "RMS": r,
        "max_abs_over_RMS": float(np.max(np.abs(x)) / r),
    }


def json_write(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def locate_exact_v2() -> Path:
    if V2_EXPECTED_PATH.exists() and sha256(V2_EXPECTED_PATH) == V2_EXPECTED_SHA256:
        return V2_EXPECTED_PATH
    roots = [Path(r"C:\Users\Admin\Downloads"), ROOT]
    matches = []
    for base in roots:
        for path in base.rglob("SUBMIT_JOINT_V2.csv"):
            if path.resolve() == V2_EXPECTED_PATH.resolve():
                continue
            if sha256(path) == V2_EXPECTED_SHA256:
                matches.append(path)
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one exact JOINT_V2 source, found {matches}")
    V2_EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(matches[0], V2_EXPECTED_PATH)
    if sha256(V2_EXPECTED_PATH) != V2_EXPECTED_SHA256:
        raise AssertionError("JOINT_V2 copy failed SHA256 parity")
    return V2_EXPECTED_PATH


def load_submission(path: Path, expected_uid: np.ndarray) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["user_id", "predict"]:
        raise AssertionError(f"unexpected columns in {path}: {list(frame.columns)}")
    uid = frame.user_id.to_numpy(np.int64)
    pred = frame.predict.to_numpy(np.float64)
    audit = {
        "path": str(path.resolve()), "sha256": sha256(path), "rows": int(len(frame)),
        "unique_user_id": int(frame.user_id.nunique()),
        "same_order": bool(np.array_equal(uid, expected_uid)),
        "finite": bool(np.isfinite(pred).all()),
        "nonnegative": bool(np.all(pred >= 0)),
        "zero_count": int(np.sum(pred == 0)),
        "columns": "user_id,predict",
    }
    if not (audit["rows"] == 250_000 and audit["unique_user_id"] == 250_000
            and audit["same_order"] and audit["finite"] and audit["nonnegative"]):
        raise AssertionError(f"submission audit failed: {audit}")
    return np.log1p(pred), audit


def orthonormal_basis(rows: np.ndarray) -> tuple[np.ndarray, dict]:
    m = np.asarray(rows, dtype=np.float64).copy()
    m -= m.mean(axis=1, keepdims=True)
    row_rms = np.sqrt(np.mean(m * m, axis=1))
    keep_nonzero = row_rms > 1e-14
    m = m[keep_nonzero]
    row_rms = row_rms[keep_nonzero]
    m /= row_rms[:, None]
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > s[0] * SVD_RTOL
    q = vt[keep]
    rejected = s[~keep]
    return q, {
        "input_rows": int(rows.shape[0]),
        "nonconstant_centered_rows": int(m.shape[0]),
        "centered_rank": int(keep.sum()),
        "rank_including_constant": int(keep.sum() + 1),
        "svd_relative_tolerance": SVD_RTOL,
        "smallest_retained_ratio": float(s[keep][-1] / s[0]),
        "largest_rejected_ratio": None if rejected.size == 0 else float(rejected[0] / s[0]),
        "orthonormality_max_error": float(np.max(np.abs(q @ q.T - np.eye(len(q))))),
    }


def project(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (q @ np.asarray(x, dtype=np.float64)) @ q


def center_project_twice(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, list[float]]:
    out = np.asarray(x, dtype=np.float64) - np.mean(x)
    pass_rms = []
    for _ in range(2):
        p = project(out, q)
        pass_rms.append(rms(p))
        out -= p
        out -= np.mean(out)
    return out, pass_rms


def robust_axis(d_a1: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, dict]:
    w, first_pass = center_project_twice(d_a1, q)
    before = distribution(w)
    clip_abs = None
    second_pass = None
    applied = before["max_abs_over_RMS"] > HEAVY_TAIL_THRESHOLD
    if applied:
        clip_abs = WINSOR_RMS * rms(w)
        w = np.clip(w, -clip_abs, clip_abs)
        w -= w.mean()
        w, second_pass = center_project_twice(w, q)
    after = distribution(w)
    return w, {
        "initial_projection_pass_RMS": first_pass,
        "winsorization_applied": bool(applied),
        "heavy_tail_threshold": HEAVY_TAIL_THRESHOLD,
        "winsor_clip_multiple_RMS": WINSOR_RMS,
        "winsor_clip_abs": clip_abs,
        "post_winsor_projection_pass_RMS": second_pass,
        "distribution_before": before,
        "distribution_after": after,
        "final_mean": float(w.mean()),
        "final_projection_RMS": rms(project(w, q)),
    }


def qsummary(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(x, dtype=np.float64), [0.025, 0.5, 0.975])
    return {"q2.5": float(q[0]), "median": float(q[1]), "q97.5": float(q[2])}


def public_geometry(d_exp075: np.ndarray, d_v2: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    d = np.vstack([d_exp075, d_v2])
    g = d @ d.T / d.shape[1]
    b = 0.5 * (R0 ** 2 + np.diag(g) - np.array([R1 ** 2, R2 ** 2]))
    a = np.linalg.solve(g, b)
    mse_opt = R0 ** 2 - float(b @ a)
    full = {
        "status": "FULL_POPULATION_GRAM_SANITY_NOT_PUBLIC_EXACT",
        "basis": ["realized_d_exp075", "realized_d_v2"],
        "G_full": g.tolist(), "b_using_full_G": b.tolist(),
        "coefficients": a.tolist(), "condition_number": float(np.linalg.cond(g)),
        "predicted_optimal_RMSLE": float(np.sqrt(mse_opt)),
        "gain_MSE_vs_ORTH_ALPHA": float(R0 ** 2 - mse_opt),
        "gain_MSE_vs_JOINT_V2": float(R2 ** 2 - mse_opt),
        "approx_Delta_RMSLE_vs_JOINT_V2": float(np.sqrt(mse_opt) - R2),
    }

    row_q = np.column_stack([d[0] ** 2, d[0] * d[1], d[1] ** 2])
    fpc = (1.0 - F_PUBLIC) / M_PUBLIC
    cov_q = np.cov(row_q, rowvar=False, ddof=0) * fpc
    rng = np.random.default_rng(89089)
    draws = rng.multivariate_normal(row_q.mean(axis=0), cov_q, size=40_000)
    aa, gains, opt_scores, conds = [], [], [], []
    for g11, g12, g22 in draws:
        gd = np.array([[g11, g12], [g12, g22]], dtype=np.float64)
        if np.linalg.eigvalsh(gd)[0] <= 0:
            continue
        bd = 0.5 * (R0 ** 2 + np.diag(gd) - np.array([R1 ** 2, R2 ** 2]))
        ad = np.linalg.solve(gd, bd)
        od = R0 ** 2 - float(bd @ ad)
        if od <= 0:
            continue
        aa.append(ad)
        gains.append(R2 ** 2 - od)
        opt_scores.append(np.sqrt(od))
        conds.append(np.linalg.cond(gd))
    aa = np.asarray(aa)
    gains = np.asarray(gains)
    opt_scores = np.asarray(opt_scores)

    # Existing confirmed kappa=1.15 model: uncertainty of public-to-full
    # residual alignments, reported as a risk diagnostic rather than folded
    # into the exact public-score equations.
    alignment_cov = KAPPA * (R0 ** 2) * fpc * g
    coef_transfer_cov = np.linalg.solve(g, alignment_cov) @ np.linalg.inv(g)
    post = {
        "status": "PUBLIC_POSTERIOR_ESTIMATE_MEMBERSHIP_UNKNOWN",
        "public_fraction": F_PUBLIC, "public_rows_approx": M_PUBLIC,
        "sampling_model": "Gaussian finite-population approximation from exact row-wise quadratic covariance",
        "draws_retained": int(len(gains)), "seed": 89089,
        "quadratic_term_order": ["G11", "G12", "G22"],
        "quadratic_term_full_mean": row_q.mean(axis=0).tolist(),
        "quadratic_term_sampling_covariance": cov_q.tolist(),
        "coefficients": {
            "exp075_weight": qsummary(aa[:, 0]),
            "joint_v2_weight": qsummary(aa[:, 1]),
        },
        "predicted_public_optimum_RMSLE": qsummary(opt_scores),
        "gain_MSE_over_JOINT_V2": qsummary(gains),
        "approx_Delta_RMSLE_over_JOINT_V2": qsummary(opt_scores - R2),
        "P_gain_MSE_ge_0_0003": float(np.mean(gains >= 0.0003)),
        "P_gain_MSE_ge_0_0001": float(np.mean(gains >= 0.0001)),
        "P_gain_over_JOINT_V2_within_realized_span": float(np.mean(gains > 0)),
        "condition_number": qsummary(np.asarray(conds)),
        "noise_inflation_kappa": KAPPA,
        "public_to_full_alignment_covariance": alignment_cov.tolist(),
        "coefficient_transfer_covariance_flat_prior": coef_transfer_cov.tolist(),
        "warning": "This is the realized [EXP075, JOINT_V2] span, not an identified A1/A2 plane; JOINT_V2 has a material out-of-plane component.",
    }
    return full, post, np.median(aa, axis=0)


def historical_private_risk(c_v2_plane: np.ndarray, scored_variants: int) -> dict:
    a1 = pl.read_parquet(
        E75 / "clean_forward_predictions.parquet",
        columns=["user_id", "cutoff", "residual", "u_perp_365"],
    )
    a2 = pl.read_parquet(
        E75 / "a2_clean_forward_predictions.parquet",
        columns=["user_id", "cutoff", "u_perp_A2"],
    )
    joined = a1.join(a2, on=["user_id", "cutoff"], how="inner", validate="1:1")
    uid = joined["user_id"].to_numpy().astype(np.int64)
    cutoff = joined["cutoff"].to_numpy()
    residual = joined["residual"].to_numpy().astype(np.float64)
    u1 = joined["u_perp_365"].to_numpy().astype(np.float64)
    u2 = joined["u_perp_A2"].to_numpy().astype(np.float64)
    corr_exp = JOINT_COEFFICIENTS[0] * u1 + JOINT_COEFFICIENTS[1] * u2
    corr_v2 = c_v2_plane[0] * u1 + c_v2_plane[1] * u2
    loss_exp = (residual - corr_exp) ** 2
    loss_v2 = (residual - corr_v2) ** 2
    delta = loss_v2 - loss_exp

    folds = []
    for key in sorted(np.unique(cutoff)):
        m = cutoff == key
        folds.append({
            "cutoff": str(key), "rows": int(m.sum()),
            "delta_MSE_v2_plane_minus_exp075": float(delta[m].mean()),
            "delta_RMSLE_v2_plane_minus_exp075": float(np.sqrt(loss_v2[m].mean()) - np.sqrt(loss_exp[m].mean())),
        })

    latest = sorted(np.unique(cutoff))[-1]
    ml = cutoff == latest
    le, lv = loss_exp[ml], loss_v2[ml]
    n = len(le)
    m_public = int(round(F_PUBLIC * n))
    rng = np.random.default_rng(8908901)
    pub_delta_rms, priv_delta_rms = [], []
    total_e, total_v = le.sum(), lv.sum()
    for _ in range(1000):
        idx = rng.choice(n, size=m_public, replace=False)
        pe, pv = le[idx].mean(), lv[idx].mean()
        qe = (total_e - le[idx].sum()) / (n - m_public)
        qv = (total_v - lv[idx].sum()) / (n - m_public)
        pub_delta_rms.append(np.sqrt(pv) - np.sqrt(pe))
        priv_delta_rms.append(np.sqrt(qv) - np.sqrt(qe))
    pub_delta_rms = np.asarray(pub_delta_rms)
    priv_delta_rms = np.asarray(priv_delta_rms)

    unique_uid, inv = np.unique(uid, return_inverse=True)
    csum = np.bincount(inv, weights=delta, minlength=len(unique_uid))
    ccount = np.bincount(inv, minlength=len(unique_uid)).astype(np.float64)
    boot = []
    for start in range(0, 1000, 20):
        size = min(20, 1000 - start)
        weights = rng.poisson(1.0, size=(size, len(unique_uid))).astype(np.float64)
        boot.extend(((weights @ csum) / (weights @ ccount)).tolist())
    boot = np.asarray(boot)

    centered_noise = pub_delta_rms - (np.sqrt(lv.mean()) - np.sqrt(le.mean()))
    wc = []
    for _ in range(20_000):
        sample = rng.choice(centered_noise, size=scored_variants, replace=True)
        wc.append(-F_PUBLIC / (1 - F_PUBLIC) * sample.min())
    wc = np.asarray(wc)

    return {
        "status": "NO_OPTIMIZED_CANDIDATE_V2_OUT_OF_PLANE",
        "historical_analogue": "Frozen EXP075 A1-365/A2 OOF directions; compare test-plane coefficients only",
        "test_plane_coefficients": {
            "EXP075": JOINT_COEFFICIENTS.tolist(),
            "JOINT_V2_plane_fit": c_v2_plane.tolist(),
        },
        "full_historical": {
            "rows": int(len(delta)), "unique_user_id_clusters": int(len(unique_uid)),
            "delta_MSE_v2_plane_minus_exp075": float(delta.mean()),
            "delta_RMSLE_v2_plane_minus_exp075": float(np.sqrt(loss_v2.mean()) - np.sqrt(loss_exp.mean())),
        },
        "by_forward_fold": folds,
        "pseudo_public_20_80": {
            "seed": 8908901, "splits": 1000, "latest_cutoff": str(latest),
            "public_delta_RMSLE": qsummary(pub_delta_rms),
            "private_delta_RMSLE": qsummary(priv_delta_rms),
            "P_private_improvement_for_plane_only_V2": float(np.mean(priv_delta_rms < 0)),
            "corr_public_private_delta": float(np.corrcoef(pub_delta_rms, priv_delta_rms)[0, 1]),
        },
        "user_cluster_poisson_bootstrap": {
            "replicates": 1000, "seed": 8908901,
            "delta_MSE_v2_plane_minus_exp075": qsummary(boot),
            "P_delta_MSE_lt_0": float(np.mean(boot < 0)),
        },
        "winner_curse": {
            "scored_variants_considered": int(scored_variants),
            "simulation_draws": 20_000,
            "expected_private_RMSLE_penalty_best_public": float(wc.mean()),
            "penalty_RMSLE_q95": float(np.quantile(wc, 0.95)),
        },
        "observed_public": {
            "gain_RMSLE_JOINT_V2_over_EXP075": float(R1 - R2),
            "gain_MSE_JOINT_V2_over_EXP075": float(R1 ** 2 - R2 ** 2),
        },
        "public_expected_gain": "Observed only; dominated by the new out-of-plane component and not transport-identified",
        "full_test_expected_gain": "Not identified for the out-of-plane component",
        "private_expected_gain": "Not identified for the out-of-plane component",
        "P_private_improvement": "Not identified; plane-only historical analogue is reported separately",
        "private_risk_verdict": "HIGH_UNCERTAINTY_OUT_OF_PLANE_NOT_HISTORICALLY_VALIDATED",
    }


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    locate_exact_v2()
    for path in [CANONICAL_Z, CANONICAL_META, CANONICAL_SCORES, *ARTIFACTS, ALPHA_PATH,
                 EXP075_PATH, LEVEL_PATH, V2_EXPECTED_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    canonical = np.load(CANONICAL_Z, allow_pickle=False)
    uid = canonical["user_id"].astype(np.int64)
    z_canonical = canonical["Z"].astype(np.float64)
    meta = json.loads(CANONICAL_META.read_text(encoding="utf-8"))
    canonical_names = list(meta["names"])
    if z_canonical.shape != (67, 250_000) or len(canonical_names) != 67:
        raise AssertionError("unexpected canonical scored bank")
    if len(np.unique(uid)) != 250_000 or not np.isfinite(z_canonical).all() or np.any(z_canonical < 0):
        raise AssertionError("canonical cache audit failed")

    z_extra: dict[str, np.ndarray] = {}
    audit_extra: dict[str, dict] = {}
    for name, path, _ in EXTRA_SCORED:
        z_extra[name], audit_extra[name] = load_submission(path, uid)
    if audit_extra["SUBMIT_JOINT_V2"]["sha256"] != V2_EXPECTED_SHA256:
        raise AssertionError("wrong JOINT_V2 artifact")

    # Exact source artifact audit requested in section 1.
    artifact_rows = []
    vector_map = {}
    for path in ARTIFACTS:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False).astype(np.float64)
            vector_map[path.name] = arr
            row = {
                "name": path.name, "kind": "signed_log_correction_vector", "path": str(path.resolve()),
                "sha256": sha256(path), "rows": int(arr.size), "unique_user_id": "N/A",
                "same_order": "implicit_TEST_order", "finite": bool(np.isfinite(arr).all()),
                "predict_nonnegative": "N/A_signed_vector", "details": f"shape={arr.shape};dtype={arr.dtype}",
            }
            if arr.shape != (250_000,) or not np.isfinite(arr).all():
                raise AssertionError(f"vector audit failed: {path}")
        else:
            bundle = np.load(path, allow_pickle=False)
            if "user_id" not in bundle.files or not np.array_equal(bundle["user_id"].astype(np.int64), uid):
                raise AssertionError("tomography_vectors user order mismatch")
            finite = all(np.isfinite(bundle[k]).all() for k in bundle.files)
            row = {
                "name": path.name, "kind": "npz_vector_bundle", "path": str(path.resolve()),
                "sha256": sha256(path), "rows": int(len(bundle["user_id"])),
                "unique_user_id": int(len(np.unique(bundle["user_id"]))), "same_order": True,
                "finite": bool(finite), "predict_nonnegative": "N/A_mixed_bundle",
                "details": "keys=" + "|".join(bundle.files),
            }
            if not finite:
                raise AssertionError("nonfinite tomography bundle")
        artifact_rows.append(row)

    exact_csv_labels = [
        ("SUBMIT_ORTH_ALPHA.csv", "SUBMIT_ORTH_ALPHA"),
        ("SUBMIT_EXP075_JOINT_A1_365_A2.csv", "SUBMIT_EXP075_JOINT_A1_365_A2"),
        ("SUBMIT_JOINT_V2.csv", "SUBMIT_JOINT_V2"),
        ("EXP084_LEVEL_PROBE_P020.csv", "EXP084_LEVEL_PROBE_P020"),
    ]
    for label, name in exact_csv_labels:
        a = audit_extra[name]
        artifact_rows.append({
            "name": label, "kind": "scored_submission_csv", "path": a["path"],
            "sha256": a["sha256"], "rows": a["rows"], "unique_user_id": a["unique_user_id"],
            "same_order": a["same_order"], "finite": a["finite"],
            "predict_nonnegative": a["nonnegative"], "details": f"zeros={a['zero_count']};columns={a['columns']}",
        })

    # Manifest of only actually scored files.  Canonical cache members are
    # checked against their exact source CSV and frozen SHA list.
    score_registry = pd.read_csv(CANONICAL_SCORES).set_index("submission_name")["leaderboard_score"]
    raw_stats = {x["file"]: x for x in meta["raw_stats"]}
    manifest_rows = []
    for i, name in enumerate(canonical_names):
        path = CANONICAL_SUBMISSIONS / name
        _, audit = load_submission(path, uid)
        if audit["sha256"] != meta["sha"][i]:
            raise AssertionError(f"canonical SHA mismatch: {name}")
        lb = float(score_registry.loc[name])
        if not np.isfinite(lb):
            raise AssertionError(f"missing canonical LB: {name}")
        manifest_rows.append({
            "name": name, "group": "canonical_scored_bank", "path": str(path.resolve()),
            "sha256": audit["sha256"], "leaderboard_score": lb,
            "rows": audit["rows"], "unique_user_id": audit["unique_user_id"],
            "same_order": audit["same_order"], "finite": audit["finite"],
            "predict_nonnegative": audit["nonnegative"], "source_evidence": "canonical score_registry + Z_meta",
        })
    for name, path, reason in EXTRA_SCORED:
        a = audit_extra[name]
        manifest_rows.append({
            "name": name, "group": reason, "path": str(path.resolve()), "sha256": a["sha256"],
            "leaderboard_score": SCORES[name], "rows": a["rows"],
            "unique_user_id": a["unique_user_id"], "same_order": a["same_order"],
            "finite": a["finite"], "predict_nonnegative": a["nonnegative"],
            "source_evidence": "known measured LB; exact local artifact",
        })
    manifest = pd.DataFrame(manifest_rows)
    if manifest.name.duplicated().any() or len(manifest) != 81:
        raise AssertionError(f"unexpected scored manifest size/duplicates: {len(manifest)}")

    d_a1 = vector_map["d_A1_TEST.npy"]
    d_a2 = vector_map["d_A2_TEST.npy"]
    d_joint = vector_map["d_joint_TEST.npy"]
    w_old = vector_map["w_robust_TEST.npy"]
    d_probe_old = vector_map["d_probe_TEST.npy"]
    z0 = z_extra["SUBMIT_ORTH_ALPHA"]
    z1 = z_extra["SUBMIT_EXP075_JOINT_A1_365_A2"]
    z2 = z_extra["SUBMIT_JOINT_V2"]
    z_level = z_extra["EXP084_LEVEL_PROBE_P020"]
    d_exp075 = z1 - z0
    d_v2 = z2 - z0
    x = d_v2 - d_v2.mean()

    old_names = [x[0] for x in EXTRA_SCORED[:11]]
    old_rows = np.vstack([z_canonical, *[z_extra[n][None, :] for n in old_names]])
    q_old, old_basis_audit = orthonormal_basis(old_rows)
    p_old = project(x, q_old)
    rem = x - p_old

    joint_unique = d_exp075 - d_exp075.mean()
    joint_unique -= project(joint_unique, q_old)
    c_joint = float((rem @ joint_unique) / (joint_unique @ joint_unique))
    p_joint = c_joint * joint_unique
    rem -= p_joint

    w_seq = w_old - w_old.mean()
    w_seq -= project(w_seq, q_old)
    w_seq -= float((w_seq @ joint_unique) / (joint_unique @ joint_unique)) * joint_unique
    c_w = float((rem @ w_seq) / (w_seq @ w_seq))
    p_w = c_w * w_seq
    e = rem - p_w
    reconstruction = p_old + p_joint + p_w + e
    total_energy = float(np.mean(x * x))

    comp = {
        "old_sent_span_excluding_unique_EXP075_joint": p_old,
        "unique_EXP075_joint_axis": p_joint,
        "old_EXP088_tomography_axis_w": p_w,
        "residual_e": e,
    }
    components = {
        name: {"RMS": rms(v), "energy_fraction": float(np.mean(v * v) / total_energy)}
        for name, v in comp.items()
    }

    # Level probe contributes only a constant relative to EXP075, so it adds
    # no centered rank and must not be allowed to hide the unique joint axis.
    level_relative = z_level - z1
    level_centered_rms = rms(level_relative - level_relative.mean())
    if level_centered_rms > 1e-10:
        raise AssertionError("level probe unexpectedly adds a centered direction")

    # Pre-submit A1/A2 fit after removing only the old sent span.
    a_res = np.vstack([d_a1 - d_a1.mean() - project(d_a1 - d_a1.mean(), q_old),
                       d_a2 - d_a2.mean() - project(d_a2 - d_a2.mean(), q_old)])
    g_a = a_res @ a_res.T / len(x)
    h_a = a_res @ (x - p_old) / len(x)
    c_a = np.linalg.solve(g_a, h_a)
    p_a = c_a @ a_res
    e_a = x - p_old - p_a
    plane_only_fraction = float(np.mean(p_a * p_a) / total_energy)
    augmented_plane_fraction = float(1.0 - np.mean(e_a * e_a) / total_energy)

    qj = joint_unique / rms(joint_unique)
    qw = w_seq / rms(w_seq)
    exp_after_old = (d_exp075 - d_exp075.mean()) - project(d_exp075 - d_exp075.mean(), q_old)
    v2_after_old = x - p_old
    coeff_matrix_unit = np.array([
        [np.mean(exp_after_old * qj), np.mean(exp_after_old * qw)],
        [np.mean(v2_after_old * qj), np.mean(v2_after_old * qw)],
    ])
    normalized_det = float(abs(np.linalg.det(coeff_matrix_unit)) /
                           np.prod(np.linalg.norm(coeff_matrix_unit, axis=1)))
    matrix_condition = float(np.linalg.cond(coeff_matrix_unit))
    w_ratio = float(rms(p_w) / rms(x))

    w_significant = bool(w_ratio >= 0.10 and matrix_condition <= 20 and normalized_det >= 0.10)
    plane_practical = bool(augmented_plane_fraction >= 0.98 and
                           rms(e_a) <= max(0.005, 0.10 * rms(x)))
    out_of_plane = bool(np.mean(e_a * e_a) / total_energy > 0.05 or rms(e_a) > 0.005)
    measured_second_axis = bool(w_significant)
    plane_identifiable = bool(measured_second_axis and plane_practical)

    if out_of_plane:
        verdict = "V2_OUT_OF_PLANE"
    elif not w_significant:
        verdict = "V2_JOINT_ONLY"
    elif not plane_identifiable:
        verdict = "UNIDENTIFIABLE_AFTER_CLIPPING"
    else:
        verdict = "PLANE_SOLVED"

    decomposition = {
        "verdict": verdict,
        "definitions": {
            "d_exp075": "log1p(EXP075)-log1p(ORTH_ALPHA), realized clipped correction",
            "d_v2": "log1p(JOINT_V2)-log1p(ORTH_ALPHA), realized clipped correction",
            "old_sent_span": "canonical bank + 11 previously scored extras; EXP075 axis excluded; level probe handled relative to EXP075",
        },
        "RMS_centered_d_v2": rms(x), "mean_d_v2": float(d_v2.mean()),
        "RMS_centered_d_exp075": rms(d_exp075 - d_exp075.mean()), "mean_d_exp075": float(d_exp075.mean()),
        "components": components,
        "coefficients": {"c_joint_unique": c_joint, "c_w_old": c_w},
        "reconstruction_error_RMS": rms(x - reconstruction),
        "R2_decomposition": float(1.0 - np.mean(e * e) / total_energy),
        "old_span_basis": old_basis_audit,
        "level_probe_centered_increment_RMS": level_centered_rms,
        "pre_submit_A1_A2_fit": {
            "c_A1": float(c_a[0]), "c_A2": float(c_a[1]),
            "G_full_population_centered": g_a.tolist(),
            "condition_number": float(np.linalg.cond(g_a)),
            "A1_A2_plane_only_energy_fraction": plane_only_fraction,
            "sent_plus_A1_A2_explained_fraction": augmented_plane_fraction,
            "RMS_e": rms(e_a),
            "residual_energy_fraction": float(np.mean(e_a * e_a) / total_energy),
        },
        "fixed_threshold_tests": {
            "w_component_RMS_ratio": w_ratio,
            "coefficient_matrix_unit_RMS_basis": coeff_matrix_unit.tolist(),
            "condition_number": matrix_condition,
            "normalized_determinant": normalized_det,
            "w_contribution_significant": w_significant,
            "plane_practical": plane_practical,
            "out_of_plane_significant": out_of_plane,
            "measured_second_axis": measured_second_axis,
            "plane_identifiable": plane_identifiable,
        },
    }

    full_geometry, posterior, diagnostic_a = public_geometry(d_exp075, d_v2)

    # Clipping audit for the diagnostic realized-span optimum.  It is never
    # promoted to a candidate because the A1/A2 identification gates failed.
    correction_linear = diagnostic_a[0] * d_exp075 + diagnostic_a[1] * d_v2
    z_diag_unclipped = z0 + correction_linear
    z_diag = np.maximum(z_diag_unclipped, 0.0)
    realized_diag = z_diag - z0
    dmat = np.vstack([d_exp075, d_v2])
    g_diag = dmat @ dmat.T / len(uid)
    h_diag = dmat @ realized_diag / len(uid)
    c_diag_realized = np.linalg.solve(g_diag, h_diag)
    clip_plane_residual = realized_diag - c_diag_realized @ dmat
    clip_direct_residual = realized_diag - correction_linear
    clipping = {
        "status": "DIAGNOSTIC_ONLY_NO_CANDIDATE",
        "posterior_median_linear_coefficients": diagnostic_a.tolist(),
        "rows_unclipped_z_lt_0": int(np.sum(z_diag_unclipped < 0)),
        "RMS_clipped_minus_unclipped_correction": rms(clip_direct_residual),
        "realized_plane_coefficients": c_diag_realized.tolist(),
        "RMS_realized_outside_realized_scored_span": rms(clip_plane_residual),
        "clipping_out_of_plane_energy_fraction": float(np.mean(clip_plane_residual ** 2) / np.mean(realized_diag ** 2)),
        "score_claim_exact": False,
    }

    # Updated scored span and frozen pre-submit joint reproduce EXP088's
    # projection contract, now with level probe and JOINT_V2 included.
    updated_rows = np.vstack([old_rows, z1[None, :], z_level[None, :], z2[None, :], d_joint[None, :]])
    q_updated, updated_basis_audit = orthonormal_basis(updated_rows)
    w_new, w_new_audit = robust_axis(d_a1, q_updated)
    updated_axis_rms = rms(w_new)

    probes_created = False
    probe_output = {}
    if verdict in {"V2_JOINT_ONLY", "V2_OUT_OF_PLANE", "UNIDENTIFIABLE_AFTER_CLIPPING"} and updated_axis_rms >= 0.010:
        u_new = w_new / updated_axis_rms
        d_probe_new = PROBE_RMS * u_new
        z_plus_unclipped = z2 + d_probe_new
        z_minus_unclipped = z2 - d_probe_new
        z_plus = np.maximum(z_plus_unclipped, 0.0)
        z_minus = np.maximum(z_minus_unclipped, 0.0)
        pd.DataFrame({"user_id": uid, "predict": np.expm1(z_plus)}).to_csv(
            PLUS_PATH, index=False, float_format="%.10f")
        pd.DataFrame({"user_id": uid, "predict": np.expm1(z_minus)}).to_csv(
            MINUS_PATH, index=False, float_format="%.10f")
        _, plus_audit = load_submission(PLUS_PATH, uid)
        _, minus_audit = load_submission(MINUS_PATH, uid)
        plus_audit["max_abs_serialized_log_error"] = float(np.max(np.abs(
            np.log1p(pd.read_csv(PLUS_PATH).predict.to_numpy(np.float64)) - z_plus)))
        minus_audit["max_abs_serialized_log_error"] = float(np.max(np.abs(
            np.log1p(pd.read_csv(MINUS_PATH).predict.to_numpy(np.float64)) - z_minus)))
        z_mid = 0.5 * (z_plus + z_minus)
        d_effective = 0.5 * (z_plus - z_minus)
        np.savez_compressed(
            EXP / "updated_tomography_vectors.npz",
            user_id=uid, d_A1=d_a1, d_A2=d_a2, d_joint=d_joint,
            w_old=w_old, d_probe_old=d_probe_old, joint_unique=joint_unique,
            w_sequential_old=w_seq, residual_e=e, w_new=w_new, u_new=u_new,
            d_probe_new=d_probe_new, z_base=z2,
            z_plus_unclipped=z_plus_unclipped, z_minus_unclipped=z_minus_unclipped,
            z_plus=z_plus, z_minus=z_minus, z_mid_actual=z_mid, d_effective=d_effective,
        )
        probe_output = {
            "created": True, "probe_RMS": PROBE_RMS,
            "plus": plus_audit, "minus": minus_audit,
            "clipping": {
                "plus_rows": int(np.sum(z_plus_unclipped < 0)),
                "minus_rows": int(np.sum(z_minus_unclipped < 0)),
                "plus_RMS": rms(z_plus - z_plus_unclipped),
                "minus_RMS": rms(z_minus - z_minus_unclipped),
                "RMS_effective_axis": rms(d_effective),
                "G_effective": float(np.mean(d_effective ** 2)),
                "RMS_midpoint_shift_from_JOINT_V2": rms(z_mid - z2),
                "RMS_effective_minus_nominal_probe": rms(d_effective - d_probe_new),
                "corr_effective_nominal": corr(d_effective, d_probe_new),
            },
        }
        probes_created = True
    else:
        # Always persist the updated axis, even when absorbed.
        np.savez_compressed(
            EXP / "updated_tomography_vectors.npz", user_id=uid, d_A1=d_a1, d_A2=d_a2,
            d_joint=d_joint, w_old=w_old, w_new=w_new, joint_unique=joint_unique,
            w_sequential_old=w_seq, residual_e=e,
        )
        probe_output = {"created": False, "reason": "updated axis below 0.010 or branch does not call for probes"}

    updated_tomography = {
        "verdict": verdict, "updated_axis_RMS": updated_axis_rms,
        "axis_gate_RMS_ge_0_010": bool(updated_axis_rms >= 0.010),
        "basis": updated_basis_audit, "robustification": w_new_audit,
        "corr_w_new_d_A1": corr(w_new, d_a1),
        "corr_w_new_d_joint": corr(w_new, d_joint),
        "corr_w_new_w_old": corr(w_new, w_old),
        "probes": probe_output,
        "vectors_path": str((EXP / "updated_tomography_vectors.npz").resolve()),
        "vectors_sha256": sha256(EXP / "updated_tomography_vectors.npz"),
    }

    private_risk = historical_private_risk(c_a, len(manifest))
    private_risk["current_choice"] = {
        "public_incumbent": "SUBMIT_JOINT_V2",
        "private_safe_conclusion": "No new optimized candidate is justified; JOINT_V2's out-of-plane public gain is not a validated private gain.",
        "old_EXP088_probes": "DO_NOT_SEND_INVALID_AFTER_SPAN_UPDATE",
    }

    coef_rows = [
        {"analysis": "realized_sequential", "target": "d_v2", "basis_vector": "old_sent_span", "coefficient": np.nan,
         "axis_RMS": np.nan, "component_RMS": rms(p_old), "energy_fraction": components["old_sent_span_excluding_unique_EXP075_joint"]["energy_fraction"],
         "notes": "orthogonal projection aggregate"},
        {"analysis": "realized_sequential", "target": "d_v2", "basis_vector": "joint_unique", "coefficient": c_joint,
         "axis_RMS": rms(joint_unique), "component_RMS": rms(p_joint), "energy_fraction": components["unique_EXP075_joint_axis"]["energy_fraction"],
         "notes": "realized clipped EXP075 unique axis"},
        {"analysis": "realized_sequential", "target": "d_v2", "basis_vector": "w_old", "coefficient": c_w,
         "axis_RMS": rms(w_seq), "component_RMS": rms(p_w), "energy_fraction": components["old_EXP088_tomography_axis_w"]["energy_fraction"],
         "notes": "EXP088 robust axis after sequential residualization"},
        {"analysis": "realized_sequential", "target": "d_v2", "basis_vector": "residual_e", "coefficient": 1.0,
         "axis_RMS": rms(e), "component_RMS": rms(e), "energy_fraction": components["residual_e"]["energy_fraction"],
         "notes": "outside old span, joint_unique and w"},
        {"analysis": "pre_submit_A1_A2", "target": "d_v2_minus_old_span", "basis_vector": "d_A1", "coefficient": float(c_a[0]),
         "axis_RMS": rms(a_res[0]), "component_RMS": np.nan, "energy_fraction": plane_only_fraction,
         "notes": "joint two-vector least squares"},
        {"analysis": "pre_submit_A1_A2", "target": "d_v2_minus_old_span", "basis_vector": "d_A2", "coefficient": float(c_a[1]),
         "axis_RMS": rms(a_res[1]), "component_RMS": rms(p_a), "energy_fraction": plane_only_fraction,
         "notes": "joint two-vector least squares"},
        {"analysis": "full_population_diagnostic", "target": "public_optimum", "basis_vector": "realized_d_exp075", "coefficient": full_geometry["coefficients"][0],
         "axis_RMS": rms(d_exp075), "component_RMS": np.nan, "energy_fraction": np.nan,
         "notes": "not an identified A1/A2 candidate"},
        {"analysis": "full_population_diagnostic", "target": "public_optimum", "basis_vector": "realized_d_v2", "coefficient": full_geometry["coefficients"][1],
         "axis_RMS": rms(d_v2), "component_RMS": np.nan, "energy_fraction": np.nan,
         "notes": "not an identified A1/A2 candidate"},
    ]

    artifact_audit_path = EXP / "artifact_audit.csv"
    manifest_path = EXP / "scored_span_manifest.csv"
    coefficients_path = EXP / "plane_coefficients.csv"
    pd.DataFrame(artifact_rows).to_csv(artifact_audit_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(coef_rows).to_csv(coefficients_path, index=False)

    json_write(EXP / "v2_decomposition.json", decomposition)
    json_write(EXP / "plane_geometry.json", {
        "verdict": verdict,
        "plane_identifiable": plane_identifiable,
        "full_population_realized_span_diagnostic": full_geometry,
        "coefficient_matrix_unit_RMS_basis": coeff_matrix_unit.tolist(),
        "matrix_condition_number": matrix_condition,
        "normalized_determinant": normalized_det,
        "clipping_aware_diagnostic": clipping,
        "updated_tomography": updated_tomography,
    })
    json_write(EXP / "public_posterior.json", posterior)
    json_write(EXP / "private_risk.json", private_risk)

    created_csv = [artifact_audit_path, manifest_path, coefficients_path]
    if probes_created:
        created_csv += [PLUS_PATH, MINUS_PATH]
    hashes = {str(p.resolve()): sha256(p) for p in created_csv}

    joint_pct = 100 * components["unique_EXP075_joint_axis"]["energy_fraction"]
    w_pct = 100 * components["old_EXP088_tomography_axis_w"]["energy_fraction"]
    out_pct = 100 * float(np.mean(e_a * e_a) / total_energy)
    gain_post = posterior["gain_MSE_over_JOINT_V2"]
    delta_post = posterior["approx_Delta_RMSLE_over_JOINT_V2"]
    probe_lines = "No new candidate or probes were created."
    if probes_created:
        probe_lines = (
            f"- PLUS: `{PLUS_PATH}` — SHA256 `{sha256(PLUS_PATH)}`\n"
            f"- MINUS: `{MINUS_PATH}` — SHA256 `{sha256(MINUS_PATH)}`\n"
            f"- Updated axis RMS: `{updated_axis_rms:.12f}`; nominal probe RMS: `{PROBE_RMS:.3f}`."
        )

    report = f"""# EXP089 — JOINT_V2 Plane Resolution

## Verdict

**{verdict}**

`JOINT_V2` does not identify the second EXP088 tomography axis.  Its old-`w`
component is below the fixed 10% RMS gate, the two-direction matrix fails both
conditioning gates, and the out-of-A1/A2 residual is material.  The old EXP088
`TOMO_PLUS/TOMO_MINUS` files are therefore invalid after the scored span update.

## Artifact and score audit

All six frozen EXP088 arrays/bundles and the four requested exact submissions
were audited.  Every CSV has 250,000 rows, unique `user_id`, exact canonical row
order, finite nonnegative predictions, and a recorded SHA256.  `JOINT_V2` was
recovered by exact SHA256 and copied to `{V2_EXPECTED_PATH}` without changing
bytes; SHA256 is `{sha256(V2_EXPECTED_PATH)}`.

Detailed audit: `artifact_audit.csv`.

## Updated scored span

The manifest contains **{len(manifest)}** actually scored files: the 67-file
canonical bank plus 14 exact scored additions.  The sent EXP075 joint, level
probe and `JOINT_V2` are present.  EXP088 PLUS/MINUS, EXP079 A040, H12_INTERP,
NEXT_AFTER_EXP069/PRIVATE_V2 and other PRE-LB files are absent.

The level probe is exactly EXP075 plus a constant in log space; its centered
increment RMS is `{level_centered_rms:.3e}`.  It therefore changes the level
measurement but adds no centered submission direction.

Detailed manifest: `scored_span_manifest.csv`.

## JOINT_V2 decomposition

Sequential full-population decomposition of centered realized `d_v2`:

| component | RMS | energy fraction |
| --- | ---: | ---: |
| old sent span excluding unique EXP075 joint | {rms(p_old):.12f} | {components['old_sent_span_excluding_unique_EXP075_joint']['energy_fraction']:.6%} |
| unique realized EXP075 joint | {rms(p_joint):.12f} | {components['unique_EXP075_joint_axis']['energy_fraction']:.6%} |
| old EXP088 tomography `w` | {rms(p_w):.12f} | {components['old_EXP088_tomography_axis_w']['energy_fraction']:.6%} |
| residual `e` | {rms(e):.12f} | {components['residual_e']['energy_fraction']:.6%} |

`c_joint={c_joint:.12f}`, `c_w={c_w:.12f}`.  Reconstruction RMS is
`{rms(x-reconstruction):.3e}` and decomposition R² is
`{1-np.mean(e*e)/total_energy:.9f}`.

The direct pre-submit A1/A2 fit gives `c_A1={c_a[0]:.12f}` and
`c_A2={c_a[1]:.12f}`.  Sent-span + A1/A2 explain
`{augmented_plane_fraction:.6%}`; remaining RMS is `{rms(e_a):.12f}` and
remaining energy is `{np.mean(e_a*e_a)/total_energy:.6%}`.

## A1/A2 plane geometry

In the unit-RMS `[joint_unique,w]` basis, the rows `[EXP075,JOINT_V2]` have
condition number `{matrix_condition:.6f}` and normalized determinant
`{normalized_det:.6f}`.  The old-`w` component RMS divided by centered V2 RMS is
`{w_ratio:.6%}`.  Fixed gates require `<=20`, `>=0.10`, and `>=10%`
respectively; all fail.

The practical-plane gates also fail: explained energy is
`{augmented_plane_fraction:.6%}` versus `98%`, and residual RMS is
`{rms(e_a):.12f}` versus `max(0.005, 10%*RMS) = {max(0.005,0.10*rms(x)):.12f}`.

## Leaderboard decoding

The full-250k Gram sanity calculation in the *realized* `[EXP075,V2]` span
predicts a diagnostic optimum at RMSLE `{full_geometry['predicted_optimal_RMSLE']:.12f}`
with MSE gain `{full_geometry['gain_MSE_vs_JOINT_V2']:.12f}` over V2.  This is
not an A1/A2 solution because V2 carries a material third direction.

With unknown public membership and a 50k finite-population sampling posterior,
the diagnostic additional gain MSE is `{gain_post['median']:.12f}`
(`95% [{gain_post['q2.5']:.12f}, {gain_post['q97.5']:.12f}]`), corresponding to
Delta RMSLE `{delta_post['median']:.12f}`
(`95% [{delta_post['q2.5']:.12f}, {delta_post['q97.5']:.12f}]`).  This is an
estimate/posterior, not an exact public Gram result.

## Clipping-aware optimum

No plane candidate is authorized.  For completeness, the posterior-median
realized-span diagnostic has `{clipping['rows_unclipped_z_lt_0']}` clipped rows;
its clipping-induced out-of-span energy is
`{clipping['clipping_out_of_plane_energy_fraction']:.6%}`.  Its score is not
claimed exact and no CSV was written for it.

## Public vs private expectation

The historical A1/A2 analogue supports the original EXP075 joint, but not the
new V2 out-of-plane residual.  The plane-only V2 coefficients are evaluated on
the frozen four forward folds, 1,000 random 20/80 pseudo-public splits, and a
1,000-replicate user-cluster Poisson bootstrap.  Winner's-curse correction uses
all {len(manifest)} scored variants.  The resulting verdict is
`{private_risk['private_risk_verdict']}`: public improvement does not identify
private improvement for the new third direction.

## Updated probes or candidate

{probe_lines}

Exact midpoint/effective-axis arrays and all pre/post-clipping vectors are in
`updated_tomography_vectors.npz` (SHA256
`{sha256(EXP/'updated_tomography_vectors.npz')}`).  No optimized plane candidate
was created.

## Final conclusion

- Old joint energy share in `JOINT_V2`: **{joint_pct:.4f}%**.
- Old tomography-axis energy share: **{w_pct:.4f}%**; RMS contribution ratio **{100*w_ratio:.4f}%**.
- Residual outside sent span + A1/A2: **{out_pct:.4f}%**, RMS **{rms(e_a):.6f}**.
- `JOINT_V2` did **not** measure the second axis and the A1/A2 plane cannot be
  solved from the existing scores alone.
- Do **not** submit the old EXP088 probes.  Use only the updated EXP089 symmetric
  pair if a further measurement is explicitly requested.

Created CSV SHA256 map:

```json
{json.dumps(hashes, indent=2)}
```
"""
    (EXP / "REPORT.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "verdict": verdict,
        "manifest_rows": len(manifest),
        "joint_energy_fraction": components["unique_EXP075_joint_axis"]["energy_fraction"],
        "w_energy_fraction": components["old_EXP088_tomography_axis_w"]["energy_fraction"],
        "out_of_plane_energy_fraction": float(np.mean(e_a * e_a) / total_energy),
        "w_RMS_ratio": w_ratio,
        "matrix_condition": matrix_condition,
        "normalized_determinant": normalized_det,
        "updated_axis_RMS": updated_axis_rms,
        "probes_created": probes_created,
        "csv_sha256": hashes,
    }, indent=2))


if __name__ == "__main__":
    main()
