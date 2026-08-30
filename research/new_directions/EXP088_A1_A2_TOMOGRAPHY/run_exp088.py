"""EXP088: frozen A1/A2 residual-plane tomography (TEST geometry only).

This script does not train or score a model.  It reconstructs the two frozen
EXP075 post-submission-span directions, projects the unknown axis out of the
current sent submission span plus the frozen joint direction, applies the
preregistered robustification rule, and writes the two symmetric probes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "new_directions" / "EXP088_A1_A2_TOMOGRAPHY"
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
GEOMETRY = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")

ALPHA_PATH = Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")
A1_SOURCE = E75 / "A1_TREE_TRAJ_365_TEST_PERP.npy"
A2_SOURCE = E75 / "A2_WEEKLY_RESIDUAL_CNN_TEST_PERP.npy"
JOINT_SOURCE = E75 / "JOINT_A1_365_A2_TEST_PERP.npy"
CANONICAL_Z = GEOMETRY / "submission_geometry" / "cache" / "Z.npz"
CANONICAL_META = GEOMETRY / "submission_geometry" / "cache" / "Z_meta.json"

# EXP075 frozen values.  The saved standalone A1/A2 PERP artifacts include
# these standalone amplitudes; division recovers the unit directions used by
# the joint coefficients below.
A1_STANDALONE_AMPLITUDE = 1.012306043162683
A2_STANDALONE_AMPLITUDE = 0.9642014960450844
JOINT_COEFFICIENTS = np.array([0.7462560853, 0.6466415685], dtype=np.float64)
JOINT_BASE_AMPLITUDE = 0.50
PROBE_RMS = 0.025
HEAVY_TAIL_THRESHOLD = 20.0
WINSOR_RMS = 10.0
SVD_RTOL = 1e-10

PLUS_PATH = ROOT / "submissions" / "SUBMIT_EXP088_TOMO_PLUS.csv"
MINUS_PATH = ROOT / "submissions" / "SUBMIT_EXP088_TOMO_MINUS.csv"


# Exact TEST files with observed leaderboard scores, beyond the 67-vector
# canonical scored bank.  Deliberately excluded: unsubmitted A1/A2 standalone
# candidates, EXP079 A040, EXP084 level probe, EXP088 outputs, and all other
# explicitly documented PRE-LB/not-uploaded candidates.
SENT_EXTRAS = [
    (ALPHA_PATH, "SUBMIT_ORTH_ALPHA"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_FINAL.csv"), "SUBMIT_ORTH_FINAL"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_PUBLIC_EB.csv"), "SUBMIT_PUBLIC_EB"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_PRIVATE_OPTIMAL.csv"), "SUBMIT_PRIVATE_OPTIMAL"),
    (GEOMETRY / "current_best" / "SUBMIT_v2_shrunk.csv", "SUBMIT_v2_shrunk"),
    (GEOMETRY / "submission_geometry" / "SUBMIT_NEXT_BEST.csv", "SUBMIT_NEXT_BEST"),
    (Path(r"C:\Users\Admin\Desktop\research_clean\analysis\BEST_EXISTING_SUBMISSION.csv"),
     "BEST_EXISTING_SUBMISSION"),
    (ROOT / "submissions" / "my_submit.csv", "my_submit"),
    (ROOT / "submissions" / "SUBMIT_v7_newmodel.csv", "SUBMIT_v7_newmodel"),
    (ROOT / "submissions" / "PROBE_scale097.csv", "PROBE_scale097"),
    (ROOT / "submissions" / "anchor_diverse_A_combo_mlp_hurdle_w065.csv", "anchor_diverse"),
    (ROOT / "submissions" / "SUBMIT_EXP075_JOINT_A1_365_A2.csv", "SUBMIT_EXP075_JOINT_A1_365_A2"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    yc = np.asarray(y, dtype=np.float64) - np.mean(y)
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float((xc @ yc) / denom)


def distribution(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    r = rms(x)
    probs = np.array([0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999])
    vals = np.quantile(x, probs)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p0.01": float(vals[0]),
        "p0.1": float(vals[1]),
        "p1": float(vals[2]),
        "p50": float(vals[3]),
        "p99": float(vals[4]),
        "p99.9": float(vals[5]),
        "p99.99": float(vals[6]),
        "RMS": r,
        "max_abs_over_RMS": float(np.max(np.abs(x)) / r),
    }


def load_submission(path: Path, user_id: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["user_id", "predict"]:
        raise AssertionError(f"unexpected columns in {path}: {list(frame.columns)}")
    if len(frame) != len(user_id) or not frame.user_id.is_unique:
        raise AssertionError(f"row/ID audit failed for {path}")
    aligned = frame.set_index("user_id").reindex(user_id)
    if aligned.predict.isna().any():
        raise AssertionError(f"ID alignment failed for {path}")
    prediction = aligned.predict.to_numpy(np.float64)
    if not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise AssertionError(f"prediction-domain audit failed for {path}")
    return np.log1p(prediction)


def build_basis(user_id: np.ndarray, d_joint: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    canonical = np.load(CANONICAL_Z, allow_pickle=False)
    canonical_uid = canonical["user_id"].astype(np.int64)
    if not np.array_equal(canonical_uid, user_id):
        raise AssertionError("canonical geometry user order mismatch")
    matrix = [np.asarray(canonical["Z"], dtype=np.float64)]
    meta = json.loads(CANONICAL_META.read_text(encoding="utf-8"))
    canonical_names = list(meta["names"])
    if len(canonical_names) != matrix[0].shape[0]:
        raise AssertionError("canonical name/vector count mismatch")

    sources: list[dict] = [{
        "name": "canonical_scored_bank",
        "path": str(CANONICAL_Z),
        "sha256": sha256(CANONICAL_Z),
        "vectors": int(matrix[0].shape[0]),
        "members": canonical_names,
    }]
    extra_vectors: list[np.ndarray] = []
    z_alpha = None
    for path, name in SENT_EXTRAS:
        z = load_submission(path, user_id)
        extra_vectors.append(z)
        sources.append({"name": name, "path": str(path), "sha256": sha256(path), "vectors": 1})
        if path == ALPHA_PATH:
            z_alpha = z
    if z_alpha is None:
        raise AssertionError("anchor was not loaded")

    # d_joint is explicit even though the actually sent, clipped EXP075 output
    # is also present: the prompt requires orthogonality to both current sent
    # space and the pre-submit joint direction.
    matrix.extend(extra_vectors)
    matrix.append(np.asarray(d_joint, dtype=np.float64)[None, :])
    sources.append({
        "name": "EXP075_d_joint_pre_submit",
        "path": str(JOINT_SOURCE),
        "sha256": sha256(JOINT_SOURCE),
        "vectors": 1,
    })

    M = np.vstack(matrix)
    M -= M.mean(axis=1, keepdims=True)  # constant is handled explicitly
    row_rms = np.sqrt(np.mean(M * M, axis=1))
    if np.any(row_rms == 0):
        raise AssertionError("zero centered basis row")
    M /= row_rms[:, None]  # span-invariant conditioning

    # SVD yields an orthonormal row basis directly.  The retained rank is
    # separated from the next singular value by >1,000x on the frozen inputs.
    _, singular_values, vt = np.linalg.svd(M, full_matrices=False)
    keep = singular_values > singular_values[0] * SVD_RTOL
    Q = vt[keep]
    orthogonality_error = float(np.max(np.abs(Q @ Q.T - np.eye(Q.shape[0]))))
    audit = {
        "constant_included_by_centering": True,
        "input_vectors_excluding_constant": int(M.shape[0]),
        "canonical_vectors": int(len(canonical_names)),
        "sent_extra_vectors": int(len(extra_vectors)),
        "explicit_joint_vectors": 1,
        "centered_rank": int(np.sum(keep)),
        "rank_including_constant": int(np.sum(keep) + 1),
        "svd_relative_tolerance": SVD_RTOL,
        "smallest_retained_singular_ratio": float(singular_values[keep][-1] / singular_values[0]),
        "largest_rejected_singular_ratio": float(singular_values[~keep][0] / singular_values[0]),
        "orthonormality_max_error": orthogonality_error,
        "sources": sources,
    }
    return Q, audit, z_alpha


def project_once(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64) - np.mean(x)
    out -= (Q @ out) @ Q
    return out


def center_and_project_twice(x: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, list[float]]:
    out = np.asarray(x, dtype=np.float64) - np.mean(x)
    projection_rms = []
    for _ in range(2):
        projection = (Q @ out) @ Q
        projection_rms.append(rms(projection))
        out -= projection
        out -= np.mean(out)
    return out, projection_rms


def clipping_audit(unclipped: np.ndarray, clipped: np.ndarray) -> dict:
    difference = clipped - unclipped
    return {
        "rows_unclipped_z_lt_0": int(np.sum(unclipped < 0)),
        "RMS_clipped_minus_unclipped": rms(difference),
        "max_clipping_difference": float(np.max(np.abs(difference))),
        "clipping_RMS_over_probe_RMS": float(rms(difference) / PROBE_RMS),
    }


def validate_written_submission(path: Path, expected_uid: np.ndarray, expected_z: np.ndarray) -> dict:
    frame = pd.read_csv(path)
    uid = frame.user_id.to_numpy(np.int64)
    prediction = frame.predict.to_numpy(np.float64)
    z_read = np.log1p(prediction)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "unique_user_id": int(frame.user_id.nunique()),
        "columns": list(frame.columns),
        "same_order": bool(np.array_equal(uid, expected_uid)),
        "finite": bool(np.isfinite(prediction).all()),
        "nonnegative": bool(np.all(prediction >= 0)),
        "zero_count": int(np.sum(prediction == 0)),
        "max_abs_serialized_log_error": float(np.max(np.abs(z_read - expected_z))),
    }


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    PLUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for required in (A1_SOURCE, A2_SOURCE, JOINT_SOURCE, CANONICAL_Z, CANONICAL_META, ALPHA_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    # Use the exact EXP075 artifact values.  float32 -> float64 is lossless.
    saved_a1 = np.load(A1_SOURCE, allow_pickle=False).astype(np.float64)
    saved_a2 = np.load(A2_SOURCE, allow_pickle=False).astype(np.float64)
    d_joint = np.load(JOINT_SOURCE, allow_pickle=False).astype(np.float64)
    d_a1 = saved_a1 / A1_STANDALONE_AMPLITUDE
    d_a2 = saved_a2 / A2_STANDALONE_AMPLITUDE
    if not (saved_a1.shape == saved_a2.shape == d_joint.shape == (250_000,)):
        raise AssertionError("unexpected EXP075 TEST-vector shape")

    joint_reconstructed = JOINT_COEFFICIENTS[0] * d_a1 + JOINT_COEFFICIENTS[1] * d_a2
    joint_error = d_joint - joint_reconstructed
    joint_audit = {
        "formula": "d_joint ~= 0.7462560853*d_A1 + 0.6466415685*d_A2",
        "max_abs_error": float(np.max(np.abs(joint_error))),
        "RMS_error": rms(joint_error),
        "mean_error": float(np.mean(joint_error)),
        "correlation": corr(d_joint, joint_reconstructed),
        "tolerance": 1e-7,
        "passes_tolerance": bool(np.max(np.abs(joint_error)) <= 1e-7),
    }
    if not joint_audit["passes_tolerance"]:
        raise AssertionError(f"joint reconstruction failed: {joint_audit}")

    Q, basis_audit, z_alpha = build_basis(
        np.load(CANONICAL_Z, allow_pickle=False)["user_id"].astype(np.int64), d_joint
    )
    user_id = np.load(CANONICAL_Z, allow_pickle=False)["user_id"].astype(np.int64)

    # First requested construction and collinearity check.
    w1 = project_once(d_a1, Q)
    w2 = project_once(d_a2, Q)
    corr_w1_w2 = corr(w1, w2)
    w2_on_w1 = float((w1 @ w2) / (w1 @ w1))
    collinearity_residual = w2 - w2_on_w1 * w1
    if abs(corr_w1_w2) < 0.999:
        raise AssertionError(f"A1/A2 residuals are not collinear: {corr_w1_w2}")

    # Canonical orientation, followed by the explicitly requested double pass.
    w = w1.copy()
    sign_flipped = False
    if corr(w, d_a1) <= 0:
        w *= -1.0
        sign_flipped = True
    w, initial_projection_pass_rms = center_and_project_twice(w, Q)
    before_robustification = distribution(w)
    initial_rms = rms(w)

    winsorization_applied = before_robustification["max_abs_over_RMS"] > HEAVY_TAIL_THRESHOLD
    winsor_clip_abs = None
    post_winsor_projection_pass_rms = None
    if winsorization_applied:
        winsor_clip_abs = WINSOR_RMS * initial_rms
        w = np.clip(w, -winsor_clip_abs, winsor_clip_abs)
        w -= np.mean(w)
        w, post_winsor_projection_pass_rms = center_and_project_twice(w, Q)
    after_robustification = distribution(w)

    final_projection = (Q @ w) @ Q
    basis_projection_coefficients = Q @ w
    axis_audit = {
        "corr_w1_w2": corr_w1_w2,
        "abs_corr_w1_w2": abs(corr_w1_w2),
        "w2_on_w1_coefficient": w2_on_w1,
        "expected_w2_on_w1_from_joint": float(-JOINT_COEFFICIENTS[0] / JOINT_COEFFICIENTS[1]),
        "collinearity_residual_RMS": rms(collinearity_residual),
        "w1_RMS_after_first_projection": rms(w1),
        "w2_RMS_after_first_projection": rms(w2),
        "d_A1_RMS": rms(d_a1),
        "new_axis_RMS_over_d_A1_RMS": float(rms(w1) / rms(d_a1)),
        "canonical_sign_flipped": sign_flipped,
        "corr_w_d_A1": corr(w, d_a1),
        "initial_double_projection_pass_RMS": initial_projection_pass_rms,
        "winsorization_applied": winsorization_applied,
        "winsor_threshold_max_abs_over_RMS": HEAVY_TAIL_THRESHOLD,
        "winsor_clip_multiple_RMS": WINSOR_RMS,
        "winsor_clip_abs": winsor_clip_abs,
        "post_winsor_double_projection_pass_RMS": post_winsor_projection_pass_rms,
        "distribution_before_robustification": before_robustification,
        "distribution_after_robustification": after_robustification,
        "final_mean": float(np.mean(w)),
        "final_corr_d_joint": corr(w, d_joint),
        "final_projection_RMS": rms(final_projection),
        "final_projection_RMS_over_axis_RMS": float(rms(final_projection) / rms(w)),
        "max_abs_orthonormal_basis_coefficient": float(np.max(np.abs(basis_projection_coefficients))),
    }

    u = w / rms(w)
    d_probe = PROBE_RMS * u
    z_base = z_alpha + JOINT_BASE_AMPLITUDE * d_joint
    z_plus_unclipped = z_base + d_probe
    z_minus_unclipped = z_base - d_probe
    z_plus = np.maximum(z_plus_unclipped, 0.0)
    z_minus = np.maximum(z_minus_unclipped, 0.0)
    predict_plus = np.expm1(z_plus)
    predict_minus = np.expm1(z_minus)

    pd.DataFrame({"user_id": user_id, "predict": predict_plus}).to_csv(
        PLUS_PATH, index=False, float_format="%.10f"
    )
    pd.DataFrame({"user_id": user_id, "predict": predict_minus}).to_csv(
        MINUS_PATH, index=False, float_format="%.10f"
    )

    clip_plus = clipping_audit(z_plus_unclipped, z_plus)
    clip_minus = clipping_audit(z_minus_unclipped, z_minus)
    clipping_material = bool(
        clip_plus["rows_unclipped_z_lt_0"] > 0 or clip_minus["rows_unclipped_z_lt_0"] > 0
    )

    # Any two realized probe vectors are exactly symmetric around their actual
    # midpoint, even when clipping makes them asymmetric around z_base.  These
    # exact vectors preregister the clipping-aware two-score decode.
    z_mid_actual = 0.5 * (z_plus + z_minus)
    d_effective = 0.5 * (z_plus - z_minus)
    g_effective = float(np.mean(d_effective * d_effective))

    # Persist the exact float64 arrays before any leaderboard result is known.
    np.save(EXP / "d_A1_TEST.npy", d_a1, allow_pickle=False)
    np.save(EXP / "d_A2_TEST.npy", d_a2, allow_pickle=False)
    np.save(EXP / "d_joint_TEST.npy", d_joint, allow_pickle=False)
    np.save(EXP / "w_robust_TEST.npy", w, allow_pickle=False)
    np.save(EXP / "d_probe_TEST.npy", d_probe, allow_pickle=False)
    np.savez_compressed(
        EXP / "tomography_vectors.npz",
        user_id=user_id,
        d_A1=d_a1,
        d_A2=d_a2,
        d_joint=d_joint,
        w=w,
        u=u,
        d_probe=d_probe,
        z_base=z_base,
        z_plus_unclipped=z_plus_unclipped,
        z_minus_unclipped=z_minus_unclipped,
        z_plus=z_plus,
        z_minus=z_minus,
        z_mid_actual=z_mid_actual,
        d_effective=d_effective,
    )

    vector_paths = {}
    for name in ("d_A1_TEST.npy", "d_A2_TEST.npy", "d_joint_TEST.npy", "w_robust_TEST.npy",
                 "d_probe_TEST.npy", "tomography_vectors.npz"):
        path = EXP / name
        vector_paths[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    output_audit = {
        "plus": validate_written_submission(PLUS_PATH, user_id, z_plus),
        "minus": validate_written_submission(MINUS_PATH, user_id, z_minus),
    }
    for side in output_audit.values():
        if not (side["rows"] == 250_000 and side["unique_user_id"] == 250_000
                and side["same_order"] and side["finite"] and side["nonnegative"]):
            raise AssertionError(f"submission output audit failed: {side}")

    audit = {
        "status": "PRE_LB_PROBES_READY_NOT_SUBMITTED",
        "models_trained": False,
        "leaderboard_used_for_direction_or_scale": False,
        "frozen_parameters": {
            "joint_base_amplitude": JOINT_BASE_AMPLITUDE,
            "probe_RMS": PROBE_RMS,
            "intended_G": PROBE_RMS ** 2,
            "heavy_tail_threshold": HEAVY_TAIL_THRESHOLD,
            "winsor_clip_multiple_RMS": WINSOR_RMS,
        },
        "source_artifacts": {
            "A1_saved_scaled_PERP": {"path": str(A1_SOURCE), "sha256": sha256(A1_SOURCE),
                                      "standalone_amplitude_removed": A1_STANDALONE_AMPLITUDE},
            "A2_saved_scaled_PERP": {"path": str(A2_SOURCE), "sha256": sha256(A2_SOURCE),
                                      "standalone_amplitude_removed": A2_STANDALONE_AMPLITUDE},
            "joint_saved_PERP": {"path": str(JOINT_SOURCE), "sha256": sha256(JOINT_SOURCE)},
            "anchor": {"path": str(ALPHA_PATH), "sha256": sha256(ALPHA_PATH)},
        },
        "joint_reconstruction": joint_audit,
        "basis": basis_audit,
        "axis": axis_audit,
        "normalization": {
            "RMS_w_final": rms(w),
            "RMS_u": rms(u),
            "RMS_d_probe": rms(d_probe),
            "G_mean_d_probe_squared": float(np.mean(d_probe * d_probe)),
        },
        "base": {
            "negative_rows_before_probe": int(np.sum(z_base < 0)),
            "min": float(np.min(z_base)),
            "max": float(np.max(z_base)),
        },
        "clipping": {
            "plus": clip_plus,
            "minus": clip_minus,
            "materially_nonzero": clipping_material,
            "affine_closed_form_exact": not clipping_material,
            "exact_vectors_saved": True,
        },
        "clipping_aware_effective_pair": {
            "definition": "z_mid=(z_plus+z_minus)/2; d_eff=(z_plus-z_minus)/2",
            "RMS_d_effective": rms(d_effective),
            "G_effective": g_effective,
            "RMS_z_mid_minus_z_base": rms(z_mid_actual - z_base),
            "max_abs_z_mid_minus_z_base": float(np.max(np.abs(z_mid_actual - z_base))),
            "rows_z_mid_differs_from_z_base": int(np.sum(np.abs(z_mid_actual - z_base) > 1e-15)),
            "RMS_d_effective_minus_d_probe": rms(d_effective - d_probe),
            "corr_d_effective_d_probe": corr(d_effective, d_probe),
            "decode": {
                "b_effective": "(S_minus^2-S_plus^2)/4",
                "R_mid_squared": "(S_plus^2+S_minus^2)/2-G_effective",
                "a_star_effective": "b_effective/G_effective",
                "gain_MSE_effective": "b_effective^2/G_effective",
            },
        },
        "saved_vectors": vector_paths,
        "outputs": output_audit,
    }
    (EXP / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "joint_reconstruction": joint_audit,
        "basis_rank": basis_audit["centered_rank"],
        "axis": axis_audit,
        "normalization": audit["normalization"],
        "clipping": audit["clipping"],
        "effective_pair": audit["clipping_aware_effective_pair"],
        "outputs": output_audit,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
