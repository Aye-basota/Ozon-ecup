from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


AMPLITUDE = 0.40
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
JOINT_COEFFICIENTS = np.asarray(
    [0.7462560852846633, 0.6466415684754089], dtype=np.float64
)
BASELINE_PUBLIC_RMSLE = 1.6461597403364463
EXPECTED_ALPHA_SHA256 = (
    "9a8adb83e7b34bb6c12b7eb51584d1bf9a93825945d285258d4e1dd991f4b838"
)
EXPECTED_EXP075_SHA256 = (
    "d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901"
)
EXPECTED_DPERP_SHA256 = (
    "e3667884a661adf64a6ce5f231956bab18e45a7e6f017e453506f5e93d3045da"
)

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
E76 = ROOT / "research" / "new_directions" / "EXP076_STRONG_BASELINE_VALIDATION_CHANNEL"
ALPHA_PATH = Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")
EXP075_PATH = ROOT / "submissions" / "SUBMIT_EXP075_JOINT_A1_365_A2.csv"
DPERP_PATH = E75 / "JOINT_A1_365_A2_TEST_PERP.npy"
SAMPLE_PATH = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw\sample_submit.csv")
EXTERNAL_ARTIFACTS = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\artifacts")
CANDIDATE_PATH = ROOT / "submissions" / "SUBMIT_EXP079_EXP075_A040.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = math.sqrt(float(x0 @ x0) * float(y0 @ y0))
    return float((x0 @ y0) / denom)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require(list(frame.columns) == ["user_id", "predict"], f"Bad columns: {path}")
    return frame


def parity_audit() -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    sample = pd.read_csv(SAMPLE_PATH)
    alpha = read_submission(ALPHA_PATH)
    sent = read_submission(EXP075_PATH)
    d_perp_stored = np.load(DPERP_PATH)

    require(len(sample) == 250_000, "sample_submit does not have 250000 rows")
    require(len(alpha) == len(sample), "ORTH_ALPHA row count mismatch")
    require(len(sent) == len(sample), "EXP075 row count mismatch")
    require(d_perp_stored.shape == (len(sample),), "D_perp shape mismatch")
    require(d_perp_stored.dtype == np.float32, "D_perp is not stored float32")

    sample_ids = sample["user_id"].to_numpy(np.int64)
    alpha_ids = alpha["user_id"].to_numpy(np.int64)
    sent_ids = sent["user_id"].to_numpy(np.int64)
    row_order_alpha = bool(np.array_equal(alpha_ids, sample_ids))
    row_order_sent = bool(np.array_equal(sent_ids, sample_ids))

    alpha_predict = alpha["predict"].to_numpy(np.float64)
    sent_predict = sent["predict"].to_numpy(np.float64)
    require(np.isfinite(alpha_predict).all(), "ORTH_ALPHA contains non-finite values")
    require((alpha_predict >= 0.0).all(), "ORTH_ALPHA contains negative values")
    require(np.isfinite(sent_predict).all(), "EXP075 contains non-finite values")
    require((sent_predict >= 0.0).all(), "EXP075 contains negative values")

    d_perp = d_perp_stored.astype(np.float64)
    z_alpha = np.log1p(alpha_predict)
    z_full = np.maximum(z_alpha + d_perp, 0.0)
    pred_full = np.expm1(z_full)
    z_sent = np.log1p(sent_predict)

    clip_rebuilt = z_alpha + d_perp <= 0.0
    clip_sent = sent_predict == 0.0
    clip_set_equal = bool(np.array_equal(clip_rebuilt, clip_sent))
    d_rebuilt = z_full - z_alpha
    d_sent = z_sent - z_alpha

    # A pre-float32 value that rounded to the stored value lies within half an ULP.
    half_float32_ulp = float(
        np.max(np.abs(np.spacing(d_perp_stored)).astype(np.float64)) / 2.0
    )
    csv_log_slack = 5.1e-11
    parity_tolerance_log = half_float32_ulp + csv_log_slack + 16 * np.finfo(float).eps
    max_abs_log_difference = float(np.max(np.abs(z_full - z_sent)))
    max_abs_predict_difference = float(np.max(np.abs(pred_full - sent_predict)))

    formatted_rebuilt = np.char.mod("%.10f", pred_full)
    formatted_sent = np.char.mod("%.10f", sent_predict)
    serialized_prediction_matches = int(np.sum(formatted_rebuilt == formatted_sent))
    with tempfile.TemporaryDirectory(prefix="exp079_parity_") as tmp:
        rebuilt_path = Path(tmp) / "SUBMIT_EXP075_REBUILT_FROM_FLOAT32.csv"
        pd.DataFrame({"user_id": sample_ids, "predict": pred_full}).to_csv(
            rebuilt_path, index=False, float_format="%.10f"
        )
        rebuilt_sha = sha256(rebuilt_path)

    alpha_sha = sha256(ALPHA_PATH)
    sent_sha = sha256(EXP075_PATH)
    dperp_sha = sha256(DPERP_PATH)
    parity_pass = bool(
        row_order_alpha
        and row_order_sent
        and len(np.unique(sent_ids)) == len(sent_ids)
        and clip_set_equal
        and max_abs_log_difference <= parity_tolerance_log
        and alpha_sha == EXPECTED_ALPHA_SHA256
        and sent_sha == EXPECTED_EXP075_SHA256
        and dperp_sha == EXPECTED_DPERP_SHA256
    )

    audit = {
        "status": "PASS" if parity_pass else "PARITY_FAIL",
        "alpha_path": str(ALPHA_PATH),
        "alpha_sha256": alpha_sha,
        "expected_alpha_sha256": EXPECTED_ALPHA_SHA256,
        "exp075_path": str(EXP075_PATH),
        "exp075_sha256": sent_sha,
        "expected_exp075_sha256": EXPECTED_EXP075_SHA256,
        "d_perp_path": str(DPERP_PATH),
        "d_perp_sha256": dperp_sha,
        "expected_d_perp_sha256": EXPECTED_DPERP_SHA256,
        "d_perp_dtype": str(d_perp_stored.dtype),
        "d_perp_shape": list(d_perp_stored.shape),
        "row_order_alpha_vs_sample": row_order_alpha,
        "row_order_exp075_vs_sample": row_order_sent,
        "unique_user_id_exp075": int(len(np.unique(sent_ids))),
        "clip_count_rebuilt": int(clip_rebuilt.sum()),
        "clip_count_sent": int(clip_sent.sum()),
        "clip_set_equal": clip_set_equal,
        "rms_d_perp_all": rms(d_perp),
        "rms_applied_rebuilt": rms(d_rebuilt),
        "rms_applied_sent": rms(d_sent),
        "correlation_applied_rebuilt_vs_sent": correlation(d_rebuilt, d_sent),
        "max_abs_log_difference": max_abs_log_difference,
        "max_abs_predict_difference": max_abs_predict_difference,
        "float32_half_ulp_bound": half_float32_ulp,
        "csv_log_serialization_slack": csv_log_slack,
        "parity_tolerance_log": parity_tolerance_log,
        "serialized_prediction_matches": serialized_prediction_matches,
        "serialized_prediction_total": int(len(sent)),
        "rebuilt_from_float32_sha256": rebuilt_sha,
        "exact_serialized_sha_match": rebuilt_sha == sent_sha,
    }
    return audit, sample, z_alpha, d_perp


SIMPLE_COMPONENTS = [
    "S1-E02",
    "S1-E03a",
    "S1-DIST",
    "S1-E10",
    "S1-E11",
    "S1-SEEDAVG5",
    "S1-B0",
    "S1-E01",
    "S1-E03b",
    "SEQ-AVG3",
    "SEQ-D3A-AVG3",
    "SEQ-D3A-BASE-AVG3",
    "ETX-AVG3",
    "ETX-AVG2",
    "ETX-01-S42",
    "PT-FULL-AVG3",
    "PT-OD-AVG3",
    "PT-SHUF-AVG3",
    "RIDGE15",
    "HOLIDAY-YOY-FAST",
    "MHZ-FULL",
    "MHZ-BASE",
    "MHZ-P30",
    "MHZ-SELF",
    "S04-A",
    "S04-B",
    "S04-C",
    "GAP-E02-K5-G090-S42",
    "GAP-E10-K5-G090-S42",
    "GAP-DIST-K5-G060-S42",
    "SAMPLE-TB1-AVG3-R300",
    "SAMPLE-BASELINE-B-AVG3-R300",
    "SAMPLE-DENSE-S3-F422-S42-R300",
    "S1-ROUNDS-R600",
    "S1-ROUNDS-R300",
]


def build_oof_component_matrix(
    a1: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    uid = a1["user_id"].to_numpy(np.int64)
    cut = a1["cutoff"].astype(str).to_numpy()
    cut_codes = {value: idx for idx, value in enumerate(sorted(set(cut)))}
    key = np.asarray([cut_codes[value] for value in cut], dtype=np.int64) * 10_000_000 + uid
    order = np.argsort(key)
    key_sorted = key[order]

    def reindex(u2: np.ndarray, c2: np.ndarray, z2: np.ndarray) -> np.ndarray:
        c2s = np.asarray(c2).astype(str)
        k2 = (
            np.asarray([cut_codes[value] for value in c2s], dtype=np.int64) * 10_000_000
            + np.asarray(u2, dtype=np.int64)
        )
        pos = np.searchsorted(key_sorted, k2)
        require(int(pos.max()) < len(key_sorted), "OOF key outside canonical rows")
        require(np.array_equal(key_sorted[pos], k2), "OOF key mismatch")
        result = np.empty(len(key), dtype=np.float64)
        result[order[pos]] = np.asarray(z2, dtype=np.float64)
        return result

    names: list[str] = []
    columns: list[np.ndarray] = []
    provenance: dict[str, str] = {}

    def append_component(name: str, file_path: Path, uid_key: str, field: str) -> None:
        require(file_path.exists(), f"Missing OOF artifact: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        u2 = data[uid_key]
        c2 = data["cutoff"]
        z2 = data[field].astype(np.float64)
        if np.array_equal(np.asarray(u2, dtype=np.int64), uid) and np.array_equal(
            np.asarray(c2).astype(str), cut
        ):
            values = z2
        else:
            values = reindex(u2, c2, z2)
        require(np.isfinite(values).all(), f"Non-finite OOF component: {name}")
        names.append(name)
        columns.append(values)
        provenance[name] = str(file_path)

    for name in SIMPLE_COMPONENTS:
        append_component(
            name,
            EXTERNAL_ARTIFACTS / f"oof_{name}.npz",
            "user_id",
            "z",
        )
    append_component(
        "BTYD:z_btyd",
        EXTERNAL_ARTIFACTS / "BTYD_STABLE_EXP051" / "oof_raw.npz",
        "user_id",
        "z_btyd",
    )
    append_component(
        "BTYD:z_strongest",
        EXTERNAL_ARTIFACTS / "BTYD_STABLE_EXP051" / "oof_raw.npz",
        "user_id",
        "z_strongest",
    )
    append_component(
        "BLOCK4:z_new_honest",
        EXTERNAL_ARTIFACTS / "oof_BLOCK4_SAF.npz",
        "uid",
        "z_new_honest",
    )
    for field in ["z_fresh", "z_vol", "z_clean"]:
        append_component(
            f"FRESH:{field}",
            EXTERNAL_ARTIFACTS / "oof_FRESH_CONTRAST_MOE.npz",
            "uid",
            field,
        )

    matrix = np.column_stack(columns)
    keep: list[int] = []
    for column_idx in range(matrix.shape[1]):
        duplicate = any(
            float(np.max(np.abs(matrix[:, column_idx] - matrix[:, prior_idx]))) < 1e-9
            for prior_idx in keep
        )
        if not duplicate:
            keep.append(column_idx)
    deduplicated = matrix[:, keep]
    deduplicated_names = [names[idx] for idx in keep]
    return deduplicated, deduplicated_names, provenance


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    diagonal_scale = float(np.diag(xtx).mean())
    system = xtx + alpha * diagonal_scale * np.eye(x.shape[1])
    system[0, 0] = xtx[0, 0]
    return np.linalg.solve(system, x.T @ y)


def project_out(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(basis, vector, rcond=None)
    return vector - basis @ coefficients


def build_composition_matched_baseline(
    target_log: np.ndarray,
    component_matrix: np.ndarray,
    component_names: list[str],
    fold_masks: list[np.ndarray],
) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(component_names)}
    families = {
        "SEQ": ["SEQ-AVG3", "SEQ-D3A-AVG3", "SEQ-D3A-BASE-AVG3"],
        "ETX": ["ETX-AVG3", "ETX-AVG2", "ETX-01-S42"],
        "TAB": [
            "S1-E03a",
            "S1-E02",
            "S1-DIST",
            "S1-E11",
            "RIDGE15",
            "HOLIDAY-YOY-FAST",
            "S04-A",
            "S04-B",
            "S1-E10",
            "SAMPLE-TB1-AVG3-R300",
            "MHZ-FULL",
            "GAP-E10-K5-G090-S42",
            "SAMPLE-DENSE-S3-F422-S42-R300",
        ],
        "BTYD": ["BTYD:z_btyd"],
    }
    frozen = {
        "SEQ": {"SEQ-AVG3": 1.0},
        "ETX": {"ETX-AVG3": 1.0},
        "TAB": {"S1-E03a": 0.10 / 0.55, "S1-E02": 0.20 / 0.55, "S1-DIST": 0.25 / 0.55},
        "BTYD": {"BTYD:z_btyd": 1.0},
    }
    alpha_composition = json.loads(
        (E76 / "out" / "s10_alpha_composition.json").read_text(encoding="utf-8")
    )
    shares = alpha_composition["SUBMIT_ORTH_ALPHA"]["shares"]

    family_predictions: dict[str, np.ndarray] = {}
    for family, columns in families.items():
        require(all(name in index for name in columns), f"Missing {family} component")
        prediction = np.full(len(target_log), np.nan, dtype=np.float64)
        family_matrix = np.hstack(
            [
                np.ones((len(target_log), 1), dtype=np.float64),
                component_matrix[:, [index[name] for name in columns]],
            ]
        )
        for fold_idx, current_mask in enumerate(fold_masks):
            if fold_idx == 0:
                prediction[current_mask] = sum(
                    weight * component_matrix[current_mask, index[name]]
                    for name, weight in frozen[family].items()
                )
            else:
                training_mask = np.logical_or.reduce(fold_masks[:fold_idx])
                coefficients = ridge_fit(
                    family_matrix[training_mask], target_log[training_mask], 3e-5
                )
                prediction[current_mask] = family_matrix[current_mask] @ coefficients
        family_predictions[family] = prediction

    mixed = sum(shares[family] * family_predictions[family] for family in families)
    matched = np.full(len(target_log), np.nan, dtype=np.float64)
    for fold_idx, current_mask in enumerate(fold_masks):
        if fold_idx == 0:
            offset = 0.0
        else:
            training_mask = np.logical_or.reduce(fold_masks[:fold_idx])
            offset = float((target_log[training_mask] - mixed[training_mask]).mean())
        matched[current_mask] = mixed[current_mask] + offset
    require(np.isfinite(matched).all(), "Composition-matched baseline is non-finite")
    return matched


def bootstrap_weighted_metrics(
    user_id: np.ndarray,
    fold_index: np.ndarray,
    error0: np.ndarray,
    error1: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    unique_user, inverse = np.unique(user_id.astype(np.int64), return_inverse=True)
    cluster = np.zeros((len(unique_user), len(FOLDS) * 3), dtype=np.float64)
    for fold_idx in range(len(FOLDS)):
        mask = fold_index == fold_idx
        inv = inverse[mask]
        cluster[:, 3 * fold_idx] = np.bincount(
            inv, minlength=len(unique_user)
        ).astype(np.float64)
        cluster[:, 3 * fold_idx + 1] = np.bincount(
            inv, weights=error0[mask] ** 2, minlength=len(unique_user)
        )
        cluster[:, 3 * fold_idx + 2] = np.bincount(
            inv, weights=error1[mask] ** 2, minlength=len(unique_user)
        )

    rng = np.random.default_rng(20260828 + 79)
    delta_mse_draws: list[float] = []
    delta_rmsle_draws: list[float] = []
    normalized_weights = FOLD_WEIGHTS / FOLD_WEIGHTS.sum()
    for _ in range(50):
        counts = rng.poisson(1.0, size=(20, len(unique_user))).astype(np.float64)
        sums = counts @ cluster
        fold_delta_mse = []
        fold_delta_rmsle = []
        for fold_idx in range(len(FOLDS)):
            denominator = sums[:, 3 * fold_idx]
            mse0 = sums[:, 3 * fold_idx + 1] / denominator
            mse1 = sums[:, 3 * fold_idx + 2] / denominator
            fold_delta_mse.append(mse1 - mse0)
            fold_delta_rmsle.append(np.sqrt(mse1) - np.sqrt(mse0))
        delta_mse_draws.extend(
            (np.column_stack(fold_delta_mse) @ normalized_weights).tolist()
        )
        delta_rmsle_draws.extend(
            (np.column_stack(fold_delta_rmsle) @ normalized_weights).tolist()
        )

    dmse = np.asarray(delta_mse_draws, dtype=np.float64)
    drmsle = np.asarray(delta_rmsle_draws, dtype=np.float64)
    summary = {
        "method": "Poisson user-cluster bootstrap; same user multiplier across folds",
        "seed": 20260828 + 79,
        "replicates": int(len(dmse)),
        "delta_mse_ci_2_5": float(np.quantile(dmse, 0.025)),
        "delta_mse_ci_97_5": float(np.quantile(dmse, 0.975)),
        "delta_rmsle_ci_2_5": float(np.quantile(drmsle, 0.025)),
        "delta_rmsle_ci_97_5": float(np.quantile(drmsle, 0.975)),
        "p_delta_mse_lt_0": float(np.mean(dmse < 0.0)),
    }
    return summary, dmse, drmsle


def historical_sbvc_audit() -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    a1 = pd.read_parquet(E75 / "clean_forward_predictions.parquet")
    a2 = pd.read_parquet(E75 / "a2_clean_forward_predictions.parquet")
    require(len(a1) == len(a2), "A1/A2 clean-forward row count mismatch")
    require(
        np.array_equal(a1["user_id"].to_numpy(), a2["user_id"].to_numpy()),
        "A1/A2 user order mismatch",
    )
    require(
        np.array_equal(a1["cutoff"].astype(str), a2["cutoff"].astype(str)),
        "A1/A2 cutoff order mismatch",
    )

    cutoff_values = a1["cutoff"].astype(str).to_numpy()
    fold_masks = [cutoff_values == cutoff for cutoff in FOLDS]
    require(all(mask.any() for mask in fold_masks), "Missing clean SBVC fold")
    fold_index = np.full(len(a1), -1, dtype=np.int8)
    for idx, mask in enumerate(fold_masks):
        fold_index[mask] = idx
    require((fold_index >= 0).all(), "Unexpected cutoff in clean-forward artifacts")

    component_matrix, component_names, provenance = build_oof_component_matrix(a1)
    target_log = a1["target_log"].to_numpy(np.float64)
    matched_z = build_composition_matched_baseline(
        target_log, component_matrix, component_names, fold_masks
    )
    raw_joint = (
        JOINT_COEFFICIENTS[0] * a1["u_raw_365"].to_numpy(np.float64)
        + JOINT_COEFFICIENTS[1] * a2["u_raw_A2"].to_numpy(np.float64)
    )

    published = pd.read_csv(E76 / "out" / "s12_sbvc_folds.csv").set_index("cutoff")
    fold_rows: list[dict] = []
    error0_all = np.empty(len(a1), dtype=np.float64)
    error1_all = np.empty(len(a1), dtype=np.float64)
    d_perp_all = np.empty(len(a1), dtype=np.float64)
    max_artifact_difference = 0.0

    for fold_idx, (cutoff, mask) in enumerate(zip(FOLDS, fold_masks)):
        n = int(mask.sum())
        ones = np.ones((n, 1), dtype=np.float64)
        residual = target_log[mask] - matched_z[mask]
        full_basis = np.hstack(
            [ones, matched_z[mask, None], component_matrix[mask]]
        )
        min_basis = np.hstack([ones, matched_z[mask, None]])
        d_perp = project_out(raw_joint[mask], full_basis)
        d_min_projection = project_out(raw_joint[mask], min_basis)
        b = float(d_perp @ residual / n)
        g = float(d_perp @ d_perp / n)
        b_min = float(d_min_projection @ residual / n)
        g_min = float(d_min_projection @ d_min_projection / n)
        artifact_differences = [
            abs(b - float(published.loc[cutoff, "b"])),
            abs(g - float(published.loc[cutoff, "G"])),
            abs(b_min - float(published.loc[cutoff, "b_s"])),
            abs(g_min - float(published.loc[cutoff, "G_s"])),
        ]
        max_artifact_difference = max(max_artifact_difference, *artifact_differences)

        corrected_residual = residual - AMPLITUDE * d_perp
        delta = corrected_residual**2 - residual**2
        baseline_rmsle = rms(residual)
        corrected_rmsle = rms(corrected_residual)
        delta_mse = float(delta.mean())
        delta_rmsle = corrected_rmsle - baseline_rmsle
        sign = "gain" if delta_mse < 0.0 else ("tie" if delta_mse == 0.0 else "loss")

        error0_all[mask] = residual
        error1_all[mask] = corrected_residual
        d_perp_all[mask] = d_perp
        fold_rows.append(
            {
                "cutoff": cutoff,
                "n": n,
                "amplitude": AMPLITUDE,
                "rho_post_projection": correlation(d_perp, residual),
                "b": b,
                "G": g,
                "baseline_RMSLE": baseline_rmsle,
                "corrected_RMSLE": corrected_rmsle,
                "Delta_MSE": delta_mse,
                "Delta_RMSLE": delta_rmsle,
                "sign": sign,
                "preclip_nonpositive": int(
                    np.sum(matched_z[mask] + AMPLITUDE * d_perp <= 0.0)
                ),
            }
        )

    # Reproducing the published b/G is a hard provenance check on the SBVC mechanism.
    require(
        max_artifact_difference <= 5e-12,
        f"SBVC reconstruction mismatch: max b/G difference={max_artifact_difference}",
    )
    folds_frame = pd.DataFrame(fold_rows)
    normalized_weights = FOLD_WEIGHTS / FOLD_WEIGHTS.sum()
    weighted_delta_mse = float(folds_frame["Delta_MSE"].to_numpy() @ normalized_weights)
    weighted_delta_rmsle = float(
        folds_frame["Delta_RMSLE"].to_numpy() @ normalized_weights
    )
    bootstrap, dmse_draws, drmsle_draws = bootstrap_weighted_metrics(
        a1["user_id"].to_numpy(np.int64),
        fold_index,
        error0_all,
        error1_all,
    )
    result = {
        "mechanism": (
            "Frozen EXP075 joint raw correction, projected out of [1, z_match, "
            "deduplicated EXP076 OOF component span] per fold; fixed amplitude 0.40"
        ),
        "amplitude": AMPLITUDE,
        "folds": FOLDS,
        "recency_weights": FOLD_WEIGHTS.tolist(),
        "joint_coefficients": JOINT_COEFFICIENTS.tolist(),
        "component_count_after_deduplication": int(component_matrix.shape[1]),
        "component_names_after_deduplication": component_names,
        "source_file_count": int(len(set(provenance.values()))),
        "max_abs_b_G_difference_vs_EXP076_s12": max_artifact_difference,
        "matched_z_min": float(matched_z.min()),
        "matched_z_max": float(matched_z.max()),
        "weighted_delta_mse": weighted_delta_mse,
        "weighted_delta_rmsle": weighted_delta_rmsle,
        "folds_not_worse": int(np.sum(folds_frame["Delta_MSE"] <= 0.0)),
        "latest_fold_delta_mse": float(folds_frame.iloc[-1]["Delta_MSE"]),
        "bootstrap": bootstrap,
    }
    return result, folds_frame, dmse_draws, drmsle_draws


def mathematical_audit(z_alpha: np.ndarray, d_perp: np.ndarray) -> dict:
    d_a100 = np.maximum(z_alpha + d_perp, 0.0) - z_alpha
    d_a040 = np.maximum(z_alpha + AMPLITUDE * d_perp, 0.0) - z_alpha
    sigma100 = rms(d_a100)
    sigma040 = rms(d_a040)
    break_even_rho = sigma040 / (2.0 * BASELINE_PUBLIC_RMSLE)
    amplitude_matched_rho = sigma040 / BASELINE_PUBLIC_RMSLE

    sbvc = json.loads((E76 / "out" / "s12_agg.json").read_text(encoding="utf-8"))
    posterior = json.loads(
        (E76 / "out" / "s18_decision.json").read_text(encoding="utf-8")
    )
    scenarios = [
        ("SBVC post-projection prior", float(sbvc["rho_strong_postproj"]), False),
        ("SBVC min-projection prior", float(sbvc["rho_strong"]), False),
        (
            "posterior lower estimate",
            float(posterior["posterior_tight_postproj"]["post_mu"]),
            False,
        ),
        (
            "posterior central estimate",
            float(posterior["posterior_wide_postproj"]["post_mu"]),
            False,
        ),
        ("realised public rho (diagnostic only)", float(posterior["observed"]["rho"]), True),
        ("rho = 0", 0.0, False),
    ]
    scenario_rows = []
    for name, rho, diagnostic_only in scenarios:
        delta_mse = sigma040**2 - 2.0 * rho * sigma040 * BASELINE_PUBLIC_RMSLE
        delta_rmsle = math.sqrt(BASELINE_PUBLIC_RMSLE**2 + delta_mse) - BASELINE_PUBLIC_RMSLE
        scenario_rows.append(
            {
                "scenario": name,
                "rho": rho,
                "diagnostic_only": diagnostic_only,
                "Delta_MSE": delta_mse,
                "Delta_RMSLE": delta_rmsle,
            }
        )

    no_signal_mse_100 = sigma100**2
    no_signal_mse_040 = sigma040**2
    no_signal_rmsle_100 = (
        math.sqrt(BASELINE_PUBLIC_RMSLE**2 + no_signal_mse_100)
        - BASELINE_PUBLIC_RMSLE
    )
    no_signal_rmsle_040 = (
        math.sqrt(BASELINE_PUBLIC_RMSLE**2 + no_signal_mse_040)
        - BASELINE_PUBLIC_RMSLE
    )
    return {
        "baseline_public_RMSLE": BASELINE_PUBLIC_RMSLE,
        "amplitude": AMPLITUDE,
        "rms_applied_delta_A100": sigma100,
        "rms_applied_delta_A040": sigma040,
        "break_even_rho_A040": break_even_rho,
        "rho_for_which_A040_is_optimal": amplitude_matched_rho,
        "scenarios": scenario_rows,
        "no_signal_downside": {
            "A100_Delta_MSE": no_signal_mse_100,
            "A100_Delta_RMSLE": no_signal_rmsle_100,
            "A040_Delta_MSE": no_signal_mse_040,
            "A040_Delta_RMSLE": no_signal_rmsle_040,
            "A040_over_A100_Delta_MSE": no_signal_mse_040 / no_signal_mse_100,
            "A040_over_A100_Delta_RMSLE": no_signal_rmsle_040 / no_signal_rmsle_100,
            "RMS_A040_over_A100": sigma040 / sigma100,
        },
    }


def write_and_check_candidate(
    sample: pd.DataFrame, z_alpha: np.ndarray, d_perp: np.ndarray
) -> tuple[dict, np.ndarray]:
    sample_ids = sample["user_id"].to_numpy(np.int64)
    delta_preclip_a100 = d_perp
    delta_preclip_a040 = AMPLITUDE * d_perp
    z_new = np.maximum(z_alpha + delta_preclip_a040, 0.0)
    predict = np.expm1(z_new)
    candidate = pd.DataFrame({"user_id": sample_ids, "predict": predict})
    candidate.to_csv(CANDIDATE_PATH, index=False, float_format="%.10f")
    candidate_sha = sha256(CANDIDATE_PATH)
    readback = pd.read_csv(CANDIDATE_PATH)
    readback_predict = readback["predict"].to_numpy(np.float64)
    readback_z = np.log1p(readback_predict)

    checks = {
        "path": str(CANDIDATE_PATH),
        "sha256": candidate_sha,
        "rows": int(len(readback)),
        "columns": list(readback.columns),
        "unique_user_id": int(readback["user_id"].nunique()),
        "same_sample_order": bool(
            np.array_equal(readback["user_id"].to_numpy(np.int64), sample_ids)
        ),
        "finite": bool(np.isfinite(readback_predict).all()),
        "nonnegative": bool((readback_predict >= 0.0).all()),
        "zero_count": int(np.sum(readback_predict == 0.0)),
        "min_predict": float(readback_predict.min()),
        "max_predict": float(readback_predict.max()),
        "mean_log1p": float(readback_z.mean()),
        "std_log1p": float(readback_z.std(ddof=0)),
        "RMS_z_new_minus_z_alpha": rms(z_new - z_alpha),
        "corr_z_new_z_alpha": correlation(z_new, z_alpha),
        "preclip_RMS_delta_A040_over_A100": rms(delta_preclip_a040)
        / rms(delta_preclip_a100),
        "preclip_corr_delta_A040_A100": correlation(
            delta_preclip_a040, delta_preclip_a100
        ),
        "max_abs_log_serialization_difference": float(
            np.max(np.abs(readback_z - z_new))
        ),
    }
    checks["format_pass"] = bool(
        checks["rows"] == 250_000
        and checks["columns"] == ["user_id", "predict"]
        and checks["unique_user_id"] == 250_000
        and checks["same_sample_order"]
        and checks["finite"]
        and checks["nonnegative"]
        and abs(checks["preclip_RMS_delta_A040_over_A100"] - 0.40) <= 1e-12
        and abs(checks["preclip_corr_delta_A040_A100"] - 1.0) <= 1e-12
    )
    return checks, z_new


def main() -> None:
    parity, sample, z_alpha, d_perp = parity_audit()
    json_dump(EXP_DIR / "parity.json", parity)
    if parity["status"] != "PASS":
        if CANDIDATE_PATH.exists():
            CANDIDATE_PATH.unlink()
        json_dump(
            EXP_DIR / "audit.json",
            {"verdict": "PARITY_FAIL", "amplitude": AMPLITUDE, "parity": parity},
        )
        print(json.dumps({"verdict": "PARITY_FAIL", "parity": parity}, indent=2))
        return

    historical, fold_metrics, dmse_draws, drmsle_draws = historical_sbvc_audit()
    fold_metrics.to_csv(EXP_DIR / "clean_sbvc_a040.csv", index=False)
    np.savez_compressed(
        EXP_DIR / "bootstrap_a040.npz",
        delta_mse=dmse_draws,
        delta_rmsle=drmsle_draws,
    )
    math_audit = mathematical_audit(z_alpha, d_perp)
    candidate, _ = write_and_check_candidate(sample, z_alpha, d_perp)

    no_signal_ratio = math_audit["no_signal_downside"]["A040_over_A100_Delta_MSE"]
    gates = {
        "1_EXP075_parity_PASS": parity["status"] == "PASS",
        "2_amplitude_exactly_0_40": AMPLITUDE == 0.40,
        "3_weighted_Delta_MSE_lt_0": historical["weighted_delta_mse"] < 0.0,
        "4_latest_fold_Delta_MSE_le_0": historical["latest_fold_delta_mse"] <= 0.0,
        "5_at_least_3_of_4_folds_not_worse": historical["folds_not_worse"] >= 3,
        "6_bootstrap_P_gain_ge_0_95": historical["bootstrap"]["p_delta_mse_lt_0"]
        >= 0.95,
        "7_no_signal_downside_materially_smaller": no_signal_ratio <= 0.25,
        "8_candidate_format_correct": candidate["format_pass"],
    }
    verdict = "GO" if all(gates.values()) else "NO_GO"
    candidate_removed = False
    if verdict != "GO" and CANDIDATE_PATH.exists():
        CANDIDATE_PATH.unlink()
        candidate_removed = True

    audit = {
        "verdict": verdict,
        "amplitude": AMPLITUDE,
        "gates": gates,
        "parity": parity,
        "clean_sbvc": historical,
        "mathematical_audit": math_audit,
        "candidate": candidate,
        "candidate_removed_after_no_go": candidate_removed,
        "candidate_exists_at_end": CANDIDATE_PATH.exists(),
        "scale_axis_closed": True,
    }
    json_dump(EXP_DIR / "audit.json", audit)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "weighted_delta_mse": historical["weighted_delta_mse"],
                "weighted_delta_rmsle": historical["weighted_delta_rmsle"],
                "bootstrap": historical["bootstrap"],
                "candidate": candidate,
                "candidate_removed_after_no_go": candidate_removed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
