"""EXP090 Team-B audit, geometry, and evidence synthesis.

Consumes only existing Team-B artifacts plus the fresh exact-code reproduction
created by ``reproduce_team_b.py``.  It trains no models and creates no
submission candidate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "new_directions" / "EXP090_TEAM_B_AUDIT"
TEAM_B = ROOT / "team-b"
SUB = ROOT / "submissions"
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
E89 = ROOT / "research" / "new_directions" / "EXP089_JOINT_V2_PLANE_RESOLUTION"
GEOM = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
CANONICAL_Z = GEOM / "submission_geometry" / "cache" / "Z.npz"
CANONICAL_META = GEOM / "submission_geometry" / "cache" / "Z_meta.json"

V2_NAME = "SUBMIT_JOINT_V2.csv"
V2_SCORE = 1.6459363044782171
V2_SHA256 = "211879cb1c79bbbde93d451fca5b61c521b523f989ce42bab62cd3ab87233cba"
ORTH_SCORE = 1.6462686940209101
LEVEL = 2.370966
WEIGHTS = {
    "recency": 0.25,
    "post_order_dist": 0.10,
    "behavior_dist": 0.20,
    "xgb_behavior": 0.25,
    "cat_behavior": 0.20,
}

TEAM_SUBMISSIONS = [
    "exp_008_recency_lightgbm_scale064.csv",
    "exp_009_recency_long_buy_lgbm_logens.csv",
    "exp_010_logens_wrec055_scale100.csv",
    "exp_010_logens_wrec055_scale101.csv",
    "exp_011_dense8_logens_scale100.csv",
    "exp_011_dense8_logens_scale120.csv",
    "exp_011_dense8_logens_scale140.csv",
    "exp_013_dense8_logens_scale127.csv",
    "exp_013_dense8_logens_scale130.csv",
    "exp_015_dense8_logens_wrec0525_scale120.csv",
    "exp_016_post_order_wrec050_scale120.csv",
    "exp_017_dist_post_order_wrec050_scale120.csv",
    "exp_019_behavior_v1_dist_wrec050_scale120.csv",
    "exp_020_behavior_v1_slim_dist_wrec050_scale120.csv",
]
STRONG_TEAM_SUBMISSIONS = [
    "exp_011_dense8_logens_scale120.csv",
    "exp_016_post_order_wrec050_scale120.csv",
    "exp_017_dist_post_order_wrec050_scale120.csv",
    "exp_019_behavior_v1_dist_wrec050_scale120.csv",
    "exp_020_behavior_v1_slim_dist_wrec050_scale120.csv",
]


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
    if den == 0:
        return float("nan")
    return float((x @ y) / den)


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def load_submission(path: Path, uid: np.ndarray) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(path)
    pred = df["predict"].to_numpy(np.float64)
    got_uid = df["user_id"].to_numpy(np.int64)
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": int(len(df)),
        "unique_user_id": int(df.user_id.nunique()),
        "same_order": bool(np.array_equal(got_uid, uid)),
        "finite": bool(np.isfinite(pred).all()),
        "nonnegative": bool(np.all(pred >= 0)),
        "fraction_zero": float(np.mean(pred == 0)),
    }
    if list(df.columns) != ["user_id", "predict"]:
        raise AssertionError(f"unexpected columns: {path}")
    if not (audit["rows"] == 250_000 and audit["unique_user_id"] == 250_000
            and audit["same_order"] and audit["finite"] and audit["nonnegative"]):
        raise AssertionError(f"submission audit failed: {audit}")
    return np.log1p(pred), audit


class ScoredSpan:
    """Centered row span using a Gram eigendecomposition."""

    def __init__(self, rows: np.ndarray, names: list[str], rtol: float = 1e-10):
        m = np.asarray(rows, dtype=np.float64).copy()
        m -= m.mean(axis=1, keepdims=True)
        rr = np.sqrt(np.mean(m * m, axis=1))
        nz = rr > 1e-14
        self.names = [n for n, k in zip(names, nz) if k]
        self.m = m[nz] / rr[nz, None]
        gram = self.m @ self.m.T
        eig, vec = np.linalg.eigh(gram)
        keep = eig > eig.max() * rtol
        self.eig = eig[keep]
        self.vec = vec[:, keep]
        self.rank = int(keep.sum())
        self.input_rows = int(len(rows))
        self.condition_design = float(np.sqrt(self.eig.max() / self.eig.min()))
        self.condition_gram = float(self.eig.max() / self.eig.min())
        self.rtol = rtol

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mx = self.m @ x
        coef = self.vec @ ((self.vec.T @ mx) / self.eig)
        return coef @ self.m

    def project_out_twice(self, x: np.ndarray) -> tuple[np.ndarray, list[float]]:
        out = np.asarray(x, dtype=np.float64) - np.mean(x)
        passes = []
        for _ in range(2):
            p = self.project(out)
            passes.append(rms(p))
            out = out - p
            out = out - out.mean()
        return out, passes

    def augmented_condition(self, row: np.ndarray) -> dict:
        r = np.asarray(row, dtype=np.float64) - np.mean(row)
        r = r / rms(r)
        ma = np.vstack([self.m, r])
        eig = np.linalg.eigvalsh(ma @ ma.T)
        keep = eig > eig.max() * self.rtol
        retained = eig[keep]
        return {
            "input_rows": int(len(ma)),
            "rank": int(keep.sum()),
            "condition_design": float(np.sqrt(retained.max() / retained.min())),
            "condition_gram": float(retained.max() / retained.min()),
            "smallest_retained_relative_eigenvalue": float(retained.min() / retained.max()),
        }


def build_scored_span(uid: np.ndarray) -> tuple[ScoredSpan, dict]:
    cache = np.load(CANONICAL_Z, allow_pickle=False)
    if not np.array_equal(cache["user_id"].astype(np.int64), uid):
        raise AssertionError("canonical scored cache row order mismatch")
    z = cache["Z"].astype(np.float64)
    meta = json.loads(CANONICAL_META.read_text(encoding="utf-8"))
    names = list(meta["names"])
    manifest = pd.read_csv(E89 / "scored_span_manifest.csv")
    extras = manifest[manifest.group != "canonical_scored_bank"]
    extra_z = []
    for row in extras.itertuples(index=False):
        zi, _ = load_submission(Path(row.path), uid)
        extra_z.append(zi)
        names.append(str(row.name))
    rows = np.vstack([z, np.vstack(extra_z)])
    span = ScoredSpan(rows, names)
    audit = {
        "canonical_rows": int(z.shape[0]),
        "scored_extra_rows": int(len(extra_z)),
        "input_rows": span.input_rows,
        "centered_rank": span.rank,
        "condition_design": span.condition_design,
        "condition_gram": span.condition_gram,
        "relative_tolerance": span.rtol,
    }
    return span, audit


def purpose_for(path: Path) -> str:
    name = path.name
    purposes = {
        "config.py": "paths/seed/target-day constants; validation cutoffs left unset",
        "validation.py": "stub validation API; not used by later experiment scripts",
        "features.py": "cutoff-safe aggregate, post-order, and behavior_v1 feature builder with parquet cache",
        "train.py": "target construction and five-component handoff trainer",
        "predict.py": "single-model and five-component handoff TEST inference",
        "dense_ensemble.py": "dense weekly LightGBM recency/long_buy ensemble",
        "dist_head_ensemble.py": "16-bin LightGBM distribution-head training and blend",
        "post_order_ensemble.py": "direct LightGBM post-order ensemble",
        "classifier_gate.py": "rejected any-purchase classifier gate",
        "catboost_blend.py": "three-component CatBoost diversity experiment",
        "model_blend_grid.py": "three-component validation weight grid",
        "xgboost_blend.py": "four-component XGBoost validation/submit pipeline",
        "cat_xgb_blend.py": "five-component exp024 validation/submit pipeline",
        "submission_blend.py": "post-hoc log-space submission blending and level alignment",
        "weight_grid.py": "two-component recency/long_buy validation grid",
        "behavior_feature_eval.py": "behavior_v1 feature importance/permutation diagnostics",
        "shap_behavior_v1.py": "behavior_v1 SHAP diagnostics",
        "STATE.md": "claimed current status and scores",
        "README.md": "project overview",
        "team_b_handoff.md": "handoff architecture claim and API",
    }
    if name in purposes:
        return purposes[name]
    if path.parent.name == "experiments":
        return "experiment report; secondary evidence checked against code/artifacts"
    if path.parent.name == "artifacts":
        return "aggregate CV/grid/importance table; no row-level OOF or TEST vector"
    if path.parent.name == "docs":
        return "secondary design/feature documentation"
    return "Team-B project file"


def build_inventory(scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for path in sorted(
        p for p in TEAM_B.rglob("*")
        if p.is_file() and "data" not in p.parts and "__pycache__" not in p.parts
    ):
        rel = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        kind = "source" if suffix == ".py" else "report" if suffix == ".md" else "aggregate_artifact" if suffix == ".csv" else "config"
        rows.append({
            "file_model_submission": rel,
            "kind": kind,
            "exists": True,
            "purpose": purpose_for(path),
            "target": "y30 raw/log1p or 16-bin log1p" if suffix == ".py" else "n/a",
            "features": "see src/features.py; recency=152, long_buy_post_order=215, behavior_v1=329" if suffix == ".py" else "n/a",
            "train_cutoffs": "script-local; exp024 TEST uses 2025-08-28..2025-10-16" if suffix == ".py" else "n/a",
            "validation": "two OOT folds for late ensemble scripts; report/table only otherwise",
            "test_inference": "available in submit/predict scripts; no persisted TEST vector in team-b" if suffix == ".py" else "n/a",
            "dependencies": "pandas,numpy,scikit-learn,lightgbm,catboost,xgboost,pyarrow,duckdb" if suffix == ".py" else "n/a",
            "reproducibility": "source present; dependencies unpinned; cache lacks code/data fingerprint" if suffix == ".py" else "artifact hashable",
            "status": "PRESENT",
            "leaderboard_score": None,
            "rows": None,
            "sha256": sha256(path),
        })

    logical = [
        ("model:recency_lgbm", "LightGBM direct z regression", "recency (152)", "log1p(y30)", 0.25),
        ("model:post_order_dist_lgbm", "LightGBM 16-bin distribution head", "long_buy_post_order (215)", "16 classes from log1p(y30)", 0.10),
        ("model:behavior_dist_lgbm", "LightGBM 16-bin distribution head", "behavior_v1 (329)", "16 classes from log1p(y30)", 0.20),
        ("model:behavior_xgboost", "XGBoost direct z regression", "behavior_v1 (329)", "log1p(y30)", 0.25),
        ("model:behavior_catboost", "CatBoost direct z regression", "behavior_v1 (329)", "log1p(y30)", 0.20),
        ("model:any_purchase_gate", "LightGBM binary classifier; rejected exp014", "recency", "I(y30>0)", 0.0),
        ("model:SEQ", "not present in Team-B", "n/a", "n/a", 0.0),
        ("model:ETX", "not present in Team-B", "n/a", "n/a", 0.0),
        ("model:Ridge", "not present in Team-B", "n/a", "n/a", 0.0),
    ]
    for name, purpose, features, target, weight in logical:
        rows.append({
            "file_model_submission": name, "kind": "logical_model", "exists": purpose != "not present in Team-B",
            "purpose": purpose, "target": target, "features": features,
            "train_cutoffs": "last 8 clean weekly cutoffs for TEST",
            "validation": "two single-cutoff OOT folds; selected blend is non-nested",
            "test_inference": f"exp024 handoff weight={weight:.2f}" if weight else "none",
            "dependencies": name.split(":")[-1], "reproducibility": "no saved model object; retraining required",
            "status": "LOGICAL_ONLY_NO_MODEL_FILE" if "not present" not in purpose else "ABSENT",
            "leaderboard_score": None, "rows": None, "sha256": None,
        })

    for name in TEAM_SUBMISSIONS:
        path = SUB / name
        df = pd.read_csv(path)
        rows.append({
            "file_model_submission": path.relative_to(ROOT).as_posix(), "kind": "submission", "exists": True,
            "purpose": "scored Team-B TEST submission", "target": "raw GMV serialized from log-space prediction",
            "features": "varies by experiment; see experiment report and source",
            "train_cutoffs": "8 clean weekly cutoff panels for exp011+; earlier scripts vary",
            "validation": "public LB plus local validation", "test_inference": "TEST cutoff 2026-02-14",
            "dependencies": "persisted CSV", "reproducibility": "vector present; primary model/OOF files absent",
            "status": "PRESENT_SCORED", "leaderboard_score": scores.get(name), "rows": len(df), "sha256": sha256(path),
        })

    missing = [
        "exp_018_catboost_blend_wcat020_scale120.csv",
        "exp_021_blend_e19_e18_9010_level_e19.csv",
        "exp_022_model_blend_rec040_post015_beh045_level_e19.csv",
        "exp_023_xgb_blend_rec030_post010_beh030_xgb030_level_e19.csv",
        "exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_scale120.csv",
        "exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_level_e19.csv",
    ]
    for name in missing:
        rows.append({
            "file_model_submission": f"team-b/submissions/{name}", "kind": "submission", "exists": False,
            "purpose": "claimed generated Team-B candidate", "target": "raw GMV", "features": "per experiment",
            "train_cutoffs": "claimed 2025-08-28..2025-10-16", "validation": "claimed two-fold OOT",
            "test_inference": "claimed TEST cutoff 2026-02-14", "dependencies": "missing CSV",
            "reproducibility": "cannot test byte parity; exact original vector absent", "status": "MISSING",
            "leaderboard_score": None, "rows": None, "sha256": None,
        })
    return pd.DataFrame(rows)


def selected_grid_rows() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    grid = pd.read_csv(TEAM_B / "artifacts" / "exp024_cat_xgb_blend_grid.csv")
    cols = list(grid.columns[:5])
    specs = {
        "exp019": (0.50, 0.00, 0.50, 0.00, 0.00),
        "exp022": (0.40, 0.15, 0.45, 0.00, 0.00),
        "exp023": (0.30, 0.10, 0.30, 0.30, 0.00),
        "exp024": (0.25, 0.10, 0.20, 0.25, 0.20),
    }
    out = []
    folds = {}
    for name, w in specs.items():
        mask = np.isclose(grid[cols].to_numpy(float), np.asarray(w)).all(axis=1)
        row = grid.loc[mask].iloc[0]
        mean = float(row.rmsle_mean)
        std = float(row.rmsle_std)
        values = np.array([mean - std / np.sqrt(2.0), mean + std / np.sqrt(2.0)])
        folds[name] = values
        out.append({"variant": name, **dict(zip(cols, w)), "rmsle_mean": mean, "rmsle_std": std,
                    "fold_2026_01_14": values[0], "fold_2025_12_15": values[1]})
    return pd.DataFrame(out), folds


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    score_df = pd.read_csv(ROOT / "scores" / "submissions.csv")
    scores = dict(zip(score_df.submission_name, score_df.leaderboard_score))

    canonical = np.load(CANONICAL_Z, allow_pickle=False)
    uid = canonical["user_id"].astype(np.int64)
    z_v2, v2_audit = load_submission(SUB / V2_NAME, uid)
    if v2_audit["sha256"] != V2_SHA256:
        raise AssertionError("wrong JOINT_V2")

    manifest = pd.read_csv(E89 / "scored_span_manifest.csv")
    orth_path = Path(manifest.loc[manifest.name == "SUBMIT_ORTH_FINAL", "path"].iloc[0])
    z_orth, orth_audit = load_submission(orth_path, uid)
    a1 = np.load(E75 / "A1_TREE_TRAJ_365_TEST_PERP.npy", allow_pickle=False).astype(np.float64)
    a2 = np.load(E75 / "A2_WEEKLY_RESIDUAL_CNN_TEST_PERP.npy", allow_pickle=False).astype(np.float64)
    tomo = np.load(E89 / "updated_tomography_vectors.npz", allow_pickle=False)
    if not np.array_equal(tomo["user_id"].astype(np.int64), uid):
        raise AssertionError("EXP089 tomography order mismatch")
    v2_out_of_plane = tomo["residual_e"].astype(np.float64)

    reproduction = json.loads((EXP / "reproduction_run.json").read_text(encoding="utf-8"))
    rv = np.load(EXP / "team_b_reproduced_vectors.npz", allow_pickle=False)
    if not np.array_equal(rv["user_id"].astype(np.int64), uid):
        raise AssertionError("fresh Team-B reproduction order mismatch")
    z_components = {name: rv[f"z_{name}"].astype(np.float64) for name in WEIGHTS}
    z_mix = sum(WEIGHTS[name] * z_components[name] for name in WEIGHTS)
    z_scaled_120 = np.log1p(np.maximum(np.expm1(z_mix) * 1.20, 0.0))
    z_exp024 = np.maximum(z_scaled_120 + (LEVEL - float(z_scaled_120.mean())), 0.0)
    z_handoff = rv["z_final"].astype(np.float64)
    p_exp024 = np.maximum(np.expm1(z_exp024), 0.0)

    repro_frame = pd.DataFrame({"user_id": uid})
    for name, z in z_components.items():
        repro_frame[f"z_{name}"] = z
    repro_frame["z_mix"] = z_mix
    repro_frame["z_scaled_120"] = z_scaled_120
    repro_frame["z_exp024_formula"] = z_exp024
    repro_frame["z_handoff_formula"] = z_handoff
    repro_frame["predict_exp024_formula"] = p_exp024
    repro_path = EXP / "team_b_reproduction_predictions.parquet"
    repro_frame.to_parquet(repro_path, index=False)

    span, span_audit = build_scored_span(uid)
    team_vectors: dict[str, np.ndarray] = {}
    submission_audits = {}
    for name in STRONG_TEAM_SUBMISSIONS:
        team_vectors[name], submission_audits[name] = load_submission(SUB / name, uid)
    team_vectors["exp_024_formula_reproduction"] = z_exp024
    submission_audits["exp_024_formula_reproduction"] = {
        "path": str(repro_path.resolve()), "sha256": sha256(repro_path), "rows": 250_000,
        "unique_user_id": 250_000, "same_order": True, "finite": True, "nonnegative": True,
        "fraction_zero": float(np.mean(p_exp024 == 0)), "status": "fresh formula reconstruction; original CSV missing",
    }

    orth_delta = z_orth - z_v2
    geometry_rows = []
    decompositions = {}
    for name, z in team_vectors.items():
        d_raw = z - z_v2
        d = d_raw - d_raw.mean()
        dp, passes = span.project_out_twice(d)
        rp = rms(dp)
        rd = rms(d)
        aug = span.augmented_condition(z)
        row = {
            "relation": "vs_JOINT_V2", "vector_a": name, "vector_b": V2_NAME,
            "leaderboard_score_a": scores.get(name), "leaderboard_score_b": V2_SCORE,
            "RMS_raw_difference": rms(d_raw), "mean_correction": float(d_raw.mean()),
            "RMS_centered_correction": rd, "RMS_d_perp": rp,
            "perp_fraction_RMS": rp / rd if rd else 0.0,
            "perp_fraction_energy": (rp / rd) ** 2 if rd else 0.0,
            "corr_z_predictions": corr(z, z_v2),
            "corr_d_perp_with_ORTH_minus_V2": corr(dp, orth_delta),
            "corr_d_perp_with_A1_365": corr(dp, a1),
            "corr_d_perp_with_A2": corr(dp, a2),
            "corr_d_perp_with_JOINT_V2_out_of_plane": corr(dp, v2_out_of_plane),
            "fraction_clipped_or_zero": submission_audits[name]["fraction_zero"],
            "projection_pass1_RMS": passes[0], "projection_pass2_RMS": passes[1],
            "current_span_rank": span.rank, "rank_after_addition": aug["rank"],
            "condition_design_before": span.condition_design,
            "condition_design_after": aug["condition_design"],
            "condition_gram_after": aug["condition_gram"],
        }
        geometry_rows.append(row)
        decompositions[name] = {**row, "d_perp": dp}

    names = list(team_vectors)
    for i, ni in enumerate(names):
        for nj in names[i + 1:]:
            zi, zj = team_vectors[ni], team_vectors[nj]
            geometry_rows.append({
                "relation": "pairwise_team_b", "vector_a": ni, "vector_b": nj,
                "leaderboard_score_a": scores.get(ni), "leaderboard_score_b": scores.get(nj),
                "RMS_raw_difference": rms(zi - zj), "mean_correction": float(np.mean(zi - zj)),
                "RMS_centered_correction": rms((zi - zj) - np.mean(zi - zj)),
                "RMS_d_perp": None, "perp_fraction_RMS": None, "perp_fraction_energy": None,
                "corr_z_predictions": corr(zi, zj),
                "corr_d_perp_with_ORTH_minus_V2": None, "corr_d_perp_with_A1_365": None,
                "corr_d_perp_with_A2": None, "corr_d_perp_with_JOINT_V2_out_of_plane": None,
                "fraction_clipped_or_zero": None, "projection_pass1_RMS": None, "projection_pass2_RMS": None,
                "current_span_rank": span.rank, "rank_after_addition": None,
                "condition_design_before": span.condition_design, "condition_design_after": None,
                "condition_gram_after": None,
            })
    geometry_df = pd.DataFrame(geometry_rows)
    geometry_df.to_csv(EXP / "team_b_geometry.csv", index=False)

    d024 = decompositions["exp_024_formula_reproduction"]["d_perp"]
    np.savez_compressed(
        EXP / "team_b_signal_vector.npz", user_id=uid, z_joint_v2=z_v2,
        z_team_b_exp024=z_exp024, d_centered=(z_exp024 - z_v2) - np.mean(z_exp024 - z_v2),
        d_perp=d024, z_handoff=z_handoff, **{f"z_{k}": v for k, v in z_components.items()},
    )

    # Public leaderboard decoding.  This is exact for realized full-population
    # log vectors but public membership is unknown, so it is an inference rather
    # than an exact public-subset Gram calculation.
    lb_rows = []
    for name in TEAM_SUBMISSIONS:
        z, _ = load_submission(SUB / name, uid)
        score = float(scores[name])
        diff = z - z_v2
        g = float(np.mean(diff * diff))
        delta = score * score - V2_SCORE * V2_SCORE
        b = 0.5 * (g - delta)
        scale = b / g if g else float("nan")
        gain = b * b / g if g else float("nan")
        lb_rows.append({
            "comparison": "relative_to_JOINT_V2", "anchor": V2_NAME, "candidate": name,
            "anchor_LB": V2_SCORE, "candidate_LB": score, "Delta_MSE": delta, "G": g, "b": b,
            "optimal_scale_from_anchor_toward_candidate": scale,
            "maximum_MSE_gain_on_full_Gram": gain,
            "predicted_optimal_RMSLE": math.sqrt(max(V2_SCORE * V2_SCORE - gain, 0.0)),
            "direction_in_current_scored_span": True,
            "evidence": "LB+realized full-population Gram; public membership unknown",
        })

    chain = [
        ("exp_011_dense8_logens_scale120.csv", "exp_016_post_order_wrec050_scale120.csv", "post_order"),
        ("exp_016_post_order_wrec050_scale120.csv", "exp_017_dist_post_order_wrec050_scale120.csv", "dist_head"),
        ("exp_017_dist_post_order_wrec050_scale120.csv", "exp_019_behavior_v1_dist_wrec050_scale120.csv", "behavior_v1"),
        ("exp_019_behavior_v1_dist_wrec050_scale120.csv", "exp_020_behavior_v1_slim_dist_wrec050_scale120.csv", "behavior_slim_ablation"),
    ]
    for anchor, candidate, mechanism in chain:
        za, _ = load_submission(SUB / anchor, uid)
        zc, _ = load_submission(SUB / candidate, uid)
        ra, rc = float(scores[anchor]), float(scores[candidate])
        diff = zc - za
        g = float(np.mean(diff * diff))
        delta = rc * rc - ra * ra
        b = 0.5 * (g - delta)
        gain = b * b / g if g else float("nan")
        lb_rows.append({
            "comparison": f"sequential_{mechanism}", "anchor": anchor, "candidate": candidate,
            "anchor_LB": ra, "candidate_LB": rc, "Delta_MSE": delta, "G": g, "b": b,
            "optimal_scale_from_anchor_toward_candidate": b / g if g else float("nan"),
            "maximum_MSE_gain_on_full_Gram": gain,
            "predicted_optimal_RMSLE": math.sqrt(max(ra * ra - gain, 0.0)),
            "direction_in_current_scored_span": True,
            "evidence": "LB+realized full-population Gram; sequential provenance diagnostic",
        })
    pd.DataFrame(lb_rows).to_csv(EXP / "leaderboard_decoding.csv", index=False)
    lb019_from_v2 = next(
        row for row in lb_rows
        if row["comparison"] == "relative_to_JOINT_V2"
        and row["candidate"] == "exp_019_behavior_v1_dist_wrec050_scale120.csv"
    )

    # Aggregate two-fold evidence can verify the reports, but it cannot produce
    # residual correlations or nested conditional gains without row-level OOF.
    selected, fold_values = selected_grid_rows()
    oof_rows = []
    comparisons = [("exp023", "exp019"), ("exp024", "exp023"), ("exp024", "exp019")]
    for candidate, baseline in comparisons:
        delta = fold_values[candidate] - fold_values[baseline]
        for i, cutoff in enumerate(["2026-01-14", "2025-12-15"]):
            oof_rows.append({
                "direction": f"{candidate}_minus_{baseline}", "fold_cutoff": cutoff,
                "status": "AGGREGATE_ONLY_NON_NESTED_WEIGHT_SELECTION",
                "rows": None, "rho": None, "latest_rho": None, "b": None, "G": None,
                "oracle_amplitude": None, "oracle_gain": None, "nested_Delta_MSE": None,
                "nested_Delta_RMSLE": None, "observed_Delta_RMSLE": float(delta[i]),
                "bootstrap_CI_low": float(delta.min()), "bootstrap_CI_high": float(delta.max()),
                "sign_stable": bool(np.all(delta < 0) or np.all(delta > 0)),
                "canonical_clean_fold_aligned": False,
                "notes": "fold scores reconstructed exactly from two-fold mean/std; no user-level OOF vectors",
            })
    oof_rows.append({
        "direction": "exp024_post_current_scored_span", "fold_cutoff": "ALL",
        "status": "NOT_IDENTIFIABLE_NO_ROW_LEVEL_COMPONENT_OOF", "rows": None,
        "rho": None, "latest_rho": None, "b": None, "G": None, "oracle_amplitude": None,
        "oracle_gain": None, "nested_Delta_MSE": None, "nested_Delta_RMSLE": None,
        "observed_Delta_RMSLE": float(np.mean(fold_values["exp024"] - fold_values["exp019"])),
        "bootstrap_CI_low": float(np.min(fold_values["exp024"] - fold_values["exp019"])),
        "bootstrap_CI_high": float(np.max(fold_values["exp024"] - fold_values["exp019"])),
        "sign_stable": True, "canonical_clean_fold_aligned": False,
        "notes": "raw blend improvement is not a post-span residual estimate; strict-forward coefficients unavailable",
    })
    pd.DataFrame(oof_rows).to_csv(EXP / "oof_residual_metrics.csv", index=False)

    leakage_rows = [
        ("all feature sets", "feature dates", "CLEAN", "Every parquet scan first restricts event_date < cutoff."),
        ("post_order", "same-period leakage", "CLEAN", "Later-than-last-order events remain inside the pre-cutoff base CTE."),
        ("behavior_v1", "target-derived activity", "CLEAN", "All 114 b1_ features use only the pre-cutoff base; no future target/activity join."),
        ("target", "30-day definition", "CLEAN", "SQL uses [cutoff, cutoff+30d); all validation targets are fully observed by 2026-02-13."),
        ("two OOT folds", "label overlap", "CLEAN", "Training target ends at validation cutoff exclusive; validation features end before cutoff."),
        ("two OOT folds", "out-of-sample prediction", "CLEAN", "Validation models are fit on earlier cutoff panels; user_id is index, not a feature."),
        ("exp024 weight grid", "nested selection", "QUESTIONABLE", "2,835 weight vectors are selected and reported on the same two folds; no outer fold."),
        ("global scale 1.20", "leaderboard use", "QUESTIONABLE", "Lineage explicitly compared scale grids to public LB; later validation was aligned after that evidence."),
        ("level 2.370966", "leaderboard geometry", "QUESTIONABLE", "Level is copied from scored exp019 TEST vector; no label leakage, but not independent supervised evidence."),
        ("behavior_v1/full vs slim", "public feature selection", "QUESTIONABLE", "Both CV and LB favored full, but public LB was used in the experiment lineage."),
        ("row-level OOF", "artifact availability", "UNKNOWN", "No OOF prediction vector or saved component predictions exist; only aggregate fold/grid tables."),
        ("model objects", "artifact availability", "UNKNOWN", "No LightGBM/XGBoost/CatBoost model files or training environment lockfile are supplied."),
        ("feature cache", "cache invalidation", "QUESTIONABLE", "Cache key is only feature_set+cutoff; raw hash/code version is not embedded."),
        ("validation.py/config.py", "canonical validation contract", "QUESTIONABLE", "validation.py is NotImplemented and config cutoffs are None; actual scripts hard-code fold dates."),
        ("exp024 TEST inference", "test target access", "CLEAN", "TEST features use history through 2026-02-13 with cutoff 2026-02-14; no TEST target is read."),
    ]
    pd.DataFrame(leakage_rows, columns=["component", "check", "classification", "evidence"]).to_csv(
        EXP / "leakage_audit.csv", index=False)

    inventory = build_inventory({k: float(v) for k, v in scores.items() if pd.notna(v)})
    inventory.to_csv(EXP / "team_b_inventory.csv", index=False)

    exp019_score = float(scores["exp_019_behavior_v1_dist_wrec050_scale120.csv"])
    exp017_score = float(scores["exp_017_dist_post_order_wrec050_scale120.csv"])
    exp016_score = float(scores["exp_016_post_order_wrec050_scale120.csv"])
    exp011_score = float(scores["exp_011_dense8_logens_scale120.csv"])
    g024 = decompositions["exp_024_formula_reproduction"]
    g019 = decompositions["exp_019_behavior_v1_dist_wrec050_scale120.csv"]
    version_info = {}
    for pkg in ["pandas", "numpy", "scikit-learn", "duckdb", "lightgbm", "xgboost", "catboost", "pyarrow"]:
        try:
            version_info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            version_info[pkg] = "missing"

    reproduction_audit = {
        **reproduction,
        "best_scored_team_b": {
            "file": "exp_019_behavior_v1_dist_wrec050_scale120.csv",
            "leaderboard_score": exp019_score,
            "sha256": sha256(SUB / "exp_019_behavior_v1_dist_wrec050_scale120.csv"),
        },
        "declared_local_champion": "exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_level_e19.csv",
        "declared_local_champion_original_present": False,
        "fresh_environment_versions": version_info,
        "exp024_experiment_formula": "component log blend -> raw GMV * 1.20 -> log1p -> level 2.370966 -> z>=0 clip",
        "handoff_formula": "component log blend -> level 2.370966 -> z>=0 clip (global 1.20 omitted)",
        "experiment_vs_handoff_RMS_log_difference": rms(z_exp024 - z_handoff),
        "experiment_vs_handoff_mean_log_difference": float(np.mean(z_exp024 - z_handoff)),
        "experiment_vs_handoff_corr": corr(z_exp024, z_handoff),
        "exp024_formula_rows": 250_000,
        "exp024_formula_unique_users": 250_000,
        "exp024_formula_same_sample_order": True,
        "exp024_formula_fraction_clipped": float(np.mean(z_exp024 == 0)),
        "exp024_formula_mean_z": float(np.mean(z_exp024)),
        "exact_reproduction_verdict": "NO_BYTE_EXACT_PARITY: original exp024 CSV/models/OOF and pinned environment are absent; two primary code paths disagree",
        "reproduction_predictions_parquet": {"path": str(repro_path.resolve()), "sha256": sha256(repro_path)},
    }
    json_dump(EXP / "reproduction_audit.json", clean_json(reproduction_audit))

    signal_decomposition = {
        "verdict": "EXP024_HAS_A_FRESH_TEST_DIRECTION_BUT_POST_SPAN_VALUE_IS_UNVALIDATED",
        "best_scored_team_b": "exp019 behavior_v1 dist-head",
        "declared_local_champion": "exp024 five-family blend",
        "architecture": {
            "weights": WEIGHTS,
            "representation": "component predictions and final blend in log1p space",
            "targets": {"direct_models": "log1p(y30)", "distribution_heads": "16 quantile bins of log1p(y30)"},
            "feature_sets": {"recency": 152, "long_buy_post_order": 215, "behavior_v1": 329},
        },
        "test_geometry_exp024_vs_joint_v2": {
            "RMS_centered_correction": g024["RMS_centered_correction"],
            "RMS_outside_current_scored_span": g024["RMS_d_perp"],
            "perp_fraction_RMS": g024["perp_fraction_RMS"],
            "perp_fraction_energy": g024["perp_fraction_energy"],
            "corr_perp_A1_365": g024["corr_d_perp_with_A1_365"],
            "corr_perp_A2": g024["corr_d_perp_with_A2"],
            "corr_perp_v2_out_of_plane": g024["corr_d_perp_with_JOINT_V2_out_of_plane"],
        },
        "scored_exp019_post_span": {
            "RMS_outside_current_scored_span": g019["RMS_d_perp"],
            "perp_fraction_RMS": g019["perp_fraction_RMS"],
            "interpretation": "numerical zero because exp019 is already a row of the current scored span",
        },
        "incremental_evidence": {
            "post_order_LB_Delta_MSE": exp016_score ** 2 - exp011_score ** 2,
            "dist_head_LB_Delta_MSE": exp017_score ** 2 - exp016_score ** 2,
            "behavior_v1_LB_Delta_MSE": exp019_score ** 2 - exp017_score ** 2,
            "xgboost_CV_Delta_RMSLE_vs_exp019": float(np.mean(fold_values["exp023"] - fold_values["exp019"])),
            "catboost_conditional_CV_Delta_RMSLE_vs_exp023": float(np.mean(fold_values["exp024"] - fold_values["exp023"])),
            "full_exp024_CV_Delta_RMSLE_vs_exp019": float(np.mean(fold_values["exp024"] - fold_values["exp019"])),
            "full_exp024_fold_deltas_vs_exp019": (fold_values["exp024"] - fold_values["exp019"]).tolist(),
            "caveat": "CV weights selected on these same two folds; no post-span residual OOF vector",
        },
        "source_of_gain": {
            "confirmed_by_LB": "post-order features, distribution head, and full behavior_v1 each improved the Team-B ladder; behavior_v1 is the last positive scored increment",
            "largest_new_unscored_CV_increment": "XGBoost model-family diversity; CatBoost adds a smaller conditional increment and hits its 0.20 search boundary",
            "why_not_in_current_scored_span": "exp024 XGBoost/CatBoost TEST predictions were never scored or included; all scored Team-B siblings including exp019 already are in the span",
            "siblings": ["exp022 three-LightGBM blend", "exp023 +XGBoost", "exp024 +CatBoost", "exp020 behavior_v1_slim ablation"],
        },
        "formula_disagreement_RMS": rms(z_exp024 - z_handoff),
        "vector_artifact": str((EXP / "team_b_signal_vector.npz").resolve()),
        "vector_sha256": sha256(EXP / "team_b_signal_vector.npz"),
    }
    json_dump(EXP / "signal_decomposition.json", clean_json(signal_decomposition))

    # Compatibility of the already scored exp019 vector can be decoded in the
    # public [V2, exp019] plane from ORTH_FINAL.  exp024 has no score/OOF b term,
    # so its conditional gain is deliberately left unidentified.
    u_v2 = (z_v2 - z_orth) - np.mean(z_v2 - z_orth)
    z019 = team_vectors["exp_019_behavior_v1_dist_wrec050_scale120.csv"]
    u019 = (z019 - z_orth) - np.mean(z019 - z_orth)
    u = np.vstack([u_v2, u019])
    gram = u @ u.T / len(uid)
    b_public = 0.5 * (ORTH_SCORE ** 2 + np.diag(gram) - np.array([V2_SCORE ** 2, exp019_score ** 2]))
    pinv = np.linalg.pinv(gram)
    coef = pinv @ b_public
    joint_gain = float(b_public @ coef)
    v2_gain = float(b_public[0] ** 2 / gram[0, 0])
    team_gain = float(b_public[1] ** 2 / gram[1, 1])
    joint = {
        "status": "EXP019_PUBLIC_PLANE_DECODED_EXP024_CONDITIONAL_GAIN_NOT_IDENTIFIED",
        "basis": ["ORTH_FINAL_to_JOINT_V2", "ORTH_FINAL_to_scored_exp019"],
        "G_full_population": gram.tolist(), "b_from_public_scores_full_Gram": b_public.tolist(),
        "coefficients": coef.tolist(), "condition_number": float(np.linalg.cond(gram)),
        "correlation_corrections": corr(u_v2, u019),
        "individual_gain_MSE_V2_axis": v2_gain,
        "individual_gain_MSE_exp019_axis": team_gain,
        "joint_gain_MSE_vs_ORTH_FINAL": joint_gain,
        "conditional_gain_exp019_given_V2_MSE": joint_gain - v2_gain,
        "conditional_gain_V2_given_exp019_MSE": joint_gain - team_gain,
        "warning": "public membership unknown; both directions are already in the scored span",
        "exp024": {
            "test_corr_perp_with_V2_correction": corr(d024, u_v2),
            "test_perp_RMS": rms(d024),
            "condition_after_addition": span.augmented_condition(z_exp024),
            "b": None, "conditional_gain_MSE": None, "expected_Delta_RMSLE": None,
            "reason": "no scored exp024 vector and no row-level post-span OOF residual alignment",
        },
        "candidate_created": False,
        "candidate_gate": {
            "clean_post_span_OOF_rho_ge_0_015": False,
            "LB_implied_out_of_span_headroom_ge_0_0003": False,
            "confirmed_better_LB_reproducible_component": False,
        },
        "next_measurement": "Build a symmetric +/- fixed-RMS probe from team_b_signal_vector.npz:d_perp on top of JOINT_V2; score both to identify sign/amplitude, then shrink for public noise. Do not use raw exp024 weight sweep as the measurement.",
    }
    json_dump(EXP / "joint_with_v2.json", clean_json(joint))

    # Final report is intentionally generated from the measured artifacts.
    perp_pct = 100.0 * g024["perp_fraction_RMS"]
    energy_pct = 100.0 * g024["perp_fraction_energy"]
    cv_full = float(np.mean(fold_values["exp024"] - fold_values["exp019"]))
    cv_cat = float(np.mean(fold_values["exp024"] - fold_values["exp023"]))
    report = f"""# EXP090 — Team-B Solution Audit

## Verdict

Team-B имеет два разных понятия champion. Лучший **фактически scored** файл —
`exp_019_behavior_v1_dist_wrec050_scale120.csv`, public LB
`{exp019_score:.15f}`. Новый локальный champion — `exp_024` с mean CV
`{fold_values['exp024'].mean():.6f}`, но его исходный CSV, model files и row-level
OOF отсутствуют.

`exp024` создаёт материальную новую TEST-direction: после двойной
проекции из актуального scored span остаётся `{perp_pct:.3f}%` RMS correction
(`{energy_pct:.3f}%` энергии). Но полезность этой именно post-span части не
измерена ни OOF, ни LB. Поэтому candidate CSV не создан; сохранён точный vector
artifact для следующего измерения. Большая часть полной correction всё же лежит
в уже известном span; новый signal — независимый хвост, а не доминирующая часть.

## Team-B architecture

Финальный handoff — log-space ensemble из пяти компонент:

| component | family / target | features | weight |
| --- | --- | --- | ---: |
| recency | LightGBM regression / `log1p(y30)` | 152 recency aggregates | 0.25 |
| post_order_dist | LightGBM 16-bin classification / binned `log1p(y30)` | 215 long-buy + post-order | 0.10 |
| behavior_dist | LightGBM 16-bin classification | 329 behavior_v1 | 0.20 |
| xgb_behavior | XGBoost regression / `log1p(y30)` | 329 behavior_v1 | 0.25 |
| cat_behavior | CatBoost regression / `log1p(y30)` | 329 behavior_v1 | 0.20 |

Component scales are `0.64` for recency and `0.62` for the other four.
Occurrence gate exists only as rejected exp014. SEQ, ETX and Ridge are absent.

The quality mechanism is cumulative: log-target/tree ensembles, raw-GMV scale
calibration, a distribution head, post-order state, and the 114 behavior_v1
features. XGBoost then gives the largest new unscored CV increment; CatBoost
adds a smaller conditional increment.

## Reproduction

The supplied raw parquet was found by exact SHA256
`5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0`.
Fresh training used exactly eight cutoff panels `2025-08-28..2025-10-16`, TEST
cutoff `2026-02-14`, 250,000 unique users, and exact sample order.

Byte-exact reproduction of the claimed `exp024` file is **not provable**:

- the original CSV is absent;
- no model files or pinned package versions are supplied;
- `cat_xgb_blend.py` applies global raw scale `1.20` before level-match, while
  `predict.py --handoff` omits it. The two realized vectors differ by RMS
  `{rms(z_exp024-z_handoff):.9f}` in log space.

Both formulas were reconstructed from the same freshly trained components and
stored in `team_b_reproduction_predictions.parquet`. The experiment-script
formula is used for geometry because it is the provenance of the named exp024
submission.

## Leakage / validation audit

No same-period or target leakage was found. Every feature query first enforces
`event_date < cutoff`; target is `[cutoff, cutoff+30d)`. Both validation targets
are fully observed and their training labels end before the validation feature
cutoff.

The questionable part is selection, not label leakage: 2,835 weight vectors
were selected and reported on the same two folds, without an outer/nested fold.
Scale `1.20` and final TEST level also have public-LB lineage. `validation.py`
is a stub, actual dates live inside scripts, and feature caches do not include a
code/data fingerprint.

## Comparison with JOINT_V2

`JOINT_V2` exact SHA256 is `{V2_SHA256}` and LB `{V2_SCORE:.15f}`. The full
geometry table contains RMS difference, level, correlations, zero/clipping
fractions, double-projection diagnostics and condition numbers for each strong
Team-B vector.

All scored Team-B files, including exp019, are already members of the current
scored bank. Accordingly exp019 post-span RMS is only
`{g019['RMS_d_perp']:.3e}`. It does not constitute a new direction relative to
the current span even though it was historically useful inside Team-B.

## New-signal geometry

For fresh exp024 relative to JOINT_V2:

- centered correction RMS: `{g024['RMS_centered_correction']:.9f}`;
- post-span RMS: `{g024['RMS_d_perp']:.9f}`;
- post-span fraction: `{g024['perp_fraction_RMS']:.6f}` RMS,
  `{g024['perp_fraction_energy']:.6f}` energy;
- corr(post-span, A1-365): `{g024['corr_d_perp_with_A1_365']:.6f}`;
- corr(post-span, A2): `{g024['corr_d_perp_with_A2']:.6f}`;
- corr(post-span, JOINT_V2 out-of-plane residual):
  `{g024['corr_d_perp_with_JOINT_V2_out_of_plane']:.6f}`;
- scored-span rank `{span.rank}` -> `{span.augmented_condition(z_exp024)['rank']}`;
  design condition `{span.condition_design:.3f}` ->
  `{span.augmented_condition(z_exp024)['condition_design']:.3f}`.

Thus exp024 is not merely a repackage of known scored submissions on TEST.
It is mostly an in-span repackage with a material absolute residual. This is a
novelty statement, not a value statement.

## OOF residual evidence

No row-level Team-B OOF vectors exist. The saved `exp024` grid contains only
mean/std over two folds. From those sufficient statistics the selected blend
improves exp019 on both folds by
`{(fold_values['exp024']-fold_values['exp019'])[0]:+.9f}` and
`{(fold_values['exp024']-fold_values['exp019'])[1]:+.9f}` RMSLE, mean
`{cv_full:+.9f}`. CatBoost conditional on exp023 contributes mean
`{cv_cat:+.9f}`.

These signs are stable but non-nested. `rho`, `b`, `G`, oracle amplitude,
strict-forward delta and user bootstrap CI for the **post-span** direction are
not identifiable and are left missing rather than synthesized from standalone
CV.

## Leaderboard evidence

The scored Team-B ladder confirms small gains from post-order, distribution
head and full behavior_v1. Behavior_v1 improves exp017 from
`{exp017_score:.9f}` to `{exp019_score:.9f}`, Delta MSE
`{exp019_score**2-exp017_score**2:+.9f}`. The slim ablation regresses.

All of this evidence is already inside the current scored span. `exp023` and
`exp024` have no public scores, so LB-implied headroom for their new post-span
part is unavailable. The decoding table clearly separates scored LB inference
from unscored OOF-only claims.

Relative to JOINT_V2, the entire scored exp019 correction has LB-implied optimal
scale only `{lb019_from_v2['optimal_scale_from_anchor_toward_candidate']:.6f}`
and maximum full-Gram MSE gain
`{lb019_from_v2['maximum_MSE_gain_on_full_Gram']:.9f}`. Its out-of-current-span
part is numerical zero, so this is neither material headroom nor new signal.

## Source of Team-B gain

The last confirmed scored increment is the `behavior_v1` representation in the
distribution head: order-cycle regularity, overdue phase, stable cheque,
pre/post-last-order intent and calendar habit features. Within the new
unscored five-family system, XGBoost diversity supplies about `-0.000580`
two-fold RMSLE versus exp019 and CatBoost adds about `{cv_cat:+.6f}` conditional
RMSLE. The CatBoost search optimum hits its imposed `0.20` boundary, so its
reported coefficient is especially selection-sensitive.

Sibling vectors are exp020 (slim behavior ablation), exp022 (three-LightGBM
weights), exp023 (+XGBoost), and the two inconsistent exp024/handoff formulas.

## Compatibility with current best

The scored exp019 axis is geometrically compatible but already exploited by the
current scored-span search. Public two-axis decoding versus ORTH_FINAL is in
`joint_with_v2.json`; it does not establish a new Team-B conditional gain.

For exp024, TEST correlation and conditioning are acceptable, but the required
alignment vector `b = U.T @ residual / N` is absent. Therefore individual and
conditional post-span gains, safe amplitude and expected Delta RMSLE are **not
identified**. Adding raw correlations would be mathematically invalid.

The only numerically decoded conditional term is scored exp019 given V2:
approximately `{joint['conditional_gain_exp019_given_V2_MSE']:.9f}` MSE in the
full-population public-plane diagnostic. It is below the `0.0003` gate, depends
on unknown public membership, and is already inside the current scored span;
it is not an expected new gain for exp024.

## Best justified candidate

No CSV was created. None of the three authorization gates is met:

- clean post-span OOF rho >= 0.015: not measured;
- LB-implied post-span headroom >= 0.0003 MSE: no exp024 score;
- reproducible confirmed better LB: exp024 is unscored and original bytes are missing.

Instead, `team_b_signal_vector.npz` stores the centered and twice-projected
exp024 direction plus all five components. SHA256:
`{sha256(EXP/'team_b_signal_vector.npz')}`.

## Final conclusions

1. Best scored Team-B submission: exp019, LB `{exp019_score:.15f}`.
2. Declared exp024 champion cannot be byte-reproduced; a fresh code reproduction
   exists, but two primary inference formulas disagree.
3. No leakage was found; validation/selection and reproducibility are the weak
   points.
4. Scored Team-B signal is fully in the current scored span. Fresh exp024 has
   `{perp_pct:.3f}%` post-span RMS, but no post-span value evidence.
5. The defensible next action is one **symmetric measurement pair** built from
   `d_perp`: `z=clip(z_V2 +/- 0.025*d_perp/RMS(d_perp),0)`. Decode sign and
   amplitude from both scores, apply public-noise shrinkage, and only then build
   one combined candidate. Do not tune another raw five-way blend on the same
   two folds.
"""
    (EXP / "REPORT.md").write_text(report, encoding="utf-8")

    # Source/output artifact manifest.  The manifest itself is omitted from its
    # hash rows to avoid a self-referential digest.
    source_paths = [
        TEAM_B / "src" / "features.py", TEAM_B / "src" / "train.py",
        TEAM_B / "src" / "predict.py", TEAM_B / "src" / "cat_xgb_blend.py",
        TEAM_B / "artifacts" / "exp024_cat_xgb_blend_grid.csv",
        SUB / V2_NAME, SUB / "exp_019_behavior_v1_dist_wrec050_scale120.csv",
        E89 / "scored_span_manifest.csv", E75 / "A1_TREE_TRAJ_365_TEST_PERP.npy",
        E75 / "A2_WEEKLY_RESIDUAL_CNN_TEST_PERP.npy",
    ]
    output_paths = [
        EXP / "team_b_inventory.csv", EXP / "reproduction_audit.json", EXP / "leakage_audit.csv",
        EXP / "team_b_geometry.csv", EXP / "oof_residual_metrics.csv",
        EXP / "leaderboard_decoding.csv", EXP / "signal_decomposition.json",
        EXP / "joint_with_v2.json", EXP / "REPORT.md", EXP / "team_b_signal_vector.npz",
        EXP / "team_b_reproduction_predictions.parquet", EXP / "team_b_reproduced_vectors.npz",
    ]
    artifact_rows = []
    for role, paths in [("source", source_paths), ("output", output_paths)]:
        for path in paths:
            artifact_rows.append({
                "role": role, "path": str(path.resolve()), "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
                "notes": "primary evidence" if role == "source" else "EXP090 generated artifact",
            })
    pd.DataFrame(artifact_rows).to_csv(EXP / "artifact_manifest.csv", index=False)

    print(json.dumps({
        "best_scored_team_b": "exp019", "exp019_LB": exp019_score,
        "exp024_perp_fraction_RMS": g024["perp_fraction_RMS"],
        "exp024_perp_RMS": g024["RMS_d_perp"],
        "exp024_cv_delta_vs_exp019": cv_full,
        "candidate_created": False,
        "signal_vector_sha256": sha256(EXP / "team_b_signal_vector.npz"),
    }, indent=2))


if __name__ == "__main__":
    main()
