"""EXP-051: stable BG/NBD revalidation and BTYD05 production candidate.

Run from the repository root with one command::

    python src/btyd_exp051.py

The runner applies one optimizer policy to all four OOF folds and the
production cutoff, performs start-to-start predictive stability audits, and
creates the fixed 5% BTYD blend only after every gate passes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.btyd05_production as production
import src.btyd_day_bgnbd as legacy
from src.btyd_stable_fit import POLICY_ID, fit_bgnbd_stable
from src.config import ARTIFACTS, ROOT, SEED
from src.validation import calibrate


EXP_NUM = 51
EXP_ID = "BTYD-STABLE-PRODUCTION"
PREFIX = "BTYD_STABLE_EXP051"
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
PRED_VAR_GATE = 1e-8
PRED_MAX_ABS_GATE = 1e-3
BLEND_SCORE_SPAN_GATE = 1e-5


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(legacy.jsonable(value), ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def _configure_oof() -> None:
    legacy.EXP_NUM = EXP_NUM
    legacy.EXP_ID = EXP_ID
    legacy.PREFIX = PREFIX
    legacy.RUN_DIR = RUN_DIR
    legacy.RESULTS = RESULTS
    legacy.fit_bgnbd = fit_bgnbd_stable

    def config() -> dict[str, Any]:
        out = _ORIGINAL_CONFIG()
        out.update({
            "experiment_number": EXP_NUM, "experiment_id": EXP_ID, "prefix": PREFIX,
            "optimizer": "analytic Jacobian L-BFGS-B plus deterministic BFGS polish",
            "optimizer_policy_id": POLICY_ID,
            "optimizer_starts": legacy.OPT_STARTS,
            "seed": SEED, "seed_source": "src/config.py",
            "predictive_stability_gates": {
                "max_pairwise_var_z": PRED_VAR_GATE,
                "max_pairwise_abs_delta_z": PRED_MAX_ABS_GATE,
                "max_fixed_005_blend_score_span": BLEND_SCORE_SPAN_GATE,
            },
        })
        return out

    legacy.experiment_config = config


_ORIGINAL_CONFIG = legacy.experiment_config


def _z_from_params(x: np.ndarray, tx: np.ndarray, T: int, mu: np.ndarray,
                   sigma: np.ndarray, params: np.ndarray) -> np.ndarray:
    _, pmf, _ = legacy.bgnbd_count_distribution(x, tx, T, params)
    z = np.empty(len(x), dtype=np.float64)
    for begin in range(0, len(x), 30_000):
        end = min(begin + 30_000, len(x))
        moments = legacy.metric_sum_moments(mu[begin:end], sigma[begin:end])
        z[begin:end] = np.sum(pmf[begin:end] * moments, axis=1)
    return z


def _pairwise(values: list[np.ndarray]) -> dict[str, float]:
    variances: list[float] = []
    maxima: list[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            delta = values[i] - values[j]
            variances.append(float(np.var(delta)))
            maxima.append(float(np.max(np.abs(delta))))
    return {"max_pairwise_var_z": max(variances),
            "max_pairwise_abs_delta_z": max(maxima)}


def predictive_stability_oof() -> dict[str, Any]:
    raw = np.load(RUN_DIR / "oof_raw.npz", allow_pickle=False)
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    global_max_var = 0.0
    global_max_abs = 0.0
    max_score_span = 0.0
    for fold in legacy.FOLD_LABELS:
        fold_mask = np.asarray(raw["cutoff"], dtype="U10") == fold
        indices = np.flatnonzero(fold_mask)
        groups = np.asarray(raw["group"])[fold_mask]
        trajectories = [np.empty(len(indices), dtype=np.float64) for _ in legacy.OPT_STARTS]
        for donor_group in (0, 1):
            recipient_local = groups == (1 - donor_group)
            recipient_global = indices[recipient_local]
            fit_path = RUN_DIR / f"fit_{fold.replace('-', '')}_donor{donor_group}.json"
            fit = json.loads(fit_path.read_text(encoding="utf-8"))
            variants: list[np.ndarray] = []
            for start in fit["starts"]:
                params = np.asarray([start["parameters"][key]
                                     for key in ("r", "alpha", "a", "b")])
                variants.append(_z_from_params(
                    np.asarray(raw["x"])[recipient_global],
                    np.asarray(raw["t_x"])[recipient_global], int(fit["T"]),
                    np.asarray(raw["mu_u"])[recipient_global],
                    np.asarray(raw["sigma_population"])[recipient_global], params))
            pair = _pairwise(variants)
            global_max_var = max(global_max_var, pair["max_pairwise_var_z"])
            global_max_abs = max(global_max_abs, pair["max_pairwise_abs_delta_z"])
            for start_i, variant in enumerate(variants):
                trajectories[start_i][recipient_local] = variant
            rows.append({"fold": fold, "donor_group": donor_group, **pair,
                         "mean_nll_spread": fit["mean_nll_spread"],
                         "max_log_parameter_spread": fit["max_log_parameter_spread"],
                         "max_gradient_norm": fit["max_gradient_norm"]})
        y = np.asarray(raw["y"])[fold_mask]
        base = np.asarray(raw["z_strongest"])[fold_mask]
        btyd_scores = [calibrate(y, z)[1] for z in trajectories]
        blend_scores = [calibrate(y, 0.95 * base + 0.05 * z)[1] for z in trajectories]
        span = float(np.ptp(blend_scores))
        max_score_span = max(max_score_span, span)
        fold_rows.append({"fold": fold, "btyd_scores": btyd_scores,
                          "fixed_005_blend_scores": blend_scores,
                          "fixed_005_blend_score_span": span})
    passed = (global_max_var <= PRED_VAR_GATE
              and global_max_abs <= PRED_MAX_ABS_GATE
              and max_score_span <= BLEND_SCORE_SPAN_GATE)
    out = {
        "status": "PASS" if passed else "FAIL",
        "policy_id": POLICY_ID,
        "gates": {"max_pairwise_var_z": PRED_VAR_GATE,
                  "max_pairwise_abs_delta_z": PRED_MAX_ABS_GATE,
                  "max_fixed_005_blend_score_span": BLEND_SCORE_SPAN_GATE},
        "maxima": {"max_pairwise_var_z": global_max_var,
                   "max_pairwise_abs_delta_z": global_max_abs,
                   "max_fixed_005_blend_score_span": max_score_span},
        "fit_rows": rows, "fold_rows": fold_rows,
    }
    _write_json(RESULTS / "oof_predictive_stability.json", out)
    return out


def predictive_stability_test() -> dict[str, Any]:
    raw = np.load(RUN_DIR / "test_raw.npz", allow_pickle=False)
    fit_details = json.loads((RESULTS / "btyd_fit_details.json").read_text(encoding="utf-8"))
    groups = np.asarray(raw["group"])
    rows: list[dict[str, Any]] = []
    trajectories = [np.empty(len(groups), dtype=np.float64) for _ in legacy.OPT_STARTS]
    global_max_var = 0.0
    global_max_abs = 0.0
    for fit in fit_details:
        donor_group = int(fit["donor_group"])
        recipient = groups == (1 - donor_group)
        variants: list[np.ndarray] = []
        for start in fit["starts"]:
            params = np.asarray([start["parameters"][key]
                                 for key in ("r", "alpha", "a", "b")])
            variants.append(_z_from_params(
                np.asarray(raw["x"])[recipient], np.asarray(raw["t_x"])[recipient],
                int(fit["T"]), np.asarray(raw["mu_u"])[recipient],
                np.asarray(raw["sigma_population"])[recipient], params))
        pair = _pairwise(variants)
        global_max_var = max(global_max_var, pair["max_pairwise_var_z"])
        global_max_abs = max(global_max_abs, pair["max_pairwise_abs_delta_z"])
        for start_i, variant in enumerate(variants):
            trajectories[start_i][recipient] = variant
        rows.append({"donor_group": donor_group, **pair,
                     "mean_nll_spread": fit["mean_nll_spread"],
                     "max_log_parameter_spread": fit["max_log_parameter_spread"],
                     "max_gradient_norm": fit["max_gradient_norm"]})
    strongest = np.asarray(raw["z_strongest"])
    centered_blends = []
    for z in trajectories:
        correction = 0.05 * (z - strongest)
        centered_blends.append(correction - correction.mean())
    blend_pair = _pairwise(centered_blends)
    passed = (global_max_var <= PRED_VAR_GATE
              and global_max_abs <= PRED_MAX_ABS_GATE
              and blend_pair["max_pairwise_abs_delta_z"] <= 0.05 * PRED_MAX_ABS_GATE)
    out = {"status": "PASS" if passed else "FAIL", "policy_id": POLICY_ID,
           "rows": rows, "maxima": {"max_pairwise_var_z": global_max_var,
                                      "max_pairwise_abs_delta_z": global_max_abs},
           "centered_fixed_005_correction": blend_pair}
    _write_json(RESULTS / "test_predictive_stability.json", out)
    return out


def _configure_production() -> None:
    production.EXP_NUM = EXP_NUM
    production.EXP_ID = EXP_ID
    production.PREFIX = PREFIX
    production.RUN_DIR = RUN_DIR
    production.RESULTS = RESULTS
    production.OOF_PATH = RUN_DIR / "oof_raw.npz"
    production.fit_bgnbd = fit_bgnbd_stable


def fresh_retrain_audit() -> dict[str, Any]:
    """Record the exact work needed for an honest deterministic FRESH rebuild."""
    parity_path = RESULTS / "fresh_parity_audit.json"
    historical_path = (ROOT / "research" / "strategies" / "results"
                       / "BTYD05_PROD_EXP050" / "fresh_parity_audit.json")
    source = parity_path if parity_path.exists() else historical_path
    if not source.exists():
        raise FileNotFoundError("EXP-050 FRESH parity audit is required")
    parity = json.loads(source.read_text(encoding="utf-8"))
    fold_targets = [f"V{tag}" for tag in ("0904", "0918", "1002", "1016")]
    out = {
        "status": "NOT_RUN_AFTER_CASE_B_SUBMISSION",
        "reason": "one honest BTYD05 submission was obtained; task requires stopping BTYD/FRESH research",
        "model_family": "SEQ-D3A-BASE TCN; hidden64, 8 blocks, kernel3, pooled [last,mean,max]",
        "semantic_protocol": {
            "clean_extra": "same CLEAN/EXTRA protocol as EXP-040",
            "crossfit": "splitmix64(user_id)&1; two-sided donor A/B",
            "heads": "CLEAN/VOL/FRESH conditional heads",
            "correction": "FRESH-CLEAN; donor-safe 0.5/99.5 winsorization; GLOBAL; center; alpha=1",
            "depth_clip": 289,
        },
        "historical_recipe": parity.get("required_training_recipe"),
        "historical_fold_checkpoint_hashes": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in parity.get("fold_checkpoints", [])
        ],
        "historical_artifact_reuse": {
            "architecture_and_semantics": True,
            "model_trajectory_or_predictions": False,
            "reason": "historical encoders used workers=3 race policy and exact TEST checkpoint is absent",
        },
        "required_new_recipe": {
            "architecture": "fixed historical SEQ-D3A-BASE family; no hyperparameter search",
            "seed": SEED, "seed_source": "src/config.py",
            "workers": 1, "materialized_batch_plan": True,
            "deterministic_cuda": True, "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "required_encoder_runs": {
            "count": 5, "targets": fold_targets + ["TEST/full-history"],
            "explanation": "four fold-compatible encoders plus one same-recipe production encoder",
        },
        "required_conditional_head_fits": {
            "count": 30,
            "explanation": "(4 OOF folds + 1 production) x 2 donor sides x CLEAN/VOL/FRESH",
        },
        "new_plan_or_model_artifacts": [],
    }
    _write_json(RESULTS / "fresh_retrain_audit.json", out)
    return out


def _finalize_summary(oof_stability: dict[str, Any], test_stability: dict[str, Any]) -> None:
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    oof = json.loads((RESULTS / "summary_oof.json").read_text(encoding="utf-8"))
    summary.update({
        "experiment": EXP_ID, "experiment_number": EXP_NUM,
        "optimizer_policy_id": POLICY_ID,
        "validation_status": "NEW EXP-051 OOF revalidation; identical policy to production",
        "oof_nested_delta_wcv": oof["nested_delta_wcv"],
        "oof_nested_fold_deltas": oof["nested_fold_deltas"],
        "oof_positive_residual_alignment_folds": oof["positive_residual_alignment_folds"],
        "oof_predictive_stability": oof_stability["status"],
        "test_predictive_stability": test_stability["status"],
        "fresh_retrain_status": "NOT_RUN_AFTER_CASE_B_SUBMISSION",
    })
    _write_json(summary_path, summary)


def persist_source_hashes() -> dict[str, str]:
    paths = [Path(__file__).resolve(), (ROOT / "src" / "btyd_stable_fit.py").resolve(),
             (ROOT / "src" / "test_btyd_stable_fit.py").resolve()]
    hashes = {str(path): legacy.sha256_file(path) for path in paths}
    _write_json(RESULTS / "source_hashes.json", hashes)
    return hashes


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _configure_oof()
    fresh_retrain_audit()
    oof_summary = legacy.run_experiment()
    # Preserve the OOF decision record before production replaces summary.json.
    _write_json(RESULTS / "summary_oof.json", oof_summary)
    oof_stability = predictive_stability_oof()
    required = [
        oof_stability["status"] == "PASS",
        oof_summary["nested_better_folds"] >= 3,
        oof_summary["nested_delta_wcv"] < 0,
        oof_summary["positive_residual_alignment_folds"] >= 3,
    ]
    if not all(required):
        _write_json(RESULTS / "summary.json", {
            "experiment": EXP_ID, "experiment_number": EXP_NUM,
            "btyd_fit_status": "PASS", "btyd_oof_revalidation": "FAIL",
            "fresh_retrain_status": "NOT_RUN", "production_support": "NOT_REACHED",
            "submission_status": "NOT_CREATED", "oof": oof_summary,
            "predictive_stability": oof_stability,
        })
        return
    print("BTYD OOF revalidation complete", flush=True)
    _configure_production()
    original_build_submission = production.build_submission
    deferred: dict[str, np.ndarray] = {}

    def defer_submission(uid: np.ndarray, z_raw: np.ndarray) -> dict[str, Any]:
        deferred["uid"] = np.asarray(uid).copy()
        deferred["z_raw"] = np.asarray(z_raw).copy()
        return {"status": "DEFERRED_UNTIL_START_PREDICTION_STABILITY_PASS"}

    production.build_submission = defer_submission
    production.main()
    if not (RUN_DIR / "test_raw.npz").exists():
        raise RuntimeError("production BTYD did not reach test scoring")
    test_stability = predictive_stability_test()
    if test_stability["status"] != "PASS":
        raise RuntimeError("test start-stability audit failed; submission remains uncreated")
    if set(deferred) != {"uid", "z_raw"}:
        raise RuntimeError("production support did not reach deferred submission stage")
    production.build_submission = original_build_submission
    submission = original_build_submission(deferred["uid"], deferred["z_raw"])
    submission["raw_candidate_sha256"] = legacy.sha256_array(deferred["z_raw"])
    support_path = RESULTS / "production_support.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    support["optimizer_policy_id"] = POLICY_ID
    support["start_prediction_stability"] = test_stability
    _write_json(support_path, support)
    _finalize_summary(oof_stability, test_stability)
    final_summary_path = RESULTS / "summary.json"
    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    final_summary["submission"] = submission
    final_summary["submission_status"] = "CREATED_BTYD05_ONLY"
    _write_json(final_summary_path, final_summary)
    _write_json(RESULTS / "submission_verification.json", submission)
    persist_source_hashes()
    print("submission created", flush=True)


if __name__ == "__main__":
    main()
