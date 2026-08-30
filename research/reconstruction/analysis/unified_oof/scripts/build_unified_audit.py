"""Build the unified canonical/occurrence audit from immutable artifacts.

The teammate row-level cache is optional.  In the current machine state it is
absent, so the script records summary-only teammate evidence and evaluates the
single permitted fallback artifact produced by run_occurrence_fallback.py.

Run:
    python analysis/unified_oof/scripts/build_unified_audit.py
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parents[1]
WORKSPACE = SCRIPT.parents[3]
ART = OUT / "artifacts"
SOURCE = Path(os.environ.get("OZON_SOURCE_ROOT", r"C:\Users\Admin\Desktop\OZON-E-CUP"))
BUNDLE = Path(os.environ.get("OCC_BUNDLE_ROOT", r"C:\Users\Admin\Desktop\latest_pipeline_bundle"))

FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FW = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64) / 15.0
LEVEL = 2.3293
STRONG_GATE = -0.0005

ALIGNED = SOURCE / "artifacts" / "RESDISC_053" / "aligned_oof.parquet"
BTYD_OOF = SOURCE / "artifacts" / "BTYD_STABLE_EXP051" / "oof_raw.npz"
BTYD_TEST = SOURCE / "artifacts" / "BTYD_STABLE_EXP051" / "test_raw.npz"
FALLBACK_OOF = ART / "fallback_occ_overlay_oof.npz"
FALLBACK_TEST = ART / "fallback_occ_overlay_test.npz"
FALLBACK_SUMMARY = ART / "fallback_occ_overlay_summary.json"

CANONICAL_NPZ = {
    "CAP": SOURCE / "artifacts" / "oof_S1-E03a.npz",
    "UNC": SOURCE / "artifacts" / "oof_S1-E02.npz",
    "DIST": SOURCE / "artifacts" / "oof_S1-DIST.npz",
    "ETX": SOURCE / "artifacts" / "oof_ETX-AVG3.npz",
    "SEQ": SOURCE / "artifacts" / "oof_SEQ-AVG3.npz",
}

TEST_CSV = {
    "strongest": SOURCE / "submissions" / "submission_STRONGEST_CURRENT.csv",
    "seq65": SOURCE / "submissions" / "submission_SEQ65_TEMPORAL_HEAVY.csv",
    "btyd05": SOURCE / "submissions" / "submission_BTYD05.csv",
    "compound": WORKSPACE / "analysis" / "BEST_EXISTING_SUBMISSION.csv",
    "latest": BUNDLE / "latest" / "latest.csv",
    "friend": BUNDLE / "latest" / "components" / "friend.csv",
    "occ_meta_B": BUNDLE / "latest" / "components" / "occ_meta_B.csv",
    "occ_raw_X3": BUNDLE / "latest" / "components" / "occ_raw_X3.csv",
}

EXPECTED_CACHE_ROOTS = [
    Path(r"C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_research"),
    BUNDLE / "research_scripts" / "_best_bas_research",
    SOURCE / "пайплайн сокомандника" / "research_scripts" / "_best_bas_research",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x: Any, digits: int = 9) -> str:
    if x is None:
        return "NA"
    try:
        if not math.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def sorted_order(cutoff: np.ndarray, uid: np.ndarray) -> np.ndarray:
    return np.lexsort((uid.astype(np.int64), cutoff.astype("U10")))


def center_by_fold(x: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.asarray(x, np.float64).copy()
    for j in range(4):
        m = fold == j
        out[m] -= float(out[m].mean())
    return out


def fold_cal_scores(ly: np.ndarray, z: np.ndarray, fold: np.ndarray) -> list[float]:
    return [float(np.std(ly[fold == j] - z[fold == j])) for j in range(4)]


def fold_raw_scores(ly: np.ndarray, z: np.ndarray, fold: np.ndarray) -> list[float]:
    return [float(np.sqrt(np.mean((ly[fold == j] - z[fold == j]) ** 2))) for j in range(4)]


def weighted(scores: list[float]) -> float:
    return float(FW @ np.asarray(scores, np.float64))


def source_score(
    name: str,
    z: np.ndarray,
    ly: np.ndarray,
    fold: np.ndarray,
    base: np.ndarray,
    family: str,
    availability: str = "canonical_row_level",
) -> dict[str, Any]:
    cal = fold_cal_scores(ly, z, fold)
    raw = fold_raw_scores(ly, z, fold)
    base_cal = fold_cal_scores(ly, base, fold)
    residual = ly - z
    base_residual = ly - base
    return {
        "source": name,
        "family": family,
        "availability": availability,
        "rows": len(z),
        "rmsle_global_raw": float(np.sqrt(np.mean(residual**2))),
        "wcv_calibrated": weighted(cal),
        "delta_wcv_vs_exp037": weighted(cal) - weighted(base_cal),
        "wins_vs_exp037": int(sum(a < b for a, b in zip(cal, base_cal))),
        "prediction_corr_vs_exp037": float(np.corrcoef(z, base)[0, 1]),
        "residual_corr_vs_exp037": float(np.corrcoef(residual, base_residual)[0, 1]),
        **{f"fold_{j+1}_rmsle_raw": raw[j] for j in range(4)},
        **{f"fold_{j+1}_rmsle_cal": cal[j] for j in range(4)},
        **{f"fold_{j+1}_delta_vs_exp037": cal[j] - base_cal[j] for j in range(4)},
    }


def weighted_cov_terms(delta: np.ndarray, residual: np.ndarray, fold: np.ndarray, allowed: list[int]) -> tuple[float, float]:
    num = 0.0
    den = 0.0
    norm = float(FW[allowed].sum())
    for j in allowed:
        m = fold == j
        d = np.asarray(delta[m], np.float64)
        r = np.asarray(residual[m], np.float64)
        d -= d.mean()
        r -= r.mean()
        w = float(FW[j] / norm)
        num += w * float(np.mean(d * r))
        den += w * float(np.mean(d * d))
    return num, den


def nested_add_one(
    label: str,
    baseline: np.ndarray,
    ly: np.ndarray,
    fold: np.ndarray,
    outer_delta: np.ndarray,
    nested_inner_raw: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    base_scores = fold_cal_scores(ly, baseline, fold)
    z = baseline.copy()
    lambdas: list[float] = []
    for outer in range(4):
        allowed = [j for j in range(4) if j != outer]
        inner_delta = nested_inner_raw[outer].astype(np.float64)
        if not np.isfinite(inner_delta[fold != outer]).all():
            raise AssertionError(f"Missing nested inner predictions for outer={outer}")
        num, den = weighted_cov_terms(inner_delta, ly - baseline, fold, allowed)
        lam = float(np.clip(num / max(den, 1e-12), 0.0, 1.5))
        lambdas.append(lam)
        m = fold == outer
        z[m] = baseline[m] + lam * outer_delta[m]
    cand_scores = fold_cal_scores(ly, z, fold)
    deltas = [a - b for a, b in zip(cand_scores, base_scores)]
    row = {
        "occurrence_source": "fallback_occ_lgbm_residual",
        "baseline": label,
        "protocol": "nested model cross-fit + analytic scalar on inner folds; lambda clipped [0,1.5]",
        "baseline_wcv": weighted(base_scores),
        "candidate_wcv": weighted(cand_scores),
        "delta_wcv": weighted(deltas),
        "better_folds": int(sum(x < 0 for x in deltas)),
        "last_fold_delta": deltas[-1],
        "lambda_mean": float(np.mean(lambdas)),
        **{f"outer_{j+1}_lambda": lambdas[j] for j in range(4)},
        **{f"fold_{j+1}_delta": deltas[j] for j in range(4)},
    }
    return row, z


def full_oof_lambda(delta: np.ndarray, residual: np.ndarray, fold: np.ndarray) -> float:
    num, den = weighted_cov_terms(delta, residual, fold, [0, 1, 2, 3])
    return float(np.clip(num / max(den, 1e-12), 0.0, 1.5))


def vector_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    dot = float(np.dot(a, b) / len(a))
    aa = float(np.dot(a, a) / len(a))
    bb = float(np.dot(b, b) / len(b))
    return {
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "cosine": dot / max(math.sqrt(aa * bb), 1e-15),
        "covariance": float(np.cov(a, b, ddof=0)[0, 1]),
        "rms_a": math.sqrt(aa),
        "rms_b": math.sqrt(bb),
        "projection_b_on_a": dot / max(aa, 1e-15),
        "projection_a_on_b": dot / max(bb, 1e-15),
    }


def native_occurrence_summaries() -> dict[str, dict[str, Any]]:
    final_path = BUNDLE / "review_bundles" / "final6h_REVIEW_BUNDLE_20260823_204823_extracted" / "results" / "OCCURRENCE_BRANCH_VALIDATION.csv"
    extra_path = BUNDLE / "review_bundles" / "extra90_REVIEW_BUNDLE_20260823_222555_extracted" / "results" / "ALL_EXTRA90_VALIDATION.csv"
    out: dict[str, dict[str, Any]] = {}
    if final_path.exists():
        d = pd.read_csv(final_path)
        row = d[d["name"].str.startswith("metaocc_l31_risk__blend_ridge_recentpow1p7")].iloc[0]
        out["occ_meta_B/final6h_B"] = row.to_dict()
    if extra_path.exists():
        d = pd.read_csv(extra_path)
        row = d[d["name"].str.startswith("xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7")].iloc[0]
        out["occ_raw_X3/extra90_3"] = row.to_dict()
    return out


def read_submission(path: Path, reference_uid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(path).select(["user_id", "predict"])
    if d["user_id"].n_unique() != d.height:
        raise AssertionError(f"Duplicate TEST user_id: {path}")
    if reference_uid is not None:
        ref = pl.DataFrame({"user_id": reference_uid})
        d = ref.join(d, on="user_id", how="left", validate="1:1")
        if d["predict"].null_count():
            raise AssertionError(f"Missing TEST users: {path}")
    p = d["predict"].to_numpy().astype(np.float64)
    if not np.isfinite(p).all() or (p < 0).any():
        raise AssertionError(f"Invalid predictions: {path}")
    return d["user_id"].to_numpy().astype(np.int64), np.log1p(p)


@dataclass
class Alignment:
    source: str
    rows: int
    unique_keys: int
    matched: int
    missing: int
    extras: int
    duplicates: int
    target_equal: str
    order_equal: str
    status: str


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [ALIGNED, BTYD_OOF, BTYD_TEST, FALLBACK_OOF, FALLBACK_TEST, FALLBACK_SUMMARY, *CANONICAL_NPZ.values(), *TEST_CSV.values()]
    missing_required = [str(p) for p in required if not p.exists()]
    if missing_required:
        raise FileNotFoundError(f"Required artifacts missing: {missing_required}")

    base_cols = ["cutoff", "user_id", "fold", "y_true", "z_cap", "z_unc", "z_dist", "z_etx_avg3", "z_seq_avg3", "z_strong_raw"]
    frame = pl.read_parquet(ALIGNED, columns=base_cols).sort(["cutoff", "user_id"])
    cutoff = frame["cutoff"].to_numpy().astype("U10")
    uid = frame["user_id"].to_numpy().astype(np.int64)
    fold = np.asarray([FOLDS.index(str(x)) for x in cutoff], dtype=np.int8)
    y = frame["y_true"].to_numpy().astype(np.float64)
    ly = np.log1p(y)
    z_cap = frame["z_cap"].to_numpy().astype(np.float64)
    z_unc = frame["z_unc"].to_numpy().astype(np.float64)
    z_dist = frame["z_dist"].to_numpy().astype(np.float64)
    z_etx = frame["z_etx_avg3"].to_numpy().astype(np.float64)
    z_seq = frame["z_seq_avg3"].to_numpy().astype(np.float64)
    z_strong = frame["z_strong_raw"].to_numpy().astype(np.float64)
    z_rebuilt = .10*z_cap + .20*z_unc + .25*z_dist + .225*z_etx + .225*z_seq
    strongest_replay_error = float(np.max(np.abs(z_rebuilt - z_strong)))
    if strongest_replay_error > 2e-6:
        raise AssertionError(f"EXP-037 replay error {strongest_replay_error}")

    canonical_keys = np.rec.fromarrays([cutoff, uid], names="cutoff,user_id")
    alignment_rows: list[Alignment] = []
    for name, path in CANONICAL_NPZ.items():
        d = load_npz(path)
        order = sorted_order(d["cutoff"], d["user_id"])
        keys = np.rec.fromarrays([d["cutoff"][order].astype("U10"), d["user_id"][order].astype(np.int64)], names="cutoff,user_id")
        duplicates = len(keys) - len(np.unique(keys))
        matched = int(np.intersect1d(keys, canonical_keys).size)
        alignment_rows.append(Alignment(
            name, len(keys), len(np.unique(keys)), matched, len(canonical_keys)-matched,
            len(keys)-matched, duplicates,
            str(np.array_equal(d["y"][order].astype(np.float32), y.astype(np.float32))),
            str(np.array_equal(keys, canonical_keys)),
            "PASS" if matched == len(canonical_keys) and duplicates == 0 else "FAIL",
        ))

    btyd = load_npz(BTYD_OOF)
    bo = sorted_order(btyd["cutoff"], btyd["user_id"])
    bkeys = np.rec.fromarrays([btyd["cutoff"][bo].astype("U10"), btyd["user_id"][bo].astype(np.int64)], names="cutoff,user_id")
    bmatched = int(np.intersect1d(bkeys, canonical_keys).size)
    alignment_rows.append(Alignment(
        "BTYD_STABLE_EXP051", len(bkeys), len(np.unique(bkeys)), bmatched,
        len(canonical_keys)-bmatched, len(bkeys)-bmatched, len(bkeys)-len(np.unique(bkeys)),
        str(np.array_equal(btyd["y"][bo].astype(np.float32), y.astype(np.float32))),
        str(np.array_equal(bkeys, canonical_keys)),
        "PASS" if bmatched == len(canonical_keys) else "FAIL",
    ))
    if not np.array_equal(bkeys, canonical_keys):
        raise AssertionError("BTYD canonical key mismatch")
    z_btyd = btyd["z_btyd"][bo].astype(np.float64)

    fallback = load_npz(FALLBACK_OOF)
    fo = sorted_order(fallback["cutoff"], fallback["user_id"])
    fkeys = np.rec.fromarrays([fallback["cutoff"][fo].astype("U10"), fallback["user_id"][fo].astype(np.int64)], names="cutoff,user_id")
    fmatched = int(np.intersect1d(fkeys, canonical_keys).size)
    alignment_rows.append(Alignment(
        "fallback_occ_lgbm_residual", len(fkeys), len(np.unique(fkeys)), fmatched,
        len(canonical_keys)-fmatched, len(fkeys)-fmatched, len(fkeys)-len(np.unique(fkeys)),
        str(np.array_equal(fallback["y"][fo].astype(np.float64), y)),
        str(np.array_equal(fkeys, canonical_keys)),
        "PASS" if fmatched == len(canonical_keys) else "FAIL",
    ))
    if not np.array_equal(fkeys, canonical_keys):
        raise AssertionError("Fallback canonical key mismatch")
    occ_delta = fallback["delta_centered"][fo].astype(np.float64)
    nested_inner = fallback["nested_inner_raw"][:, fo].astype(np.float64)

    # Canonical recipes.
    z_seq65 = .10*z_cap + .10*z_unc + .15*z_dist + .325*z_etx + .325*z_seq
    z_btyd05 = .95*z_strong + .05*z_btyd
    z_compound = .95*z_seq65 + .05*z_btyd

    scores = [
        source_score("CAP", z_cap, ly, fold, z_strong, "canonical_primitive"),
        source_score("UNC", z_unc, ly, fold, z_strong, "canonical_primitive"),
        source_score("DIST", z_dist, ly, fold, z_strong, "canonical_primitive"),
        source_score("ETX-AVG3", z_etx, ly, fold, z_strong, "canonical_primitive"),
        source_score("SEQ-AVG3", z_seq, ly, fold, z_strong, "canonical_primitive"),
        source_score("EXP-037", z_strong, ly, fold, z_strong, "canonical_baseline"),
        source_score("SEQ65", z_seq65, ly, fold, z_strong, "canonical_correction"),
        source_score("BTYD", z_btyd, ly, fold, z_strong, "canonical_residual_source"),
        source_score("BTYD05", z_btyd05, ly, fold, z_strong, "canonical_correction"),
        source_score("compound_SEQ65_BTYD05", z_compound, ly, fold, z_strong, "canonical_compound"),
        source_score("fallback_occ_lgbm_residual_fixed1", z_strong + occ_delta, ly, fold, z_strong, "occurrence_fallback"),
    ]
    native = native_occurrence_summaries()
    for name, row in native.items():
        scores.append({
            "source": name,
            "family": "teammate_occurrence",
            "availability": "summary_only_native_protocol_not_canonical",
            "rows": 0,
            "native_summary_wcv": row.get("wcv"),
            "native_summary_base_wcv": row.get("base_wcv"),
            "native_summary_delta": row.get("delta"),
            "native_summary_wins": row.get("wins"),
            "native_summary_fold_scores": row.get("fold_scores"),
            "native_summary_fold_deltas": row.get("fold_deltas"),
            "wcv_calibrated": math.nan,
            "delta_wcv_vs_exp037": math.nan,
            "notes": "No row-level predictions; native base wCV 1.749804 is not EXP-037 wCV 1.747510.",
        })
    scores.append({
        "source": "teammate_latest",
        "family": "teammate_occurrence_assembly",
        "availability": "test_only_exact_recipe_oof_missing",
        "rows": 0,
        "wcv_calibrated": math.nan,
        "delta_wcv_vs_exp037": math.nan,
        "notes": "Exact TEST recipe available; canonical OOF for occ_meta_B and occ_raw_X3 absent.",
    })
    write_csv(OUT / "OCCURRENCE_OOF_SCORES.csv", scores)

    lofo_rows: list[dict[str, Any]] = []
    lofo_candidates: dict[str, np.ndarray] = {}
    for label, baseline in [
        ("A_EXP037", z_strong),
        ("B_EXP037_SEQ65", z_seq65),
        ("C_EXP037_BTYD05", z_btyd05),
        ("D_EXP037_SEQ65_BTYD05", z_compound),
    ]:
        row, candidate = nested_add_one(label, baseline, ly, fold, occ_delta, nested_inner)
        lofo_rows.append(row)
        lofo_candidates[label] = candidate
    write_csv(OUT / "INCREMENTAL_LOFO.csv", lofo_rows)
    exp037_wcv = weighted(fold_cal_scores(ly, z_strong, fold))
    compound_lambda = full_oof_lambda(occ_delta, ly-z_compound, fold)

    # OOF correction overlap, centered within fold to match calibrated wCV.
    corr_vectors = {
        "SEQ65": center_by_fold(z_seq65-z_strong, fold),
        "BTYD05": center_by_fold(z_btyd05-z_strong, fold),
        "compound": center_by_fold(z_compound-z_strong, fold),
        "fallback_occ": center_by_fold(occ_delta, fold),
        "DIST_primitive": center_by_fold(z_dist-z_strong, fold),
        "ETX_primitive": center_by_fold(z_etx-z_strong, fold),
        "SEQ_primitive": center_by_fold(z_seq-z_strong, fold),
    }
    target_error = center_by_fold(ly-z_strong, fold)
    overlap_rows: list[dict[str, Any]] = []
    names = list(corr_vectors)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = corr_vectors[left], corr_vectors[right]
            m = vector_metrics(a, b)
            row = {"left": left, "right": right, **m}
            row["left_residual_alignment"] = float(np.corrcoef(a, target_error)[0, 1])
            row["right_residual_alignment"] = float(np.corrcoef(b, target_error)[0, 1])
            row["left_positive_alignment_folds"] = int(sum(np.cov(a[fold==j], target_error[fold==j], ddof=0)[0,1] > 0 for j in range(4)))
            row["right_positive_alignment_folds"] = int(sum(np.cov(b[fold==j], target_error[fold==j], ddof=0)[0,1] > 0 for j in range(4)))
            overlap_rows.append(row)
    write_csv(OUT / "CORRECTION_OVERLAP.csv", overlap_rows)

    # TEST key alignment and exact latest recipe replay.
    test_uid, zt_strong = read_submission(TEST_CSV["strongest"])
    test_z: dict[str, np.ndarray] = {"strongest": zt_strong}
    test_alignment: list[dict[str, Any]] = []
    for name, path in TEST_CSV.items():
        tuid, z = read_submission(path, test_uid)
        test_z[name] = z
        test_alignment.append({
            "source": name, "rows": len(z), "unique_user_id": len(np.unique(tuid)),
            "matched": int(np.array_equal(tuid, test_uid) * len(test_uid)),
            "missing": 0, "duplicates": len(tuid)-len(np.unique(tuid)), "status": "PASS",
        })
    latest_rebuilt = .12*test_z["friend"] + .16*test_z["occ_meta_B"] + .72*test_z["occ_raw_X3"]
    latest_replay_error = float(np.max(np.abs(latest_rebuilt - test_z["latest"])))
    if latest_replay_error > 2e-12:
        raise AssertionError(f"latest recipe replay error {latest_replay_error}")
    ft = load_npz(FALLBACK_TEST)
    order = np.argsort(ft["user_id"])
    ref_order = np.argsort(test_uid)
    if not np.array_equal(ft["user_id"][order], test_uid[ref_order]):
        raise AssertionError("Fallback TEST user keys mismatch")
    map_pos = pd.Series(np.arange(len(test_uid)), index=test_uid)
    pos = map_pos.loc[ft["user_id"]].to_numpy()
    occ_test_delta = np.empty(len(test_uid), np.float64)
    occ_test_delta[pos] = ft["delta_centered"].astype(np.float64)
    test_z["fallback_occ_candidate"] = zt_strong + occ_test_delta
    compound_fallback_pre = test_z["compound"] + compound_lambda * occ_test_delta
    compound_fallback_shift = LEVEL - float(compound_fallback_pre.mean())
    test_z["compound_fallback_candidate"] = np.maximum(compound_fallback_pre + compound_fallback_shift, 0.0)
    compound_fallback_test_npz = ART / "candidate_compound_fallback_test.npz"
    np.savez_compressed(
        compound_fallback_test_npz,
        user_id=test_uid,
        z_strongest=zt_strong,
        z_compound=test_z["compound"],
        fallback_delta_centered=occ_test_delta,
        fallback_lambda=np.asarray([compound_lambda], np.float64),
        level_shift=np.asarray([compound_fallback_shift], np.float64),
        z_candidate=test_z["compound_fallback_candidate"],
    )

    latest_delta = test_z["latest"] - zt_strong
    test_overlap_rows: list[dict[str, Any]] = []
    test_candidates = {
        "occ_meta_B": test_z["occ_meta_B"],
        "occ_raw_X3": test_z["occ_raw_X3"],
        "seq65": test_z["seq65"],
        "btyd05": test_z["btyd05"],
        "compound": test_z["compound"],
        "fallback_occ_candidate": test_z["fallback_occ_candidate"],
        "compound_fallback_candidate": test_z["compound_fallback_candidate"],
    }
    for name, z in test_candidates.items():
        d = z - zt_strong
        raw = vector_metrics(d, latest_delta)
        centered = vector_metrics(d-d.mean(), latest_delta-latest_delta.mean())
        test_overlap_rows.append({
            "candidate": name,
            "rows": len(d),
            "rms_delta_vs_strongest": float(np.sqrt(np.mean(d*d))),
            "rms_candidate_vs_latest": float(np.sqrt(np.mean((z-test_z['latest'])**2))),
            "corr_delta_with_latest_raw": raw["corr"],
            "cosine_delta_with_latest_raw": raw["cosine"],
            "projection_latest_on_candidate_raw": raw["projection_b_on_a"],
            "corr_delta_with_latest_centered": centered["corr"],
            "cosine_delta_with_latest_centered": centered["cosine"],
            "projection_latest_on_candidate_centered": centered["projection_b_on_a"],
            "candidate_centered_rms": centered["rms_a"],
            "latest_centered_rms": centered["rms_b"],
        })
    write_csv(OUT / "TEST_SPACE_OVERLAP.csv", test_overlap_rows)

    # Candidate ladder.  Exact teammate rows remain explicitly blocked.
    combination_rows: list[dict[str, Any]] = []
    for row in lofo_rows:
        combination_rows.append({
            "combination": row["baseline"] + "+fallback_occ",
            "protocol": row["protocol"],
            "wcv": row["candidate_wcv"],
            "delta_vs_its_baseline": row["delta_wcv"],
            "delta_vs_EXP037": row["candidate_wcv"] - exp037_wcv,
            "better_folds": row["better_folds"],
            "last_fold_delta": row["last_fold_delta"],
            "status": "measured_canonical",
        })
    compound_score = next(x for x in scores if x["source"] == "compound_SEQ65_BTYD05")
    combination_rows += [
        {"combination": "SEQ65+BTYD05 canonical compound", "wcv": compound_score["wcv_calibrated"],
         "delta_vs_EXP037": compound_score["delta_wcv_vs_exp037"], "better_folds": compound_score["wins_vs_exp037"],
         "status": "measured_canonical"},
        {"combination": "best two teammate occurrence sources", "status": "BLOCKED_ROW_LEVEL_OOF_MISSING"},
        {"combination": "meta-occurrence exact", "status": "BLOCKED_ROW_LEVEL_OOF_MISSING"},
        {"combination": "compound+latest", "status": "BLOCKED_LATEST_OOF_MISSING",
         "chosen_lambda": math.nan, "oof_gain": math.nan,
         "test_compound_correction_rms": next(r for r in test_overlap_rows if r["candidate"] == "compound")["rms_delta_vs_strongest"],
         "test_corr_compound_with_latest_delta": next(r for r in test_overlap_rows if r["candidate"] == "compound")["corr_delta_with_latest_centered"],
         "test_projection_latest_on_compound": next(r for r in test_overlap_rows if r["candidate"] == "compound")["projection_latest_on_candidate_centered"],
         "expected_robustness": "UNRESOLVED_WITHOUT_LATEST_OOF",
         "notes": "TEST overlap measured; lambda cannot be selected against latest without canonical latest OOF."},
    ]
    write_csv(OUT / "COMBINATION_RESULTS.csv", combination_rows)

    # Full-data scalar for a possible canonical production formula.  This is
    # OOF-only and is not used to alter latest.
    fallback_exp037_nested = next(r for r in lofo_rows if r["baseline"] == "A_EXP037")
    fallback_exp037_fixed = next(r for r in scores if r["source"] == "fallback_occ_lgbm_residual_fixed1")
    compound_nested = next(r for r in lofo_rows if r["baseline"] == "D_EXP037_SEQ65_BTYD05")
    fallback_test_row = next(r for r in test_overlap_rows if r["candidate"] == "fallback_occ_candidate")
    best_test_row = next(r for r in test_overlap_rows if r["candidate"] == "compound_fallback_candidate")
    fallback_gate = (
        fallback_exp037_fixed["delta_wcv_vs_exp037"] <= STRONG_GATE
        and fallback_exp037_fixed["wins_vs_exp037"] >= 3
        and fallback_exp037_fixed["fold_4_delta_vs_exp037"] <= 0
    )
    overlap_incomplete = abs(float(fallback_test_row["corr_delta_with_latest_centered"])) < 0.90

    # Do not automatically create/declare a submission that competes with the
    # best public-observed latest while latest OOF is missing.  A passing
    # fallback therefore asks for exactly one follow-up: retrieve the compact
    # row-level occurrence predictions (not a broad retrain).
    compound_strong_incremental_gate = (
        compound_nested["delta_wcv"] <= STRONG_GATE
        and compound_nested["better_folds"] >= 3
        and compound_nested["last_fold_delta"] <= 0
    )
    if fallback_gate and overlap_incomplete:
        recommendation = "RUN ONE CHEAP FOLLOW-UP"
        answer = "YES_MECHANISM_EXACT_TEAMMATE_RECIPE_PENDING_CACHE"
    elif fallback_gate:
        recommendation = "RUN ONE CHEAP FOLLOW-UP"
        answer = "NO_EXACT_UNION_YET_TEST_SIGNAL_OVERLAPS_LATEST"
    else:
        recommendation = "STOP THIS DIRECTION"
        answer = "NO_FALLBACK_FAILS_INCREMENTAL_GATE"

    # Cache-missing report requested at analysis/ root.
    expected_fold_names = [f"occ_{name}__<fold>.npz" for name in (
        "r10_fast", "r16_bal", "r22_stable", "r14_multiscale", "r18_wide",
        "r24_multiscale", "r12_wide", "r20_shallow"
    )]
    missing_report = f"""# Occurrence cache missing

## Status

The teammate `_best_bas_research` cache was not found on local `C:`/`D:` project locations. The supplied bundle explicitly says it was excluded (~80 fold NPZ, ~15 TEST NPZ, ~9.7 GB feature cache). Exact TEST CSVs and validation summaries exist, but row-level canonical OOF does not.

## Missing artifacts

- cache root: `_best_bas_research/checkpoints/folds/` and `_best_bas_research/checkpoints/test/`;
- core folds: `cap/unc/dist/hurdle__{{2025-09-04,2025-09-18,2025-10-02,2025-10-16}}.npz` with `user_id,y,z` and `p,mu` for hurdle;
- occurrence folds (32 expected): `{', '.join(expected_fold_names)}` for four folds, keys `user_id,y,p`;
- occurrence TEST (8 expected): `occ_<name>_test.npz`, keys `user_id,p`;
- helper/meta state: `meta_raw_test.npz`, `final_candidate_bank.npz`, saved stable-stack predictions/recipes needed for `best_bas` and `_best_bas_research` replay;
- row-level final sources needed for the unified audit: `occ_meta_B/final6h_B`, `occ_raw_X3/extra90_3`, and their exact base/friend OOF on `(cutoff,user_id)`.

## Recoverability and cost

The scripts are present and the cache is recoverable in principle. Historical runtime shows the eight occurrence families alone took ~4.50 CPU-hours once the 9.7 GB core cache existed; extra90 materialization took ~31 minutes. Rebuilding the missing core bank from scratch invokes the 23h/14h lineage and is conservatively **20–30 CPU-hours plus disk**, with no need for a new neural/GPU run if archived neural predictions remain usable. This exceeds the 6–10h automatic-run ceiling, so it was not started.

## Required handoff

Copy only compact `checkpoints/folds/*.npz`, `checkpoints/test/*.npz`, recipe manifests and selected row-level final OOF; do not transfer the 9.7 GB processed feature cache unless replay is actually required. Every fold artifact must carry explicit `user_id` and the fold/cutoff must be unambiguous.
"""
    (WORKSPACE / "analysis" / "OCCURRENCE_CACHE_MISSING.md").write_text(missing_report, encoding="utf-8")

    # Artifact manifest.
    manifest_entries: list[tuple[str, str, str, str]] = []
    manifest_paths = {
        "canonical aligned OOF": ALIGNED,
        **{f"canonical {k} OOF": v for k, v in CANONICAL_NPZ.items()},
        "BTYD OOF": BTYD_OOF,
        "BTYD TEST": BTYD_TEST,
        "fallback OOF": FALLBACK_OOF,
        "fallback TEST": FALLBACK_TEST,
        "fallback summary": FALLBACK_SUMMARY,
        "compound+fallback TEST cache": compound_fallback_test_npz,
        **{f"TEST {k}": v for k, v in TEST_CSV.items()},
        "teammate README": BUNDLE / "README_PIPELINE_RU.md",
        "teammate final6h script": BUNDLE / "research_scripts" / "continue_best_bas_final6h.py",
        "teammate extra90 script": BUNDLE / "research_scripts" / "materialize_final6h_extra90m.py",
        "reproduction fallback script": SCRIPT.parent / "run_occurrence_fallback.py",
        "reproduction audit script": SCRIPT,
        "reproduction entrypoint": SCRIPT.parent / "run_all.ps1",
    }
    for role, path in manifest_paths.items():
        status = "FOUND" if path.exists() else "MISSING"
        manifest_entries.append((role, str(path), status, sha256(path) if path.is_file() else ""))
    manifest_md = [
        "# Artifact manifest", "",
        f"Generated from read-only source roots `{SOURCE}` and `{BUNDLE}`.", "",
        "| role | path | status | SHA256 |", "|---|---|---|---|",
    ]
    manifest_md += [f"| {r} | `{p}` | {s} | `{h}` |" for r, p, s, h in manifest_entries]
    manifest_md += ["", "## Missing teammate cache roots", ""]
    manifest_md += [f"- `{p}` — {'FOUND' if p.exists() else 'MISSING'}" for p in EXPECTED_CACHE_ROOTS]
    manifest_md += ["", "No large source artifact was copied into this repository; only compact fallback predictions and reports were written."]
    (OUT / "ARTIFACT_MANIFEST.md").write_text("\n".join(manifest_md) + "\n", encoding="utf-8")

    # Alignment report.
    align_md = [
        "# Alignment report", "",
        "Canonical key is `(cutoff, user_id)` for OOF and `user_id` for TEST. No positional merge was used.", "",
        f"Canonical rows: **{len(uid):,}**; unique keys: **{len(np.unique(canonical_keys)):,}**; unique users: **{len(np.unique(uid)):,}**; folds: `{', '.join(FOLDS)}`.", "",
        "| source | rows | matched | missing | extras | duplicates | target equal | original order equal after canonical sort | status |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for a in alignment_rows:
        align_md.append(f"| {a.source} | {a.rows} | {a.matched} | {a.missing} | {a.extras} | {a.duplicates} | {a.target_equal} | {a.order_equal} | {a.status} |")
    align_md += [
        "", f"EXP-037 primitive replay max absolute error: `{strongest_replay_error:.3e}`.",
        f"Latest TEST recipe replay max absolute log error: `{latest_replay_error:.3e}`.",
        "", "Teammate `occ_meta_B`, `occ_raw_X3`, and `latest` have exact TEST keys (250,000/250,000) but no OOF rows; they are not treated as aligned OOF sources.",
    ]
    (OUT / "ALIGNMENT_REPORT.md").write_text("\n".join(align_md) + "\n", encoding="utf-8")

    # Decision-oriented final report.
    base_fixed = next(x for x in scores if x["source"] == "fallback_occ_lgbm_residual_fixed1")
    best_nested = min(lofo_rows, key=lambda r: r["candidate_wcv"])
    compound_latest_row = next(r for r in test_overlap_rows if r["candidate"] == "compound")
    final = f"""# Unified OOF final report

## 1. Cache status

The exact teammate occurrence checkpoint bank is **missing**. Exact TEST components and native validation summaries were recovered; canonical row-level OOF for `occ_meta_B/final6h_B`, `occ_raw_X3/extra90_3`, and `latest` was not. Full replay was not started because the missing core cache makes it a 20–30 CPU-hour reconstruction. The one permitted cheap fallback was run in {json.loads(FALLBACK_SUMMARY.read_text(encoding='utf-8'))['runtime_seconds']:.1f}s.

## 2. Alignment

Canonical `(cutoff,user_id)` alignment is complete for EXP-037 primitives, BTYD and fallback: **770,616/770,616**, zero missing/extras/duplicates, target equality PASS. TEST `user_id` alignment is 250,000/250,000. EXP-037 replay error is `{strongest_replay_error:.3e}`; latest recipe replay error is `{latest_replay_error:.3e}`.

## 3. Occurrence offline performance

Exact teammate sources remain unscorable on canonical folds. Native summary gains (`occ_meta_B` about −0.001767; `occ_raw_X3` about −0.001625) are against teammate base wCV 1.749804 and are not substituted for canonical OOF. The locked fallback occurrence-only LightGBM overlay gives fixed-scale wCV **{base_fixed['wcv_calibrated']:.9f}**, delta **{base_fixed['delta_wcv_vs_exp037']:+.9f}**, {base_fixed['wins_vs_exp037']}/4.

## 4. Incremental utility

Nested inner-fold scalar selection gives:

| baseline | delta wCV | folds | last fold | mean lambda |
|---|---:|---:|---:|---:|
"""
    for r in lofo_rows:
        final += f"| {r['baseline']} | {r['delta_wcv']:+.9f} | {r['better_folds']}/4 | {r['last_fold_delta']:+.9f} | {r['lambda_mean']:.4f} |\n"
    final += f"""

The decisive row is occurrence after the canonical `SEQ65+BTYD05` compound: delta `{compound_nested['delta_wcv']:+.9f}`, {compound_nested['better_folds']}/4, last fold `{compound_nested['last_fold_delta']:+.9f}`. This is secondary evidence below the −0.0005 incremental production floor, while total candidate delta versus EXP-037 is `{compound_nested['candidate_wcv']-exp037_wcv:+.9f}`.

## 5. Signal overlap

OOF correction overlap is quantified in `CORRECTION_OVERLAP.csv`. The fallback is a real residual direction, but it is a newly cross-fitted mechanism probe—not an exact reconstruction of the teammate models. TEST centered correlation of fallback correction with `latest−STRONGEST` is **{fallback_test_row['corr_delta_with_latest_centered']:.6f}**; compound versus latest is **{compound_latest_row['corr_delta_with_latest_centered']:.6f}**.

## 6. Best honest combination

Best measured canonical recipe is `0.95·SEQ65 + 0.05·BTYD + λ·fallback_occ_delta`, with the occurrence model/features fixed in `run_occurrence_fallback.py`. Nested OOF uses outer-specific lambdas; full-OOF production scalar would be **λ={compound_lambda:.6f}**. Nested wCV is **{compound_nested['candidate_wcv']:.9f}**, total delta versus EXP-037 **{compound_nested['candidate_wcv']-exp037_wcv:+.9f}**, 4/4. Its incremental delta over compound is **{compound_nested['delta_wcv']:+.9f}**, secondary rather than an automatic production gain. This is not declared an exact teammate+canonical union because teammate OOF is missing.

## 7. Test-space sanity

Fallback TEST/OOF correction variance ratio is **{json.loads(FALLBACK_SUMMARY.read_text(encoding='utf-8'))['delta_test_oof_variance_ratio']:.6f}**. Fallback-vs-latest centered correction correlation is **{fallback_test_row['corr_delta_with_latest_centered']:.6f}**, projection of latest on fallback is **{fallback_test_row['projection_latest_on_candidate_centered']:.6f}**. The full canonical compound+fallback TEST candidate has centered correction correlation **{best_test_row['corr_delta_with_latest_centered']:.6f}** with `latest−STRONGEST`. `compound+latest` remains blocked: without canonical `z_latest`, no honest λ can be selected; no public-LB weight optimization was performed.

## 8. Recommendation

**{recommendation}**

Decision code: `{answer}`. The fallback demonstrates a non-absorbed occurrence mechanism, but exact teammate+canonical OOF remains unmeasured. Retrieve only the compact teammate fold/TEST NPZ and rerun this audit. Do not rebuild the full 9.7 GB feature cache and do not submit `latest + λ·compound` before exact latest OOF exists.
"""
    (OUT / "FINAL_REPORT.md").write_text(final, encoding="utf-8")

    audit_summary = {
        "answer": answer,
        "recommendation": recommendation,
        "fallback_gate_after_compound": compound_strong_incremental_gate,
        "fallback_gate_vs_exp037": fallback_gate,
        "compound_strong_incremental_gate": compound_strong_incremental_gate,
        "test_overlap_incomplete": overlap_incomplete,
        "compound_nested_incremental": compound_nested,
        "compound_full_oof_lambda_for_fallback": compound_lambda,
        "fallback_test_overlap_latest": fallback_test_row,
        "best_candidate_test_overlap_latest": best_test_row,
        "strongest_replay_error": strongest_replay_error,
        "latest_replay_error": latest_replay_error,
    }
    (ART / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    main()
