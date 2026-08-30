"""Reproducible OOF/test audit for the strongest compatible prediction sources.

The script is read-only with respect to the source repository.  It aligns every
OOF source by (cutoff, user_id), reproduces STRONGEST_CURRENT, evaluates fixed
pre-existing recipes, performs a leave-one-fold-out two-dimensional weight
audit, and writes one JSON intermediate used by the final report/CSV builder.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0]) / 15.0
LEVEL = 2.3293


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\Users\Admin\Desktop\OZON-E-CUP"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/intermediate/prediction_audit.json"),
    )
    return parser.parse_args()


def key_order(cutoff: np.ndarray, user_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key = np.rec.fromarrays([cutoff.astype("U10"), user_id.astype(np.int64)], names="cutoff,user_id")
    order = np.argsort(key, order=("cutoff", "user_id"))
    return key[order], order


def load_standard_oof(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        user_key = "user_id" if "user_id" in data.files else "uid"
        key, order = key_order(data["cutoff"], data[user_key])
        return {
            "key": key,
            "cutoff": data["cutoff"][order].astype("U10"),
            "user_id": data[user_key][order].astype(np.int64),
            "y": data["y"][order].astype(np.float64),
            "z": data["z"][order].astype(np.float64),
        }


def align_npz_field(
    path: Path,
    reference_key: np.ndarray,
    field: str,
    user_field: str = "user_id",
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        key, order = key_order(data["cutoff"], data[user_field])
        if not np.array_equal(key, reference_key):
            raise AssertionError(f"OOF key mismatch: {path}")
        return data[field][order].astype(np.float64)


def fold_scores(y: np.ndarray, z: np.ndarray, cutoff: np.ndarray) -> list[float]:
    ly = np.log1p(y)
    return [float(np.std(ly[cutoff == fold] - z[cutoff == fold])) for fold in FOLDS]


def weighted_cv(scores: list[float]) -> float:
    return float(FOLD_WEIGHTS @ np.asarray(scores, dtype=np.float64))


def calibrated_prediction(y: np.ndarray, z: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    ly = np.log1p(y)
    out = z.copy()
    for fold in FOLDS:
        mask = cutoff == fold
        out[mask] += float(np.mean(ly[mask] - z[mask]))
    return out


def model_metrics(y: np.ndarray, z: np.ndarray, cutoff: np.ndarray) -> dict[str, object]:
    scores = fold_scores(y, z, cutoff)
    return {
        "fold_scores": dict(zip(FOLDS, scores)),
        "wcv": weighted_cv(scores),
        "mean_cv": float(np.mean(scores)),
    }


def paired_cluster_se(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    cutoff: np.ndarray,
    user_id: np.ndarray,
) -> float:
    """First-order user-cluster SE for the weighted difference of fold RMSLEs."""
    ly = np.log1p(y)
    unique_users, inverse = np.unique(user_id, return_inverse=True)
    contribution = np.zeros(len(unique_users), dtype=np.float64)
    for fold_weight, fold in zip(FOLD_WEIGHTS, FOLDS):
        mask = cutoff == fold
        eb = ly[mask] - base[mask]
        ec = ly[mask] - candidate[mask]
        eb = eb - eb.mean()
        ec = ec - ec.mean()
        rb = float(np.sqrt(np.mean(eb**2)))
        rc = float(np.sqrt(np.mean(ec**2)))
        influence = (ec**2 - rc**2) / (2.0 * rc) - (eb**2 - rb**2) / (2.0 * rb)
        scaled = fold_weight * influence / int(mask.sum())
        contribution += np.bincount(inverse[mask], weights=scaled, minlength=len(unique_users))
    return float(np.sqrt(np.sum(contribution**2)))


def pair_metrics(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ly = np.log1p(y)
    ra = ly - a
    rb = ly - b
    d = a - b
    return {
        "prediction_corr": float(np.corrcoef(a, b)[0, 1]),
        "residual_corr": float(np.corrcoef(ra, rb)[0, 1]),
        "error_covariance": float(np.mean((ra - ra.mean()) * (rb - rb.mean()))),
        "disagreement_variance": float(np.var(d)),
        "mean_abs_disagreement": float(np.mean(np.abs(d))),
    }


def wcv_on_subset(
    y: np.ndarray,
    z_calibrated: np.ndarray,
    cutoff: np.ndarray,
    subset: np.ndarray,
) -> float | None:
    ly = np.log1p(y)
    values: list[float] = []
    for fold in FOLDS:
        mask = subset & (cutoff == fold)
        if int(mask.sum()) < 50:
            return None
        values.append(float(np.sqrt(np.mean((ly[mask] - z_calibrated[mask]) ** 2))))
    return weighted_cv(values)


def segment_masks(features: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
    positive = y > 0
    pos_y = y[positive]
    q25, q75, q95 = np.quantile(pos_y, [0.25, 0.75, 0.95])
    rec = features["rec_buy"]
    freq = features["w180_days_buy"]
    tenure = features["tenure_frac"]
    present = features["w365_days_present"]
    conversion = features["w180_cart2ord"]
    t25, t75 = np.quantile(tenure, [0.25, 0.75])
    p25, p75 = np.quantile(present, [0.25, 0.75])
    c25, c75 = np.quantile(conversion, [0.25, 0.75])
    return {
        "target:zero": ~positive,
        "target:positive_low_q25": positive & (y <= q25),
        "target:positive_mid_q25_q75": positive & (y > q25) & (y <= q75),
        "target:positive_high_q75_q95": positive & (y > q75) & (y <= q95),
        "target:positive_extreme_q95": positive & (y > q95),
        "recency:0_7": rec <= 7,
        "recency:8_14": (rec > 7) & (rec <= 14),
        "recency:15_60": (rec > 14) & (rec <= 60),
        "recency:61_365": (rec > 60) & (rec <= 365),
        "recency:never": rec > 365,
        "frequency:w180_0": freq == 0,
        "frequency:w180_1_3": (freq >= 1) & (freq <= 3),
        "frequency:w180_4_15": (freq >= 4) & (freq <= 15),
        "frequency:w180_16_plus": freq >= 16,
        "tenure:bottom_quartile": tenure <= t25,
        "tenure:top_quartile": tenure >= t75,
        "history_presence:bottom_quartile": present <= p25,
        "history_presence:top_quartile": present >= p75,
        "conversion:w180_bottom_quartile": conversion <= c25,
        "conversion:w180_top_quartile": conversion >= c75,
    }


def load_segment_features(source_root: Path, reference_key: np.ndarray) -> dict[str, np.ndarray]:
    import polars as pl

    path = source_root / "artifacts" / "RESDISC_053" / "aligned_oof.parquet"
    cols = [
        "cutoff",
        "user_id",
        "rec_buy",
        "w180_days_buy",
        "tenure_frac",
        "w365_days_present",
        "w180_cart2ord",
    ]
    frame = pl.read_parquet(path, columns=cols)
    cutoff = frame["cutoff"].to_numpy()
    user_id = frame["user_id"].to_numpy()
    key, order = key_order(cutoff, user_id)
    if not np.array_equal(key, reference_key):
        raise AssertionError("segment feature key mismatch")
    return {name: frame[name].to_numpy()[order].astype(np.float64) for name in cols[2:]}


def nested_grid_audit(
    y: np.ndarray,
    cutoff: np.ndarray,
    strong: np.ndarray,
    seq65: np.ndarray,
    btyd: np.ndarray,
) -> dict[str, object]:
    seq_weights = [0.45, 0.55, 0.65, 0.75]
    btyd_weights = [0.0, 0.025, 0.05, 0.075, 0.10]
    candidates: dict[tuple[float, float], np.ndarray] = {}
    for seq_weight in seq_weights:
        alpha = (seq_weight - 0.45) / 0.20
        z_seq = strong + alpha * (seq65 - strong)
        for btyd_weight in btyd_weights:
            candidates[(seq_weight, btyd_weight)] = (1.0 - btyd_weight) * z_seq + btyd_weight * btyd

    fold_score_cache = {
        key: np.asarray(fold_scores(y, z, cutoff), dtype=np.float64) for key, z in candidates.items()
    }
    full_rank = sorted(
        (
            (weighted_cv(scores.tolist()), abs(sw - 0.45) + bw, sw, bw)
            for (sw, bw), scores in fold_score_cache.items()
        )
    )
    heldout_scores: list[float] = []
    selections: list[dict[str, float | str]] = []
    for outer_idx, outer_fold in enumerate(FOLDS):
        train_weights = np.delete(FOLD_WEIGHTS, outer_idx)
        train_weights = train_weights / train_weights.sum()
        ranked = []
        for (sw, bw), scores in fold_score_cache.items():
            train_score = float(train_weights @ np.delete(scores, outer_idx))
            ranked.append((round(train_score / 1e-5) * 1e-5, abs(sw - 0.45) + bw, train_score, sw, bw))
        _, _, train_score, sw, bw = min(ranked)
        outer_score = float(fold_score_cache[(sw, bw)][outer_idx])
        heldout_scores.append(outer_score)
        selections.append(
            {
                "outer_fold": outer_fold,
                "selected_sequence_weight": sw,
                "selected_btyd_weight": bw,
                "training_wcv": train_score,
                "heldout_rmsle": outer_score,
            }
        )
    nested_wcv = weighted_cv(heldout_scores)
    base_wcv = weighted_cv(fold_scores(y, strong, cutoff))
    best = full_rank[0]
    return {
        "grid_sequence_weights": seq_weights,
        "grid_btyd_weights": btyd_weights,
        "full_oof_best": {"wcv": best[0], "sequence_weight": best[2], "btyd_weight": best[3]},
        "lofo_selections": selections,
        "nested_wcv": nested_wcv,
        "nested_delta_vs_strongest": nested_wcv - base_wcv,
    }


def load_test_array(artifacts: Path, name: str, reference_uid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(artifacts / f"ztest_{name}.npy").astype(np.float64)
    uid = np.load(artifacts / f"uid_{name}.npy").astype(np.int64)
    if reference_uid is not None and not np.array_equal(uid, reference_uid):
        raise AssertionError(f"test uid mismatch: {name}")
    return uid, z


def main() -> None:
    args = parse_args()
    artifacts = args.source_root / "artifacts"

    component_names = ["S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3"]
    loaded = {name: load_standard_oof(artifacts / f"oof_{name}.npz") for name in component_names}
    reference = loaded["S1-E03a"]
    for name, data in loaded.items():
        if not np.array_equal(data["key"], reference["key"]):
            raise AssertionError(f"component key mismatch: {name}")
        if not np.array_equal(data["y"], reference["y"]):
            raise AssertionError(f"component target mismatch: {name}")

    y = reference["y"]
    cutoff = reference["cutoff"]
    Z = {name: data["z"] for name, data in loaded.items()}
    strong = (
        0.10 * Z["S1-E03a"]
        + 0.20 * Z["S1-E02"]
        + 0.25 * Z["S1-DIST"]
        + 0.225 * Z["ETX-AVG3"]
        + 0.225 * Z["SEQ-AVG3"]
    )
    seq65 = (
        0.10 * Z["S1-E03a"]
        + 0.10 * Z["S1-E02"]
        + 0.15 * Z["S1-DIST"]
        + 0.325 * Z["ETX-AVG3"]
        + 0.325 * Z["SEQ-AVG3"]
    )

    btyd_path = artifacts / "BTYD_STABLE_EXP051" / "oof_raw.npz"
    btyd = align_npz_field(btyd_path, reference["key"], "z_btyd")
    btyd_strong = align_npz_field(btyd_path, reference["key"], "z_strongest")
    if float(np.max(np.abs(btyd_strong - strong))) > 5e-7:
        raise AssertionError("BTYD baseline does not reconstruct STRONGEST_CURRENT")
    btyd05 = 0.95 * strong + 0.05 * btyd
    compound = 0.95 * seq65 + 0.05 * btyd

    extra: dict[str, np.ndarray] = {}
    for name in ["S04-B", "RIDGE15", "SEQ-D3A-AVG3"]:
        extra[name] = align_npz_field(artifacts / f"oof_{name}.npz", reference["key"], "z")
    fresh_path = artifacts / "oof_FRESH_CONTRAST_MOE.npz"
    fresh_base = align_npz_field(fresh_path, reference["key"], "z_base", user_field="uid")
    fresh_correction = align_npz_field(
        fresh_path, reference["key"], "fresh_processed_nested", user_field="uid"
    )
    if float(np.max(np.abs(fresh_base - strong))) > 5e-7:
        raise AssertionError("FRESH baseline does not reconstruct STRONGEST_CURRENT")
    extra["FRESH-corrected"] = strong + fresh_correction

    btyd05_fresh = btyd05 + fresh_correction
    compound_fresh = compound + fresh_correction

    sources: dict[str, np.ndarray] = {
        "CAP": Z["S1-E03a"],
        "UNC": Z["S1-E02"],
        "DIST": Z["S1-DIST"],
        "ETX-AVG3": Z["ETX-AVG3"],
        "SEQ-AVG3": Z["SEQ-AVG3"],
        **extra,
        "BTYD": btyd,
        "STRONGEST_CURRENT": strong,
        "SEQ65": seq65,
        "BTYD05": btyd05,
        "COMPOUND_SEQ65_BTYD05": compound,
        "BTYD05_FRESH": btyd05_fresh,
        "COMPOUND_SEQ65_BTYD05_FRESH": compound_fresh,
    }
    models = {name: model_metrics(y, z, cutoff) for name, z in sources.items()}
    expected_baseline = 1.7475098627078645
    if abs(float(models["STRONGEST_CURRENT"]["wcv"]) - expected_baseline) > 1e-9:
        raise AssertionError("STRONGEST_CURRENT score mismatch")

    diversity: list[dict[str, object]] = []
    for a_name, b_name in itertools.combinations(sources, 2):
        row: dict[str, object] = {"source_a": a_name, "source_b": b_name, "oof_rows": len(y)}
        row.update(pair_metrics(y, sources[a_name], sources[b_name]))
        diversity.append(row)

    features = load_segment_features(args.source_root, reference["key"])
    masks = segment_masks(features, y)
    segment_candidates = {
        "STRONGEST_CURRENT": strong,
        "SEQ65": seq65,
        "BTYD05": btyd05,
        "COMPOUND_SEQ65_BTYD05": compound,
    }
    calibrated = {name: calibrated_prediction(y, z, cutoff) for name, z in segment_candidates.items()}
    segments: list[dict[str, object]] = []
    for segment, mask in masks.items():
        base_score = wcv_on_subset(y, calibrated["STRONGEST_CURRENT"], cutoff, mask)
        if base_score is None:
            continue
        for name, z_cal in calibrated.items():
            score = wcv_on_subset(y, z_cal, cutoff, mask)
            if score is None:
                continue
            segments.append(
                {
                    "segment": segment,
                    "model": name,
                    "rows": int(mask.sum()),
                    "wcv": score,
                    "delta_vs_strongest": score - base_score,
                }
            )

    base_wcv = float(models["STRONGEST_CURRENT"]["wcv"])
    seq_delta = float(models["SEQ65"]["wcv"]) - base_wcv
    btyd_delta = float(models["BTYD05"]["wcv"]) - base_wcv
    compound_delta = float(models["COMPOUND_SEQ65_BTYD05"]["wcv"]) - base_wcv
    correction_seq = seq65 - strong
    correction_btyd = btyd05 - strong
    ly = np.log1p(y)
    residual = ly - strong
    interaction = {
        "seq65_delta": seq_delta,
        "btyd05_delta": btyd_delta,
        "compound_delta": compound_delta,
        "arithmetic_sum_of_standalone_deltas": seq_delta + btyd_delta,
        "interaction_vs_arithmetic_sum": compound_delta - seq_delta - btyd_delta,
        "correction_pearson": float(np.corrcoef(correction_seq, correction_btyd)[0, 1]),
        "correction_covariance": float(np.mean((correction_seq - correction_seq.mean()) * (correction_btyd - correction_btyd.mean()))),
        "seq_residual_alignment": float(np.corrcoef(correction_seq - correction_seq.mean(), residual)[0, 1]),
        "btyd_residual_alignment": float(np.corrcoef(correction_btyd - correction_btyd.mean(), residual)[0, 1]),
        "fresh_delta": float(models["FRESH-corrected"]["wcv"]) - base_wcv,
        "btyd05_fresh_delta": float(models["BTYD05_FRESH"]["wcv"]) - base_wcv,
        "compound_fresh_delta": float(models["COMPOUND_SEQ65_BTYD05_FRESH"]["wcv"]) - base_wcv,
        "fresh_seq_correction_pearson": float(np.corrcoef(fresh_correction, correction_seq)[0, 1]),
        "fresh_btyd_correction_pearson": float(np.corrcoef(fresh_correction, correction_btyd)[0, 1]),
        "paired_user_cluster_se": {
            name: paired_cluster_se(y, strong, sources[name], cutoff, reference["user_id"])
            for name in [
                "SEQ65",
                "BTYD05",
                "COMPOUND_SEQ65_BTYD05",
                "FRESH-corrected",
                "BTYD05_FRESH",
                "COMPOUND_SEQ65_BTYD05_FRESH",
            ]
        },
    }
    grid = nested_grid_audit(y, cutoff, strong, seq65, btyd)

    test_raw_names = [
        "S1-CAP",
        "S1-UNC",
        "S1-DIST",
        "SEQ-01",
        "SEQ-C289-S43",
        "SEQ-C289-S44",
        "ETX-01-S42-DCW",
        "ETX-01-S43-DCW",
        "ETX-01-S44-DCW",
    ]
    test_parts: dict[str, np.ndarray] = {}
    test_uid: np.ndarray | None = None
    for name in test_raw_names:
        uid, z = load_test_array(artifacts, name, test_uid)
        if test_uid is None:
            test_uid = uid
        test_parts[name] = z
    assert test_uid is not None
    seq_test = (test_parts["SEQ-01"] + test_parts["SEQ-C289-S43"] + test_parts["SEQ-C289-S44"]) / 3.0
    etx_test = (
        test_parts["ETX-01-S42-DCW"]
        + test_parts["ETX-01-S43-DCW"]
        + test_parts["ETX-01-S44-DCW"]
    ) / 3.0
    strong_test = (
        0.10 * test_parts["S1-CAP"]
        + 0.20 * test_parts["S1-UNC"]
        + 0.25 * test_parts["S1-DIST"]
        + 0.225 * seq_test
        + 0.225 * etx_test
    )
    seq65_test = (
        0.10 * test_parts["S1-CAP"]
        + 0.10 * test_parts["S1-UNC"]
        + 0.15 * test_parts["S1-DIST"]
        + 0.325 * seq_test
        + 0.325 * etx_test
    )
    with np.load(artifacts / "BTYD_STABLE_EXP051" / "test_raw.npz", allow_pickle=False) as data:
        if not np.array_equal(data["user_id"].astype(np.int64), test_uid):
            raise AssertionError("BTYD test uid mismatch")
        btyd_test = data["z_btyd"].astype(np.float64)
        btyd_test_strong = data["z_strongest"].astype(np.float64)
    if float(np.max(np.abs(btyd_test_strong - strong_test))) > 5e-7:
        raise AssertionError("BTYD test baseline mismatch")
    btyd05_test = 0.95 * strong_test + 0.05 * btyd_test
    compound_test = 0.95 * seq65_test + 0.05 * btyd_test
    test_sources = {
        "STRONGEST_CURRENT": strong_test,
        "SEQ65": seq65_test,
        "BTYD": btyd_test,
        "BTYD05": btyd05_test,
        "COMPOUND_SEQ65_BTYD05": compound_test,
    }
    test_diversity: list[dict[str, object]] = []
    for a_name, b_name in itertools.combinations(test_sources, 2):
        d = test_sources[a_name] - test_sources[b_name]
        test_diversity.append(
            {
                "source_a": a_name,
                "source_b": b_name,
                "test_rows": len(test_uid),
                "test_prediction_corr": float(np.corrcoef(test_sources[a_name], test_sources[b_name])[0, 1]),
                "test_disagreement_variance": float(np.var(d)),
                "test_mean_abs_disagreement": float(np.mean(np.abs(d))),
            }
        )
    test_regime = {
        "seq65_correction_variance_ratio_test_over_oof": float(np.var(seq65_test - strong_test) / np.var(seq65 - strong)),
        "btyd05_correction_variance_ratio_test_over_oof": float(np.var(btyd05_test - strong_test) / np.var(btyd05 - strong)),
        "compound_correction_variance_ratio_test_over_oof": float(np.var(compound_test - strong_test) / np.var(compound - strong)),
        "compound_test_mean_raw_z": float(np.mean(compound_test)),
        "compound_test_level_shift": float(LEVEL - np.mean(compound_test)),
    }

    report = {
        "source_root": str(args.source_root),
        "rows": len(y),
        "folds": FOLDS,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "models": models,
        "interaction": interaction,
        "nested_grid": grid,
        "diversity": diversity,
        "test_diversity": test_diversity,
        "test_regime": test_regime,
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"interaction": interaction, "nested_grid": grid, "test_regime": test_regime}, indent=2))


if __name__ == "__main__":
    main()
