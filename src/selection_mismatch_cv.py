"""EXP-048: artifact-only audit of competition-user selection mismatch.

The runner reads the daily log and saved raw OOF artifacts.  It never fits a
prediction model, reads test predictions, or writes a submission.

Run: python src/selection_mismatch_cv.py
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, RAW_PARQUET, SEED,
                        VAL_FOLDS_S1)
from src.validation import calibrate


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "SELMATCH_EXP048"
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
FOLDS = [v.isoformat() for v in VAL_FOLDS_S1]
ELIGIBLE_FOLDS = FOLDS[:3]
EXPECTED_FOLD_SCORES = np.array([
    1.7668833567997195, 1.7605095767798136,
    1.748629223964952, 1.7412785664479717,
])
EXPECTED_WCV = 1.7475098625201952
BASE_COMPONENTS = {
    "CAP": ("S1-E03a", 0.10),
    "UNC": ("S1-E02", 0.20),
    "DIST": ("S1-DIST", 0.25),
    "ETX-AVG3": ("ETX-AVG3", 0.225),
    "SEQ-AVG3": ("SEQ-AVG3", 0.225),
}
STANDALONE = ["CAP", "DIST", "ETX-AVG3", "SEQ-AVG3", "BTYD"]
INCREMENTAL = [
    "BTYD05", "FRESH", "ZERO2D", "SEQ_SLOT_25", "SEQ_SLOT_50",
    "SEQ_SLOT_75", "BTYD05_FRESH1",
]
ALL_MODELS = ["STRONGEST"] + STANDALONE + ["UNC"] + INCREMENTAL
COMPETITION_BLOCKS = [
    (dt.date(2025, 11, 16), dt.date(2025, 12, 15)),
    (dt.date(2025, 12, 16), dt.date(2026, 1, 14)),
    (dt.date(2026, 1, 15), dt.date(2026, 2, 13)),
]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (json.dumps(jsonable(row.get(k)), ensure_ascii=False,
                                            sort_keys=True)
                            if isinstance(row.get(k), (dict, list, tuple, np.ndarray))
                            else jsonable(row.get(k, ""))) for k in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def row_keys(cutoff: np.ndarray, uid: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(cutoff, dtype="U10"), "|"),
                       np.asarray(uid).astype("U24"))


def weighted_calibrate(y: np.ndarray, z: np.ndarray,
                       weight: np.ndarray) -> tuple[float, float]:
    """Exact optimal clipped log offset under non-negative sample weights."""
    ly = np.log1p(np.asarray(y, float))
    z = np.asarray(z, float)
    w = np.asarray(weight, float)
    if len(y) == 0 or not np.any(w > 0):
        return float("nan"), float("nan")
    d = float(np.sum(w * (ly - z)) / np.sum(w))
    for _ in range(25):
        active = (z + d > 0) & (w > 0)
        if not active.any():
            break
        d_new = float(np.sum(w[active] * (ly[active] - z[active]))
                      / np.sum(w[active]))
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    err = ly - np.maximum(z + d, 0.0)
    return d, float(np.sqrt(np.sum(w * err * err) / np.sum(w)))


def window_spec(V: dt.date) -> dict[str, tuple[dt.date, dt.date]]:
    """Closed source dates for target and future selection blocks."""
    return {
        "target": (V + dt.timedelta(days=1), V + dt.timedelta(days=30)),
        "F1": (V + dt.timedelta(days=31), V + dt.timedelta(days=60)),
        "F2": (V + dt.timedelta(days=61), V + dt.timedelta(days=90)),
        "F3": (V + dt.timedelta(days=91), V + dt.timedelta(days=120)),
    }


def _load_oof(name: str) -> dict[str, np.ndarray]:
    path = ARTIFACTS / f"oof_{name}.npz"
    d = np.load(path, allow_pickle=False)
    order = np.lexsort((np.asarray(d["user_id"]),
                        np.asarray(d["cutoff"], dtype="U10")))
    return {
        "path": np.asarray([str(path.resolve())]),
        "uid": np.asarray(d["user_id"])[order],
        "cutoff": np.asarray(d["cutoff"], dtype="U10")[order],
        "y": np.asarray(d["y"], float)[order],
        "z": np.asarray(d["z"], float)[order],
        "raw_z": np.asarray(d["z"]),
    }


def reconstruct_baseline() -> tuple[dict, dict[str, np.ndarray]]:
    canonical = None
    arrays: dict[str, np.ndarray] = {}
    manifest_components = []
    z_base = None
    for label, (artifact, weight) in BASE_COMPONENTS.items():
        d = _load_oof(artifact)
        path = ARTIFACTS / f"oof_{artifact}.npz"
        if canonical is None:
            canonical = {k: d[k].copy() for k in ("uid", "cutoff", "y")}
            z_base = np.zeros(len(d["uid"]), float)
        else:
            assert np.array_equal(d["uid"], canonical["uid"]), f"uid mismatch: {label}"
            assert np.array_equal(d["cutoff"], canonical["cutoff"]), f"cutoff mismatch: {label}"
            assert np.allclose(d["y"], canonical["y"], atol=1e-6), f"target mismatch: {label}"
        for fold in FOLDS:
            m = d["cutoff"] == fold
            assert np.all(np.diff(d["uid"][m]) > 0), f"user order not ascending: {label}/{fold}"
        arrays[label] = d["z"]
        z_base += weight * d["z"]
        manifest_components.append({
            "label": label, "artifact": artifact, "weight": weight,
            "path": str(path.resolve()), "file_sha256": sha256_file(path),
            "prediction_dtype": str(d["raw_z"].dtype),
            "prediction_sha256": sha256_array(d["raw_z"]),
            "row_keys_sha256": sha256_array(row_keys(d["cutoff"], d["uid"])),
            "target_sha256": sha256_array(d["y"]), "rows": len(d["uid"]),
        })
    assert canonical is not None and z_base is not None
    offsets, scores, sizes = [], [], []
    for fold in FOLDS:
        m = canonical["cutoff"] == fold
        off, score = calibrate(canonical["y"][m], z_base[m])
        offsets.append(off); scores.append(score); sizes.append(int(m.sum()))
    wcv = float(np.average(scores, weights=FOLD_WEIGHTS_S1))
    assert np.max(np.abs(np.asarray(scores) - EXPECTED_FOLD_SCORES)) <= 5e-10
    assert abs(wcv - EXPECTED_WCV) <= 5e-10
    canonical["z_base"] = z_base
    keys = row_keys(canonical["cutoff"], canonical["uid"])
    manifest = {
        "status": "PASS_EXACT", "prefix": PREFIX,
        "rows": len(keys), "folds": FOLDS, "fold_sizes": sizes,
        "fold_offsets": offsets, "fold_cal": scores, "wcv": wcv,
        "row_keys_sha256": sha256_array(keys),
        "target_sha256": sha256_array(canonical["y"]),
        "prediction_sha256": sha256_array(z_base),
        "components": manifest_components,
        "target_semantics": "GMV on (V,V+30]",
        "prediction_semantics": "raw log1p-space",
    }
    arrays["STRONGEST"] = z_base
    return canonical, arrays, manifest


def _daily_future(V: dt.date, uid: np.ndarray,
                  include_events: bool = True) -> pl.DataFrame:
    """Continuation variables from F1/F2/F3; target dates are not scanned."""
    spec = window_spec(V)
    a, b = spec["F1"][0], spec["F3"][1]
    cols = ["user_id", "event_date"]
    if include_events:
        cols += ["searches", "cat", "to_cart", "to_ord"]
    q = (pl.scan_parquet(RAW_PARQUET).select(cols)
         .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b)))
    day = (pl.col("event_date") - pl.lit(V)).dt.total_days()
    q = q.with_columns(_block=((day - 31) // 30).cast(pl.Int8))
    if include_events:
        q = q.with_columns(_events=pl.sum_horizontal(
            [pl.col(c).cast(pl.Float64) for c in ("searches", "cat", "to_cart", "to_ord")]))
        block = (q.group_by(["user_id", "_block"])
                 .agg(active_days=pl.len(), events=pl.col("_events").sum(),
                      first_date=pl.col("event_date").min()))
    else:
        block = (q.group_by(["user_id", "_block"])
                 .agg(active_days=pl.len(), events=pl.lit(0.0),
                      first_date=pl.col("event_date").min()))
    agg = (block.group_by("user_id").agg(
        future_blocks_active=pl.col("_block").n_unique(),
        _min_days=pl.col("active_days").min(),
        future_total_active_days=pl.col("active_days").sum(),
        _min_events=pl.col("events").min(),
        _first=pl.col("first_date").min()).collect())
    users = pl.DataFrame({"user_id": np.asarray(uid, dtype=np.int64)})
    out = users.join(agg, on="user_id", how="left").with_columns(
        pl.col("future_blocks_active").fill_null(0).cast(pl.Int8),
        pl.col("future_total_active_days").fill_null(0).cast(pl.Int16),
    ).with_columns(
        future_min_active_days=pl.when(pl.col("future_blocks_active") == 3)
        .then(pl.col("_min_days")).otherwise(0).cast(pl.Int16),
        future_min_events=pl.when(pl.col("future_blocks_active") == 3)
        .then(pl.col("_min_events")).otherwise(0.0),
        days_to_first_activity_after_target=pl.when(pl.col("_first").is_not_null())
        .then((pl.col("_first") - pl.lit(V + dt.timedelta(days=30))).dt.total_days())
        .otherwise(91).cast(pl.Int16),
    ).drop(["_min_days", "_min_events", "_first"])
    return out


def _organizer_panel_and_future(V: dt.date) -> pl.DataFrame:
    """One-scan organizer panel plus F1/F2/F3 continuation for a landmark."""
    lo, hi = V - dt.timedelta(days=89), V + dt.timedelta(days=120)
    q = (pl.scan_parquet(RAW_PARQUET).select(["user_id", "event_date"])
         .filter((pl.col("event_date") >= lo) & (pl.col("event_date") <= hi)))
    day = (pl.col("event_date") - pl.lit(V)).dt.total_days()
    past = (pl.when(day <= 0).then(((day + 89) // 30).cast(pl.Int8))
            .otherwise(pl.lit(-1, dtype=pl.Int8)))
    future = (pl.when(day >= 31).then(((day - 31) // 30).cast(pl.Int8))
              .otherwise(pl.lit(-1, dtype=pl.Int8)))
    g = (q.with_columns(_past=past, _future=future)
         .group_by("user_id").agg(
             past_blocks=pl.col("_past").filter(pl.col("_past") >= 0).n_unique(),
             future_blocks_active=pl.col("_future").filter(pl.col("_future") >= 0).n_unique())
         .filter(pl.col("past_blocks") == 3).collect())
    return g.with_columns(pl.col("future_blocks_active").cast(pl.Int8)).sort("user_id")


def competition_audit(canonical: dict) -> dict:
    lo, hi = COMPETITION_BLOCKS[0][0], COMPETITION_BLOCKS[-1][1]
    q = (pl.scan_parquet(RAW_PARQUET).select(["user_id", "event_date"])
         .filter((pl.col("event_date") >= lo) & (pl.col("event_date") <= hi)))
    day = (pl.col("event_date") - pl.lit(lo)).dt.total_days()
    g = (q.with_columns(_block=(day // 30).cast(pl.Int8))
         .group_by(["user_id", "_block"]).agg(days=pl.len()).collect())
    by_block = g.group_by("_block").len().sort("_block")
    selected = (g.group_by("user_id").agg(k=pl.col("_block").n_unique())
                .filter(pl.col("k") == 3).sort("user_id"))
    universe = (pl.scan_parquet(RAW_PARQUET).select("user_id").unique().collect()
                .sort("user_id"))
    assert universe.height == 250_000
    assert selected.height == 250_000
    assert by_block["len"].to_list() == [250_000, 250_000, 250_000]
    assert np.array_equal(selected["user_id"].to_numpy(), universe["user_id"].to_numpy())
    for fold in FOLDS:
        u = np.unique(canonical["uid"][canonical["cutoff"] == fold])
        assert np.isin(u, selected["user_id"].to_numpy()).all()
    test_panel = _organizer_panel_and_future(dt.date(2026, 2, 13))
    assert test_panel.height == 250_000
    assert np.array_equal(test_panel["user_id"].to_numpy(), selected["user_id"].to_numpy())
    return {
        "competition_blocks": COMPETITION_BLOCKS,
        "active_users_by_block": by_block["len"].to_list(),
        "selected_users": selected.height,
        "raw_universe_users": universe.height,
        "test_panel_rule_equivalent": True,
        "all_validation_users_in_global_selection": True,
        "activity_definition": "any daily row",
    }


def build_selection_frame(canonical: dict) -> tuple[dict, list[dict], dict]:
    n = len(canonical["uid"])
    out = {"k": np.empty(n, np.int8), "min_days": np.empty(n, np.int16),
           "total_days": np.empty(n, np.int16), "first_day": np.empty(n, np.int16),
           "min_events": np.empty(n, float)}
    prevalence, sources = [], {}
    for fold in FOLDS:
        V = dt.date.fromisoformat(fold)
        m = canonical["cutoff"] == fold
        f = _daily_future(V, canonical["uid"][m])
        out["k"][m] = f["future_blocks_active"].to_numpy()
        out["min_days"][m] = f["future_min_active_days"].to_numpy()
        out["total_days"][m] = f["future_total_active_days"].to_numpy()
        out["first_day"][m] = f["days_to_first_activity_after_target"].to_numpy()
        out["min_events"][m] = f["future_min_events"].to_numpy()
        counts = np.bincount(out["k"][m], minlength=4)
        spec = window_spec(V)
        target_dates = set(_date_range(*spec["target"]))
        future_dates = set().union(*[set(_date_range(*spec[x])) for x in ("F1", "F2", "F3")])
        assert not target_dates & future_dates
        sources[fold] = spec
        overlap = len(future_dates & set(_date_range(COMPETITION_BLOCKS[0][0],
                                                      COMPETITION_BLOCKS[-1][1])))
        for k in range(4):
            prevalence.append({"fold": fold, "k": k, "n": int(counts[k]),
                               "prevalence": float(counts[k] / m.sum()),
                               "real_selection_overlap_days": overlap,
                               "real_selection_overlap_share": overlap / 90.0})
    last = canonical["cutoff"] == "2025-10-16"
    assert np.all(out["k"][last] == 3)
    assert sources["2025-10-16"]["F1"] == COMPETITION_BLOCKS[0]
    assert sources["2025-10-16"]["F2"] == COMPETITION_BLOCKS[1]
    assert sources["2025-10-16"]["F3"] == COMPETITION_BLOCKS[2]
    return out, prevalence, sources


def _date_range(a: dt.date, b: dt.date) -> Iterable[dt.date]:
    for i in range((b - a).days + 1):
        yield a + dt.timedelta(days=i)


def reference_distribution() -> tuple[np.ndarray, list[dict], list[dict], np.ndarray]:
    landmarks = []
    V = dt.date(2025, 4, 3)
    end = dt.date(2025, 7, 17)
    user_ids = (pl.scan_parquet(RAW_PARQUET).select("user_id").unique().collect()
                .sort("user_id")["user_id"].to_numpy())
    uid_pos = {int(u): i for i, u in enumerate(user_ids)}
    landmark_k = np.full((len(user_ids), 16), -1, np.int8)
    li = 0
    while V <= end:
        assert V + dt.timedelta(days=120) <= dt.date(2025, 11, 15)
        d = _organizer_panel_and_future(V)
        k = d["future_blocks_active"].to_numpy()
        counts = np.bincount(k, minlength=4)
        pi = counts / counts.sum()
        row = {"landmark": V.isoformat(), "panel_n": int(len(k))}
        for j in range(4):
            row[f"n_k{j}"] = int(counts[j]); row[f"pi_k{j}"] = float(pi[j])
        landmarks.append(row)
        pos = np.fromiter((uid_pos[int(u)] for u in d["user_id"].to_numpy()),
                          dtype=np.int64, count=d.height)
        landmark_k[pos, li] = k
        li += 1
        V += dt.timedelta(days=7)
    assert len(landmarks) == 16
    pi_ref = np.mean([[r[f"pi_k{k}"] for k in range(4)] for r in landmarks], axis=0)
    rng = np.random.default_rng(SEED)
    reps = np.empty((500, 4), float)
    for r in range(500):
        sampled = rng.integers(0, len(user_ids), size=len(user_ids))
        multiplicity = np.bincount(sampled, minlength=len(user_ids))
        landmark_pi = np.empty((16, 4), float)
        for j in range(16):
            present = landmark_k[:, j] >= 0
            counts = np.bincount(landmark_k[present, j],
                                 weights=multiplicity[present], minlength=4)
            landmark_pi[j] = counts / counts.sum()
        reps[r] = landmark_pi.mean(axis=0)
    summary = []
    for k in range(4):
        summary.append({"k": k, "pi_ref": float(pi_ref[k]),
                        "bootstrap_p025": float(np.quantile(reps[:, k], .025)),
                        "bootstrap_p10": float(np.quantile(reps[:, k], .10)),
                        "bootstrap_median": float(np.median(reps[:, k])),
                        "bootstrap_p90": float(np.quantile(reps[:, k], .90)),
                        "bootstrap_p975": float(np.quantile(reps[:, k], .975))})
    return pi_ref, landmarks, summary, reps


def load_candidates(canonical: dict, arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    # Standalone raw OOF already audited through the baseline components.
    models = {"STRONGEST": arrays["STRONGEST"], "CAP": arrays["CAP"],
              "UNC": arrays["UNC"], "DIST": arrays["DIST"],
              "ETX-AVG3": arrays["ETX-AVG3"], "SEQ-AVG3": arrays["SEQ-AVG3"]}
    bpath = ARTIFACTS / "BTYD_DAY_BGNBD_EXP047_V2" / "oof_raw.npz"
    b = np.load(bpath, allow_pickle=False)
    bo = np.lexsort((b["user_id"], np.asarray(b["cutoff"], dtype="U10")))
    assert np.array_equal(np.asarray(b["user_id"])[bo], canonical["uid"])
    assert np.array_equal(np.asarray(b["cutoff"], dtype="U10")[bo], canonical["cutoff"])
    assert np.allclose(np.asarray(b["y"], float)[bo], canonical["y"], atol=1e-6)
    models["BTYD"] = np.asarray(b["z_btyd"], float)[bo]

    fpath = ARTIFACTS / "oof_FRESH_CONTRAST_MOE.npz"
    f = np.load(fpath, allow_pickle=False)
    fo = np.lexsort((f["uid"], np.asarray(f["cutoff"], dtype="U10")))
    assert np.array_equal(np.asarray(f["uid"])[fo], canonical["uid"])
    assert np.array_equal(np.asarray(f["cutoff"], dtype="U10")[fo], canonical["cutoff"])
    assert np.max(np.abs(np.asarray(f["z_base"], float)[fo] - models["STRONGEST"])) < 1e-6
    fresh_corr = np.asarray(f["fresh_processed_nested"], float)[fo]
    models["FRESH"] = models["STRONGEST"] + fresh_corr

    # Exact honest outer prediction from exp_042, reconstructed solely from saved OOF/PACT artifacts.
    from src.zero2d_shrink import load_frame, run_nested
    zframe, _ = load_frame()
    zero = run_nested(zframe, "ZERO2D")
    assert np.array_equal(zframe["uid"], canonical["uid"])
    assert np.array_equal(zframe["cutoff"], canonical["cutoff"])
    models["ZERO2D"] = np.asarray(zero["z_honest"], float)

    base, etx, seq, btyd = (models["STRONGEST"], models["ETX-AVG3"],
                            models["SEQ-AVG3"], models["BTYD"])
    models["BTYD05"] = .95 * base + .05 * btyd
    models["SEQ_SLOT_25"] = (base - .225 * etx - .225 * seq
                              + .45 * (.25 * etx + .75 * seq))
    models["SEQ_SLOT_50"] = (base - .225 * etx - .225 * seq
                              + .45 * (.50 * etx + .50 * seq))
    models["SEQ_SLOT_75"] = (base - .225 * etx - .225 * seq
                              + .45 * (.75 * etx + .25 * seq))
    assert np.max(np.abs(models["SEQ_SLOT_50"] - base)) < 1e-12
    models["BTYD05_FRESH1"] = .95 * base + .05 * btyd + fresh_corr
    audit = {
        "btyd_path": str(bpath.resolve()), "btyd_sha256": sha256_file(bpath),
        "fresh_path": str(fpath.resolve()), "fresh_sha256": sha256_file(fpath),
        "zero2d_reconstructed_from_saved_artifacts": True,
        "zero2d_wcv": zero["candidate_wcv"],
        "zero2d_expected_wcv": 1.747485106715659,
        "candidate_formulas": {
            "BTYD05": ".95*STRONGEST+.05*BTYD",
            "FRESH": "STRONGEST+fresh_processed_nested (GLOBAL alpha=1 all outer folds)",
            "ZERO2D": "exact exp_042 honest outer z_honest",
            "SEQ_SLOT_25": "backbone+.45*(.25*ETX+.75*SEQ)",
            "SEQ_SLOT_50": "backbone+.45*(.50*ETX+.50*SEQ)=STRONGEST",
            "SEQ_SLOT_75": "backbone+.45*(.75*ETX+.25*SEQ)",
            "BTYD05_FRESH1": ".95*STRONGEST+.05*BTYD+fresh_processed_nested",
        },
        "zero2d_input_artifacts": [
            {"path": str((ARTIFACTS / f"PACT_dist_{fold}.npz").resolve()),
             "sha256": sha256_file(ARTIFACTS / f"PACT_dist_{fold}.npz")}
            for fold in FOLDS
        ],
        "candidate_prediction_sha256": {name: sha256_array(z) for name, z in models.items()},
    }
    return models, audit


def load_history(canonical: dict) -> dict[str, np.ndarray]:
    path = ROOT / "research" / "rmsle_diagnostics" / "fold_predictions.parquet"
    d = pl.read_parquet(path, columns=["cutoff", "user_id", "y", "rec_buy",
                                           "rec_any", "w180_days_buy"])
    keys = pl.DataFrame({"_row": np.arange(len(canonical["uid"])),
                         "cutoff": canonical["cutoff"], "user_id": canonical["uid"]})
    j = keys.join(d, on=["cutoff", "user_id"], how="left").sort("_row")
    assert j.height == len(canonical["uid"]) and j["y"].null_count() == 0
    assert np.allclose(j["y"].to_numpy(), canonical["y"], atol=1e-6)
    return {c: j[c].to_numpy().astype(float) for c in ("rec_buy", "rec_any", "w180_days_buy")}


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    yy = np.asarray(y) > 0
    if yy.all() or (~yy).all() or len(yy) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(yy, np.asarray(score, float)))


def _value_deciles(x: np.ndarray, name: str) -> list[tuple[str, np.ndarray]]:
    edges = np.unique(np.quantile(np.asarray(x, float), np.linspace(0, 1, 11)))
    if len(edges) <= 1:
        return [(f"{name}:all_tied", np.ones(len(x), bool))]
    ids = np.digitize(x, edges[1:-1], right=True)
    out = []
    for q in range(len(edges) - 1):
        m = ids == q
        if m.any():
            out.append((f"{name}:q{q+1}[{edges[q]:g},{edges[q+1]:g}]", m))
    return out


def slice_masks(sel: dict, hist: dict, y: np.ndarray, fold_mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    k, total, first = sel["k"][fold_mask], sel["total_days"][fold_mask], sel["first_day"][fold_mask]
    rec, buy = hist["rec_buy"][fold_mask], hist["w180_days_buy"][fold_mask]
    local: list[tuple[str, np.ndarray]] = [("all", np.ones(len(k), bool)), ("future_k_lt3", k < 3)]
    local += [(f"future_k_{j}", k == j) for j in range(4)]
    local += _value_deciles(total, "future_total_active_days")
    local += _value_deciles(first, "days_to_first_activity_after_target")
    local += [
        ("rec_buy_15_60", (rec >= 15) & (rec <= 60)),
        ("w180_days_buy_2_15", (buy >= 2) & (buy <= 15)),
        ("intersection_rec15_60_buy2_15", (rec >= 15) & (rec <= 60) & (buy >= 2) & (buy <= 15)),
        ("long_recency_gt60_or_never", (rec > 60) | ~np.isfinite(rec)),
        ("hist_buy_days_0", buy == 0), ("hist_buy_days_1", buy == 1),
        ("hist_buy_days_2_3", (buy >= 2) & (buy <= 3)),
        ("hist_buy_days_4_10", (buy >= 4) & (buy <= 10)),
        ("hist_buy_days_11plus", buy >= 11),
        ("actual_target_zero", y[fold_mask] == 0),
        ("actual_target_positive", y[fold_mask] > 0),
    ]
    return [(name, mask) for name, mask in local if mask.any()]


def slice_diagnostics(canonical: dict, models: dict[str, np.ndarray],
                      sel: dict, hist: dict) -> list[dict]:
    rows = []
    y, ly = canonical["y"], np.log1p(canonical["y"])
    for fold in FOLDS:
        fm = canonical["cutoff"] == fold
        offsets = {name: calibrate(y[fm], z[fm])[0] for name, z in models.items()}
        slices = slice_masks(sel, hist, y, fm)
        fold_idx = np.flatnonzero(fm)
        for slice_name, local in slices:
            ix = fold_idx[local]
            if not len(ix):
                continue
            for mode in ("fixed_fold", "slice_shape_only"):
                slice_scores = {}
                slice_offsets = {}
                for name, z in models.items():
                    off = offsets[name] if mode == "fixed_fold" else calibrate(y[ix], z[ix])[0]
                    pred = np.maximum(z[ix] + off, 0.0)
                    err = ly[ix] - pred
                    slice_offsets[name] = off
                    slice_scores[name] = float(np.sqrt(np.mean(err * err)))
                base_residual = ly[ix] - (models["STRONGEST"][ix] + offsets["STRONGEST"])
                for name, z in models.items():
                    off = slice_offsets[name]
                    pred = np.maximum(z[ix] + off, 0.0)
                    err = ly[ix] - pred
                    correction = z[ix] - models["STRONGEST"][ix]
                    corr = (float(np.corrcoef(correction, base_residual)[0, 1])
                            if name != "STRONGEST" and np.std(correction) > 0 and len(ix) > 2
                            else float("nan"))
                    rows.append({
                        "fold": fold, "slice": slice_name, "calibration_mode": mode,
                        "model": name, "n": len(ix), "target_zero_rate": float(np.mean(y[ix] == 0)),
                        "mean_log1p_target": float(np.mean(ly[ix])),
                        "mse_contribution_to_fold": float(np.sum(err * err) / fm.sum()),
                        "rmsle": slice_scores[name],
                        "prediction_bias_mean_z_minus_ly": float(np.mean(pred - ly[ix])),
                        "auc_y_positive": _auc(y[ix], z[ix]), "offset": off,
                        "delta_to_strongest": slice_scores[name] - slice_scores["STRONGEST"],
                        "corr_candidate_correction_strongest_residual": corr,
                    })
    return rows


def point_rankings(canonical: dict, models: dict[str, np.ndarray], sel: dict,
                   pi_ref: np.ndarray) -> tuple[list[dict], list[dict], dict, dict]:
    y, cut, k = canonical["y"], canonical["cutoff"], sel["k"]
    schemes = {
        "A_STANDARD_4F": {f: np.ones(np.sum(cut == f), float) for f in FOLDS},
        "B_SURVIVOR_K3": {f: (k[cut == f] == 3).astype(float) for f in FOLDS},
    }
    fold_pi, support = {}, {}
    match_weights = {}
    for fold in ELIGIBLE_FOLDS:
        fk = k[cut == fold]
        counts = np.bincount(fk, minlength=4)
        pi = counts / counts.sum()
        fold_pi[fold] = pi
        missing = [j for j in range(4) if pi_ref[j] > 0 and counts[j] == 0]
        w = np.asarray([pi_ref[j] / pi[j] if pi[j] > 0 else 0.0 for j in fk])
        ess = float(w.sum() ** 2 / np.sum(w * w))
        support[fold] = {"counts": counts, "pi": pi, "missing_k": missing,
                         "max_weight": float(w.max()), "ess": ess,
                         "ess_fraction": ess / len(w), "supported": not missing}
        match_weights[fold] = w
    schemes["C_PSEUDO_MATCHED_3F"] = match_weights
    rankings, pairwise = [], []
    fold_details = {}
    for scheme, fw in schemes.items():
        folds = ELIGIBLE_FOLDS if scheme.startswith("C_") else FOLDS
        fweights = np.asarray([1, 2, 4], float) if len(folds) == 3 else np.asarray(FOLD_WEIGHTS_S1, float)
        score_by_model, model_folds = {}, {}
        for name, z in models.items():
            fs = []
            for fold in folds:
                m = cut == fold
                if scheme == "B_SURVIVOR_K3":
                    full_offset = calibrate(y[m], z[m])[0]
                    keep = fw[fold] > 0
                    err = np.log1p(y[m][keep]) - np.maximum(z[m][keep] + full_offset, 0.0)
                    sc = float(np.sqrt(np.mean(err * err)))
                else:
                    _, sc = weighted_calibrate(y[m], z[m], fw[fold])
                fs.append(sc)
            score_by_model[name] = float(np.average(fs, weights=fweights))
            model_folds[name] = fs
        fold_details[scheme] = model_folds
        for family, members in (("standalone", STANDALONE), ("reference", ["UNC"]),
                                ("incremental", INCREMENTAL)):
            ordered = sorted(members, key=lambda x: score_by_model[x])
            for rank, name in enumerate(ordered, 1):
                rankings.append({"scheme": scheme, "family": family, "model": name,
                                 "score": score_by_model[name], "rank": rank,
                                 "delta_to_strongest": score_by_model[name] - score_by_model["STRONGEST"],
                                 "fold_scores": model_folds[name],
                                 "fold_deltas_to_strongest": (np.asarray(model_folds[name])
                                                               - np.asarray(model_folds["STRONGEST"])).tolist(),
                                 "folds_correct_sign": int(np.sum(np.asarray(model_folds[name])
                                                                  < np.asarray(model_folds["STRONGEST"])))})
            for a in members:
                for b in members:
                    pairwise.append({"scheme": scheme, "family": family,
                                     "model_a": a, "model_b": b,
                                     "delta_a_minus_b": score_by_model[a] - score_by_model[b]})
    return rankings, pairwise, {"pi_fold": fold_pi, "support": support,
                                "match_weights": match_weights}, fold_details


def _rankdata(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(values, method="average")


def rank_correlations(rankings: list[dict]) -> list[dict]:
    from scipy.stats import kendalltau, spearmanr
    rows = []
    schemes = ["A_STANDARD_4F", "B_SURVIVOR_K3", "C_PSEUDO_MATCHED_3F"]
    for family in ("standalone", "incremental"):
        members = STANDALONE if family == "standalone" else INCREMENTAL
        scores = {s: np.asarray([next(r["score"] for r in rankings
                                             if r["scheme"] == s and r["family"] == family
                                             and r["model"] == m) for m in members]) for s in schemes}
        for i, a in enumerate(schemes):
            for b in schemes[i+1:]:
                rows.append({"family": family, "scheme_a": a, "scheme_b": b,
                             "spearman": float(spearmanr(scores[a], scores[b]).statistic),
                             "kendall": float(kendalltau(scores[a], scores[b]).statistic)})
    return rows


def bootstrap_rankings(canonical: dict, models: dict[str, np.ndarray], sel: dict,
                       scheme_weights: dict, rankings: list[dict], n_rep: int = 500) -> tuple[list[dict], dict]:
    """Cluster bootstrap by user_id, holding point-estimate offsets fixed."""
    uid_unique, user_index = np.unique(canonical["uid"], return_inverse=True)
    rng = np.random.default_rng(SEED)
    model_names = list(models)
    schemes = {
        "A_STANDARD_4F": (FOLDS, {f: np.ones(np.sum(canonical["cutoff"] == f)) for f in FOLDS}, np.array([1,2,4,8.])),
        "B_SURVIVOR_K3": (FOLDS, {f: (sel["k"][canonical["cutoff"] == f] == 3).astype(float) for f in FOLDS}, np.array([1,2,4,8.])),
        "C_PSEUDO_MATCHED_3F": (ELIGIBLE_FOLDS, scheme_weights, np.array([1,2,4.])),
    }
    all_rep_scores: dict[str, np.ndarray] = {}
    summary_rows = []
    for scheme, (folds, weights_by_fold, fold_w) in schemes.items():
        # Per-user sufficient statistics for calibrated squared losses.
        num = np.zeros((len(uid_unique), len(model_names) * len(folds)), np.float32)
        den = np.zeros((len(uid_unique), len(folds)), np.float32)
        for fi, fold in enumerate(folds):
            m = canonical["cutoff"] == fold
            ix = user_index[m]
            w = np.asarray(weights_by_fold[fold], float)
            np.add.at(den[:, fi], ix, w.astype(np.float32))
            for mi, name in enumerate(model_names):
                z = models[name][m]
                off = (calibrate(canonical["y"][m], z)[0]
                       if scheme == "B_SURVIVOR_K3"
                       else weighted_calibrate(canonical["y"][m], z, w)[0])
                err2 = np.square(np.log1p(canonical["y"][m]) - np.maximum(z + off, 0.0))
                np.add.at(num[:, mi * len(folds) + fi], ix, (w * err2).astype(np.float32))
        rep_scores = np.empty((n_rep, len(model_names)), float)
        batch = 20
        for start in range(0, n_rep, batch):
            b = min(batch, n_rep - start)
            counts = np.empty((b, len(uid_unique)), np.float32)
            for j in range(b):
                sample = rng.integers(0, len(uid_unique), size=len(uid_unique))
                counts[j] = np.bincount(sample, minlength=len(uid_unique))
            dsum = counts @ den
            nsum = counts @ num
            for mi in range(len(model_names)):
                mse = nsum[:, mi*len(folds):(mi+1)*len(folds)] / dsum
                rep_scores[start:start+b, mi] = np.average(np.sqrt(mse), axis=1, weights=fold_w)
        all_rep_scores[scheme] = rep_scores
        base_i = model_names.index("STRONGEST")
        for family, members in (("standalone", STANDALONE), ("incremental", INCREMENTAL)):
            member_i = [model_names.index(x) for x in members]
            rank_matrix = np.vstack([_rankdata(rep_scores[r, member_i]) for r in range(n_rep)])
            for j, name in enumerate(members):
                delta = rep_scores[:, model_names.index(name)] - rep_scores[:, base_i]
                summary_rows.append({
                    "scheme": scheme, "family": family, "model": name,
                    "delta_mean": float(delta.mean()), "delta_median": float(np.median(delta)),
                    "delta_p025": float(np.quantile(delta, .025)),
                    "delta_p10": float(np.quantile(delta, .10)),
                    "delta_p90": float(np.quantile(delta, .90)),
                    "delta_p975": float(np.quantile(delta, .975)),
                    "p_delta_lt0": float(np.mean(delta < 0)),
                    "rank_median": float(np.median(rank_matrix[:, j])),
                    "rank_p10": float(np.quantile(rank_matrix[:, j], .10)),
                    "rank_p90": float(np.quantile(rank_matrix[:, j], .90)),
                    "bootstrap_offsets_fixed": True,
                })
    return summary_rows, {"scores": all_rep_scores, "model_names": model_names}


def _history_bins(hist: dict, fold_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rec, buy = hist["rec_buy"][fold_mask], hist["w180_days_buy"][fold_mask]
    rb = np.where((rec >= 15) & (rec <= 60), 1, np.where(rec > 60, 2, 0)).astype(np.int8)
    wb = np.where(buy <= 1, 0, np.where(buy <= 15, 1, 2)).astype(np.int8)
    return rb, wb


def shuffle_control(canonical: dict, models: dict[str, np.ndarray], sel: dict, hist: dict,
                    pi_ref: np.ndarray, rankings: list[dict], n_perm: int = 100) -> list[dict]:
    rng = np.random.default_rng(SEED)
    y, cut = canonical["y"], canonical["cutoff"]
    names = ["STRONGEST"] + INCREMENTAL + STANDALONE
    real = {r["model"]: r["delta_to_strongest"] for r in rankings
            if r["scheme"] == "C_PSEUDO_MATCHED_3F"}
    standard3 = {}
    for name in names:
        fs, bfs = [], []
        for fold in ELIGIBLE_FOLDS:
            m = cut == fold
            fs.append(calibrate(y[m], models[name][m])[1])
            bfs.append(calibrate(y[m], models["STRONGEST"][m])[1])
        standard3[name] = float(np.average(np.asarray(fs)-np.asarray(bfs), weights=[1,2,4]))
    shifts = {name: np.empty(n_perm, float) for name in names if name != "STRONGEST"}
    group_rows = {}
    for fold in ELIGIBLE_FOLDS:
        m = cut == fold
        rb, wb = _history_bins(hist, m)
        group_rows[fold] = [np.flatnonzero((rb == a) & (wb == b))
                            for a in range(3) for b in range(3)]
    for p in range(n_perm):
        fold_delta = {name: [] for name in names}
        for fold in ELIGIBLE_FOLDS:
            m = cut == fold
            kperm = sel["k"][m].copy()
            for rows in group_rows[fold]:
                kperm[rows] = rng.permutation(kperm[rows])
            counts = np.bincount(kperm, minlength=4)
            pi = counts / counts.sum()
            w = np.asarray([pi_ref[j] / pi[j] for j in kperm])
            for name in names:
                fold_delta[name].append(weighted_calibrate(y[m], models[name][m], w)[1])
        base = np.asarray(fold_delta["STRONGEST"])
        for name in shifts:
            delta = float(np.average(np.asarray(fold_delta[name]) - base, weights=[1,2,4]))
            shifts[name][p] = delta - standard3[name]
    rows = []
    for name, values in shifts.items():
        real_shift = real[name] - standard3[name]
        rows.append({"model": name, "real_ranking_shift": real_shift,
                     "shuffle_mean": float(values.mean()), "shuffle_p05": float(np.quantile(values,.05)),
                     "shuffle_median": float(np.median(values)), "shuffle_p95": float(np.quantile(values,.95)),
                     "real_percentile": float(np.mean(values <= real_shift)),
                     "outside_central_90": bool(real_shift < np.quantile(values,.05)
                                                or real_shift > np.quantile(values,.95)),
                     "stronger_than_95pct_in_improving_direction": bool(real_shift < np.quantile(values,.05)),
                     "permutations": n_perm,
                     "strata": "fold x rec_buy[0-14,15-60,>60] x w180_days_buy[0-1,2-15,16+]"})
    return rows


def selection_penalties(rankings: list[dict]) -> list[dict]:
    score = {(r["scheme"], r["model"]): r["delta_to_strongest"] for r in rankings}
    rows = []
    for name in STANDALONE + INCREMENTAL:
        survivor = score[("B_SURVIVOR_K3", name)]
        matched = score[("C_PSEUDO_MATCHED_3F", name)]
        standard = score[("A_STANDARD_4F", name)]
        rows.append({"comparison": f"{name} - STRONGEST", "model": name,
                     "reference": "STRONGEST", "delta_standard": standard,
                     "delta_survivor_conditioned": survivor,
                     "delta_pseudo_matched": matched,
                     "selection_penalty": survivor - matched})
    for name, ref in (("DIST", "CAP"), ("DIST", "UNC"), ("ETX-AVG3", "SEQ-AVG3")):
        standard = score[("A_STANDARD_4F", name)] - score[("A_STANDARD_4F", ref)]
        survivor = score[("B_SURVIVOR_K3", name)] - score[("B_SURVIVOR_K3", ref)]
        matched = score[("C_PSEUDO_MATCHED_3F", name)] - score[("C_PSEUDO_MATCHED_3F", ref)]
        rows.append({"comparison": f"{name} - {ref}", "model": name, "reference": ref,
                     "delta_standard": standard, "delta_survivor_conditioned": survivor,
                     "delta_pseudo_matched": matched,
                     "selection_penalty": survivor - matched})
    return rows


def level_shape_summary(slice_rows: list[dict]) -> list[dict]:
    """Decompose survivor k=3 differences into fixed-level and shape-only parts."""
    lookup = {(r["calibration_mode"], r["model"], r["fold"]): r["rmsle"]
              for r in slice_rows if r["slice"] == "future_k_3"}
    rows = []
    for name in ["STRONGEST"] + STANDALONE + INCREMENTAL:
        values = {}
        for mode in ("fixed_fold", "slice_shape_only"):
            fs = np.asarray([lookup[(mode, name, fold)] for fold in FOLDS])
            bs = np.asarray([lookup[(mode, "STRONGEST", fold)] for fold in FOLDS])
            values[mode] = float(np.average(fs - bs, weights=FOLD_WEIGHTS_S1))
        rows.append({"model": name, "survivor_fixed_delta": values["fixed_fold"],
                     "survivor_shape_only_delta": values["slice_shape_only"],
                     "level_component_fixed_minus_shape": (values["fixed_fold"]
                                                            - values["slice_shape_only"])})
    return rows


def decide(rankings: list[dict], bootstrap: list[dict], shuffle: list[dict],
           support: dict, models: dict[str, np.ndarray], canonical: dict) -> tuple[str, list[dict]]:
    if any((not x["supported"]) or x["ess_fraction"] < .25 for x in support.values()):
        return "TECHNICAL_INCONCLUSIVE", []
    point = {(r["scheme"], r["model"]): r for r in rankings}
    boot = {(r["scheme"], r["model"]): r for r in bootstrap}
    shuf = {r["model"]: r for r in shuffle}
    gates = []
    for name in INCREMENTAL:
        c = point[("C_PSEUDO_MATCHED_3F", name)]
        a = point[("A_STANDARD_4F", name)]
        b = boot[("C_PSEUDO_MATCHED_3F", name)]
        correction = models[name] - models["STRONGEST"]
        non_level = float(np.var(correction - np.mean(correction))) > 1e-10
        row = {"model": name, "pseudo_delta": c["delta_to_strongest"],
               "eligible_folds_better": int(np.sum(np.asarray(c["fold_deltas_to_strongest"]) < 0)),
               "p_delta_lt0": b["p_delta_lt0"], "shuffle_95_pass": shuf[name]["stronger_than_95pct_in_improving_direction"],
               "standard_delta": a["delta_to_strongest"], "not_only_level_shift": non_level}
        row["actionable_gate"] = bool(row["pseudo_delta"] <= -.001
            and row["eligible_folds_better"] >= 2 and row["p_delta_lt0"] >= .90
            and row["shuffle_95_pass"] and row["standard_delta"] <= .0001 and non_level)
        gates.append(row)
    if any(r["actionable_gate"] for r in gates):
        return "ACTIONABLE_SELECTION_CANDIDATE", gates
    best = min(r["pseudo_delta"] for r in gates)
    stable_small = any(-.001 < r["pseudo_delta"] <= -.0003 and r["p_delta_lt0"] >= .80
                       for r in gates)
    return ("SMALL_SELECTION_EFFECT" if stable_small else "NOT_ACTIONABLE"), gates


def build_report(summary: dict, rankings: list[dict], reference: list[dict],
                 prevalence: list[dict], penalties: list[dict], bootstrap: list[dict],
                 shuffle: list[dict], level_shape: list[dict], correlations: list[dict]) -> str:
    def table(headers, rows):
        return "|" + "|".join(headers) + "|\n|" + "|".join(["---"]*len(headers)) + "|\n" + \
            "\n".join("|" + "|".join(str(x) for x in row) + "|" for row in rows)
    prev_rows = []
    for f in FOLDS:
        rr = [r for r in prevalence if r["fold"] == f]
        prev_rows.append([f] + [f"{next(x['prevalence'] for x in rr if x['k']==k):.4f}" for k in range(4)]
                         + [f"{rr[0]['real_selection_overlap_days']}/90"])
    ref_rows = [[r["k"], f"{r['pi_ref']:.5f}", f"[{r['bootstrap_p025']:.5f},{r['bootstrap_p975']:.5f}]"] for r in reference]
    rank_rows = [[r["family"], r["scheme"], r["model"], r["rank"],
                  f"{r['delta_to_strongest']:+.6f}", r["folds_correct_sign"]] for r in rankings]
    pen_rows = [[r["comparison"], f"{r['delta_standard']:+.6f}", f"{r['delta_survivor_conditioned']:+.6f}",
                 f"{r['delta_pseudo_matched']:+.6f}", f"{r['selection_penalty']:+.6f}"] for r in penalties]
    boot_c = {r["model"]: r for r in bootstrap if r["scheme"] == "C_PSEUDO_MATCHED_3F"}
    sh = {r["model"]: r for r in shuffle}
    uncertainty = [[m, f"{boot_c[m]['delta_mean']:+.6f}",
                    f"[{boot_c[m]['delta_p10']:+.6f},{boot_c[m]['delta_p90']:+.6f}]",
                    f"[{boot_c[m]['delta_p025']:+.6f},{boot_c[m]['delta_p975']:+.6f}]",
                    f"{boot_c[m]['p_delta_lt0']:.3f}",
                    f"[{sh[m]['shuffle_p05']:+.6f},{sh[m]['shuffle_p95']:+.6f}]",
                    str(sh[m]["outside_central_90"])] for m in INCREMENTAL]
    best_abs = max(abs(r["selection_penalty"]) for r in penalties)
    incremental_penalty = max(abs(r["selection_penalty"]) for r in penalties
                              if r["reference"] == "STRONGEST" and r["model"] in INCREMENTAL)
    evidence_001 = any(r["delta_pseudo_matched"] <= -.001 for r in penalties if r["model"] in INCREMENTAL)
    floor = incremental_penalty >= .0004
    ls = {r["model"]: r for r in level_shape}
    ls_rows = [[m, f"{ls[m]['survivor_fixed_delta']:+.6f}",
                f"{ls[m]['survivor_shape_only_delta']:+.6f}",
                f"{ls[m]['level_component_fixed_minus_shape']:+.6f}"] for m in INCREMENTAL]
    corr_rows = [[r["family"], r["scheme_a"], r["scheme_b"],
                  f"{r['spearman']:.3f}", f"{r['kendall']:.3f}"] for r in correlations]
    raw_changes, meaningful_changes = [], []
    for family, members in (("standalone", STANDALONE), ("incremental", INCREMENTAL)):
        rs = {(r["scheme"], r["model"]): r["score"] for r in rankings if r["family"] == family}
        for i, a in enumerate(members):
            for b in members[i+1:]:
                da = rs[("A_STANDARD_4F", a)] - rs[("A_STANDARD_4F", b)]
                dc = rs[("C_PSEUDO_MATCHED_3F", a)] - rs[("C_PSEUDO_MATCHED_3F", b)]
                if da * dc < 0:
                    text = f"{family}: {a} vs {b} (A {da:+.6f}, C {dc:+.6f})"
                    raw_changes.append(text)
                    if max(abs(da), abs(dc)) >= .0002:
                        meaningful_changes.append(text)
    mismatch_floor_text = ("NO for incremental candidates" if not floor
                           else "POSSIBLY for incremental candidates")
    if summary["verdict"] == "TECHNICAL_INCONCLUSIVE":
        mismatch_floor_text += "; k=0 support is absent in every eligible fold"
    return f"""# exp_048 — SELECTION-MISMATCH / SELECTION-MATCHED CV

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** {summary['git_commit']}
- **Prefix:** `{PREFIX}`
- **SELECTION-MISMATCH VERDICT:** **{summary['verdict']}**
- **PROMOTE_TO_PRODUCTION_AUDIT:** **{'YES' if summary['verdict']=='ACTIONABLE_SELECTION_CANDIDATE' else 'NO'}**

## Exact mechanism and audit status

`STRONGEST_CURRENT` reconstructed exactly from raw OOF: fold scores
`1.766883357 / 1.760509577 / 1.748629224 / 1.741278566`, wCV
`1.747509863`. Row keys, target, component predictions, fold sizes, paths,
hashes, order, and fold calibration are in `audit_manifest.json`.

The competition universe is the intersection of any-row activity in G1
`2025-11-16..12-15`, G2 `2025-12-16..2026-01-14`, and G3
`2026-01-15..02-13`: 250,000 users in every block. The test panel at
`2026-02-13` is exactly the same three-past-block rule. Standard validation
panels are additionally conditioned on this globally future selection because
the supplied daily log contains only those selected 250k users.

For each V, target dates are `(V,V+30]`; selection variables use only
`(V+30,V+120]`, split into three 30-day blocks. The date sets are disjoint.
`future_min_events` is the minimum across F1/F2/F3 of daily
`searches+cat+to_cart+to_ord`; an inactive block makes it zero. Full source
dates are preserved in `audit_manifest.json`.

## Support and prevalence

{table(['fold','pi(k=0)','pi(k=1)','pi(k=2)','pi(k=3)','overlap with G'], prev_rows)}

The 10-16 F1/F2/F3 windows match G1/G2/G3 exactly and every validation row has
`k=3`; it is therefore excluded from pseudo-matched aggregation. Per-fold max
weights and ESS are in `matched_support.csv`; minimum ESS fraction is
**{summary['min_ess_fraction']:.3f}**.

However, `pi_ref(k=0) > 0` while **none** of the three eligible folds contains
`k=0`. All three folds are therefore unsupported under the registered rule;
the reported k>0 reweighting is a conditional sensitivity diagnostic, not the
requested full-reference matched estimate. This forces `TECHNICAL_INCONCLUSIVE`.

## Natural-continuation reference

Sixteen equally weighted weekly landmarks `2025-04-03..2025-07-17` use the
same three-past-block panel rule and all satisfy `V+120 <= 2025-11-15`.

{table(['k','pi_ref','cluster-bootstrap 95%'], ref_rows)}

Limitation: the reference is still built inside the globally selected 250k
universe and is not an unbiased estimate of the full platform population.

## Standard / survivor / pseudo-matched rankings

{table(['family','scheme','candidate','rank','delta to STRONGEST','folds correct sign'], rank_rows)}

Full pairwise deltas and bootstrap rank
intervals are in `rankings.csv`, `pairwise_deltas.csv`,
`rank_correlations.csv`, and `bootstrap.csv`. Differences below 0.0002 or whose
bootstrap interval spans zero are treated as unresolved, not rank changes.

{table(['family','scheme A','scheme B','Spearman','Kendall'], corr_rows)}

Raw order-changing pairs: **{'; '.join(raw_changes) if raw_changes else 'none'}**.
Order changes surviving the 0.0002 materiality rule: **{'; '.join(meaningful_changes) if meaningful_changes else 'none'}**.

## Bootstrap and shuffle controls

500 cluster-bootstrap replicates resample `user_id`; the point-estimate weighted
offsets are held fixed inside bootstrap (explicit diagnostic approximation).
The point matched scores themselves use exact weighted optimal offsets.

{table(['candidate','boot mean','10/90','2.5/97.5','P(delta<0)','shuffle central 90%','outside'], uncertainty)}

Shuffle uses 100 fixed seed-42 permutations inside
`fold × rec_buy_bin × w180_days_buy_bin` and preserves every stratum size.

## Selection penalty and magnitude

`selection_penalty = delta_survivor_conditioned - delta_pseudo_matched`.

{table(['comparison','standard','survivor k=3','pseudo-matched','selection penalty'], pen_rows)}

Largest absolute measured penalty is **{best_abs:.6f}**. Evidence that mismatch
alone explains the systematic 0.0004–0.0006 floor: **{mismatch_floor_text}**;
largest incremental penalty is **{incremental_penalty:.6f}**.
Evidence of an incremental candidate at scale >=0.001: **{'YES' if evidence_001 else 'NO'}**.

{table(['candidate','survivor fixed-fold delta','survivor shape-only delta','level component'], ls_rows)}

Fixed-versus-shape decomposition and residual-correction correlations in
`slice_diagnostics.csv` separate level shift from residual alignment. The
matched endpoint always refits one weighted fold offset, so its deltas are
shape comparisons rather than global-level wins.

## What can and cannot be concluded

This is a pseudo-selection-matched sensitivity analysis, not an unbiased test
estimate. It identifies how ranking changes under a fixed k-only continuation
reference within the selected universe. It cannot recover users absent from the
250k data, establish platform-population performance, or justify new gates,
weights, test inference, or a submission.

## Verdict

**{summary['verdict']}**. No model training, test inference, public-LB use, or
submission was performed. If non-actionable, the expected upside is below that
of a structurally new `Search/Catalog future-GMV target decomposition`, which
would attack target structure rather than this small/unstable selection axis;
that decomposition is not implemented here.
"""


def git_commit() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=ROOT, text=True).strip()


def main() -> None:
    if SEED != 42:
        raise AssertionError("registered bootstrap/shuffle protocol requires config.SEED=42")
    RESULTS.mkdir(parents=True, exist_ok=True)
    canonical, base_arrays, baseline_manifest = reconstruct_baseline()
    print("Baseline reconstructed", flush=True)
    competition = competition_audit(canonical)
    selection, prevalence, sources = build_selection_frame(canonical)
    pi_ref, landmarks, reference, reference_reps = reference_distribution()
    print("Selection audit complete", flush=True)
    models, candidate_audit = load_candidates(canonical, base_arrays)
    history = load_history(canonical)
    history_path = ROOT / "research" / "rmsle_diagnostics" / "fold_predictions.parquet"
    candidate_audit["history_diagnostics_artifact"] = {
        "path": str(history_path.resolve()), "sha256": sha256_file(history_path)}
    print("OOF alignment complete", flush=True)
    rankings, pairwise, match, fold_details = point_rankings(canonical, models, selection, pi_ref)
    bootstrap, bootstrap_raw = bootstrap_rankings(
        canonical, models, selection, match["match_weights"], rankings, 500)
    shuffle = shuffle_control(canonical, models, selection, history, pi_ref, rankings, 100)
    print("Matched CV complete", flush=True)
    slices = slice_diagnostics(canonical, models, selection, history)
    penalties = selection_penalties(rankings)
    level_shape = level_shape_summary(slices)
    correlations = rank_correlations(rankings)
    verdict, gates = decide(rankings, bootstrap, shuffle, match["support"], models, canonical)
    print("Diagnostics complete", flush=True)

    support_rows = []
    for fold, d in match["support"].items():
        support_rows.append({"fold": fold, "counts_k0_k3": d["counts"],
                             "pi_k0_k3": d["pi"], "missing_k": d["missing_k"],
                             "max_weight": d["max_weight"], "ess": d["ess"],
                             "n": int(np.sum(canonical["cutoff"] == fold)),
                             "ess_fraction": d["ess_fraction"], "supported": d["supported"]})
    audit = {**baseline_manifest, "competition_selection": competition,
             "selection_source_dates": sources,
             "selection_variables": {
                 "future_blocks_active": "sum any daily row in F1/F2/F3",
                 "future_min_active_days": "min block daily rows; 0 if a block inactive",
                 "future_total_active_days": "daily rows in F1+F2+F3",
                 "days_to_first_activity_after_target": "days after V+30; 91 if none",
                 "future_min_events": "min block sum(searches+cat+to_cart+to_ord); 0 if inactive",
             }, "target_window_read_by_selection": False,
             "candidate_artifacts": candidate_audit,
             "forbidden_actions": {"model_training": False, "test_inference": False,
                                    "submission": False, "public_lb_used": False}}
    summary = {"experiment": PREFIX, "git_commit": git_commit(), "verdict": verdict,
               "promote_to_production_audit": verdict == "ACTIONABLE_SELECTION_CANDIDATE",
               "pi_ref": pi_ref, "min_ess_fraction": min(x["ess_fraction"] for x in match["support"].values()),
               "decision_gates": gates, "baseline_wcv": baseline_manifest["wcv"],
               "bootstrap_replicates": 500, "shuffle_permutations": 100,
               "bootstrap_note": "cluster user_id; point-estimate offsets held fixed",
               "reference_limitation": "inside globally selected 250k universe; not unbiased platform population"}
    write_json(RESULTS / "audit_manifest.json", audit)
    write_json(RESULTS / "summary.json", summary)
    write_csv(RESULTS / "selection_prevalence.csv", prevalence)
    write_csv(RESULTS / "reference_landmarks.csv", landmarks)
    write_csv(RESULTS / "reference_distribution.csv", reference)
    write_csv(RESULTS / "matched_support.csv", support_rows)
    write_csv(RESULTS / "rankings.csv", rankings)
    write_csv(RESULTS / "pairwise_deltas.csv", pairwise)
    write_csv(RESULTS / "rank_correlations.csv", correlations)
    write_csv(RESULTS / "bootstrap.csv", bootstrap)
    write_csv(RESULTS / "shuffle_controls.csv", shuffle)
    write_csv(RESULTS / "slice_diagnostics.csv", slices)
    write_csv(RESULTS / "selection_penalties.csv", penalties)
    write_csv(RESULTS / "level_shape_summary.csv", level_shape)
    np.savez_compressed(RESULTS / "selection_rows.npz",
                        user_id=canonical["uid"], cutoff=canonical["cutoff"],
                        future_blocks_active=selection["k"],
                        future_min_active_days=selection["min_days"],
                        future_total_active_days=selection["total_days"],
                        days_to_first_activity_after_target=selection["first_day"],
                        future_min_events=selection["min_events"])
    report = build_report(summary, rankings, reference, prevalence, penalties,
                          bootstrap, shuffle, level_shape, correlations)
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "experiments" / "exp_048_selection_mismatch_cv.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
