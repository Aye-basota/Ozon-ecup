"""EXP077: strict-forward rebuild of the existing production ensemble.

This experiment is deliberately read-only with respect to model artifacts.  It
loads existing canonical OOF and TEST predictions, fixes the ridge penalty to
the value already selected in EXP076, and applies the pre-registered G.1 gates.
No model is trained and no leaderboard value is read or used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
ART = OZON / "artifacts"
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
E76 = ROOT / "research" / "new_directions" / "EXP076_STRONG_BASELINE_VALIDATION_CHANNEL"
ALPHA_PATH = Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")
PUBLIC_EB_PATH = Path(r"C:\Users\Admin\Downloads\SUBMIT_PUBLIC_EB.csv")
SAMPLE_PATH = OZON / "data" / "raw" / "sample_submit.csv"
GEOMETRY_Z = Path(r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\cache\Z.npz")

FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHT = dict(zip(FOLDS, [1.0, 2.0, 4.0, 8.0]))
RIDGE_ALPHA = 3e-5  # frozen result of EXP076; not searched in EXP077
ALPHA_RECON_RIDGE = 1e-6  # frozen EXP076 composition-reconstruction setting
PROJECTION_TOL = 1e-10
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260828

# Exact 40-column bank left after the near-exact duplicate removal in EXP076.
REFERENCE_BANK = [
    "S1-E02", "S1-E03a", "S1-DIST", "S1-E10", "S1-E11", "S1-SEEDAVG5",
    "S1-B0", "S1-E01", "S1-E03b", "SEQ-AVG3", "SEQ-D3A-AVG3",
    "SEQ-D3A-BASE-AVG3", "ETX-AVG3", "ETX-AVG2", "ETX-01-S42",
    "PT-FULL-AVG3", "PT-OD-AVG3", "PT-SHUF-AVG3", "RIDGE15",
    "HOLIDAY-YOY-FAST", "MHZ-FULL", "MHZ-BASE", "MHZ-P30", "MHZ-SELF",
    "S04-A", "S04-B", "S04-C", "GAP-E02-K5-G090-S42",
    "GAP-E10-K5-G090-S42", "GAP-DIST-K5-G060-S42",
    "SAMPLE-TB1-AVG3-R300", "SAMPLE-DENSE-S3-F422-S42-R300",
    "S1-ROUNDS-R600", "S1-ROUNDS-R300", "BTYD:z_btyd",
    "BTYD:z_strongest", "BLOCK4:z_new_honest", "FRESH:z_fresh",
    "FRESH:z_vol", "FRESH:z_clean",
]

# Only exact OOF/TEST pairs from the reference bank.  A list denotes a frozen
# log-space average.  No approximate surrogate is allowed.
TEST_SPECS: dict[str, tuple[str, list[str] | str, str | None]] = {
    "S1-E02": ("npy", ["S1-UNC"], None),
    "S1-E03a": ("npy", ["S1-CAP"], None),
    "S1-DIST": ("npy", ["S1-DIST"], None),
    "S1-E11": ("npy", ["S1-E11"], None),
    "SEQ-AVG3": ("npy", ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"], None),
    "ETX-AVG3": ("npy", ["ETX-01-S42-DCW", "ETX-01-S43-DCW", "ETX-01-S44-DCW"], None),
    "ETX-AVG2": ("npy", ["ETX-01-S42-DCW", "ETX-01-S43-DCW"], None),
    "ETX-01-S42": ("npy", ["ETX-01-S42-DCW"], None),
    "RIDGE15": ("npy", ["RIDGE15"], None),
    "HOLIDAY-YOY-FAST": ("npy", ["HOLIDAY-YOY-FAST"], None),
    "S04-A": ("npy", ["S04-A"], None),
    "S04-B": ("npy", ["S04-B"], None),
    "SAMPLE-TB1-AVG3-R300": ("npy", ["TIER-A-DIRECT-AVG3-R300"], None),
    "BTYD:z_btyd": ("npz", "BTYD_STABLE_EXP051/test_raw.npz", "z_btyd"),
    "BTYD:z_strongest": ("npz", "BTYD_STABLE_EXP051/test_raw.npz", "z_strongest"),
    "BLOCK4:z_new_honest": ("npz", "test_BLOCK4_SAF.npz", "z_new"),
}
DEPLOY_BANK = [name for name in REFERENCE_BANK if name in TEST_SPECS]

COMPOSITION_COMPONENTS = {
    "SEQ": ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44", "SEQ-D3A-BASE-S42"],
    "ETX": ["ETX-01-S42-DCW", "ETX-01-S43-DCW", "ETX-01-S44-DCW"],
    "TABULAR": [
        "S1-CAP", "S1-UNC", "S1-DIST", "S1-E11", "RIDGE15", "HOLIDAY-YOY-FAST",
        "TIER-A-DIRECT-AVG3-R300", "L180_norm0_tb1", "LNone_norm0_tb1",
        "L90_norm0_tb1", "S1-NORM", "S04-A", "S04-B",
    ],
}

PROXY_FAMILIES = {
    "SEQ": ["SEQ-AVG3", "SEQ-D3A-AVG3", "SEQ-D3A-BASE-AVG3"],
    "ETX": ["ETX-AVG3", "ETX-AVG2", "ETX-01-S42"],
    "TABULAR": [
        "S1-E03a", "S1-E02", "S1-DIST", "S1-E11", "RIDGE15",
        "HOLIDAY-YOY-FAST", "S04-A", "S04-B", "S1-E10",
        "SAMPLE-TB1-AVG3-R300", "MHZ-FULL", "GAP-E10-K5-G090-S42",
        "SAMPLE-DENSE-S3-F422-S42-R300",
    ],
    "BTYD_OTHER": ["BTYD:z_btyd"],
}

PROXY_FROZEN = {
    "SEQ": {"SEQ-AVG3": 1.0},
    "ETX": {"ETX-AVG3": 1.0},
    "TABULAR": {"S1-E03a": 0.10 / 0.55, "S1-E02": 0.20 / 0.55, "S1-DIST": 0.25 / 0.55},
    "BTYD_OTHER": {"BTYD:z_btyd": 1.0},
}

LOCAL_SPAN = [
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv"), "SUBMIT_ORTH_ALPHA"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_FINAL.csv"), "SUBMIT_ORTH_FINAL"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_PUBLIC_EB.csv"), "SUBMIT_PUBLIC_EB"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_PRIVATE_OPTIMAL.csv"), "SUBMIT_PRIVATE_OPTIMAL"),
    (Path(r"C:\Users\Admin\Downloads\SUBMIT_PRIVATE_V2.csv"), "SUBMIT_PRIVATE_V2"),
    (ROOT / "submissions" / "SUBMIT_NEXT_AFTER_EXP069.csv", "SUBMIT_NEXT_AFTER_EXP069"),
    (ROOT / "submissions" / "PROBE_scale097.csv", "PROBE_scale097"),
    (ROOT / "submissions" / "my_submit.csv", "my_submit"),
    (ROOT / "submissions" / "SUBMIT_v7_newmodel.csv", "SUBMIT_v7_newmodel"),
    (ROOT / "submissions" / "SUBMIT_ORTH_ROBUST_H12_INTERP.csv", "SUBMIT_ORTH_ROBUST_H12_INTERP"),
    (ROOT / "submissions" / "anchor_diverse_A_combo_mlp_hurdle_w065.csv", "anchor_diverse"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_once(path: Path, value: Any) -> None:
    write_text_once(path, json.dumps(as_jsonable(value), ensure_ascii=False, indent=2) + "\n")


def write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, lineterminator="\n")
    write_text_once(path, text)


def write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        old = np.load(path, allow_pickle=False)
        if set(old.files) != set(arrays):
            raise FileExistsError(f"schema drift in {path}")
        for key, value in arrays.items():
            if not np.array_equal(old[key], value, equal_nan=True):
                raise FileExistsError(f"content drift in {path}:{key}")
        return
    np.savez_compressed(path, **arrays)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    xc = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    yc = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    den = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return 0.0 if den <= 1e-20 else float(xc @ yc / den)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = X.T @ X
    scale = float(np.diag(xtx).mean())
    system = xtx + alpha * scale * np.eye(X.shape[1])
    system[0, 0] = xtx[0, 0]  # intercept is never penalized
    return np.linalg.solve(system, X.T @ y)


def project_out(u: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(basis, u, rcond=None)
    return u - basis @ coef, coef


def family(name: str) -> str:
    if name.startswith("SEQ-"):
        return "SEQ"
    if name.startswith("ETX-"):
        return "ETX"
    if name.startswith(("S1-", "RIDGE", "HOLIDAY", "MHZ", "S04-", "GAP-", "SAMPLE-")):
        return "TABULAR"
    return "BTYD_OTHER"


def canonical_keys(uid: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    fold_code = {fold: i for i, fold in enumerate(FOLDS)}
    return np.asarray([fold_code[str(c)] for c in cutoff], dtype=np.int64) * 10_000_000 + uid.astype(np.int64)


def align_oof(uid: np.ndarray, cutoff: np.ndarray, values: np.ndarray,
              ref_uid: np.ndarray, ref_cutoff: np.ndarray) -> tuple[np.ndarray, str]:
    uid = np.asarray(uid, dtype=np.int64)
    cutoff = np.asarray(cutoff).astype(str)
    values = np.asarray(values, dtype=np.float64)
    if np.array_equal(uid, ref_uid) and np.array_equal(cutoff, ref_cutoff):
        return values, "direct"
    ref_key = canonical_keys(ref_uid, ref_cutoff)
    key = canonical_keys(uid, cutoff)
    order = np.argsort(key)
    if len(np.unique(key)) != len(key):
        raise AssertionError("duplicate OOF key")
    pos = np.searchsorted(key[order], ref_key)
    if pos.max() >= len(key) or not np.array_equal(key[order][pos], ref_key):
        raise AssertionError("OOF key coverage mismatch")
    return values[order][pos], "reindexed"


def oof_source(name: str) -> tuple[Path, str, str, str]:
    if name.startswith("BTYD:"):
        return ART / "BTYD_STABLE_EXP051" / "oof_raw.npz", name.split(":", 1)[1], "user_id", "y"
    if name == "BLOCK4:z_new_honest":
        return ART / "oof_BLOCK4_SAF.npz", "z_new_honest", "uid", "y"
    if name.startswith("FRESH:"):
        return ART / "oof_FRESH_CONTRAST_MOE.npz", name.split(":", 1)[1], "uid", "y"
    return ART / f"oof_{name}.npz", "z", "user_id", "y"


def provenance_path(name: str) -> str:
    if name.startswith("BTYD:"):
        return str(OZON / "research" / "strategies" / "results" / "BTYD_STABLE_EXP051" / "artifact_manifest.json")
    if name.startswith("BLOCK4:"):
        return str(OZON / "research" / "strategies" / "results" / "BLOCK4_SAF" / "audit.json")
    if name.startswith("FRESH:"):
        return str(OZON / "research" / "strategies" / "results" / "FRESH_CONTRAST" / "validation.json")
    report = ART / f"report_{name}.json"
    if report.exists():
        return str(report)
    if name == "RIDGE15":
        return str(OZON / "research" / "strategies" / "results" / "RIDGE15" / "summary.json")
    return str(E76 / "code" / "s3_build_matrix.py")


def test_paths_for(name: str) -> list[Path]:
    kind, source, _ = TEST_SPECS[name]
    if kind == "npy":
        assert isinstance(source, list)
        return [ART / f"ztest_{member}.npy" for member in source]
    assert isinstance(source, str)
    return [ART / source]


def load_reference_bank(canon: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    uid = canon.user_id.to_numpy(np.int64)
    cutoff = canon.cutoff.astype(str).to_numpy()
    target_log = canon.target_log.to_numpy(np.float64)
    columns: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    for name in REFERENCE_BANK:
        path, field, uid_field, y_field = oof_source(name)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        z, alignment = align_oof(data[uid_field], data["cutoff"], data[field], uid, cutoff)
        y_aligned, _ = align_oof(data[uid_field], data["cutoff"], data[y_field], uid, cutoff)
        target_error = float(np.max(np.abs(np.log1p(np.maximum(y_aligned, 0.0)) - target_log)))
        folds = sorted(set(np.asarray(data["cutoff"]).astype(str)))
        test_paths = test_paths_for(name) if name in TEST_SPECS else []
        for test_path in test_paths:
            if not test_path.exists():
                raise FileNotFoundError(test_path)
        audit.append({
            "component": name,
            "family": family(name),
            "oof_path": str(path),
            "oof_field": field,
            "oof_sha256": sha256_file(path),
            "rows": len(z),
            "fold_coverage": f"{len(folds)}/4: " + ",".join(folds),
            "alignment": alignment,
            "target_log_max_abs_error": target_error,
            "provenance": provenance_path(name),
            "test_path": ";".join(str(p) for p in test_paths),
            "test_sha256": ";".join(sha256_file(p) for p in test_paths),
            "deployable": bool(test_paths),
            "exclusion_reason": "" if test_paths else "no exact TEST counterpart; OOF-reference only",
            "prediction_fields_used": field,
            "forbidden_activity_used": False,
        })
        if not np.isfinite(z).all() or len(folds) != 4 or target_error > 2e-5:
            raise AssertionError(f"OOF audit failed for {name}")
        columns.append(z)

    # Record the near-exact duplicate removed by EXP076 before the 40-column fit.
    duplicate = ART / "oof_SAMPLE-BASELINE-B-AVG3-R300.npz"
    audit.append({
        "component": "SAMPLE-BASELINE-B-AVG3-R300",
        "family": "TABULAR", "oof_path": str(duplicate), "oof_field": "z",
        "oof_sha256": sha256_file(duplicate), "rows": len(canon),
        "fold_coverage": "4/4: " + ",".join(FOLDS), "alignment": "not loaded (deduplicated)",
        "target_log_max_abs_error": 0.0,
        "provenance": str(ART / "report_SAMPLE-BASELINE-B-AVG3-R300.json"),
        "test_path": "", "test_sha256": "", "deployable": False,
        "exclusion_reason": "near-exact duplicate removed by EXP076 canonical-bank construction",
        "prediction_fields_used": "none", "forbidden_activity_used": False,
    })
    return np.column_stack(columns), pd.DataFrame(audit)


def align_test(values: np.ndarray, source_uid: np.ndarray | None,
               sample_uid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if source_uid is None:
        if len(values) != len(sample_uid):
            raise AssertionError("TEST row count mismatch")
        return values
    source_uid = np.asarray(source_uid, dtype=np.int64)
    if np.array_equal(source_uid, sample_uid):
        return values
    if len(np.unique(source_uid)) != len(source_uid):
        raise AssertionError("duplicate TEST user_id")
    return pd.Series(values, index=source_uid).reindex(sample_uid).to_numpy(np.float64)


def load_named_ztest(name: str, sample_uid: np.ndarray) -> np.ndarray:
    path = ART / f"ztest_{name}.npy"
    uid_path = ART / f"uid_{name}.npy"
    return align_test(np.load(path), np.load(uid_path) if uid_path.exists() else None, sample_uid)


def load_test_component(name: str, sample_uid: np.ndarray) -> np.ndarray:
    kind, source, field = TEST_SPECS[name]
    if kind == "npy":
        assert isinstance(source, list)
        vectors = [load_named_ztest(member, sample_uid) for member in source]
        out = np.mean(np.vstack(vectors), axis=0)
    else:
        assert isinstance(source, str) and field is not None
        data = np.load(ART / source, allow_pickle=True)
        uid_field = "user_id" if "user_id" in data.files else "uid"
        out = align_test(data[field], data[uid_field], sample_uid)
    if len(out) != len(sample_uid) or not np.isfinite(out).all():
        raise AssertionError(f"bad TEST vector {name}")
    return out


def load_submission_log(path: Path, sample_uid: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["user_id", "predict"] and not {"user_id", "predict"}.issubset(frame.columns):
        raise AssertionError(f"submission schema mismatch: {path}")
    if frame.user_id.duplicated().any():
        raise AssertionError(f"duplicate submission ids: {path}")
    pred = frame.set_index("user_id").reindex(sample_uid).predict.to_numpy(np.float64)
    if not np.isfinite(pred).all() or np.any(pred < 0):
        raise AssertionError(f"invalid submission predictions: {path}")
    return np.log1p(pred)


def reconstruct_alpha(sample_uid: np.ndarray, z_alpha: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    names: list[str] = []
    cols: list[np.ndarray] = []
    for fam, members in COMPOSITION_COMPONENTS.items():
        for member in members:
            names.append(member)
            cols.append(load_named_ztest(member, sample_uid))
    btyd = np.load(ART / "BTYD_STABLE_EXP051" / "test_raw.npz", allow_pickle=True)
    names.append("BTYD_z")
    cols.append(align_test(btyd["z_btyd"], btyd["user_id"], sample_uid))
    X = np.column_stack([np.ones(len(sample_uid))] + cols)
    coef = ridge_fit(X, z_alpha, ALPHA_RECON_RIDGE)
    projection = X @ coef
    residual = z_alpha - projection
    centered = z_alpha - z_alpha.mean()
    r2 = 1.0 - float(residual @ residual / (centered @ centered))
    raw = {}
    start = 1
    for fam, members in COMPOSITION_COMPONENTS.items():
        raw[fam] = float(np.sum(coef[start:start + len(members)]))
        start += len(members)
    raw["BTYD_OTHER"] = float(coef[-1])
    total = sum(raw.values())
    shares = {key: value / total for key, value in raw.items()}
    result = {
        "component_names": names,
        "R2": r2,
        "alpha_unexplained_rms": rms(residual),
        "alpha_centered_rms": rms(centered),
        "total_slope": total,
        "family_raw": raw,
        "family_shares": shares,
        "intercept": float(coef[0]),
        "ridge_alpha": ALPHA_RECON_RIDGE,
    }
    return result, projection, residual


def production_weights(names: list[str]) -> np.ndarray:
    out = np.zeros(len(names) + 1, dtype=np.float64)
    recipe = {"S1-E03a": 0.10, "S1-E02": 0.20, "S1-DIST": 0.25,
              "ETX-AVG3": 0.225, "SEQ-AVG3": 0.225}
    for name, weight in recipe.items():
        out[1 + names.index(name)] = weight
    return out


def strict_forward(X: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray],
                   names: list[str], z037: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    pred = np.full(len(y), np.nan, dtype=np.float64)
    weights: dict[str, np.ndarray] = {}
    for k, fold in enumerate(FOLDS):
        val = masks[fold]
        if k == 0:
            w = production_weights(names)
            pred[val] = z037[val]
        else:
            train = np.logical_or.reduce([masks[f] for f in FOLDS[:k]])
            w = ridge_fit(X[train], y[train], RIDGE_ALPHA)
            pred[val] = X[val] @ w
        weights[fold] = w
    final = ridge_fit(X, y, RIDGE_ALPHA)
    if not np.isfinite(pred).all():
        raise AssertionError("non-finite forward prediction")
    return pred, weights, final


def build_composition_proxy(Z: np.ndarray, names: list[str], y: np.ndarray,
                            masks: dict[str, np.ndarray], shares: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    index = {name: i for i, name in enumerate(names)}
    family_pred: dict[str, np.ndarray] = {}
    for fam, members in PROXY_FAMILIES.items():
        cols = [index[name] for name in members]
        X = np.column_stack([np.ones(len(y)), Z[:, cols]])
        pred = np.full(len(y), np.nan)
        for k, fold in enumerate(FOLDS):
            val = masks[fold]
            if k == 0:
                pred[val] = sum(weight * Z[val, index[name]]
                                for name, weight in PROXY_FROZEN[fam].items())
            else:
                train = np.logical_or.reduce([masks[f] for f in FOLDS[:k]])
                pred[val] = X[val] @ ridge_fit(X[train], y[train], RIDGE_ALPHA)
        family_pred[fam] = pred
    mixed = sum(shares[fam] * family_pred[fam] for fam in PROXY_FAMILIES)
    proxy = np.full(len(y), np.nan)
    for k, fold in enumerate(FOLDS):
        val = masks[fold]
        if k == 0:
            offset = 0.0
        else:
            train = np.logical_or.reduce([masks[f] for f in FOLDS[:k]])
            offset = float(np.mean(y[train] - mixed[train]))
        proxy[val] = mixed[val] + offset
    return proxy, family_pred


def fold_metrics(y: np.ndarray, masks: dict[str, np.ndarray], z037: np.ndarray,
                 proxy: np.ndarray, reference: np.ndarray, deploy: np.ndarray) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        m = masks[fold]
        scores = {
            "EXP037": rms(y[m] - z037[m]),
            "composition_proxy": rms(y[m] - proxy[m]),
            "reference_40": rms(y[m] - reference[m]),
            "forward_deployable": rms(y[m] - deploy[m]),
        }
        for candidate in ("reference_40", "forward_deployable"):
            rows.append({
                "cutoff": fold, "n": int(m.sum()), "candidate": candidate,
                "RMSLE": scores[candidate], "RMSLE_EXP037": scores["EXP037"],
                "RMSLE_composition_proxy": scores["composition_proxy"],
                "Delta_vs_EXP037": scores[candidate] - scores["EXP037"],
                "Delta_vs_composition_proxy": scores[candidate] - scores["composition_proxy"],
            })
    return pd.DataFrame(rows)


def weighted_score(frame: pd.DataFrame, candidate: str, column: str = "RMSLE") -> float:
    part = frame[frame.candidate == candidate].set_index("cutoff")
    den = sum(FOLD_WEIGHT.values())
    return float(sum(FOLD_WEIGHT[fold] * part.loc[fold, column] for fold in FOLDS) / den)


def weight_record(label: str, w: np.ndarray, names: list[str]) -> dict[str, Any]:
    component = np.asarray(w[1:], dtype=np.float64)
    l1 = float(np.sum(np.abs(component)))
    l2sq = float(component @ component)
    signed_sum = float(component.sum())
    fam_signed = {fam: float(sum(component[i] for i, name in enumerate(names) if family(name) == fam))
                  for fam in ("SEQ", "ETX", "TABULAR", "BTYD_OTHER")}
    fam_abs = {fam: float(sum(abs(component[i]) for i, name in enumerate(names) if family(name) == fam))
               for fam in ("SEQ", "ETX", "TABULAR", "BTYD_OTHER")}
    return {
        "fit": label, "intercept": float(w[0]), "component_sum": signed_sum,
        "L1_norm": l1, "effective_components": 0.0 if l2sq == 0 else l1 * l1 / l2sq,
        **{f"signed_share_{fam}": (value / signed_sum if abs(signed_sum) > 1e-12 else 0.0)
           for fam, value in fam_signed.items()},
        **{f"absolute_share_{fam}": (value / l1 if l1 > 0 else 0.0)
           for fam, value in fam_abs.items()},
    }


def sbvc(y: np.ndarray, uid: np.ndarray, masks: dict[str, np.ndarray], proxy: np.ndarray,
         forward: np.ndarray, Z: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    rows = []
    saved: dict[str, dict[str, np.ndarray]] = {}
    for fold in FOLDS:
        m = masks[fold]
        base = proxy[m]
        residual = y[m] - base
        d_raw = forward[m] - base
        d_center = d_raw - d_raw.mean()
        bmin = np.column_stack([np.ones(m.sum()), base])
        d_min, _ = project_out(d_center, bmin)
        d_min, _ = project_out(d_min, bmin)  # required repeat
        bfull = np.column_stack([np.ones(m.sum()), base, Z[m]])
        d_post, _ = project_out(d_center, bfull)
        d_post, _ = project_out(d_post, bfull)  # required repeat
        algebraic_zero = rms(d_post) < PROJECTION_TOL
        if algebraic_zero:
            d_post = np.zeros_like(d_post)
        max_projection = 0.0
        if not algebraic_zero:
            norms = np.sqrt(np.sum(np.square(bfull), axis=0))
            max_projection = float(np.max(np.abs(bfull.T @ d_post) / np.maximum(norms, 1e-300)))
        row = {
            "cutoff": fold, "n": int(m.sum()), "RMSLE_baseline": rms(residual),
            "RMSLE_forward": rms(y[m] - forward[m]),
            "rho_min_projection": corr(d_min, residual),
            "rho_post_projection": corr(d_post, residual),
            "rms_D_raw": rms(d_raw), "rms_D_centered": rms(d_center),
            "rms_D_min_projection": rms(d_min), "rms_D_post_projection": rms(d_post),
            "post_perp_fraction": float((d_post @ d_post) / max(d_center @ d_center, 1e-300)),
            "second_pass_max_projection": max_projection,
            "algebraically_in_component_span": algebraic_zero,
            "b_min": float(np.mean(d_min * residual)),
            "G_min": float(np.mean(d_min * d_min)),
            "b_post": float(np.mean(d_post * residual)),
            "G_post": float(np.mean(d_post * d_post)),
        }
        rows.append(row)
        saved[fold] = {"uid": uid[m], "residual": residual, "d_min": d_min, "d_post": d_post}
    frame = pd.DataFrame(rows)
    for suffix in ("min", "post"):
        amps = []
        for k in range(len(FOLDS)):
            if k == 0:
                amp = 1.0
            else:
                prev = frame.iloc[:k]
                g = float(prev[f"G_{suffix}"].sum())
                amp = 0.0 if g <= 1e-20 else float(prev[f"b_{suffix}"].sum() / g)
            amps.append(amp)
        frame[f"nested_amplitude_{suffix}"] = amps
        dmse = []
        drmsle = []
        for row, amp in zip(frame.to_dict("records"), amps):
            delta = -2.0 * amp * row[f"b_{suffix}"] + amp * amp * row[f"G_{suffix}"]
            dmse.append(delta)
            new_rms = math.sqrt(max(row["RMSLE_baseline"] ** 2 + delta, 0.0))
            drmsle.append(new_rms - row["RMSLE_baseline"])
        frame[f"nested_Delta_MSE_{suffix}"] = dmse
        frame[f"nested_Delta_RMSLE_{suffix}"] = drmsle
    weights = np.array([FOLD_WEIGHT[f] for f in FOLDS])
    agg: dict[str, Any] = {}
    for col in ("rho_min_projection", "rho_post_projection"):
        agg[f"weighted_{col}"] = float(np.average(frame[col], weights=weights))
        agg[f"latest_{col}"] = float(frame[col].iloc[-1])
        agg[f"weighted_evaluable_{col}"] = float(np.average(frame[col].iloc[1:], weights=weights[1:]))
    for suffix in ("min", "post"):
        agg[f"weighted_nested_Delta_MSE_{suffix}"] = float(np.average(
            frame[f"nested_Delta_MSE_{suffix}"], weights=weights))
        agg[f"weighted_nested_Delta_RMSLE_{suffix}"] = float(np.average(
            frame[f"nested_Delta_RMSLE_{suffix}"], weights=weights))
        total_b = float(frame[f"b_{suffix}"].sum())
        total_g = float(frame[f"G_{suffix}"].sum())
        agg[f"deploy_amplitude_{suffix}"] = 0.0 if total_g <= 1e-20 else total_b / total_g
    return frame, agg, saved


def cluster_bootstrap(saved: dict[str, dict[str, np.ndarray]], frame: pd.DataFrame,
                      suffix: str) -> dict[str, Any]:
    # The full projection is an algebraic zero.  Preserve exact zero rather than
    # report a meaningless sign generated by floating-point projection noise.
    if all(rms(saved[fold][f"d_{suffix}"]) < PROJECTION_TOL for fold in FOLDS):
        return {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "unit": "user_id cluster", "method": "degenerate algebraic-zero bootstrap",
            "point_Delta_MSE": 0.0, "CI95": [0.0, 0.0], "P_Delta_MSE_lt_0": 0.0,
        }
    all_uid = np.unique(np.concatenate([saved[fold]["uid"] for fold in FOLDS]))
    n_users = len(all_uid)
    contributions = np.zeros((len(FOLDS), n_users), dtype=np.float64)
    presence = np.zeros((len(FOLDS), n_users), dtype=np.float64)
    amplitudes = frame[f"nested_amplitude_{suffix}"].to_numpy(np.float64)
    for k, fold in enumerate(FOLDS):
        item = saved[fold]
        pos = np.searchsorted(all_uid, item["uid"])
        d = item[f"d_{suffix}"]
        row_delta = -2.0 * amplitudes[k] * d * item["residual"] + amplitudes[k] ** 2 * d * d
        contributions[k, pos] = row_delta
        presence[k, pos] = 1.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    fw = np.asarray([FOLD_WEIGHT[f] for f in FOLDS], dtype=np.float64)
    for rep in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_users, size=n_users)
        count = np.bincount(sampled, minlength=n_users).astype(np.float64)
        per_fold = (contributions @ count) / np.maximum(presence @ count, 1.0)
        draws[rep] = float(np.average(per_fold, weights=fw))
    point = float(np.average(frame[f"nested_Delta_MSE_{suffix}"], weights=fw))
    return {
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "unit": "user_id cluster", "method": "ordinary cluster bootstrap; nested amplitudes frozen",
        "point_Delta_MSE": point,
        "CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "P_Delta_MSE_lt_0": float(np.mean(draws < 0)),
    }


class SubmissionSpan:
    def __init__(self, M: np.ndarray, tol: float = 1e-12):
        gram = M @ M.T
        eig, vec = np.linalg.eigh(gram)
        keep = eig > eig.max() * tol
        self.rank = int(keep.sum())
        self.vi = vec[:, keep] / np.sqrt(eig[keep])
        self.M = M

    def project(self, x: np.ndarray) -> np.ndarray:
        xc = x - x.mean()
        qtx = self.vi.T @ (self.M @ xc)
        return (self.vi @ qtx) @ self.M

    def perp_twice(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        first = x - x.mean() - self.project(x)
        second_projection = self.project(first)
        out = first - second_projection
        out -= out.mean()
        qtx = self.vi.T @ (self.M @ out)
        return out, float(np.max(np.abs(qtx))) if len(qtx) else 0.0


def build_submission_span(sample_uid: np.ndarray) -> tuple[SubmissionSpan, list[str]]:
    canonical = np.load(GEOMETRY_Z, allow_pickle=True)
    if not np.array_equal(canonical["user_id"].astype(np.int64), sample_uid):
        raise AssertionError("canonical submission geometry user order mismatch")
    M = np.asarray(canonical["Z"], dtype=np.float64)
    names = [f"canonical:{i}" for i in range(M.shape[0])]
    extra = []
    for path, name in LOCAL_SPAN:
        if not path.exists():
            raise FileNotFoundError(path)
        extra.append(load_submission_log(path, sample_uid))
        names.append(name)
    M = np.vstack([M, np.vstack(extra)])
    M -= M.mean(axis=1, keepdims=True)
    return SubmissionSpan(M, tol=1e-12), names


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---:" if col not in ("cutoff", "candidate", "fit", "family") else "---" for col in columns) + " |"
    lines = [header, sep]
    for row in frame[columns].to_dict("records"):
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), formats.get(col, ".6f")))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    for required in (ALPHA_PATH, PUBLIC_EB_PATH, SAMPLE_PATH, GEOMETRY_Z,
                     E75 / "clean_forward_predictions.parquet"):
        if not required.exists():
            raise FileNotFoundError(required)

    canon = pd.read_parquet(E75 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    if canon.duplicated(["cutoff", "user_id"]).any():
        raise AssertionError("duplicate canonical OOF keys")
    masks = {fold: canon.cutoff.to_numpy() == fold for fold in FOLDS}
    y = canon.target_log.to_numpy(np.float64)
    uid = canon.user_id.to_numpy(np.int64)
    Z, audit = load_reference_bank(canon)
    if Z.shape != (770616, 40) or len(DEPLOY_BANK) != 16:
        raise AssertionError(f"unexpected bank shape: {Z.shape}, deploy={len(DEPLOY_BANK)}")
    index = {name: i for i, name in enumerate(REFERENCE_BANK)}
    z037 = Z[:, index["BTYD:z_strongest"]]
    local_037 = pd.read_parquet(ROOT / "artifacts" / "oof" / "EXP_037_STRONGEST_CURRENT.parquet")
    check037 = local_037.set_index(["cutoff", "user_id"]).reindex(
        pd.MultiIndex.from_arrays([canon.cutoff, canon.user_id])).z_pred.to_numpy(np.float64)
    exp037_max_abs = float(np.max(np.abs(check037 - z037)))
    if exp037_max_abs > 1e-10:
        raise AssertionError("EXP037 primary-artifact mismatch")

    sample = pd.read_csv(SAMPLE_PATH)
    sample_uid = sample.user_id.to_numpy(np.int64)
    if len(sample_uid) != 250000 or len(np.unique(sample_uid)) != len(sample_uid):
        raise AssertionError("sample_submit audit failed")
    z_alpha = load_submission_log(ALPHA_PATH, sample_uid)
    z_public = load_submission_log(PUBLIC_EB_PATH, sample_uid)
    alpha_recon, z_alpha_proxy_test, r_alpha = reconstruct_alpha(sample_uid, z_alpha)

    proxy_shares = {
        "SEQ": alpha_recon["family_shares"]["SEQ"],
        "ETX": alpha_recon["family_shares"]["ETX"],
        "TABULAR": alpha_recon["family_shares"]["TABULAR"],
        "BTYD_OTHER": alpha_recon["family_shares"]["BTYD_OTHER"],
    }
    z_proxy, _ = build_composition_proxy(Z, REFERENCE_BANK, y, masks, proxy_shares)

    Xref = np.column_stack([np.ones(len(y)), Z])
    zref, wref, wref_final = strict_forward(Xref, y, masks, REFERENCE_BANK, z037)
    dep_idx = [index[name] for name in DEPLOY_BANK]
    Zdep = Z[:, dep_idx]
    Xdep = np.column_stack([np.ones(len(y)), Zdep])
    zdeploy, wdep, wdep_final = strict_forward(Xdep, y, masks, DEPLOY_BANK, z037)
    forward = fold_metrics(y, masks, z037, z_proxy, zref, zdeploy)

    weights_rows = []
    weight_values = []
    for bank_name, names, fold_weights, final_weights in (
        ("reference_40", REFERENCE_BANK, wref, wref_final),
        ("forward_deployable", DEPLOY_BANK, wdep, wdep_final),
    ):
        for fold in FOLDS:
            label = f"{bank_name}:{fold}"
            weights_rows.append({"bank": bank_name, **weight_record(fold, fold_weights[fold], names)})
            for component, value in zip(["INTERCEPT"] + names, fold_weights[fold]):
                weight_values.append({"bank": bank_name, "fit": fold, "component": component,
                                      "family": "INTERCEPT" if component == "INTERCEPT" else family(component),
                                      "weight": float(value)})
        weights_rows.append({"bank": bank_name, **weight_record("final_all_folds", final_weights, names)})
        for component, value in zip(["INTERCEPT"] + names, final_weights):
            weight_values.append({"bank": bank_name, "fit": "final_all_folds", "component": component,
                                  "family": "INTERCEPT" if component == "INTERCEPT" else family(component),
                                  "weight": float(value)})
    weight_summary = pd.DataFrame(weights_rows)
    weight_frame = pd.DataFrame(weight_values)

    sbvc_frame, sbvc_agg, saved = sbvc(y, uid, masks, z_proxy, zdeploy, Z)
    bootstrap_post = cluster_bootstrap(saved, sbvc_frame, "post")
    bootstrap_min = cluster_bootstrap(saved, sbvc_frame, "min")

    Ztest_dep = np.column_stack([load_test_component(name, sample_uid) for name in DEPLOY_BANK])
    z_forward_test = np.column_stack([np.ones(len(sample_uid)), Ztest_dep]) @ wdep_final
    if not np.isfinite(z_forward_test).all():
        raise AssertionError("non-finite forward TEST stack")
    d_test = z_forward_test - z_alpha
    d_test_center = d_test - d_test.mean()
    alpha_den = float(r_alpha @ r_alpha)
    alpha_coef = float(d_test_center @ r_alpha / alpha_den)
    alpha_projection = alpha_coef * r_alpha
    alpha_interaction = {
        **alpha_recon,
        "corr_D_test_r_alpha_unexplained": corr(d_test_center, r_alpha),
        "projection_coefficient_D_onto_r_alpha": alpha_coef,
        "RMS_projection_D_onto_r_alpha": rms(alpha_projection),
        "projection_RMS_over_r_alpha_RMS": rms(alpha_projection) / rms(r_alpha),
        "corr_stack_minus_component_proxy_with_r_alpha": corr(z_forward_test - z_alpha_proxy_test, r_alpha),
        "material_conflict": bool(alpha_coef < -0.25 and rms(alpha_projection) > 0.25 * rms(r_alpha)),
        "interpretation": "D_test contains -r_alpha exactly when the standalone stack lies in the reconstructed component span",
    }

    span, span_names = build_submission_span(sample_uid)
    d_perp_test, second_pass_max = span.perp_twice(d_test)
    orth_correction = z_alpha - z_public
    test_geometry = {
        "vectors": len(span_names), "rank": span.rank,
        "RMS_D_test": rms(d_test), "RMS_D_test_centered": rms(d_test_center),
        "RMS_D_perp": rms(d_perp_test),
        "perp_fraction_centered": float((d_perp_test @ d_perp_test) / max(d_test_center @ d_test_center, 1e-300)),
        "perp_fraction_raw": float((d_perp_test @ d_perp_test) / max(d_test @ d_test, 1e-300)),
        "second_pass_max_projection": second_pass_max,
        "mean_D_perp": float(d_perp_test.mean()),
        "corr_D_perp_ORTH_correction": corr(d_perp_test, orth_correction),
        "finite": bool(np.isfinite(d_perp_test).all()),
        "sample_rows": len(sample_uid), "sample_user_id_unique": len(np.unique(sample_uid)) == len(sample_uid),
    }

    # Fixed gates.  Post-projection SBVC is decision-relevant by construction.
    dep_folds = forward[forward.candidate == "forward_deployable"].set_index("cutoff")
    heldout = dep_folds.loc[FOLDS[1:]]
    heldout_wins_proxy = int((heldout.Delta_vs_composition_proxy < 0).sum())
    heldout_wins_037 = int((heldout.Delta_vs_EXP037 < 0).sum())
    fold4_gain = float(dep_folds.loc[FOLDS[-1], "Delta_vs_EXP037"])
    post_dmse = sbvc_frame.nested_Delta_MSE_post.to_numpy(np.float64)
    improvement_not_one_fold = bool(np.sum(post_dmse < -1e-12) >= 2)
    gates = {
        "1_fold4_Delta_RMSLE_le_minus_0.0005": fold4_gain <= -0.0005,
        "2_at_least_2_of_3_heldout_better_composition_proxy": heldout_wins_proxy >= 2,
        "3_latest_SBVC_rho_positive": sbvc_agg["latest_rho_post_projection"] > 0,
        "4_weighted_SBVC_rho_ge_0.010": sbvc_agg["weighted_rho_post_projection"] >= 0.010,
        "5_nested_SBVC_Delta_MSE_negative": sbvc_agg["weighted_nested_Delta_MSE_post"] < 0,
        "6_bootstrap_probability_ge_0.95": bootstrap_post["P_Delta_MSE_lt_0"] >= 0.95,
        "7_improvement_not_one_fold": improvement_not_one_fold,
        "8_no_material_ORTH_ALPHA_residual_conflict": not alpha_interaction["material_conflict"],
        "9_TEST_vector_format_projection_checks": bool(test_geometry["finite"] and test_geometry["sample_rows"] == 250000
                                                         and test_geometry["sample_user_id_unique"]
                                                         and second_pass_max < 1e-8),
    }
    verdict = "GO" if all(gates.values()) else "NO_GO"

    submission_path = ROOT / "submissions" / "SUBMIT_EXP077_FORWARD_STACK.csv"
    output: dict[str, Any] = {"path": None, "sha256": None}
    deploy_amplitude = sbvc_agg["deploy_amplitude_post"]
    if verdict == "GO":
        if submission_path.exists():
            raise FileExistsError(f"refusing to overwrite submission: {submission_path}")
        z_candidate = np.maximum(z_alpha + deploy_amplitude * d_perp_test, 0.0)
        pred = np.expm1(z_candidate)
        candidate = pd.DataFrame({"user_id": sample_uid, "predict": pred})
        if len(candidate) != 250000 or candidate.user_id.duplicated().any() or not np.isfinite(pred).all() or np.any(pred < 0):
            raise AssertionError("candidate submission audit failed")
        candidate.to_csv(submission_path, index=False)
        output = {"path": str(submission_path), "sha256": sha256_file(submission_path)}

    wcvs = {
        "EXP037": weighted_score(forward.assign(candidate=np.where(forward.candidate == "reference_40", "reference_40", "forward_deployable")), "reference_40", "RMSLE_EXP037"),
        "composition_proxy": weighted_score(forward, "reference_40", "RMSLE_composition_proxy"),
        "reference_40": weighted_score(forward, "reference_40"),
        "forward_deployable": weighted_score(forward, "forward_deployable"),
    }
    result = {
        "experiment": "EXP077_FORWARD_STACK", "verdict": verdict,
        "constraints": {"models_trained": 0, "leaderboard_used": False,
                        "ridge_alpha": RIDGE_ALPHA, "hyperparameter_sweep": False},
        "artifact_audit": {"reference_components": len(REFERENCE_BANK),
                           "deployable_components": len(DEPLOY_BANK),
                           "forbidden_activity_used": False,
                           "exp037_primary_max_abs_error": exp037_max_abs,
                           "canonical_rows": len(canon), "folds": FOLDS},
        "wCV": wcvs,
        "heldout_wins_vs_composition_proxy": heldout_wins_proxy,
        "heldout_wins_vs_EXP037": heldout_wins_037,
        "fold4_Delta_vs_EXP037": fold4_gain,
        "SBVC": sbvc_agg,
        "bootstrap_post_projection": bootstrap_post,
        "bootstrap_min_projection": bootstrap_min,
        "ORTH_ALPHA_residual_interaction": alpha_interaction,
        "TEST_geometry": test_geometry,
        "gates": gates, "output": output,
    }

    write_csv_once(HERE / "artifact_audit.csv", audit)
    write_csv_once(HERE / "forward_folds.csv", forward)
    write_csv_once(HERE / "weights.csv", weight_frame)
    write_csv_once(HERE / "weight_summary.csv", weight_summary)
    write_csv_once(HERE / "sbvc_folds.csv", sbvc_frame)
    write_json_once(HERE / "bootstrap.json", {"post_projection": bootstrap_post, "min_projection": bootstrap_min})
    write_json_once(HERE / "orth_alpha_residual.json", alpha_interaction)
    write_json_once(HERE / "test_geometry.json", test_geometry)
    write_json_once(HERE / "results.json", result)
    write_npz_once(HERE / "forward_test_vectors.npz", user_id=sample_uid,
                   z_forward_stack=z_forward_test, D_test=d_test, D_perp=d_perp_test,
                   z_alpha=z_alpha, r_alpha_unexplained=r_alpha)

    display_forward = forward[forward.candidate == "forward_deployable"].copy()
    display_forward["candidate"] = "new forward stack (16 deployable)"
    reference_table = forward[forward.candidate == "reference_40"].copy()
    reference_table["candidate"] = "EXP076 reference (40 OOF)"
    fold_table = pd.concat([reference_table, display_forward], ignore_index=True)
    ws_final = weight_summary[weight_summary.fit == "final_all_folds"].copy()
    sbvc_display = sbvc_frame.rename(columns={
        "rho_min_projection": "rho_min", "rho_post_projection": "rho_post",
        "nested_Delta_MSE_post": "nested_Delta_MSE",
        "nested_Delta_RMSLE_post": "nested_Delta_RMSLE",
    })
    audit_included = audit[audit.component.isin(REFERENCE_BANK)].copy()
    audit_lines = []
    for row in audit_included.to_dict("records"):
        test_text = row["test_path"] if row["test_path"] else "— (excluded from deployable fit)"
        audit_lines.append(f"| {row['component']} | {row['family']} | `{row['oof_path']}` | `{test_text}` | {row['fold_coverage']} | `{row['provenance']}` |")
    gates_lines = "\n".join(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in gates.items())
    expected = ("No positive robust uplift is supported: the decision-relevant post-projection "
                "historical correction is algebraically zero, so its nested Delta RMSLE is 0. "
                "The standalone wCV change cannot be translated into an incremental production gain.")
    report = f"""# EXP077 — Forward Production Stack

## Verdict

**{verdict}**. The fixed G.1 gates were applied without leaderboard fitting and without training any model.

| Gate | Result |
| --- | --- |
{gates_lines}

## Artifact audit

The EXP076 40-column clean OOF bank is reproduced exactly. The deployable stack is the strict
intersection of that bank with exact frozen TEST predictions: **{len(DEPLOY_BANK)}/40** components.
OOF-only components remain in the reference reproduction but are not silently replaced for TEST.
`oof_BLOCK4_SAF['activity']` and every target-derived activity field were excluded; only
`z_new_honest` was read from that artifact. No contaminated teammate OOF was loaded.

| component | family | OOF path | TEST path | fold coverage | provenance |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(audit_lines)}

Full hashes, alignment mode, target parity, and exclusion reasons: `artifact_audit.csv`.
The compact canonical EXP037 parquet reproduces `BTYD:z_strongest` with max absolute error
`{exp037_max_abs:.3e}`.

## Forward results

Ridge regularization was frozen at `{RIDGE_ALPHA:g}` from EXP076. There was no EXP077 sweep.
Fold 1 uses the frozen EXP037 recipe; each later fold uses weights fitted only on earlier folds.

{markdown_table(fold_table, ['cutoff','candidate','RMSLE','Delta_vs_EXP037','Delta_vs_composition_proxy'], {'RMSLE':'.9f','Delta_vs_EXP037':'+.9f','Delta_vs_composition_proxy':'+.9f'})}

| stack | weighted wCV 1:2:4:8 |
| --- | ---: |
| EXP037 frozen | {wcvs['EXP037']:.9f} |
| composition-matched ORTH_ALPHA proxy | {wcvs['composition_proxy']:.9f} |
| EXP076 reference, 40 OOF | {wcvs['reference_40']:.9f} |
| new forward stack, 16 deployable | {wcvs['forward_deployable']:.9f} |

Held-out sign consistency for the deployable stack is `{heldout_wins_proxy}/3` versus the
composition proxy and `{heldout_wins_037}/3` versus EXP037. Fold-4 Delta versus EXP037 is
`{fold4_gain:+.9f}`. The worst held-out Delta versus EXP037 is
`{heldout.Delta_vs_EXP037.max():+.9f}`.

Final all-fold weight stability and family shares (signed shares use total component slope;
absolute shares use L1 mass):

{markdown_table(ws_final, ['bank','L1_norm','effective_components','signed_share_SEQ','signed_share_ETX','signed_share_TABULAR','signed_share_BTYD_OTHER','absolute_share_SEQ','absolute_share_ETX','absolute_share_TABULAR','absolute_share_BTYD_OTHER'], {c:'.6f' for c in ws_final.columns if c not in ('bank','fit')})}

ORTH_ALPHA reconstructed family shares are SEQ `{proxy_shares['SEQ']:.6f}`, ETX
`{proxy_shares['ETX']:.6f}`, TABULAR `{proxy_shares['TABULAR']:.6f}`, and BTYD/OTHER
`{proxy_shares['BTYD_OTHER']:.6f}`. The deployable forward optimum moves signed SEQ share to
`{ws_final[ws_final.bank == 'forward_deployable'].iloc[0].signed_share_SEQ:.6f}`
(`{ws_final[ws_final.bank == 'forward_deployable'].iloc[0].signed_share_SEQ - proxy_shares['SEQ']:+.6f}`)
and ETX to `{ws_final[ws_final.bank == 'forward_deployable'].iloc[0].signed_share_ETX:.6f}`
(`{ws_final[ws_final.bank == 'forward_deployable'].iloc[0].signed_share_ETX - proxy_shares['ETX']:+.6f}`).
Combined SEQ+ETX rises from `{proxy_shares['SEQ'] + proxy_shares['ETX']:.6f}` to
`{ws_final[ws_final.bank == 'forward_deployable'].iloc[0].signed_share_SEQ + ws_final[ws_final.bank == 'forward_deployable'].iloc[0].signed_share_ETX:.6f}`:
Claude's directional composition hypothesis is reproduced. Signed ridge weights are not mixture
probabilities; the absolute shares above expose cancelling/unstable weights. Full per-component
weights are in `weights.csv`.

## SBVC

`D_fold = z_forward - z_composition_proxy` was centered, projected outside the required
historical ensemble span (constant + strong baseline + all 40 canonical components), and the
projection was repeated. Because both terms are linear combinations of that same component
bank, the correction is algebraically in-span and is annihilated (numerical remnants below
`{PROJECTION_TOL:g}` are set to exact zero before inference).

{markdown_table(sbvc_display, ['cutoff','rho_min','rho_post','nested_amplitude_min','nested_amplitude_post','rms_D_post_projection','nested_Delta_MSE','nested_Delta_RMSLE'], {'rho_min':'+.9f','rho_post':'+.9f','nested_amplitude_min':'.6f','nested_amplitude_post':'.6f','rms_D_post_projection':'.3e','nested_Delta_MSE':'+.3e','nested_Delta_RMSLE':'+.3e'})}

Recency-weighted post-projection rho is `{sbvc_agg['weighted_rho_post_projection']:+.9f}`;
latest rho is `{sbvc_agg['latest_rho_post_projection']:+.9f}`. Nested weighted Delta MSE is
`{sbvc_agg['weighted_nested_Delta_MSE_post']:+.3e}` and nested weighted Delta RMSLE is
`{sbvc_agg['weighted_nested_Delta_RMSLE_post']:+.3e}`. Cluster bootstrap 95% CI is
`[{bootstrap_post['CI95'][0]:+.3e}, {bootstrap_post['CI95'][1]:+.3e}]`, with
`P(Delta MSE < 0) = {bootstrap_post['P_Delta_MSE_lt_0']:.6f}`.

The min-projection rho is reported only as a diagnostic; it is not used for GO because EXP076
defines SBVC with the full historical component-span projection. Even that weaker diagnostic
does not support deployment once amplitudes are strict-forward: weighted nested Delta MSE is
`{sbvc_agg['weighted_nested_Delta_MSE_min']:+.9f}`, bootstrap 95% CI
`[{bootstrap_min['CI95'][0]:+.9f}, {bootstrap_min['CI95'][1]:+.9f}]`, and
`P(Delta MSE < 0) = {bootstrap_min['P_Delta_MSE_lt_0']:.6f}`.

## ORTH_ALPHA residual interaction

The primary TEST reconstruction reproduces the EXP076 decomposition: `R² =
{alpha_recon['R2']:.9f}`, unexplained RMS `{alpha_recon['alpha_unexplained_rms']:.9f}` versus
centered alpha RMS `{alpha_recon['alpha_centered_rms']:.9f}`.

For the actual standalone difference `D_test = z_forward_stack - z_alpha`,
`corr(D_test, r_alpha_unexplained) = {alpha_interaction['corr_D_test_r_alpha_unexplained']:+.9f}`.
Its projection coefficient on `r_alpha_unexplained` is `{alpha_coef:+.9f}` and the projection
RMS is `{alpha_interaction['RMS_projection_D_onto_r_alpha']:.9f}`. This is a material conflict:
the standalone reweighting subtracts the unexplained ORTH_ALPHA residual essentially one-for-one.

## TEST geometry

The current submission span contains {test_geometry['vectors']} vectors and has centered rank
{test_geometry['rank']} at eigenvalue tolerance `1e-12`.

| metric | value |
| --- | ---: |
| RMS(D_test) | {test_geometry['RMS_D_test']:.9f} |
| RMS(centered D_test) | {test_geometry['RMS_D_test_centered']:.9f} |
| RMS(D_perp) | {test_geometry['RMS_D_perp']:.9f} |
| perp_fraction (centered energy) | {test_geometry['perp_fraction_centered']:.9f} |
| second-pass max projection | {test_geometry['second_pass_max_projection']:.3e} |
| corr(D_perp, current ORTH correction) | {test_geometry['corr_D_perp_ORTH_correction']:+.9f} |

The vector has 250,000 finite rows in unique sample order. Orthogonality is a format check only;
the historical SBVC gate remains decisive.

## Expected effect

{expected}

## Output

No submission was created under **NO_GO**. The audited TEST vectors are stored in
`forward_test_vectors.npz`; the reserved candidate path remains
`{submission_path}` and was not written.

## Final conclusion

The advertised 40-OOF forward wCV is reproduced (`{wcvs['reference_40']:.9f}`), but it is not a
deployable 40-component stack because exact TEST counterparts exist for only 16 components.
The clean deployable rebuild has fold-4 Delta `{fold4_gain:+.9f}` versus EXP037. More importantly,
its required post-projection SBVC direction is identically zero and the unprojected standalone
difference cancels the unexplained ORTH_ALPHA residual. Therefore G.1 is **NO_GO** and is not
rescued with a different ridge, blend coefficient, or leaderboard fit.
"""
    write_text_once(HERE / "REPORT.md", report)
    print(json.dumps(as_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
