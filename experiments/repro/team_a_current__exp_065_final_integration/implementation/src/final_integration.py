"""EXP-065: independently rebuild and package the two final candidates.

Run: python src/final_integration.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ANCHOR_BAND, ARTIFACTS, ROOT
from src.data import sample_submit


RESULTS = ROOT / "research" / "strategies" / "results" / "FINAL_INTEGRATION_EXP065"
PACKAGE = ROOT / "submissions" / "FINAL_20260825_A1"
EXPECTED_A = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"
EXPECTED_B = "c3cfb4d90f50ceff8f5d8f8aaca072664966fb91018eb0a3fa01195dc38c2932"
LEVEL = 2.3293
COMPONENTS = {
    "CAP": (["S1-CAP"], 0.10, "z_cap"),
    "UNC": (["S1-UNC"], 0.20, "z_unc"),
    "DIST": (["S1-DIST"], 0.25, "z_dist"),
    "SEQ_AVG3": (["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"], 0.225, "z_seq_avg3"),
    "ETX_AVG3": (["ETX-01-S42-DCW", "ETX-01-S43-DCW", "ETX-01-S44-DCW"],
                 0.225, "z_etx_avg3"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_csv(path: Path, expected_users: np.ndarray) -> dict:
    frame = pl.read_csv(path)
    assert frame.columns == ["user_id", "predict"]
    assert frame.height == 250_000 and frame["user_id"].n_unique() == 250_000
    uid = frame["user_id"].to_numpy()
    pred = frame["predict"].to_numpy().astype(float)
    assert np.array_equal(uid, expected_users)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    level = float(np.log1p(pred).mean())
    assert ANCHOR_BAND[0] <= level <= ANCHOR_BAND[1]
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "rows": int(len(pred)), "unique_users": int(len(np.unique(uid))),
        "finite": True, "nonnegative": True, "mean_log1p": level,
        "mean_predict": float(pred.mean()), "min": float(pred.min()), "max": float(pred.max()),
        "zero_fraction": float((pred == 0).mean()),
        "quantiles": {str(q): float(np.quantile(pred, q)) for q in (0.001, 0.01, 0.5, 0.99, 0.999)},
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    allowed = {"candidate_A_STRONGEST_CURRENT.csv", "candidate_B_BTYD05_HEDGE.csv"}
    unexpected = {p.name for p in PACKAGE.iterdir()} - allowed
    assert not unexpected, f"unexpected files in final package: {sorted(unexpected)}"
    expected_users = sample_submit()["user_id"].to_numpy()
    test_components: dict[str, np.ndarray] = {}
    source_files: dict[str, dict] = {}
    component_uid: np.ndarray | None = None
    primitive_z: list[np.ndarray] = []
    primitive_w: list[float] = []
    for logical, (names, logical_weight, _) in COMPONENTS.items():
        seeds = []
        for name in names:
            z_path, u_path = ARTIFACTS / f"ztest_{name}.npy", ARTIFACTS / f"uid_{name}.npy"
            z, uid = np.load(z_path).astype(float), np.load(u_path)
            assert len(z) == 250_000 and np.isfinite(z).all()
            assert np.array_equal(uid, expected_users), f"{name}: sample user order mismatch"
            if component_uid is None:
                component_uid = uid
            assert np.array_equal(uid, component_uid)
            seeds.append(z)
            primitive_z.append(z)
            primitive_w.append(logical_weight / len(names))
            source_files[name] = {
                "z_sha256": sha256(z_path), "uid_sha256": sha256(u_path),
                "mean_z": float(z.mean()), "min_z": float(z.min()), "max_z": float(z.max()),
            }
        test_components[logical] = np.mean(seeds, axis=0)
    assert abs(sum(primitive_w) - 1.0) < 1e-12
    z_mix = np.average(np.vstack(primitive_z), axis=0, weights=np.asarray(primitive_w))
    level_shift = LEVEL - float(z_mix.mean())
    z_final = np.maximum(z_mix + level_shift, 0.0)
    pred = np.maximum(np.expm1(z_final), 0.0)

    a_path = PACKAGE / "candidate_A_STRONGEST_CURRENT.csv"
    pl.DataFrame({"user_id": expected_users, "predict": pred}).write_csv(a_path, float_precision=6)
    a = validate_csv(a_path, expected_users)
    assert a["sha256"] == EXPECTED_A, (a["sha256"], EXPECTED_A)

    b_source = ROOT / "submissions" / "submission_BTYD05.csv"
    assert sha256(b_source) == EXPECTED_B
    b_path = PACKAGE / "candidate_B_BTYD05_HEDGE.csv"
    shutil.copyfile(b_source, b_path)
    b = validate_csv(b_path, expected_users)
    assert b["sha256"] == EXPECTED_B

    aligned_path = ARTIFACTS / "RESDISC_053" / "aligned_oof.parquet"
    aligned = pd.read_parquet(
        aligned_path, columns=["z_strong_raw"] + [v[2] for v in COMPONENTS.values()])
    oof_strong = aligned["z_strong_raw"].to_numpy(float)
    test_strong = z_mix
    support: dict[str, dict] = {}
    for logical, (_, _, oof_col) in COMPONENTS.items():
        d_oof = aligned[oof_col].to_numpy(float) - oof_strong
        d_test = test_components[logical] - test_strong
        support[logical] = {
            "var_difference_oof": float(np.var(d_oof)),
            "var_difference_test": float(np.var(d_test)),
            "test_oof_ratio": float(np.var(d_test) / np.var(d_oof)),
            "test_corr_with_ensemble": float(np.corrcoef(test_components[logical], test_strong)[0, 1]),
        }
    pairwise = []
    for left, right in itertools.combinations(COMPONENTS, 2):
        loof, rooff = COMPONENTS[left][2], COMPONENTS[right][2]
        v_oof = float(np.var(aligned[loof].to_numpy(float) - aligned[rooff].to_numpy(float)))
        v_test = float(np.var(test_components[left] - test_components[right]))
        pairwise.append({
            "left": left, "right": right, "var_oof": v_oof, "var_test": v_test,
            "test_oof_ratio": v_test / v_oof,
        })
    critical = next(row for row in pairwise
                    if {row["left"], row["right"]} == {"SEQ_AVG3", "ETX_AVG3"})
    assert 0.6 <= critical["test_oof_ratio"] <= 1.2

    btyd_summary_path = RESULTS.parent / "BTYD_STABLE_EXP051" / "summary.json"
    btyd_summary = json.loads(btyd_summary_path.read_text(encoding="utf-8"))
    assert btyd_summary["btyd_production_status"] == "PASS"
    assert btyd_summary["test_support_status"] == "PASS"
    assert 0.6 <= float(btyd_summary["variance_ratio"]) <= 1.4
    btyd_oof = ARTIFACTS / "BTYD_STABLE_EXP051" / "oof_raw.npz"
    btyd_test = ARTIFACTS / "BTYD_STABLE_EXP051" / "test_raw.npz"
    source_a = ROOT / "submissions" / "submission_STRONGEST_CURRENT.csv"
    assert sha256(source_a) == EXPECTED_A
    package_files = sorted(p.name for p in PACKAGE.iterdir())
    assert package_files == ["candidate_A_STRONGEST_CURRENT.csv", "candidate_B_BTYD05_HEDGE.csv"]
    za = np.log1p(pl.read_csv(a_path)["predict"].to_numpy())
    zb = np.log1p(pl.read_csv(b_path)["predict"].to_numpy())
    summary = {
        "experiment_id": 65, "prefix": "FINAL_INTEGRATION_EXP065",
        "development_reference": "STRONGEST-CURRENT / exp_037",
        "decision": {"verdict": "ACCEPT", "strongest_current_changed": False,
                     "leaderboard_uploaded": False, "exactly_two_candidates": True},
        "candidate_A": a, "candidate_B": b,
        "candidate_pair": {
            "correlation_log1p": float(np.corrcoef(za, zb)[0, 1]),
            "var_log1p_difference": float(np.var(za - zb)),
            "max_abs_log1p_difference": float(np.max(np.abs(za - zb))),
        },
        "rebuild": {
            "recipe_weights": dict(zip(
                [n for names, _, _ in COMPONENTS.values() for n in names], primitive_w)),
            "raw_mix_mean_z": float(z_mix.mean()), "level_shift": level_shift,
            "registered_source_sha256": sha256(source_a), "byte_identical_to_registered": True,
            "component_sources": source_files,
        },
        "regime_support": {
            "aligned_oof_sha256": sha256(aligned_path),
            "component_to_ensemble": support, "pairwise": pairwise,
            "critical_etx_seq_ratio": critical["test_oof_ratio"], "critical_gate_pass": True,
        },
        "btyd_provenance": {
            "summary_sha256": sha256(btyd_summary_path),
            "oof_raw_sha256": sha256(btyd_oof), "test_raw_sha256": sha256(btyd_test),
            "oof_fixed_delta_wcv": -0.000320983,
            "test_correction_variance_ratio": btyd_summary["variance_ratio"],
            "production_status": btyd_summary["btyd_production_status"],
            "support_status": btyd_summary["test_support_status"],
        },
        "package_files": package_files,
    }
    (RESULTS / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(pairwise).to_csv(RESULTS / "component_pairwise_regime.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
