"""Build the fixed SEQ65_TEMPORAL_HEAVY production-regime probe.

No model is trained here.  The script accepts only the immutable production
predictions bundled with STRONGEST_CURRENT, verifies their SHA256 hashes and
alignment, reproduces the champion, evaluates the fixed recipe on existing OOF,
and writes one submission plus a diagnostics JSON file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT / "artifacts"
SUBMISSIONS = ROOT / "submissions"
OUT_DIR = ROOT / "research" / "strategies" / "results" / "SEQ65_TEMPORAL_HEAVY"
OUTPUT = SUBMISSIONS / "submission_SEQ65_TEMPORAL_HEAVY.csv"
DIAGNOSTICS = OUT_DIR / "diagnostics.json"
REFERENCE = SUBMISSIONS / "submission_STRONGEST_CURRENT.csv"
LEVEL = 2.3293

REFERENCE_SHA256 = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"
UID_SHA256 = "50e5ba9b71a510b05126d5f325d9c63186ca09975680c66e4ee024e3e0fd576a"
EXPECTED_Z_SHA256 = {
    "S1-CAP": "d6b3c59920d816cb54c5b65e7daf8de0cea3edc338bedb8ea78e3fb01086e7d9",
    "S1-UNC": "c2a94114f709f8127f4d7ce61dea545a103e4e66851188d153a40d7bc2757773",
    "S1-DIST": "ad974c09e0c97dedcc622877a5937f9d40c7f4f6604c7bb5eb08d3b7c73fe966",
    "SEQ-01": "c20ae75cee1eef216ae86a6fa5f594850369cdd92d1a946675942481d194b630",
    "SEQ-C289-S43": "66bdc5718af747ae0fd3059f18fa97395b4a5715d7085422064bb7d3a6ed022b",
    "SEQ-C289-S44": "7662569a90c9ff78c32b6126e09b2b4571a7b46e2d75047c5993c0eef25fd631",
    "ETX-01-S42-DCW": "2a9f9955503578fb48b959c7253f1d8a7de0c1ffb85704dfb2ff85253fea1c39",
    "ETX-01-S43-DCW": "eba71ea4cc7eb43958fa5b3fae9ae6812052643293677f85d8acac7b86283c04",
    "ETX-01-S44-DCW": "eb69c69fbabef0648ae555bcda4ada05d3ec3d7bbf8214b10fa2ebaab4a19c34",
}

STRONG_WEIGHTS = {
    "S1-CAP": 0.10,
    "S1-UNC": 0.20,
    "S1-DIST": 0.25,
    "SEQ-01": 0.075,
    "SEQ-C289-S43": 0.075,
    "SEQ-C289-S44": 0.075,
    "ETX-01-S42-DCW": 0.075,
    "ETX-01-S43-DCW": 0.075,
    "ETX-01-S44-DCW": 0.075,
}
SEQ65_WEIGHTS = {
    "S1-CAP": 0.10,
    "S1-UNC": 0.10,
    "S1-DIST": 0.15,
    "SEQ-01": 0.325 / 3.0,
    "SEQ-C289-S43": 0.325 / 3.0,
    "SEQ-C289-S44": 0.325 / 3.0,
    "ETX-01-S42-DCW": 0.325 / 3.0,
    "ETX-01-S43-DCW": 0.325 / 3.0,
    "ETX-01-S44-DCW": 0.325 / 3.0,
}
OOF_NAMES = ("S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3")
OOF_STRONG_WEIGHTS = np.asarray([0.10, 0.20, 0.25, 0.225, 0.225], dtype=np.float64)
OOF_SEQ65_WEIGHTS = np.asarray([0.10, 0.10, 0.15, 0.325, 0.325], dtype=np.float64)
FOLD_WEIGHTS = np.asarray([1, 2, 4, 8], dtype=np.float64) / 15.0
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_production() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    uid_ref = None
    z_by_name: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name, expected_z_hash in EXPECTED_Z_SHA256.items():
        z_path = ARTIFACTS / f"ztest_{name}.npy"
        uid_path = ARTIFACTS / f"uid_{name}.npy"
        z_hash = sha256(z_path)
        uid_hash = sha256(uid_path)
        assert z_hash == expected_z_hash, f"wrong production z hash: {name}"
        assert uid_hash == UID_SHA256, f"wrong production uid hash: {name}"
        z = np.load(z_path)
        uid = np.load(uid_path)
        assert z.ndim == uid.ndim == 1 and len(z) == len(uid) == 250_000
        assert np.isfinite(z).all() and (z >= 0).all(), f"invalid z: {name}"
        if uid_ref is None:
            uid_ref = uid
        else:
            assert np.array_equal(uid, uid_ref), f"uid alignment mismatch: {name}"
        z_by_name[name] = z.astype(np.float64, copy=False)
        hashes[f"ztest_{name}.npy"] = z_hash
        hashes[f"uid_{name}.npy"] = uid_hash
    assert uid_ref is not None
    return uid_ref, z_by_name, hashes


def blend(z_by_name: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    assert set(weights) == set(z_by_name)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    return sum(weights[name] * z_by_name[name] for name in weights)


def production_level(z_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact project policy: shift raw z mean to LEVEL, then floor z at zero."""
    delta = LEVEL - float(z_raw.mean())
    z_cal = np.maximum(z_raw + delta, 0.0)
    assert abs(float((z_raw + delta).mean()) - LEVEL) < 1e-12
    return z_cal, delta


def verify_champion(uid: np.ndarray, z_cal: np.ndarray) -> dict[str, float | str]:
    assert sha256(REFERENCE) == REFERENCE_SHA256, "champion CSV hash mismatch"
    ref = pl.read_csv(REFERENCE)
    assert ref.columns == ["user_id", "predict"] and ref.height == 250_000
    assert np.array_equal(ref["user_id"].to_numpy(), uid), "champion/production uid mismatch"
    z_file = np.log1p(ref["predict"].to_numpy())
    max_error = float(np.max(np.abs(z_file - z_cal)))
    assert max_error < 1e-6, "production semantics do not reconstruct champion"
    return {"sha256": REFERENCE_SHA256, "max_abs_z_reconstruction_error": max_error}


def oof_diagnostics() -> dict[str, object]:
    loaded = []
    for name in OOF_NAMES:
        path = ARTIFACTS / f"oof_{name}.npz"
        assert path.exists(), f"missing existing OOF: {name}"
        d = np.load(path)
        key = np.rec.fromarrays([d["cutoff"], d["user_id"]], names="cutoff,user_id")
        order = np.argsort(key, order=("cutoff", "user_id"))
        loaded.append((d, order, key[order]))

    base_d, base_order, base_key = loaded[0]
    y = base_d["y"][base_order].astype(np.float64)
    cutoff = base_d["cutoff"][base_order]
    parts = []
    for name, (d, order, key) in zip(OOF_NAMES, loaded):
        assert np.array_equal(key, base_key), f"OOF alignment mismatch: {name}"
        assert np.array_equal(d["y"][order], base_d["y"][base_order]), f"OOF target mismatch: {name}"
        parts.append(d["z"][order].astype(np.float64))

    Z = np.vstack(parts)
    z_strong = np.average(Z, axis=0, weights=OOF_STRONG_WEIGHTS)
    z_seq65 = np.average(Z, axis=0, weights=OOF_SEQ65_WEIGHTS)
    ly = np.log1p(y)
    folds = sorted(set(cutoff.tolist()))
    assert len(folds) == 4

    def fold_scores(z: np.ndarray) -> list[float]:
        return [float(np.std(ly[cutoff == fold] - z[cutoff == fold])) for fold in folds]

    strong_fold = fold_scores(z_strong)
    seq65_fold = fold_scores(z_seq65)
    strong_wcv = float(FOLD_WEIGHTS @ strong_fold)
    seq65_wcv = float(FOLD_WEIGHTS @ seq65_fold)
    delta = seq65_wcv - strong_wcv
    assert delta <= 0.005, f"gross OOF deterioration {delta:+.6f}; submission blocked"
    return {
        "informational_only": True,
        "folds": folds,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "strongest_fold_rmsle": strong_fold,
        "seq65_fold_rmsle": seq65_fold,
        "delta_by_fold": [new - old for new, old in zip(seq65_fold, strong_fold)],
        "strongest_wcv": strong_wcv,
        "seq65_wcv": seq65_wcv,
        "delta_wcv": delta,
    }


def main() -> None:
    uid, z_by_name, component_hashes = load_production()
    z_strong_raw = blend(z_by_name, STRONG_WEIGHTS)
    z_seq65_raw = blend(z_by_name, SEQ65_WEIGHTS)
    z_strong, delta_strong = production_level(z_strong_raw)
    z_seq65, delta_seq65 = production_level(z_seq65_raw)
    champion = verify_champion(uid, z_strong)
    oof = oof_diagnostics()

    delta_before = z_seq65_raw - z_strong_raw
    delta_after = z_seq65 - z_strong
    production = {
        "var_delta_z": float(np.var(delta_after)),
        "pearson_z": float(np.corrcoef(z_seq65, z_strong)[0, 1]),
        "mean_delta_z_before_normalization": float(delta_before.mean()),
        "mean_delta_z_after_normalization": float(delta_after.mean()),
        "max_abs_delta_z": float(np.max(np.abs(delta_after))),
        "delta_z_quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(delta_after, QUANTILES))},
        "strongest_raw_mean_z": float(z_strong_raw.mean()),
        "seq65_raw_mean_z": float(z_seq65_raw.mean()),
        "strongest_shift": delta_strong,
        "seq65_shift": delta_seq65,
        "strongest_final_mean_log1p": float(z_strong.mean()),
        "seq65_final_mean_log1p": float(z_seq65.mean()),
        "seq65_pre_floor_mean_z": float((z_seq65_raw + delta_seq65).mean()),
    }

    pred = np.maximum(np.expm1(z_seq65), 0.0)
    sub = pl.DataFrame({"user_id": uid, "predict": pred.astype(np.float64)})
    assert sub.columns == ["user_id", "predict"] and sub.height == 250_000
    assert sub["user_id"].n_unique() == 250_000
    assert np.array_equal(sub["user_id"].to_numpy(), pl.read_csv(REFERENCE)["user_id"].to_numpy())
    assert np.isfinite(pred).all() and (pred >= 0).all()
    if OUTPUT.exists():
        existing = pl.read_csv(OUTPUT)
        assert existing.columns == sub.columns and existing.height == sub.height
        assert np.array_equal(existing["user_id"].to_numpy(), uid)
        assert np.max(np.abs(existing["predict"].to_numpy() - pred)) < 5.1e-7
    else:
        sub.write_csv(OUTPUT, float_precision=6)

    disk = pl.read_csv(OUTPUT)
    p_disk = disk["predict"].to_numpy()
    assert disk.columns == ["user_id", "predict"] and disk.height == 250_000
    assert np.array_equal(disk["user_id"].to_numpy(), uid)
    assert np.isfinite(p_disk).all() and (p_disk >= 0).all()
    max_reconstruction_error = float(np.max(np.abs(np.log1p(p_disk) - z_seq65)))
    assert max_reconstruction_error < 1e-6
    submission_hash = sha256(OUTPUT)

    report = {
        "purpose": "large representation-balance LB probe",
        "base": "STRONGEST_CURRENT",
        "only_structural_change": "sequence total weight 0.45 -> 0.65",
        "level": LEVEL,
        "semantics": "log-space blend; global shift to raw mean z=2.3293; floor z at 0; expm1",
        "weights_grouped": {"CAP": 0.10, "UNC": 0.10, "DIST": 0.15, "ETX-AVG3": 0.325, "SEQ-AVG3": 0.325},
        "component_hashes": component_hashes,
        "champion_reconstruction": champion,
        "production_diagnostics": production,
        "oof": oof,
        "submission": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "sha256": submission_hash,
            "rows": disk.height,
            "columns": disk.columns,
            "unique_user_id": disk["user_id"].n_unique(),
            "nan_or_inf": int((~np.isfinite(p_disk)).sum()),
            "negative": int((p_disk < 0).sum()),
            "zeros": int((p_disk == 0).sum()),
            "min": float(p_disk.min()),
            "max": float(p_disk.max()),
            "mean_log1p": float(np.log1p(p_disk).mean()),
            "max_abs_z_reconstruction_error": max_reconstruction_error,
        },
    }
    DIAGNOSTICS.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("components verified")
    print("blend built")
    print("submission verified")
    print(json.dumps({"production": production, "oof": oof, "submission": report["submission"]}, indent=2))


if __name__ == "__main__":
    main()
