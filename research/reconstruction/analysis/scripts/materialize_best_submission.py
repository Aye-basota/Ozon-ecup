"""Materialize the locked no-training SEQ65+BTYD05 submission.

No validation or leaderboard information is read.  The script accepts only the
immutable primitive test predictions, verifies their hashes/alignment, applies
the fixed recipe once, and writes one CSV plus a compact JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


LEVEL = 2.3293
UID_SHA256 = "50e5ba9b71a510b05126d5f325d9c63186ca09975680c66e4ee024e3e0fd576a"
BTYD_TEST_SHA256 = "5222d26166c600ba201958937d7226ba535a49a1c7aeb2a8dc3b328b437e5a43"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(r"C:\Users\Admin\Desktop\OZON-E-CUP"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.source_root / "artifacts"
    z: dict[str, np.ndarray] = {}
    uid_ref: np.ndarray | None = None
    for name, expected_hash in EXPECTED_Z_SHA256.items():
        z_path = artifacts / f"ztest_{name}.npy"
        uid_path = artifacts / f"uid_{name}.npy"
        assert sha256(z_path) == expected_hash
        assert sha256(uid_path) == UID_SHA256
        current_uid = np.load(uid_path).astype(np.int64)
        if uid_ref is None:
            uid_ref = current_uid
        else:
            assert np.array_equal(current_uid, uid_ref)
        z[name] = np.load(z_path).astype(np.float64)
    assert uid_ref is not None and len(uid_ref) == 250_000

    seq = (z["SEQ-01"] + z["SEQ-C289-S43"] + z["SEQ-C289-S44"]) / 3.0
    etx = (z["ETX-01-S42-DCW"] + z["ETX-01-S43-DCW"] + z["ETX-01-S44-DCW"]) / 3.0
    strong = 0.10 * z["S1-CAP"] + 0.20 * z["S1-UNC"] + 0.25 * z["S1-DIST"] + 0.225 * etx + 0.225 * seq
    seq65 = 0.10 * z["S1-CAP"] + 0.10 * z["S1-UNC"] + 0.15 * z["S1-DIST"] + 0.325 * etx + 0.325 * seq

    btyd_path = artifacts / "BTYD_STABLE_EXP051" / "test_raw.npz"
    assert sha256(btyd_path) == BTYD_TEST_SHA256
    with np.load(btyd_path, allow_pickle=False) as data:
        assert np.array_equal(data["user_id"].astype(np.int64), uid_ref)
        assert float(np.max(np.abs(data["z_strongest"].astype(np.float64) - strong))) <= 5e-7
        z_btyd = data["z_btyd"].astype(np.float64)

    raw = 0.95 * seq65 + 0.05 * z_btyd
    shift = LEVEL - float(raw.mean())
    final_z = np.maximum(raw + shift, 0.0)
    predict = np.expm1(final_z)
    assert np.isfinite(predict).all() and (predict >= 0).all()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": uid_ref, "predict": predict}).write_csv(args.output, float_precision=6)
    disk = pl.read_csv(args.output)
    assert disk.columns == ["user_id", "predict"] and disk.height == 250_000
    assert np.array_equal(disk["user_id"].to_numpy(), uid_ref)
    assert float(np.max(np.abs(np.log1p(disk["predict"].to_numpy()) - final_z))) <= 1e-6
    manifest = {
        "recipe": "0.95*SEQ65 + 0.05*BTYD; one global shift to 2.3293; floor z; expm1",
        "output": str(args.output),
        "sha256": sha256(args.output),
        "rows": 250_000,
        "raw_mean_z": float(raw.mean()),
        "shift": shift,
        "disk_mean_log1p": float(np.log1p(disk["predict"].to_numpy()).mean()),
        "zeros": int((disk["predict"].to_numpy() == 0).sum()),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
