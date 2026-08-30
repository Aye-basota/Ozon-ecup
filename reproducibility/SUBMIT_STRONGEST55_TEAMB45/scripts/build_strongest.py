"""Build STRONGEST_CURRENT from nine component vectors.

The three tabular vectors may come from a fresh training run while the six
SEQ/ETX vectors stay frozen for the bounded reproducibility audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen" / "strongest" / "predictions"
REFERENCE = ROOT / "frozen" / "strongest" / "submission_STRONGEST_CURRENT.csv"
EXPECTED_SHA256 = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"
LEVEL = 2.3293
COMPONENTS = (
    ("S1-CAP", 0.100),
    ("S1-UNC", 0.200),
    ("S1-DIST", 0.250),
    ("SEQ-01", 0.075),
    ("SEQ-C289-S43", 0.075),
    ("SEQ-C289-S44", 0.075),
    ("ETX-01-S42-DCW", 0.075),
    ("ETX-01-S43-DCW", 0.075),
    ("ETX-01-S44-DCW", 0.075),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular-dir", type=Path, default=FROZEN)
    parser.add_argument("--frozen-dir", type=Path, default=FROZEN)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    tabular_names = {"S1-CAP", "S1-UNC", "S1-DIST"}
    uid_ref = None
    z_parts = []
    component_stats = {}
    for name, _ in COMPONENTS:
        source = args.tabular_dir if name in tabular_names else args.frozen_dir
        z = np.load(source / f"ztest_{name}.npy")
        uid = np.load(source / f"uid_{name}.npy")
        if len(z) != 250_000 or len(uid) != 250_000 or not np.isfinite(z).all():
            raise AssertionError(f"invalid component {name}")
        if uid_ref is None:
            uid_ref = uid
        elif not np.array_equal(uid_ref, uid):
            raise AssertionError(f"user_id mismatch: {name}")
        z_parts.append(z.astype(np.float64, copy=False))
        component_stats[name] = {"source": str(source), "mean_log": float(z.mean())}

    weights = np.asarray([weight for _, weight in COMPONENTS], dtype=np.float64)
    z_mix = np.average(np.vstack(z_parts), axis=0, weights=weights)
    delta = LEVEL - float(z_mix.mean())
    prediction = np.maximum(np.expm1(np.maximum(z_mix + delta, 0.0)), 0.0)

    reference = pl.read_csv(args.reference)
    order = reference.select("user_id").with_row_index("_order")
    submission = (
        pl.DataFrame({"user_id": uid_ref, "predict": prediction.astype(np.float64)})
        .join(order, on="user_id", how="inner")
        .sort("_order")
        .drop("_order")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The archived STRONGEST_CURRENT component uses LF line endings.
    submission.write_csv(args.output, float_precision=6, line_terminator="\n")

    output_prediction = pl.read_csv(args.output)["predict"].to_numpy()
    reference_prediction = reference["predict"].to_numpy()
    diff = np.log1p(output_prediction) - np.log1p(reference_prediction)
    report = {
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "expected_sha256": EXPECTED_SHA256,
        "delta": delta,
        "max_abs_log_error_vs_reference": float(np.max(np.abs(diff))),
        "rms_log_error_vs_reference": float(np.sqrt(np.mean(diff**2))),
        "components": component_stats,
    }
    if args.require_exact and report["sha256"] != EXPECTED_SHA256:
        raise AssertionError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
