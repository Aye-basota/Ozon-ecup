"""Build the preregistered observable H12 interpolation submission.

No fitting, calibration, segment logic, or alternative scale is performed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANCHOR = ROOT / "research/new_directions/CLAUDE_PUBLIC_CEILING/SUBMIT_PUBLIC_EB.csv"
H21 = ROOT / "submissions/SUBMIT_ORTH_FINAL.csv"
OUTPUT = ROOT / "submissions/SUBMIT_ORTH_ROBUST_H12_INTERP.csv"
ALPHA = 12.0 / 21.0
EXPECTED_ROWS = 250_000


def main() -> None:
    anchor = pd.read_csv(ANCHOR)
    h21 = pd.read_csv(H21)

    assert list(anchor.columns) == ["user_id", "predict"]
    assert list(h21.columns) == ["user_id", "predict"]
    assert len(anchor) == len(h21) == EXPECTED_ROWS
    assert anchor["user_id"].is_unique and h21["user_id"].is_unique
    assert np.array_equal(anchor["user_id"].to_numpy(), h21["user_id"].to_numpy())

    anchor_predict = anchor["predict"].to_numpy(np.float64)
    h21_predict = h21["predict"].to_numpy(np.float64)
    assert np.isfinite(anchor_predict).all() and np.isfinite(h21_predict).all()
    assert (anchor_predict >= 0.0).all() and (h21_predict >= 0.0).all()

    za = np.log1p(anchor_predict)
    z21 = np.log1p(h21_predict)
    z = za + ALPHA * (z21 - za)
    prediction = np.maximum(np.expm1(z), 0.0)

    assert np.isfinite(prediction).all()
    assert (prediction >= 0.0).all()

    submission = pd.DataFrame(
        {"user_id": anchor["user_id"].to_numpy(), "predict": prediction}
    )
    submission.to_csv(OUTPUT, index=False, float_format="%.10f", lineterminator="\n")

    written = pd.read_csv(OUTPUT)
    assert len(written) == EXPECTED_ROWS
    assert list(written.columns) == ["user_id", "predict"]
    assert written["user_id"].is_unique
    assert np.array_equal(written["user_id"].to_numpy(), anchor["user_id"].to_numpy())
    assert np.isfinite(written["predict"].to_numpy(np.float64)).all()
    assert (written["predict"].to_numpy(np.float64) >= 0.0).all()

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"path={OUTPUT}")
    print(f"sha256={digest}")
    print(f"rows={len(written)}")


if __name__ == "__main__":
    main()
