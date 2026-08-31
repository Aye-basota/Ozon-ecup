from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Public-LB calibrated final blend in z=log1p space.
# The weights are deliberately convex and conservative:
#   12% fixed teammate champion + 16% previous meta-occ B + 72% best raw-occ X3.
# This keeps the strong production ensemble dominant while mixing two different
# occurrence corrections. No model training is performed.
W_FRIEND = 0.12
W_B = 0.16
W_X3 = 0.72


def log(*x):
    print(*x, flush=True)


def find_one(root: Path, patterns: list[str], label: str) -> Path:
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))
        hits.extend(root.glob(f"**/{pat}"))
    # de-duplicate while keeping newest files first
    uniq = {}
    for p in hits:
        try:
            if p.is_file():
                uniq[str(p.resolve())] = p
        except OSError:
            pass
    hits = list(uniq.values())
    if not hits:
        raise FileNotFoundError(f"Cannot find {label}. Tried: {patterns}")
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer canonical generated locations/names when several renamed copies exist.
    pref = [p for p in hits if "submissions" in {q.lower() for q in p.parts}]
    chosen = pref[0] if pref else hits[0]
    log(f"{label}: {chosen}")
    return chosen


def read_submit(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "user_id" not in df.columns or "predict" not in df.columns:
        raise ValueError(f"Bad submission format: {path}")
    uid = df["user_id"].to_numpy(np.int64)
    pred = df["predict"].to_numpy(np.float64)
    if len(uid) != 250_000:
        log(f"WARNING: {path.name} has {len(uid):,} rows (expected 250,000)")
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"NaN/inf in {path}")
    pred = np.clip(pred, 0.0, None)
    return uid, np.log1p(pred)


def main() -> None:
    root = Path(__file__).resolve().parent
    comp = root / "components"
    out = root / "latest_rebuilt.csv"

    friend = comp / "friend.csv"
    b = comp / "occ_meta_B.csv"
    x3 = comp / "occ_raw_X3.csv"

    uid_f, zf = read_submit(friend)
    uid_b, zb = read_submit(b)
    uid_x, zx = read_submit(x3)
    if not (np.array_equal(uid_f, uid_b) and np.array_equal(uid_f, uid_x)):
        raise ValueError("user_id order differs between submissions")

    z = W_FRIEND * zf + W_B * zb + W_X3 * zx
    z = np.maximum(z, 0.0)
    pred = np.expm1(z)
    if not np.all(np.isfinite(pred)) or np.any(pred < 0):
        raise RuntimeError("Invalid final predictions")

    pd.DataFrame({"user_id": uid_f, "predict": pred}).to_csv(out, index=False)
    ref = pd.read_csv(root / "latest.csv")
    zr = np.log1p(np.clip(ref["predict"].to_numpy(np.float64), 0.0, None))
    max_err = float(np.max(np.abs(z-zr)))
    print("DONE:", out)
    print("weights friend/B/X3 =", W_FRIEND, W_B, W_X3)
    print("max |log1p(rebuilt)-log1p(reference)| =", max_err)
    if max_err > 1e-10:
        raise RuntimeError(f"Rebuild mismatch: {max_err}")

if __name__ == "__main__":
    main()
