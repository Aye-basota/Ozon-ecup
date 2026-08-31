# Public-LB calibrated final blend in z=log1p space.

## Catalogue metadata

- **Catalogue ID:** `teammate_research__make_last20`
- **Namespace:** `teammate_research`
- **Experiment ID:** `make_last20`
- **Original source:** `пайплайн сокомандника/research_scripts/make_last20.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** teammate research runner
- **Model:** ensemble, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** "submission_extra90_3_*.csv",
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# make_last20

Original script: `пайплайн сокомандника/research_scripts/make_last20.py`

```python
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                    help="best_bas directory; default = script directory")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV; default = <root>/last.csv")
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out if args.out is not None else root / "last.csv").resolve()

    friend = find_one(root, [
        "submission_STRONGEST_CURRENT/submission/submission_STRONGEST_CURRENT.csv",
        "submission_STRONGEST_CURRENT.csv",
    ], "FRIEND")
    b = find_one(root, [
        "submission_final6h_B_*.csv",
        "*final6h_B*.csv",
    ], "PREVIOUS_B")
    x3 = find_one(root, [
        "submission_extra90_3_*.csv",
        "*extra90_3*.csv",
    ], "EXTRA90_3")

    uid_f, zf = read_submit(friend)
    uid_b, zb = read_submit(b)
    uid_x, zx = read_submit(x3)
    if not (np.array_equal(uid_f, uid_b) and np.array_equal(uid_f, uid_x)):
        raise ValueError("user_id order differs between submissions")

    # Convex blend in the competition's natural z=log1p space.
    z = W_FRIEND * zf + W_B * zb + W_X3 * zx
    z = np.maximum(z, 0.0)
    pred = np.expm1(z)
    if not np.all(np.isfinite(pred)) or np.any(pred < 0):
        raise RuntimeError("Invalid final predictions")

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": uid_f, "predict": pred}).to_csv(out, index=False)

    # Diagnostics: enough to catch wrong files/order immediately.
    def dist(a: np.ndarray, c: np.ndarray):
        d = a - c
        return (float(np.corrcoef(a, c)[0, 1]),
                float(np.sqrt(np.mean(d*d))),
                float(np.mean(np.abs(d))),
                float(np.mean(np.abs(d) > 0.02)))

    log("DONE:", out)
    log("weights friend/B/x3 =", W_FRIEND, W_B, W_X3)
    log("rows =", len(pred), "mean_log1p =", float(z.mean()),
        "min/max =", float(pred.min()), float(pred.max()))
    log("distance vs FRIEND corr/rmssd/mae/pct>|.02| =", dist(z, zf))
    log("distance vs X3     corr/rmssd/mae/pct>|.02| =", dist(z, zx))


if __name__ == "__main__":
    main()

```
