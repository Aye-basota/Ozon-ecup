"""Blend already generated submission CSV files in log-space.

This is a cheap ensemble layer: it does not retrain models, it combines existing
`submissions/*.csv` files with columns `user_id,predict`.

Examples:
    python src/submission_blend.py stats --files exp_019.csv exp_018.csv
    python src/submission_blend.py blend --files exp_019.csv exp_018.csv --weights 0.8 0.2 \
        --out exp_021_blend.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS


def _path(name: str) -> Path:
    p = Path(name)
    return p if p.exists() else SUBMISSIONS / name


def load_submission(name: str) -> pd.DataFrame:
    df = pd.read_csv(_path(name))
    if list(df.columns) != ["user_id", "predict"]:
        raise ValueError(f"{name}: expected columns user_id,predict, got {list(df.columns)}")
    if len(df) != 250_000:
        raise ValueError(f"{name}: expected 250000 rows, got {len(df)}")
    if df["user_id"].duplicated().any():
        raise ValueError(f"{name}: duplicated user_id")
    pred = df["predict"].to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError(f"{name}: NaN/inf predictions")
    if (pred < 0).any():
        raise ValueError(f"{name}: negative predictions")
    return df


def load_stack(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    users = None
    zs = []
    for name in files:
        df = load_submission(name)
        uid = df["user_id"].to_numpy()
        if users is None:
            users = uid
        elif not np.array_equal(users, uid):
            raise ValueError(f"{name}: user_id order differs from first file")
        zs.append(np.log1p(df["predict"].clip(lower=0).to_numpy()))
    assert users is not None
    return users, np.vstack(zs)


def print_stats(files: list[str]) -> None:
    _, z = load_stack(files)
    rows = []
    for name, zi in zip(files, z):
        pred = np.expm1(zi)
        rows.append({
            "file": Path(name).name,
            "mean_log1p": zi.mean(),
            "mean_pred": pred.mean(),
            "p_zero": float((pred == 0).mean()),
            "max_pred": pred.max(),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    labels = [Path(f).name[:18] for f in files]
    print("\nlog prediction correlation:")
    print(pd.DataFrame(np.corrcoef(z), index=labels, columns=labels).round(6).to_string())

    print("\nvar(log_pred_i - log_pred_j):")
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            print(f"{labels[i]:>18} vs {labels[j]:<18} {np.var(z[i] - z[j]):.6f}")


def blend(files: list[str], weights: list[float], out: str, level: float | None) -> None:
    users, z = load_stack(files)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(files):
        raise ValueError(f"got {len(w)} weights for {len(files)} files")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    w = w / w.sum()
    z_mix = np.average(z, axis=0, weights=w)
    if level is not None:
        z_mix = np.maximum(z_mix + (level - float(z_mix.mean())), 0.0)
    pred = np.maximum(np.expm1(z_mix), 0.0)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / out
    pd.DataFrame({"user_id": users, "predict": pred}).to_csv(out_path, index=False)
    print(f"saved {out_path} rows={len(users)}")
    print(f"weights={dict(zip([Path(f).name for f in files], np.round(w, 4)))}")
    print(
        f"mean_log1p={np.log1p(pred).mean():.6f} "
        f"mean_pred={pred.mean():.6f} max={pred.max():.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--files", nargs="+", required=True)

    p_blend = sub.add_parser("blend")
    p_blend.add_argument("--files", nargs="+", required=True)
    p_blend.add_argument("--weights", nargs="+", type=float, required=True)
    p_blend.add_argument("--out", required=True)
    p_blend.add_argument("--level", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "stats":
        print_stats(args.files)
    elif args.cmd == "blend":
        blend(args.files, args.weights, args.out, args.level)


if __name__ == "__main__":
    main()
