"""Журнал экспериментов: одна строка в experiments/log.csv на эксперимент.

Каждая запись самодостаточна: по ней через несколько часов можно понять,
почему текущая версия решения выглядит именно так.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, EXPERIMENTS

LOG = EXPERIMENTS / "log.csv"
FIELDS = ["exp_id", "timestamp", "commit", "description", "scenario", "n_features",
          "model", "params", "cutoffs", "L", "panel_blocks", "fold_scores", "cv_mean",
          "cv_std", "bias_mean", "best_offset", "cv_mean_calib", "delta_vs_b0",
          "runtime_s", "verdict", "conclusion"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


def log_experiment(**kw) -> None:
    row = {k: "" for k in FIELDS}
    row.update({k: v for k, v in kw.items() if k in FIELDS})
    row["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
    row["commit"] = row["commit"] or git_commit()
    for k, v in list(row.items()):
        if isinstance(v, (list, tuple)):
            row[k] = json.dumps([round(x, 5) if isinstance(x, float) else x for x in v])
        elif isinstance(v, dict):
            row[k] = json.dumps(v, sort_keys=True, default=str)
        elif isinstance(v, float):
            row[k] = f"{v:.5f}"
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def save_oof(exp_id: str, user_ids, cutoffs, z, y) -> Path:
    """OOF-предсказания эксперимента: (exp_id, cutoff, user_id, z, y)."""
    p = ARTIFACTS / f"oof_{exp_id}.npz"
    np.savez_compressed(p, user_id=np.asarray(user_ids), cutoff=np.asarray(cutoffs, dtype="U10"),
                        z=np.asarray(z, dtype=np.float32), y=np.asarray(y, dtype=np.float32))
    return p


def load_oof(exp_id: str):
    return np.load(ARTIFACTS / f"oof_{exp_id}.npz", allow_pickle=False)
