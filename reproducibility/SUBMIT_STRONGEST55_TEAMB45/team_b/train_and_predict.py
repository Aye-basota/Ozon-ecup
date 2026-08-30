"""Train the complete team-b-final tabular ensemble and write its prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    raw_data = args.raw_data.expanduser().resolve()
    os.environ["ECUP_RAW_DATA_DIR"] = str(raw_data.parent)
    os.environ["ECUP_TEAM_PROCESSED_DIR"] = str(args.processed_dir.expanduser().resolve())
    sys.path.insert(0, str(ROOT))

    import importlib.metadata
    import numpy as np
    import pandas as pd

    from src.predict import (
        CURRENT_LOG_SCALE,
        CURRENT_WEIGHT,
        TEAM_WEIGHT,
        predict_final_gmv,
    )
    from src.train import train_submit_models

    started = time.perf_counter()
    print(f"read raw data: {raw_data}", flush=True)
    df_raw = pd.read_parquet(raw_data)
    print("train team-b-final tabular models", flush=True)
    models = train_submit_models(df_raw)
    train_seconds = time.perf_counter() - started

    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        with args.model_output.open("wb") as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved models: {args.model_output}", flush=True)

    print("predict team-b-final", flush=True)
    prediction = predict_final_gmv(models, df_raw)
    frame = pd.DataFrame({
        "user_id": prediction.index.to_numpy(dtype=np.int64),
        "predict": prediction.to_numpy(dtype=np.float64),
    })
    if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
        raise AssertionError("team-b prediction has wrong user count")
    if not np.isfinite(frame.predict).all() or (frame.predict < 0).any():
        raise AssertionError("team-b prediction contains invalid values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the archived Team-B component byte-for-byte across platforms.
    frame.to_csv(args.output, index=False, lineterminator="\r\n")

    report = {
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "rows": len(frame),
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - started,
        "current_weight": CURRENT_WEIGHT,
        "team_weight": TEAM_WEIGHT,
        "current_log_scale": CURRENT_LOG_SCALE,
        "mean_log1p": float(np.log1p(frame.predict).mean()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "duckdb", "scikit-learn", "lightgbm", "catboost", "xgboost"]
        },
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
