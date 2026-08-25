"""Fast environment, path, import, and key-artifact validation."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import competition, paths


REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "yaml",
    "sklearn",
    "lightgbm",
    "catboost",
    "src.data.loaders",
    "src.features.canonical",
    "src.models.tabular",
    "src.metrics.rmsle",
    "src.validation.folds",
    "src.validation.evaluate",
    "src.validation.workflow",
    "src.blending.align",
]


def main() -> None:
    checks: dict[str, object] = {"imports": {}, "paths": {}, "artifacts": {}}
    failures: list[str] = []
    for module in REQUIRED_IMPORTS:
        try:
            imported = importlib.import_module(module)
            checks["imports"][module] = getattr(imported, "__version__", "ok")
        except Exception as exc:
            checks["imports"][module] = f"ERROR: {exc}"
            failures.append(f"import {module}: {exc}")

    path_cfg = paths()
    required_paths = {
        "data_root": path_cfg.data_root,
        "raw_parquet": path_cfg.raw_parquet,
        "sample_submission": path_cfg.sample_submission,
        "old_repo_root": path_cfg.old_repo_root,
        "external_artifacts_root": path_cfg.external_artifacts_root,
        "submission_geometry_root": path_cfg.submission_geometry_root,
    }
    for name, path in required_paths.items():
        exists = path.exists()
        checks["paths"][name] = {"path": str(path), "exists": exists}
        if not exists:
            failures.append(f"missing path {name}: {path}")

    baseline = ROOT / str(competition()["offline_baseline"]["artifact"])
    external_oof = [
        path_cfg.external_artifacts_root / "oof_S1-E03a.npz",
        path_cfg.external_artifacts_root / "oof_S1-E02.npz",
        path_cfg.external_artifacts_root / "oof_S1-DIST.npz",
        path_cfg.external_artifacts_root / "oof_SEQ-AVG3.npz",
        path_cfg.external_artifacts_root / "oof_ETX-AVG3.npz",
    ]
    geometry_incumbent = path_cfg.submission_geometry_root / "submission_geometry" / "SUBMIT_NEXT_BEST.csv"
    for path in [baseline, *external_oof, geometry_incumbent]:
        exists = path.exists()
        checks["artifacts"][str(path)] = exists
        if not exists:
            failures.append(f"missing key artifact: {path}")

    if path_cfg.raw_parquet.exists():
        metadata = pq.ParquetFile(path_cfg.raw_parquet).metadata
        checks["raw_dataset"] = {"rows": metadata.num_rows, "columns": metadata.num_columns}
        if metadata.num_rows != 30_631_006 or metadata.num_columns != 18:
            failures.append("raw dataset geometry differs from audited source")
    if baseline.exists():
        metadata = pq.ParquetFile(baseline).metadata
        checks["baseline"] = {"rows": metadata.num_rows, "columns": metadata.num_columns}
        if metadata.num_rows != 770_616:
            failures.append("canonical baseline row count differs from 770616")

    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
