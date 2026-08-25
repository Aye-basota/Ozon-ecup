"""Repository and external-path configuration.

Machine-specific paths live only in ``config/paths.local.yaml`` (ignored by
Git) or in the file selected by ``ECUP_PATHS_CONFIG``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


@dataclass(frozen=True)
class Paths:
    data_root: Path
    old_repo_root: Path
    external_artifacts_root: Path
    submission_geometry_root: Path

    @property
    def raw_parquet(self) -> Path:
        return self.data_root / str(competition()["data"]["raw_parquet"])

    @property
    def sample_submission(self) -> Path:
        return self.data_root / str(competition()["data"]["sample_submission"])

    @property
    def processed_root(self) -> Path:
        return self.data_root / str(competition()["features"]["cache_dir"])


def _absolute(value: Any, config_path: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


@lru_cache(maxsize=1)
def paths() -> Paths:
    configured = os.environ.get("ECUP_PATHS_CONFIG")
    config_path = Path(configured) if configured else ROOT / "config" / "paths.local.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. Copy config/paths.example.yaml to "
            "config/paths.local.yaml and set absolute paths."
        )
    raw = _read_yaml(config_path)
    required = (
        "data_root",
        "old_repo_root",
        "external_artifacts_root",
        "submission_geometry_root",
    )
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f"Missing path keys in {config_path}: {missing}")
    return Paths(**{key: _absolute(raw[key], config_path) for key in required})


@lru_cache(maxsize=1)
def competition() -> dict[str, Any]:
    return _read_yaml(ROOT / "config" / "competition.yaml")


def clear_settings_cache() -> None:
    """Useful in tests that temporarily select a different local config."""
    paths.cache_clear()
    competition.cache_clear()
