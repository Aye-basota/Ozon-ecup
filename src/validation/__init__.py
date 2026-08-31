"""Compatibility exports for original and clean-workspace validation layouts.

The protected original implementation stays byte-unchanged in
``src/validation.py``.  This package bridge only resolves Python's module/package
name precedence after both histories were merged.
"""
from __future__ import annotations

from pathlib import Path

from src.validation.evaluate import evaluate_oof as _clean_evaluate_oof
from src.validation.folds import canonical_folds as _clean_canonical_folds
from src.validation.folds import cutoff_grid as _clean_cutoff_grid


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "validation.py"
exec(compile(_LEGACY_PATH.read_text(encoding="utf-8"), str(_LEGACY_PATH), "exec"), globals())

evaluate_oof = _clean_evaluate_oof
canonical_folds = _clean_canonical_folds
# Bare ``cutoff_grid`` belongs to the original API; the clean grid remains
# available at ``src.validation.folds.cutoff_grid``.
clean_cutoff_grid = _clean_cutoff_grid
__all__ = sorted(name for name in globals() if not name.startswith("__"))
