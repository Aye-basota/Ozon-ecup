"""Compatibility exports for original and clean-workspace model layouts."""
from __future__ import annotations

from pathlib import Path

from src.models.tabular import fit_model as _clean_fit_model
from src.models.tabular import predict_model as _clean_predict_model


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "models.py"
exec(compile(_LEGACY_PATH.read_text(encoding="utf-8"), str(_LEGACY_PATH), "exec"), globals())

# The clean workflow deliberately keeps these two generic entry points.
fit_model = _clean_fit_model
predict_model = _clean_predict_model
__all__ = sorted(
    name for name in globals() if not name.startswith("__")
)
