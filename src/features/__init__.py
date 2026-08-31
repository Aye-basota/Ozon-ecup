"""Compatibility surface for the two preserved Team-A source layouts.

The merged history contains both the original ``src/features.py`` pipeline and
the later clean-workspace ``src/features/canonical.py`` package.  Python gives
the package precedence for ``import src.features``; without this bridge every
original experiment imports the wrong implementation.  Bare imports therefore
retain the original behavior, while the clean pipeline remains explicitly
available as ``src.features.canonical``.
"""
from __future__ import annotations

from pathlib import Path


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "features.py"
exec(compile(_LEGACY_PATH.read_text(encoding="utf-8"), str(_LEGACY_PATH), "exec"), globals())
__all__ = sorted(name for name in globals() if not name.startswith("__"))
