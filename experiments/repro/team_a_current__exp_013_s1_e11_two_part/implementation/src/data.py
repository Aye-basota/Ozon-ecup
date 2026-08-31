"""Загрузка сырых данных. Единственная точка чтения train.parquet."""
from __future__ import annotations

import polars as pl

from src.config import RAW_PARQUET, SAMPLE_SUBMIT

# 10 информативных колонок из 18: остальные — точные производные (eda_findings §1.3).
USE_COLS = ["user_id", "event_date", "searches", "cat", "search_to_cart", "search_to_ord",
            "cat_to_cart", "cat_to_ord", "to_cart", "to_ord", "gmv", "gmv_search", "gmv_cat"]

_CACHE: dict[str, object] = {}


def load() -> pl.DataFrame:
    """Полный дневной лог, 30.6 млн строк. Кэшируется в памяти процесса."""
    if "df" not in _CACHE:
        _CACHE["df"] = pl.read_parquet(RAW_PARQUET, columns=USE_COLS)
    return _CACHE["df"]  # type: ignore[return-value]


def sample_submit() -> pl.DataFrame:
    if "ss" not in _CACHE:
        _CACHE["ss"] = pl.read_csv(SAMPLE_SUBMIT)
    return _CACHE["ss"]  # type: ignore[return-value]
