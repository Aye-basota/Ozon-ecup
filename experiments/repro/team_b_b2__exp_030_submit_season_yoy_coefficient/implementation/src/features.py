"""Feature pipeline for the E-CUP LTV baseline.

Cutoff is the first day of the target window:
- features use only rows with event_date < cutoff;
- target uses rows with cutoff <= event_date < cutoff + 30 days.

Run: python src/features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, TARGET_DAYS


DATA_START = pd.Timestamp("2025-01-01")
DATASET_CACHE = ROOT / "artifacts" / "datasets"

FEATURES = [
    "w180_days_buy",
    "w180_orders",
    "w365_days_buy",
    "w365_orders",
    "w90_days_buy",
    "rec_over_buygap",
    "w90_orders",
    "lifetime_gmv_per_day",
    "rec_buy",
    "tenure",
    "w180_lgmv_std",
    "w365_gmv",
    "w14_searches",
    "w7_searches",
    "rec_cart",
    "w30_gmv",
    "rec_any",
    "w180_gmv",
    "w14_days_present",
    "w365_lgmv_mean",
    "w180_lgmv_mean",
    "all_days_buy",
    "w60_gmv",
    "w7_days_present",
    "first_buy_age",
    "gap_cv",
    "w60_orders",
    "w60_days_buy",
    "rec_over_gap",
    "w365_srch_per_day",
    "w14_days_search",
    "w365_days_cat",
    "w30_searches",
    "w365_buyday_rate",
    "w365_lgmv_std",
    "buygap_cv",
    "dlog_buyd_90_365",
    "w14_carts",
    "rec_cat",
    "w365_lgmv",
    "buygap_mean",
    "w90_gmv",
    "w30_orders",
    "w30_days_cart",
    "w365_gmv_max",
    "w60_lgmv",
    "w30_days_buy",
    "w14_gmv",
    "buygap_std",
    "w14_days_cart",
    "lag1_gmv",
    "lag1_orders",
    "lag1_carts",
    "lag1_searches",
    "lag2_gmv",
    "lag2_orders",
    "lag2_carts",
    "lag2_searches",
    "lag3_gmv",
    "lag3_orders",
    "lag3_carts",
    "lag3_searches",
    "lag7_gmv",
    "lag7_orders",
    "lag7_carts",
    "lag7_searches",
    "lag14_gmv",
    "lag14_orders",
    "lag14_carts",
    "lag14_searches",
    "lag21_gmv",
    "lag21_orders",
    "lag21_carts",
    "lag21_searches",
    "lag28_gmv",
    "lag28_orders",
    "lag28_carts",
    "lag28_searches",
    "var_predictability",
]

CHECK_COLS = ["gmv", "to_ord", "to_cart", "searches"]
DERIVED_COLS = ["lgmv", "is_buy_day", "is_cart_day", "is_search_day", "is_cat_day"]


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(map(str, paths)))


def train_path() -> Path:
    return _first_existing([ROOT / "data" / "train.parquet", ROOT / "data" / "raw" / "train.parquet"])


def sample_submit_path() -> Path:
    return _first_existing(
        [
            ROOT / "data" / "sample_submit.csv",
            ROOT / "data" / "sample_submission.csv",
            ROOT / "sample_submit.csv",
            ROOT / "sample_submission.csv",
            ROOT / "data" / "raw" / "sample_submit.csv",
            ROOT / "data" / "raw" / "sample_submission.csv",
        ]
    )


def load_train_data(path: Path | None = None) -> pd.DataFrame:
    path = path or train_path()
    df = pd.read_parquet(path)
    df["event_date"] = pd.to_datetime(df["event_date"])

    for col in CHECK_COLS:
        df[col] = df[col].fillna(0)

    dup_count = int(df.duplicated(["user_id", "event_date"]).sum())
    if dup_count:
        value_cols = [col for col in df.columns if col not in ["user_id", "event_date"]]
        df = df.groupby(["user_id", "event_date"], as_index=False, sort=False)[value_cols].sum()
        print(f"Aggregated duplicate (user_id, event_date) rows: {dup_count}")

    return add_derived_columns(df)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in DERIVED_COLS if col not in df.columns]
    if not missing:
        return df

    df = df.copy()
    df["lgmv"] = np.log1p(df["gmv"].clip(lower=0)).astype("float32")
    df["is_buy_day"] = (df["to_ord"] > 0).astype("int8")
    df["is_cart_day"] = (df["to_cart"] > 0).astype("int8")
    df["is_search_day"] = (df["search"] == 1).astype("int8")
    df["is_cat_day"] = (df["cat"] == 1).astype("int8")
    return df


def load_sample_submit(path: Path | None = None) -> pd.DataFrame:
    path = path or sample_submit_path()
    sample = pd.read_csv(path)
    if "user_id" not in sample.columns:
        raise ValueError(f"{path} must contain user_id")
    return sample


def get_all_users(df: pd.DataFrame) -> pd.Index:
    return pd.Index(np.sort(df["user_id"].unique()), name="user_id")


def validate_data_report(df: pd.DataFrame, sample_submit: pd.DataFrame | None = None) -> pd.Index:
    all_users = get_all_users(df)
    print(f"train_shape={df.shape}")
    print(f"date_min={df['event_date'].min().date()}")
    print(f"date_max={df['event_date'].max().date()}")
    assert df["event_date"].min() == pd.Timestamp("2025-01-01")
    assert df["event_date"].max() == pd.Timestamp("2026-02-13")
    print(f"unique_users={len(all_users)}")
    print(f"duplicates_user_date={int(df.duplicated(['user_id', 'event_date']).sum())}")

    for col in CHECK_COLS:
        nan_count = int(df[col].isna().sum())
        neg_count = int((df[col].fillna(0) < 0).sum())
        print(f"{col}_nan={nan_count}")
        print(f"{col}_negative={neg_count}")
        assert neg_count == 0

    buy_users = df.loc[df["to_ord"].fillna(0) > 0, "user_id"].nunique()
    print(f"buy_users={buy_users}")
    print(f"buy_user_share={buy_users / len(all_users):.6f}")
    print("gmv_quantiles=")
    print(df["gmv"].fillna(0).quantile([0.5, 0.9, 0.99]).to_string())

    if sample_submit is not None:
        sample_users = pd.Index(sample_submit["user_id"].unique()).sort_values()
        print(f"sample_shape={sample_submit.shape}")
        print(f"sample_columns={list(sample_submit.columns)}")
        print(f"sample_unique_users={len(sample_users)}")
        assert len(sample_submit) == len(sample_users)
        assert set(sample_users) == set(all_users)
        print("sample_user_set_matches_ALL_USERS=True")

    return all_users


def _covered_days(cutoff: pd.Timestamp, days: int) -> int:
    return min(days, (cutoff - DATA_START).days + 1)


def _days_since(cutoff: pd.Timestamp, dates: pd.Series, default: float = 999.0) -> pd.Series:
    out = (cutoff - dates).dt.days.astype("float32")
    return out.fillna(default)


def _add_window_aggs(
    X: pd.DataFrame,
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    days: int,
    aggregations: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=days)
    part = history.loc[(history["event_date"] >= start) & (history["event_date"] < cutoff)]
    if part.empty:
        for name in aggregations:
            X[name] = 0.0
        return X

    agg = part.groupby("user_id", sort=False).agg(**aggregations)
    return X.join(agg, how="left")


def _add_daily_lag_features(X: pd.DataFrame, history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cols = ["gmv", "to_ord", "to_cart", "searches"]
    rename_base = {"gmv": "gmv", "to_ord": "orders", "to_cart": "carts", "searches": "searches"}
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        lag_date = cutoff - pd.Timedelta(days=lag)
        part = history.loc[history["event_date"] == lag_date, ["user_id"] + cols]
        if part.empty:
            for suffix in rename_base.values():
                X[f"lag{lag}_{suffix}"] = 0.0
            continue

        lag_frame = part.set_index("user_id")[cols].rename(
            columns={col: f"lag{lag}_{name}" for col, name in rename_base.items()}
        )
        X = X.join(lag_frame, how="left")
    return X


def _gap_stats(days: pd.DataFrame, all_users: pd.Index) -> pd.DataFrame:
    if days.empty:
        return pd.DataFrame(index=all_users, columns=["gap_mean", "gap_std", "gap_count"], dtype="float32")

    ordered = days.sort_values(["user_id", "event_date"])
    gaps = ordered.groupby("user_id", sort=False)["event_date"].diff().dt.days
    gap_frame = pd.DataFrame({"user_id": ordered["user_id"].to_numpy(), "gap": gaps.to_numpy()}).dropna()
    if gap_frame.empty:
        return pd.DataFrame(index=all_users, columns=["gap_mean", "gap_std", "gap_count"], dtype="float32")

    stats = gap_frame.groupby("user_id", sort=False)["gap"].agg(
        gap_mean="mean", gap_std="std", gap_count="count"
    )
    return stats.reindex(all_users)


def _dataset_cache_paths(cutoff: pd.Timestamp) -> tuple[Path, Path]:
    stamp = cutoff.date().isoformat()
    return DATASET_CACHE / f"X_{stamp}.parquet", DATASET_CACHE / f"y_{stamp}.parquet"


def _read_cached_dataset(
    cutoff: pd.Timestamp,
    all_users: pd.Index,
    need_target: bool,
) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame | None:
    X_path, y_path = _dataset_cache_paths(cutoff)
    if not X_path.exists() or (need_target and not y_path.exists()):
        return None

    X = pd.read_parquet(X_path)
    if list(X.columns) != FEATURES or not X.index.equals(all_users):
        return None

    if not need_target:
        return X.astype("float32")

    y_frame = pd.read_parquet(y_path)
    y = y_frame.iloc[:, 0]
    if not y.index.equals(all_users):
        return None
    return X.astype("float32"), y.astype("float32")


def _write_cached_dataset(
    cutoff: pd.Timestamp,
    X: pd.DataFrame,
    y: pd.Series | None = None,
) -> None:
    X_path, y_path = _dataset_cache_paths(cutoff)
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    X.to_parquet(X_path)
    if y is not None:
        y.rename("gmv").to_frame().to_parquet(y_path)


def build_features(
    cutoff_date: str | pd.Timestamp,
    history: pd.DataFrame | None = None,
    all_users: pd.Index | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the fixed 50-feature user table for one cutoff."""
    cutoff = pd.Timestamp(cutoff_date)

    if history is None:
        df = load_train_data()
        history = df.loc[df["event_date"] < cutoff]
    else:
        history = add_derived_columns(history)
        assert (history["event_date"] < cutoff).all()

    if all_users is None:
        all_users = get_all_users(history)
    all_users = pd.Index(all_users, name="user_id")
    X = pd.DataFrame(index=all_users)

    if history.empty:
        for feature in FEATURES:
            X[feature] = 0.0
        recency_cols = ["rec_buy", "rec_cart", "rec_any", "first_buy_age", "rec_cat", "buygap_mean"]
        X[recency_cols] = 999.0
        return X[FEATURES].astype("float32")

    all_agg = history.groupby("user_id", sort=False).agg(
        w365_days_buy=("is_buy_day", "sum"),
        w365_orders=("to_ord", "sum"),
        w365_gmv=("gmv", "sum"),
        w365_lgmv_mean=("lgmv", "mean"),
        w365_lgmv_std=("lgmv", "std"),
        w365_days_cat=("is_cat_day", "sum"),
        w365_lgmv=("lgmv", "sum"),
        w365_gmv_max=("lgmv", "max"),
        _w365_searches=("searches", "sum"),
    )
    X = X.join(all_agg, how="left")

    X = _add_window_aggs(
        X,
        history,
        cutoff,
        180,
        {
            "w180_days_buy": ("is_buy_day", "sum"),
            "w180_orders": ("to_ord", "sum"),
            "w180_lgmv_std": ("lgmv", "std"),
            "w180_gmv": ("gmv", "sum"),
            "w180_lgmv_mean": ("lgmv", "mean"),
        },
    )
    X = _add_window_aggs(
        X,
        history,
        cutoff,
        90,
        {
            "w90_days_buy": ("is_buy_day", "sum"),
            "w90_orders": ("to_ord", "sum"),
            "w90_gmv": ("gmv", "sum"),
        },
    )
    X = _add_window_aggs(
        X,
        history,
        cutoff,
        60,
        {
            "w60_gmv": ("gmv", "sum"),
            "w60_orders": ("to_ord", "sum"),
            "w60_days_buy": ("is_buy_day", "sum"),
            "w60_lgmv": ("lgmv", "sum"),
        },
    )
    X = _add_window_aggs(
        X,
        history,
        cutoff,
        30,
        {
            "w30_gmv": ("gmv", "sum"),
            "w30_searches": ("searches", "sum"),
            "w30_orders": ("to_ord", "sum"),
            "w30_days_cart": ("is_cart_day", "sum"),
            "w30_days_buy": ("is_buy_day", "sum"),
        },
    )
    X = _add_window_aggs(
        X,
        history,
        cutoff,
        14,
        {
            "w14_searches": ("searches", "sum"),
            "w14_days_present": ("event_date", "size"),
            "w14_days_search": ("is_search_day", "sum"),
            "w14_carts": ("to_cart", "sum"),
            "w14_gmv": ("gmv", "sum"),
            "w14_days_cart": ("is_cart_day", "sum"),
        },
    )
    X = _add_window_aggs(
        X,
        history,
        cutoff,
        7,
        {
            "w7_searches": ("searches", "sum"),
            "w7_days_present": ("event_date", "size"),
        },
    )

    first_date = history.groupby("user_id", sort=False)["event_date"].min().reindex(all_users)
    last_any = history.groupby("user_id", sort=False)["event_date"].max().reindex(all_users)
    X["tenure"] = ((cutoff - first_date).dt.days + 1).fillna(0).astype("float32")
    X["rec_any"] = _days_since(cutoff, last_any)

    buy_days = history.loc[history["is_buy_day"] > 0, ["user_id", "event_date"]]
    if buy_days.empty:
        X["rec_buy"] = 999.0
        X["first_buy_age"] = 999.0
    else:
        last_buy = buy_days.groupby("user_id", sort=False)["event_date"].max().reindex(all_users)
        first_buy = buy_days.groupby("user_id", sort=False)["event_date"].min().reindex(all_users)
        X["rec_buy"] = _days_since(cutoff, last_buy)
        X["first_buy_age"] = _days_since(cutoff, first_buy)

    cart_days = history.loc[history["is_cart_day"] > 0, ["user_id", "event_date"]]
    if cart_days.empty:
        X["rec_cart"] = 999.0
    else:
        last_cart = cart_days.groupby("user_id", sort=False)["event_date"].max().reindex(all_users)
        X["rec_cart"] = _days_since(cutoff, last_cart)

    cat_days = history.loc[history["is_cat_day"] > 0, ["user_id", "event_date"]]
    if cat_days.empty:
        X["rec_cat"] = 999.0
    else:
        last_cat = cat_days.groupby("user_id", sort=False)["event_date"].max().reindex(all_users)
        X["rec_cat"] = _days_since(cutoff, last_cat)

    presence_stats = _gap_stats(history[["user_id", "event_date"]], all_users)
    gap_mean = presence_stats["gap_mean"]
    gap_std = presence_stats["gap_std"].fillna(0)
    gap_count = presence_stats["gap_count"].fillna(0)
    X["gap_cv"] = np.where((gap_count >= 2) & (gap_mean > 0), gap_std / gap_mean, 0.0)
    X["rec_over_gap"] = np.where(gap_mean.notna(), X["rec_any"] / (gap_mean + 1.0), 999.0)

    buy_stats = _gap_stats(buy_days, all_users)
    buygap_mean_raw = buy_stats["gap_mean"]
    buygap_std = buy_stats["gap_std"].fillna(0)
    X["buygap_mean"] = buygap_mean_raw.fillna(999.0)
    X["buygap_std"] = np.where(buygap_mean_raw.notna(), buygap_std, 0.0)
    X["buygap_cv"] = np.where(
        buygap_mean_raw.notna() & (buygap_mean_raw > 0), buygap_std / buygap_mean_raw, 0.0
    )
    X["rec_over_buygap"] = np.where(
        buygap_mean_raw.notna(), X["rec_buy"] / (buygap_mean_raw + 1.0), 999.0
    )

    covered_365 = _covered_days(cutoff, 365)
    X["all_days_buy"] = X["w365_days_buy"]
    X["lifetime_gmv_per_day"] = X["w365_gmv"] / np.maximum(X["tenure"], 1.0)
    X["w365_srch_per_day"] = X["_w365_searches"] / covered_365
    X["w365_buyday_rate"] = X["w365_days_buy"] / covered_365
    X["dlog_buyd_90_365"] = np.log1p(X["w90_days_buy"] / 90.0) - np.log1p(
        np.maximum(X["w365_days_buy"] - X["w90_days_buy"], 0.0) / 275.0
    )
    X = _add_daily_lag_features(X, history, cutoff)
    amount_cv = X["w365_lgmv_std"] / (X["w365_lgmv_mean"].abs() + 1.0)
    X["var_predictability"] = amount_cv + X["buygap_cv"].clip(lower=0, upper=10) + X["gap_cv"].clip(lower=0, upper=10)

    for feature in FEATURES:
        if feature not in X.columns:
            X[feature] = 0.0

    X = X[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert X.shape[0] == len(all_users)
    assert list(X.columns) == FEATURES
    assert not np.isinf(X.to_numpy()).any()
    assert not X.isna().any().any()
    return X.astype("float32")


def make_dataset(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    all_users: pd.Index | np.ndarray,
    need_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    all_users = pd.Index(all_users, name="user_id")

    cached = _read_cached_dataset(cutoff, all_users, need_target)
    if cached is not None:
        return cached

    history = df.loc[df["event_date"] < cutoff]
    X = build_features(cutoff, history=history, all_users=all_users)
    assert X.shape[0] == len(all_users)

    if not need_target:
        _write_cached_dataset(cutoff, X)
        return X

    target_end = cutoff + pd.Timedelta(days=TARGET_DAYS)
    future = df.loc[(df["event_date"] >= cutoff) & (df["event_date"] < target_end)]
    unique_target_dates = int(future["event_date"].nunique())
    assert unique_target_dates == TARGET_DAYS, (
        f"Target window for cutoff={cutoff.date()} has {unique_target_dates} dates, "
        f"expected {TARGET_DAYS}"
    )

    y = future.groupby("user_id", sort=False)["gmv"].sum().reindex(all_users, fill_value=0.0)
    y = y.astype("float32")
    assert y.shape[0] == len(all_users)
    assert not y.isna().any()
    _write_cached_dataset(cutoff, X, y)
    return X, y


def main() -> None:
    df = load_train_data()
    sample_submit = load_sample_submit()
    validate_data_report(df, sample_submit)
    all_users = get_all_users(df)
    X, y = make_dataset(df, "2026-01-14", all_users)
    print(f"check_X_shape={X.shape}")
    print(f"check_y_shape={y.shape}")
    print(f"feature_count={len(FEATURES)}")


if __name__ == "__main__":
    main()
