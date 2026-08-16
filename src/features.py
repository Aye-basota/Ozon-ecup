"""Единый пайплайн фичей: build_features(cutoff_date).

Один и тот же код строит фичи для train, val и test.
Фичи считаются ТОЛЬКО на данных до cutoff — никакого лукапа.
Новые фичи — только новые колонки, чужие не переписываем.

Запуск: python src/features.py  (строит и сохраняет фичи в data/processed/)
"""
import pandas as pd
import duckdb
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, DATA_RAW

RAW_TRAIN = DATA_RAW / "train.parquet"
DEFAULT_FEATURE_SET = "recency"

BASE_COLS = [
    "search",
    "cat",
    "has_search_to_cart",
    "has_search_to_ord",
    "has_cat_to_cart",
    "has_cat_to_ord",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
    "to_cart",
    "to_ord",
    "gmv",
    "searches",
]

WINDOWS = [7, 14, 30, 60, 120]
LONG_WINDOWS = [90, 180, 365]
TREND_COLS = ["active_days", "searches", "to_cart", "to_ord", "gmv"]
RECENT_PREVIOUS_WINDOWS = [(7, 14), (30, 60), (60, 120)]
SHORT_LONG_WINDOWS = [(7, 30), (14, 60), (30, 120)]
FEATURE_SETS = {
    "baseline",
    "conversions",
    "trends",
    "recency",
    "conversions_recency",
    "long_buy",
    "long_buy_post_order",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Деление для фичей-конверсий: если знаменатель 0, возвращаем 0."""
    numerator_values = numerator.to_numpy(dtype="float64", copy=False)
    denominator_values = denominator.to_numpy(dtype="float64", copy=False)
    result = np.divide(
        numerator_values,
        denominator_values,
        out=np.zeros_like(numerator_values, dtype="float64"),
        where=denominator_values != 0,
    )
    return pd.Series(result, index=numerator.index)


def _window_series(features: pd.DataFrame, col: str, days: int) -> pd.Series:
    if col == "active_days":
        return features[f"active_days_{days}d"]
    return features[f"{col}_sum_{days}d"]


def _add_conversion_features(features: pd.DataFrame) -> pd.DataFrame:
    ratio_features = {}
    for suffix in ["all", *[f"{days}d" for days in WINDOWS]]:
        ratio_features[f"search_to_cart_rate_{suffix}"] = _safe_ratio(
            features[f"search_to_cart_sum_{suffix}"],
            features[f"searches_sum_{suffix}"],
        )
        ratio_features[f"search_to_order_rate_{suffix}"] = _safe_ratio(
            features[f"search_to_ord_sum_{suffix}"],
            features[f"searches_sum_{suffix}"],
        )
        ratio_features[f"search_cart_to_order_rate_{suffix}"] = _safe_ratio(
            features[f"search_to_ord_sum_{suffix}"],
            features[f"search_to_cart_sum_{suffix}"],
        )
        ratio_features[f"cat_cart_to_order_rate_{suffix}"] = _safe_ratio(
            features[f"cat_to_ord_sum_{suffix}"],
            features[f"cat_to_cart_sum_{suffix}"],
        )
        ratio_features[f"total_cart_to_order_rate_{suffix}"] = _safe_ratio(
            features[f"to_ord_sum_{suffix}"],
            features[f"to_cart_sum_{suffix}"],
        )
        ratio_features[f"avg_order_value_{suffix}"] = _safe_ratio(
            features[f"gmv_sum_{suffix}"],
            features[f"to_ord_sum_{suffix}"],
        )
        ratio_features[f"gmv_per_cart_{suffix}"] = _safe_ratio(
            features[f"gmv_sum_{suffix}"],
            features[f"to_cart_sum_{suffix}"],
        )
        ratio_features[f"search_avg_order_value_{suffix}"] = _safe_ratio(
            features[f"gmv_search_sum_{suffix}"],
            features[f"search_to_ord_sum_{suffix}"],
        )
        ratio_features[f"cat_avg_order_value_{suffix}"] = _safe_ratio(
            features[f"gmv_cat_sum_{suffix}"],
            features[f"cat_to_ord_sum_{suffix}"],
        )
        ratio_features[f"search_gmv_share_{suffix}"] = _safe_ratio(
            features[f"gmv_search_sum_{suffix}"],
            features[f"gmv_sum_{suffix}"],
        )
        ratio_features[f"cat_gmv_share_{suffix}"] = _safe_ratio(
            features[f"gmv_cat_sum_{suffix}"],
            features[f"gmv_sum_{suffix}"],
        )
    return pd.concat([features, pd.DataFrame(ratio_features, index=features.index)], axis=1)


def _add_trend_features(features: pd.DataFrame) -> pd.DataFrame:
    trend_features = {}
    for col in TREND_COLS:
        for recent_days, total_days in RECENT_PREVIOUS_WINDOWS:
            recent = _window_series(features, col, recent_days)
            total = _window_series(features, col, total_days)
            previous = total - recent
            trend_features[f"{col}_recent_prev_diff_{recent_days}d"] = recent - previous
            trend_features[f"{col}_recent_prev_ratio_{recent_days}d"] = _safe_ratio(recent, previous)
            trend_features[f"{col}_recent_total_share_{recent_days}of{total_days}d"] = _safe_ratio(
                recent,
                total,
            )

        for short_days, long_days in SHORT_LONG_WINDOWS:
            short_daily = _window_series(features, col, short_days) / short_days
            long_daily = _window_series(features, col, long_days) / long_days
            trend_features[f"{col}_daily_ratio_{short_days}d_to_{long_days}d"] = _safe_ratio(
                short_daily,
                long_daily,
            )
    return pd.concat([features, pd.DataFrame(trend_features, index=features.index)], axis=1)


def _add_post_order_features(features: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    query = f"""
        WITH base AS (
            SELECT *
            FROM read_parquet($path)
            WHERE event_date < CAST($cutoff AS DATE)
        ),
        last_order AS (
            SELECT
                user_id,
                MAX(CASE WHEN to_ord > 0 THEN event_date END) AS last_order_date
            FROM base
            GROUP BY user_id
        )
        SELECT
            b.user_id,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date THEN 1 ELSE 0 END)
                AS post_order_active_days,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date AND b.search > 0 THEN 1 ELSE 0 END)
                AS post_order_search_days,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date AND b.cat > 0 THEN 1 ELSE 0 END)
                AS post_order_cat_days,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date AND b.to_cart > 0 THEN 1 ELSE 0 END)
                AS post_order_cart_days,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date THEN b.search ELSE 0 END)
                AS post_order_search_sum,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date THEN b.cat ELSE 0 END)
                AS post_order_cat_sum,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date THEN b.searches ELSE 0 END)
                AS post_order_searches_sum,
            SUM(CASE WHEN lo.last_order_date IS NOT NULL AND b.event_date > lo.last_order_date THEN b.to_cart ELSE 0 END)
                AS post_order_cart_sum
        FROM base AS b
        JOIN last_order AS lo USING (user_id)
        GROUP BY b.user_id
    """
    with duckdb.connect() as con:
        duckdb_tmp = DATA_PROCESSED / "duckdb_tmp"
        duckdb_tmp.mkdir(parents=True, exist_ok=True)
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET temp_directory = ?", [str(duckdb_tmp)])
        post_order = con.execute(
            query,
            {"path": str(RAW_TRAIN), "cutoff": cutoff_date},
        ).fetchdf()

    features = features.merge(post_order, on="user_id", how="left").fillna(0)
    has_order = features["recency_to_ord_days"] < 9999
    derived = {
        "has_activity_after_last_order": (features["post_order_active_days"] > 0).astype("int8"),
        "has_search_after_last_order": (features["post_order_search_days"] > 0).astype("int8"),
        "has_cart_after_last_order": (features["post_order_cart_days"] > 0).astype("int8"),
        "no_activity_after_last_order": (
            (features["recency_to_ord_days"] < 9999) & (features["post_order_active_days"] == 0)
        ).astype("int8"),
        "last_search_after_last_order": (
            (features["recency_to_ord_days"] < 9999)
            & (features["recency_search_days"] < features["recency_to_ord_days"])
        ).astype("int8"),
        "last_cart_after_last_order": (
            (features["recency_to_ord_days"] < 9999)
            & (features["recency_to_cart_days"] < features["recency_to_ord_days"])
        ).astype("int8"),
        "days_last_order_to_last_search": (features["recency_to_ord_days"] - features["recency_search_days"]).where(
            has_order,
            0,
        ),
        "days_last_order_to_last_cart": (features["recency_to_ord_days"] - features["recency_to_cart_days"]).where(
            has_order,
            0,
        ),
        "post_order_active_share": _safe_ratio(features["post_order_active_days"], features["recency_to_ord_days"]),
        "post_order_search_share_all": _safe_ratio(features["post_order_searches_sum"], features["searches_sum_all"]),
        "post_order_cart_share_all": _safe_ratio(features["post_order_cart_sum"], features["to_cart_sum_all"]),
        "post_order_searches_per_day": _safe_ratio(
            features["post_order_searches_sum"],
            features["post_order_active_days"],
        ),
        "post_order_cart_per_day": _safe_ratio(features["post_order_cart_sum"], features["post_order_active_days"]),
    }
    return pd.concat([features, pd.DataFrame(derived, index=features.index)], axis=1)


def build_features(cutoff_date: str, feature_set: str = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    """Построить фичи для всех пользователей на дату cutoff.

    Возвращает DataFrame: index — user_id, колонки — фичи.
    Таргет (сумма заказов за 30 дней после cutoff) добавляет train.py,
    здесь его нет, чтобы не было лукапа.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Неизвестный feature_set={feature_set}. Доступно: {sorted(FEATURE_SETS)}")

    if not RAW_TRAIN.exists():
        raise FileNotFoundError(f"Не найден файл с сырыми данными: {RAW_TRAIN}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_PROCESSED / f"features_{feature_set}_{cutoff_date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path).sort_values("user_id").set_index("user_id")

    select_parts = [
        "user_id",
        "COUNT(*) AS active_days_all",
        "DATE_DIFF('day', MAX(event_date), CAST($cutoff AS DATE)) AS recency_days",
    ]

    for col in BASE_COLS:
        select_parts.append(f"SUM({col}) AS {col}_sum_all")
        select_parts.append(f"AVG({col}) AS {col}_mean_all")
        select_parts.append(f"MAX({col}) AS {col}_max_all")

    for days in WINDOWS:
        condition = (
            f"event_date >= CAST($cutoff AS DATE) - INTERVAL {days} DAY "
            "AND event_date < CAST($cutoff AS DATE)"
        )
        select_parts.append(f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) AS active_days_{days}d")
        for col in BASE_COLS:
            select_parts.append(
                f"SUM(CASE WHEN {condition} THEN {col} ELSE 0 END) AS {col}_sum_{days}d"
            )

    if feature_set in {"recency", "conversions_recency", "long_buy", "long_buy_post_order"}:
        event_conditions = {
            "search": "search > 0",
            "cat": "cat > 0",
            "to_cart": "to_cart > 0",
            "to_ord": "to_ord > 0",
            "gmv": "gmv > 0",
            "search_to_cart": "search_to_cart > 0",
            "cat_to_cart": "cat_to_cart > 0",
        }
        for name, condition in event_conditions.items():
            select_parts.append(
                "COALESCE("
                f"DATE_DIFF('day', MAX(CASE WHEN {condition} THEN event_date END), CAST($cutoff AS DATE)), "
                f"9999) AS recency_{name}_days"
            )

    if feature_set in {"long_buy", "long_buy_post_order"}:
        select_parts.extend(
            [
                "DATE_DIFF('day', MIN(event_date), CAST($cutoff AS DATE)) AS tenure_days",
                "COALESCE(DATE_DIFF('day', MIN(CASE WHEN to_ord > 0 THEN event_date END), CAST($cutoff AS DATE)), 9999) AS first_buy_age_days",
                "SUM(CASE WHEN to_ord > 0 THEN 1 ELSE 0 END) AS all_days_buy",
                "STDDEV_SAMP(CASE WHEN gmv > 0 THEN LN(1 + gmv) END) AS all_lgmv_std",
                "AVG(CASE WHEN gmv > 0 THEN LN(1 + gmv) END) AS all_lgmv_mean",
            ]
        )
        for days in LONG_WINDOWS:
            condition = (
                f"event_date >= CAST($cutoff AS DATE) - INTERVAL {days} DAY "
                "AND event_date < CAST($cutoff AS DATE)"
            )
            select_parts.extend(
                [
                    f"SUM(CASE WHEN {condition} AND to_ord > 0 THEN 1 ELSE 0 END) AS w{days}_days_buy",
                    f"SUM(CASE WHEN {condition} THEN to_ord ELSE 0 END) AS w{days}_orders",
                    f"SUM(CASE WHEN {condition} THEN gmv ELSE 0 END) AS w{days}_gmv",
                    f"AVG(CASE WHEN {condition} AND gmv > 0 THEN LN(1 + gmv) END) AS w{days}_lgmv_mean",
                    f"STDDEV_SAMP(CASE WHEN {condition} AND gmv > 0 THEN LN(1 + gmv) END) AS w{days}_lgmv_std",
                ]
            )

    query = f"""
        SELECT
            {", ".join(select_parts)}
        FROM read_parquet($path)
        WHERE event_date < CAST($cutoff AS DATE)
        GROUP BY user_id
    """

    with duckdb.connect() as con:
        duckdb_tmp = DATA_PROCESSED / "duckdb_tmp"
        duckdb_tmp.mkdir(parents=True, exist_ok=True)
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET temp_directory = ?", [str(duckdb_tmp)])
        features = con.execute(
            query,
            {"path": str(RAW_TRAIN), "cutoff": cutoff_date},
        ).fetchdf()

    features = features.fillna(0)

    for days in WINDOWS:
        features[f"gmv_per_active_day_{days}d"] = _safe_ratio(
            features[f"gmv_sum_{days}d"],
            features[f"active_days_{days}d"],
        )
        features[f"orders_per_active_day_{days}d"] = _safe_ratio(
            features[f"to_ord_sum_{days}d"],
            features[f"active_days_{days}d"],
        )

    if feature_set in {"conversions", "conversions_recency"}:
        features = _add_conversion_features(features)
    elif feature_set == "trends":
        features = _add_trend_features(features)
    elif feature_set in {"long_buy", "long_buy_post_order"}:
        long_features = {
            "lifetime_gmv_per_day": _safe_ratio(features["gmv_sum_all"], features["tenure_days"]),
            "buyday_rate_all": _safe_ratio(features["all_days_buy"], features["active_days_all"]),
            "recency_buy_over_tenure": _safe_ratio(features["recency_to_ord_days"], features["tenure_days"]),
            "recency_buy_over_buydays": _safe_ratio(features["recency_to_ord_days"], features["all_days_buy"]),
        }
        for days in LONG_WINDOWS:
            long_features[f"w{days}_buyday_rate"] = _safe_ratio(
                features[f"w{days}_days_buy"],
                features[f"active_days_{days if days != 365 else 120}d"] if days in WINDOWS else pd.Series(days, index=features.index),
            )
            long_features[f"w{days}_avg_order_value"] = _safe_ratio(
                features[f"w{days}_gmv"],
                features[f"w{days}_orders"],
            )
            long_features[f"w{days}_orders_per_buyday"] = _safe_ratio(
                features[f"w{days}_orders"],
                features[f"w{days}_days_buy"],
            )
            long_features[f"w{days}_gmv_per_buyday"] = _safe_ratio(
                features[f"w{days}_gmv"],
                features[f"w{days}_days_buy"],
            )
            long_features[f"w{days}_orders_per_day"] = features[f"w{days}_orders"] / days
            long_features[f"w{days}_gmv_per_day"] = features[f"w{days}_gmv"] / days
        features = pd.concat([features, pd.DataFrame(long_features, index=features.index)], axis=1)
        if feature_set == "long_buy_post_order":
            features = _add_post_order_features(features, cutoff_date)

    features = features.sort_values("user_id")
    features.to_parquet(cache_path, index=False)
    return features.set_index("user_id")


if __name__ == "__main__":
    from src.config import CUTOFF_TEST

    if CUTOFF_TEST is None:
        raise SystemExit("Передайте cutoff через build_features(cutoff_date) или задайте CUTOFF_TEST")
    build_features(CUTOFF_TEST)
