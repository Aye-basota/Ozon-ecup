import duckdb
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED


DEFAULT_FEATURE_SET = "recency"
FEATURE_SETS = {"recency", "long_buy_post_order", "behavior_v1"}

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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = numerator.to_numpy(dtype="float64", copy=False)
    denominator_values = denominator.to_numpy(dtype="float64", copy=False)
    result = np.divide(
        numerator_values,
        denominator_values,
        out=np.zeros_like(numerator_values, dtype="float64"),
        where=denominator_values != 0,
    )
    return pd.Series(result, index=numerator.index)


def _fill_no_history_users(features: pd.DataFrame) -> pd.DataFrame:
    features = features.fillna(0)
    if "active_days_all" not in features.columns:
        return features

    no_history = features["active_days_all"] == 0
    if not no_history.any():
        return features

    feature_cols = [col for col in features.columns if col != "user_id"]
    sentinel_cols = [
        col for col in feature_cols
        if col.startswith("recency") or col == "first_buy_age_days"
    ]

    features.loc[no_history, feature_cols] = 0
    features.loc[no_history, sentinel_cols] = 9999
    return features


def _with_all_users(features: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    users = pd.DataFrame({"user_id": df_raw["user_id"].unique()})
    features = users.merge(features, on="user_id", how="left")
    return _fill_no_history_users(features)


def _query_df(query: str, df_raw: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    duckdb_tmp = DATA_PROCESSED / "duckdb_tmp"
    duckdb_tmp.mkdir(parents=True, exist_ok=True)

    if not pd.api.types.is_datetime64_any_dtype(df_raw["event_date"]):
        df_raw = df_raw.copy()
        df_raw["event_date"] = pd.to_datetime(df_raw["event_date"])

    with duckdb.connect() as con:
        con.register("raw_events", df_raw)
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET temp_directory = ?", [str(duckdb_tmp)])
        return con.execute(
            query.replace("read_parquet($path)", "raw_events"),
            {"cutoff": cutoff_date},
        ).fetchdf()


def _add_post_order_features(features: pd.DataFrame, df_raw: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    query = """
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
    post_order = _query_df(query, df_raw, cutoff_date)

    features = features.merge(post_order, on="user_id", how="left").fillna(0)
    has_order = features["recency_to_ord_days"] < 9999
    derived = {
        "has_activity_after_last_order": (features["post_order_active_days"] > 0).astype("int8"),
        "has_search_after_last_order": (features["post_order_search_days"] > 0).astype("int8"),
        "has_cart_after_last_order": (features["post_order_cart_days"] > 0).astype("int8"),
        "no_activity_after_last_order": (
            (features["recency_to_ord_days"] < 9999)
            & (features["post_order_active_days"] == 0)
        ).astype("int8"),
        "last_search_after_last_order": (
            (features["recency_to_ord_days"] < 9999)
            & (features["recency_search_days"] < features["recency_to_ord_days"])
        ).astype("int8"),
        "last_cart_after_last_order": (
            (features["recency_to_ord_days"] < 9999)
            & (features["recency_to_cart_days"] < features["recency_to_ord_days"])
        ).astype("int8"),
        "days_last_order_to_last_search": (
            features["recency_to_ord_days"] - features["recency_search_days"]
        ).where(has_order, 0),
        "days_last_order_to_last_cart": (
            features["recency_to_ord_days"] - features["recency_to_cart_days"]
        ).where(has_order, 0),
        "post_order_active_share": _safe_ratio(
            features["post_order_active_days"],
            features["recency_to_ord_days"],
        ),
        "post_order_search_share_all": _safe_ratio(
            features["post_order_searches_sum"],
            features["searches_sum_all"],
        ),
        "post_order_cart_share_all": _safe_ratio(
            features["post_order_cart_sum"],
            features["to_cart_sum_all"],
        ),
        "post_order_searches_per_day": _safe_ratio(
            features["post_order_searches_sum"],
            features["post_order_active_days"],
        ),
        "post_order_cart_per_day": _safe_ratio(
            features["post_order_cart_sum"],
            features["post_order_active_days"],
        ),
    }
    return pd.concat([features, pd.DataFrame(derived, index=features.index)], axis=1)


def _add_behavior_v1_features(features: pd.DataFrame, df_raw: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    query = """
        WITH base AS (
            SELECT *
            FROM read_parquet($path)
            WHERE event_date < CAST($cutoff AS DATE)
        ),
        daily AS (
            SELECT
                user_id,
                event_date,
                MAX(CASE WHEN search > 0 THEN 1 ELSE 0 END) AS has_search_day,
                MAX(CASE WHEN cat > 0 THEN 1 ELSE 0 END) AS has_cat_day,
                MAX(CASE WHEN to_cart > 0 THEN 1 ELSE 0 END) AS has_cart_day,
                MAX(CASE WHEN to_ord > 0 THEN 1 ELSE 0 END) AS has_order_day,
                SUM(searches) AS searches_day,
                SUM(to_cart) AS carts_day,
                SUM(to_ord) AS orders_day,
                SUM(gmv) AS gmv_day,
                SUM(gmv_search) AS gmv_search_day,
                SUM(gmv_cat) AS gmv_cat_day,
                SUM(search_to_cart) AS search_to_cart_day,
                SUM(search_to_ord) AS search_to_ord_day,
                SUM(cat_to_cart) AS cat_to_cart_day,
                SUM(cat_to_ord) AS cat_to_ord_day
            FROM base
            GROUP BY user_id, event_date
        ),
        active_gaps AS (
            SELECT
                user_id,
                DATE_DIFF('day', LAG(event_date) OVER (PARTITION BY user_id ORDER BY event_date), event_date)
                    AS gap_days
            FROM daily
        ),
        active_gap_stats AS (
            SELECT
                user_id,
                AVG(gap_days) AS b1_active_gap_mean_all,
                MEDIAN(gap_days) AS b1_active_gap_median_all,
                QUANTILE_CONT(gap_days, 0.90) AS b1_active_gap_p90_all,
                STDDEV_SAMP(gap_days) AS b1_active_gap_std_all,
                MAX(gap_days) AS b1_active_gap_max_all
            FROM active_gaps
            WHERE gap_days IS NOT NULL
            GROUP BY user_id
        ),
        order_days AS (
            SELECT *
            FROM daily
            WHERE orders_day > 0
        ),
        order_gaps AS (
            SELECT
                user_id,
                DATE_DIFF('day', LAG(event_date) OVER (PARTITION BY user_id ORDER BY event_date), event_date)
                    AS gap_days
            FROM order_days
        ),
        order_gap_stats AS (
            SELECT
                user_id,
                AVG(gap_days) AS b1_order_gap_mean_all,
                MEDIAN(gap_days) AS b1_order_gap_median_all,
                QUANTILE_CONT(gap_days, 0.90) AS b1_order_gap_p90_all,
                STDDEV_SAMP(gap_days) AS b1_order_gap_std_all,
                MIN(gap_days) AS b1_order_gap_min_all,
                MAX(gap_days) AS b1_order_gap_max_all
            FROM order_gaps
            WHERE gap_days IS NOT NULL
            GROUP BY user_id
        ),
        last_order AS (
            SELECT *
            FROM (
                SELECT
                    user_id,
                    event_date AS last_order_date,
                    gmv_day AS b1_last_order_day_gmv,
                    orders_day AS b1_last_order_day_orders,
                    searches_day AS b1_last_order_day_searches,
                    carts_day AS b1_last_order_day_carts,
                    has_search_day AS b1_last_order_had_search_same_day,
                    has_cart_day AS b1_last_order_had_cart_same_day,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_date DESC) AS rn
                FROM order_days
            )
            WHERE rn = 1
        ),
        pre_last_order AS (
            SELECT
                b.user_id,
                COUNT(DISTINCT b.event_date) AS b1_pre_last_order_active_days_7d,
                SUM(b.searches) AS b1_pre_last_order_searches_7d,
                SUM(b.to_cart) AS b1_pre_last_order_carts_7d
            FROM base AS b
            JOIN last_order AS lo USING (user_id)
            WHERE b.event_date >= lo.last_order_date - INTERVAL 7 DAY
              AND b.event_date < lo.last_order_date
            GROUP BY b.user_id
        ),
        post_last_order AS (
            SELECT
                d.user_id,
                SUM(d.gmv_search_day) AS b1_post_order_gmv_search_sum,
                SUM(d.gmv_cat_day) AS b1_post_order_gmv_cat_sum,
                SUM(d.search_to_cart_day) AS b1_post_order_search_to_cart_sum,
                SUM(d.search_to_ord_day) AS b1_post_order_search_to_ord_sum,
                SUM(d.cat_to_cart_day) AS b1_post_order_cat_to_cart_sum,
                SUM(d.cat_to_ord_day) AS b1_post_order_cat_to_ord_sum,
                SUM(CASE WHEN d.has_cart_day = 1 AND d.has_order_day = 0 THEN 1 ELSE 0 END)
                    AS b1_post_order_cart_no_order_days,
                SUM(CASE WHEN d.has_search_day = 1 AND d.has_cart_day = 0 THEN 1 ELSE 0 END)
                    AS b1_post_order_search_no_cart_days
            FROM daily AS d
            JOIN last_order AS lo USING (user_id)
            WHERE d.event_date > lo.last_order_date
            GROUP BY d.user_id
        ),
        order_ranked AS (
            SELECT
                user_id,
                gmv_day,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY gmv_day DESC) AS gmv_rank
            FROM order_days
        ),
        order_amount_stats AS (
            SELECT
                user_id,
                MEDIAN(gmv_day) AS b1_order_gmv_median_all,
                QUANTILE_CONT(gmv_day, 0.25) AS b1_order_gmv_p25_all,
                QUANTILE_CONT(gmv_day, 0.75) AS b1_order_gmv_p75_all,
                QUANTILE_CONT(gmv_day, 0.90) AS b1_order_gmv_p90_all,
                MAX(gmv_day) AS b1_order_gmv_max_all,
                AVG(gmv_day) AS b1_order_gmv_mean_all,
                STDDEV_SAMP(gmv_day) AS b1_order_gmv_std_all,
                SUM(gmv_day) AS b1_order_gmv_sum_all,
                COUNT(*) AS b1_order_days_count_all,
                SUM(CASE WHEN orders_day > 1 THEN 1 ELSE 0 END) AS b1_multi_order_day_count_all
            FROM order_days
            GROUP BY user_id
        ),
        order_top_stats AS (
            SELECT
                user_id,
                SUM(CASE WHEN gmv_rank = 1 THEN gmv_day ELSE 0 END) AS b1_order_gmv_top1_sum,
                SUM(CASE WHEN gmv_rank <= 3 THEN gmv_day ELSE 0 END) AS b1_order_gmv_top3_sum
            FROM order_ranked
            GROUP BY user_id
        ),
        behavior_windows AS (
            SELECT
                user_id,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 30 DAY AND has_search_day = 1 THEN 1 ELSE 0 END)
                    AS b1_search_days_30d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 1 THEN 1 ELSE 0 END)
                    AS b1_search_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 30 DAY AND has_cart_day = 1 THEN 1 ELSE 0 END)
                    AS b1_cart_days_30d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_cart_day = 1 THEN 1 ELSE 0 END)
                    AS b1_cart_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 30 DAY AND has_cart_day = 1 AND has_order_day = 0 THEN 1 ELSE 0 END)
                    AS b1_cart_no_order_days_30d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_cart_day = 1 AND has_order_day = 0 THEN 1 ELSE 0 END)
                    AS b1_cart_no_order_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 30 DAY AND has_search_day = 1 AND has_cart_day = 0 THEN 1 ELSE 0 END)
                    AS b1_search_no_cart_days_30d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 1 AND has_cart_day = 0 THEN 1 ELSE 0 END)
                    AS b1_search_no_cart_days_90d,
                MAX(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 30 DAY THEN searches_day ELSE 0 END)
                    AS b1_searches_day_max_30d,
                MAX(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY THEN searches_day ELSE 0 END)
                    AS b1_searches_day_max_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 1 AND has_cat_day = 1 THEN 1 ELSE 0 END)
                    AS b1_route_mixed_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 1 AND has_cat_day = 0 THEN 1 ELSE 0 END)
                    AS b1_route_search_only_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 0 AND has_cat_day = 1 THEN 1 ELSE 0 END)
                    AS b1_route_cat_only_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 1 AND has_cat_day = 1 THEN has_order_day ELSE 0 END)
                    AS b1_route_mixed_order_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 1 AND has_cat_day = 0 THEN has_order_day ELSE 0 END)
                    AS b1_route_search_only_order_days_90d,
                SUM(CASE WHEN event_date >= CAST($cutoff AS DATE) - INTERVAL 90 DAY AND has_search_day = 0 AND has_cat_day = 1 THEN has_order_day ELSE 0 END)
                    AS b1_route_cat_only_order_days_90d
            FROM daily
            GROUP BY user_id
        ),
        calendar_stats AS (
            SELECT
                user_id,
                SUM(CASE WHEN EXTRACT(DOW FROM event_date) IN (0, 6) THEN 1 ELSE 0 END)
                    AS b1_weekend_active_days_all,
                SUM(CASE WHEN EXTRACT(DOW FROM event_date) IN (0, 6) AND has_order_day = 1 THEN 1 ELSE 0 END)
                    AS b1_weekend_order_days_all,
                SUM(CASE WHEN EXTRACT(DOW FROM event_date) IN (0, 6) THEN gmv_day ELSE 0 END)
                    AS b1_weekend_gmv_all,
                SUM(CASE WHEN EXTRACT(DAY FROM event_date) BETWEEN 1 AND 7 AND has_order_day = 1 THEN 1 ELSE 0 END)
                    AS b1_month_start_order_days_all,
                SUM(CASE WHEN EXTRACT(DAY FROM event_date) BETWEEN 8 AND 20 AND has_order_day = 1 THEN 1 ELSE 0 END)
                    AS b1_month_mid_order_days_all,
                SUM(CASE WHEN EXTRACT(DAY FROM event_date) >= 21 AND has_order_day = 1 THEN 1 ELSE 0 END)
                    AS b1_month_end_order_days_all,
                SUM(CASE WHEN EXTRACT(DAY FROM event_date) BETWEEN 8 AND 12 AND has_order_day = 1 THEN 1 ELSE 0 END)
                    AS b1_payday_10_order_days_all,
                SUM(CASE WHEN EXTRACT(DAY FROM event_date) BETWEEN 23 AND 27 AND has_order_day = 1 THEN 1 ELSE 0 END)
                    AS b1_payday_25_order_days_all
            FROM daily
            GROUP BY user_id
        )
        SELECT
            d.user_id,
            ags.* EXCLUDE (user_id),
            ogs.* EXCLUDE (user_id),
            lo.* EXCLUDE (user_id, last_order_date, rn),
            plo.* EXCLUDE (user_id),
            pso.* EXCLUDE (user_id),
            oas.* EXCLUDE (user_id),
            ots.* EXCLUDE (user_id),
            bw.* EXCLUDE (user_id),
            cs.* EXCLUDE (user_id)
        FROM (SELECT DISTINCT user_id FROM daily) AS d
        LEFT JOIN active_gap_stats AS ags USING (user_id)
        LEFT JOIN order_gap_stats AS ogs USING (user_id)
        LEFT JOIN last_order AS lo USING (user_id)
        LEFT JOIN pre_last_order AS plo USING (user_id)
        LEFT JOIN post_last_order AS pso USING (user_id)
        LEFT JOIN order_amount_stats AS oas USING (user_id)
        LEFT JOIN order_top_stats AS ots USING (user_id)
        LEFT JOIN behavior_windows AS bw USING (user_id)
        LEFT JOIN calendar_stats AS cs USING (user_id)
    """
    behavior = _query_df(query, df_raw, cutoff_date)

    features = features.merge(behavior, on="user_id", how="left").fillna(0)
    order_gap_cv = _safe_ratio(
        features["b1_order_gap_std_all"],
        features["b1_order_gap_mean_all"],
    )
    active_gap_cv = _safe_ratio(
        features["b1_active_gap_std_all"],
        features["b1_active_gap_mean_all"],
    )
    search_intent_score = (
        np.log1p(features["searches_sum_30d"])
        * _safe_ratio(features["b1_search_days_30d"], features["active_days_30d"])
        / (features["recency_search_days"] + 1)
    )
    cart_pressure_30d = (
        np.log1p(features["to_cart_sum_30d"])
        / (features["recency_to_cart_days"] + 1)
    )
    cart_pressure_90d = (
        np.log1p(features["to_cart_sum_120d"])
        / (features["recency_to_cart_days"] + 1)
    )
    cart_friction_score = _safe_ratio(
        features["b1_cart_no_order_days_90d"],
        features["b1_cart_days_90d"],
    ) * np.log1p(features["to_cart_sum_120d"])
    order_regularity_score = _safe_ratio(
        pd.Series(1.0, index=features.index),
        1.0 + order_gap_cv,
    )
    expected_overdue_ratio = _safe_ratio(
        features["recency_to_ord_days"] - features["b1_order_gap_median_all"],
        features["b1_order_gap_p90_all"] + 1,
    )
    derived = {
        "b1_order_gap_cv_all": order_gap_cv,
        "b1_active_gap_cv_all": active_gap_cv,
        "b1_order_gap_last_over_mean": _safe_ratio(
            features["recency_to_ord_days"],
            features["b1_order_gap_mean_all"],
        ),
        "b1_order_gap_last_over_median": _safe_ratio(
            features["recency_to_ord_days"],
            features["b1_order_gap_median_all"],
        ),
        "b1_order_cycle_phase": _safe_ratio(
            features["recency_to_ord_days"],
            features["b1_order_gap_median_all"] + 1,
        ),
        "b1_expected_next_order_overdue_days": (
            features["recency_to_ord_days"] - features["b1_order_gap_median_all"]
        ),
        "b1_expected_next_order_overdue_ratio": expected_overdue_ratio,
        "b1_order_regularity_score": order_regularity_score,
        "b1_active_gap_last_over_mean": _safe_ratio(
            features["recency_days"],
            features["b1_active_gap_mean_all"],
        ),
        "b1_searches_per_search_day_30d": _safe_ratio(
            features["searches_sum_30d"],
            features["b1_search_days_30d"],
        ),
        "b1_searches_per_search_day_90d": _safe_ratio(
            features["searches_sum_120d"],
            features["b1_search_days_90d"],
        ),
        "b1_search_day_share_active_30d": _safe_ratio(
            features["b1_search_days_30d"],
            features["active_days_30d"],
        ),
        "b1_search_day_share_active_90d": _safe_ratio(
            features["b1_search_days_90d"],
            features["active_days_120d"],
        ),
        "b1_search_spike_ratio_30d": _safe_ratio(
            features["b1_searches_day_max_30d"],
            _safe_ratio(features["searches_sum_30d"], features["b1_search_days_30d"]),
        ),
        "b1_search_spike_ratio_90d": _safe_ratio(
            features["b1_searches_day_max_90d"],
            _safe_ratio(features["searches_sum_120d"], features["b1_search_days_90d"]),
        ),
        "b1_search_intent_score": search_intent_score,
        "b1_cart_no_order_share_30d": _safe_ratio(
            features["b1_cart_no_order_days_30d"],
            features["b1_cart_days_30d"],
        ),
        "b1_cart_no_order_share_90d": _safe_ratio(
            features["b1_cart_no_order_days_90d"],
            features["b1_cart_days_90d"],
        ),
        "b1_cart_pressure_30d": cart_pressure_30d,
        "b1_cart_pressure_90d": cart_pressure_90d,
        "b1_cart_friction_score": cart_friction_score,
        "b1_route_mixed_day_share_90d": _safe_ratio(
            features["b1_route_mixed_days_90d"],
            features["active_days_120d"],
        ),
        "b1_route_search_only_to_order_rate_90d": _safe_ratio(
            features["b1_route_search_only_order_days_90d"],
            features["b1_route_search_only_days_90d"],
        ),
        "b1_route_cat_only_to_order_rate_90d": _safe_ratio(
            features["b1_route_cat_only_order_days_90d"],
            features["b1_route_cat_only_days_90d"],
        ),
        "b1_route_mixed_to_order_rate_90d": _safe_ratio(
            features["b1_route_mixed_order_days_90d"],
            features["b1_route_mixed_days_90d"],
        ),
        "b1_order_gmv_iqr_all": (
            features["b1_order_gmv_p75_all"] - features["b1_order_gmv_p25_all"]
        ),
        "b1_order_gmv_max_over_median": _safe_ratio(
            features["b1_order_gmv_max_all"],
            features["b1_order_gmv_median_all"],
        ),
        "b1_order_gmv_last_over_median": _safe_ratio(
            features["b1_last_order_day_gmv"],
            features["b1_order_gmv_median_all"],
        ),
        "b1_order_gmv_last_over_p75": _safe_ratio(
            features["b1_last_order_day_gmv"],
            features["b1_order_gmv_p75_all"],
        ),
        "b1_order_gmv_cv_all": _safe_ratio(
            features["b1_order_gmv_std_all"],
            features["b1_order_gmv_mean_all"],
        ),
        "b1_order_gmv_top1_share": _safe_ratio(
            features["b1_order_gmv_top1_sum"],
            features["b1_order_gmv_sum_all"],
        ),
        "b1_order_gmv_top3_share": _safe_ratio(
            features["b1_order_gmv_top3_sum"],
            features["b1_order_gmv_sum_all"],
        ),
        "b1_multi_order_day_share_all": _safe_ratio(
            features["b1_multi_order_day_count_all"],
            features["b1_order_days_count_all"],
        ),
        "b1_last_order_day_is_multi": (
            features["b1_last_order_day_orders"] > 1
        ).astype("int8"),
        "b1_weekend_active_share_all": _safe_ratio(
            features["b1_weekend_active_days_all"],
            features["active_days_all"],
        ),
        "b1_weekend_order_share_all": _safe_ratio(
            features["b1_weekend_order_days_all"],
            features["b1_order_days_count_all"],
        ),
        "b1_weekend_gmv_share_all": _safe_ratio(
            features["b1_weekend_gmv_all"],
            features["gmv_sum_all"],
        ),
        "b1_month_start_order_share_all": _safe_ratio(
            features["b1_month_start_order_days_all"],
            features["b1_order_days_count_all"],
        ),
        "b1_month_end_order_share_all": _safe_ratio(
            features["b1_month_end_order_days_all"],
            features["b1_order_days_count_all"],
        ),
        "b1_payday_order_share_all": _safe_ratio(
            features["b1_payday_10_order_days_all"] + features["b1_payday_25_order_days_all"],
            features["b1_order_days_count_all"],
        ),
        "b1_post_order_cart_no_order_share": _safe_ratio(
            features["b1_post_order_cart_no_order_days"],
            features["post_order_cart_days"],
        ),
        "b1_post_order_intent_without_buy_score": (
            np.log1p(features["post_order_searches_sum"] + features["post_order_cart_sum"])
            * (features["b1_post_order_search_to_ord_sum"] == 0).astype("int8")
        ),
        "b1_pre_last_order_search_lift_7d": _safe_ratio(
            features["b1_pre_last_order_searches_7d"],
            features["searches_sum_all"] / (features["tenure_days"] / 7 + 1),
        ),
        "b1_pre_last_order_cart_lift_7d": _safe_ratio(
            features["b1_pre_last_order_carts_7d"],
            features["to_cart_sum_all"] / (features["tenure_days"] / 7 + 1),
        ),
        "b1_fresh_buyer_high_intent_score": (
            (features["recency_to_ord_days"] <= 14).astype("int8")
            * (search_intent_score + cart_pressure_30d)
        ),
        "b1_stale_buyer_high_intent_score": (
            (features["recency_to_ord_days"] > 30).astype("int8")
            * (features["recency_to_ord_days"] < 9999).astype("int8")
            * (search_intent_score + cart_pressure_30d)
        ),
        "b1_cart_heavy_no_order_score": (
            cart_friction_score * np.log1p(features["to_cart_sum_30d"])
        ),
        "b1_regular_buyer_overdue_score": (
            order_regularity_score * np.maximum(expected_overdue_ratio, 0)
        ),
        "b1_silent_high_value_score": (
            np.log1p(features["gmv_sum_all"])
            * (features["recency_days"] > 30).astype("int8")
            * (features["recency_to_ord_days"] < 9999).astype("int8")
        ),
        "b1_post_order_browser_value_score": (
            (features["has_activity_after_last_order"] > 0).astype("int8")
            * np.log1p(features["b1_order_gmv_median_all"])
        ),
    }
    return pd.concat([features, pd.DataFrame(derived, index=features.index)], axis=1)


def build_features(df_raw: pd.DataFrame, cutoff_date: str, feature_set: str = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    cutoff_date = pd.to_datetime(cutoff_date).date().isoformat()
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set}; expected one of {sorted(FEATURE_SETS)}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_PROCESSED / f"team_features_{feature_set}_{cutoff_date}.parquet"
    legacy_cache_path = DATA_PROCESSED / f"features_{feature_set}_{cutoff_date}.parquet"
    if not cache_path.exists() and legacy_cache_path.exists():
        cache_path = legacy_cache_path
    if cache_path.exists():
        features = pd.read_parquet(cache_path)
        features = _with_all_users(features, df_raw)
        return features.sort_values("user_id").set_index("user_id")

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
        select_parts.append(
            f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) AS active_days_{days}d"
        )
        for col in BASE_COLS:
            select_parts.append(
                f"SUM(CASE WHEN {condition} THEN {col} ELSE 0 END) AS {col}_sum_{days}d"
            )

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

    if feature_set in {"long_buy_post_order", "behavior_v1"}:
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
    features = _query_df(query, df_raw, cutoff_date).fillna(0)
    features = _with_all_users(features, df_raw)

    for days in WINDOWS:
        features[f"gmv_per_active_day_{days}d"] = _safe_ratio(
            features[f"gmv_sum_{days}d"],
            features[f"active_days_{days}d"],
        )
        features[f"orders_per_active_day_{days}d"] = _safe_ratio(
            features[f"to_ord_sum_{days}d"],
            features[f"active_days_{days}d"],
        )

    if feature_set in {"long_buy_post_order", "behavior_v1"}:
        long_features = {
            "lifetime_gmv_per_day": _safe_ratio(features["gmv_sum_all"], features["tenure_days"]),
            "buyday_rate_all": _safe_ratio(features["all_days_buy"], features["active_days_all"]),
            "recency_buy_over_tenure": _safe_ratio(
                features["recency_to_ord_days"],
                features["tenure_days"],
            ),
            "recency_buy_over_buydays": _safe_ratio(
                features["recency_to_ord_days"],
                features["all_days_buy"],
            ),
        }
        for days in LONG_WINDOWS:
            day_index = pd.Series(days, index=features.index)
            long_features[f"w{days}_buyday_rate"] = _safe_ratio(
                features[f"w{days}_days_buy"],
                day_index,
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
        features = _add_post_order_features(features, df_raw, cutoff_date)
        if feature_set == "behavior_v1":
            features = _add_behavior_v1_features(features, df_raw, cutoff_date)

    features = _fill_no_history_users(features)
    features = features.sort_values("user_id")
    features.to_parquet(cache_path, index=False)
    return features.set_index("user_id")
