import pandas as pd
import numpy as np


WINDOWS = [7, 14, 30, 60, 90, 180, 365]
EPS = 1e-9


def build_features(df_raw, cutoff_date):
    df = df_raw.copy()
    cutoff_date = pd.to_datetime(cutoff_date)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df_history = df[df['event_date'] < cutoff_date]

    df_users = pd.DataFrame({
        'user_id': df_raw['user_id'].unique()
    })

    # Base aggregates by rolling windows.
    for window in WINDOWS:
        start_date = cutoff_date - pd.Timedelta(days=window)
        df_window = df_history[df_history['event_date'] >= start_date]
        df_window_buy = df_window[df_window['to_ord'] > 0].copy()
        df_window_cart = df_window[df_window['to_cart'] > 0]
        df_window_search = df_window[df_window['search'] > 0]
        df_window_cat = df_window[df_window['cat'] > 0]

        df_users[f'w{window}_gmv'] = df_users['user_id'].map(
            df_window.groupby('user_id')['gmv'].sum()
        ).fillna(0)

        df_users[f'w{window}_orders'] = df_users['user_id'].map(
            df_window.groupby('user_id')['to_ord'].sum()
        ).fillna(0)

        df_users[f'w{window}_carts'] = df_users['user_id'].map(
            df_window.groupby('user_id')['to_cart'].sum()
        ).fillna(0)

        df_users[f'w{window}_searches'] = df_users['user_id'].map(
            df_window.groupby('user_id')['searches'].sum()
        ).fillna(0)

        df_users[f'w{window}_days_present'] = df_users['user_id'].map(
            df_window.groupby('user_id')['event_date'].nunique()
        ).fillna(0)

        df_users[f'w{window}_days_buy'] = df_users['user_id'].map(
            df_window_buy.groupby('user_id')['event_date'].nunique()
        ).fillna(0)

        df_users[f'w{window}_days_cart'] = df_users['user_id'].map(
            df_window_cart.groupby('user_id')['event_date'].nunique()
        ).fillna(0)

        df_users[f'w{window}_days_search'] = df_users['user_id'].map(
            df_window_search.groupby('user_id')['event_date'].nunique()
        ).fillna(0)

        df_users[f'w{window}_days_cat'] = df_users['user_id'].map(
            df_window_cat.groupby('user_id')['event_date'].nunique()
        ).fillna(0)

        df_users[f'w{window}_lgmv'] = np.log1p(df_users[f'w{window}_gmv'])

        df_window_buy['log_gmv'] = np.log1p(df_window_buy['gmv'])

        df_users[f'w{window}_lgmv_mean'] = df_users['user_id'].map(
            df_window_buy.groupby('user_id')['log_gmv'].mean()
        ).fillna(0)

        df_users[f'w{window}_lgmv_std'] = df_users['user_id'].map(
            df_window_buy.groupby('user_id')['log_gmv'].std()
        ).fillna(0)

        df_users[f'w{window}_srch_per_day'] = (
            df_users[f'w{window}_searches']
            / df_users[f'w{window}_days_present'].clip(lower=1)
        )

        df_users[f'w{window}_buyday_rate'] = (
            df_users[f'w{window}_days_buy']
            / df_users[f'w{window}_days_present'].clip(lower=1)
        )

    # Derived ratios by the same rolling windows.
    df_window_features = pd.DataFrame(index=df_users.index)

    for window in WINDOWS:
        df_window_features[f'w{window}_aov'] = (
            df_users[f'w{window}_gmv']
            / df_users[f'w{window}_orders'].clip(lower=1)
        )

        df_window_features[f'w{window}_active_day_rate'] = (
            df_users[f'w{window}_days_present'] / window
        )

        df_window_features[f'w{window}_orders_per_active_day'] = (
            df_users[f'w{window}_orders']
            / df_users[f'w{window}_days_present'].clip(lower=1)
        )

        df_window_features[f'w{window}_orders_per_buy_day'] = (
            df_users[f'w{window}_orders']
            / df_users[f'w{window}_days_buy'].clip(lower=1)
        )

        df_window_features[f'w{window}_gmv_per_active_day'] = (
            df_users[f'w{window}_gmv']
            / df_users[f'w{window}_days_present'].clip(lower=1)
        )

        df_window_features[f'w{window}_search_to_cart'] = (
            df_users[f'w{window}_carts']
            / df_users[f'w{window}_searches'].clip(lower=1)
        )

        df_window_features[f'w{window}_cart_to_order'] = (
            df_users[f'w{window}_orders']
            / df_users[f'w{window}_carts'].clip(lower=1)
        )

        df_window_features[f'w{window}_search_to_order'] = (
            df_users[f'w{window}_orders']
            / df_users[f'w{window}_searches'].clip(lower=1)
        )

        df_window_features[f'w{window}_searches_per_order'] = (
            df_users[f'w{window}_searches']
            / df_users[f'w{window}_orders'].clip(lower=1)
        )

        df_window_features[f'w{window}_carts_per_order'] = (
            df_users[f'w{window}_carts']
            / df_users[f'w{window}_orders'].clip(lower=1)
        )

        df_window_features[f'w{window}_searches_per_cart'] = (
            df_users[f'w{window}_searches']
            / df_users[f'w{window}_carts'].clip(lower=1)
        )

        df_window_features[f'w{window}_carts_per_buy_day'] = (
            df_users[f'w{window}_carts']
            / df_users[f'w{window}_days_buy'].clip(lower=1)
        )

    df_users = pd.concat([df_users, df_window_features], axis=1).copy()

    # Lifetime aggregates.
    df_users['all_gmv'] = df_users['user_id'].map(
        df_history.groupby('user_id')['gmv'].sum()
    ).fillna(0)

    df_users['all_orders'] = df_users['user_id'].map(
        df_history.groupby('user_id')['to_ord'].sum()
    ).fillna(0)

    df_users['all_days_present'] = df_users['user_id'].map(
        df_history.groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    df_users['all_days_buy'] = df_users['user_id'].map(
        df_history[df_history['to_ord'] > 0]
        .groupby('user_id')['event_date']
        .nunique()
    ).fillna(0)

    df_users['all_aov'] = (
        df_users['all_gmv'] / df_users['all_orders'].clip(lower=1)
    )

    # First and last event dates.
    first_seen = df_history.groupby('user_id')['event_date'].min()
    first_buy = df_history[df_history['to_ord'] > 0].groupby('user_id')['event_date'].min()
    last_any = df_history.groupby('user_id')['event_date'].max()
    last_buy = df_history[df_history['to_ord'] > 0].groupby('user_id')['event_date'].max()
    last_cart = df_history[df_history['to_cart'] > 0].groupby('user_id')['event_date'].max()
    last_search = df_history[df_history['search'] > 0].groupby('user_id')['event_date'].max()
    last_cat = df_history[df_history['cat'] > 0].groupby('user_id')['event_date'].max()

    # User age and recency.
    df_users['tenure'] = (
        cutoff_date - df_users['user_id'].map(first_seen)
    ).dt.days.fillna(0)

    df_users['first_buy_age'] = (
        cutoff_date - df_users['user_id'].map(first_buy)
    ).dt.days.fillna(999)

    df_users['rec_any'] = (
        cutoff_date - df_users['user_id'].map(last_any)
    ).dt.days.fillna(999)

    df_users['rec_buy'] = (
        cutoff_date - df_users['user_id'].map(last_buy)
    ).dt.days.fillna(999)

    df_users['rec_cart'] = (
        cutoff_date - df_users['user_id'].map(last_cart)
    ).dt.days.fillna(999)

    df_users['rec_search'] = (
        cutoff_date - df_users['user_id'].map(last_search)
    ).dt.days.fillna(999)

    df_users['rec_cat'] = (
        cutoff_date - df_users['user_id'].map(last_cat)
    ).dt.days.fillna(999)

    df_users['days_to_first_buy'] = (
        df_users['user_id'].map(first_buy)
        - df_users['user_id'].map(first_seen)
    ).dt.days.fillna(999)

    # Lifetime status and density.
    df_users['has_ever_bought'] = (df_users['all_orders'] > 0).astype(int)

    df_users['is_repeat_buyer'] = (
        df_users['all_days_buy'] >= 2
    ).astype(int)

    df_users['lifetime_gmv_per_day'] = (
        df_users['all_gmv'] / df_users['tenure'].clip(lower=1)
    )

    df_users['orders_per_30d_lifetime'] = (
        df_users['all_orders'] / df_users['tenure'].clip(lower=1) * 30
    )

    df_users['purchase_density_since_first_buy'] = (
        df_users['all_orders'] / df_users['first_buy_age'].clip(lower=1) * 30
    )

    # Recent month as a share of lifetime history.
    df_users['w30_orders_share_of_lifetime'] = (
        df_users['w30_orders'] / df_users['all_orders'].clip(lower=1)
    )

    df_users['w30_gmv_share_of_lifetime'] = (
        df_users['w30_gmv'] / df_users['all_gmv'].clip(lower=1)
    )

    df_users['w30_active_days_share_of_lifetime'] = (
        df_users['w30_days_present']
        / df_users['all_days_present'].clip(lower=1)
    )

    # Trend from shorter to longer windows.
    df_users['orders_trend_7_30'] = (
        (df_users['w7_orders'] / 7)
        / ((df_users['w30_orders'] / 30) + EPS)
    )

    df_users['gmv_trend_7_30'] = (
        (df_users['w7_gmv'] / 7)
        / ((df_users['w30_gmv'] / 30) + EPS)
    )

    df_users['carts_trend_7_30'] = (
        (df_users['w7_carts'] / 7)
        / ((df_users['w30_carts'] / 30) + EPS)
    )

    df_users['searches_trend_7_30'] = (
        (df_users['w7_searches'] / 7)
        / ((df_users['w30_searches'] / 30) + EPS)
    )

    df_users['orders_trend_30_90'] = (
        (df_users['w30_orders'] / 30)
        / ((df_users['w90_orders'] / 90) + EPS)
    )

    df_users['gmv_trend_30_90'] = (
        (df_users['w30_gmv'] / 30)
        / ((df_users['w90_gmv'] / 90) + EPS)
    )

    # AOV changes.
    df_users['w7_aov_vs_w90_aov'] = (
        df_users['w7_aov'] / df_users['w90_aov'].clip(lower=1)
    )

    df_users['w30_aov_vs_all_aov'] = (
        df_users['w30_aov'] / df_users['all_aov'].clip(lower=1)
    )

    # Last 7 days against the previous 23 days.
    df_users['orders_prev_23d'] = df_users['w30_orders'] - df_users['w7_orders']
    df_users['gmv_prev_23d'] = df_users['w30_gmv'] - df_users['w7_gmv']
    df_users['searches_prev_23d'] = df_users['w30_searches'] - df_users['w7_searches']
    df_users['carts_prev_23d'] = df_users['w30_carts'] - df_users['w7_carts']
    df_users['days_present_prev_23d'] = df_users['w30_days_present'] - df_users['w7_days_present']

    df_users['orders_acceleration_7_vs_prev23'] = (
        (df_users['w7_orders'] / 7)
        / ((df_users['orders_prev_23d'].clip(lower=1) / 23) + EPS)
    )

    df_users['gmv_acceleration_7_vs_prev23'] = (
        (df_users['w7_gmv'] / 7)
        / ((df_users['gmv_prev_23d'].clip(lower=1) / 23) + EPS)
    )

    df_users['search_acceleration_7_vs_prev23'] = (
        (df_users['w7_searches'] / 7)
        / ((df_users['searches_prev_23d'].clip(lower=1) / 23) + EPS)
    )

    df_users['cart_acceleration_7_vs_prev23'] = (
        (df_users['w7_carts'] / 7)
        / ((df_users['carts_prev_23d'].clip(lower=1) / 23) + EPS)
    )

    # Current intent flags.
    df_users['recent_cart_no_buy_7d'] = (
        (df_users['w7_carts'] > 0)
        & (df_users['w7_orders'] == 0)
    ).astype(int)

    df_users['recent_search_no_buy_7d'] = (
        (df_users['w7_searches'] > 0)
        & (df_users['w7_orders'] == 0)
    ).astype(int)

    df_users['cart_reactivation_7d'] = (
        (df_users['w7_carts'] > 0)
        & (df_users['rec_buy'] > 30)
    ).astype(int)

    df_users['search_reactivation_7d'] = (
        (df_users['w7_searches'] > 0)
        & (df_users['days_present_prev_23d'] == 0)
        & (df_users['tenure'] > 30)
    ).astype(int)

    df_users['cart_vs_buy_recency_gap'] = (
        df_users['rec_buy'] - df_users['rec_cart']
    )

    df_users['search_vs_buy_recency_gap'] = (
        df_users['rec_buy'] - df_users['rec_search']
    )

    # Activity gaps and streaks.
    df_active_days = (
        df_history[['user_id', 'event_date']]
        .drop_duplicates()
        .sort_values(['user_id', 'event_date'])
        .copy()
    )
    df_active_days['gap'] = df_active_days.groupby('user_id')['event_date'].diff().dt.days

    df_buy_days = (
        df_history[df_history['to_ord'] > 0][['user_id', 'event_date']]
        .drop_duplicates()
        .sort_values(['user_id', 'event_date'])
        .copy()
    )
    df_buy_days['buygap'] = df_buy_days.groupby('user_id')['event_date'].diff().dt.days

    df_active_days['streak_break'] = df_active_days['gap'].ne(1).astype(int)
    df_active_days['streak_id'] = df_active_days.groupby('user_id')['streak_break'].cumsum()
    df_active_days['active_streak_len'] = (
        df_active_days.groupby(['user_id', 'streak_id']).cumcount() + 1
    )

    last_active_streak = df_active_days.groupby('user_id')['active_streak_len'].last()
    last_active_is_yesterday = df_users['rec_any'] == 1

    df_users['current_active_streak'] = (
        df_users['user_id'].map(last_active_streak).fillna(0)
    )
    df_users['current_active_streak'] = (
        df_users['current_active_streak'].where(last_active_is_yesterday, 0)
    )

    df_active_days_30 = df_active_days[
        df_active_days['event_date'] >= cutoff_date - pd.Timedelta(days=30)
    ].copy()
    df_active_days_30['gap_30'] = (
        df_active_days_30.groupby('user_id')['event_date'].diff().dt.days
    )
    df_active_days_30['streak_break_30'] = (
        df_active_days_30['gap_30'].ne(1).astype(int)
    )
    df_active_days_30['streak_id_30'] = (
        df_active_days_30.groupby('user_id')['streak_break_30'].cumsum()
    )
    df_active_days_30['active_streak_len_30'] = (
        df_active_days_30.groupby(['user_id', 'streak_id_30']).cumcount() + 1
    )
    df_users['max_active_streak_30d'] = df_users['user_id'].map(
        df_active_days_30.groupby('user_id')['active_streak_len_30'].max()
    ).fillna(0)

    # Regularity of user activity.
    df_users['gap_mean'] = df_users['user_id'].map(
        df_active_days.groupby('user_id')['gap'].mean()
    ).fillna(999)

    df_users['gap_std'] = df_users['user_id'].map(
        df_active_days.groupby('user_id')['gap'].std()
    ).fillna(0)

    df_users['gap_cv'] = (
        df_users['gap_std'] / df_users['gap_mean'].clip(lower=1)
    )

    df_users['buygap_mean'] = df_users['user_id'].map(
        df_buy_days.groupby('user_id')['buygap'].mean()
    ).fillna(999)

    df_users['buygap_std'] = df_users['user_id'].map(
        df_buy_days.groupby('user_id')['buygap'].std()
    ).fillna(0)

    df_users['buygap_cv'] = (
        df_users['buygap_std'] / df_users['buygap_mean'].clip(lower=1)
    )

    df_users['rec_over_gap'] = (
        df_users['rec_any'] / df_users['gap_mean'].clip(lower=1)
    )

    df_users['rec_over_buygap'] = (
        df_users['rec_buy'] / df_users['buygap_mean'].clip(lower=1)
    )

    # Recent burst of search/cart activity.
    df_users['search_burst_7_30'] = (
        df_users['w7_searches'] / df_users['w30_searches'].clip(lower=1)
    )

    df_users['cart_burst_7_30'] = (
        df_users['w7_carts'] / df_users['w30_carts'].clip(lower=1)
    )

    # Same calendar month one year ago.
    year_ago_date = cutoff_date - pd.DateOffset(years=1)
    year_ago_start = year_ago_date.replace(day=1).normalize()
    year_ago_end = year_ago_start + pd.offsets.MonthBegin(1)
    year_ago_days = (year_ago_end - year_ago_start).days

    df_year_ago = df_history[
        (df_history['event_date'] >= year_ago_start)
        & (df_history['event_date'] < year_ago_end)
    ]

    df_year_ago_buy = df_year_ago[df_year_ago['to_ord'] > 0].copy()
    df_year_ago_cart = df_year_ago[df_year_ago['to_cart'] > 0]
    df_year_ago_search = df_year_ago[df_year_ago['search'] > 0]
    df_year_ago_cat = df_year_ago[df_year_ago['cat'] > 0]

    df_users['has_data_year_ago'] = int(len(df_year_ago) > 0)

    df_users['has_user_data_year_ago'] = (
        df_users['user_id'].isin(df_year_ago['user_id']).astype(int)
    )

    df_users['year_ago_gmv'] = df_users['user_id'].map(
        df_year_ago.groupby('user_id')['gmv'].sum()
    ).fillna(0)

    df_users['year_ago_orders'] = df_users['user_id'].map(
        df_year_ago.groupby('user_id')['to_ord'].sum()
    ).fillna(0)

    df_users['year_ago_carts'] = df_users['user_id'].map(
        df_year_ago.groupby('user_id')['to_cart'].sum()
    ).fillna(0)

    df_users['year_ago_searches'] = df_users['user_id'].map(
        df_year_ago.groupby('user_id')['searches'].sum()
    ).fillna(0)

    df_users['year_ago_days_present'] = df_users['user_id'].map(
        df_year_ago.groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    df_users['year_ago_days_buy'] = df_users['user_id'].map(
        df_year_ago_buy.groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    df_users['year_ago_days_cart'] = df_users['user_id'].map(
        df_year_ago_cart.groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    df_users['year_ago_days_search'] = df_users['user_id'].map(
        df_year_ago_search.groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    df_users['year_ago_days_cat'] = df_users['user_id'].map(
        df_year_ago_cat.groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    df_users['year_ago_lgmv'] = np.log1p(df_users['year_ago_gmv'])

    df_year_ago_buy['log_gmv'] = np.log1p(df_year_ago_buy['gmv'])

    df_users['year_ago_lgmv_mean'] = df_users['user_id'].map(
        df_year_ago_buy.groupby('user_id')['log_gmv'].mean()
    ).fillna(0)

    df_users['year_ago_lgmv_std'] = df_users['user_id'].map(
        df_year_ago_buy.groupby('user_id')['log_gmv'].std()
    ).fillna(0)

    df_users['year_ago_has_buy'] = (
        df_users['year_ago_orders'] > 0
    ).astype(int)

    df_users['year_ago_aov'] = (
        df_users['year_ago_gmv']
        / df_users['year_ago_orders'].clip(lower=1)
    )

    df_users['year_ago_active_day_rate'] = (
        df_users['year_ago_days_present'] / year_ago_days
    )

    df_users['year_ago_buyday_rate'] = (
        df_users['year_ago_days_buy']
        / df_users['year_ago_days_present'].clip(lower=1)
    )

    df_users['year_ago_srch_per_day'] = (
        df_users['year_ago_searches']
        / df_users['year_ago_days_present'].clip(lower=1)
    )

    df_users['w30_gmv_vs_year_ago'] = (
        df_users['w30_gmv'] / df_users['year_ago_gmv'].clip(lower=1)
    )

    df_users['w30_orders_vs_year_ago'] = (
        df_users['w30_orders'] / df_users['year_ago_orders'].clip(lower=1)
    )

    df_users['w30_activity_vs_year_ago'] = (
        df_users['w30_days_present']
        / df_users['year_ago_days_present'].clip(lower=1)
    )

    return df_users


def get_target(df_raw: pd.DataFrame, cutoff_date: pd.Timestamp, window: int):
    df = df_raw.copy()
    window = pd.to_timedelta(window, unit='D')

    start = cutoff_date
    end = cutoff_date + window
    df['event_date'] = pd.to_datetime(df['event_date'])
    df_history = df[
        (df['event_date'] >= start)
        & (df['event_date'] < end)
    ]

    df_target = pd.DataFrame({
        'user_id': df['user_id'].unique()
    })

    df_target['target_gmv'] = df_target['user_id'].map(
        df_history.groupby('user_id')['gmv'].sum()
    ).fillna(0)

    df_target['target_gmv_above_zero'] = (
        df_target['target_gmv'] > 0
    ).astype(int)

    df_target['target_log_gmv'] = np.log1p(df_target['target_gmv'])

    return df_target


def build_df(df_raw, cutoff_date, window):
    X = build_features(df_raw, cutoff_date)
    target_df = get_target(df_raw, cutoff_date, window)

    df_processed = X.merge(target_df, on='user_id', how='left')
    return df_processed
