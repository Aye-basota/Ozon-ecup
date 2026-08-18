import pandas as pd
import numpy as np

WINDOWS = [7, 14, 30, 60, 90, 180, 365]


def build_features(df_raw, cutoff_date):
    df = df_raw.copy()
    cutoff_date = pd.to_datetime(cutoff_date)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df_history = df[df['event_date'] < cutoff_date]

    df_users = pd.DataFrame({
        'user_id': df_raw['user_id'].unique()
    })

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
            df_users[f'w{window}_searches'] / df_users[f'w{window}_days_present'].clip(lower=1)
        )

        df_users[f'w{window}_buyday_rate'] = (
            df_users[f'w{window}_days_buy'] / df_users[f'w{window}_days_present'].clip(lower=1)
        )

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
        df_history[df_history['to_ord'] > 0].groupby('user_id')['event_date'].nunique()
    ).fillna(0)

    first_seen = df_history.groupby('user_id')['event_date'].min()
    first_buy = df_history[df_history['to_ord'] > 0].groupby('user_id')['event_date'].min()
    last_any = df_history.groupby('user_id')['event_date'].max()
    last_buy = df_history[df_history['to_ord'] > 0].groupby('user_id')['event_date'].max()
    last_cart = df_history[df_history['to_cart'] > 0].groupby('user_id')['event_date'].max()
    last_cat = df_history[df_history['cat'] > 0].groupby('user_id')['event_date'].max()

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

    df_users['rec_cat'] = (
        cutoff_date - df_users['user_id'].map(last_cat)
    ).dt.days.fillna(999)

    df_users['lifetime_gmv_per_day'] = (
        df_users['all_gmv'] / df_users['tenure'].clip(lower=1)
    )

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

    return df_users

def get_target(df_raw: pd.DataFrame, cutoff_date: pd.Timestamp, window: int):
    df = df_raw.copy()
    window = pd.to_timedelta(window, unit='D')

    start = cutoff_date
    end = cutoff_date + window
    df['event_date'] = pd.to_datetime(df['event_date'])
    df_history = df[(df['event_date'] >= start)
                    & (df['event_date'] < end)]
 
    df_target = pd.DataFrame({
        'user_id': df['user_id'].unique()
    })
    df_target['target_gmv'] = df_target['user_id'].map(
        df_history.groupby('user_id')['gmv'].sum()
    ).fillna(0)

    return df_target

def build_df(df_raw, cutoff_date, window):
    X = build_features(df_raw, cutoff_date)
    target_df = get_target(df_raw, cutoff_date, window)

    df_processed = X.merge(target_df, on='user_id', how='left')
    return df_processed