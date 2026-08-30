"""Stage 2: what IS sample_submit.predict? Test against every 30-day window."""
import datetime as dt

import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw"
OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"

ss = pl.read_parquet(OUT + r"\sample_submit.parquet").sort("user_id")
gmv = (pl.scan_parquet(RAW + r"\train.parquet")
       .select("user_id", "event_date", "gmv", "gmv_search", "gmv_cat")
       .filter(pl.col("gmv") > 0)
       .collect())
print("rows with gmv>0:", gmv.height)

END = dt.date(2026, 2, 13)
all_users = ss.select("user_id")


def window_sum(start, end, col="gmv"):
    w = (gmv.filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end))
         .group_by("user_id").agg(pl.col(col).sum().alias("y")))
    return all_users.join(w, on="user_id", how="left").with_columns(pl.col("y").fill_null(0.0)).sort("user_id")


def compare(name, df):
    y = df["y"].to_numpy()
    p = ss["predict"].to_numpy()
    exact = np.isclose(y, p, rtol=0, atol=1e-6).mean()
    close = np.isclose(y, p, rtol=1e-5, atol=1e-4).mean()
    r_log = np.corrcoef(np.log1p(y), np.log1p(p))[0, 1]
    rmsle = np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2))
    print(f"{name:34s} zero%={100*(y==0).mean():6.2f} mean={y.mean():9.3f} "
          f"med={np.median(y):8.3f} max={y.max():11.2f} | exact={100*exact:6.2f}% "
          f"close={100*close:6.2f}% rlog={r_log:6.4f} RMSLE={rmsle:7.4f}")
    return exact


print()
print("=" * 118)
print("TARGET SAMPLE_SUBMIT:                zero%= 45.93 mean=   84.034 "
      "med=   7.893 max=   53746.95")
print("=" * 118)

# Every 30-day window ending on a date near the end of history
for back in [0, 1, 2, 3, 7, 14, 21, 28, 30, 31, 45, 60, 90, 120, 180, 365]:
    end = END - dt.timedelta(days=back)
    start = end - dt.timedelta(days=29)
    compare(f"gmv 30d [{start}..{end}]", window_sum(start, end))

print()
print("--- other window lengths ending 2026-02-13 ---")
for L in [7, 14, 28, 30, 31, 60, 90, 180, 365, 409]:
    start = END - dt.timedelta(days=L - 1)
    compare(f"gmv {L:3d}d ending {END}", window_sum(start, END))

print()
print("--- gmv_search / gmv_cat only, last 30d ---")
compare("gmv_search 30d", window_sum(END - dt.timedelta(days=29), END, "gmv_search"))
compare("gmv_cat 30d", window_sum(END - dt.timedelta(days=29), END, "gmv_cat"))

print()
print("=" * 118)
print("HISTORICAL 30d-TARGET DISTRIBUTION AT VARIOUS CUTOFFS (what a real target looks like)")
print("=" * 118)
for cutoff in ["2025-04-15", "2025-07-15", "2025-10-15", "2025-11-15", "2025-12-15", "2026-01-14"]:
    T = dt.date.fromisoformat(cutoff)
    df = window_sum(T + dt.timedelta(days=1), T + dt.timedelta(days=30))
    y = df["y"].to_numpy()
    print(f"cutoff {cutoff} -> target window [{T+dt.timedelta(days=1)}..{T+dt.timedelta(days=30)}]  "
          f"zero%={100*(y==0).mean():6.2f} mean={y.mean():9.3f} med={np.median(y):8.3f} "
          f"q75={np.quantile(y,.75):9.3f} max={y.max():11.2f} sum={y.sum():,.0f}")

print()
print("=" * 118)
print("MONTHLY GMV TOTALS (is there trend/seasonality that explains submit mean=84?)")
print("=" * 118)
m = (gmv.with_columns(pl.col("event_date").dt.truncate("1mo").alias("mo"))
     .group_by("mo").agg(pl.col("gmv").sum().alias("gmv_sum"),
                         pl.col("user_id").n_unique().alias("buyers"),
                         pl.len().alias("n_rows"))
     .sort("mo"))
print(m)
