"""Stage 3: hidden structure — attribution lag, cohorts, user_id semantics, seasonality."""
import datetime as dt

import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw"
OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"

lf = pl.scan_parquet(RAW + r"\train.parquet")

print("=" * 100)
print("A. DELAYED ATTRIBUTION: orders/gmv on days with no search/cat session")
print("=" * 100)
a = lf.select(
    rows_no_search_but_sord=((pl.col("searches") == 0) & (pl.col("search_to_ord") > 0)).sum(),
    rows_no_search_but_scart=((pl.col("searches") == 0) & (pl.col("search_to_cart") > 0)).sum(),
    rows_no_cat_but_cord=((pl.col("cat") == 0) & (pl.col("cat_to_ord") > 0)).sum(),
    rows_no_cat_but_ccart=((pl.col("cat") == 0) & (pl.col("cat_to_cart") > 0)).sum(),
    gmv_when_no_session=pl.when((pl.col("searches") == 0) & (pl.col("cat") == 0))
                          .then(pl.col("gmv")).otherwise(0.0).sum(),
    gmv_total=pl.col("gmv").sum(),
    rows_no_session_any_gmv=((pl.col("searches") == 0) & (pl.col("cat") == 0) & (pl.col("gmv") > 0)).sum(),
    rows_no_session_at_all=((pl.col("searches") == 0) & (pl.col("cat") == 0)).sum(),
).collect()
for c in a.columns:
    print(f"  {c:28s} = {a[c][0]:,.2f}")
print(f"  -> share of GMV from 'no-session' days: {100*a['gmv_when_no_session'][0]/a['gmv_total'][0]:.3f}%")

print()
print("=" * 100)
print("B. USER-LEVEL SPAN: cohorts / first & last activity")
print("=" * 100)
span = (lf.group_by("user_id").agg(
    first_d=pl.col("event_date").min(),
    last_d=pl.col("event_date").max(),
    n_rows=pl.len(),
    n_gmv_days=(pl.col("gmv") > 0).sum(),
    gmv_tot=pl.col("gmv").sum(),
    first_buy=pl.when(pl.col("gmv") > 0).then(pl.col("event_date")).min(),
    last_buy=pl.when(pl.col("gmv") > 0).then(pl.col("event_date")).max(),
).collect())
span.write_parquet(OUT + r"\user_span.parquet")

fd = span["first_d"]
print("first_activity_date distribution:")
print(fd.describe())
fc = (span.with_columns(pl.col("first_d").dt.truncate("1mo").alias("mo"))
      .group_by("mo").agg(pl.len().alias("n_users")).sort("mo"))
print(fc)
print()
print("last_activity_date distribution (monthly):")
lc = (span.with_columns(pl.col("last_d").dt.truncate("1mo").alias("mo"))
      .group_by("mo").agg(pl.len().alias("n_users")).sort("mo"))
print(lc)
print()
print(f"users whose FIRST row is 2025-01-01 (left-censored): "
      f"{(fd == dt.date(2025,1,1)).sum():,} ({100*(fd==dt.date(2025,1,1)).mean():.2f}%)")
print(f"users whose LAST row is 2026-02-13: {(span['last_d'] == dt.date(2026,2,13)).sum():,}")
print(f"users who never bought: {(span['gmv_tot'] == 0).sum():,} ({100*(span['gmv_tot']==0).mean():.2f}%)")

print()
print("=" * 100)
print("C. USER_ID SEMANTICS — does the id encode registration order?")
print("=" * 100)
s2 = span.with_columns(
    first_doy=(pl.col("first_d") - dt.date(2025, 1, 1)).dt.total_days(),
    last_doy=(pl.col("last_d") - dt.date(2025, 1, 1)).dt.total_days(),
)
uid = s2["user_id"].to_numpy().astype(float)
for col in ["first_doy", "last_doy", "n_rows", "n_gmv_days"]:
    v = s2[col].to_numpy().astype(float)
    print(f"  corr(user_id, {col:10s}) pearson={np.corrcoef(uid, v)[0,1]:+.4f}  "
          f"spearman={np.corrcoef(np.argsort(np.argsort(uid)), np.argsort(np.argsort(v)))[0,1]:+.4f}")
gm = np.log1p(s2["gmv_tot"].to_numpy())
print(f"  corr(user_id, log1p(gmv_total))  pearson={np.corrcoef(uid, gm)[0,1]:+.4f}  "
      f"spearman={np.corrcoef(np.argsort(np.argsort(uid)), np.argsort(np.argsort(gm)))[0,1]:+.4f}")

# bucket user_id into deciles and look at stats
s3 = s2.with_columns((pl.col("user_id").rank("ordinal") * 20 // (s2.height + 1)).alias("uid_bucket"))
print()
print(s3.group_by("uid_bucket").agg(
    uid_min=pl.col("user_id").min(), uid_max=pl.col("user_id").max(),
    n=pl.len(), mean_first_doy=pl.col("first_doy").mean(),
    frac_first_jan1=(pl.col("first_d") == dt.date(2025, 1, 1)).mean(),
    mean_rows=pl.col("n_rows").mean(), mean_gmv=pl.col("gmv_tot").mean(),
    frac_never_bought=(pl.col("gmv_tot") == 0).mean(),
).sort("uid_bucket"))

# gaps in the user_id space
print()
uids = np.sort(s2["user_id"].to_numpy())
gaps = np.diff(uids)
print(f"user_id: n={len(uids)} min={uids[0]} max={uids[-1]} span={uids[-1]-uids[0]+1:,} "
      f"density={len(uids)/(uids[-1]-uids[0]+1):.4f}")
print(f"gap stats: mean={gaps.mean():.3f} med={np.median(gaps)} max={gaps.max()} "
      f"frac_gap_eq_1={(gaps==1).mean():.4f}")

print()
print("=" * 100)
print("D. DAILY TIME SERIES + SEASONALITY OF THE TARGET WINDOW (Feb14-Mar15)")
print("=" * 100)
daily = (lf.group_by("event_date").agg(
    gmv=pl.col("gmv").sum(), n_rows=pl.len(),
    n_users=pl.col("user_id").n_unique(),
    n_buyers=(pl.col("gmv") > 0).sum(),
    orders=pl.col("to_ord").sum(), carts=pl.col("to_cart").sum(),
    searches=pl.col("searches").sum(),
).sort("event_date").collect())
daily.write_parquet(OUT + r"\daily.parquet")

d = daily.with_columns(dow=pl.col("event_date").dt.weekday())
print("day-of-week effect (1=Mon):")
print(d.group_by("dow").agg(gmv_mean=pl.col("gmv").mean(), rows_mean=pl.col("n_rows").mean()).sort("dow"))

print()
print("--- 2025 analogue of the target window vs the 30d before it ---")


def rng(a, b):
    return daily.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))


w25_hist = rng(dt.date(2025, 1, 15), dt.date(2025, 2, 13))   # analogue of feature window
w25_tgt = rng(dt.date(2025, 2, 14), dt.date(2025, 3, 15))    # analogue of target window
w26_hist = rng(dt.date(2026, 1, 15), dt.date(2026, 2, 13))
for nm, w in [("2025 hist [01-15..02-13]", w25_hist), ("2025 TGT  [02-14..03-15]", w25_tgt),
              ("2026 hist [01-15..02-13]", w26_hist)]:
    print(f"  {nm}: gmv={w['gmv'].sum():14,.0f}  buyers_rows={w['n_buyers'].sum():10,}  "
          f"rows={w['n_rows'].sum():10,}  users={w['n_users'].sum():10,}")
r25 = w25_tgt["gmv"].sum() / w25_hist["gmv"].sum()
print(f"\n  >>> 2025 seasonal ratio  TGT/HIST = {r25:.4f}")
print(f"  >>> implied 2026 target total GMV = {w26_hist['gmv'].sum()*r25:,.0f} "
      f"(vs persistence {w26_hist['gmv'].sum():,.0f})")

print()
print("--- daily GMV around Feb-Mar 2025 (holiday spikes) ---")
sub = rng(dt.date(2025, 2, 1), dt.date(2025, 3, 20)).with_columns(
    (pl.col("gmv") / 1e6).round(3).alias("gmv_M"))
for r in sub.iter_rows(named=True):
    bar = "#" * int(r["gmv"] / 1e6 * 12)
    print(f"  {r['event_date']} {r['event_date'].strftime('%a')} {r['gmv_M']:7.3f}M {bar}")
