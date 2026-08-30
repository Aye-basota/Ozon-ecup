"""Stage 4: exact panel-selection rule, seasonality decomposition, baseline RMSLE anchors."""
import datetime as dt

import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw"
OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"

span = pl.read_parquet(OUT + r"\user_span.parquet")
daily = pl.read_parquet(OUT + r"\daily.parquet")

print("=" * 100)
print("A. EXACT PANEL SELECTION RULE")
print("=" * 100)
print("first_d: min =", span["first_d"].min(), " max =", span["first_d"].max())
print("last_d : min =", span["last_d"].min(), " max =", span["last_d"].max())
END = dt.date(2026, 2, 13)
print(f"END - max(first_d) = {(END - span['first_d'].max()).days} days")
print(f"END - min(last_d)  = {(END - span['last_d'].min()).days} days")
print()
print("last_d daily counts, earliest 15:")
print(span.group_by("last_d").agg(pl.len().alias("n")).sort("last_d").head(15))
print("first_d daily counts, latest 15:")
print(span.group_by("first_d").agg(pl.len().alias("n")).sort("first_d").tail(15))
print()
print("first_buy: min =", span["first_buy"].min(), " max =", span["first_buy"].max())
print("last_buy : min =", span["last_buy"].min(), " max =", span["last_buy"].max())
print("n_rows: min =", span["n_rows"].min(), " -> is there a min-activity filter?")
print(span.group_by("n_rows").agg(pl.len().alias("n")).sort("n_rows").head(12))

print()
print("=" * 100)
print("B. SEASONALITY: ratio next30 / last30 of TOTAL GMV across the whole year")
print("=" * 100)
d = daily.sort("event_date")
dates = d["event_date"].to_list()
g = d["gmv"].to_numpy()
idx = {dd: i for i, dd in enumerate(dates)}


def wsum(a, b):
    ia, ib = idx.get(a), idx.get(b)
    if ia is None or ib is None:
        return None
    return g[ia:ib + 1].sum()


rows = []
T = dt.date(2025, 1, 30)
while T <= dt.date(2026, 1, 14):
    hist = wsum(T - dt.timedelta(days=29), T)
    tgt = wsum(T + dt.timedelta(days=1), T + dt.timedelta(days=30))
    if hist and tgt:
        rows.append((T, hist, tgt, tgt / hist))
    T += dt.timedelta(days=7)
rt = pl.DataFrame({"cutoff": [r[0] for r in rows], "hist30": [r[1] for r in rows],
                   "tgt30": [r[2] for r in rows], "ratio": [r[3] for r in rows]})
for r in rt.iter_rows(named=True):
    bar = "#" * int((r["ratio"] - 0.8) * 60)
    print(f"  T={r['cutoff']}  hist30={r['hist30']/1e6:6.2f}M tgt30={r['tgt30']/1e6:6.2f}M "
          f"ratio={r['ratio']:.4f} {bar}")
print()
print(f"  median ratio over the year = {rt['ratio'].median():.4f}   mean = {rt['ratio'].mean():.4f}")
r_feb = rt.filter(pl.col("cutoff") == dt.date(2025, 2, 13))
print(f"  ratio at the YoY-analogue cutoff 2025-02-13 = "
      f"{wsum(dt.date(2025,2,14), dt.date(2025,3,15))/wsum(dt.date(2025,1,15), dt.date(2025,2,13)):.4f}")

# secular growth: same-window YoY
print()
print("  --- secular growth (trailing 30d GMV, month over month) ---")
for m in range(1, 14):
    a = dt.date(2025, 1, 1) + dt.timedelta(days=30 * (m - 1))
    b = a + dt.timedelta(days=29)
    s = wsum(a, b)
    if s:
        print(f"    [{a}..{b}] {s/1e6:6.3f}M")

print()
print("=" * 100)
print("C. BASELINE RMSLE ANCHORS AT HISTORICAL CUTOFFS")
print("=" * 100)
gmv = (pl.scan_parquet(RAW + r"\train.parquet").select("user_id", "event_date", "gmv")
       .filter(pl.col("gmv") > 0).collect())
all_users = span.select("user_id").sort("user_id")


def win(a, b):
    w = (gmv.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
         .group_by("user_id").agg(pl.col("gmv").sum().alias("v")))
    return (all_users.join(w, on="user_id", how="left")
            .with_columns(pl.col("v").fill_null(0.0)).sort("user_id")["v"].to_numpy())


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(p, 0))) ** 2)))


print(f"{'cutoff T':12s} {'zero%':>6s} {'persist30':>10s} {'persist*1.16':>12s} {'mean30of90':>11s} "
      f"{'const=med':>10s} {'const=opt':>10s} {'oracle_expmZ':>12s}")
cutoffs = ["2025-02-13", "2025-04-15", "2025-06-15", "2025-08-15", "2025-10-15",
           "2025-11-15", "2025-12-15", "2026-01-14"]
anchor = {}
for c in cutoffs:
    T = dt.date.fromisoformat(c)
    y = win(T + dt.timedelta(days=1), T + dt.timedelta(days=30))
    p30 = win(T - dt.timedelta(days=29), T)
    p90 = win(T - dt.timedelta(days=89), T) / 3.0
    z = np.log1p(y)
    # best constant in log space
    best_c = float(np.expm1(z.mean()))
    grid = np.linspace(0, 60, 601)
    consts = [rmsle(y, gc) for gc in grid]
    best_const = grid[int(np.argmin(consts))]
    anchor[c] = dict(persist=rmsle(y, p30), persist116=rmsle(y, p30 * 1.1628),
                     mean90=rmsle(y, p90), const=min(consts))
    print(f"{c:12s} {100*(y==0).mean():5.1f}% {rmsle(y,p30):10.4f} {rmsle(y,p30*1.1628):12.4f} "
          f"{rmsle(y,p90):11.4f} {rmsle(y,np.median(y)):10.4f} {min(consts):10.4f} "
          f"(bestconst={best_const:.2f}, expm1(mean z)={best_c:.2f})")

print()
print("=" * 100)
print("D. FORWARD-LOOKING SELECTION BIAS: activity rate in (T, T+30] by cutoff")
print("=" * 100)
act = (pl.scan_parquet(RAW + r"\train.parquet").select("user_id", "event_date").collect())


def act_rate(a, b):
    w = act.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b)).select("user_id").unique()
    return w.height / 250000


print(f"{'cutoff T':12s} {'P(any activity in next 30d)':>28s} {'P(gmv>0 in next 30d)':>22s}")
for c in cutoffs:
    T = dt.date.fromisoformat(c)
    y = win(T + dt.timedelta(days=1), T + dt.timedelta(days=30))
    print(f"{c:12s} {act_rate(T+dt.timedelta(days=1), T+dt.timedelta(days=30)):28.4f} "
          f"{(y>0).mean():22.4f}")
print()
print("NOTE: every user is guaranteed >=1 activity in [min(last_d)..2026-02-13] by construction,")
print("      so historical cutoffs cannot show true churn -> local CV is optimistic.")
