# is there any extra rule, e.g. min #active days or min gmv?

## Catalogue metadata

- **Catalogue ID:** `eda__e06_panel_logseason`
- **Namespace:** `eda`
- **Experiment ID:** `e06_panel_logseason`
- **Original source:** `research/eda/e06_panel_logseason.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** EDA experiment/script
- **Model:** Unknown / not recoverable from repository history
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** print(f"  => at the YoY analogue the log-level RISES by {yoy['dm'][0]:+.4f} vs the trailing window,")
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e06_panel_logseason

Original script: `research/eda/e06_panel_logseason.py`

```python
"""Stage 5: verify the panel rule exactly; decompose seasonality in LOG space (= metric space)."""
import datetime as dt

import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw"
OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"
END = dt.date(2026, 2, 13)

span = pl.read_parquet(OUT + r"\user_span.parquet")
act = pl.scan_parquet(RAW + r"\train.parquet").select("user_id", "event_date", "gmv").collect()
all_u = span.select("user_id").sort("user_id")
N = all_u.height

print("=" * 100)
print("A. PANEL RULE — exact integer verification")
print("=" * 100)
print(f"users with last_d  >= 2026-01-15 : {(span['last_d'] >= dt.date(2026,1,15)).sum():,} / {N:,}")
print(f"users with first_d <= 2025-12-15 : {(span['first_d'] <= dt.date(2025,12,15)).sum():,} / {N:,}")
print(f"  => RULE: active in [T-29..T] AND first activity <= T-60,  T = {END}")

# is there any extra rule, e.g. min #active days or min gmv?
print(f"\nmin rows per user = {span['n_rows'].min()}, users with 0 gmv ever = "
      f"{(span['gmv_tot']==0).sum():,}  -> no min-purchase filter")

# max inactivity gap per user (does the panel forbid long gaps?)
gaps = (act.select("user_id", "event_date").sort(["user_id", "event_date"])
        .with_columns((pl.col("event_date").diff().dt.total_days()
                       .over("user_id")).alias("gap"))
        .group_by("user_id").agg(pl.col("gap").max().alias("max_gap")))
print(f"\nmax inactivity gap: p50={gaps['max_gap'].median()} p95={gaps['max_gap'].quantile(.95)} "
      f"p99={gaps['max_gap'].quantile(.99)} max={gaps['max_gap'].max()}  -> no max-gap rule")

print("\nexact activity counts in (T, T+30] for late cutoffs:")
for c in ["2025-11-15", "2025-11-30", "2025-12-15", "2025-12-16", "2025-12-31", "2026-01-14"]:
    T = dt.date.fromisoformat(c)
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=30)
    n = act.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))["user_id"].n_unique()
    ov = max(0, (min(b, END) - max(a, dt.date(2026, 1, 15))).days + 1)
    print(f"  T={c}  active={n:,} ({100*n/N:6.3f}%)  overlap with guaranteed window = {ov} days")

print()
print("=" * 100)
print("B. SEASONALITY IN LOG SPACE  (m = mean log1p(y); this is what RMSLE cares about)")
print("=" * 100)
gmv_pos = act.filter(pl.col("gmv") > 0)


def wvec(a, b):
    w = (gmv_pos.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
         .group_by("user_id").agg(pl.col("gmv").sum().alias("v")))
    return (all_u.join(w, on="user_id", how="left")
            .with_columns(pl.col("v").fill_null(0.0)).sort("user_id")["v"].to_numpy())


rows = []
T = dt.date(2025, 1, 30)
while T <= dt.date(2026, 1, 14):
    y = wvec(T + dt.timedelta(days=1), T + dt.timedelta(days=30))
    x = wvec(T - dt.timedelta(days=29), T)
    zy, zx = np.log1p(y), np.log1p(x)
    rows.append(dict(cutoff=T, m_y=zy.mean(), m_x=zx.mean(), dm=zy.mean() - zx.mean(),
                     p_y=(y > 0).mean(), p_x=(x > 0).mean(),
                     mu_y=zy[y > 0].mean(), mu_x=zx[x > 0].mean(),
                     ratio_raw=y.sum() / x.sum()))
    T += dt.timedelta(days=14)
df = pl.DataFrame(rows)
print(f"{'cutoff':12s} {'m_x':>7s} {'m_y':>7s} {'dm':>7s} | {'P(x>0)':>7s} {'P(y>0)':>7s} {'dP':>7s} | "
      f"{'mu_x':>6s} {'mu_y':>6s} {'dmu':>6s} | {'rawratio':>8s}")
for r in df.iter_rows(named=True):
    star = "  <== YoY ANALOGUE" if r["cutoff"] == dt.date(2025, 2, 13) else ""
    star = "  <== TEST-LIKE(contaminated)" if r["cutoff"] >= dt.date(2025, 12, 16) else star
    print(f"{str(r['cutoff']):12s} {r['m_x']:7.4f} {r['m_y']:7.4f} {r['dm']:+7.4f} | "
          f"{r['p_x']:7.4f} {r['p_y']:7.4f} {r['p_y']-r['p_x']:+7.4f} | "
          f"{r['mu_x']:6.3f} {r['mu_y']:6.3f} {r['mu_y']-r['mu_x']:+6.3f} | {r['ratio_raw']:8.4f}{star}")

print()
clean = df.filter(pl.col("cutoff") <= dt.date(2025, 12, 15))
print(f"CLEAN cutoffs (T<=2025-12-15): dm mean={clean['dm'].mean():+.4f} median={clean['dm'].median():+.4f} "
      f"std={clean['dm'].std():.4f}")
yoy = df.filter(pl.col("cutoff") == dt.date(2025, 2, 13))
print(f"YoY analogue T=2025-02-13:     dm={yoy['dm'][0]:+.4f}  dP={yoy['p_y'][0]-yoy['p_x'][0]:+.4f}  "
       f"dmu={yoy['mu_y'][0]-yoy['mu_x'][0]:+.4f}")
print(f"  => at the YoY analogue the log-level RISES by {yoy['dm'][0]:+.4f} vs the trailing window,")
print(f"     driven by dP={yoy['p_y'][0]-yoy['p_x'][0]:+.4f} (extensive) and "
      f"dmu={yoy['mu_y'][0]-yoy['mu_x'][0]:+.4f} (intensive)")

print()
print("=" * 100)
print("C. RE-APPLYING THE PANEL RULE AT HISTORICAL CUTOFFS (test-like sub-panels)")
print("=" * 100)
first_d = span.select("user_id", "first_d", "last_d").sort("user_id")
print(f"{'cutoff':12s} {'panel_n':>9s} {'%':>6s} {'P(y>0)|panel':>13s} {'P(y>0)|all':>11s} "
      f"{'m_y|panel':>10s} {'m_y|all':>9s}")
for c in ["2025-04-15", "2025-06-15", "2025-08-15", "2025-10-15", "2025-11-15", "2025-12-15"]:
    T = dt.date.fromisoformat(c)
    recent = (act.filter((pl.col("event_date") >= T - dt.timedelta(days=29))
                         & (pl.col("event_date") <= T)).select("user_id").unique())
    panel = (recent.join(first_d.filter(pl.col("first_d") <= T - dt.timedelta(days=60)),
                         on="user_id", how="inner").select("user_id"))
    y = wvec(T + dt.timedelta(days=1), T + dt.timedelta(days=30))
    ydf = all_u.with_columns(pl.Series("y", y))
    yp = ydf.join(panel, on="user_id", how="inner")["y"].to_numpy()
    print(f"{c:12s} {panel.height:9,} {100*panel.height/N:5.1f}% {(yp>0).mean():13.4f} "
          f"{(y>0).mean():11.4f} {np.log1p(yp).mean():10.4f} {np.log1p(y).mean():9.4f}")
print()
print("TEST panel at T=2026-02-13 is 100% of users by construction (rule was applied there).")

```
