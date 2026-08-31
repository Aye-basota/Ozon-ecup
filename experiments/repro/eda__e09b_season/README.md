# e09b_season

## Catalogue metadata

- **Catalogue ID:** `eda__e09b_season`
- **Namespace:** `eda`
- **Experiment ID:** `e09b_season`
- **Original source:** `research/eda/e09b_season.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** EDA experiment/script
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** print("PLATFORM-LEVEL DAILY CALENDAR FACTOR for the target window (from 2025)")
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e09b_season

Original script: `research/eda/e09b_season.py`

```python
"""Stage 8b: isolate seasonality with a CONSTANT panel definition across the year."""
import datetime as dt
import time

import numpy as np
import polars as pl

import fe

t0 = time.time()
fe.load()

print("=" * 96)
print("SEASONALITY WITH A FIXED 1-BLOCK PANEL (constructible from 2025-02-13 onwards)")
print("=" * 96, flush=True)
rows = []
T = dt.date(2025, 2, 13)
while T <= dt.date(2025, 12, 15):
    u = fe.panel_users(T, 1)
    Y = fe.target(T, u)
    X = (fe.load().lazy()
         .filter((pl.col("event_date") > T - dt.timedelta(days=30)) & (pl.col("event_date") <= T))
         .group_by("user_id").agg(pl.col("gmv").sum().alias("x")).collect())
    d = (u.join(X, on="user_id", how="left").with_columns(pl.col("x").fill_null(0.0))
         .join(Y, on="user_id", how="inner"))
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    zx, zy = np.log1p(x), np.log1p(y)
    rows.append(dict(T=T, n=u.height, m_x=zx.mean(), m_y=zy.mean(), dm=zy.mean() - zx.mean(),
                     dP=float((y > 0).mean() - (x > 0).mean()),
                     dmu=float(zy[y > 0].mean() - zx[x > 0].mean()),
                     contaminated=(T + dt.timedelta(days=30)) >= dt.date(2025, 11, 16)))
    print(f"  {T}  n={u.height:7,}  m_x={zx.mean():.4f} m_y={zy.mean():.4f} "
          f"dm={zy.mean()-zx.mean():+.4f}  dP={(y>0).mean()-(x>0).mean():+.4f} "
          f"dmu={zy[y>0].mean()-zx[x>0].mean():+.4f}"
          f"{'   <== CONTAMINATED' if rows[-1]['contaminated'] else ''}"
          f"{'   <== YoY ANALOGUE' if T == dt.date(2025,2,13) else ''}", flush=True)
    T += dt.timedelta(days=14)

df = pl.DataFrame(rows)
cl = df.filter(~pl.col("contaminated"))
print()
print(f"clean cutoffs (n={cl.height}): dm mean={cl['dm'].mean():+.4f} median={cl['dm'].median():+.4f} "
      f"std={cl['dm'].std():.4f}  min={cl['dm'].min():+.4f} max={cl['dm'].max():+.4f}")
feb = df.filter(pl.col("T") == dt.date(2025, 2, 13))
print(f"YoY analogue  dm={feb['dm'][0]:+.4f}  =>  seasonal excess over median = "
      f"{feb['dm'][0]-cl['dm'].median():+.4f}")
print(f"              dP={feb['dP'][0]:+.4f} (median {cl['dP'].median():+.4f}), "
      f"dmu={feb['dmu'][0]:+.4f} (median {cl['dmu'].median():+.4f})")

print()
print("=" * 96)
print("PLATFORM-LEVEL DAILY CALENDAR FACTOR for the target window (from 2025)")
print("=" * 96)
daily = (fe.load().lazy().group_by("event_date").agg(pl.col("gmv").sum().alias("g"),
                                                     pl.col("user_id").n_unique().alias("u"))
         .sort("event_date").collect())
d25 = daily.filter((pl.col("event_date") >= dt.date(2025, 1, 15))
                   & (pl.col("event_date") <= dt.date(2025, 3, 15)))
base = d25.filter(pl.col("event_date") <= dt.date(2025, 2, 13))["g"].mean()
print(f"  mean daily GMV in the 2025 feature window [01-15..02-13] = {base/1e6:.4f}M")
tgt = d25.filter(pl.col("event_date") >= dt.date(2025, 2, 14))
print(f"  mean daily GMV in the 2025 target  window [02-14..03-15] = {tgt['g'].mean()/1e6:.4f}M "
       f"  ratio = {tgt['g'].mean()/base:.4f}")
print("\n  per-day factor (target day / feature-window mean), 2025:")
for r in tgt.iter_rows(named=True):
    f = r["g"] / base
    print(f"    {r['event_date']} {r['event_date'].strftime('%a')}  factor={f:5.3f} "
          f"{'#'*int((f-0.8)*40)}")
print(f"\ntotal {time.time()-t0:.0f}s")

```
