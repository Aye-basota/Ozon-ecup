# e07_blocks_anchor

## Catalogue metadata

- **Catalogue ID:** `eda__e07_blocks_anchor`
- **Namespace:** `eda`
- **Experiment ID:** `e07_blocks_anchor`
- **Original source:** `research/eda/e07_blocks_anchor.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** EDA experiment/script
- **Model:** Unknown / not recoverable from repository history
- **Features:** holiday/YoY features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** f"(=> mean predicted GMV level expm1 ~ {np.expm1(m_x_test+dm):.3f})")
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e07_blocks_anchor

Original script: `research/eda/e07_blocks_anchor.py`

```python
"""Stage 6: 30-day block activity (true panel rule), test-level anchors, all-zero-row value."""
import datetime as dt

import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw"
OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"
END = dt.date(2026, 2, 13)
N = 250000

act = pl.scan_parquet(RAW + r"\train.parquet").select("user_id", "event_date", "gmv", "searches", "cat", "to_ord", "to_cart").collect()
span = pl.read_parquet(OUT + r"\user_span.parquet")
all_u = span.select("user_id").sort("user_id")

print("=" * 100)
print("A. ACTIVITY PER 30-DAY BLOCK COUNTING BACK FROM 2026-02-13")
print("=" * 100)
print(f"{'block':>5s}  {'window':26s} {'users active':>13s} {'%':>8s}")
block_users = []
for k in range(13):
    b = END - dt.timedelta(days=30 * k)
    a = b - dt.timedelta(days=29)
    if a < dt.date(2025, 1, 1):
        a = dt.date(2025, 1, 1)
    u = act.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b)).select("user_id").unique()
    block_users.append(u)
    print(f"{k:>5d}  [{a}..{b}] {u.height:13,} {100*u.height/N:7.3f}%")

print()
print("--- cumulative: users active in ALL of the first k blocks ---")
cur = block_users[0]
for k in range(1, 6):
    cur = cur.join(block_users[k], on="user_id", how="inner")
    print(f"  active in all blocks 0..{k}: {cur.height:,} ({100*cur.height/N:.3f}%)")

print()
print("--- sanity: arbitrary 30-day windows NOT aligned to the block grid ---")
for off in [5, 10, 15, 20, 25, 35, 45]:
    b = END - dt.timedelta(days=off)
    a = b - dt.timedelta(days=29)
    u = act.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))["user_id"].n_unique()
    print(f"  [{a}..{b}] (offset {off:2d}) {u:,} ({100*u/N:.3f}%)")

print()
print("--- 90-day and 60-day trailing windows ---")
for L in [60, 90, 120]:
    a = END - dt.timedelta(days=L - 1)
    u = act.filter(pl.col("event_date") >= a)["user_id"].n_unique()
    print(f"  last {L}d [{a}..{END}] {u:,} ({100*u/N:.3f}%)")

print()
print("=" * 100)
print("B. TEST-LEVEL ANCHORS  (what mean(log1p(pred)) should be on the submission)")
print("=" * 100)
ss = pl.read_parquet(OUT + r"\sample_submit.parquet")
x_test = ss.sort("user_id")["predict"].to_numpy()          # = trailing 30d GMV at T=2026-02-13
m_x_test = float(np.log1p(x_test).mean())
p_x_test = float((x_test > 0).mean())
mu_x_test = float(np.log1p(x_test[x_test > 0]).mean())
print(f"  trailing-30d at T=2026-02-13:  m_x={m_x_test:.4f}  P(x>0)={p_x_test:.4f}  mu_x={mu_x_test:.4f}")
print()
print("  scenarios for the target level E[log1p(y_test)]:")
for nm, dm in [("year-median drift", 0.0887), ("clean-cutoff mean", 0.0841),
               ("YoY analogue 2025-02-13", 0.1759), ("no drift", 0.0),
               ("post-holiday (2026-01-01)", -0.2813)]:
    print(f"    {nm:28s} dm={dm:+.4f}  ->  E[log1p(y)] = {m_x_test+dm:.4f}  "
          f"(=> mean predicted GMV level expm1 ~ {np.expm1(m_x_test+dm):.3f})")
print()
print("  YoY extensive/intensive split at 2025-02-13: dP=+0.0357, dmu=+0.0656")
print(f"  => implied test P(y>0) ~ {p_x_test+0.0357:.4f}, implied mu_y ~ {mu_x_test+0.0656:.4f}")
print(f"  => cross-check E[log1p(y)] = P*mu = {(p_x_test+0.0357)*(mu_x_test+0.0656):.4f}")

print()
print("=" * 100)
print("C. VALUE OF 'PRESENCE-ONLY' ROWS (all-zero rows) — do they predict future GMV?")
print("=" * 100)
T = dt.date(2025, 10, 15)
w = act.filter((pl.col("event_date") > T - dt.timedelta(days=30)) & (pl.col("event_date") <= T))
feat = w.group_by("user_id").agg(
    n_rows=pl.len(),
    n_zero_rows=((pl.col("searches") == 0) & (pl.col("cat") == 0) & (pl.col("to_cart") == 0)
                 & (pl.col("to_ord") == 0) & (pl.col("gmv") == 0)).sum(),
    n_search_days=(pl.col("searches") > 0).sum(),
    gmv30=pl.col("gmv").sum(),
)
gp = (act.filter((pl.col("event_date") > T) & (pl.col("event_date") <= T + dt.timedelta(days=30)))
      .group_by("user_id").agg(pl.col("gmv").sum().alias("y")))
d = (feat.join(gp, on="user_id", how="left").with_columns(pl.col("y").fill_null(0.0))
     .with_columns(zfrac=pl.col("n_zero_rows") / pl.col("n_rows")))
print("users with >=1 row in the last 30d at T=2025-10-15:", d.height)
print("\nfuture GMV by number of presence-only rows, CONTROLLING for #search-days and gmv30 bucket:")
d2 = d.with_columns(
    act_b=pl.when(pl.col("n_search_days") <= 2).then(pl.lit("act 0-2"))
           .when(pl.col("n_search_days") <= 6).then(pl.lit("act 3-6"))
           .when(pl.col("n_search_days") <= 14).then(pl.lit("act 7-14"))
           .otherwise(pl.lit("act 15+")),
    gmv_b=pl.when(pl.col("gmv30") == 0).then(pl.lit("gmv30=0")).otherwise(pl.lit("gmv30>0")),
    z_b=pl.when(pl.col("n_zero_rows") == 0).then(pl.lit("0 zerorows"))
         .when(pl.col("n_zero_rows") <= 2).then(pl.lit("1-2"))
         .when(pl.col("n_zero_rows") <= 5).then(pl.lit("3-5")).otherwise(pl.lit("6+")),
)
r = (d2.group_by(["gmv_b", "act_b", "z_b"])
     .agg(n=pl.len(), P_buy=(pl.col("y") > 0).mean(), m_y=pl.col("y").log1p().mean())
     .sort(["gmv_b", "act_b", "z_b"]))
print(r.to_pandas().to_string(index=False))

```
