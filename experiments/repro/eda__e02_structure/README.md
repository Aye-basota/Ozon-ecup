# what do "zero" rows look like — is `cat` ever >0 while search==0?

## Catalogue metadata

- **Catalogue ID:** `eda__e02_structure`
- **Namespace:** `eda`
- **Experiment ID:** `e02_structure`
- **Original source:** `research/eda/e02_structure.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** EDA experiment/script
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e02_structure

Original script: `research/eda/e02_structure.py`

```python
"""Stage 1: global structure + sample_submit forensics."""
import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw"
OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"

lf = pl.scan_parquet(RAW + r"\train.parquet")

print("=" * 70)
print("1. GLOBAL")
print("=" * 70)
g = lf.select(
    n_rows=pl.len(),
    n_users=pl.col("user_id").n_unique(),
    dmin=pl.col("event_date").min(),
    dmax=pl.col("event_date").max(),
    n_dates=pl.col("event_date").n_unique(),
    uid_min=pl.col("user_id").min(),
    uid_max=pl.col("user_id").max(),
).collect()
print(g)

n_days = (g["dmax"][0] - g["dmin"][0]).days + 1
print(f"calendar days in range: {n_days}, distinct dates present: {g['n_dates'][0]}")

print()
print("=" * 70)
print("2. COLUMN IDENTITIES / CONSISTENCY")
print("=" * 70)
cons = lf.select(
    gmv_eq=(((pl.col("gmv_search") + pl.col("gmv_cat")) - pl.col("gmv")).abs() > 1e-6).sum(),
    ord_eq=((pl.col("search_to_ord") + pl.col("cat_to_ord")) != pl.col("to_ord")).sum(),
    cart_eq=((pl.col("search_to_cart") + pl.col("cat_to_cart")) != pl.col("to_cart")).sum(),
    hs_cart=((pl.col("has_search_to_cart") != (pl.col("search_to_cart") > 0).cast(pl.Int64))).sum(),
    hs_ord=((pl.col("has_search_to_ord") != (pl.col("search_to_ord") > 0).cast(pl.Int64))).sum(),
    hc_cart=((pl.col("has_cat_to_cart") != (pl.col("cat_to_cart") > 0).cast(pl.Int64))).sum(),
    hc_ord=((pl.col("has_cat_to_ord") != (pl.col("cat_to_ord") > 0).cast(pl.Int64))).sum(),
    search_flag=((pl.col("search") != (pl.col("searches") > 0).cast(pl.Int64))).sum(),
    gmv_search_pos_no_ord=((pl.col("gmv_search") > 0) & (pl.col("search_to_ord") == 0)).sum(),
    gmv_cat_pos_no_ord=((pl.col("gmv_cat") > 0) & (pl.col("cat_to_ord") == 0)).sum(),
    ord_pos_no_gmv=((pl.col("to_ord") > 0) & (pl.col("gmv") <= 0)).sum(),
    gmv_neg=(pl.col("gmv") < 0).sum(),
).collect()
for c in cons.columns:
    print(f"  {c:26s} violations = {cons[c][0]:,}")

print()
print("=" * 70)
print("3. PER-COLUMN STATS")
print("=" * 70)
cols = ["search", "cat", "has_search_to_cart", "has_search_to_ord", "has_cat_to_cart",
        "has_cat_to_ord", "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
        "to_cart", "to_ord", "searches", "gmv_search", "gmv_cat", "gmv"]
stats = lf.select(
    [pl.col(c).min().alias(f"{c}|min") for c in cols]
    + [pl.col(c).max().alias(f"{c}|max") for c in cols]
    + [pl.col(c).mean().alias(f"{c}|mean") for c in cols]
    + [(pl.col(c) == 0).mean().alias(f"{c}|zerofrac") for c in cols]
    + [pl.col(c).n_unique().alias(f"{c}|nuniq") for c in cols]
    + [pl.col(c).null_count().alias(f"{c}|nulls") for c in cols]
).collect()
print(f"{'col':20s} {'min':>10s} {'max':>12s} {'mean':>12s} {'zero%':>8s} {'nuniq':>9s} {'nulls':>7s}")
for c in cols:
    print(f"{c:20s} {stats[f'{c}|min'][0]:>10.4g} {stats[f'{c}|max'][0]:>12.6g} "
          f"{stats[f'{c}|mean'][0]:>12.6g} {100*stats[f'{c}|zerofrac'][0]:>7.2f}% "
          f"{stats[f'{c}|nuniq'][0]:>9,} {stats[f'{c}|nulls'][0]:>7,}")

print()
print("=" * 70)
print("4. ALL-ZERO ROWS (row exists but nothing happened)")
print("=" * 70)
allzero = (pl.col("searches") == 0) & (pl.col("cat") == 0) & (pl.col("to_cart") == 0) \
          & (pl.col("to_ord") == 0) & (pl.col("gmv") == 0) & (pl.col("search") == 0)
az = lf.select(n_allzero=allzero.sum(), frac=allzero.mean()).collect()
print(az)
# what do "zero" rows look like — is `cat` ever >0 while search==0?
print(lf.group_by(["search", "cat"]).agg(pl.len().alias("n"), pl.col("gmv").mean().alias("gmv_mean"))
      .sort("n", descending=True).collect())

print()
print("=" * 70)
print("5. ROWS PER USER")
print("=" * 70)
rpu = lf.group_by("user_id").agg(pl.len().alias("n")).collect()
print(rpu["n"].describe())
print("quantiles:", {q: float(rpu["n"].quantile(q)) for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]})

print()
print("=" * 70)
print("6. SAMPLE_SUBMIT FORENSICS")
print("=" * 70)
ss = pl.read_csv(RAW + r"\sample_submit.csv")
print("shape", ss.shape, "cols", ss.columns)
users_train = rpu.select("user_id")
print("users in train:", users_train.height, "users in submit:", ss.height)
inter = ss.join(users_train, on="user_id", how="inner").height
print("intersection:", inter)
print("submit users NOT in train:", ss.height - inter)
print("train users NOT in submit:", users_train.height - inter)

pv = ss["predict"]
print("\npredict stats:")
print(pv.describe())
print("zeros:", int((pv == 0).sum()), f"({100*(pv==0).mean():.2f}%)")
print("n unique:", pv.n_unique())
# how many decimal digits -> is it float model output or an integer-ish sum?
print("\ntop-20 most frequent predict values:")
print(ss.group_by("predict").agg(pl.len().alias("n")).sort("n", descending=True).head(20))

ss.write_parquet(OUT + r"\sample_submit.parquet")
rpu.write_parquet(OUT + r"\rows_per_user.parquet")
print("\nsaved intermediates")

```
