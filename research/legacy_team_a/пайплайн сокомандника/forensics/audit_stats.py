from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.parquet"
SAMPLE = ROOT / "data" / "sample_submit.csv"
LATEST = ROOT / "latest" / "latest.csv"
COMPONENTS = {
    "friend": ROOT / "latest" / "components" / "friend.csv",
    "occ_meta_B": ROOT / "latest" / "components" / "occ_meta_B.csv",
    "occ_raw_X3": ROOT / "latest" / "components" / "occ_raw_X3.csv",
}

DATA_START = dt.date(2025, 1, 1)
CORRIDOR_END = dt.date(2025, 10, 16)
TEST_CUTOFF = dt.date(2026, 2, 13)
FOLDS = [
    dt.date(2025, 9, 4),
    dt.date(2025, 9, 18),
    dt.date(2025, 10, 2),
    dt.date(2025, 10, 16),
]


def jprint(label: str, value) -> None:
    print(label)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def cutoff_grid() -> list[dt.date]:
    start = DATA_START + dt.timedelta(days=90)
    out, cur = [], CORRIDOR_END
    while cur >= start:
        out.append(cur)
        cur -= dt.timedelta(days=7)
    return sorted(out)


def panel_users(cutoff: dt.date, blocks: int) -> pl.DataFrame:
    users = None
    for k in range(blocks):
        end = cutoff - dt.timedelta(days=30 * k)
        start = end - dt.timedelta(days=29)
        block = (
            pl.scan_parquet(DATA)
            .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end))
            .select("user_id")
            .unique()
            .collect()
        )
        users = block if users is None else users.join(block, on="user_id", how="inner")
    assert users is not None
    return users.sort("user_id")


def prediction_stats(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    p = df["predict"].to_numpy(np.float64)
    z = np.log1p(np.maximum(p, 0.0))
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return df, {
        "rows": len(df),
        "unique_users": int(df.user_id.nunique()),
        "mean_pred": float(p.mean()),
        "median_pred": float(np.median(p)),
        "mean_log1p": float(z.mean()),
        "std_log1p": float(z.std()),
        "zero_share": float((p == 0).mean()),
        **{f"p{int(q*100):02d}": float(np.quantile(p, q)) for q in qs},
        "max": float(p.max()),
    }


def prediction_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, stats = {}, {}
    for name, path in {"latest": LATEST, **COMPONENTS}.items():
        frames[name], stats[name] = prediction_stats(path)
    sample = pd.read_csv(SAMPLE)
    uid = sample.user_id.to_numpy(np.int64)
    for name, df in frames.items():
        assert np.array_equal(df.user_id.to_numpy(np.int64), uid), name
    jprint("PREDICTION_STATS", stats)

    f = frames["friend"]
    last = frames["latest"]
    zf = np.log1p(np.maximum(f.predict.to_numpy(np.float64), 0.0))
    zl = np.log1p(np.maximum(last.predict.to_numpy(np.float64), 0.0))
    d = zl - zf
    qgrid = [0, .001, .01, .05, .10, .25, .50, .75, .90, .95, .99, .999, 1]
    geom = {
        "corr_log1p": float(np.corrcoef(zl, zf)[0, 1]),
        "corr_raw": float(np.corrcoef(last.predict, f.predict)[0, 1]),
        "mean_d": float(d.mean()),
        "std_d": float(d.std()),
        "var_d": float(d.var()),
        "rmssd": float(np.sqrt(np.mean(d * d))),
        "mae_d": float(np.mean(np.abs(d))),
        "share_d_positive": float((d > 0).mean()),
        "share_abs_gt_002": float((np.abs(d) > .02).mean()),
        "share_abs_gt_005": float((np.abs(d) > .05).mean()),
        "share_abs_gt_010": float((np.abs(d) > .10).mean()),
        "quantiles": {str(q): float(np.quantile(d, q)) for q in qgrid},
    }
    jprint("LATEST_MINUS_FRIEND_GEOMETRY", geom)

    dx = {}
    for name in ("occ_meta_B", "occ_raw_X3"):
        z = np.log1p(np.maximum(frames[name].predict.to_numpy(np.float64), 0.0))
        dd = zl - z
        dx[name] = {
            "corr_latest_component": float(np.corrcoef(zl, z)[0, 1]),
            "std_latest_minus_component": float(dd.std()),
            "mae_latest_minus_component": float(np.mean(np.abs(dd))),
            "var_latest_minus_component": float(dd.var()),
        }
    jprint("LATEST_DISTANCE_TO_COMPONENTS", dx)

    order = np.argsort(d)
    extremes = pd.DataFrame({
        "user_id": uid,
        "friend_pred": f.predict.to_numpy(np.float64),
        "latest_pred": last.predict.to_numpy(np.float64),
        "d_log": d,
    })
    jprint("MOST_NEGATIVE_D", extremes.iloc[order[:20]].to_dict("records"))
    jprint("MOST_POSITIVE_D", extremes.iloc[order[-20:][::-1]].to_dict("records"))
    return extremes, frames["occ_raw_X3"]


def data_construction() -> None:
    scan = pl.scan_parquet(DATA)
    overview = scan.select(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("event_date").min().alias("date_min"),
        pl.col("event_date").max().alias("date_max"),
        pl.struct(["user_id", "event_date"]).n_unique().alias("unique_user_days"),
        pl.col("gmv").is_null().sum().alias("gmv_nulls"),
        (pl.col("gmv") > 0).sum().alias("positive_gmv_rows"),
    ).collect().to_dicts()[0]
    jprint("RAW_DATA_OVERVIEW", overview)

    cuts = cutoff_grid()
    one_block_panels = []
    cutoff_rows = []
    for i, cutoff in enumerate(cuts):
        p = panel_users(cutoff, 1).with_columns(pl.lit(i).cast(pl.Int16).alias("cutoff_i"))
        one_block_panels.append(p)
        cutoff_rows.append({"cutoff": cutoff.isoformat(), "n_train_panel_b1": p.height})
    examples = pl.concat(one_block_panels)
    repeat = examples.group_by("user_id").len().select(
        pl.len().alias("unique_train_users"),
        pl.col("len").mean().alias("mean_states_per_user"),
        pl.col("len").median().alias("median_states_per_user"),
        *[pl.col("len").quantile(q).alias(f"q{int(q*100):02d}") for q in (.01,.05,.10,.25,.75,.90,.95,.99)],
        pl.col("len").max().alias("max_states_per_user"),
    ).to_dicts()[0]
    repeat["total_training_examples_full_fit"] = examples.height
    jprint("FULL_TABLE_TRAIN_POPULATION", repeat)
    jprint("ONE_BLOCK_PANEL_BY_CUTOFF", cutoff_rows)

    one_counts = {dt.date.fromisoformat(x["cutoff"]): x["n_train_panel_b1"] for x in cutoff_rows}
    fold_rows = []
    for val in FOLDS:
        tr = [c for c in cuts if c + dt.timedelta(days=30) <= val]
        val_users = panel_users(val, 3)
        start, end = val + dt.timedelta(days=1), val + dt.timedelta(days=30)
        pos = (
            pl.scan_parquet(DATA)
            .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end) & (pl.col("gmv") > 0))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
            .collect()
        )
        val_target = val_users.join(pos, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
        fold_rows.append({
            "val": val.isoformat(),
            "train_cutoffs": len(tr),
            "train_first": tr[0].isoformat(),
            "train_last": tr[-1].isoformat(),
            "train_examples_b1": int(sum(one_counts[c] for c in tr)),
            "val_panel_b3": val_users.height,
            "target_zero_share": float((val_target["target"] == 0).mean()),
            "mean_log1p_target": float(val_target["target"].log1p().mean()),
        })
    test_panel = panel_users(TEST_CUTOFF, 3)
    sample = pl.read_csv(SAMPLE, columns=["user_id"]).sort("user_id")
    fold_rows.append({
        "val": "TEST",
        "train_cutoffs": len(cuts),
        "train_first": cuts[0].isoformat(),
        "train_last": cuts[-1].isoformat(),
        "train_examples_b1": int(sum(one_counts.values())),
        "val_panel_b3": test_panel.height,
        "sample_exact_match": bool(test_panel.equals(sample)),
    })
    jprint("FOLD_POPULATIONS", fold_rows)

    last10 = cuts[-10:]
    lag = np.asarray([(TEST_CUTOFF - x).days for x in last10], np.float64)
    weights = np.exp(-lag / 55.0)
    ridge_fw = np.asarray([1., 2., 4., 8.]) ** 1.7
    jprint("TEMPORAL_WEIGHTING", {
        "occ_r10_test_cutoffs": [x.isoformat() for x in last10],
        "occ_r10_test_lag_days": lag.astype(int).tolist(),
        "occ_r10_tau55_relative_weights": (weights / weights.max()).tolist(),
        "occ_r10_normalized_cutoff_weights_ignoring_panel_size": (weights / weights.sum()).tolist(),
        "ridge_fold_weights_power1p7_normalized": (ridge_fw / ridge_fw.sum()).tolist(),
        "adjacent_target_window_overlap_days": 23,
        "adjacent_target_window_overlap_share": 23/30,
    })


def activity_profile(extremes: pd.DataFrame) -> None:
    cutoff = TEST_CUTOFF
    start = cutoff - dt.timedelta(days=180) + dt.timedelta(days=1)
    base = (
        pl.scan_parquet(DATA)
        .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= cutoff))
        .with_columns((pl.lit(cutoff) - pl.col("event_date")).dt.total_days().alias("age"))
    )
    exprs = [
        pl.col("age").min().alias("rec_any"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).min().alias("rec_buy"),
    ]
    for w in (7, 30, 60, 90, 180):
        m = pl.col("age") < w
        exprs.extend([
            m.sum().alias(f"w{w}_days_present"),
            (m & (pl.col("gmv") > 0)).sum().alias(f"w{w}_days_buy"),
            pl.when(m).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"w{w}_gmv"),
            pl.when(m).then(pl.col("to_ord")).otherwise(0).sum().alias(f"w{w}_orders"),
            pl.when(m).then(pl.col("searches")).otherwise(0).sum().alias(f"w{w}_searches"),
        ])
    agg = base.group_by("user_id").agg(exprs).collect().to_pandas()
    df = extremes.merge(agg, on="user_id", how="left")
    d = df.d_log.to_numpy(np.float64)
    corr_cols = ["friend_pred","rec_any","rec_buy","w7_days_present","w30_days_present","w90_days_present",
                 "w180_days_present","w30_days_buy","w90_days_buy","w180_days_buy","w30_gmv","w90_gmv","w180_gmv",
                 "w30_orders","w90_orders","w30_searches","w90_searches"]
    corr = {}
    for c in corr_cols:
        x = df[c].fillna(999 if c.startswith("rec_") else 0).to_numpy(np.float64)
        if "gmv" in c or c == "friend_pred":
            x = np.log1p(np.maximum(x, 0))
        corr[c] = float(np.corrcoef(d, x)[0, 1])
    jprint("CORRECTION_CORRELATIONS_WITH_TEST_HISTORY", corr)

    def segment(col: str, bins, labels) -> list[dict]:
        s = pd.cut(df[col].fillna(999), bins=bins, labels=labels, right=False, include_lowest=True)
        out=[]
        for label in labels:
            m=(s==label).to_numpy()
            if not m.any(): continue
            out.append({"segment":str(label),"n":int(m.sum()),"share":float(m.mean()),
                        "mean_d":float(d[m].mean()),"std_d":float(d[m].std()),
                        "share_d_negative":float((d[m]<0).mean()),"mean_friend_z":float(np.log1p(df.friend_pred.to_numpy()[m]).mean())})
        return out

    jprint("D_BY_REC_BUY", segment("rec_buy", [-1,1,8,31,61,91,181,10000], ["0d","1-7d","8-30d","31-60d","61-90d","91-180d","no-buy-180d"]))
    jprint("D_BY_W90_BUY_DAYS", segment("w90_days_buy", [-1,1,2,4,8,16,10000], ["0","1","2-3","4-7","8-15","16+"]))
    jprint("D_BY_W90_PRESENT_DAYS", segment("w90_days_present", [-1,31,46,61,76,86,91,10000], ["<=30","31-45","46-60","61-75","76-85","86-90","other"]))

    top = df.reindex(df.d_log.abs().sort_values(ascending=False).index).head(30)
    cols = ["user_id","friend_pred","latest_pred","d_log","rec_buy","w30_days_present","w90_days_present","w30_days_buy","w90_days_buy","w30_gmv","w90_gmv"]
    jprint("LARGEST_ABS_CORRECTIONS_WITH_HISTORY", top[cols].to_dict("records"))


def main() -> None:
    extremes, _ = prediction_geometry()
    data_construction()
    activity_profile(extremes)


if __name__ == "__main__":
    main()
