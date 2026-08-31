"""Target-free calendar-composition supplement for EXP085; no model training."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp085_core", HERE / "run_forensic_audit.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import EXP085 core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def profile_delta(hist: np.ndarray, hist_dates: pd.DatetimeIndex,
                  future_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    overall = hist.mean(axis=1)
    dow_mean = np.column_stack([
        hist[:, hist_dates.dayofweek == k].mean(axis=1) for k in range(7)
    ])
    dow_expected = dow_mean[:, future_dates.dayofweek].sum(axis=1)
    dow_delta = dow_expected - len(future_dates) * overall

    # Exact day-of-month means with four pseudo-days of shrinkage to the user's
    # own overall rate; this is frozen and target-free.
    dom_mean = np.empty((len(hist), 31), np.float64)
    for k in range(1, 32):
        mask = hist_dates.day == k
        n = int(mask.sum())
        dom_mean[:, k - 1] = (hist[:, mask].sum(axis=1) + 4 * overall) / (n + 4)
    dom_expected = dom_mean[:, future_dates.day.to_numpy() - 1].sum(axis=1)
    dom_delta = dom_expected - len(future_dates) * overall
    return dow_delta, dom_delta


def main() -> None:
    prod = core.reconstruct_production()
    uid_all = np.load(core.PROCESSED / "seq_uid_v1.npy", mmap_mode="r")
    gmv = np.load(core.PROCESSED / "seq_gmv_v1.npy", mmap_mode="r")
    all_dates = pd.date_range("2025-01-01", periods=gmv.shape[1], freq="D")
    store = {}
    rows = []
    for f in core.FOLDS:
        m = prod["masks"][f]
        ids = prod["uid"][m]
        idx = np.searchsorted(uid_all, ids)
        d = int((np.datetime64(f) - core.DATA_START).astype("timedelta64[D]").astype(int))
        h0 = max(0, d - 364)
        hist = np.asarray(gmv[idx, h0:d + 1], np.float64)
        hist_dates = all_dates[h0:d + 1]
        future_dates = pd.date_range(pd.Timestamp(f) + pd.Timedelta(days=1), periods=30, freq="D")
        ddow, ddom = profile_delta(hist, hist_dates, future_dates)
        pred_y = np.expm1(prod["z"][m])
        p_dow = np.log1p(np.maximum(pred_y + ddow, 0)) - prod["z"][m]
        p_dom = np.log1p(np.maximum(pred_y + ddom, 0)) - prod["z"][m]
        D = np.column_stack([p_dow, p_dom])
        fit = core.gain(prod["r"][m], D, prod["bases"][f])
        store[f] = (D, fit)
        rows.append({
            "cutoff": f,
            "oracle_headroom": fit["gain"],
            "oracle_headroom_debiased": fit["debiased_gain"],
            "oracle_rho": fit["rho"],
            "rank": fit["rank"],
            "proxy_dow_rms": core.rms(p_dow),
            "proxy_dom_rms": core.rms(p_dom),
            "corr_dow_dom": core.corr(p_dow, p_dom),
        })
    w = np.asarray([core.FOLD_WEIGHT[f] for f in core.FOLDS], float)
    agg = {"cutoff": "weighted_1_2_4_8"}
    for col in rows[0]:
        if col != "cutoff":
            agg[col] = float(np.average([r[col] for r in rows], weights=w))
    donor = store["2025-09-04"][1]
    Dv = store["2025-10-16"][0]
    mv = prod["masks"]["2025-10-16"]
    q = core.project_out(Dv, prod["bases"]["2025-10-16"]) @ donor["beta"]
    rv = prod["r"][mv]
    agg.update({
        "purged_rho": core.corr(q, rv),
        "purged_correction_rms": core.rms(q),
        "purged_Delta_MSE": float(np.mean((rv - q) ** 2 - rv ** 2)),
        "oracle_gate_ge_0_001": bool(agg["oracle_headroom_debiased"] >= 0.001),
        "observable_gate": bool(abs(core.corr(q, rv)) >= 0.020 or np.mean((rv - q) ** 2 - rv ** 2) <= -0.001),
    })
    rows.append(agg)
    pd.DataFrame(rows).to_csv(HERE / "calendar_profile_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
