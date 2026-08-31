"""STRATEGY_02 Variant B: denser cutoff grid at matched training volume.

Only the temporal grid and a deterministic target-free user hash sample change:
step 7 / all users versus step 3 / 42.2% users.  The fraction was fixed from
panel sizes before fitting and keeps every validation fold within 1% of the
baseline row count.

Examples (from repository root):
  python src/densityval.py --run --seeds 42 --rounds 450 --curve 150 200 250 300 450
  python src/densityval.py --run --seeds 43 44 --rounds 300 --curve 300
  python src/densityval.py --summarize
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import ARTIFACTS, DATA_PROCESSED, FOLD_WEIGHTS_S1, SEED, VAL_FOLDS_S1
from src.report import evaluate, load_report, save_report
from src.tracking import load_oof, save_oof
from src.train import Setup, block_rows, run


BASE_STEP = 7
DENSE_STEP = 3
DENSE_ROW_FRAC = 0.422
DEFAULT_CURVE = (150, 200, 250, 300, 450)
RESULT_DIR = Path("research/strategies/results/STRATEGY_02")
FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]
W = np.asarray(FOLD_WEIGHTS_S1, float)
W /= W.sum()


def dense_id(seed: int) -> str:
    return f"SAMPLE-DENSE-S3-F422-S{seed}"


def dense_setup(rounds: int, seed: int) -> Setup:
    return Setup(L=0, min_history=90, step=DENSE_STEP, panel_blocks=3,
                 train_blocks=1, model="direct", rounds=rounds,
                 params={"seed": seed}, cutoffs="all", norm_long=True,
                 row_frac=DENSE_ROW_FRAC)


def base_setup(rounds: int, seed: int) -> Setup:
    return Setup(L=0, min_history=90, step=BASE_STEP, panel_blocks=3,
                 train_blocks=1, model="direct", rounds=rounds,
                 params={"seed": seed}, cutoffs="all", norm_long=True)


def baseline_alias(seed: int, rounds: int) -> str | None:
    if seed == SEED:
        candidate = f"S1-ROUNDS-R{rounds}"
        if (ARTIFACTS / f"report_{candidate}.json").exists():
            return candidate
    if rounds == 300:
        candidate = f"S1-SEED{seed}"
        if (ARTIFACTS / f"report_{candidate}.json").exists():
            return candidate
    return None


def dense_candidate(seed: int, rounds: int) -> tuple[str, dict] | None:
    base = dense_id(seed)
    snap = f"{base}-R{rounds}"
    if (ARTIFACTS / f"report_{snap}.json").exists():
        return snap, load_report(snap)
    if (ARTIFACTS / f"report_{base}.json").exists():
        rep = load_report(base)
        if int(rep.get("params", {}).get("rounds", -1)) == rounds:
            return base, rep
    return None


def run_dense(seeds, rounds: int, curve, force: bool) -> None:
    snapshots = tuple(sorted(set(int(x) for x in curve if int(x) <= rounds)))
    for seed in seeds:
        exp = dense_id(seed)
        report = ARTIFACTS / f"report_{exp}.json"
        oof = ARTIFACTS / f"oof_{exp}.npz"
        if not force and report.exists() and oof.exists() and all(
                dense_candidate(seed, r) for r in snapshots):
            print(f"[skip] {exp}: requested artifacts already exist", flush=True)
            continue
        run(exp, f"S_02B dense step=3 row_frac={DENSE_ROW_FRAC}, seed={seed}",
            dense_setup(rounds, seed), save_model_feats=True, snap=snapshots,
            snap_save=True, no_log=True)


def collect_capacity(curve) -> list[dict]:
    rows = []
    for arm in ("baseline", "dense"):
        for rounds in curve:
            if arm == "baseline":
                exp = baseline_alias(SEED, int(rounds))
                found = (exp, load_report(exp)) if exp else None
            else:
                found = dense_candidate(SEED, int(rounds))
            if found is None:
                continue
            exp, rep = found
            rows.append(dict(arm=arm, exp_id=exp, seed=SEED, rounds=int(rounds),
                             wcv=rep["wcv"], fold_cal=rep["fold_cal"],
                             fold_scores=rep["fold_scores"], mean_z=rep["mean_z"]))
    return rows


def select_capacity(rows: list[dict], arm: str) -> dict:
    candidates = [r for r in rows if r["arm"] == arm]
    if not candidates:
        raise ValueError(f"нет capacity curve для {arm}")
    best_wcv = min(candidates, key=lambda r: (r["wcv"], r["rounds"]))
    best_last = min(candidates,
                    key=lambda r: (r["fold_cal"][-1], r["wcv"], r["rounds"]))
    return dict(best_wcv=best_wcv, best_last_fold=best_last, decision=best_last)


def exp_for(arm: str, seed: int, rounds: int) -> str | None:
    if arm == "baseline":
        return baseline_alias(seed, rounds)
    found = dense_candidate(seed, rounds)
    return found[0] if found else None


def aligned_average(exp_ids: list[str], out: str) -> dict:
    ds = [load_oof(e) for e in exp_ids]
    uid = np.asarray(ds[0]["user_id"])
    cut = np.asarray(ds[0]["cutoff"])
    y = np.asarray(ds[0]["y"])
    zs = []
    for exp, d in zip(exp_ids, ds):
        assert np.array_equal(uid, d["user_id"]), f"{exp}: user_id не выровнены"
        assert np.array_equal(cut, d["cutoff"]), f"{exp}: cutoff не выровнены"
        assert np.array_equal(y, d["y"]), f"{exp}: target не выровнен"
        zs.append(np.asarray(d["z"], float))
    z = np.mean(zs, axis=0)
    rep = evaluate(y, z, cut)
    save_oof(out, uid, cut, z, y)
    save_report(out, rep, extra=dict(description="S_02B avg3 log-space average",
                                     seeds=exp_ids, n_seeds=len(exp_ids)))
    return dict(exp_id=out, wcv=rep["wcv"], fold_cal=rep["fold_cal"],
                fold_bias=rep["fold_bias"], mean_z=rep["mean_z"])


def grid_audit() -> list[dict]:
    rows = []
    for V in VAL_FOLDS_S1:
        sb, sd = base_setup(1, SEED), dense_setup(1, SEED)
        cb, cd = sb.train_cutoffs(V), sd.train_cutoffs(V)
        assert all(T + dt.timedelta(days=30) <= V for T in cb + cd)
        rb = sum(block_rows(T, sb) for T in cb)
        rd = sum(block_rows(T, sd) for T in cd)
        rows.append(dict(
            validation_cutoff=V.isoformat(), baseline_step=BASE_STEP,
            baseline_cutoffs=len(cb), baseline_first=cb[0].isoformat(),
            baseline_last=cb[-1].isoformat(), baseline_rows=rb,
            dense_step=DENSE_STEP, dense_cutoffs=len(cd),
            dense_first=cd[0].isoformat(), dense_last=cd[-1].isoformat(),
            dense_row_frac=DENSE_ROW_FRAC, dense_rows=rd,
            row_delta_frac=rd / rb - 1,
            dense_grid=" ".join(T.isoformat() for T in cd),
            anti_lookup_ok=True,
        ))
    assert max(abs(r["row_delta_frac"]) for r in rows) <= 0.01
    return rows


def diagnostic_metrics(exps: list[str], base: str) -> dict:
    """AUC and the two pre-registered hard segments on capacity-matched avg3 OOF."""
    from src.blend import aligned
    from src.ptime_eval import auc, calibrated, order_of, seg_frame, seg_metrics, segments

    Z, y, cut = aligned(exps)
    uid, cut2 = order_of(exps[0])
    assert np.array_equal(cut, cut2)
    df = (pl.DataFrame({"cutoff": cut, "user_id": uid})
          .join(seg_frame(), on=["cutoff", "user_id"], how="left"))
    Zc = calibrated(Z, y, cut)
    masks = segments(df)
    wanted = ("ВСЕ", "rec_buy 15-60", "w180_days_buy 2-15")
    rows = []
    for name in wanted:
        rows += seg_metrics(name, masks[name], Zc, y, cut, exps)
    ib = exps.index(base)
    pooled = [auc(y > 0, Zc[i]) for i in range(len(exps))]
    fold_rows = []
    for i, exp in enumerate(exps):
        all_row = next(r for r in rows if r["segment"] == "ВСЕ" and r["exp"] == exp)
        all_row["auc_pooled"] = pooled[i]
        for c, sc, ac in zip(FOLDS, all_row["rmsle_folds"], all_row["auc_folds"]):
            fold_rows.append(dict(exp=exp, cutoff=c, rmsle_cal=sc, auc=ac))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in r.items()}
                  for r in rows]).write_csv(RESULT_DIR / "variant_B_segment_metrics.csv")
    pl.DataFrame(fold_rows).write_csv(RESULT_DIR / "variant_B_fold_metrics.csv")

    def row(exp, segment):
        return next(r for r in rows if r["exp"] == exp and r["segment"] == segment)

    candidate = exps[1]
    return dict(
        base=base, candidate=candidate,
        auc_weighted={e: row(e, "ВСЕ")["auc"] for e in exps},
        auc_pooled=dict(zip(exps, pooled)),
        delta_auc_weighted=row(candidate, "ВСЕ")["auc"] - row(base, "ВСЕ")["auc"],
        segments={name: {
            "base_rmsle": row(base, name)["rmsle"],
            "candidate_rmsle": row(candidate, name)["rmsle"],
            "delta_rmsle": row(candidate, name)["rmsle"] - row(base, name)["rmsle"],
            "base_auc": row(base, name)["auc"],
            "candidate_auc": row(candidate, name)["auc"],
            "delta_auc": row(candidate, name)["auc"] - row(base, name)["auc"],
        } for name in wanted[1:]},
    )


def diversity(exp_a: str, exp_b: str) -> dict:
    from src.blend import aligned
    Z, y, _ = aligned([exp_a, exp_b])
    ly = np.log1p(y)
    return dict(var_delta=float(np.var(Z[0] - Z[1])),
                residual_corr=float(np.corrcoef(ly - Z[0], ly - Z[1])[0, 1]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                        for k, v in row.items()})


def summarize(curve, out_name: str) -> dict:
    rows = collect_capacity(curve)
    capacity = {arm: select_capacity(rows, arm) for arm in ("baseline", "dense")}
    decision_rounds = {arm: capacity[arm]["decision"]["rounds"] for arm in capacity}
    single = {arm: capacity[arm]["decision"] for arm in capacity}
    single_delta = single["dense"]["wcv"] - single["baseline"]["wcv"]
    single_fd = (np.asarray(single["dense"]["fold_cal"])
                 - np.asarray(single["baseline"]["fold_cal"])).tolist()

    avg3 = {}
    seed_rows = []
    for arm in ("baseline", "dense"):
        r = decision_rounds[arm]
        exps = [exp_for(arm, seed, r) for seed in (42, 43, 44)]
        for seed, exp in zip((42, 43, 44), exps):
            if exp:
                rep = load_report(exp)
                seed_rows.append(dict(arm=arm, seed=seed, rounds=r, exp_id=exp,
                                      wcv=rep["wcv"], fold_cal=rep["fold_cal"]))
        if all(exps):
            avg3[arm] = aligned_average(exps, f"SAMPLE-{arm.upper()}-B-AVG3-R{r}")

    if len(avg3) == 2:
        fd = (np.asarray(avg3["dense"]["fold_cal"])
              - np.asarray(avg3["baseline"]["fold_cal"])).tolist()
        delta = avg3["dense"]["wcv"] - avg3["baseline"]["wcv"]
        wins = sum(x < 0 for x in fd)
        diagnostics = diagnostic_metrics(
            [avg3["baseline"]["exp_id"], avg3["dense"]["exp_id"]],
            avg3["baseline"]["exp_id"])
        div = diversity(avg3["baseline"]["exp_id"], avg3["dense"]["exp_id"])
        evidence = "avg3"
    else:
        fd, delta, wins = single_fd, single_delta, sum(x < 0 for x in single_fd)
        diagnostics = diagnostic_metrics(
            [single["baseline"]["exp_id"], single["dense"]["exp_id"]],
            single["baseline"]["exp_id"])
        div = diversity(single["baseline"]["exp_id"], single["dense"]["exp_id"])
        evidence = "seed42"

    if delta <= -0.0005 and wins >= 3 and fd[-1] < 0:
        verdict = "PASS"
    elif diagnostics and (diagnostics["delta_auc_weighted"] >= 0.0005 or all(
            s["delta_rmsle"] < 0 and s["delta_auc"] > 0
            for s in diagnostics["segments"].values())):
        verdict = "NEUTRAL"
    else:
        verdict = "FAIL"

    audit = grid_audit()
    summary = dict(
        design=dict(baseline_step=BASE_STEP, dense_step=DENSE_STEP,
                    dense_row_frac=DENSE_ROW_FRAC, hash="(user_id*2654435761)%1000 < 422",
                    only_change="temporal cutoff density with volume-matching user sample"),
        capacity=capacity, decision_rounds=decision_rounds,
        seed42=dict(delta_wcv=single_delta, fold_delta=single_fd,
                    wins=sum(x < 0 for x in single_fd)),
        avg3=avg3, delta_wcv=delta, fold_delta=fd, fold_wins=wins,
        latest_fold_win=fd[-1] < 0, evidence=evidence,
        diagnostics=diagnostics, diversity=div, grid_audit=audit, verdict=verdict,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"sample_design_{out_name}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(RESULT_DIR / "variant_B_capacity_curve.csv", rows)
    write_csv(RESULT_DIR / "variant_B_seed_robustness.csv", seed_rows)
    write_csv(RESULT_DIR / "variant_B_cutoff_grid.csv", audit)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"saved {path}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--rounds", type=int, default=max(DEFAULT_CURVE))
    ap.add_argument("--curve", nargs="+", type=int, default=list(DEFAULT_CURVE))
    ap.add_argument("--out", default="S1-SAMPLE-B-FINAL")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.run:
        run_dense(a.seeds, a.rounds, a.curve, a.force)
    else:
        summarize(a.curve, a.out)


if __name__ == "__main__":
    main()
