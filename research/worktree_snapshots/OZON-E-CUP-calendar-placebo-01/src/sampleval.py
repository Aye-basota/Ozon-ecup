"""STRATEGY_02 Variant A: train_blocks=0 with a capacity-matched baseline.

The registered S_05 snapshots are reused for the train_blocks=1 baseline.  The
new arm is trained with the same features, target, folds, loss and parameters;
only the training panel changes.

Examples:
  python -m src.sampleval --run --train-blocks 0 --seeds 42
  python -m src.sampleval --run --train-blocks 0 --seeds 43 44 --rounds 300 --curve 300
  python -m src.sampleval --summarize
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, CUTOFF_TEST, SEED, VAL_FOLDS_S1
from src.report import evaluate, load_report, save_report
from src.tracking import load_oof, save_oof
from src.train import Setup, run


DEFAULT_CURVE = (150, 200, 250, 300, 450)


def experiment_id(train_blocks: int, seed: int) -> str:
    return f"SAMPLE-TB{train_blocks}-S{seed}"


def setup_for(train_blocks: int, rounds: int, seed: int) -> Setup:
    return Setup(L=0, min_history=90, step=7, panel_blocks=3,
                 train_blocks=train_blocks, model="direct", rounds=rounds,
                 params={"seed": seed}, cutoffs="all", norm_long=True)


def _baseline_alias(seed: int, rounds: int) -> str | None:
    """Exact S_05 artifacts for train_blocks=1; avoids repeating that experiment."""
    if seed == 42:
        candidate = f"S1-ROUNDS-R{rounds}"
        if (ARTIFACTS / f"report_{candidate}.json").exists():
            return candidate
        if rounds == 300 and (ARTIFACTS / "report_S1-SEED42.json").exists():
            return "S1-SEED42"
    if rounds == 300:
        candidate = f"S1-SEED{seed}"
        if (ARTIFACTS / f"report_{candidate}.json").exists():
            return candidate
    return None


def _candidate(train_blocks: int, seed: int, rounds: int) -> tuple[str, dict] | None:
    if train_blocks == 1:
        alias = _baseline_alias(seed, rounds)
        return (alias, load_report(alias)) if alias else None
    base = experiment_id(train_blocks, seed)
    snap = f"{base}-R{rounds}"
    if (ARTIFACTS / f"report_{snap}.json").exists():
        return snap, load_report(snap)
    if (ARTIFACTS / f"report_{base}.json").exists():
        rep = load_report(base)
        if int(rep.get("params", {}).get("rounds", -1)) == rounds:
            return base, rep
    return None


def run_grid(train_blocks, seeds, rounds: int, curve, force: bool) -> None:
    snapshots = tuple(sorted(set(int(x) for x in curve if int(x) <= rounds)))
    for tb in train_blocks:
        for seed in seeds:
            # The exact baseline already exists from S_05; do not spend compute twice.
            if tb == 1 and all(_baseline_alias(seed, r) for r in snapshots):
                print(f"[reuse] train_blocks=1 seed={seed}: S_05 artifacts", flush=True)
                continue
            exp = experiment_id(tb, seed)
            report = ARTIFACTS / f"report_{exp}.json"
            oof = ARTIFACTS / f"oof_{exp}.npz"
            if not force and report.exists() and oof.exists():
                print(f"[skip] {exp}: artifacts already exist", flush=True)
                continue
            run(exp, f"S_02A train_blocks={tb}, seed={seed}", setup_for(tb, rounds, seed),
                save_model_feats=True, snap=snapshots, snap_save=True, no_log=True)


def collect_rows(train_blocks, seeds, curve) -> list[dict]:
    rows = []
    for tb in train_blocks:
        for seed in seeds:
            for rounds in curve:
                found = _candidate(tb, seed, int(rounds))
                if found is None:
                    continue
                exp, rep = found
                rows.append(dict(
                    exp_id=exp, train_blocks=int(tb), seed=int(seed), rounds=int(rounds),
                    wcv=rep["wcv"], fold_cal=rep["fold_cal"], fold_scores=rep["fold_scores"],
                    fold_bias=rep["fold_bias"], mean_z=rep["mean_z"], oof_bias=rep["oof_bias"],
                ))
    return rows


def select_capacity(rows: list[dict], train_blocks: int, seed: int = SEED) -> dict:
    candidates = [r for r in rows if r["train_blocks"] == train_blocks and r["seed"] == seed]
    if not candidates:
        raise ValueError(f"нет capacity curve для train_blocks={train_blocks}, seed={seed}")
    best_wcv = min(candidates, key=lambda r: (r["wcv"], r["rounds"]))
    best_last = min(candidates, key=lambda r: (r["fold_cal"][-1], r["wcv"], r["rounds"]))
    return dict(best_wcv=best_wcv, best_last_fold=best_last, decision=best_last)


def _aligned_average(exp_ids: list[str], out: str) -> dict:
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
    save_report(out, rep, extra=dict(description="S_02A 3-seed log-space average",
                                     seeds=exp_ids, n_seeds=len(exp_ids)))
    return dict(exp_id=out, wcv=rep["wcv"], fold_cal=rep["fold_cal"],
                fold_bias=rep["fold_bias"], mean_z=rep["mean_z"], oof_bias=rep["oof_bias"])


def _exp_for_seed(train_blocks: int, seed: int, rounds: int) -> str | None:
    found = _candidate(train_blocks, seed, rounds)
    return found[0] if found else None


def _diversity(exp_a: str, exp_b: str) -> dict:
    """Prediction difference after exact OOF alignment.

    STRATEGY_05 established Var(delta)=0.00712 as the seed-noise floor, so
    reporting only a raw correlation would overstate structural diversity.
    """
    from src.blend import aligned

    Z, y, _ = aligned([exp_a, exp_b])
    ly = np.log1p(y)
    return dict(exp_a=exp_a, exp_b=exp_b, var_delta=float(np.var(Z[0] - Z[1])),
                residual_corr=float(np.corrcoef(ly - Z[0], ly - Z[1])[0, 1]))


def _training_rows_last_fold(train_blocks: int) -> int:
    from src.features import features_cached, panel_users

    s = setup_for(train_blocks, 1, SEED)
    cuts = s.train_cutoffs(VAL_FOLDS_S1[-1])
    if train_blocks == 0:
        return sum(features_cached(T, None, True).height for T in cuts)
    return sum(panel_users(T, train_blocks).height for T in cuts)


def adversarial_population_auc(rounds: int = 250) -> dict:
    """Compare each training-panel population at the latest fold with the test panel."""
    from src.adversarial import adv_auc
    from src.data import load
    from src.features import feature_names, make_xy, to_np

    V = VAL_FOLDS_S1[-1]
    load()
    Xt, _ = make_xy(CUTOFF_TEST, None, 3, with_target=False, norm_long=True)
    feats = feature_names(Xt)
    At = to_np(Xt, feats)
    result = {}
    for tb in (1, 0):
        Xv, _ = make_xy(V, None, tb, norm_long=True)
        auc, drivers = adv_auc(to_np(Xv, feats), At, feats, rounds=rounds)
        result[str(tb)] = dict(auc=auc, drivers=drivers, n=Xv.height)
    path = ARTIFACTS / "sampleval_adversarial.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    return result


def summarize(rows: list[dict], out_name: str) -> dict:
    capacity = {str(tb): select_capacity(rows, tb) for tb in (1, 0)}
    decision_rounds = {tb: capacity[str(tb)]["decision"]["rounds"] for tb in (1, 0)}
    single = {tb: capacity[str(tb)]["decision"] for tb in (1, 0)}
    delta_single = single[0]["wcv"] - single[1]["wcv"]
    fold_delta_single = (np.asarray(single[0]["fold_cal"]) - np.asarray(single[1]["fold_cal"])).tolist()

    avg3 = {}
    for tb in (1, 0):
        exps = [_exp_for_seed(tb, seed, decision_rounds[tb]) for seed in (42, 43, 44)]
        if all(exps):
            avg3[tb] = _aligned_average(exps, f"SAMPLE-TB{tb}-AVG3-R{decision_rounds[tb]}")

    comparison = dict(
        decision_rule="round minimizing calibrated score on fold 2025-10-16",
        baseline_rounds=decision_rounds[1], variant_rounds=decision_rounds[0],
        seed42_delta_wcv=delta_single, seed42_fold_delta=fold_delta_single,
        seed42_wins=sum(x < 0 for x in fold_delta_single),
        seed42_last_fold_win=fold_delta_single[-1] < 0,
    )
    if len(avg3) == 2:
        fd = (np.asarray(avg3[0]["fold_cal"]) - np.asarray(avg3[1]["fold_cal"])).tolist()
        comparison.update(avg3_delta_wcv=avg3[0]["wcv"] - avg3[1]["wcv"],
                          avg3_fold_delta=fd, avg3_wins=sum(x < 0 for x in fd),
                          avg3_last_fold_win=fd[-1] < 0)
        d, wins, last = comparison["avg3_delta_wcv"], comparison["avg3_wins"], comparison["avg3_last_fold_win"]
        evidence = "avg3"
    else:
        d, wins, last = delta_single, comparison["seed42_wins"], comparison["seed42_last_fold_win"]
        evidence = "seed42"
    if d <= -0.002 and wins >= 3 and last:
        verdict = "ACCEPT"
    elif d <= -0.0005 and wins >= 3 and last:
        verdict = "CONTINUE"
    else:
        verdict = "REJECT"

    diversity = {
        "seed42_capacity_matched": _diversity(single[0]["exp_id"], single[1]["exp_id"]),
    }
    if len(avg3) == 2:
        diversity["avg3_capacity_matched"] = _diversity(avg3[0]["exp_id"], avg3[1]["exp_id"])
        if (ARTIFACTS / "oof_S1-DIST.npz").exists():
            diversity["avg3_variant_vs_dist"] = _diversity(avg3[0]["exp_id"], "S1-DIST")

    row_counts = {str(tb): _training_rows_last_fold(tb) for tb in (1, 0)}
    adv_path = ARTIFACTS / "sampleval_adversarial.json"
    adversarial = json.loads(adv_path.read_text(encoding="utf-8")) if adv_path.exists() else None
    summary = dict(
        capacity=capacity, comparison=comparison, avg3=avg3,
        decision_evidence=evidence, verdict=verdict, training_rows_last_fold=row_counts,
        row_increase=float(row_counts["0"] / row_counts["1"] - 1),
        diversity=diversity, adversarial=adversarial,
    )
    path = ARTIFACTS / f"sample_design_{out_name}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_csv(ARTIFACTS / f"sample_design_{out_name}_curve.csv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"saved {path}", flush=True)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                        for k, v in row.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    ap.add_argument("--train-blocks", nargs="+", type=int, choices=[0, 1], default=[1, 0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--rounds", type=int, default=max(DEFAULT_CURVE))
    ap.add_argument("--curve", nargs="+", type=int, default=list(DEFAULT_CURVE))
    ap.add_argument("--out", default="S1-SAMPLE-A")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--adversarial", action="store_true")
    a = ap.parse_args()
    if a.run:
        run_grid(a.train_blocks, a.seeds, a.rounds, a.curve, a.force)
    if a.adversarial:
        adversarial_population_auc()
    rows = collect_rows((1, 0), (42, 43, 44), a.curve)
    if rows:
        summarize(rows, a.out)


if __name__ == "__main__":
    main()
