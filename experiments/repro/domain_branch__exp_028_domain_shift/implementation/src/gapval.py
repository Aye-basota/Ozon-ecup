"""STRATEGY_01: gap-axis validation with fixed training-set size.

The runner changes only two properties of the training cutoff list: the minimum
train->validation gap and the number of retained cutoffs.  Features, target,
validation panels and model parameters remain those of the registered probes.

Examples:
  python -m src.gapval --run
  python -m src.gapval --summarize
  python -m src.gapval --run --probes e10 e03a --gaps 120 --seeds 43 44 --rounds 200
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, SEED, VAL_FOLDS_S1
from src.report import load_report
from src.train import Setup, run


PROBES = {
    "e10": dict(model="direct", L=0, norm_long=True),
    "e03a": dict(model="direct", L=180, norm_long=False),
    "e02": dict(model="direct", L=0, norm_long=False),
    "dist": dict(model="dist", L=0, norm_long=True),
}
DEFAULT_GAPS = (30, 60, 90, 120)
DEFAULT_CURVE = (50, 75, 100, 150, 200, 250, 300)


def experiment_id(probe: str, n_cutoffs: int, gap: int, seed: int) -> str:
    return f"GAP-{probe.upper()}-K{n_cutoffs}-G{gap:03d}-S{seed}"


def setup_for(probe: str, gap: int, n_cutoffs: int, rounds: int, seed: int,
              vals=None) -> Setup:
    spec = PROBES[probe]
    return Setup(**spec, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                 cutoffs="all", min_gap=gap, n_cutoffs=n_cutoffs, rounds=rounds,
                 params={"seed": seed}, vals=vals)


def run_grid(probes, gaps, seeds, n_cutoffs: int, rounds: int, curve, force: bool,
             vals=None) -> None:
    snapshots = tuple(sorted(set(int(x) for x in curve if int(x) <= rounds)))
    for probe in probes:
        for gap in gaps:
            for seed in seeds:
                exp = experiment_id(probe, n_cutoffs, gap, seed)
                report = ARTIFACTS / f"report_{exp}.json"
                oof = ARTIFACTS / f"oof_{exp}.npz"
                if not force and report.exists() and oof.exists():
                    print(f"[skip] {exp}: artifacts already exist", flush=True)
                    continue
                s = setup_for(probe, gap, n_cutoffs, rounds, seed, vals=vals)
                run(exp, f"S_01 {probe}, k={n_cutoffs}, G={gap}, seed={seed}", s,
                    save_model_feats=True, snap=snapshots, snap_save=True, no_log=True)


def _candidate_report(base: str, rounds: int) -> tuple[str, dict] | None:
    """Load a round snapshot, or the base run when it was trained for that round."""
    snap = ARTIFACTS / f"report_{base}-R{rounds}.json"
    if snap.exists():
        return f"{base}-R{rounds}", load_report(f"{base}-R{rounds}")
    p = ARTIFACTS / f"report_{base}.json"
    if p.exists():
        rep = load_report(base)
        if int(rep.get("params", {}).get("rounds", -1)) == rounds:
            return base, rep
    return None


def _actual_gaps(probe: str, gap: int, n_cutoffs: int, rounds: int, seed: int) -> list[int]:
    s = setup_for(probe, gap, n_cutoffs, rounds, seed)
    return [(V - max(s.train_cutoffs(V))).days for V in VAL_FOLDS_S1]


def collect_rows(probes, gaps, seeds, n_cutoffs: int, curve) -> list[dict]:
    rows = []
    for probe in probes:
        for gap in gaps:
            for seed in seeds:
                base = experiment_id(probe, n_cutoffs, gap, seed)
                for rounds in curve:
                    found = _candidate_report(base, int(rounds))
                    if found is None:
                        continue
                    exp, rep = found
                    actual = _actual_gaps(probe, gap, n_cutoffs, int(rounds), seed)
                    rows.append(dict(
                        exp_id=exp, probe=probe, requested_gap=int(gap),
                        actual_gap=int(round(np.mean(actual))), n_cutoffs=n_cutoffs,
                        seed=int(seed), rounds=int(rounds), wcv=rep["wcv"],
                        fold_cal=rep["fold_cal"], fold_bias=rep["fold_bias"],
                        fold_scores=rep["fold_scores"], mean_z=rep["mean_z"],
                    ))
    return rows


def select_capacity(rows: list[dict], seed: int = SEED) -> list[dict]:
    selected = []
    keys = sorted({(r["probe"], r["requested_gap"]) for r in rows if r["seed"] == seed})
    for key in keys:
        candidates = [r for r in rows if (r["probe"], r["requested_gap"]) == key
                      and r["seed"] == seed]
        if candidates:
            selected.append(min(candidates, key=lambda r: (r["wcv"], r["rounds"])))
    return selected


def _slope(x, y) -> float:
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def _spearman(xs, ys) -> float | None:
    if len(xs) < 2:
        return None
    rx = np.argsort(np.argsort(np.asarray(xs, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(ys, float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _reference_wcv(probe: str) -> tuple[str, float] | None:
    refs = {"e10": "S1-SEED42", "e03a": "S1-E03a", "e02": "S1-E02", "dist": "S1-DIST"}
    try:
        exp = refs[probe]
        try:
            rep = load_report(exp)
        except FileNotFoundError:
            from src.report import from_oof
            rep = from_oof(exp)
        return exp, float(rep["wcv"])
    except (FileNotFoundError, TypeError):
        return None


def _variant_c_identity() -> list[dict]:
    """Audit the pre-registered f4 control using the exact production cutoff sets.

    The production f4 change removes the five cutoffs after 2025-09-11.  For the
    2025-10-16 fold those dates are ineligible already, at both requested gaps;
    therefore the two local training sets are identical and no model run can
    make this control informative.
    """
    V = dt.date(2025, 10, 16)
    full = Setup(L=0, min_history=90).grid()
    f4 = full[:-5]
    out = []
    for gap in (30, 60):
        a = [T for T in full if T + dt.timedelta(days=max(30, gap)) <= V]
        b = [T for T in f4 if T + dt.timedelta(days=max(30, gap)) <= V]
        # This is the only literal local analogue of "remove the five freshest
        # eligible cutoffs".  It is intentionally reported separately: unlike
        # production f4, it changes the validation training sample at both gaps
        # and therefore cannot reproduce the registered LB intervention.
        local_drop_five = a[:-5]
        out.append(dict(requested_gap=gap, full_n=len(a), f4_n=len(b),
                        identical=a == b, latest=a[-1].isoformat(),
                        local_drop_five_n=len(local_drop_five),
                        local_drop_five_identical=local_drop_five == a,
                        interpretation=(
                            "production full/f4 sets are identical after validation eligibility; "
                            "a local five-cutoff drop is a different intervention"
                        )))
    return out


def summarize(rows: list[dict], n_cutoffs: int, out_name: str) -> dict:
    selected = select_capacity(rows)
    by = {(r["probe"], r["requested_gap"]): r for r in selected}

    # Registered output shape: every metric is addressable by the explicit
    # <validation cutoff>|G<requested gap> key, while reports remain ordinary
    # project reports whose experiment id already contains G.
    fold_gap_metrics = {}
    for r in selected:
        probe_rows = fold_gap_metrics.setdefault(r["probe"], {})
        for V, score, bias, raw in zip(
                VAL_FOLDS_S1, r["fold_cal"], r["fold_bias"], r["fold_scores"]):
            probe_rows[f"{V.isoformat()}|G{r['requested_gap']}"] = dict(
                actual_gap=r["actual_gap"], rounds=r["rounds"],
                rmsle_cal=score, rmsle_raw=raw, bias=bias)

    slopes = {}
    for probe in sorted({r["probe"] for r in selected}):
        rs = [by[(probe, g)] for g in DEFAULT_GAPS if (probe, g) in by]
        if len(rs) < 2:
            continue
        x = [r["actual_gap"] for r in rs]
        fold_slopes = [_slope(x, [r["fold_bias"][i] for r in rs]) for i in range(4)]
        slopes[probe] = dict(
            mean_bias_per_gap=[float(np.mean(r["fold_bias"])) for r in rs],
            weighted_bias_per_gap=[float(np.average(r["fold_bias"], weights=FOLD_WEIGHTS_S1))
                                   for r in rs],
            actual_gaps=x, mean_slope=_slope(x, [np.mean(r["fold_bias"]) for r in rs]),
            weighted_slope=_slope(
                x, [np.average(r["fold_bias"], weights=FOLD_WEIGHTS_S1) for r in rs]),
            fold_slopes=fold_slopes,
        )

    fixed_slopes = {}
    for probe in sorted({r["probe"] for r in rows}):
        fixed_slopes[probe] = {}
        for rounds in sorted({r["rounds"] for r in rows if r["probe"] == probe}):
            rs = [r for r in rows if r["probe"] == probe and r["rounds"] == rounds
                  and r["seed"] == SEED]
            by_gap = {r["requested_gap"]: r for r in rs}
            if not all(g in by_gap for g in DEFAULT_GAPS):
                continue
            ordered = [by_gap[g] for g in DEFAULT_GAPS]
            x = [r["actual_gap"] for r in ordered]
            fixed_slopes[probe][str(rounds)] = dict(
                mean_slope=_slope(x, [np.mean(r["fold_bias"]) for r in ordered]),
                weighted_slope=_slope(
                    x, [np.average(r["fold_bias"], weights=FOLD_WEIGHTS_S1)
                        for r in ordered]),
                fold_slopes=[_slope(x, [r["fold_bias"][i] for r in ordered])
                             for i in range(4)],
            )

    model_delta = []
    for gap in DEFAULT_GAPS:
        if ("e10", gap) in by and ("e03a", gap) in by:
            a, b = by[("e10", gap)], by[("e03a", gap)]
            model_delta.append(dict(
                requested_gap=gap, actual_gap=a["actual_gap"],
                e10_rounds=a["rounds"], e03a_rounds=b["rounds"],
                e10=a["wcv"], e03a=b["wcv"], delta_e03a_minus_e10=b["wcv"] - a["wcv"],
                fold_delta=(np.asarray(b["fold_cal"]) - np.asarray(a["fold_cal"])).tolist(),
            ))

    gcv = []
    for probe in sorted({r["probe"] for r in selected}):
        r = by.get((probe, 120))
        if r is None:
            continue
        ref = _reference_wcv(probe)
        gcv.append(dict(probe=probe, rounds=r["rounds"], gcv=r["wcv"],
                         fold_cal=r["fold_cal"], fold_std=float(np.std(r["fold_cal"])),
                         reference_exp=ref[0] if ref else None,
                         reference_wcv=ref[1] if ref else None,
                         gcv_minus_wcv=r["wcv"] - ref[1] if ref else None))

    common = [p for p in ("e10", "e03a", "e02", "dist")
              if (p, 30) in by and (p, 120) in by]
    rank = dict(
        probes=common,
        gap30=[by[(p, 30)]["wcv"] for p in common],
        gap120=[by[(p, 120)]["wcv"] for p in common],
    )
    rank["spearman"] = _spearman(rank["gap30"], rank["gap120"])

    summary = dict(
        n_cutoffs=n_cutoffs, capacity_rule="argmin weighted calibrated fold score per probe and gap",
        rows=len(rows), selected=selected, fold_gap_metrics=fold_gap_metrics, bias_slopes=slopes,
        bias_slopes_fixed_round=fixed_slopes, e03a_vs_e10=model_delta,
        gcv=gcv, rank_correlation=rank, variant_c_cutoff_identity=_variant_c_identity(),
        notes=[
            "requested G values map to actual gaps imposed by the 7-day grid",
            "gCV is a secondary criterion: every G=120 arm uses only five cutoffs",
            "gCV-wCV references use the registered production-capacity experiment for each probe",
        ],
    )
    path = ARTIFACTS / f"gap_axis_{out_name}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_csv(ARTIFACTS / f"gap_axis_{out_name}_curve.csv", rows)
    _write_csv(ARTIFACTS / f"gap_axis_{out_name}_selected.csv", selected)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"saved {path}", flush=True)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="train missing grid arms")
    mode.add_argument("--summarize", action="store_true", help="summarize existing artifacts")
    ap.add_argument("--probes", nargs="+", choices=list(PROBES), default=list(PROBES))
    ap.add_argument("--gaps", nargs="+", type=int, default=list(DEFAULT_GAPS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--n-cutoffs", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=max(DEFAULT_CURVE))
    ap.add_argument("--curve", nargs="+", type=int, default=list(DEFAULT_CURVE))
    ap.add_argument("--out", default="S1-GAPAXIS")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--val", nargs="*", default=None,
                    help="explicit validation cutoffs; use with --no-summary for partial controls")
    ap.add_argument("--no-summary", action="store_true")
    a = ap.parse_args()
    if a.run:
        vals = [dt.date.fromisoformat(v) for v in a.val] if a.val else None
        run_grid(a.probes, a.gaps, a.seeds, a.n_cutoffs, a.rounds, a.curve, a.force, vals)
    if a.no_summary:
        return
    rows = collect_rows(a.probes, a.gaps, a.seeds, a.n_cutoffs, a.curve)
    summarize(rows, a.n_cutoffs, a.out)


if __name__ == "__main__":
    main()
