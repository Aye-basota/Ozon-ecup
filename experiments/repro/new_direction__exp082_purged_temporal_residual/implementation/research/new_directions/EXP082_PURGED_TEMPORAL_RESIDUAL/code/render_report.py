from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXP = Path(__file__).resolve().parents[1]


def load_json(name: str):
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def f(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    verdict = load_json("verdict.json")
    leakage = load_json("leakage_assertions.json")
    fidelity = load_json("baseline_fidelity_audit.json")
    feature_gate = load_json("residual_feature_fidelity_gate.json")
    results = load_json("purged_results.json")
    headroom = load_json("mathematical_headroom.json")
    protocol = pd.read_csv(EXP / "protocol_comparison.csv")
    primary = results["LightGBM_AB_mean"]
    boot = primary["bootstrap"]
    all_rhos = [row["rho_post_projection"] for result in results.values() for row in result["folds"]]

    lines = [
        "# EXP082 — Fully Purged Temporal Residual Validation",
        "",
        "## Verdict",
        "",
        f"**{verdict['verdict']}**. Statistical verdict of the conservative core-162 diagnostic: "
        f"**{verdict['statistical_verdict_core_subset']}**.",
        "",
        "The four-fold temporal protocol itself is valid, and the production-like baseline passes its fidelity gate. "
        "However, the exact frozen EXP081 nonlinear learner is not reproducible on the requested dates: its full 40-model "
        "prediction bank and three auxiliary structural channels have saved OOF only on the old 14-day canonical folds. "
        "The new-fold diagnostic uses 162 cutoff-reproducible columns rather than the frozen 200-column feature matrix. "
        "No future prediction, imputation, distillation, leaderboard signal, or target-derived activity was used.",
        "",
        "## Purged fold construction",
        "",
        "| Cutoff | Target window | N | Spacing | Previous labels known | Outside survivorship interval |",
        "|---|---|---:|---:|:---:|:---:|",
    ]
    for row in leakage["rows"]:
        spacing = "—" if row["spacing_from_previous_days"] is None else str(row["spacing_from_previous_days"])
        lines.append(
            f"| {row['cutoff']} | ({row['cutoff']}, {row['target_end']}] | {row['n']:,} | {spacing} | "
            f"{'PASS' if row['previous_target_known'] else 'FAIL'} | "
            f"{'PASS' if row['final_outside_survivorship_interval'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Result: 4 folds / 3 genuine purged transitions. For every transition "
        "`target_end(previous) <= cutoff(current)`; the last target end is 2025-11-15.",
        "",
        "## Production baseline fidelity",
        "",
        "Frozen composition: `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-S42 + 0.225 ETX-S42`. "
        "SEQ/ETX use the allowed frozen single-seed approximation; no weights or model settings were tuned.",
        "",
        f"- correlation rebuilt S42 baseline vs exact AVG3 at 2025-10-16: `{fidelity['corr_rebuilt_vs_exact_AVG3']:.9f}` "
        f"(gate >= 0.995: {'PASS' if fidelity['corr_gate_0_995'] else 'FAIL'});",
        f"- correlation vs composition-matched EXP076 proxy: `{fidelity['corr_rebuilt_vs_composition_matched']:.9f}`;",
        f"- RMS(rebuilt − exact AVG3): `{fidelity['RMS_rebuilt_minus_AVG3']:.6f}`;",
        f"- candidate-rho geometry gate <= 0.003: {'PASS' if fidelity['fidelity_pass'] else 'FAIL'}.",
        "",
        "| Diagnostic correction | rho rebuilt | rho EXP076 matched | absolute difference | gate |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in fidelity["candidate_rho_rows"]:
        lines.append(
            f"| {row['candidate']} | {f(row['rho_rebuilt_single_seed'])} | "
            f"{f(row['rho_composition_matched_EXP076'])} | {f(row['absolute_difference'])} | "
            f"{'PASS' if row['passes_0_003'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Every component artifact is separately audited for SHA256 parity, cutoff-safe last training target, "
        "unchanged config, and runtime below six hours in `production_component_audit.csv`.",
        "",
        "## Residual learner reproduction",
        "",
        "LightGBM A and B use the exact frozen depth/leaves/regularization/tree-count recipes from EXP081; "
        "A/B mean is their arithmetic mean. Preprocessing on common state/RFM and cohort columns is frozen. "
        "The feature fidelity gate nevertheless fails because bank-wide disagreement and interactions cannot be "
        "reconstructed exactly without the missing historical 41-model predictions.",
        "",
        f"- EXP081 feature count: `{feature_gate['exp081_feature_count']}`;",
        f"- cutoff-reproducible diagnostic feature count: `{feature_gate['reproduced_feature_count']}`;",
        f"- exact feature/preprocessing fidelity: `{'PASS' if feature_gate['pass'] else 'FAIL'}`;",
        f"- derived disagreement formula match: `{'PASS' if feature_gate['derived_disagreement_formula_match'] else 'FAIL'}`.",
        "",
        "Therefore the core-162 metrics below are conservative diagnostic evidence, not an exact reproduction "
        "capable of authorizing STRONG_GO or a TEST submission.",
        "",
        "## Purged results",
        "",
        "Primary diagnostic candidate: LightGBM A/B mean. Deployable amplitude for each row was fitted only "
        "from user-disjoint cross-fitted predictions on fully available earlier folds.",
        "",
        "| Validation cutoff | Train folds | rho raw | rho vs strong residual | rho post-projection | b | G | oracle amp | deployed amp | ΔMSE | ΔRMSLE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary["folds"]:
        train = ", ".join(row["train_folds"])
        lines.append(
            f"| {row['cutoff']} | {train} | {f(row['rho_raw'])} | {f(row['rho_vs_strong_residual'])} | "
            f"{f(row['rho_post_projection'])} | {row['b']:.6g} | {row['G']:.6g} | "
            f"{f(row['oracle_amplitude'], 5)} | {f(row['deployable_amplitude'], 5)} | "
            f"{row['Delta_MSE']:.6g} | {row['Delta_RMSLE']:.6g} |"
        )
    lines += [
        "",
        f"Recency-weighted post-projection rho: **{f(primary['weighted_purged_post_projection_rho'])}**. "
        f"Latest rho: **{f(primary['latest_purged_post_projection_rho'])}**. "
        f"Positive transitions: **{primary['positive_rho_transitions']}/3**.",
        "",
        f"Nested ΔMSE: **{primary['nested_Delta_MSE']:.8f}**; nested ΔRMSLE: "
        f"**{primary['nested_Delta_RMSLE']:.8f}**. Cluster-bootstrap 95% CI for ΔMSE: "
        f"**[{boot['CI95_Delta_MSE'][0]:.8f}, {boot['CI95_Delta_MSE'][1]:.8f}]**, "
        f"`P(ΔMSE < 0)={boot['P_Delta_MSE_lt_0']:.4f}`.",
        "",
        "Leave-one-transition-out metrics are saved in `bootstrap.json`; they are used to assess dependence on one fold.",
        "",
        "## Same-period vs ordered vs fully-purged",
        "",
        "| Protocol | Candidate fidelity | rho | latest rho | ΔMSE | P(gain) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in protocol.iterrows():
        lines.append(
            f"| {row['Protocol']} | {row['candidate']} | {f(row['rho'])} | {f(row['latest_rho'])} | "
            f"{row['Delta_MSE']:.6g} | {row['P_gain']:.4f} |"
        )
    lines += [
        "",
        "Because Protocol C lacks exact full-200 feature fidelity, this table is directional rather than a strictly "
        "identical-candidate causal comparison. It does not use same-period targets as temporal evidence.",
        "",
        "## Projection / novelty",
        "",
        "| Cutoff | RMS(u_raw) | RMS(u_perp) | perp fraction | second-pass RMS | relative error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary["folds"]:
        lines.append(
            f"| {row['cutoff']} | {f(row['RMS_u_raw'])} | {f(row['RMS_u_perp'])} | "
            f"{f(row['perp_fraction'])} | {row['second_pass_projection_error_RMS']:.3e} | "
            f"{row['second_pass_relative_error']:.3e} |"
        )
    lines += [
        "",
        "Only `u_perp` is used for predictive-signal claims and deployed-gain arithmetic.",
        "",
        "## Mathematical headroom",
        "",
        f"Required gap ΔMSE: **{headroom['required_Delta_MSE_gain']:.10f}**. "
        f"Weighted purged rho²: **{headroom['achieved_purged_rho_squared']:.8f}**.",
        "",
        "| Headroom definition | MSE gain | Fraction of required gap |",
        "|---|---:|---:|",
        f"| Correlation-only mathematical ceiling | {headroom['correlation_only_MSE_headroom']:.8f} | {100*headroom['correlation_only_fraction_of_gap']:.2f}% |",
        f"| Nested deployed point gain | {headroom['nested_point_MSE_gain']:.8f} | {100*headroom['nested_point_fraction_of_gap']:.2f}% |",
        f"| Robust 95% gain | {headroom['robust_95pct_MSE_headroom']:.8f} | {100*headroom['robust_95pct_fraction_of_gap']:.2f}% |",
        "",
        f"Maximum individual purged post-projection rho across A/B/mean diagnostics: **{max(all_rhos):.6f}**.",
        "",
        "## Output",
        "",
        "No `SUBMIT_EXP082_PURGED_RESIDUAL.csv` was created. Exact residual-feature fidelity and STRONG_GO are both "
        "required before TEST inference; leaderboard fitting or automatic submission was never used.",
        "",
        "## Final conclusion",
        "",
        "The requested four-fold purged clock is technically valid and the rebuilt production baseline is faithful. "
        "The exact EXP081 nonlinear mechanism cannot be adjudicated as requested because its full historical feature bank "
        "does not exist on the 35-day folds, and rebuilding that bank would require replaying dozens of separate legacy "
        "model pipelines rather than the authorized production recipe. The core-162 result quantifies the available "
        "temporal evidence but cannot upgrade to STRONG_GO.",
        "",
        "To continue scientifically, the next useful information channel is not another learner on the same schema: use "
        "additional future labels on an independently frozen cohort, a genuinely new raw field, entity relations, or "
        "rules-permitted external data. For an exact rerun of this particular hypothesis, first materialize the frozen "
        "40-model bank predictions at all four purged cutoffs; do not impute them from future canonical folds.",
        "",
        "No leaderboard data was read or used.",
    ]
    (EXP / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(EXP / "REPORT.md")


if __name__ == "__main__":
    main()
