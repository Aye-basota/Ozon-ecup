# Audit of excluded secondary summaries

This audit treats every scanned summary/navigation document as an object of
forensic review, never as a source of experiment facts. Every factual resolution
comes from `registry/experiments.jsonl`, primary experiment records, machine
artifacts/manifests, the submission audit, or the normalized leaderboard
chronology. For all documents below: **`used_for_facts=no`**.

## Scope and exclusions

Scanned exactly these 17 secondary documents from the source repository:

1. `STATE.md`
2. `HISTORY.md`
3. `README.md`
4. `DISCRIPTION.md`
5. `experiments/DISCRIPTION.md`
6. `experiments/README.md`
7. `experiments/INDEPENDENT_STAGE1_FINDINGS.md`
8. `PROBABLY_EXP.md`
9. `research/README.md`
10. `research/eda/README.md`
11. `research/rmsle_diagnostics/README.md`
12. `research/strategies/STRATEGIES_INDEX.md`
13. `research/strategy_1_results.md`
14. `research/strategy_comparison.md`
15. `research/strategy_NN_report.md`
16. `пайплайн сокомандника/friend_original/submission_STRONGEST_CURRENT/README.md`
17. `пайплайн сокомандника/provenance/PROJECT_STATE_AFTER_PHASE11.md`

`DISCRIPTION.md` and `experiments/DISCRIPTION.md` are exact SHA-256 duplicates
(`a3943710…2bf1`). They contain task/background text but no concrete experiment
result conflict. `research/eda/README.md` and the teammate production-bundle
README likewise yielded no concrete result conflict after primary/machine
comparison.

Not opened in this audit: `AGENTS.md`; `docs/superpowers/plans/*`;
`docs/superpowers/specs/*`; prompt/TODO files; `research/strategy_1.md`,
`research/strategy_2.md`, `research/strategy_NN_1.md`, and
`research/strategy_NN_2.md`. Those are instruction/design documents rather than
the requested result/comparison summaries. The strategy index was scanned only
because it was explicitly in scope, and none of its instructions or future
recommendations were followed.

## Result

The machine-readable file
`evidence/secondary_summary_conflicts.jsonl` contains **22 material rows**:

- 15 high-severity conflicts;
- 7 medium-severity conflicts.

The dominant patterns are:

- validation overgeneralization: a Team-A four-fold wCV protocol was repeatedly
  described as universal, while the normalized registry contains 42 distinct
  comparison classes;
- wrong comparison ancestry: L=180 was sometimes judged against old B0 although
  its actual parent was dense-cutoff S1-E02, reversing the sign of the causal
  comparison;
- materially stale current/final or artifact-presence labels after EXP-037 and
  EXP-067;
- diagnostic lookup/projection claims later weakened or rejected by controlled
  model experiments (three-block training, anniversary, renewal, global regime,
  distribution-head effect size);
- two unnamespaced numeric collisions (`EXP-057`, `EXP-058`) and one LB score
  (`EXP-015`/F4) lacking a unique submission binding;
- stage-verdict collapse in EXP-051, where a primary nested research gate and a
  fixed-weight production-support gate legitimately disagree.

## Reading rule

Rows classified as `stale_experiment_status` or `stale_current_solution_label`
do not imply that the secondary document was false on its original date. They
mean that it cannot be used as present-tense project memory after later primary
evidence appeared. Likewise, a diagnostic signal is retained as a diagnostic;
it is not promoted to a measured model gain when a later controlled experiment
produced a different endpoint.

No future ML recommendation was extracted or produced. No source file was
modified. As a final read-only check, the source repository's
`git status --porcelain=v1 -uall` remained byte-for-byte equal to the captured
pre-audit status (624 lines before and after).
