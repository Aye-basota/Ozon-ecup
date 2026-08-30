# OZON E-CUP research memory — clean reconstruction

This repository is the normalized, evidence-backed memory of the neighboring
`OZON-E-CUP` hackathon repository. The reconstruction performed no training,
inference, tuning, weight search, or submission generation. The source
repository was treated as read-only.

## Headline inventory

- 124 primary experiment-report rows, all represented in the registry;
- 1 additional machine-only experiment (unnumbered S04);
- 13 runtime-backed teammate review/training units;
- 138 central registry rows and 134 novelty-level units after collapsing four
  exact-document/replay rows;
- 12 bottom-up research families;
- 11 duplicate/rerun/reuse clusters, with all related records preserved;
- 1,134 granular machine metric/manifest records;
- 614 OOF/test/model component groups;
- 11 repository-internal score-to-existing-submission links and zero
  independently platform-verified leaderboard exports.

These counts have different grains by design. A primary report, a machine-only
hypothesis, a top-level review run, a child training unit, a fold, and a weight
sweep are not silently equated.

## Start here

Read [`reports/START_HERE.md`](reports/START_HERE.md), then use:

- [`registry/experiments.csv`](registry/experiments.csv) for filtering;
- [`registry/experiments.jsonl`](registry/experiments.jsonl) for lossless nested
  FACTS/INTERPRETATION;
- [`experiments/normalized/`](experiments/normalized/) for one readable card per
  registry unit;
- [`ensembles/SOLUTION_ANCESTRY.md`](ensembles/SOLUTION_ANCESTRY.md) for the
  decision genealogy and parallel pipelines;
- [`reports/REPOSITORY_RECONSTRUCTION.md`](reports/REPOSITORY_RECONSTRUCTION.md)
  for scope, counts, contradictions, missing evidence, and completeness.

## Repository map

| Path | Contents |
|---|---|
| `registry/` | central experiments, run metrics, components, families, dedup, ID collisions |
| `experiments/` | immutable primary-report snapshots by namespace plus normalized cards |
| `families/` | descriptive family pages; scores grouped only by compatible protocol |
| `baselines/` | per-research-line baseline chronology |
| `submissions/` | artifact audit and recipe/provenance registry |
| `leaderboard/` | confirmed repository-internal chronology and unverified claims |
| `ensembles/` | recipes, ancestry graph, and directed lineage edges |
| `artifacts/` | checksummed manifests, duplicate groups, component linkage, orphan audit |
| `code_index/` | script inventory, report links, and immutable historical code snapshots |
| `evidence/` | machine evidence copies and forensic audits |
| `contradictions/` | conflicts, caveats, provenance gaps, and missing manifested files |
| `inventory/` | full file/dataset/Git/worktree inventory and exclusion audit |
| `reports/` | reconstruction report, completeness audit, and source-integrity proof |
| `tools/` | deterministic reconstruction/audit scripts; no ML execution |

## Evidence and comparability

Evidence priority is artifact → run-written metric → manifest → concrete
config/code → log → primary experiment report. Instruction/state/roadmap/master
summary documents were inventoried but not used as fact sources. See
[`evidence/SOURCE_POLICY.md`](evidence/SOURCE_POLICY.md).

Seventeen scoped old summary/navigation documents were also checked against
canonical evidence. Their 22 material discrepancies are explicitly marked
`secondary-only; used_for_facts=no` in the contradiction registry.

`comparison_class` prevents local CV, weighted CV, standalone scores, ensemble
scores, public LB, simulation checks, and incompatible folds/train coverage from
being pooled. Family best/median values are dictionaries keyed by comparison
class, never a cross-protocol global ranking.

Missing values are the literal `unknown`. A recorded LB score linked inside the
repository is still not independent platform confirmation unless an upload or
platform export survives; none does here.

## Rebuild the derived indexes

From this repository:

```powershell
python tools/build_registry.py C:\Users\Admin\Desktop\research_clean
python tools/check_completeness.py C:\Users\Admin\Desktop\research_clean
```

The source-integrity verifier is intentionally separate because it re-hashes
tens of gigabytes of source/worktree content:

```powershell
python tools/verify_source_unchanged.py C:\Users\Admin\Desktop\research_clean C:\Users\Admin\Desktop\OZON-E-CUP
```

The final recorded run is PASS: main 3,599/3,599 files and all 3,549 inventoried
linked-worktree files retained their pre-audit size/SHA-256 and Git status.
