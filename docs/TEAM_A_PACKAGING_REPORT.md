# Team-A packaging report

Date: 2026-08-31. Branch: `team-a`. Base commit:
`80e911f6b3dae0bac66e6351af5b40b5e8132dc9`.

## Source locations

The forensic scan covered the requested roots and every related source root
discovered through paths, manifests and reports:

- `C:\Users\Admin\Desktop\e-cup-research-clean`;
- `C:\Users\Admin\Desktop\OZON-E-CUP`;
- `C:\Users\Admin\Desktop\submission_geometry_research`;
- `C:\Users\Admin\Downloads` (competition-matched files only);
- `C:\Users\Admin\Desktop\latest_pipeline_bundle`;
- `C:\Users\Admin\Desktop\research_clean`;
- five historical OZON worktrees: calendar, domain, global-regime/occurrence,
  renewal and strategy-2;
- `C:\Users\Admin\Desktop\OZON-ECUP2-WORK` raw/audit workspace.

Original files were neither moved nor deleted.

## Inventory results

- inventoried files: **8,022**;
- unique SHA256 values: **5,340**;
- byte-identical duplicates: **2,682**;
- hash failures: **0**;
- unified experiment/report lineage rows: **95**;
- included unique source/report/manifest files: **1,985**, plus the canonical
  `exp_001…exp_071` report series and selected final artifacts.

One copied historical teammate script containing an unresolved merge marker was
left inventory-only; its original file and SHA remain preserved outside this
branch.

Large raw, model, OOF, TEST and cache artifacts remain external. Every located
source path, type, experiment, destination, SHA, duplicate status and known Git
commit is recorded in `docs/TEAM_A_SOURCE_INVENTORY.csv`.

## Model families in the final solution

- Team-A tabular: CAP, UNC and DIST (LightGBM, including a 16-bin distribution head);
- Team-A neural: SEQ temporal encoder and ETX sparse-event transformer;
- Team-B: LightGBM regression/classification, post-order and behavior
  distribution heads, XGBoost and CatBoost;
- prediction-space ancestors of JOINT: submission geometry, ORTH and A1/A2/EXP075.

## Final solution status

### SUBMIT_STRONGEST55_TEAMB45

- frozen components, exact builder, weights, clipping, level alignment and
  reference CSV: **complete**;
- expected/result SHA: `1ce85203…a14fb4`;
- precomputed byte-identical rebuild: **YES**;
- prior Team-B raw retrain: **byte-identical**;
- full historical STRONGEST retrain: **not byte-identical** because three booster
  weights/exact feature matrices and the SEQ-01 seed-42 checkpoint were not retained;
- actual final LB: **not found**; 1.64823 is explicitly retained as a forecast.

### SUBMIT_JOINT86_TEAMB14

- exact 86/14 builder, log-space formula, clipping, level alignment, both frozen
  components and reference CSV: **complete**;
- expected/result SHA: `85d9cd64…dac02`;
- precomputed byte-identical rebuild: **YES**;
- recorded public LB: **1.6458200196207617**, source copied from the teammate request;
- 86% Team-A component: `SUBMIT_JOINT_V2`, SHA `211879cb…33cba`, recorded public
  LB 1.6459363044782171;
- upstream JOINT_V2 generation script: **PROVENANCE_INCOMPLETE**. Geometry,
  ORTH, A1/A2, EXP075 and EXP089 evidence is retained, but no script producing
  the exact SHA was found. No synthetic provenance was invented.

## Canonical repository layout

```text
README.md
requirements.txt
configs/ + config/
src/
scripts/
experiments/
research/
  legacy_team_a/
  new_directions/
  provenance/
  reconstruction/
  submission_geometry/
  worktree_snapshots/
reproducibility/
  SUBMIT_STRONGEST55_TEAMB45/
  SUBMIT_JOINT86_TEAMB14/
submissions/
docs/
```

No empty architecture-only directories were added.

## Audit and Git status

The exact smoke audit checks imports, row/ID alignment, finite/nonnegative
predictions, SHA256, maximum absolute prediction difference and RMS log-space
difference. Its machine-readable output is `docs/FINAL_REPRODUCTION_AUDIT.json`.

The final commit SHA and push URL are intentionally reported by `git log -1`
and the final task response: a commit cannot contain its own SHA without
changing that SHA. Push status is filled by the delivery step, not inferred.
