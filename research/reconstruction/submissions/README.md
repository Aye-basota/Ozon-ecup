# Submission registry

`registry.csv` is the final path-level forensic registry:

- 36 valid existing CSV paths: 25 main-worktree, 1 linked Strategy-2 copy, and
  10 teammate review-bundle files;
- 33 unique SHA-256 values and 3 duplicate paths;
- 20 exact recorded recipes, 3 recorded semantic recipes, 3 numerically
  reconstructed S04 recipes, and 10 producer-script semantic recipes;
- 0 artifacts with an unknown forensic recipe;
- 11 strong repository-internal score-to-artifact links, with no independent
  platform export.

`audit.csv` preserves the broader initial candidate audit, including manifests
and expected-but-missing paths. `inventory.csv` is the raw filename-oriented
inventory. Where an exact creation-time recipe did not survive, `registry.csv`
labels the reconstruction/evidence type instead of presenting it as original
metadata.

