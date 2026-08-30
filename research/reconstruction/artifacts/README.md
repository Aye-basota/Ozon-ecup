# Artifact index

Large source artifacts were not copied. `manifest.csv` records original path,
type/purpose, size, SHA-256, experiment association, and schema/array metadata
where practical. Linked-worktree artifacts are covered by
`evidence/worktree_artifacts_manifest.csv`; selected small JSON/CSV/log evidence
was copied under `evidence/worktree_artifacts/`.

Key indexes:

- `../registry/components.csv`: 614 component groups with OOF/test/model pairing;
- `exact_duplicates.csv`: 412 byte-identical groups in the combined unique-file
  inventory;
- `cross_worktree_duplicates.csv`: duplicated content across Git/worktree
  namespaces, including ordinary branch copies;
- `component_linkage_audit.csv`: report/run association audit;
- `orphan_prediction_resolution.csv`: the 18 initially unlinked prediction
  candidates and their code/report-backed resolution;
- `worktree_unique_files.csv`: files present in linked worktrees but not the main
  worktree by SHA-256.

Byte identity proves content identity, not that two experiment hypotheses are
duplicates. Experiment-level deduplication lives in
`../registry/deduplication.csv`.

