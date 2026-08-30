# Inventory

The main worktree inventory covers 3,599 non-Git/non-cache files. Six linked
worktrees add 1,293 content-unique paths relative to main. Inventories retain
path, size, timestamp, SHA-256, inferred file role, experiment association, and
schema metadata where practical.

Important files:

- `files.csv` and `source_checksums_sha256.csv`: main-worktree inventory;
- `worktrees.csv` and `worktree_files.csv`: six linked worktrees, excluding their
  shared `data` junction;
- `git_refs.csv` and `git_report_occurrences.csv`: branch/tag/report archaeology;
- `dataset_fingerprints.csv`: 21 normalized dataset/cache families;
- `report_linkage_audit.csv`: all 88 current-worktree report candidates;
- `excluded_interpretive_documents.csv`: 24 instruction/state/summary documents
  explicitly excluded as fact sources;
- `excluded_sources_used_for_facts.csv`: expected to contain zero data rows;
- `source_snapshot_before.csv` and `git_status_before.txt`: pre-audit integrity
  snapshot.

The raw train Parquet has 30,631,006 rows and SHA-256
`5f3aa...67c0`; the full exact hash is in `dataset_fingerprints.csv` and
`files.csv`.

