# Code index

`scripts.csv` inventories 222 code/config units in the main worktree.
`report_code_links.csv` contains explicit primary-report references, and
`script_linkage_audit.csv` separates:

- 100 scripts explicitly linked to a primary report;
- 4 shared pipeline dependencies;
- 118 scripts without an explicit primary-report link.

The final category means “orphan candidate,” not “unused.” File names alone were
not used to assign experiments.

Historical branch-only code referenced by reports is preserved in 136 immutable
files under `snapshots/`, indexed by `git_referenced_code.csv`. No branch was
checked out over the source worktree.

