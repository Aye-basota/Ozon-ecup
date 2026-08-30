# Contradictions and evidence gaps

`registry.csv/jsonl` retains report-vs-artifact tensions, metric/protocol
comparability warnings, missing provenance, checksum mismatches, and material ID
collisions. Rows are categorized so an audit caveat is not counted as a serious
numeric contradiction.

The current registry contains 106 primary/machine forensic rows and 22 rows
from the excluded-summary conflict audit. Every latter row is marked
`secondary-only; used_for_facts=no`; see
`../evidence/secondary_summary_conflicts_audit.md`.

`manifested_submissions_missing.csv` contains two recipes whose expected CSV was
not materialized in the available worktrees. `json_parse_errors.csv` records
machine-evidence parse failures; it contains no failure rows in this audit.

No conflict was resolved by guessing. The normalized experiment card preserves
both the measured FACTS and the original INTERPRETATION.
