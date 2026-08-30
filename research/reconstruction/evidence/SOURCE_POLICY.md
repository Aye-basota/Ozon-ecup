# Evidence policy

This repository is a forensic reconstruction of `OZON-E-CUP`. It contains no
new model research, training, inference, tuning, or submission generation.

## Evidence order

Facts are recorded in this order of preference:

1. prediction, OOF, checkpoint, and submission artifacts;
2. metrics written directly by a run;
3. experiment manifests;
4. concrete configuration and experiment code;
5. logs;
6. a primary report for one experiment;
7. secondary descriptions, only as non-authoritative context.

`evidence_strength` describes the strongest surviving source. A submission and
score linked inside the repository are not called independently verified unless
an export from the competition platform is present.

## Excluded interpretive sources

Agent instructions, state files, histories, roadmaps, TODOs, strategy indexes,
master/executive summaries, and recommendation documents were inventoried but
were not used to establish experiment facts. Their paths are listed in
`inventory/excluded_interpretive_documents.csv` with `used_for_facts=no`.

Primary reports about a single concrete experiment remain admissible evidence.
If such a report quotes an excluded summary, that particular claim is marked as
provenance-contaminated and is not promoted to fact without independent support.

`secondary_summary_conflicts.jsonl` uses selected excluded summaries only as
objects whose claims are tested against primary/machine evidence. Its rows carry
`used_for_facts=no`; they never populate experiment results or fill missing data.

## Conflicts and comparability

Conflicting values are retained; no value is selected by guesswork. Local CV,
weighted CV, calibrated OOF, standalone scores, ensemble scores, public LB,
simulation checks, and different fold/train-coverage protocols remain separate.
`comparison_class` is the machine-readable guardrail for this separation.

## Copies and checksums

Large source artifacts are referenced by path and SHA-256 instead of copied.
Small evidence files and immutable Git snapshots may be copied into `evidence/`.
An identical checksum proves byte identity, not semantic equivalence.
