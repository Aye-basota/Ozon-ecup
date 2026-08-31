# EXP074 — SEQ Temporal DIST16 Head

## Verdict: `TECHNICAL_BLOCK`

`EXP074` was free in the current and historical experiment registries. Reconnaissance then
found a mandatory precondition failure: none of the four exact fold-specific historical
plain `SEQ-01` seed-42 checkpoints that produced the current `SEQ-AVG3` OOF member were
saved. The exact historical seed-42 production checkpoint is also absent. Per the
experiment protocol, the run stopped before fitting any head.

This is not a compute failure. Historical OOF predictions exist, but a final embedding
cannot be reconstructed from a scalar prediction. Retraining seed 42 or substituting
`SEQ-D3A`, `SEQ-03A`, `DETSEQ01`, or the compiled control would change the frozen encoder
and invalidate the stated hypothesis.

## Encoder parity

Status: **NOT RUN — exact checkpoint unavailable**.

The exact representation and hook were recovered: historical plain `SEQ-01` is the
17-channel, hidden-64, eight-block dilated TCN in `src/seq.py`; the head input is
`concat(last, mean, max)` of the final normalized hidden sequence, width 192.
`src/seq_cond.py` already exposes this representation under `torch.no_grad`.

The authoritative weight archive contains no `model_SEQ-01-S42-V*.pt`. This matches the
historical EXP-026 record that original `SEQ-01` weights were not saved. Available
seed-42 substitutes fail prediction parity:

| substitute | fold | max abs log error | RMS log error | threshold |
|---|---|---:|---:|---:|
| `SEQ-D3A-BASE-S42` | 2025-10-16 | 1.703125 | 0.164831179 | 0.000001 |
| `SEQ-03A-BASE-S42` | 2025-10-16 | 1.703125 | 0.164831179 | 0.000001 |
| `SEQ-01C-S42` | 2025-09-04 | 1.509765625 | 0.144327147 | 0.000001 |

The row keys and targets are exact in these comparisons; the prediction mismatch is the
encoder/run mismatch. `SEQ-D3A-BASE` was explicitly not used as current SEQ.

An additional repository-wide recovery audit covered all nine ZIP archives, all 203
Torch-like files, every Git ref/reflog/stash and all unreachable Git objects. Of 169
architecture-compatible files, 134 unique tensor states remained after content
deduplication. Every state was executed with the exact historical architecture on fixed
canonical probes for all four fold contexts and production TEST. There were **zero exact
matches** at the `1e-6` threshold; the best max log errors by context were `0.46875`,
`0.43359375`, `0.4375`, `0.5` and `0.40625`. This rules out a usable state dict hidden
under a later or misleading filename. Full evidence is in `deep_recovery_audit.json`.

## Recovered DIST16 contract

The exact historical S1-DIST implementation was recovered without guessing. It uses one
exact-zero class plus 15 positive quantile classes. Edge zero is `1e-9`; positive edges
are fold-training-only quantiles at `1/15..14/15`; class centres are training-row mean
`z30` values; prediction is `sum(p_k * m_k)`. Empty centres inherit from the left.

The exact 250-round LightGBM capacity is: learning rate `0.05`, 127 leaves, minimum leaf
200, feature/bagging fractions `0.7/0.8`, bagging frequency 1, L2 5, `max_bin=63`,
row-wise mode, seed 42. DIST uses multiclass/multi-logloss with 16 classes; matched DIRECT
would change only objective/metric to regression/RMSE.

## Pilot and requested metrics

Status: **NOT RUN — stop rule applied before model fitting**.

Consequently there are no valid DIRECT, DIST16 or SHUFFLED-DIST16 scores; no fixed
replacement delta; no `A`, `Q`, analytic alpha, user-half result, correlation or
unexplained-variance estimate; and no `pilot_rows.parquet`. Reporting any such number from
a retrained or D3A encoder would be mislabeled evidence.

## Full/nested wCV, seed robustness and bootstrap

Status: **NOT RUN**. The pilot could not legally start, so full four-fold validation,
LOFO alpha selection, seeds 42/43/44 and 1,000 user-cluster bootstrap replicates were not
run. Independently, the exact current seed-43 checkpoint for fold `2025-10-16` is also
missing, which would block the later AVG3 stage even if seed 42 were recovered.

## Diversity/span and OOF/TEST regime

No candidate correction exists, so correction correlations, donor-fold projection,
orthogonal endpoint, OOF/TEST RMS ratio and TEST span distance are not defined.

The required inputs themselves passed reconnaissance: aligned OOF has 770,616 unique
canonical rows and all requested donor columns; aligned TEST has 250,000 unique finite
rows in exact sample order. The current geometry bank is the exact 65-unique-source,
rank-57 basis (`Z.npz` SHA256
`443e80472f321e796faccfe0e9ea6954653397eb74311e7abef7471ea3064dfe`). The public
incumbent was identified only as the TEST anchor and was not used for validation.

## Artifacts

- `run_reconnaissance.py` — read-only reproducible registry/checkpoint/input audit.
- `test_reconnaissance.py` — tests registry freedom, checkpoint absence, substitute parity,
  exact DIST16 semantics, aligned inputs and geometry inventory.
- `technical_block.json` — machine-readable stop record.
- `deep_recovery_audit.json` — exhaustive ZIP, checkpoint, Git-object and functional
  parity recovery record for `C:/Users/Admin/Desktop/OZON-E-CUP`.
- `reconnaissance.md` — detailed provenance, hashes and recovered contracts.
- `report.md` — this decision record.

No OOF, TEST, builder or submission was created. In particular,
`submissions/SUBMIT_EXP074_SEQ_TEMPORAL_DIST16.csv` does not exist and no leaderboard
upload was attempted. Original artifacts and registries were not modified.

## Verification

`python run_reconnaissance.py` completed successfully. Pytest result: **5 passed**.

## Recommended next step

The requested local repository has now been exhausted. Search only archival/off-machine
storage for the exact original `SEQ-01` state dicts, at
minimum all seed-42 fold checkpoints and its production checkpoint, plus
`SEQ-01-S43-V1016.pt` for the requested AVG3 confirmation. Accept them only if the hooked
original head reproduces the retained scalar OOF/test predictions with max log error
`<=1e-6` in canonical row order. If those weights cannot be recovered, close EXP074 as
non-runnable; a fully checkpointed deterministic plain-SEQ retrain must receive a new
experiment ID because it tests a different frozen representation.
