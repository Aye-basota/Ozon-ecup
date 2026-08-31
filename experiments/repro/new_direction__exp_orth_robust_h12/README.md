# EXP ORTH-ROBUST H12

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp_orth_robust_h12`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP_ORTH_ROBUST_H12`
- **Original source:** `research/new_directions/EXP_ORTH_ROBUST_H12`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** ## Forward validation
- **Known score:** | `Delta RMSLE = LB_h21 - LB_anchor` | `-0.000055980023302` |
- **Seed:** `reason = exact five-member averaged predictions, per-model configs/seeds,
- **Postprocessing:** None documented
- **Submission:** Searched the clean repository, `OZON-E-CUP`, `submission_geometry_research`,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: the report survives, but no experiment launcher was recoverable from this package
- **Notes:** Directory-level audit unit: 1 files, 0 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP ORTH-ROBUST H12

## Verdict

**BLOCKED**

`SUBMIT_ORTH_ROBUST_H12.csv` не создан. Строгий gate остановил эксперимент до
генерации кандидата: локально отсутствует исходный вектор `v` до clipping, а
восстановление из `SUBMIT_ORTH_FINAL.csv` неоднозначно на clipped-строках.

## Baseline

- Anchor path: `research/new_directions/CLAUDE_PUBLIC_CEILING/SUBMIT_PUBLIC_EB.csv`
- Дубликат anchor: `research/new_directions/CLAUDE_PRIVATE_V2/SUBMIT_PUBLIC_EB.csv`
- Оба anchor-файла byte-identical, SHA256
  `66d1f4df2af22d41a0a59276e2818127114ea26dc27b782c6fabf83beff9c778`.
- Anchor LB: `1.6463246740442117`.
- H21 path: `submissions/SUBMIT_ORTH_FINAL.csv`.
- H21 LB: `1.6462686940209101`.
- H21 SHA256:
  `102496bdfbec6f959b88339a11c6c8b669d088b81277df103ffd72e39ec4cbdb`.
- Claude narrative report found at
  `C:/Users/Admin/Downloads/SUBMIT_ORTH_FINAL_reasoning.md`.

## Reproduction

### Artifact audit

Searched the clean repository, `OZON-E-CUP`, `submission_geometry_research`,
`research_clean`, `OZON-ECUP2-WORK`, Downloads, Claude local history/cache,
PowerShell history, temp files, and the recycle bin for the generator, original
pre-clip direction, five GBM predictions/models/configs, and historical caches.
The only ORTH-specific artifacts found are the downloaded H21 CSV and the
narrative Markdown report. The report describes four GBM configurations plus the
original model, four training windows and weights `0.20/0.20/0.25/0.35`, but does
not contain exact hyperparameters, seeds, prediction arrays, projection basis, or
generation code.

The H21 CSV has a `Zone.Identifier` pointing to the leaderboard object store and
is byte-identical to `C:/Users/Admin/Downloads/SUBMIT_ORTH_FINAL.csv`; it is a
download of the submitted artifact, not a local pipeline output.

### Format checks

| check | result |
|---|---:|
| rows | `250000` |
| columns | `user_id,predict` |
| unique `user_id` | `250000` |
| same order as anchor | `True` |
| finite predictions | `True` |
| negative predictions | `0` |
| zero predictions / clipped rows | `815` |
| min / max predict | `0.0 / 3353.7524240845` |

### Parity status

Let `z_a = log1p(anchor)` and `z_21 = log1p(H21)`. Defining the circular,
post-clipping surrogate `v_visible = (z_21-z_a)/21` reproduces the existing CSV
byte-for-byte when serialized with `float_format="%.10f"`: max absolute prediction
difference `1.82e-12`, identical SHA256. This is **not accepted as Claude pipeline
parity**, because the direction was derived from the target output rather than
from the five GBM predictions and original projection cache.

Clipping makes the missing source material decisive:

- `v` is directly identifiable on `249185/250000` rows;
- `815` pre-clip coordinates are lost because `z_21 = 0` only implies
  `z_a + 21 v <= 0`;
- `686` of those rows also have `z_a = 0` and are guaranteed to remain clipped at
  H12;
- the remaining `129` rows have positive `z_a`, so their H12 values are genuinely
  ambiguous (`0 <= z_12 <= (3/7) z_a` from the available files alone);
- across those 129 rows the largest possible prediction under that row-wise bound
  is `0.0700751` and the sum of the row-wise upper bounds is `0.683922`.

Therefore `CLAUDE_PIPELINE_PARITY = BLOCKED` and the exact H12 file is not
identified by the available artifacts.

### Direction and orthogonality

The Claude narrative states that the original direction had unit Euclidean norm,
`<v,1> = -5.8e-8`, and `<v,z_TEAM_EB> = 3.8e-7`, after orthogonalization to the
full submission span. Those claims cannot be independently replayed without the
pre-clip vector and exact basis.

Forensic metrics for `v_visible` (not the original `v`) are:

| metric | value |
|---|---:|
| `rms(v_visible)` | `0.001998932251` |
| `mean(v_visible)` | `2.666382339e-6` |
| Euclidean norm | `0.9994661255` |
| known non-clipped energy | `0.9988043273` |
| `max_k abs(<v_visible,phi_k>)` on maximal local rank-68 span | `1.302830659e-4` |
| `rms(P_span(v_visible))` | `1.534174880e-4` |
| `rms(v_visible-P_span(v_visible))` | `0.001993036181` |
| orthogonal fraction | `0.9970503902` |
| max projection after a second projection | `2.73e-16` |

The nonzero first-pass projection is expected from replacing every unknown
pre-clip value by its clipping boundary; it is also direct evidence that
`v_visible` must not be used as the production direction.

## New LB diagnostic

Using the preregistered unit-direction geometry with `N=250000`, anchor score
`S0`, step `h=21`, and

`S(h)^2 - S0^2 = h^2/N - 2 h rho S0/sqrt(N)`, the actual scores give:

| metric | value |
|---|---:|
| `Delta RMSLE = LB_h21 - LB_anchor` | `-0.000055980023302` |
| `Delta MSE = LB_h21^2 - LB_anchor^2` | `-0.000184319453467` |
| implied `rho_public` | `0.014088520675` |
| implied optimal `h` | `11.597139604` |
| H21 break-even `rho` | `0.012755685638` |

The score confirms the positive sign out of sample: H21 improved the anchor.
The implied signal is only `2.84%` above the most test-like historical value
`rho=0.0137` and is therefore consistent with it, while being much weaker than
the earlier `rho=0.0348` fold and Claude's central `rho=0.0265`. This calculation
is diagnostic only; it was not used to select `h=12`.

## Forward validation

The only available values are those recorded in Claude's narrative report:

| train -> apply | reported rho | reported optimal gain |
|---|---:|---:|
| `303+333 -> 364` | `+0.0348` | `-0.00103` |
| `303+333 -> 371` | `+0.0137` | `-0.00016` |

`371` remains the primary test-like comparison. Reverse-time folds are not used
as evidence of forward generalization.

`HISTORICAL_EXACT_CHECK = BLOCKED`

`reason = exact five-member averaged predictions, per-model configs/seeds,
pre-orthogonalized fold predictions, and the fold-specific projection bases were
not found locally. The narrative values cannot establish that the exact final
averaged direction, rather than an earlier single GBM, produced them.`

Consequently exact fold-specific optimal `h` and exact fold deltas at H12/H21
were not recomputed or claimed.

## H12 vs H21

The following table is a preregistered mathematical risk analysis on the anchor
scale. Negative delta is improvement. `rho_public` is included only as a separate
diagnostic row.

| rho scenario | optimal h | expected Delta RMSLE H12 | expected Delta RMSLE H21 |
|---|---:|---:|---:|
| `0` | `0.0000` | `+0.000174926` | `+0.000535652` |
| `0.010` | `8.2316` | `-0.000065066` | `+0.000115735` |
| historical test-like `0.0137` | `11.2773` | `-0.000153872` | `-0.000039662` |
| public-implied diagnostic `0.0140885` | `11.5971` | `-0.000163197` | `-0.000055980` |
| historical central `0.0265` | `21.8138` | `-0.000461129` | `-0.000577362` |

Break-even `rho` is `0.007288963` for H12 versus `0.012755686` for H21. At
`rho=0`, H12 carries only `32.66%` of H21's RMSLE downside. At the historical
central `rho=0.0265`, H12 retains `79.87%` of H21's modeled gain; at the
test-like `rho=0.0137`, H12 is near the optimum and has about `3.88x` the modeled
gain of H21. Thus the risk hypothesis is supported mathematically: H12 preserves
most plausible upside while materially reducing null-signal downside.

This does not override the artifact gate. The pre-clipping scale ratio
`12/21 = 0.5714285714` and direction correlation `1` are identities for the exact
pipeline, but cannot be globally verified without the missing 815 coordinates.

## Output

- Requested path:
  `C:/Users/Admin/Desktop/e-cup-research-clean/submissions/SUBMIT_ORTH_ROBUST_H12.csv`
- Status: **NOT CREATED**.
- SHA256: `N/A`.
- Reason: exact `v` and independent H21 pipeline parity are missing, so creating a
  candidate would violate GO conditions 1, 2 and 7.
- Existing submissions and score/history files were not modified.

## Final conclusion

The actual H21 score validates the direction's sign and implies
`rho=0.0140885`, almost exactly the test-like historical regime where H12 is near
optimal and H21 is near break-even. The risk math therefore favors H12: about
one-third of the null downside while retaining about 80% of central-scenario
upside. Nevertheless the original pre-clip direction and exact five-GBM pipeline
are absent, and 129 H12 predictions cannot be recovered from the clipped H21 CSV;
under the preregistered gates the only defensible verdict is **BLOCKED** until the
original `v` or equivalent raw averaged prediction/projection cache is supplied.
