# Team-A solution architecture

## End-to-end dependency graph

```text
train.parquet + sample_submit.csv
        |
        +--> build_features(cutoff) --> CAP / UNC / DIST (LightGBM)
        |                              |
        +--> sequence cache ---------->+--> SEQ-01 + SEQ-C289 seeds
        |                              |
        +--> sparse event cache ------>+--> ETX-DCW seeds
                                       |
                                       v
                        STRONGEST_CURRENT (log1p, level 2.3293)
                                       |
                                       +------ 55% --------------------+
                                                                         |
raw --> Team-B features --> LGBM / DIST / XGB / CatBoost --> Team-B ---+--> STRONGEST55/TEAMB45
                              |                                          |
                              +---------------- 14% ----------------------+--> JOINT86/TEAMB14
                                                                         ^
scored submission bank --> geometry --> ORTH --> A1/A2/EXP075 --> JOINT_V2 (86%)
```

All arrows above are backed by located code or frozen artifacts except the last
materialization step that produced the exact `SUBMIT_JOINT_V2.csv` bytes. The
geometry, ORTH, A1/A2 and EXP075 audits exist, but no primary script that emits
SHA `211879cb…33cba` was found. The final 86/14 builder itself is complete.

## STRONGEST_CURRENT

The exact inner blend is built from nine TEST vectors in `log1p`:

| Family | Components | Total weight | Role |
|---|---|---:|---|
| CAP | `S1-CAP` | 0.10 | capped-history tabular member |
| UNC | `S1-UNC` | 0.20 | uncapped/direct tabular member |
| DIST | `S1-DIST` | 0.25 | 16-bin distribution head |
| SEQ | seed 42 plus clipped-depth seeds 43/44 | 0.225 | sequence encoder; TEST depth clipped to 289 |
| ETX | DCW-corrected seeds 42/43/44 | 0.225 | sparse event transformer; depth/day-of-week correction |

After the weighted average, one constant shift sets
`mean(log1p(prediction)) = 2.3293`, then log predictions and raw predictions are
clipped nonnegative. The resulting CSV SHA is `abc2218…e04bda` and its recorded
public LB is 1.6496571902356205.

## Team-B vector

`final_classic_ml.csv` has two layers. The current branch combines LightGBM and
CatBoost direct/hurdle models; the Team-B branch combines five members:

| Member | Family | Weight inside Team-B branch |
|---|---|---:|
| recency | LightGBM regression | 0.25 |
| post-order distribution | LightGBM 16-bin classifier | 0.10 |
| behavior distribution | LightGBM 16-bin classifier | 0.20 |
| behavior regression | XGBoost | 0.25 |
| behavior regression | CatBoost | 0.20 |

The packaged production vector was fresh-trained byte-identically in the pinned
Team-B environment: SHA `4ed2916b…44aba`.

## Final outer blends

Both outer blends use the same historical level-alignment function. For anchor
`z_a=log1p(anchor)` and raw Team-B `z_b`, a scalar `s` is found by 100 bisection
iterations so that `mean(max(z_b+s,0))=mean(z_a)`. The final prediction is
`max(expm1(w*z_a + (1-w)*max(z_b+s,0)),0)`.

| Solution | Anchor | Anchor weight | Team-B weight | Realized Team-B shift | Output SHA |
|---|---|---:|---:|---:|---|
| STRONGEST55/TEAMB45 | STRONGEST_CURRENT | 0.55 | 0.45 | -0.12190138468055683 | `1ce85203…a14fb4` |
| JOINT86/TEAMB14 | SUBMIT_JOINT_V2 | 0.86 | 0.14 | -0.1214326530964569 | `85d9cd64…dac02` |

## Historical-only families

Dense supervision, personal-time, hazard/multi-horizon, BTYD, occurrence,
fingerprint, funnel/order, residual correction, Ridge, renewal, calendar and
platform-detrending families remain in the experiment history with their
original ACCEPT/REJECT/NO_GO verdicts. They are not silently promoted into the
final model. Submission geometry and ORTH are historical ancestors of JOINT but
are public-LB-fitted prediction-space methods, not fold-safe model training.

## Leakage boundary

Located feature code consistently routes feature construction through a cutoff
and applies no event later than it (family-specific code uses `<` or `<=`). The canonical validation target uses the
next 30 days. Known contaminated late cutoffs and other rejected validation
schemes remain documented in the preserved `STATE.md`; they are not used to
claim final OOF performance.
