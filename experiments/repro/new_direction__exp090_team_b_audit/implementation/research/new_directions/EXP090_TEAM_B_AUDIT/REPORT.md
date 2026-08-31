# EXP090 — Team-B Solution Audit

## Verdict

Team-B имеет два разных понятия champion. Лучший **фактически scored** файл —
`exp_019_behavior_v1_dist_wrec050_scale120.csv`, public LB
`1.654502353530087`. Новый локальный champion — `exp_024` с mean CV
`1.706955`, но его исходный CSV, model files и row-level
OOF отсутствуют.

`exp024` создаёт материальную новую TEST-direction: после двойной
проекции из актуального scored span остаётся `18.074%` RMS correction
(`3.267%` энергии). Но полезность этой именно post-span части не
измерена ни OOF, ни LB. Поэтому candidate CSV не создан; сохранён точный vector
artifact для следующего измерения. Большая часть полной correction всё же лежит
в уже известном span; новый signal — независимый хвост, а не доминирующая часть.

## Team-B architecture

Финальный handoff — log-space ensemble из пяти компонент:

| component | family / target | features | weight |
| --- | --- | --- | ---: |
| recency | LightGBM regression / `log1p(y30)` | 152 recency aggregates | 0.25 |
| post_order_dist | LightGBM 16-bin classification / binned `log1p(y30)` | 215 long-buy + post-order | 0.10 |
| behavior_dist | LightGBM 16-bin classification | 329 behavior_v1 | 0.20 |
| xgb_behavior | XGBoost regression / `log1p(y30)` | 329 behavior_v1 | 0.25 |
| cat_behavior | CatBoost regression / `log1p(y30)` | 329 behavior_v1 | 0.20 |

Component scales are `0.64` for recency and `0.62` for the other four.
Occurrence gate exists only as rejected exp014. SEQ, ETX and Ridge are absent.

The quality mechanism is cumulative: log-target/tree ensembles, raw-GMV scale
calibration, a distribution head, post-order state, and the 114 behavior_v1
features. XGBoost then gives the largest new unscored CV increment; CatBoost
adds a smaller conditional increment.

## Reproduction

The supplied raw parquet was found by exact SHA256
`5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0`.
Fresh training used exactly eight cutoff panels `2025-08-28..2025-10-16`, TEST
cutoff `2026-02-14`, 250,000 unique users, and exact sample order.

Byte-exact reproduction of the claimed `exp024` file is **not provable**:

- the original CSV is absent;
- no model files or pinned package versions are supplied;
- `cat_xgb_blend.py` applies global raw scale `1.20` before level-match, while
  `predict.py --handoff` omits it. The two realized vectors differ by RMS
  `0.045847096` in log space.

Both formulas were reconstructed from the same freshly trained components and
stored in `team_b_reproduction_predictions.parquet`. The experiment-script
formula is used for geometry because it is the provenance of the named exp024
submission.

## Leakage / validation audit

No same-period or target leakage was found. Every feature query first enforces
`event_date < cutoff`; target is `[cutoff, cutoff+30d)`. Both validation targets
are fully observed and their training labels end before the validation feature
cutoff.

The questionable part is selection, not label leakage: 2,835 weight vectors
were selected and reported on the same two folds, without an outer/nested fold.
Scale `1.20` and final TEST level also have public-LB lineage. `validation.py`
is a stub, actual dates live inside scripts, and feature caches do not include a
code/data fingerprint.

## Comparison with JOINT_V2

`JOINT_V2` exact SHA256 is `211879cb1c79bbbde93d451fca5b61c521b523f989ce42bab62cd3ab87233cba` and LB `1.645936304478217`. The full
geometry table contains RMS difference, level, correlations, zero/clipping
fractions, double-projection diagnostics and condition numbers for each strong
Team-B vector.

All scored Team-B files, including exp019, are already members of the current
scored bank. Accordingly exp019 post-span RMS is only
`4.529e-08`. It does not constitute a new direction relative to
the current span even though it was historically useful inside Team-B.

## New-signal geometry

For fresh exp024 relative to JOINT_V2:

- centered correction RMS: `0.159131583`;
- post-span RMS: `0.028760795`;
- post-span fraction: `0.180736` RMS,
  `0.032665` energy;
- corr(post-span, A1-365): `-0.009993`;
- corr(post-span, A2): `0.011197`;
- corr(post-span, JOINT_V2 out-of-plane residual):
  `-0.000800`;
- scored-span rank `66` -> `67`;
  design condition `48347.856` ->
  `48649.911`.

Thus exp024 is not merely a repackage of known scored submissions on TEST.
It is mostly an in-span repackage with a material absolute residual. This is a
novelty statement, not a value statement.

## OOF residual evidence

No row-level Team-B OOF vectors exist. The saved `exp024` grid contains only
mean/std over two folds. From those sufficient statistics the selected blend
improves exp019 on both folds by
`-0.000737707` and
`-0.000750484` RMSLE, mean
`-0.000744096`. CatBoost conditional on exp023 contributes mean
`-0.000164196`.

These signs are stable but non-nested. `rho`, `b`, `G`, oracle amplitude,
strict-forward delta and user bootstrap CI for the **post-span** direction are
not identifiable and are left missing rather than synthesized from standalone
CV.

## Leaderboard evidence

The scored Team-B ladder confirms small gains from post-order, distribution
head and full behavior_v1. Behavior_v1 improves exp017 from
`1.654631819` to `1.654502354`, Delta MSE
`-0.000428419`. The slim ablation regresses.

All of this evidence is already inside the current scored span. `exp023` and
`exp024` have no public scores, so LB-implied headroom for their new post-span
part is unavailable. The decoding table clearly separates scored LB inference
from unscored OOF-only claims.

Relative to JOINT_V2, the entire scored exp019 correction has LB-implied optimal
scale only `0.004545`
and maximum full-Gram MSE gain
`0.000000589`. Its out-of-current-span
part is numerical zero, so this is neither material headroom nor new signal.

## Source of Team-B gain

The last confirmed scored increment is the `behavior_v1` representation in the
distribution head: order-cycle regularity, overdue phase, stable cheque,
pre/post-last-order intent and calendar habit features. Within the new
unscored five-family system, XGBoost diversity supplies about `-0.000580`
two-fold RMSLE versus exp019 and CatBoost adds about `-0.000164` conditional
RMSLE. The CatBoost search optimum hits its imposed `0.20` boundary, so its
reported coefficient is especially selection-sensitive.

Sibling vectors are exp020 (slim behavior ablation), exp022 (three-LightGBM
weights), exp023 (+XGBoost), and the two inconsistent exp024/handoff formulas.

## Compatibility with current best

The scored exp019 axis is geometrically compatible but already exploited by the
current scored-span search. Public two-axis decoding versus ORTH_FINAL is in
`joint_with_v2.json`; it does not establish a new Team-B conditional gain.

For exp024, TEST correlation and conditioning are acceptable, but the required
alignment vector `b = U.T @ residual / N` is absent. Therefore individual and
conditional post-span gains, safe amplitude and expected Delta RMSLE are **not
identified**. Adding raw correlations would be mathematically invalid.

The only numerically decoded conditional term is scored exp019 given V2:
approximately `0.000020810` MSE in the
full-population public-plane diagnostic. It is below the `0.0003` gate, depends
on unknown public membership, and is already inside the current scored span;
it is not an expected new gain for exp024.

## Best justified candidate

No CSV was created. None of the three authorization gates is met:

- clean post-span OOF rho >= 0.015: not measured;
- LB-implied post-span headroom >= 0.0003 MSE: no exp024 score;
- reproducible confirmed better LB: exp024 is unscored and original bytes are missing.

Instead, `team_b_signal_vector.npz` stores the centered and twice-projected
exp024 direction plus all five components. SHA256:
`fb0b7545afd2b7ccf000805356102fbcb7eba9403571ae027e6ad4ba66aeba37`.

## Final conclusions

1. Best scored Team-B submission: exp019, LB `1.654502353530087`.
2. Declared exp024 champion cannot be byte-reproduced; a fresh code reproduction
   exists, but two primary inference formulas disagree.
3. No leakage was found; validation/selection and reproducibility are the weak
   points.
4. Scored Team-B signal is fully in the current scored span. Fresh exp024 has
   `18.074%` post-span RMS, but no post-span value evidence.
5. The defensible next action is one **symmetric measurement pair** built from
   `d_perp`: `z=clip(z_V2 +/- 0.025*d_perp/RMS(d_perp),0)`. Decode sign and
   amplitude from both scores, apply public-noise shrinkage, and only then build
   one combined candidate. Do not tune another raw five-way blend on the same
   two folds.
