# Session state — 2026-08-25 A1

## Frozen development reference

`SESSION_BASELINE = DEVELOPMENT_REFERENCE = STRONGEST-CURRENT / exp_037`.

- Composition in `log1p` space: `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 ETX-AVG3 + 0.225 SEQ-AVG3`.
- Validation: calibrated wCV `1.7475098625`; fold scores `1.7668833568 / 1.7605095768 / 1.7486292240 / 1.7412785664`; weights `1:2:4:8`.
- Champion evidence: honest slot LOFO `-0.00092`, `4/4` including `2025-10-16`; public LB `1.6496571`.
- OOF sources and SHA256: `oof_S1-E03a.npz` `38fb0270...`; `oof_S1-E02.npz` `2a8e543f...`; `oof_S1-DIST.npz` `7ef12519...`; `oof_ETX-AVG3.npz` `890aef1a...`; `oof_SEQ-AVG3.npz` `8e8ec790...`. Exact prediction hash: `b30008842ff0fb1d36682a13308c7847c5de3ae3853024187c7f93bee6a04b91`; row-key hash `9e9c9de2d280e856eb1172830519fb044c346a319d99b9ed33d0834d04ab067a`.
- Test recipe: `research/strategies/results/ETX2/make_submission.sh`; ETX uses `DCW`, TCN uses `clip289`; exact CSV SHA256 `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.
- Code provenance: git `a28a71f` plus the registered working-tree implementation described by `exp_037`; `features.py` and `seq.py` are currently shared/dirty and must not be rolled back.

## Compact Research Map

- Dominant validated families: dense cutoff tabular aggregates; distribution head; depth-clipped TCN; DCW event transformer; fixed log-space ensemble.
- Development-only hedge: stable BTYD at fixed 0.05 (`exp_051`, `-0.000321`, 4/4), below production gate.
- Closed families: contaminated late supervision; tabular rounds/seeds refresh; generic residual/post-calibration/gating; classic BTYD variants; zero/hard-zero correction; future-event auxiliary supervision; conditional-fresh ensemble/fine-tune; block differencing; dense/multi-horizon auxiliary heads; personal-time tabular representation; fixed burst/gap episodes; landmark outcome memory; late SSL adaptation; dataset fingerprint; production-state reweighting; channel-Shapley monetary split; ridge on existing 227 columns.
- Disguised repeats excluded: TCN capacity/epochs, neighboring blend weights, another residual learner over the same 227 columns, multi-horizon heads on the same supervision, threshold sweeps for burst/gap, level search.
- Session loop closed after four clean hypothesis rejections and one accepted final integration. H4 explicit event-order and H5 direct occurrence integration are closed; H6 production-regime audit passed.

## Current-champion provenance audit

- **Validation-first champion remains `STRONGEST-CURRENT / exp_037`.** It is the strongest candidate under the frozen canonical four-fold protocol and remains `SESSION_BASELINE` for the whole session.
- `пайплайн сокомандника/latest/latest.csv` is the public-score leader at `1.6492175622` (CSV SHA256 `7ef5b2c5...e722`), built as `.12 friend(exp037) + .16 occ_meta_B + .72 occ_raw_X3`.
- That final blend is explicitly documented as **Public-LB calibrated**, so it is not eligible as a validation-first integration reference. Its clean-fold occurrence candidates were trained/evaluated from a weaker proxy baseline and are worse than exact exp_037 on all four absolute fold scores (best extra90 xmeta weighted gap approximately `+0.00043`).
- Consequence: public champion and legitimate research champion are distinct. Public evidence motivates one pre-registered occurrence audit, but cannot select its weights or override the canonical decision gates.

## Candidate hypotheses and choice

1. **H1 OPEN-FUNNEL (selected):** counts and age of Search/Cart activity occurring after the last order encode unresolved intent absent from window marginals.
2. **H2 MONETARY-TAIL (excluded after source audit):** too close conceptually to the closed monetary persistence/BTYD and channel-shape evidence in `exp_047/052`; a new quantile implementation would not establish a new source of information.
3. **H3 PLATFORM-DETREND (selected pivot):** normalize each user's historical daily activity by the cutoff-safe platform-wide state on that date, testing whether relative demand survives marginal counts and removes macro growth/seasonality.

H1 had the best initial information-value/cost ratio. After its rejection, H2 is disqualified as a disguised conceptual repeat; H3 is the highest-EV independent pivot and still admits an artifact-first matched-placebo test.

## Experiment ledger

- `EXP-061 OPEN-FUNNEL`: **REJECT**; exact audits PASS, but REAL/SHUFFLED/CONTROL selected scale 0 on every fold/half. Information gained: unresolved funnel marginals are conditionally redundant with the champion.
- `EXP-062 PLATFORM-DETREND`: **REJECT**; exact audits PASS, but REAL/PLACEBO/CONTROL selected scale 0 on every fold/half. Information gained: same-day platform normalization is conditionally redundant with the champion.
- `EXP-063 OCCURRENCE-REVISIT`: **REJECT**; exact replay PASS, but E11 nested LOFO is `+0.0000105`, 1/4, latest alpha 0, and does not separate from direct control. Information gained: teammate occurrence public gains do not transfer to exact canonical OOF.
- `EXP-064 EVENT-ORDER`: **REJECT**; exact state-multiset shuffle audits PASS and changes >97.6% rows, but REAL/SHUFFLED/CONTROL select scale 0 on every fold/half. Information gained: explicit adjacent funnel order is conditionally redundant with the champion.
- `EXP-065 FINAL-INTEGRATION`: **ACCEPT package, strongest unchanged**; A rebuilt byte-identical exp037, ETX↔SEQ ratio `0.775`; B byte-identical BTYD05 with support ratio `1.1734`; all CSV audits PASS, no LB upload.

## Best next action

Session complete under the compute-budget stop: no new candidate reached the frozen development/production success gates. Retain exp037 as validation-first champion; use the two verified files in `submissions/FINAL_20260825_A1/` only if an external submission decision is later authorized.

## Final session status

- `RESEARCH_STATUS = REJECT`: no new predictive signal passed the frozen development gate.
- `PRODUCTION_CANDIDATE = NO` for a **new incremental candidate**; existing exp037 remains fully production-ready and is packaged as Submission A.
- Highest-EV unresolved explanation: the remaining intensity error is driven by basket/item/order composition that is not observable in the daily aggregate schema (`user_id,date,Search/Cart/Order counts, channel GMV`).
- Exact next experiment, conditional on new source availability: `BASKET-COMPOSITION-MATCHED` — add cutoff-safe per-order/item log-value dispersion and category-mix history, compare against a within-user date/value-marginal shuffle, then run the same cross-user residual preflight. Current repository has no order/item/category identifiers, so implementing it now would fabricate information rather than test the hypothesis.
