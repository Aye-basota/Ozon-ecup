# Reconstructed primary report — exp_058 EXACT-ANNIVERSARY-WINDOW

## Catalogue metadata

- **Catalogue ID:** `independent_anniversary__exp_058_exact_anniversary`
- **Namespace:** `independent_anniversary`
- **Experiment ID:** `exp_058_exact_anniversary`
- **Original source:** `experiments/exp_058_exact_anniversary.md`
- **Source ref:** `LINKED_WORKTREE:exp/058-exact-anniversary`
- **Source commit:** `PRIMARY_CARD_NOT_IN_MERGED_GIT; normalized row SHA256=ef7120958e4609afc702dbb778085b8349b7790e2d16659e270e44b681ab21b9`
- **Kind:** reconstructed primary experiment report
- **Model:** Ridge
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | interpretation_reported | **REJECT.** REAL выглядит полезнее baseline и shuffled, но exact calendar<br>alignment не подтверждён: same-family shifted old window сильнее REAL на обеих<br>половинах. Значит измеренный gain нельзя приписать exact anniversary; он<br>совместим с дополнительной old-history capacity/persistent user level и с<br>разной поддержкой pre-window scale denominators. По preregistered stop-rule<br>окна, Ridge alpha и shrink не тюнились.<br><br>- `ORDINARY TEMPORAL CV: unavailab
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** | interpretation_reported | **REJECT.** REAL выглядит полезнее baseline и shuffled, но exact calendar<br>alignment не подтверждён: same-family shifted old window сильнее REAL на обеих<br>половинах. Значит измеренный gain нельзя приписать exact anniversary; он<br>совместим с дополнительной old-history capacity/persistent user level и с<br>разной поддержкой pre-window scale denominators. По preregistered stop-rule<br>окна, Ridge alpha и shrink не тюнились.<br><br>- `ORDINARY TEMPORAL CV: unavailab
- **Submission:** | interpretation_reported | **REJECT.** REAL выглядит полезнее baseline и shuffled, но exact calendar<br>alignment не подтверждён: same-family shifted old window сильнее REAL на обеих<br>половинах. Значит измеренный gain нельзя приписать exact anniversary; он<br>совместим с дополнительной old-history capacity/persistent user level и с<br>разной поддержкой pre-window scale denominators. По preregistered stop-rule<br>окна, Ridge alpha и shrink не тюнились.<br><br>- `ORDINARY TEMPORAL CV: unavailab
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: normalized report fields, implementation and tests survive; the original Markdown bytes and ignored input artifacts do not
- **Notes:** Rejected experiment; no submission was created. No missing field was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Reconstructed primary report — exp_058 EXACT-ANNIVERSARY-WINDOW

The original Markdown card is absent from the merged tree. The independent reconstruction preserved its normalized primary-report row verbatim; every field is reproduced below.

| Registry field | Preserved value |
|---|---|
| experiment_id | independent_anniversary:exp_058 |
| namespace | independent_anniversary |
| local_id | exp_058 |
| canonical_name | exp_058 — EXACT-ANNIVERSARY-WINDOW |
| family_inferred | temporal_and_calendar |
| date_reported | 2026-08-25 |
| hypothesis_reported | unknown |
| change_reported | unknown |
| facts_reported | unknown |
| interpretation_reported | **REJECT.** REAL выглядит полезнее baseline и shuffled, но exact calendar<br>alignment не подтверждён: same-family shifted old window сильнее REAL на обеих<br>половинах. Значит измеренный gain нельзя приписать exact anniversary; он<br>совместим с дополнительной old-history capacity/persistent user level и с<br>разной поддержкой pre-window scale denominators. По preregistered stop-rule<br>окна, Ridge alpha и shrink не тюнились.<br><br>- `ORDINARY TEMPORAL CV: unavailable`<br>- `PSEUDO-PRODUCTION CROSS-FIT: performed`<br>- `TRANSFER RISK: one-month calendar extrapolation`<br>- Production candidate/audit: **not reached after REJECT**<br>- Submission: **not created**<br>- LB: **не отправляли**<br>- Tests: **29 passed** (`experiment + pipeline + validation`) |
| config_reported | unknown |
| cv_candidate | unknown |
| delta_candidate | unknown |
| lb_candidate | unknown |
| runtime_candidate | unknown |
| verdict_reported | reject |
| source_ref | LINKED_WORKTREE:exp/058-exact-anniversary |
| source_path | experiments/exp_058_exact_anniversary.md |
| clean_evidence_path | experiments/independent_anniversary/exp_058_exact_anniversary.md |
| sha256 | ef7120958e4609afc702dbb778085b8349b7790e2d16659e270e44b681ab21b9 |
| evidence_tier | 7_primary_experiment_report |
