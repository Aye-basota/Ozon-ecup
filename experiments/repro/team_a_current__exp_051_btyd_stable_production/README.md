# exp_051 — BTYD stable fit + production

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_051_btyd_stable_production`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_051_btyd_stable_production`
- **Original source:** `experiments/exp_051_btyd_stable_production.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** dilated TCN, sequence model, BG/NBD, BTYD
- **Features:** freshness/conditional features, history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** ## BTYD OOF REVALIDATION
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** Final mean `log1p` после штатного level shift: `2.32930000017`.
- **Submission:** `submissions/submission_BTYD05.csv`: 250,000 строк, exact schema/order,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_051 — BTYD stable fit + production

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево
- **Код:** `src/btyd_stable_fit.py`, `src/btyd_exp051.py`
- **Запуск:** `python src/btyd_exp051.py`
- **Статус:** **CASE B — `submission_BTYD05.csv` CREATED, leaderboard upload не выполнялся**

## Гипотеза

Production blocker `exp_050` вызван преждевременной остановкой finite-difference
L-BFGS-B, а не сменой статистической модели. Аналитический градиент и строгий
deterministic polish должны одинаково стабилизировать те же BG/NBD fits на OOF
и test cutoff, сохранив residual signal `exp_047`.

## Что изменено относительно базы

Изменена только numerical optimization procedure неизменённого basic BG/NBD:
те же 3 starts и log-bounds, analytic Jacobian, строгий L-BFGS-B и BFGS polish.
Likelihood, purchase-day, common origin, A/B hash split, monetary `K=3`,
aggregation и production weight `0.05` не менялись.

## BTYD FIT STATUS

**PASS.** Причина `exp_050` — optimization convergence: старый solver завершался
по relative function reduction при gradient до `0.002274`; на почти плоской
dropout-оси finite differences приводили три старта к разным недополированным
точкам. Это не оказалось реальной non-identifiability финального максимума.

- OOF 8/8 fits: max NLL spread `5.42e-11`, max log-param spread `0.001351`,
  max gradient norm `6.82e-8`; исходные gates сохранены и пройдены.
- Production donor 0/1: NLL spread `1.31e-12 / 7.53e-13`, log-param spread
  `0.000287 / 0.0000439`, max gradient `9.73e-8 / 1.20e-7`.
- Production parameters: donor 0 `(r=.612636, alpha=13.13969, a=.381899,
  b=983.2023)`; donor 1 `(.614722, 13.21532, .298215, 768.5915)`.

## BTYD OOF REVALIDATION

**PASS production gate.** Полный новый OOF построен тем же optimizer policy.

| Fold | nested delta | fixed BTYD05 delta |
|---|---:|---:|
| 2025-09-04 | −0.000654668 | −0.000654668 |
| 2025-09-18 | −0.000570928 | −0.000570928 |
| 2025-10-02 | −0.000175076 | −0.000241882 |
| 2025-10-16 | −0.000192613 | −0.000256336 |

- Nested LOFO: **−0.000269182, 4/4**, практически exact `exp_047`
  (`−0.000269184`); selected weights `.05/.05/.10/.10`.
- Fixed production `w=.05`: **−0.000320983, 4/4**.
- Residual alignment положительна 4/4; pooled centered alignment `0.02486`.
- Start prediction stability OOF: max `Var(z_i-z_j)=1.52e-10`,
  max `|delta z|=0.000831`; fixed-5% score span `2.62e-10`. Все заранее
  записанные predictive gates пройдены.

Старый `exp_047` research promotion gate `nested <= -0.0003` формально всё ещё
даёт `REJECT`; это не использовано как новый research claim. Текущий явный
production gate требовал сохранения materially close signal и ≥3/4 — он PASS.

## FRESH RETRAIN STATUS

**NOT RUN after CASE B submission.** Exact family аудирован: `SEQ-D3A-BASE`
TCN, hidden 64, 8 blocks, kernel 3, pooled `[last,mean,max]`, CLEAN/EXTRA,
two-sided A/B, CLEAN/VOL/FRESH heads, `FRESH-CLEAN`, donor-safe 0.5/99.5%
winsorization, GLOBAL, centering, alpha 1, depth clip 289.

Historical fold encoders имеют `workers=3` race policy, exact TEST encoder и
saved heads отсутствуют. Честный DET-PAIR rebuild потребовал бы **5 новых
encoder runs** (4 folds + production) и 30 conditional-head fits. После
получения одного честного CASE B candidate задание требует остановиться, поэтому
plans/checkpoints не создавались; audit сохранён в `fresh_retrain_audit.json`.

## FRESH OOF REVALIDATION

**NOT RUN.** Старое `−0.000225` не приписывалось новой trajectory; FRESH не
включён в submission.

## PRODUCTION SUPPORT

**PASS.** Exact 250,000-user alignment с raw/sample/STRONGEST, hash sides и
schema/order подтверждены; NaN/inf/negative/duplicates/missing = 0.

- `Var(correction_test)/Var(correction_oof)=1.17342` внутри `[0.6,1.4]`.
- Test start stability: max `Var(z_i-z_j)=5.42e-13`, max `|delta z|=7.81e-5`;
  для centered 5% correction max `|delta|=3.90e-6`.
- Direct BTYD clipping 0; tail-at-cap30 mean `7.85e-6`; QMC-grid excursions:
  `mu<min 0.0016%`, остальные 0.
- Final mean `log1p` после штатного level shift: `2.32930000017`.

## SUBMISSION STATUS

**CREATED — CASE B.** Фиксированный recipe:

```text
BTYD05 = 0.95 * STRONGEST_CURRENT + 0.05 * BTYD
```

`submissions/submission_BTYD05.csv`: 250,000 строк, exact schema/order,
нулей `0.0508%`, SHA256
`c3cfb4d90f50ceff8f5d8f8aaca072664966fb91018eb0a3fa01195dc38c2932`.
Leaderboard upload не выполнялся.

## Вердикт и вывод

**SUCCESS / STOP BRANCH.** Production blocker снят без изменения model или
ослабления gates; новый OOF воспроизвёл сигнал `exp_047`, test support PASS,
один честный BTYD05 candidate создан. Согласно заданию BTYD/FRESH research
остановлен; FRESH, новые BTYD variants и следующий target-decomposition branch
не запускались.

## Артефакты и hashes

- Config: `artifacts/BTYD_STABLE_EXP051/config.json`
  (`9b22fe14bb9733478f5302eeb048d25b6de5ae65ec2999217d4558c67a00c77b`).
- OOF: `artifacts/BTYD_STABLE_EXP051/oof_raw.npz`
  (`093358cc2da37fd5aa545ec3c68f693e5de1d13889bcc5006d62a5ed131f87c7`).
- Test raw: `artifacts/BTYD_STABLE_EXP051/test_raw.npz`
  (`5222d26166c600ba201958937d7226ba535a49a1c7aeb2a8dc3b328b437e5a43`).
- Results: `research/strategies/results/BTYD_STABLE_EXP051/` — fit details,
  OOF/test predictive stability, production support, FRESH audit, summary.
- Tests: `python -m pytest src/test_btyd_stable_fit.py
  src/test_btyd_day_bgnbd.py src/test_validation.py -q` → **42 passed**.
