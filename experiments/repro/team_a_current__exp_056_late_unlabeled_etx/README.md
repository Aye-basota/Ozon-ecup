# exp_056 — LATE-UNLABELED-ETX-ADAPT

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_056_late_unlabeled_etx`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_056_late_unlabeled_etx`
- **Original source:** `experiments/exp_056_late_unlabeled_etx.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** calibration diagnostic
- **Features:** calendar features, history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Starting checkpoint: `model_ETX-01-S42-V0904.pt`, SHA256 `558fd1554076ec03ea976aa31808697003e98591c03b7ddb9403a2a9fbc1ad53`; validation checkpoint `2025-09-04`, seed 42 из `config.py`.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** ETX `d_model=128`, 5 blocks, 8 heads, head dim 16, FFN 384, dropout 0.10; batch/chunk 512/128; AdamW `lr=1.5e-3`, `wd=1e-2`, warmup 500, `lambda_ssl=0.25`, SmoothL1 с fixed per-channel constant normalization, grad clip 1.0, bf16/TF32/eager, workers=1, deterministic CUDA, seed 42. На 8 GB GPU два loss terms накоплены отдельными backward до общего clip/optimizer step, чтобы не держать две ETX graphs одновременно; batch/objective/RNG contract не менялся.
- **Postprocessing:** `Var(z_late−z_control)=0.00017742`; `corr(correction, CONTROL residual)=+0.001432`; calibrated mean correction `≈0`, поэтому различие не является только level shift, но residual alignment практически нулевая.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_056 — LATE-UNLABELED-ETX-ADAPT

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`

## Гипотеза

Поздние неразмеченные input histories имеют заметный temporal/domain mismatch относительно supervised-era ETX. Если дообучить exact saved `ETX-01-S42-V0904` encoder на masked reconstruction позднего коридора, сохраняя frozen direct head и одинаковый clean direct rehearsal, то `LATE-SSL` должен улучшить прогноз относительно причинного контроля `CONTROL-CLEANSSL`.

## Что изменено относительно базы

Один paired input-only adaptation epoch: на каждом optimizer step одинаковый clean direct batch плюс masked reconstruction 15% реальных event tokens; arms отличаются только source dates неразмеченной истории.

## Exact audit и paired contract

- Starting checkpoint: `model_ETX-01-S42-V0904.pt`, SHA256 `558fd1554076ec03ea976aa31808697003e98591c03b7ddb9403a2a9fbc1ad53`; validation checkpoint `2025-09-04`, seed 42 из `config.py`.
- Original legal rehearsal cutoffs: weekly Thursday `2025-04-03…2025-07-31`; последний target заканчивается `2025-08-30 <= 2025-09-04`.
- Evaluation/tokenizer policy: exact 14 normalized behavioral channels, `n_tok=192`, history/static depth cap 212, query weekday Thursday. Production `2026-02-13` использован только для input embedding diagnostics с Thursday query context; direct head там не вызывался.
- CONTROL: weekly Thursday `2025-05-22…2025-07-31`; LATE: `2025-08-07…2025-10-16`.
- Exact matching по token count (строже preregistered decile) и buy-event-day activity bin: **2,095,321** examples в каждой arm, **4,094** одинаковых optimizer steps. Materialized plan SHA256 `c917c736441f6eb3e2c080e4d3aa95b13debeadff773faf9407cf0b1e55a674d`.
- **17,664,047 / 117,165,174 = 15.0762%** реальных токенов masked; один и тот же mask/LR/direct plan. Reconstruction target — exact normalized 14-vector; time/calendar/position channels не маскировались.
- Direct head, starting encoder, reconstruction-head init, optimizer and RNG hashes совпали между arms. 100-step replay `deterministic_replay_v2.json`: все model/reconstruction/optimizer/RNG/prediction hashes и snapshots совпали точно.
- Primary validation: `2025-10-16`, exact 3-block order, `n=197,379`, order SHA256 `227e156cea4c1a5a59485eb8ba1c4e8843aa1697b92ed889f1f36c2f93171d65`. Validation target впервые прочитан только на analysis stage.
- Tests: `python -m pytest src/test_late_unlabeled_etx.py -q` → **12 passed**.

## Domain-shift audit без target

| Пара | Adversarial AUC | PSI mean / max | Frozen ETX MMD² |
|---|---:|---:|---:|
| supervised A vs LATE | 0.897178 | 0.022672 / 0.136470 | 0.003111 |
| supervised A vs PRODUCTION | 0.901138 | 0.052515 / 0.382737 | 0.011460 |
| CONTROL vs LATE | 0.865693 | 0.011823 / 0.093194 | 0.001098 |

Сдвиг хорошо детектируется даже без explicit cutoff, weekday, raw depth и user_id. Однако adaptation его не уменьшила: MMD² LATE↔PRODUCTION стало `0.005648` BASE → `0.005754` CONTROL → `0.005792` LATE; CONTROL↔LATE — `0.001098` → `0.001234` → `0.001244`.

## Результат на 2025-10-16

| Модель | RMSLE raw | RMSLE calibrated | AUC(y>0) | zero RMSLE | positive RMSLE |
|---|---:|---:|---:|---:|---:|
| no adaptation | 1.754175 | 1.749787 | 0.842369 | 1.860276 | 1.676244 |
| CONTROL-CLEANSSL | 1.753266 | 1.749372 | 0.842329 | 1.858439 | 1.676823 |
| LATE-SSL | 1.753265 | 1.749403 | 0.842334 | 1.858300 | 1.676974 |

- Primary causal contrast `LATE−CONTROL`: raw **−0.00000149**, calibrated **+0.00003163**.
- Fixed candidate slot: `LATE_SLOT−BASE_SLOT` raw **+0.00000367**, calibrated **+0.00000723**.
- Fixed user halves: standalone **+0.00005845 / +0.00000504**; slot **+0.00001297 / +0.00000153** — обе половины неправильного знака.
- `Var(z_late−z_control)=0.00017742`; `corr(correction, CONTROL residual)=+0.001432`; calibrated mean correction `≈0`, поэтому различие не является только level shift, но residual alignment практически нулевая.
- Все preregistered behavioral segments ухудшились; только `y=0` улучшился на `−0.000139`, тогда как `y>0` ухудшился на `+0.000151`.
- Clean direct holdout MSE BASE/CONTROL/LATE: **3.138318 / 3.141582 / 3.141207** — обе adaptation arms деградировали относительно starting checkpoint.
- Reconstruction честно улучшилась: CONTROL first20→last20 `1.092995→0.792168`, LATE `1.029612→0.773096`, но direct forecast не улучшился.
- Full folds, production/test prediction, submission и LB не запускались.

## Вердикт и вывод

**REJECT; `PROMOTE_TO_FULL_FOLDS=NO`.** Поздний input-only reconstruction учит источник и обнаруживает реальный domain shift, но causal `LATE−CONTROL` после калибровки хуже, обе fixed user halves неправильного знака, fixed slot хуже, clean rehearsal деградирует, а embedding MMD не уменьшается. Не повторять эту exact one-epoch masked-reconstruction ветку и не спасать её sweep по mask/lambda/epoch/head.

## Конфиг прогона

ETX `d_model=128`, 5 blocks, 8 heads, head dim 16, FFN 384, dropout 0.10; batch/chunk 512/128; AdamW `lr=1.5e-3`, `wd=1e-2`, warmup 500, `lambda_ssl=0.25`, SmoothL1 с fixed per-channel constant normalization, grad clip 1.0, bf16/TF32/eager, workers=1, deterministic CUDA, seed 42. На 8 GB GPU два loss terms накоплены отдельными backward до общего clip/optimizer step, чтобы не держать две ETX graphs одновременно; batch/objective/RNG contract не менялся.

Артефакты и подробные таблицы: `research/strategies/results/LATE_SSL_EXP056/`.
