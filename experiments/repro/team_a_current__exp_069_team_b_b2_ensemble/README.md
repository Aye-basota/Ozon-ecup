# exp_069 — TEAM-B-B2 ensemble

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_069_team_b_b2_ensemble`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_069_team_b_b2_ensemble`
- **Original source:** `experiments/exp_069_team_b_b2_ensemble.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** CatBoost, ensemble, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Наш replay: **1.747509863**; по фолдам `1.766883 / 1.760510 / 1.748629 / 1.741279`.
- **Known score:** PREPARED по запросу, но REJECT как новый чемпион:** LOFO подтверждает неслучайность optimum веса на этих фолдах, однако выигрыш `0.000151` ниже project gate `0.0005` и ожидаемого public noise `0.00025`. `STRONGEST-CURRENT` остаётся research/private-safe anchor; соседние веса, поздние загрязнённые cutoff'ы и public-LB tuning не продолжать.
- **Seed:** Pinned `features.py/train.py/predict.py` из team SHA `88dc691…`; exact 4 модели команды (LGBM classifier, positive LGBM, all-user LGBM, CatBoost), seed **42** только из `config.py`, 260 features. S1 validation dates 09-04/09-18/10-02/10-16; team feature cutoff = `V+1`, latest safe single train snapshot starts `V-29` and its target ends on `V`. Production team model: cutoff `2025-10-17` (history through clean `2025-10-16`), test feature cutoff `2026-02-14` (history through `2026-02-13`). Training
- **Postprocessing:** Pinned `features.py/train.py/predict.py` из team SHA `88dc691…`; exact 4 модели команды (LGBM classifier, positive LGBM, all-user LGBM, CatBoost), seed **42** только из `config.py`, 260 features. S1 validation dates 09-04/09-18/10-02/10-16; team feature cutoff = `V+1`, latest safe single train snapshot starts `V-29` and its target ends on `V`. Production team model: cutoff `2025-10-17` (history through clean `2025-10-16`), test feature cutoff `2026-02-14` (history through `2026-02-13`). Training
- **Submission:** LB: не отправляли. Submission создан: `submissions/submission_TEAM_B_B2_OPTIMAL_ENSEMBLE.csv`, SHA256 `9b2cc1b4cd3de6394f1e891904f58ab6e5c5f8c7b3505d83d797bd706bad899f`.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_069 — TEAM-B-B2 ensemble

- **Дата:** 2026-08-27
- **Автор:** A1
- **Коммит:** `a28a71f` + working tree; team source pinned at `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`

## Гипотеза

Финальный four-model tabular pipeline команды B2 может быть слабее `STRONGEST-CURRENT` отдельно, но дать полезную ошибочную декорреляцию. Вес подбирается только на канонических OOF S1 в raw-GMV формуле `w*our + (1-w)*team`; public LB не используется.

## Что изменено относительно базы

Добавлен один внешний член `team-b-B2` без изменений рабочего `src/features.py`: pinned team-код запускается изолированно через `src/team_b_b2_ensemble.py`.

## Результат

- Наш replay: **1.747509863**; по фолдам `1.766883 / 1.760510 / 1.748629 / 1.741279`.
- Team B2: **1.750956977**; по фолдам `1.771174 / 1.763374 / 1.752128 / 1.744740`.
- Математический optimum: **w(our)=0.835139593**, `w(team)=0.164860407`.
- Blend: **1.747357382**, дельта к our **−0.000152481**, лучше 3/4, включая 10-16; 09-04 хуже на +0.000040.
- Честный LOFO: **1.747358931**, дельта **−0.000150931**; held-out веса our `0.8276 / 0.8416 / 0.8325 / 0.8427`.
- OOF→TEST `Var(z_our-z_team)` = `0.019255→0.024607`, ratio **1.278**; перенос слабее подтверждённого режима основных компонентов.
- LB: не отправляли. Submission создан: `submissions/submission_TEAM_B_B2_OPTIMAL_ENSEMBLE.csv`, SHA256 `9b2cc1b4cd3de6394f1e891904f58ab6e5c5f8c7b3505d83d797bd706bad899f`.

## Вердикт и вывод

**PREPARED по запросу, но REJECT как новый чемпион:** LOFO подтверждает неслучайность optimum веса на этих фолдах, однако выигрыш `0.000151` ниже project gate `0.0005` и ожидаемого public noise `0.00025`. `STRONGEST-CURRENT` остаётся research/private-safe anchor; соседние веса, поздние загрязнённые cutoff'ы и public-LB tuning не продолжать.

## Конфиг прогона

Pinned `features.py/train.py/predict.py` из team SHA `88dc691…`; exact 4 модели команды (LGBM classifier, positive LGBM, all-user LGBM, CatBoost), seed **42** только из `config.py`, 260 features. S1 validation dates 09-04/09-18/10-02/10-16; team feature cutoff = `V+1`, latest safe single train snapshot starts `V-29` and its target ends on `V`. Production team model: cutoff `2025-10-17` (history through clean `2025-10-16`), test feature cutoff `2026-02-14` (history through `2026-02-13`). Training 5 model sets: **19.1 min**. Финальный raw blend получил frozen production log-level `2.3293` одним глобальным сдвигом `−0.02302193`; 250,000 rows/order/schema/finite/nonnegative PASS. Synthetic post-cutoff perturbation PASS; team targets совпали с canonical OOF (`max abs <= 9.47e-4`, floating aggregation only); 20 focused tests PASS.
