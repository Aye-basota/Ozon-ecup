# exp_011 — S2-E02: сила EB-усадки и sigma ablation

## Catalogue metadata

- **Catalogue ID:** `strategy_2__exp_011_s2_k_calibration`
- **Namespace:** `strategy_2`
- **Experiment ID:** `exp_011_s2_k_calibration`
- **Original source:** `git:3c1d86d836c7:experiments/exp_011_s2_k_calibration.md`
- **Source ref:** `3c1d86d836c7b73519abe99f94686431852187cc`
- **Source commit:** `2e1d89d7904ee161939d9c6eed44fe16d4e4c549`
- **Kind:** git-history experiment card
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** на двух out-of-time фолдах. `sigma_scale` разрешено выбирать только внутри train.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_011 — S2-E02: сила EB-усадки и sigma ablation

- **Дата:** 2026-08-11
- **Автор:** A2
- **Коммит:** working tree, `team-a-strategy-2-impl`

## Гипотеза

`K=3` недостаточно усаживает пользовательский чек; оптимум должен быть устойчив
на двух out-of-time фолдах. `sigma_scale` разрешено выбирать только внутри train.

## Что изменено относительно базы

Проверены `K={1,2,3,5,8,15}` и `sigma_scale={0.8,0.9,1.0}`.

## Результат

- Двухфолдовый CV по K: 1.76950 / 1.76704 / 1.76582 / **1.76492** /
  **1.76485** / 1.76561.
- `K=5`: 1.77351 / 1.75634; `K=8`: 1.77371 / 1.75598.
- Разница K=5 и K=8 = 0.00007; выбран центр плато `K=5`.
- Train-only sigma-choice: 1.0 на сентябрьском пути, 0.8 на октябрьском;
  итог **1.76481**, выигрыш всего 0.00011.
- LB: не отправляли.

## Вердикт и вывод

Более сильная усадка подтверждена, но точный минимум неразличим. Принят `K=5`;
sigma выбирается вложенно, фиксировать 0.8 нельзя.

## Конфиг прогона

Фолды 2025-09-18/2025-10-16, offset-Poisson+hurdle, hybrid QMC/FW, 600 rounds,
seed из `src/config.py`.
