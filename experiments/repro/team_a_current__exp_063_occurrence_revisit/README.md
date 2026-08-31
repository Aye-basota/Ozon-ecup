# exp_063 — OCCURRENCE-REVISIT

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_063_occurrence_revisit`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_063_occurrence_revisit`
- **Original source:** `experiments/exp_063_occurrence_revisit.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Nested E11 дельты: `+0.0000677 / +0.0000641 / −0.0000096 / 0`; wCV **+0.0000105**, 1/4, latest `0`.
- **Known score:** Nested E11 дельты: `+0.0000677 / +0.0000641 / −0.0000096 / 0`; wCV **+0.0000105**, 1/4, latest `0`.
- **Seed:** Seed/model training отсутствуют; exact 770,616-row aligned OOF SHA256 `40aa9719...0ed0a`; `oof_S1-E11.npz` SHA256 `39f566b4...0c44`, `oof_S1-E10.npz` `b4e43277...0f5c`; canonical folds 2025-09-04/09-18/10-02/10-16 and weights 1:2:4:8. Runner `python src/occurrence_revisit.py`; frozen protocol and results in `research/strategies/results/OCCURRENCE_REVISIT_EXP063/`.
- **Postprocessing:** Только log-space member `z(alpha)=(1-alpha)·z_exp037+alpha·z_E11` на заранее фиксированной сетке `0/.025/.05/.075/.10/.15/.20`; модели и признаки не переобучались.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_063 — OCCURRENCE-REVISIT

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** a28a71f + working tree

## Гипотеза

Существующий two-part `S1-E11` (вероятность покупки × conditional positive GMV) может быть комплементарен exact ETX+SEQ чемпиону, хотя старый `exp_016` измерял его только против более слабой смеси. Artifact-only nested LOFO против exp_037 с same-feature direct `S1-E10` контролем должен отделить реальную occurrence-информацию от выигрыша на слабой базе/public-LB calibration.

## Что изменено относительно базы

Только log-space member `z(alpha)=(1-alpha)·z_exp037+alpha·z_E11` на заранее фиксированной сетке `0/.025/.05/.075/.10/.15/.20`; модели и признаки не переобучались.

## Результат

- Nested E11 дельты: `+0.0000677 / +0.0000641 / −0.0000096 / 0`; wCV **+0.0000105**, 1/4, latest `0`.
- Held-out alpha E11: `.075/.075/.05/0`; E10 direct control: `0/0/0/0`, nested delta `0`.
- Лучший fixed E11 alpha `.05`: лишь **−0.0000098** wCV; ниже noise floor примерно в 50 раз.
- Standalone residual corr E11 vs champion `0.997894`; `Var(E11−strong)=0.01300`.
- LB: не отправляли; test audit/submission не запускались.

## Вердикт и вывод

**REJECT.** Exact alignment/target/finiteness/base-replay audits прошли, но frozen gates провалены: latest выбирает ноль, всего 1/4, nested delta неправильного знака и E11 не отделяется от direct control. Direct occurrence-member integration закрыта; не спасать соседними весами, retraining, segment gates или public-LB calibration.

## Конфиг прогона

Seed/model training отсутствуют; exact 770,616-row aligned OOF SHA256 `40aa9719...0ed0a`; `oof_S1-E11.npz` SHA256 `39f566b4...0c44`, `oof_S1-E10.npz` `b4e43277...0f5c`; canonical folds 2025-09-04/09-18/10-02/10-16 and weights 1:2:4:8. Runner `python src/occurrence_revisit.py`; frozen protocol and results in `research/strategies/results/OCCURRENCE_REVISIT_EXP063/`.
