# exp_044 — fresh conditional supervision при paired fine-tune SEQ-01

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_044_fresh_conditional_ft`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_044_fresh_conditional_ft`
- **Original source:** `experiments/exp_044_fresh_conditional_ft.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** dilated TCN, sequence model
- **Features:** freshness/conditional features, history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Если физически свежие positive-only cutoff'ы несут полезный условный сигнал, то при одинаковом объёме conditional-supervision они должны перестроить encoder plain `SEQ-01` полезнее, чем старые CLEAN-positive доноры. Проверяем только разность `FT-FRESH − FT-VOL` на заново обученных детерминированных baseline seeds 42/43/44 и одном фолде 2025-10-16.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Итог: `artifacts/FRESH_COND_FT_EXP044/analysis.json`, `seed_summary.csv`, `segment_summary.csv`.
- **Postprocessing:** None documented
- **Submission:** LB/test/full-fold/LOFO/submission: **не запускались**.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_044 — fresh conditional supervision при paired fine-tune SEQ-01

- **Дата:** 2026-08-22
- **Автор:** A1
- **Коммит:** a28a71f

## Гипотеза

Если физически свежие positive-only cutoff'ы несут полезный условный сигнал, то при одинаковом объёме conditional-supervision они должны перестроить encoder plain `SEQ-01` полезнее, чем старые CLEAN-positive доноры. Проверяем только разность `FT-FRESH − FT-VOL` на заново обученных детерминированных baseline seeds 42/43/44 и одном фолде 2025-10-16.

## Что изменено относительно базы

После нового plain `SEQ-01` checkpoint заморожена direct head и на один CLEAN epoch совместно оптимизированы encoder и отдельная conditional head; arms различаются только 128 добавленными positive-only примерами в conditional half-batch: равнообъёмный CLEAN `VOL` против физически свежего `FRESH`.

## Технический гейт

- Baselines `DETSEQ01-S42/S43/S44-V1016` обучены с нуля, без D3A, старых checkpoint'ов и `depth_aug`: 17 каналов, hidden 64, 8 TCN blocks, window 365, batch 1024, 4 CLEAN epochs.
- Детерминированная политика: `workers=1`, materialized index/LR plans, fixed Python/NumPy/Torch/CUDA RNG, deterministic CUDA, bf16, TF32, eager, отдельный процесс на run.
- Один baseline checkpoint, direct rows/order, common CLEAN-positive rows, added slot IDs, conditional-head init и LR multiplier общие внутри каждой seed-пары. `FRESH` и `VOL` имеют по 619,776 added slots; donor pools по 887,996 строк, sampling с replacement.
- EXTRA только group B, только `z>0`, `depth_clip=289`, conditional loss; EXTRA не попадает в direct loss. Conditional head `192→64→GELU→Dropout(0.1)→1`; direct head заморожена и не используется в conditional inference.
- Fine-tune: AdamW, encoder LR `3e-5`, conditional-head LR `1e-3`, weight decay `1e-2`, `lambda_cond=0.25`, warmup 300 + общий cosine multiplier, 4,842 шага. Snapshots: 0/1/100/1000/2421/4842.
- Strict replay `FT-VOL-S42`, два отдельных процесса ×100 шагов: prediction/model/optimizer/head/Python+NumPy+Torch+CUDA RNG совпали точно; `Var(Δz)=0`, `max|Δz|=0`, SHA256 обоих прогнозов `67ee85d8…c3798`.
- Технический verdict: **PASS**. Все 25 обязательных artifact-backed invariants прошли; вместе с regression-тестами SEQ/DET/COND — **143 passed**.

## Результат

Baseline и full-step RMSLE_cal на фолде 2025-10-16:

| Seed | baseline | FT-VOL | FT-FRESH | FRESH−VOL | VOL−base | FRESH−base |
|------|---------:|-------:|---------:|----------:|---------:|-----------:|
| 42 | 1.739396619 | 1.739459703 | 1.739352194 | **−0.000107510** | +0.000063084 | −0.000044425 |
| 43 | 1.737668562 | 1.737790116 | 1.737774927 | **−0.000015188** | +0.000121553 | +0.000106365 |
| 44 | 1.738704254 | 1.738940963 | 1.738799379 | **−0.000141583** | +0.000236709 | +0.000095126 |

Paired diagnostics на endpoint:

| Seed | ΔAUC | Δpositive | Δaux RMSE | Var(Δz) | max\|Δz\| | Pearson | mean Δz | offset diff |
|------|-----:|----------:|----------:|--------:|----------:|--------:|--------:|------------:|
| 42 | −0.000007643 | −0.000130994 | −0.002716322 | 0.000072789 | 0.06250 | 0.999984958 | +0.001140598 | −0.001140726 |
| 43 | −0.000020324 | +0.000574353 | −0.003220702 | 0.000134269 | 0.09375 | 0.999976038 | −0.001382819 | +0.001331464 |
| 44 | −0.000010538 | +0.000442978 | −0.002735936 | 0.000121943 | 0.09375 | 0.999976678 | −0.000242079 | +0.000242137 |

Prediction SHA-256 (`baseline / VOL / FRESH`):

- S42: `e14e7fb3b5b4fa3d85c2862884d716acf843201b9f6ac039d5a786031c7c91b0` / `97d24e0b3cf8739e1bddb511bb36ecc09394f13d6cce39a6c0709a1853d3b6a2` / `9017800fa131914a2dcff7029c0bc8dc525ff4d73330a3a84f088921c0f671ea`.
- S43: `97a996d9a1b0efd35adc4a0fe21a3178ca5bfa6f6129422181080e0a4e71f0a1` / `f5d5edb4cdec52fca558c4d87369f66453292c6de65d4bf5f17da7160c46f086` / `9c0bc3356ee83538331e0879dbe9cc8ed65e9f2b1dba9bead2e4e2bc06391ae8`.
- S44: `b8863e9412aaf666e91bc2ea706fb43077dfa36aca35bff7e33eacb0ba0ef25f` / `bd072024759e3298187c528444466c0794320b6f2f69451eada4b290fce7097f` / `e5957d0e9935a044ed1fceb6babbc0ab14a82f9272d3e45c2201143fcb475382`.

- Mean paired delta: **−0.000088094**; median **−0.000107510**; sample sd **0.000065396**; отрицательных seeds **3/3**.
- Полупрогон, step 2421: deltas `−0.000101522 / +0.000003037 / −0.000185233`, mean **−0.000094573**. Знак и масштаб результата не являются артефактом последней половины schedule.
- Mean `FRESH−baseline`: **+0.000052355**. То есть FRESH немного лучше VOL, но не лучше исходного checkpoint в среднем.
- Mean AUC delta `−0.000012835`; positive-only error delta **+0.000295446** (лучше лишь 1/3); auxiliary conditional RMSE delta `−0.002890987`.
- Conditional loss действительно учится лучше на FRESH: full train loss 1.2668–1.2717 против VOL 1.2750–1.2798. На step 4842 conditional→encoder gradients ненулевые во всех runs: weighted norms 0.0753–0.1748; direct norms 0.2227–0.3568. Direct-head hash неизменен.
- Pair movement мал и почти общий: pooled `Var(z_FRESH−z_VOL)=0.000110732`, `max|Δz|=0.09375`, Pearson 0.999977–0.999985. Mean `Δz` и разность calibration offsets взаимно компенсируются в пределах `5.1e-5` по seed.
- Сегменты не дают устойчивого механизма. `rec_buy 15–60`: delta `−0.000067/−0.000044/−0.000088`; `w180_days_buy≥16`: `−0.000131/+0.000052/−0.000355`; «никогда не покупал»: `−0.000047/+0.000300/+0.000039`.
- LB/test/full-fold/LOFO/submission: **не запускались**.

## Вердикт и вывод

**SIGNAL: REJECT. PROMOTE TO FULL FOLDS: NO.** Средняя paired delta выше rejection boundary `−0.0003`, а positive-only error в среднем ухудшился. Физически свежая conditional supervision распознаётся auxiliary head и даёт маленький однонаправленный сдвиг против volume control, но практически полезного улучшения direct forecast нет.

Доказан только controlled paired contrast на **новых deterministic plain SEQ-01 baselines**. Результат нельзя переносить на исторический `SEQ-AVG3`; он ничего не утверждает о `STRONGEST_CURRENT`, LOFO, test, LB или submission.

## Конфиг прогона

Fold `2025-10-16`; seeds только из `src/config.py`: 42/43/44. CLEAN train corridor и 24 cutoff'а наследуют SEQ-01. Baseline AdamW: LR 0.003, wd 0.01, warmup 300, 4 epochs. Fine-tune и conditional-конфиг приведены в техническом гейте. Полное baseline+arms GPU-время: 26,869 s (7.46 h), без коротких replay/анализа.

## Артефакты

- Runner: `src/fresh_cond_ft.py`; tests: `src/test_fresh_cond_ft.py`.
- Итог: `artifacts/FRESH_COND_FT_EXP044/analysis.json`, `seed_summary.csv`, `segment_summary.csv`.
- Plans и init: `artifacts/FRESH_COND_FT_EXP044/plans/`; baseline/arm snapshots и predictions: `baselines/`, `arms/`.
- Strict replay: `artifacts/FRESH_COND_FT_EXP044/integration_replay/comparison.json`.
