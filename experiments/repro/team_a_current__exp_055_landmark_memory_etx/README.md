# exp_055 — retrospective landmark outcome memory

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_055_landmark_memory_etx`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_055_landmark_memory_etx`
- **Original source:** `experiments/exp_055_landmark_memory_etx.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, sequence model
- **Features:** recency, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** SHUFFLED outcome; shared feature/config/validation code не менялся.
- **Known score:** wCV `1.7475098625`; максимум расхождения с registered values `4.48e-10`.
- **Seed:** lr .03, feature/bagging .8, lambda_l2 20, max_bin 63, seed 42.
- **Postprocessing:** None documented
- **Submission:** GPU/full folds/test inference/public LB/submission: **не запускались**.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_055 — retrospective landmark outcome memory

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`

## Гипотеза

Cutoff-safe последовательность прошлых пар `historical state → realized next-30d GMV`
может содержать информацию для текущего direct-прогноза, которой нет в
`STRONGEST-CURRENT`. Главный причинный контроль — та же landmark representation,
но с outcome, детерминированно перемешанными между пользователями внутри
`query cutoff × lag × donor-only past30-GMV decile`.

Raw same-user correlation `corr(z_T,z_{T+60})=0.4980` восстановлена как факт N9 из
`research/strategy_NN_report.md` и `research/strategy_NN_1.md` (также процитирована
в `exp_024`). Это autocorrelation сырого target, а не evidence incremental utility.

## Что изменено относительно базы

Новые isolated runner/tests и cutoff-safe landmark caches: 16 фиксированных лагов
`30,45,...,255`, 20 numeric fields, QUERY с zero/masked outcome, REAL и matched
SHUFFLED outcome; shared feature/config/validation code не менялся.

## Exact baseline audit

- `BASE_HEAD=a28a71f`, совпадение с baseline parallel experiment: **PASS**.
- `STRONGEST-CURRENT` fold scores:
  `[1.7668833568, 1.7605095768, 1.7486292240, 1.7412785664]`.
- wCV `1.7475098625`; максимум расхождения с registered values `4.48e-10`.
- Exact OOF alignment: 770,616 строк, 0 missing, 0 duplicates; key SHA256
  `9e9c9de2d280e856eb1172830519fb044c346a319d99b9ed33d0834d04ab067a`.
- File SHA256: CAP `38fb0270...6d525`, UNC `2a8e543f...874ff`, DIST
  `7ef12519...4c4e`, ETX-AVG3 `890aef1a...bf2f`, SEQ-AVG3
  `8e8ec790...646c`; reconstructed STRONGEST prediction
  `b3000884...4b91`.

## Dataset, leakage и control audit

- Landmark state строго `(t-30,t]`, historical outcome строго `(t,t+30]`;
  каждый valid token удовлетворяет `t+30<=T`. Недоступная история — PAD/mask.
- QUERY outcome всегда zero/masked; мутации после `T` и в current target
  `(T,T+30]` не меняют input. User ID не является feature.
- REAL cache shape `(770616,17,20)`, `float16`; 16 landmark tokens + QUERY.
- SHUFFLED state/token shape идентичны REAL; меняется только materialized outcome
  field. Seed только `config.SEED=42`; permutation сохранена. Multiset outcome
  сохранён в каждом stratum; changed fraction `0.669736/0.669749` для направлений
  `0→1/1→0`.
- Similarity использует только 18 state fields, donor-only standardization и cosine;
  summaries фиксированы: last/mean/median/trend/sim-weighted/nearest/max-sim/count.

### Методологическое ограничение preregistered mutation test

При stride 15 и окнах длиной 30 raw-окна landmarks неизбежно перекрываются.
Поэтому порча сырого события внутри `(t,t+30]` не может менять только outcome одного
landmark: она также попадает в соседние outcome и/или state windows. Проверка
«только соответствующий landmark» корректна и выполнена для materialized outcome
channel — именно на нём определён REAL→SHUFFLED intervention. Отдельный тест
фиксирует неизбежный raw-overlap; leakage current target при этом отсутствует.

## CPU pre-flight

Fixed EXP-053 LightGBM probe: exact COMBINED `227 state + 34 disagreement`, signed
fold-calibrated STRONGEST residual, donor early folds одной user-half → recipient
10-16 другой half, scale только `{0,.25,.5,1}`.

| Gate | Результат | PASS |
|---|---:|:---:|
| REAL late delta ≤ −0.0005 | `0.000000` (scale `0/0`) | no |
| REAL − SHUFFLED ≤ −0.0004 | `0.000000` | no |
| Обе recipient halves лучше | `0.000000 / 0.000000` | no |
| Residual alignment >0 в обеих halves | `0.006877 / 0.011361` | yes |
| Nearest-state лучше last-outcome control | обе scale `0/0`, RMSLE tie | no |
| Pooled partial residual corr ≥0.02 | `0.005814` | no |

Late-fold RMSLE у BASE/REAL/SHUFFLED/NEAREST_ONLY/LAST_ONLY одинаков:
`1.7412785664`. SHUFFLED alignment (`0.007061/0.011069`) не хуже по смыслу REAL;
scale-selection отклонил все ненулевые corrections.

Diagnostics на recipient halves:

- `corr(current target, nearest)` = `0.480859 / 0.491288`;
- `corr(current target, last)` = `0.542185 / 0.548327`;
- `corr(STRONGEST residual, nearest innovation)` = `0.001037 / 0.010590`;
- после контроля `w30/w90/w180 GMV + recency` partial target-nearest correlation
  остаётся `0.130914 / 0.140863`, но в incremental RMSLE не переносится.

## Результат

- CV по фолдам кандидата: не запускался — CPU gate закрыт.
- Baseline wCV: `1.747509863` (лучший на момент: `exp_037`,
  `STRONGEST-CURRENT 1.74751`).
- Pilot: **не запускался**, status `PROHIBITED_BY_PREFLIGHT`.
- GPU/full folds/test inference/public LB/submission: **не запускались**.
- CPU runtime: `861.26 s`.
- Tests: `25 passed` focused и `142 passed` adjacent non-slow regression suite;
  analysis-only replay canonical hashes: **PASS**.

## Вердикт и вывод

**NO_GO_PREFLIGHT; PROMOTE_TO_FULL_FOLDS=NO.** Исторические outcomes сильно
коррелируют с текущим target на raw/conditional axes, но не дают измеримой
incremental correction поверх `STRONGEST-CURRENT`; REAL не отделяется от SHUFFLED.
Не спасать эту exact ветку sweep-ом stride/window/lags, state vector, shuffle strata,
probe scales или architecture.

## Конфиг прогона

- Landmarks: 16, stride 15, lags 30..255; state/outcome windows по 30 дней.
- Probe: LightGBM regression_l1, 200 rounds, 31 leaves, min leaf 2000,
  lr .03, feature/bagging .8, lambda_l2 20, max_bin 63, seed 42.
- Preregistered neural contract (CPU hash only): ETX d128, 5 blocks, 8 heads,
  head_dim16, FFN384, dropout .1, direct head, 4 epochs; identical REAL/SHUF model,
  optimizer, batch and LR hashes. Neural training не разрешён gate-ом.

## Артефакты

- `artifacts/LANDMARK_MEMORY_EXP055/`: baseline manifest, REAL/SHUF token caches,
  masks, past30 GMV, materialized permutations/manifests, memory summaries,
  CPU predictions, paired CPU contract/hashes.
- `research/strategies/results/LANDMARK_MEMORY_EXP055/`: schedule/config,
  safe-window audit, direction/pooled/correlation tables, pre-flight verdict,
  summary и reproducibility hashes.
