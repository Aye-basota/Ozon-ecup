# Logged run — SEQ-DEPTH

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_depth`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-DEPTH`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_027: разбор провала SEQAVG3-MIX на LB. Кросс-фолдовый стресс по глубине (ранняя модель -> поздняя панель, +49/+63/+77) и замер цены ухода канала avail из прожитого режима. Inference-only.
- **Known score:** conclusion:** ПРИЧИНА ПРОВАЛА НАЙДЕНА. Разложение тестового прогноза по точному пути (без допущений): глубина Var(dz)=0.00809 при старой доле SEQ 0.30 и 0.01819 при новой 0.45, усреднение 0.00128, веса 0.00175, итого 0.01323. Ось глубины в 6.3 раза больше усреднения и в 4.6 раза больше весов; усреднение и веса вместе объясняют <15% провала даже по верхней оценке. Факт LB +0.0051372 ПРЕВЫШАЕТ верхнюю оценку чистого шума +0.00401 (128%), то есть перестройка сдвинула прогнозы систематически против
- **Seed:** conclusion:** ПРИЧИНА ПРОВАЛА НАЙДЕНА. Разложение тестового прогноза по точному пути (без допущений): глубина Var(dz)=0.00809 при старой доле SEQ 0.30 и 0.01819 при новой 0.45, усреднение 0.00128, веса 0.00175, итого 0.01323. Ось глубины в 6.3 раза больше усреднения и в 4.6 раза больше весов; усреднение и веса вместе объясняют <15% провала даже по верхней оценке. Факт LB +0.0051372 ПРЕВЫШАЕТ верхнюю оценку чистого шума +0.00401 (128%), то есть перестройка сдвинула прогнозы систематически против
- **Postprocessing:** None documented
- **Submission:** conclusion:** ПРИЧИНА ПРОВАЛА НАЙДЕНА. Разложение тестового прогноза по точному пути (без допущений): глубина Var(dz)=0.00809 при старой доле SEQ 0.30 и 0.01819 при новой 0.45, усреднение 0.00128, веса 0.00175, итого 0.01323. Ось глубины в 6.3 раза больше усреднения и в 4.6 раза больше весов; усреднение и веса вместе объясняют <15% провала даже по верхней оценке. Факт LB +0.0051372 ПРЕВЫШАЕТ верхнюю оценку чистого шума +0.00401 (128%), то есть перестройка сдвинула прогнозы систематически против
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-DEPTH

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-DEPTH
- **timestamp:** 2026-08-13T20:46:55
- **commit:** 560b24b
- **description:** exp_027: разбор провала SEQAVG3-MIX на LB. Кросс-фолдовый стресс по глубине (ранняя модель -> поздняя панель, +49/+63/+77) и замер цены ухода канала avail из прожитого режима. Inference-only.
- **scenario:** S1
- **n_features:** 17
- **model:** tcn-diagnostic
- **params:** {"alpha_grid": [0, 0.25, 0.5, 0.75, 1], "availprobe_configs": 5, "crossdepth_pairs": 7, "extrap": [49, 63, 77], "mode": "inference-only", "seeds": [42, 43, 44]}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ACCEPT
- **conclusion:** ПРИЧИНА ПРОВАЛА НАЙДЕНА. Разложение тестового прогноза по точному пути (без допущений): глубина Var(dz)=0.00809 при старой доле SEQ 0.30 и 0.01819 при новой 0.45, усреднение 0.00128, веса 0.00175, итого 0.01323. Ось глубины в 6.3 раза больше усреднения и в 4.6 раза больше весов; усреднение и веса вместе объясняют <15% провала даже по верхней оценке. Факт LB +0.0051372 ПРЕВЫШАЕТ верхнюю оценку чистого шума +0.00401 (128%), то есть перестройка сдвинула прогнозы систематически против истины. КРОСС-ФОЛДОВЫЙ СТРЕСС (новая команда seq crossdepth: модель раннего фолда на поздней панели; пара 09-04->10-16 даёт +77 при тестовых +76; 7 пар, 3 сида, 3 размера): монотонность НЕ сохраняется, появляется внутренний оптимум (+42/+52/+65 при экстраполяции +49/+63/+77), у сида 43 деградация после +49 на +0.00113. НО полная глубина всё равно лучше обрезки (-0.00366 при +77), а усадка z(a)=(1-a)z_clip+a z_full даёт a*=0.75 устойчиво 7/7 LOFO и -0.00410. То есть стресс предсказывает, что глубина должна была ПОМОЧЬ — провал он не воспроизводит. МЕХАНИЗМ. Возмущение на тесте в 2.4 раза больше исторического аналога того же размера (Var 0.08990 против 0.03816). Причина структурная: у ВСЕХ 29 обучающих cutoff-ов и всех 4 фолдов в окне 365 не меньше 76 позиций с avail=0 (данные с 2025-01-01), а на тесте при полной глубине их РОВНО НОЛЬ — avail=1 всюду. Это единственная точка задачи с постоянным avail, обучающих примеров у неё ноль, и воспроизвести её на истории НЕВОЗМОЖНО. Прямой замер цены этого одного бита при неизменных данных (seq availprobe): +0.00339/+0.00355/+0.00387/+0.00443/+0.00398 на 5 конфигурациях, 3 сида, свой и чужой фолд. Это БОЛЬШЕ выигрыша от добавленных данных (-0.0037). Естественный контроль: у 8-11% пользователей в добавке нет ни одной строки, для них clip->full меняет ТОЛЬКО avail — эффект +0.00045, 6 пар из 7. СЛЕДСТВИЕ: OOF и LOFO всех фолдов считались в режиме avail с нулями, то есть были корректным прогнозом для сабмита С ОБРЕЗКОЙ и никогда — для полной глубины. exp_026 сравнил локальные числа одного рецепта с LB другого. ПОЛИТИКА ГЛУБИНЫ НА ТЕСТЕ: --depth-clip 289 (a=0). Историческое a*=0.75 переносу не подлежит: любое a>0 подмешивает непокрытый режим пропорционально. Гейтинг по активности в добавке отклонён замером (-0.00267 против -0.00291 у простой полной глубины). LOFO СЕМЕЙСТВ (опора — отправленный SEQ-01-MIX, CAP>=0.10): CAP+E02+DIST+SEQ и полная смесь по -0.00055 4/4; CAP+ROUNDS+DIST+SEQ -0.00044; CAP+DIST+SEQ -0.00041; CAP+SEQ +0.00069 1/4; без SEQ +0.00113 0/4. S1-E10 избыточен (тот же результат при вдвое более устойчивых весах), DIST выбрасывать нельзя, SEQ несёт всю смесь. Ни одно семейство не берёт порог отправки -0.0020. КОД: seq crossdepth и seq availprobe; train_test теперь ВСЕГДА сохраняет веса тестовой модели (их отсутствие и было причиной, по которой clip289 для сидов 43/44 пришлось восстанавливать допущением). test_seq.py 37 проверок. Детали exp_027
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
