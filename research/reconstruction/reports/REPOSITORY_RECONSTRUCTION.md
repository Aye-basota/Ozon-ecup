# Repository reconstruction

## 1. Что было найдено

Исходный `OZON-E-CUP` оказался не одной линейной историей, а совокупностью
главного worktree, Git-истории и шести связанных worktree. В основном дереве
проиндексировано 3 599 файлов вне `.git`/cache; ещё 1 293 пути из linked
worktree уникальны по содержимому относительно main. Полный source footprint
включал около 51.9 GiB, поэтому большие data/OOF/test/checkpoint artifacts не
копировались: для них сохранены пути, размеры, SHA-256 и schema/array metadata.

Найдены и нормализованы:

- 124 строки первичных experiment reports: 65 Team A current, 4 Team A
  Strategy 2, 33 Team B core, 17 Team B alternate и 5 independent branches;
- 124/124 отчёта имеют отдельный design-evidence record с train construction,
  features, target, folds, seeds и существенными hyperparameters; отсутствующее
  сохранено как `unknown`/`not_applicable`, а прямые и унаследованные fold dates
  различены;
- один дополнительный реально выполненный, но безномерный machine-only S04 run;
- 13 runtime-backed teammate review/training run units;
- 1 134 granular machine metric/manifest records;
- 614 групп OOF/test/model components;
- 222 code/config units главного worktree и 136 report-referenced исторических
  code snapshots из Git refs;
- 21 семейство datasets/cache artifacts;
- 11 сильных repository-internal связок public-LB score ↔ существующий CSV.

Основные file-role counts главного worktree: 649 datasets, 447 OOF predictions,
64 test predictions, 204 checkpoints/models, 55 fold artifacts, 223 metric/
diagnostic files, 51 submission-role files, 35 manifests, 62 training scripts,
92 inference scripts, 79 ensemble scripts, 31 feature-generation scripts и 5
postprocessing scripts. Один файл может иметь несколько ролей.

Старые AGENTS/STATE/HISTORY/roadmap/TODO/master-summary документы
зарегистрированы отдельно, но ни одно их утверждение не использовано как факт.

## 2. Сколько экспериментов

Центральный registry содержит **138 evidence units**:

| Grain | Count |
|---|---:|
| primary report rows | 124 |
| machine-only experiment units | 1 |
| teammate top-level/child run units | 13 |
| total registry rows | 138 |

После схлопывания четырёх строк, являющихся точной duplicate-document или
seed/replay реализацией уже учтённой гипотезы, остаётся **134 novelty-level
units**. Обе величины сохранены: 138 — для lossless истории запусков, 134 — для
счёта уникальных экспериментальных единиц на уровне новизны.

У каждого registry row есть namespaced ID, family, parent baseline, изменение,
model/validation, CV/delta/LB, status, evidence strength, artifacts,
compatibility tags, reproducibility, FACTS, INTERPRETATION и conflicts.
Отсутствующие значения записаны как `unknown`.

## 3. Duplicate/rerun/reuse

Выделено **11 experiment-level clusters**. Информация не удалялась.

Ключевые случаи:

- второй manifest EXP-032 описывает тот же experiment;
- EXP-030B/030C — seed/multiseed rerun-линия EXP-030;
- `team_b_core:EXP-031` — точный semantic replay `team_b_core:EXP-005`;
- два EXP-020 baseline NPZ — array-identical aliases при разных container hashes;
- EXP-029/030 используют один и тот же V1016 baseline artifact;
- EXP-032B bitwise переиспользует conditional predictions EXP-032, меняя
  extensive component;
- EXP-038 содержит stochastic same-seed BASE-R2 noise control;
- второй run внутри EXP-043 exact-identical первому;
- EXP-051 численно replay-ит OOF EXP-047, но имеет другой production optimizer
  и test artifacts;
- EXP-067 V1/V2 — частично дублированные audit directories;
- EXP-061/062/064 и связанные preflights приходят к baseline-identical zero
  correction, оставаясь разными гипотезами.

Отдельно учтены 7 teammate package copies, 412 exact-content groups в
объединённом уникальном файловом инвентаре и 2 152 cross-worktree checksum
groups. Эти файловые дубликаты не объявлялись experiment duplicates без
семантического evidence.

## 4. Research families

Bottom-up taxonomy содержит **12 families**:

1. behavioral occurrence and BTYD;
2. calibration and postprocessing;
3. domain shift, unlabeled data, and dataset identity;
4. ensembles, stacking, and component selection;
5. neural sequence and event models;
6. production integration and provenance;
7. residual and correction models;
8. tabular models and feature engineering;
9. target distribution and decomposition;
10. temporal history and calendar;
11. train-example construction;
12. validation, reproducibility, and diagnostics.

В `registry/family_summary.csv` best и median представлены отдельно для каждого
`comparison_class`. Несовместимые folds, train coverage, standalone/ensemble,
LOFO, simulation, AUC и LB не агрегируются.

## 5. Подтверждённые submissions и leaderboard

Подтверждено **11 repository-internal score-to-existing-CSV links**:

- три calibration-level файла S1-BEST;
- EXP-MIN и EXP-SIM;
- S1-DIST-MIX;
- S1-MIX-E11;
- S2-BEST из linked Strategy-2 worktree;
- SEQ-01-MIX;
- SEQ-AVG3-MIX;
- STRONGEST_CURRENT.

Для каждого сохранены filename, exact score, recipe, SHA-256, lineage и evidence.
**Ни для одного нет независимого platform export/screenshot**, поэтому термин
«confirmed» означает только сильную внутреннюю repository linkage.

Path-level submission registry содержит 36 валидных существующих CSV с 33
уникальными SHA-256. У всех 36 есть forensic recipe: 20 exact-recorded, 3
semantic-recorded, 3 numerically reconstructed S04 и 10 producer-script
semantic recipes; `unknown` recipes — 0.

Ещё 19 records находятся в `leaderboard/report_only_claims.csv`: 17
experiment-level claims и 2 teammate context constants. Там явно различены
отсутствующий CSV, неоднозначная привязка к нескольким CSV и существующий
artifact без SHA-bound upload/score event. `latest.csv`, например, точно
реконструируется как artifact, но его LB event остаётся неподтверждённым.

## 6. Baseline chronology

Baseline не один; он менялся внутри нескольких несовместимых линий.

- **Team A current, ранняя tabular линия:** S1-B0 → dense/history/normalized
  components → S1-BEST → S1-DIST-MIX.
- **Team A current, sequence линия:** S1-DIST-MIX → SEQ-01-MIX →
  SEQ-AVG3/clip289 → ETX+SEQ STRONGEST_CURRENT. Ранний equal-mean CV и поздний
  calibrated 1:2:4:8 wCV хранятся раздельно.
- **Strategy 2:** pure FW/structural aggregation → hybrid QMC/FW → count hurdle
  → K-shrink monetary component → S2-BEST. Simulation, single-fold screen и
  final four-fold RMSLE — разные comparison classes.
- **Team B core:** собственный LightGBM baseline `team_b_core:EXP-001` с двумя
  decision folds и отдельным fold-3 diagnostic.
- **Team B alternate:** собственный HGBR single-cutoff baseline, затем отдельные
  recency/LightGBM/scale и two-fold alignment anchors.
- **Teammate review:** baseline `STRONGEST_CURRENT + table_core` относится к
  review walk-forward stack и не равен полному STRONGEST CV.
- **Independent branches:** renewal, calendar, domain, global-regime и exact-
  anniversary используют branch-specific anchors; их bare IDs и абсолютные
  метрики не переносятся в другие линии.

Полный список из 82 явно упомянутых baseline/reference recipes находится в
`baselines/chronology.csv`.

## 7. Параллельные research lines и ancestry

Обнаружены как минимум пять одновременно развивавшихся pipeline lines:

1. основной Team A tabular→distribution→TCN→ETX ensemble;
2. отдельный Team A Strategy-2 structural count/monetary pipeline;
3. Team B core LightGBM feature/calibration/hurdle/seasonality pipeline;
4. Team B alternate HGBR→recency LightGBM→scale/log-ensemble pipeline;
5. teammate fixed-stack→occurrence→cached-meta review pipeline.

Кроме них существовали параллельные conditional S04, BG/NBD residual,
postprocessing, residual, validation/provenance и independent mechanism-audit
ветви. Полный directed graph содержит 232 edges; читаемая principal genealogy
дана в `ensembles/SOLUTION_ANCESTRY.md`.

## 8. Серьёзные contradictions

Все **128** conflict/caveat rows сохранены в `contradictions/registry.csv`:
106 получены из primary/machine forensic audit, ещё 22 — из отдельной проверки
17 старых summary/navigation документов (15 high, 7 medium). Последние помечены
`secondary-only; used_for_facts=no`; canonical resolution всегда опирается на
primary report или machine evidence. Audit caveats и comparability warnings
помечены отдельно от material tensions. Наиболее важные:

- `EXP-057` и `EXP-058` означают разные эксперименты в current и independent
  namespaces; всего разрешено 33 cross-namespace local-ID collisions;
- EXP-015 F4 score нельзя однозначно связать с одним из двух существующих CSV;
- несколько старых summary приписывают E03a неверный parent baseline B0;
  primary report сравнивает его с dense-cutoff S1-E02, поэтому causal delta
  меняет знак;
- Team B имеет 12 report-only LB records и 13 названных, но отсутствующих CSV;
- в Team B несколько local scale/calibration winners расходятся с report-only LB;
- EXP-026 одновременно меняет seed average, blend weights и test depth; его
  ранняя production interpretation устаревает после EXP-027;
- S04 final recipe не имеет исходного manifest: он восстановлен уникальным
  exact 0.05-grid match всех 250 000 строк;
- EXP-035 machine table численно предпочитает D3A average, а созданный submission
  использует SEQ-AVG3; причина выбора не зафиксирована;
- EXP-041/042 создают submissions после validation REJECT; EXP-042 также имеет
  test-reference lineage mismatch;
- EXP-048 смешивает fold sets; EXP-049 является исправленным three-fold audit и
  не сравнивается с исходным числом напрямую;
- EXP-051 OOF summary говорит REJECT/no promotion, production support — PASS и
  создаёт submission;
- S2 machine grid минимально предпочитает K=8, downstream policy явно фиксирует
  K=5;
- independent global-regime не имеет требуемого canonical OOF и публикует только
  re-anchored comparison;
- exact-anniversary retrain проходит parity tolerance, но не bitwise identity;
- teammate outer manifest проходит 90/91 из-за serialization hash reconstructed
  CSV при численном совпадении до 8.88e-16; 37 одинаковых candidate names имеют
  разные metric versions между stages;
- EXP-067/latest artifact identity сильная, но reported LB event имеет только
  circular/secondary provenance и не повышен до confirmed.

Ни один конфликт не разрешён угадыванием или переписыванием неудобного verdict.

## 9. Каких artifacts не хватает

- 13 названных Team B submission CSV;
- три old teammate score-bearing artifacts/привязки: ridge, `ranker_safe`,
  `class1_occ`;
- два manifested, но не materialized current CSV без LB score:
  `submission_ETX_SEQ_mix.csv` и `submission_SEQAVG3_clip289_full5.csv`;
- DIST G90 selected arm для EXP-019;
- machine result artifacts для EXP-028 preflight;
- seed-43/44 full-depth checkpoints EXP-026 при наличии test predictions;
- canonical candidate OOF EXP-027;
- full-panel/LOFO validation для EXP-032/032B;
- часть exact production checkpoints/weights и полных runtime traces поздних
  sequence experiments;
- canonical row-level OOF/ancestry для двух late components в EXP-066–068;
- exact historical member banks/meta matrices для replay EXP-068;
- любые независимые competition-platform LB exports.

Отсутствие artifact не заменялось значением из summary-документа.

## 10. Completeness

Проверки перед завершением:

- 124/124 primary report rows присутствуют в registry;
- все 88 report candidates текущего worktree связаны с primary report, machine
  audit или exact teammate/package copy; unresolved report orphans — 0;
- 18 первоначально не связанных prediction components разрешены через code,
  report и recipe evidence; unresolved среди этой выборки — 0;
- глобальные experiment IDs уникальны; bare-ID collisions вынесены отдельно;
- 118 main scripts не имеют explicit primary-report link и сохранены как orphan
  candidates, а не объявлены неиспользованными;
- 19 registry units не имеют canonical numeric CV/delta/LB: 13 — runtime-only
  teammate units, остальные — blocked/preflight/manifest/production-only случаи;
- 19 unverified LB claim records не смешаны с 11 confirmed internal links;
- 24 исключённых interpretive documents инвентаризированы; использованных как
  source of facts — 0;
- 17 scoped summary/navigation документов отдельно сопоставлены с canonical
  evidence: 22 материальных расхождения сохранены как secondary-only и не
  использованы для заполнения experiment FACTS;
- source worktree и шесть linked worktrees повторно хешируются относительно
  pre-audit snapshot; результат находится в
  `reports/source_unchanged_verification.json`.

Финальная integrity-проверка дала PASS: 3 599/3 599 файлов main и все 3 549
проинвентаризированных файлов шести linked worktrees имеют прежние size/SHA-256;
missing/extra/changed = 0. `git status` каждого worktree также точно совпал со
снимком до аудита, включая уже существовавшие пользовательские изменения main и
три untracked файла exact-anniversary worktree.

Полнота **высокая для доступных primary reports, machine artifacts, Git refs,
linked worktrees, file identity и внутренней submission provenance**. Она
**ограничена для внешнего leaderboard, удалённых Team B CSV, исторических OOF
banks и runs, от которых остался только primary report**. Эти ограничения
являются частью registry, а не скрытой неопределённостью.
