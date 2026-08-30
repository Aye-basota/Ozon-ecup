# Forensic-аудит пайплайна сокомандника и leaderboard evidence

Дата среза: 2026-08-25. Исходный repository: `C:/Users/Admin/Desktop/OZON-E-CUP`. Аудит read-only; исходный repository не изменялся.

## Границы и правила evidence

Проверены каталоги `пайплайн сокомандника/`, `submissions/`, `deliverables/`, `weights_archives/`, включая извлечённые и архивные review bundles. Источники-инструкции и вторичные state/summary-файлы (`AGENTS*`, `STATE*`, `HISTORY*`, `README*`, roadmap/TODO/master-summary/provenance) не использовались как источник экспериментальных фактов. `REPORT_RU.txt` использовался только для навигации; числа брались из CSV/JSON/runtime/manifests/scripts и реальных prediction/submission artifacts. Первичные карточки конкретных экспериментов `exp_*.md` использованы как допустимое evidence после сверки с артефактами.

Термин «подтверждённый LB» ниже означает надёжную внутреннюю repository-связку `score ↔ конкретный существующий CSV`: первичный experiment report и/или отслеживаемый исторический ledger плюс реальный artifact. В repository нет platform export, API response, screenshot или иного независимого снимка leaderboard. Поэтому даже 11 подтверждённых строк — repository-internal verification, не внешняя аттестация платформой.

## Инвентаризация зоны

| Root | Files | Bytes | Существенные типы |
|---|---:|---:|---|
| `пайплайн сокомандника/` | 147 | 372,279,298 | 55 CSV, 30 Python, 18 NPY, 14 Markdown, 5 JSON, 5 logs, 5 PT, 3 ZIP, manifests |
| `submissions/` | 25 | 103,620,464 | 25 CSV, включая 2 файла в `FINAL_20260825_A1/` |
| `deliverables/` | 3 | 209,339,985 | 2 ZIP и вторичная LB-таблица |
| `weights_archives/` | 4 | 80,057,515 | 4 checkpoint ZIP |
| **Итого** | **179** | **765,297,262** | все 179 disk-файлов SHA-256 hashed |

Полная hash-дедупликация дала 153 уникальных SHA-256, 9 duplicate-групп и 35 входящих в них файлов.

## Подтверждённый production package `STRONGEST_CURRENT`

`deliverables/submission_STRONGEST_CURRENT_artifacts_2026-08-20.zip`:

- SHA-256 `d9bb7276f889826873b5a4d33e8466b625cc324f75647599c278acd774da8a43`;
- 72 ZIP entries; все 72 byte-exact совпадают с extracted package;
- nested `MANIFEST.sha256`: 71 payload entry, 71 OK, 0 mismatch, 0 missing;
- extracted package имеет один дополнительный ожидаемый файл — сгенерированный `submission_STRONGEST_CURRENT_rebuilt.csv`;
- canonical, rebuilt, root `submissions/submission_STRONGEST_CURRENT.csv`, `latest/components/friend.csv` и final candidate A имеют один SHA-256: `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.

Первичная recipe из `pipeline/build_submission.py` — взвешивание `z=log1p(predict)`:

| Component | Weight |
|---|---:|
| S1-CAP | 0.100 |
| S1-UNC | 0.200 |
| S1-DIST | 0.250 |
| SEQ-01 | 0.075 |
| SEQ-C289-S43 | 0.075 |
| SEQ-C289-S44 | 0.075 |
| ETX-01-S42-DCW | 0.075 |
| ETX-01-S43-DCW | 0.075 |
| ETX-01-S44-DCW | 0.075 |

После смеси применяется shift к target level 2.3293 и `max(z, 0)`. Независимый numeric rebuild дал:

- mean до shift: `2.464661871972273`;
- shift: `-0.13536187197227312`;
- mean после clipping/CSV round-trip: `2.3293213697927495`;
- max absolute log error к canonical CSV: `4.96611577169119e-7`;
- при штатной шестизначной сериализации итог byte-identical canonical CSV.

В package находятся 9 `ztest_*.npy`, 9 соответствующих `uid_*.npy`, 5 test checkpoints и 5 run logs. Все UID-массивы byte-identical, `int64`, длины 250,000, без дублей, в exact sample order. Девять `ztest` — `float64`, длины 250,000. Пять test checkpoint также дословно входят в соответствующие weight archives.

## Восстановленная ancestry production-решения

Ancestry является DAG, а не одной линейной веткой:

```text
tabular:  exp_006 S1-BEST ──> exp_014 S1-DIST ───────────────┐
sequence: exp_025 SEQ-01 -> exp_027 depth -> exp_035 AVG3 ───┼─> exp_037 STRONGEST_CURRENT
event:    exp_036 ETX-01 -------------------------------> exp_037
                                                               |
                                                               v
fixedstack review -> final6h occurrence -> extra90 cached meta
       \                                      /
        + friend=.12 + occ_meta_B=.16 + occ_raw_X3=.72 -> latest.csv
```

Семь первичных experiment units package:

| ID | Date | Фактическая роль | Primary measured result | LB |
|---|---|---|---|---:|
| exp_006 | 2026-08-10 | исходная LightGBM-смесь | CV mean 1.75886 vs B0 1.76879 | 1.6512802628833827 |
| exp_014 | 2026-08-11 | distribution head / DIST blend | CV mean 1.75834 | 1.6507774106 |
| exp_025 | 2026-08-13 | raw-sequence TCN как компонент | mixture wCV 1.748343; honest LOFO -0.00106, 4/4 | 1.650176372731295 |
| exp_027 | 2026-08-13 | inference-only depth diagnostic | local wCV 1.74774, но full-depth LB transfer отрицательный | 1.6553135958569027 |
| exp_035 | 2026-08-19 | clip289 SEQ seed-average | fixed mixture wCV 1.74777; LOFO -0.00055, 4/4 | not submitted |
| exp_036 | 2026-08-20 | sparse event Transformer | ETX-S42 wCV 1.74953; ETX+SEQ LOFO -0.00091, 4/4 | not submitted |
| exp_037 | 2026-08-20 | ETX-AVG3 + SEQ-AVG3 join | full mixture wCV 1.74751; LOFO -0.00092, 4/4 | 1.6496571 |

Записанные verdict не принимались без чисел. Например, exp_025 имеет standalone SEQ-01 хуже табличной смеси (`1.75270`), но положительный ensemble LOFO; эти два факта сохранены отдельно. exp_027 локально выглядел лучше, но реальный LB full-depth submission хуже на `+0.0051372231`; это не смешивается с clip289-кандидатом exp_035.

## Review bundles: реальные run units

Три top-level run manifest полностью присутствуют; их ZIP и extracted trees byte-exact:

| Run | Start/finish | Runtime | Новое обучение | Candidate table | Materialized CSV |
|---|---|---:|---|---:|---:|
| `fixedstack_combo_10h_2026-08-23_001` | 04:18:50–05:46:54 | 1.470208674 h | `recent_hurdle_fast12`, `recent_hurdle_stable18` OOF | 50 canonical names | 4 |
| `final6h_fixedfriend_2026-08-23_001` | 16:18:39–20:48:23 | 4.496829306 h | 8 occurrence configs × 4 folds + TEST; stable18 TEST | 166 canonical names | 2 |
| `extra90m_cached_meta_2026-08-23_001` | finish 22:25:55 | 31.111260 min | none; cached outputs only | 83 canonical names | 4 |

Archive fingerprints and extraction:

- fixedstack: SHA `f6921f298acfb540190f2d7d0ce36ad9473fa7c6161b949e809d2916a3e9f791`, 24/24 exact;
- final6h: SHA `fe84467066993e7aab40efe8b54343efc5fb9366eee5d5e187b7b21f865807fe`, 15/15 exact;
- extra90: SHA `e51822f916f926286be25ec7d2251c1887dfeae61999145786af4382be595144`, 13/13 exact.

Внутри run-time evidence есть 10 child training units: 2 hurdle variants и 8 occurrence configs. Final6h runtime содержит 41 строки: stable18 TEST плюс `8 × (4 folds + TEST)`. Суммарные runtime occurrence-конфигураций: `occ_r10_fast` 0.255321 h, `r12_wide` 0.374100 h, `r14_multiscale` 0.349096 h, `r16_bal` 0.469936 h, `r18_wide` 0.584617 h, `r20_shallow` 0.589509 h, `r22_stable` 0.618921 h, `r24_multiscale` 0.607721 h.

Таким образом, точный count run-level единиц этой линии:

- 3 top-level review runs;
- 10 runtime-backed child training units;
- **13 manifested/runtime-backed run units**;
- отдельно 191 deduplicated candidate-evaluation names;
- отдельно 7 предшествующих primary package experiment units.

## Нормализация review-метрик и deduplication

Review protocol не совпадает с full STRONGEST_CURRENT CV. `table_core` — нормированная пропорция `CAP/UNC/DIST = 0.10/0.20/0.25` внутри table-slot 0.55. Его baseline `wCV = 1.749803702558191`. Meta-model fit walk-forward: для каждой даты используются только предыдущие folds; на первом fold meta-training отсутствует, поэтому fold остаётся baseline. Fold weights — 1:2:4:8. Все `delta`/`delta_table` в bundles относятся к этой table-side метрике.

Следовательно, значения `1.74798...` review-кандидатов нельзя напрямую сравнивать с full STRONGEST_CURRENT `1.74751`, standalone model score или public LB. Они сопоставимы только внутри одного review protocol против `table_core`.

Полный candidate audit:

- 22 CSV с полным набором `wcv/base_wcv/delta/fold_scores`;
- 1,111 строк metric evidence;
- 191 уникальное имя кандидата;
- 264 уникальные пары `(name, metric signature)`;
- 183 имени присутствуют в canonical `ALL_*` tables;
- ещё 8 `__slotbeta750/875` вариантов присутствуют только в `FINAL_CANDIDATE_METRICS.csv`;
- 37 имён меняют метрики между fixedstack stages: 36 имеют 3 версии, 1 имеет 2 версии;
- это объясняется изменением expert pool после добавления fast12/stable18, поэтому старые значения не перезаписаны и не смешаны;
- по финальной canonical версии 174 кандидата имеют отрицательную delta против `table_core`, 17 — положительную, 0 exact ties.

26 bottom-up families найдены непосредственно в таблицах: occurrence overlay/meta/meta-risk, ridge subset/slot-strength/shrink/temporal/pred-only, adaptive blend, super-ridge/recent/pband/simplex, xmeta plain/risk, raw new/raw occurrence, local bias/trust, candidate pband/simplex, hierarchical, greedy, pband, simplex и occurrence calibration. Самая крупная — `occurrence_overlay` (80); затем `ridge_subset` (12); по 10 записей имеют `adaptive_blend`, `occurrence_meta` и `occurrence_meta_risk`.

Лучший table-side metric, без LB-интерпретации:

`xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`: wCV `1.7479830389407023`, delta `-0.0018206636174889232`, recent wins 3/3, latest delta `-0.0020723716645272283`.

## Materialized review submissions

Создано 10 CSV, 9 уникальных SHA-256. Все 10 имеют 250,000 строк, exact schema/sample UID order, finite/nonnegative predictions и mean `log1p` около 2.3293. Ни у одного нет подтверждённого LB score.

Точные дубликаты:

- final6h branch A = extra90 rank 1 (`c2efd7f100e0c3fabc6399d75cddfcede0fa463a8a94cd4633720125f7ee5e15`);
- final6h branch B = `latest/components/occ_meta_B.csv` (`8d90a0bb1afdfe48cb6181cf1869eb30095da729e36dbea81e2051730e7989d8`);
- extra90 rank 3 = `latest/components/occ_raw_X3.csv` (`0ac3f241685d6562c1f5b54993065475845464a0b217a56dbb5d79b42dd27356`).

Последние selected metrics — факты review protocol, не рекомендации:

- fixedstack: четыре materialized branches, delta_table от `-0.0014245197` до `-0.0015469442`;
- final6h A: `-0.0016507644`; B occurrence-meta-risk: `-0.0017665444`;
- extra90: anchor `-0.0016507644`, xmeta-risk `-0.0018206636`, raw-occ `-0.0016246012`, hierarchical `-0.0013961545`.

## `latest.csv`: recipe восстановлена, LB не подтверждён

`latest/rebuild_latest.py` задаёт единственную точную recipe:

```text
z_latest = 0.12 * z(friend)
         + 0.16 * z(occ_meta_B)
         + 0.72 * z(occ_raw_X3)
predict = expm1(max(z_latest, 0))
```

Post-blend level normalization отсутствует. Компоненты имеют ясную ancestry: `friend` — byte-copy STRONGEST_CURRENT; `occ_meta_B` — byte-copy final6h B; `occ_raw_X3` — byte-copy extra90 rank 3.

- `latest.csv` SHA: `7ef5b2c58925bd28c5bc7eb83b9cfd4785c608a0c8b2a6d7a3277730cba8e722`;
- `latest_rebuilt.csv` SHA: `a9dc2dabdc693cd510c0428c501154898ec11f1e15b0694261471757cae274a1`;
- max log error: `8.881784197001252e-16`;
- max raw prediction error: `2.8421709430404007e-13`;
- mean log1p canonical: `2.32930256438812`.

Файлы не byte-identical только из-за CSV serialization. Outer `пайплайн сокомандника/MANIFEST.sha256` ошибочно требует для rebuilt файла reference hash canonical latest. Итог outer manifest: 90/91 OK, 1 mismatch, 0 missing.

Заявление `latest → public LB 1.64921756224069` не внесено в verified registry. Число действительно переписано в exp_067 audit manifest и позднюю working-tree строку ledger рядом с `latest.csv`, но сам exp_067 прямо фиксирует upstream provenance как `EXTERNALLY_REPORTED` из README/text provenance. Это downstream-транскрипция claim, а не независимый upload record. SHA и recipe надёжно связывают **artifact с его реконструкцией**, но не связывают **LB event со SHA artifact**: отсутствуют original teammate score registry/run manifest с парой `7ef5... ↔ 1.6492175622`, дата загрузки и platform export. Canonical OOF также отсутствует, но это отдельный CV-reproducibility caveat, не причина отказа именно в LB binding.

## Root submissions и recipe coverage

В `submissions/` найдено 25 CSV, 23 уникальных hash. Все проходят 250,000-row/schema/order/finite/nonnegative проверки. Две exact duplicate пары:

- `submission_STRONGEST_CURRENT.csv` = `FINAL_20260825_A1/candidate_A_STRONGEST_CURRENT.csv`;
- `submission_BTYD05.csv` = `FINAL_20260825_A1/candidate_B_BTYD05_HEDGE.csv`.

После forensic reconstruction нет ни одного существующего submission с полностью неизвестной recipe. Однако три S04-файла не имели recorded-at-creation ledger/report recipe:

- `submission_s04_a.csv` точно восстанавливается из `ztest_S04-A` с level shift 2.3293;
- `submission_s04_b.csv` — из `ztest_S04-B` с тем же level shift;
- `submission_s04_blend.csv` — `0.30 S1-UNC + 0.10 S1-CAP + 0.15 S1-DIST + 0.45 S04-B` в log space, intercept `-0.1480783154`;
- max log reconstruction error каждого около `5e-7`, согласуется с шестизначным CSV rounding.

То есть count `submission without any recipe after audit = 0`; count `recipe not recorded at creation, reconstructed forensically = 3`.

## Подтверждённая leaderboard chronology

В `leaderboard_verified.jsonl` записано ровно 11 score↔existing artifact связок:

| Date | File | Exact public score | SHA-256 |
|---|---|---:|---|
| 2026-08-10 | `submission_strategy_1_level_low.csv` | 1.6519789982910107 | `44a310304845e90243f0b50166d12351e19be180b7a5e99ec2ae819c4bd5f6ad` |
| 2026-08-10 | `submission_strategy_1.csv` | 1.6512802628833827 | `b70afce2f4352edb2beb205a17ffb04292ba3ff595f802cfdf6d69edfbdd1de3` |
| 2026-08-10 | `submission_strategy_1_level_high.csv` | 1.6529908823677866 | `d4ce4f929f012da69649de0b29525d7c18fe9d3dc306be072d50cc8b7bceb0ba` |
| 2026-08-10 | `experimental_submission_1.csv` | 1.667424590457357 | `1b950d09c6aefb69fec179b7323d47108b69930deb2c73d3907988f14e58fad6` |
| 2026-08-10 | `experimental_submission_2.csv` | 1.6682180280505314 | `319b777d624f9cd3d7c295829a06db514c89291c5da59c4f3f67ab1c23954b89` |
| 2026-08-11 | `submission_dist_head.csv` | 1.6507774106 | `5e0b6494aa106c95f96784fae79a28191ee15f4820929635df0e0ac09ed8c72f` |
| 2026-08-11 | `submission_candidate_e11mix.csv` | 1.6510029 | `11bd0dded68d960e8272b3e3cb62bfbe6a66553edf4b0b7b089e96895fed6a8e` |
| 2026-08-11 | `submission_strategy_2.csv` (copied from linked worktree) | 1.6619324597771563 | `cab04e8cc94066819ebe1c548624267768c380b1c98b183331c7d13625b01668` |
| 2026-08-13 | `submission_SEQ01_mix.csv` | 1.650176372731295 | `ce2f535561a3673c29726833b96fa4444e3b3dc51912c58799ea55a41ef67964` |
| 2026-08-13 | `submission_SEQAVG3_mix.csv` | 1.6553135958569027 | `25c1cc5edc559de46c6f2950be78054dd6ece3446c901a2295fb4cbf3f66227b` |
| 2026-08-20 | `submission_STRONGEST_CURRENT.csv` | 1.6496571 | `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda` |

Для E11 нет противоречия между exp_013 и exp_016: exp_013 фиксирует более раннюю standalone E11-модель как не отправленную; exp_016 и ledger относятся к поздней смеси, включающей E11.

S2 после reconciliation также считается strong repository-internal link. Copied artifact `evidence/worktree_artifacts/team_a_s2/submissions/submission_strategy_2.csv` и source в linked worktree byte-identical, размер 6,358,662 bytes, SHA `cab04e8c...b01668`; `worktree_artifacts_manifest.csv` фиксирует эту пару. `s2_final.json` называет тот же output path и подтверждает 250,000 строк, exact order, finite/nonnegative и mean log1p 2.3293. Primary exp_012 называет этот filename и exact LB `1.6619324597771563`; ledger округляет совместимо до `1.6619325`. Внешнего platform export всё равно нет, поэтому strength тот же repository-internal, что и у остальных 10 строк, а не platform-independent.

## LB claims, которые не прошли verify

Четыре score claims в teammate line оставлены вне verified chronology:

1. `latest = 1.64921756224069`: artifact существует и SHA-bound к recipe/rebuild; score переписан в exp_067 manifest, но тот же manifest помечает его `EXTERNALLY_REPORTED`, то есть LB event не SHA-bound.
2. `known_ridge_submission_public = 1.6492897556391737`: hard-coded в final6h manifest/script, конкретный preserved CSV не связан.
3. `ranker_safe = 1.654133685532829`: hard-coded constant; diversity CSV доказывает, что старый prediction был загружен во время run, но submission CSV не сохранён.
4. `class1_occ = 1.688068573391526`: аналогично — runtime-derived diversity evidence есть, submission CSV отсутствует.

Отдельные категории:

- S2 больше не orphan: artifact сохранён из linked worktree и включён в verified chronology;
- ambiguous score с двумя существующими artifacts: `1.6512012383165489` — primary exp_015 прямо говорит «лучший из двух» F4-файлов без идентификации; tracked ledger округляет и назначает score mix-файлу, поэтому оба исключены из verified registry;
- существующие review submissions: 10 файлов, но подтверждённых LB у них 0.

Итого по score-bearing missing/unbound artifacts: три логические записи — unnamed known-ridge, old ranker_safe и old class1_occ. Это script/run-context claims без сохранённой однозначной artifact-привязки.

## Weight archives

| Archive | SHA-256 | Entries | Unique payloads | Reproducibility |
|---|---|---:|---:|---|
| `ETX-01_weights.zip` | `4d1a0763dd793049016b97883f02a29353280d42bfdc43779fa60b68898c03a1` | 10 | 10 | partial: S43 только TEST; S44 без V1016 |
| `TCN_SEQ-01_weights.zip` | `686932a06cbf2f44835a41eec6fff4722c58464e344583da6727d6e818c757f3` | 10 | 10 | partial: неполные S42/S43 grids |
| `TCN_DETSEQ01_weights.zip` | `0498d8da965604253c0603c8af568943a55aa6aaacf2a320e87f942275012f0c` | 3 | 3 | только fold V1016 × seeds 42/43/44 |
| `TCN_SEQ-D3A_weights.zip` | `e3d2ca469c2d876318d576ae9bc35654b203c098f2cede4e2c1e7e20db45b794` | 29 | 28 | heterogeneous incomplete grids |

Всего 52 checkpoint entries, 51 уникальный payload. В D3A archive `model_SEQ-03A-BASE-S42-V1016.pt` и `model_SEQ-D3A-BASE-S42-V1016.pt` byte-identical. Пять archive entries — exact copies test-models strong package. Ни у одного weight archive нет sidecar manifest, связывающего каждый checkpoint с полным config/data fingerprint/metric. Эти архивы являются artifact collections, а не полной гарантией OOF replay.

## Dataset package

`deliverables/ozon_e_cup_base_data.zip` SHA `6c451f69f8a52def781ae814822a611398a68fc974e7b307e0bd09ad542d4afc`, 4 entries. Внутренние fingerprints:

- `train.parquet`: `5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0`;
- `sample_submit.csv`: `06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1`.

Оба совпадают с data-файлами, на которые опирается teammate package.

## Exact duplicate groups

Девять disk-level SHA-групп:

1. 9 одинаковых UID NPY (`50e5ba...`).
2. 6 одинаковых canonical combo validation snapshots (`6029a9...`).
3. 5 одинаковых primitive validation snapshots (`2a53f5...`).
4. 5 копий STRONGEST_CURRENT CSV (`abc221...`).
5. extra90 rank3 = latest occ_raw_X3 (`0ac3f2...`).
6. final6h B = latest occ_meta_B (`8d90a0...`).
7. final6h A = extra90 rank1 (`c2efd7...`).
8. BTYD submission = final candidate B (`c3cfb4...`).
9. 2 одинаковых stable-branch validation snapshots (`f7a5de...`).

## Существенные contradictions и confounders

1. **Outer manifest mismatch:** `latest_rebuilt.csv` numeric-exact, но не byte-exact; manifest содержит неправильный expected serialization hash.
2. **Latest LB unverified:** score не имеет primary upload event или platform evidence.
3. **F4 ambiguity:** один score, два реально существующих файла, primary report не различает их.
4. **Old best_bas score claims:** ridge/ranker/class1 score constants не привязаны к preserved files.
5. **Same-name metric drift:** 37 review names имеют 2–3 разные metric versions из-за stage-dependent expert pool.
6. **Metric incomparability:** table-slot review wCV, full ensemble wCV, standalone wCV и LB нельзя смешивать.
7. **Incomplete replay evidence:** review bundles сохраняют result tables и selected submissions, но не canonical OOF arrays/models всех experts; full candidate replay невозможен только из bundle.
8. **Dirty ledger caveat:** рабочая копия `experiments/submissions.csv` изменена относительно HEAD; для исторических LB-связок использован tracked HEAD ledger и primary reports, а не поздние добавленные summary-строки. Исключение reconciliation — S2 подтверждён independent linked-worktree artifact chain.

## Параллельные research lines

Найдены как минимум три реально независимые линии:

- A1/package production DAG: tabular distribution + TCN + ETX, завершается reproducible STRONGEST_CURRENT;
- best_bas teammate line: fixed table stack → recent hurdle → occurrence probability/meta → cached extra90; завершается артефактом latest, но без canonical OOF/LB event;
- checkpoint exploration line: SEQ-01, DETSEQ, D3A variants и ETX seed/fold grids; частично пересекается с production package, но содержит неполные и отдельные variant grids.

Review manifests также ссылаются на более ранние `run_best_bas_research_23h.py`, `continue_best_bas_12h_v2.py` и fixed-stack parent. Их old predictions использовались во время runs, но standalone manifests/submission files этих ранних прогонов не сохранены в текущем teammate tree. Поэтому они зарегистрированы как ancestry/context, а не как полностью воспроизводимые отдельные runs.

## Completeness verdict для этой зоны

- 7/7 primary experiment reports внутри strong package связаны с experiment records.
- 3/3 review run manifests связаны с run records; 3/3 archives совпадают с extracted contents.
- 10/10 runtime-backed child training units восстановлены.
- 191 review candidate names deduplicated; все 264 metric versions учтены агрегатно и конфликтный count зафиксирован.
- 10/10 materialized review submissions валидированы; 9 unique hashes.
- 25/25 root submissions валидированы; 23 unique hashes.
- 11 score↔existing artifact LB rows имеют сильную внутреннюю связку; одиннадцатая — reconciled S2 из linked worktree.
- 4 unverified teammate score claims, 3 missing/unbound old best_bas artifacts и 1 ambiguous F4 score зарегистрированы отдельно.
- 4/4 checkpoint archives проиндексированы с SHA и completeness caveats.
- STRONGEST_CURRENT полностью воспроизводим до byte-identical submission при наличии dataset package.
- latest recipe воспроизводима численно, но её local validation ancestry и LB event неполны.

Оценка полноты именно для audited teammate/LB зоны: высокая для inventory, hashes, packages, submission recipes и review metric tables; средняя для полного обучения/replay best_bas ветки; недостаточная для независимой platform-verification LB и для старых best_bas submissions, которые не были сохранены.

Machine-readable companion files: `evidence/teammate_records.jsonl` и `evidence/leaderboard_verified.jsonl`.
