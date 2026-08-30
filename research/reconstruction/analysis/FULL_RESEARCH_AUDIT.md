# Full research audit

Дата аудита: 2026-08-25. Метрика: RMSLE, меньше — лучше. Во всех таблицах `delta < 0` означает улучшение.

## Executive verdict

1. **Research/private-safe baseline сейчас — `team_a_current:EXP-037 / STRONGEST_CURRENT`, а не teammate `latest.csv`.** У `EXP-037` есть полный canonical OOF на 770 616 строк, четыре временных fold, точная reconstruction из primitive components, test-regime audit и repository-linked LB `1.6496571`. У `latest.csv` test recipe воспроизводится, но canonical OOF двух из трёх верхнеуровневых компонентов отсутствует, а `1.64921756224069` только externally reported и не связан SHA→score событием.
2. **Гипотеза о недооценке micro-gains подтверждается для зрелой стадии проекта, но не для ранней.** В начале крупные изменения train construction и базовой композиции действительно дали скачки. После `S1-BEST` весь подтверждённый LB-прогресс пришёл тремя шагами `−0.000503`, `−0.000601`, `−0.000519`; ни одного скачка `>0.001` на LB в этой сильной линии больше не было.
3. **Существующий no-training compound найден.** Фиксированное, не подогнанное по LB объединение `EXP-059` и `EXP-051` даёт `1.746947164` против `1.747509863`, то есть `−0.000562699`, на 4/4 folds. Корреляция corrections `−0.0126`; interaction относительно арифметической суммы всего `−0.0000038`. Paired user-cluster SE ≈ `0.0000586`.
4. **Если отправлять один файл без новых training runs, выбран fixed compound `SEQ65 + BTYD05`, а не public-selected teammate recipe.** Точный recipe дан в `BEST_EXISTING_SUBMISSION.md`.

## 1. Evidence и метод

Факты проверялись в порядке prediction/OOF/submission artifacts → run metrics/manifests → configs/code → logs → primary experiment reports. Старые master summaries и рекомендации не использовались как источники результата.

Вычислительный аудит:

- выровнял OOF по exact `(cutoff, user_id)`;
- побитово/численно восстановил `EXP-037` из `CAP/UNC/DIST/ETX-AVG3/SEQ-AVG3`;
- пересчитал per-fold RMSLE как standard deviation log-residual после fold-optimal shift;
- посчитал prediction/residual correlations, error covariance и disagreement;
- проверил fixed blends и leave-one-fold-out двумерную сетку;
- посчитал conditional performance с одной общей fold-calibration, без segment-specific подгонки;
- сравнил OOF/test variance corrections;
- нормализовал experiment distribution, не объединяя несовместимые comparison classes.

Ограничения:

- platform export отсутствует для всех LB-связок; `verified` ниже означает repository-internal linkage;
- абсолютные CV разных команд не сравниваются напрямую из-за разных cutoff/train coverage;
- teammate `latest` нельзя честно включить в OOF blend: row-level canonical OOF `occ_meta_B` и `occ_raw_X3` не сохранился;
- offline compound — новое вычисление на старых artifacts, не public-LB optimization.

## 2. Распределение результатов

Единица счёта — 134 novelty-level units: из 138 registry rows схлопнуты exact replay, seed replay, multiseed replay и duplicate manifest. Для каждой единицы взята одна primary parent-aligned RMSLE delta; production revalidation `EXP-051` не считается второй новой победой после `EXP-047`. `neutral` задан как `|delta| < 0.0001`. 34 единицы без честной сопоставимой delta оставлены отдельной строкой.

| bucket | count | доля от 100 comparable units |
|---|---:|---:|
| gain `>0.003` | 6 | 6% |
| gain `0.001–0.003` | 7 | 7% |
| gain `0.0005–0.001` | 19 | 19% |
| gain `0.0001–0.0005` | 15 | 15% |
| neutral `|delta|<0.0001` | 29 | 29% |
| negative `delta>=+0.0001` | 24 | 24% |
| inconclusive / no comparable delta | 34 | — |

Что стоит за шестью крупными gains:

- `team_a_current:EXP-003` dense cutoffs `−0.00697` — ранняя перестройка train examples;
- `team_a_current:EXP-006` S1-BEST `−0.00993` — ранняя композиция нескольких tabular sources;
- `team_a_s2:EXP-010` `−0.00830` — single-fold hurdle внутри слабой S2-линии;
- `team_b_alt:EXP-005/008` `−0.03845/−0.03956` — single-fold scale/calibration и накопленный stack, не независимые mature-baseline breakthroughs;
- `team_b_core:EXP-015` `−0.00828` локально, но LB ухудшился примерно на `+0.009`.

Иными словами, крупные локальные числа преимущественно возникали на слабых bases, при смене калибровки или на одном/двух folds. Они не описывают экономику зрелого `EXP-037`.

### Откуда реально пришёл leaderboard progress

Repository-linked strong line после первого S1-BEST:

| submission | LB | gain к предыдущему |
|---|---:|---:|
| `submission_strategy_1.csv` | 1.651280263 | — |
| `submission_dist_head.csv` (`EXP-014`) | 1.650777411 | `−0.000502852` |
| `submission_SEQ01_mix.csv` (`EXP-025`) | 1.650176373 | `−0.000601038` |
| `submission_STRONGEST_CURRENT.csv` (`EXP-037`) | 1.649657100 | `−0.000519273` |

Суммарно после S1-BEST: `−0.001623163`, целиком тремя micro-gains. Это не арифметическое сложение локальных CV-дельт, а фактическая chronology лучших submission scores.

### Ответ на исследовательский вопрос №1

**Стратегической ошибкой была не первоначальная ставка на крупные изменения, а сохранение крупного acceptance gate после насыщения сильной линии.** В ранней фазе dense cutoffs и базовый ensemble оправданно искали тысячные. После `EXP-014/025` data показывают другой режим: устойчивые `4/4` улучшения `0.0002–0.0009` стали нормой, а проекты часто помечали их `rejected_below_gate` вместо систематического compatibility audit.

## 3. Настоящий сильнейший baseline

| pipeline | honest local evidence | LB evidence | coverage / composition | verdict |
|---|---|---|---|---|
| `team_a_current:EXP-037` | wCV `1.747509863`; folds `1.766883/1.760510/1.748629/1.741279`; exact 770,616-row OOF | `1.6496571`, strong repository link | 4 temporal folds; `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 ETX + 0.225 SEQ`; exact test rebuild | **research/private-safe baseline** |
| teammate `latest.csv` / `EXP-067` | canonical OOF missing for `occ_meta_B`, `occ_raw_X3`; exact latest wCV unknown | externally reported `1.649217562`; no SHA→score event | test recipe `.12 friend + .16 occ_meta_B + .72 occ_raw_X3`; exact test reconstruction | best public claim, **not a baseline** |
| teammate final6h/extra90 candidates | reported deltas `−0.001651…−0.001821` vs teammate base `1.749804`, only 3 recent wins; fold 1 delta is 0 | no manifest LB events | many Ridge/meta/occurrence searches; corr to friend `0.99963–0.99972` | exploration source; repeated-selection risk |
| Team B alternate `EXP-017` | two-fold mean `1.708295`, protocol/cutoffs incompatible | `1.6546318191` report-linked only | recency + post-order DIST, scale 1.2 | independent but weaker line |
| Team B core `EXP-015/029` | best local calibration `1.708737`; hurdle `1.716161`; protocol incompatible | calibration LB ≈`1.670`; hurdle `1.6584166` report-only | 2 decision folds + diagnostic fold | not competitive/private-safe |
| Strategy 2 `EXP-012` | mean `1.76831`; S1 blend gain ≈`0.0001` rejected | `1.66193246` | structural count×value/hurdle | useful diversity lesson, not baseline |

Почему `EXP-037` остаётся baseline:

- полный aligned OOF и targets;
- все primitive test components и exact recipe;
- 4-fold robustness и fixed fold weights;
- test/OOF regime checks;
- лучший подтверждённый pipeline без missing canonical state;
- teammate public leader имеет более низкий заявленный LB, но validation и provenance недостаточны для private-safe research.

## 4. OOF и diversity audit

Standalone score не равен ensemble value:

| source | wCV | комментарий |
|---|---:|---|
| CAP | 1.760642 | слабее, но выполняет extrapolation insurance role |
| UNC | 1.751507 | tabular core |
| DIST | 1.750616 | distribution head |
| ETX-AVG3 | 1.748610 | strongest standalone member |
| SEQ-AVG3 | 1.749634 | neural sequence diversity |
| BTYD | 1.800700 | очень слаб standalone |
| `STRONGEST_CURRENT` | 1.747510 | exact baseline |
| `BTYD05` | 1.747189 | слабый BTYD полезен только как 5% residual source |

Ключевые diversity numbers:

- `corr(STRONGEST, BTYD)=0.95741`, `Var(BTYD−STRONGEST)=0.22545`;
- после 5% веса `corr(STRONGEST, BTYD05)=0.999882`, но этого достаточно для `−0.000321`;
- `SEQ65` близок к baseline (`corr=0.999849`, `Var(delta)=0.0007425`), однако улучшает 4/4;
- corr corrections `SEQ65−STRONGEST` и `BTYD05−STRONGEST` = **`−0.01263`**, то есть источники почти ортогональны как directions;
- residual alignment corrections: `0.01418` для SEQ65 и `0.02591` для BTYD05.

Это прямое опровержение правила «standalone хуже → бесполезен»: BTYD хуже baseline на `0.0532` wCV, но его 5%-direction — второй лучший готовый micro-gain.

## 5. Ensemble audit и новое offline исследование

Исторически:

- `EXP-014`: 1771 weight combinations с шагом 0.05; OOF optimum выбрасывал CAP, но production recipe сохранял 0.10 CAP как shift insurance;
- `EXP-025`: LOFO с фиксированным CAP дал `−0.00106`, 4/4; свободный optimum опять выбрасывал страховку;
- `EXP-035/036/037`: последовательные slot replacement/coauthor tests создали champion;
- `EXP-047/051`: nested weights `.05/.05/.10/.10`, production fixed `.05`; fixed curve оптимальна примерно `.05–.10`;
- `EXP-059`: один заранее заданный shift sequence-slot `0.45→0.65`; соседние веса не подбирались;
- teammate line делала широкий meta/weight search на неполном common OOF и затем public-selected assembly.

### Fixed compound: новый факт

Без weight search и без LB:

```text
SEQ65 = 0.10 CAP + 0.10 UNC + 0.15 DIST + 0.325 ETX-AVG3 + 0.325 SEQ-AVG3
COMPOUND = 0.95 * SEQ65 + 0.05 * BTYD
```

| model | 09-04 | 09-18 | 10-02 | 10-16 | wCV | delta |
|---|---:|---:|---:|---:|---:|---:|
| STRONGEST | 1.766883 | 1.760510 | 1.748629 | 1.741279 | 1.747510 | — |
| SEQ65 | 1.766538 | 1.760279 | 1.748357 | 1.741069 | 1.747272 | `−0.000238` |
| BTYD05 | 1.766229 | 1.759939 | 1.748387 | 1.741022 | 1.747189 | `−0.000321` |
| **fixed compound** | **1.765895** | **1.759715** | **1.748119** | **1.740801** | **1.746947** | **`−0.000563`** |

Arithmetic standalone sum `−0.0005589`; observed `−0.0005627`; interaction `−0.0000038`. Test/OOF variance ratio общей correction = `1.2167`, внутри ранее использовавшегося production gate `[0.6,1.4]`.

Leave-one-fold-out 2D diagnostic grid (`sequence weight 0.45/0.55/0.65/0.75`, BTYD `0/.025/.05/.075/.10`) даёт nested `−0.000566`. Первые три outer folds выбирают `.65/.075`, последний `.75/.10`. Финальный recommendation **не использует этот optimum**: фиксируются уже существующие `.65` и `.05`, чтобы исключить OOF weight overfit.

## 6. Compatibility audit

Полная 48×48 matrix находится в `COMPATIBILITY_MATRIX.csv`. Главные отношения:

Коды matrix: `T+` — historically tested positive; `A+` — near-additive в новом fixed OOF audit; `A-` — positive total с antagonistic interaction; `A?` — потенциально additive, не проверено; `N` — nested/absorbed; `R` — redundant; `X` — mutually exclusive slot; `C` — correlated/uncertain; `U` — incompatible protocol/unproven pairability.

- **absorbed/nested:** `EXP-003`, `005`, `006`, `014`, `025`, `026`, `027`, `035`, `036` уже вошли в lineage `EXP-037`; складывать их deltas нельзя;
- **falsified transfer:** `EXP-017/018` дали seed/round gains на E10, но `EXP-046` factorial на production UNC/CAP свёл перенос к `−0.0000526` (seed only) и `−0.0000024` (primary combined); эта microgain уже проверена и практически поглощена;
- **genuinely additive:** `EXP-059` и `EXP-051` — разные sources, correction corr `−0.0126`, fixed combo 4/4;
- **positive but antagonistic:** `EXP-040 FRESH` + `EXP-047/051 BTYD`; pair улучшает, но interaction около `+0.00008`, поэтому нельзя суммировать;
- **nested compound:** `EXP-049` уже является BTYD+FRESH validation combination, а не третьим независимым сигналом;
- **unproven cross-pipeline:** Team B/S2/teammate gains нельзя объявить additive без common row-level OOF.

## 7. Ablation reconstruction

### Champion lineage

- `EXP-014` заменил часть tabular mixture на DIST: ensemble `−0.00071`, 4/4.
- `EXP-025` добавил TCN: honest LOFO `−0.00106`, 4/4; repository-linked LB `−0.000601`.
- `EXP-035` улучшил sequence slot на `−0.000575`, 4/4.
- `EXP-036` показал ETX как coauthor: `−0.000823`, 4/4, хотя standalone replacement отвергнут.
- `EXP-037` собрал equal ETX/TCN halves в slot: `−0.000833`, 4/4.

### Уже выполненный factorial tabular debt (`EXP-046`)

| contrast inside champion | delta | folds |
|---|---:|---:|
| AVG3 only @600 | `−0.00005256` | 4/4 |
| rounds 300 only | `+0.00003314` | 1/4 |
| AVG3@300 primary | `−0.00000245` | 3/4 |

Это закрывает соблазн механически перенести `EXP-017/018` в champion.

### Новый add-one/remove-one вокруг champion

| recipe | delta | folds | paired cluster SE |
|---|---:|---:|---:|
| add SEQ65 reweight | `−0.000238` | 4/4 | `0.000045` |
| add BTYD05 | `−0.000321` | 4/4 | `0.000042` |
| add both | `−0.000563` | 4/4 | `0.000059` |
| add FRESH only | `−0.000225` | 4/4 | `0.000037` |
| BTYD05 + FRESH | `−0.000467` | 4/4 | `0.000063` |
| SEQ65 + BTYD05 + FRESH | `−0.000721` | 4/4 | `0.000074` |

Последняя строка пока OOF-only: exact production FRESH отсутствует.

## 8. Error analysis

Conditional score использует общую fold calibration, не подгоняется внутри сегмента.

Для fixed compound:

| segment | delta vs STRONGEST |
|---|---:|
| actual zero | `−0.002211` |
| positive low quartile | `+0.002900` |
| positive middle 50% | `+0.002187` |
| positive 75–95% | `−0.001789` |
| positive top 5% | `−0.007108` |
| rec_buy 15–60 | `−0.000802` |
| w180 buy-days 1–3 | `−0.000681` |
| w180 buy-days 4–15 | `−0.000620` |
| sparse history bottom quartile | `−0.000452` |
| dense history top quartile | `−0.000511` |

Интерпретация: общий gain приходит из будущих нулей, high/extreme GMV и recency 15–60, ценой low/mid positive buyers. Это подтверждает mechanism BTYD/temporal balance, но создаёт private-LB risk при сдвиге target mixture. Нельзя строить target-segment gate по истинному `y`; `EXP-053` показал, что cutoff-safe gate почти не предсказывает нужное направление (`AUC 0.5268`, bounded gain `−0.0000064`).

## 9. Failed experiments: что именно отвергнуто

| class | examples | conclusion |
|---|---|---|
| fundamentally falsified in tested mechanism | `EXP-039 BLOCK4`, `EXP-042 ZERO2D`, `EXP-061/062/064` | correction direction или схлопнулась в zero, или ухудшила; не тюнить дальше |
| implementation failure, hypothesis survived | `EXP-050` unstable BG/NBD optimizer | исправлено analytic-gradient solver в `EXP-051`; production PASS |
| real but small standalone/residual signal | `EXP-040 FRESH`, `EXP-047/051 BTYD`, `EXP-059 SEQ65` | кандидаты для compound, не для standalone champion replacement |
| ensemble-only value | BTYD standalone `1.800700` → BTYD05 `1.747189` | слабая standalone модель полезна за residual diversity |
| inconclusive / design problem | `EXP-048` mixed 3/4-fold estimands, `EXP-032/032B` half-panel, `EXP-038` single fold | не переносить delta без corrected full protocol |
| provenance/confounding problem | teammate latest / `EXP-067` | exact test recipe есть, canonical OOF и SHA→LB link нет |

## 10. Public-LB overfitting

- Все LB values — repository-internal или externally reported, platform export отсутствует.
- `EXP-016` локально `−0.00038` 4/4, но LB ухудшился на `+0.00023`: correlated weight search и удаление CAP были плохим transfer.
- Team B core fixed shift локально `−0.00828`, LB примерно `+0.009` хуже.
- teammate line провела много Ridge/meta/occurrence selections; её lower public claim нельзя использовать для выбора весов или baseline.
- Новый fixed compound использует только два заранее существовавших recipes; public scores не участвовали в выборе `.65/.05`.

## 11. Research saturation

| family | evidence | saturation verdict |
|---|---|---|
| residual/correction models | 6 units; последние `EXP-053/054/055/061/064` дали `−0.000006…0` | **saturated** для generic residual maps/preflights |
| calibration/postprocessing | 13 units; ZERO2D controls и late detrend ≈0; крупный Core shift не перенёсся | **saturated/high transfer risk** |
| tabular refresh | 14 units; `EXP-046` exact factorial поглотил seed/round gain | **saturated** для rounds/seed on existing slots |
| occurrence/BTYD | classic BG/NBD небольшой, но 4/4; production готов | family tuning saturated, **existing residual direction usable** |
| neural sequence/event | 8 novelty units; крупные ранние gains, поздние architecture variants mixed | architecture tuning near-saturated; composition reweight ещё даёт micro-gain |
| ensembles/component selection | 21 units; late `EXP-040/059` still `−0.000225/−0.000238` | не saturated именно по missing compatible combinations |
| target decomposition | S2 слаб; hurdle/calibration variants mixed; FNL one-fold | saturated для очередного hurdle/count tweak без нового evidence |
| teammate occurrence stack | apparent recent-fold gains, missing canonical OOF | not falsified, but **provenance-limited**, not ready for optimization |

## 12. Opportunity gaps и rational strategy balance

### Strategy A — exploitation (примерно 70–80% ближайшего бюджета)

1. Материализовать fixed `SEQ65+BTYD05` compound.
2. Если independent rebuild проходит, productionize exact FRESH и проверить только один frozen triple recipe.

### Strategy B — exploration (примерно 20–30%)

Восстановить common canonical OOF teammate occurrence/table components и провести один locked nested comparison against `EXP-037`; не продолжать широкий meta-search до прохождения этого gate.

Баланс смещён в exploitation: доступный compound уже даёт statistically clear local gain того же порядка, из которого исторически складывался LB progress. Новый breakthrough сейчас имеет меньший Expected Value, чем корректная сборка накопленного evidence.

## 13. Ответы A–D

### A. Была ли ошибка в ставке на крупные отдельные improvements?

**Да, после насыщения сильной линии; нет, в самом начале.** Ранние structural changes дали нужные скачки. После S1-BEST реальный LB прогресс целиком состоит из трёх gains примерно по `0.0005–0.0006`, а registry содержит 34 measured gains в диапазоне `0.0001–0.001` против только 13 gains `>=0.001`.

### B. Какой baseline использовать сейчас?

**`team_a_current:EXP-037 / submission_STRONGEST_CURRENT.csv`**, wCV `1.747509863`, SHA256 `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.

### C. Есть ли no-training комбинация с высокой вероятностью лучше current submission?

**Да: fixed `0.95*SEQ65 + 0.05*BTYD`.** OOF `−0.000562699`, 4/4, correction corr `−0.0126`, paired SE `0.0000586`, test/OOF correction variance ratio `1.2167`.

### D. Какой один submission отправить сейчас?

**Fixed compound `SEQ65 + BTYD05`**, собранный из primitive test arrays, с одним global shift до level `2.3293`, затем `z=max(z,0)` и `predict=expm1(z)`. Не смешивать уже округлённые/shifted CSV в raw space. Полный exact recipe — в `BEST_EXISTING_SUBMISSION.md`.
