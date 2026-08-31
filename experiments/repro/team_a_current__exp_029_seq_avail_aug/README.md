# exp_029 — SEQ-AVAIL-AUG: train-time augmentation канала `avail`

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_029_seq_avail_aug`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_029_seq_avail_aug`
- **Original source:** `experiments/exp_029_seq_avail_aug.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** dilated TCN, sequence model, ensemble
- **Features:** calendar features, history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Validation/test не аугментируются. Проверены две точки:
- **Known score:** ключевой сегмент `rec_buy 15–60` на +0.00309 RMSLE / −0.00263 AUC; `A25`
- **Seed:** python -m src.seq fold --val 2025-10-16 --epochs 4 --seed 42 --exp SEQ-03A-A25-S42 --aug avail_drop --aug-p 0.25
- **Postprocessing:** дают 1.43× (`B`) и 1.34× (`A25`) этого уровня. Но изменение вредное. `B` портит
- **Submission:** single-fold diagnostics уже отрицательны. Ни один из двух последних сабмитов
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_029 — SEQ-AVAIL-AUG: train-time augmentation канала `avail`

- **Дата:** 2026-08-14
- **Автор:** A1
- **Коммит:** 560b24b + рабочее дерево
- **Код:** `src/seq.py`, `src/test_seq.py`; отчёт — `research/strategies/results/SEQ4/`
- **Вычисления:** только fold `2025-10-16`, seed 42; полный 4-fold/3-seed прогон не запускался

## Гипотеза

`exp_027` отделил полезность дополнительных дней истории от OOD канала `avail`:
при полной глубине test получает `avail ≡ 1`, хотя ни один train cutoff такого
состояния не видел. Проверялось, можно ли ввести этот край в train простой
аугментацией одного канала, не меняя архитектуру, target, supervision и ensemble.

## Что изменено относительно базы

Train-only `sample_avail_shift` переводит последние позиции ведущего префикса
`avail=0` в `avail=1`; 14 behavioral channels и target остаются побитово прежними.
Validation/test не аугментируются. Проверены две точки:

- `B`: `avail_bnd(p=0.5, full=0.5)` — 25% примеров получают `avail ≡ 1`, ещё
  примерно 25% — случайную промежуточную границу;
- `A25`: `avail_drop(p=0.25)` — те же 25% крайних `avail ≡ 1`, без jitter
  промежуточных границ.

BASE использует `aug=none`. Контроль переобучен в том же локальном eager-режиме:
его 1.74808 отличается от исторического seed-42 1.74704 на +0.00104, поэтому все
дельты считаются только к свежему BASE. Это укладывается в известную высокую
нестабильность TCN (`exp_025`/`exp_026`).

## Корректность и анти-лукап

`python -m pytest src/test_seq.py -q` — **63 passed**. Отдельно проверено:

- меняется только колонка `avail`; behavioral/calendar channels и target те же;
- открываемый нулевой префикс поведения остаётся строго нулевым;
- граница остаётся монотонной, диапазон включает `avail ≡ 1`;
- augmentation живёт только в `Batcher`, inference детерминирован;
- поток случайности augmentation отделён от shuffle BASE;
- ни один день после cutoff не попадает во вход.

## Быстрый gate: fold 2025-10-16, seed 42

| вариант | RMSLE cal | Δ | AUC(y>0) | ΔAUC | Var(z−BASE) | corr остатков |
|---|---:|---:|---:|---:|---:|---:|
| BASE | **1.74808** | — | **0.84248** | — | — | — |
| `B` | 1.74986 | **+0.00178** | 0.84123 | **−0.00125** | 0.03236 | 0.99471 |
| `A25` | 1.74913 | **+0.00105** | 0.84190 | **−0.00057** | 0.03044 | 0.99502 |

Изменение не seed noise: `Var(z₄₂−z₄₃)=0.02270` на том же фолде, а варианты
дают 1.43× (`B`) и 1.34× (`A25`) этого уровня. Но изменение вредное. `B` портит
ключевой сегмент `rec_buy 15–60` на +0.00309 RMSLE / −0.00263 AUC; `A25`
мягче (+0.00079 / −0.00062), но общий STOP-порог +0.001 всё равно пересечён.

## `availprobe` и `availcurve`

| вариант | ΔRMSLE при `avail≡1` | Var(Δz) | corr | диапазон `availcurve` |
|---|---:|---:|---:|---:|
| BASE | **+0.00651** | 0.02307 | 0.99517 | 0 … **+0.00651** |
| `B` | **−0.00020** | 0.00060 | 0.99987 | −0.00022 … +0.00017 |
| `A25` | **−0.00001** | 0.00167 | 0.99965 | −0.00001 … +0.00014 |

Representation shift действительно устранён: край перестал быть аномальным,
а вся кривая стала плоской. Это не сомнительный нулевой эффект — зависимость
BASE воспроизводится свежим контролем и падает на 93–97% по `Var(Δz)`.

## Cross-depth +77: польза реальных дней отдельно от границы

На тех же checkpoint'ах вход обрезался до 212 дней и затем раскрывался до 289:
это **+77 реальных дней** без ухода в `avail ≡ 1`. Исчезновение границы измерено
отдельно выше при неизменных данных.

| вариант | cal@212 | cal@254 | cal@289 | gain +77 | optimum | full−optimum |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 1.75116 | 1.74736 | 1.74808 | **−0.00308** | 275 | +0.00087 |
| `B` | 1.75151 | 1.74835 | 1.74986 | **−0.00165** | 261 | +0.00154 |
| `A25` | 1.75119 | 1.74770 | 1.74913 | **−0.00206** | 254 | +0.00143 |

Дополнительные дни формально остаются полезны относительно 212, но gain
сократился на **47%** у `B` и **33%** у `A25`. Optimum отступил с 275 до 261/254,
то есть augmentation купила инвариантность именно частичным отказом от длинной
истории. После максимальной train-глубины 254 дополнительные 35 дней вредят
сильнее BASE: +0.00151 / +0.00143 против +0.00072.

## Gate и вердикт

### `REJECT`

Условия CONTINUE должны выполняться одновременно. Выполнены только OOD-часть и
критерий «изменение больше seed noise». Оба train-only варианта ухудшают последний
fold и AUC; `B` существенно, `A25` на самом STOP-пороге. Польза длинной истории
не возвращена, а ослаблена; это прямо совпадает с зарегистрированным STOP-сценарием
«augmentation заставляет модель игнорировать длинную историю».

- **Устранена ли проблема `avail ≡ 1`?** Да, технически полностью.
- **Удалось ли вернуть пользу более длинной истории?** Нет: gain +77 уменьшился.
- **Нужен ли полный 4-fold/3-seed?** Нет; gate на 10-16 не пройден.
- **Нужны ли LOFO/SEQAVG3 и LB-submit?** Нет: без полного OOF они запрещены, а
  single-fold diagnostics уже отрицательны. Ни один из двух последних сабмитов
  на этот вариант тратить нельзя.

Политика production остаётся прежней: **SEQ на test только с `--depth-clip 289`**.

## Воспроизведение

```text
python -m pytest src/test_seq.py -q
python -m src.seq fold --val 2025-10-16 --epochs 4 --seed 42 --exp SEQ-03A-BASE-S42 --aug none
python -m src.seq fold --val 2025-10-16 --epochs 4 --seed 42 --exp SEQ-03A-B-S42 --aug avail_bnd --aug-p 0.5 --aug-full 0.5
python -m src.seq fold --val 2025-10-16 --epochs 4 --seed 42 --exp SEQ-03A-A25-S42 --aug avail_drop --aug-p 0.25
python -m src.seq availprobe --ckpt <EXP>-V1016
python -m src.seq availcurve --ckpt <EXP>-V1016 --shifts 0 13 26 38 51 64 76
python -m src.seq depth --ckpt <EXP>-V1016 --depths 212 230 247 254 261 275 289
```

Первый технический запуск A25 был оборван оболочкой после одной эпохи и не
оставил checkpoint/OOF; приведённые числа относятся только к чистому полному
перезапуску. Успешные времена: BASE ~78 мин, `B` ~84 мин, `A25` ~83 мин.
