# exp_040 — FRESH-CONTRAST-MOE: incremental COND-FRESH residual

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_040_fresh_contrast_moe`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_040_fresh_contrast_moe`
- **Original source:** `experiments/exp_040_fresh_contrast_moe.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** dilated TCN, sequence model, ensemble
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** исходная сторона `EXTRA-B -> validation-A`: 887 996 positive строк,
- **Known score:** | **wCV** | **1.747510** | **1.747285** | **−0.000225** | **4/4** |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** 4/4, поэтому это не только сдвиг уровня (уровень обнулён preprocessing и fold
- **Submission:** 11. Submission: **нет**.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_040 — FRESH-CONTRAST-MOE: incremental COND-FRESH residual

- **Дата:** 2026-08-21
- **Автор:** A1
- **Коммит:** рабочее дерево поверх `a28a71f`
- **Код:** `src/fresh_contrast.py`, `src/test_fresh_contrast.py`
- **Результаты:** `research/strategies/results/FRESH_CONTRAST/`
- **Запуск:** `python src/fresh_contrast.py`

## Гипотеза

`exp_032/032b` доказал, что EXTRA-supervision улучшает conditional intensity,
но абсолютный two-part член теряет diversity в смеси. Здесь от него оставлена
только разность при общем боевом extensive `P=1-p0` из `S1-DIST`:

```text
d_fresh = z_COND_FRESH - z_COND_CLEAN
z_new   = z_STRONGEST_CURRENT + alpha * processed(d_fresh)
```

Проверялись только `GLOBAL` и заранее заданный `HIGH16`, только alpha
`0/0.25/0.5/0.75/1.0`. `d_vol` — симметричный контроль объёма.

## Full-panel cross-fit и воспроизведение базы

Получен полный двухсторонний OOF на 770 616 строках четырёх folds:

- исходная сторона `EXTRA-B -> validation-A`: 887 996 positive строк,
  97 648 уникальных доноров; predictions `EXP-032b` переиспользованы;
- достроенная сторона `EXTRA-A -> validation-B`: 884 573 positive строк,
  97 317 уникальных доноров;
- recipient A/B по folds: `94 238/94 280`, `95 497/95 528`,
  `96 810/96 884`, `98 707/98 672`;
- пересечение EXTRA donors с recipients своей головы — 0 на обеих сторонах;
- TCN/ETX не обучались: выполнены только inference четырёх сохранённых frozen
  fold-энкодеров и две недостающие головы `VOL/FRESH` на fold;
- `d_fresh=fresh-clean`, `d_vol=vol-clean` построчно; разность с
  `P_DIST*(mu_variant-mu_clean)` не выше `1.6e-15`;
- `z_STRONGEST_CURRENT` побитово совпал с `oof_BLOCK4_SAF.npz`; baseline:
  **1.747509862**, fold scores `1.766883/1.760510/1.748629/1.741279`.

Штатная база не менялась:

```text
0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 ETX-AVG3 + 0.225 SEQ-AVG3
```

## Настоящий nested LOFO: REJECT

Для каждого outer held-out fold variant/alpha выбирались на остальных трёх.
При выборе каждый из трёх training folds winsorized границами только двух других
training folds; outer fold не участвовал ни в labels, ни в границах. На held-out
применялись границы всех трёх training folds. После clipping 0.5/99.5% correction
центрировалась; у HIGH16 gate применялся до центрирования.

| fold | STRONGEST | FRESH candidate | delta | variant/alpha без fold |
|---|---:|---:|---:|---|
| 2025-09-04 | 1.766883 | 1.766640 | **−0.000243** | GLOBAL / 1.00 |
| 2025-09-18 | 1.760510 | 1.760346 | **−0.000164** | GLOBAL / 1.00 |
| 2025-10-02 | 1.748629 | 1.748435 | **−0.000194** | GLOBAL / 1.00 |
| 2025-10-16 | 1.741279 | 1.741025 | **−0.000253** | GLOBAL / 1.00 |
| **wCV** | **1.747510** | **1.747285** | **−0.000225** | **4/4** |

Знак устойчив и последний fold лучше, но величина в 2.2 раза слабее минимального
порога `−0.0005` и в 4.4 раза слабее STRONG ACCEPT `−0.0010`.

### Заранее заданные fixed curves

| alpha | GLOBAL delta wCV | folds | HIGH16 delta wCV | folds |
|---:|---:|---:|---:|---:|
| 0.00 | +0.000000 | 0/4 | +0.000000 | 0/4 |
| 0.25 | −0.000096 | 4/4 | −0.000022 | 2/4 |
| 0.50 | −0.000166 | 4/4 | **−0.000027** | 2/4 |
| 0.75 | −0.000209 | 4/4 | −0.000016 | 2/4 |
| 1.00 | **−0.000225** | 4/4 | +0.000012 | 1/4 |

`GLOBAL` монотонно до края разрешённой сетки, но меньший шаг alpha запрещён и
не нужен: даже наблюдаемый максимум далеко от gate. `HIGH16` около нуля и
неустойчив (лучший fixed alpha 0.5, только 2/4), поэтому specialist-ветка закрыта.

## VOL-control и causal contrast

Тот же nested LOFO для `d_vol` выбрал `HIGH16 0.75 / HIGH16 1.0 / GLOBAL 1.0 /
GLOBAL 1.0` и дал:

```text
VOL nested LOFO       +0.000008
FRESH nested LOFO     -0.000225
FRESH minus VOL       -0.000233
```

VOL не объясняет слабое улучшение, но FRESH не набирает требуемого causal
отрыва `−0.0004` для STRONG ACCEPT. Вывод держится прежде всего на абсолютном
нулевом размере residual gain, а не на провале control.

## Hash halves и diagnostics

При выбранных full-panel nested candidates и отдельной fold calibration:

| группа | delta wCV |
|---|---:|
| A (`head EXTRA-B`) | **−0.000249** |
| B (`head EXTRA-A`) | **−0.000201** |

Знак одинаков на обеих половинах; one-sided artifact `exp_032` исключён.

- `Var(d_fresh)` по folds: `0.000951 / 0.000537 / 0.000614 / 0.001305`;
  `Var(d_vol)`: `0.000092 / 0.000088 / 0.000076 / 0.000059`.
- `corr(d_fresh, ly-z_STRONGEST)` = `0.0115 / 0.0134 / 0.0147 / 0.0157`;
  после fold-safe winsor `0.0167 / 0.0137 / 0.0149 / 0.0171`.
- `corr(residual_before,residual_after)` = `0.99985..0.99991`.
- AUC(y>0) wCV-average: `0.843543 -> 0.843674`, delta **+0.000131**.
- positive-only RMSE: `1.674533 -> 1.677213`, то есть **+0.002680** хуже.
- processed correction: mean numerical zero, pooled std `0.02591`,
  `p01/p05/p50/p95/p99 = -0.05799/-0.04052/-0.00282/0.04735/0.07759`;
  weighted clipped fraction 3.23%.

Freshness contrast имеет слабую положительную residual correlation и улучшает
4/4, поэтому это не только сдвиг уровня (уровень обнулён preprocessing и fold
calibration). Но он не исправляет исходно мотивировавшую magnitude-ошибку:
positive-only RMSE ухудшается, а небольшой выигрыш идёт через форму/ranking.

### Где лежит наблюдаемый gain

| сегмент | delta RMSLE |
|---|---:|
| `rec_buy 15-60` | **−0.000327** |
| `w180_days_buy 2-15` | −0.000277 |
| `w180_days_buy 0-1` | −0.000223 |
| never purchased | −0.000167 |
| `w180_days_buy >=16` | **−0.000046** |

Именно HIGH16, где standalone contrast был крупнейшим, поверх
`STRONGEST_CURRENT` почти исчерпан. Residual gain лежит в средней частоте и
проблемном `rec_buy 15-60`, но его величина всё равно ниже разрешения проекта;
новые segment weights постановкой запрещены и не проверялись.

## Test regime и submission

По зарегистрированному протоколу production выполняется только если validation
не REJECT. Здесь `nested LOFO > −0.0005`, поэтому test heads не считались,
test/OOF regime audit неприменим и
`submissions/submission_FRESH_CONTRAST_MOE.csv` **не создан**. Это не блокер, а
обязательная остановка по gate.

## Вердикт и прямые ответы

**REJECT.** Свежая conditional supervision реальна как standalone signal, но
её incremental часть поверх `STRONGEST_CURRENT` уже практически содержится в
чемпионе. `COND-FRESH` как ensemble residual закрыт; автоматического перехода к
neural fine-tune нет.

1. Полный two-sided cross-fit: **да**, 770 616/770 616 OOF строк.
2. FRESH nested LOFO: **−0.000225**.
3. VOL-control: **+0.000008**; causal contrast **−0.000233**.
4. Без held-out везде выбран **GLOBAL, alpha=1.0**; HIGH16 закрыт.
5. Улучшено **4/4** folds.
6. 2025-10-16: **−0.000253**.
7. A/B: одинаковый минус, **−0.000249/−0.000201**.
8. Residual correlation есть, но мала: **0.0115..0.0157** raw.
9. Gain — `rec_buy 15-60`/`w180 2-15`; HIGH16 почти ноль.
10. Test/OOF regime: не запускался после validation REJECT.
11. Submission: **нет**.
12. Реального LB submit не достоин: ожидаемая дельта ниже project floor.
13. Финальный статус: **REJECT**.

## Тесты и воспроизведение

```text
python -m pytest src/test_fresh_contrast.py src/test_seq_cond.py \
  src/test_pipeline.py src/test_validation.py -q
# 60 passed

python src/fresh_contrast.py
```

Основные artifacts: `artifacts/oof_FRESH_CONTRAST_MOE.npz`, четыре
`FRESH_CONTRAST_MOE_fold_*.npz`, четыре mirror-head artifacts; таблицы
`fixed_summary.csv`, `nested_lofo.csv`, `fold_diagnostics.csv`, `segments.csv`,
`hash_groups.csv`. Повторный runner использует resume и заканчивается за ~10 с.
