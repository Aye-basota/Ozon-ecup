# EXP084 — One-Shot Global Level Probe

Статус: **PRE-LB / probe создан, но не отправлен**. Это один математический
probe общей константы в log-пространстве; модели не обучались, scale search не
проводился, других значений `c` не создавалось и не проверялось.

## Frozen setup

```text
R0 = 1.646143314225527
c  = +0.0200000000
```

Anchor:

```text
C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP075_JOINT_A1_365_A2.csv
SHA256 d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901
```

Путь является актуальным: файл byte-identical независимой копии
`SUBMIT_EXP075_JOINT_A1_365_A2_CONFIRMED.csv`; тот же SHA зафиксирован в
`EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS/checksums_final.sha256` и в post-mortem
EXP076. Никакой другой anchor не подставлялся.

Единственное преобразование:

```python
z_anchor = np.log1p(predict_anchor)
z_probe = np.maximum(z_anchor + 0.02, 0.0)
predict_probe = np.expm1(z_probe)
```

## Formula frozen before LB

После ручного получения `R1` разрешена только подстановка в следующие заранее
зафиксированные формулы:

```text
Delta_RMSLE = R1 - R0
Delta_MSE   = R1^2 - R0^2

r_bar = mean_public(target_log - z_anchor)
      = (c^2 - Delta_MSE) / (2c)

c_public_opt = r_bar
gain_MSE_public_opt = r_bar^2
gain_RMSLE_public_opt ≈ r_bar^2 / (2 * R0)
```

Для uncertainty после LB:

```text
SE_public_sampling = 0.007018778644
z_score = r_bar / SE_public_sampling
95% sampling CI for the corresponding full-population mean
    ≈ r_bar ± 0.013756553358
```

Это декодирование одной scalar-оси, не разрешение на создание поправки
`c_public_opt`, второй probe или подбор scale.

## Clipping audit

Проверено на полном anchor до создания probe:

| Проверка | Значение |
| --- | ---: |
| `count(z_anchor == 0)` | 1,219 |
| `count(z_anchor + 0.02 <= 0)` | 0 |
| `max(abs(z_clipped - z_affine))` | 0.0 |
| `RMS(z_clipped - z_affine)` | 0.0 |
| точная поправка к affine `Delta_MSE`-формуле | 0.0 |

Следовательно, clipping не активируется ни на одной строке. Для этого файла
`z_probe = z_anchor + c` точно, и формула
`Delta_MSE = c^2 - 2*c*r_bar` не требует clipping-correction.

Максимальная ошибка после десятичной сериализации и повторного чтения CSV
составляет `4.440892098500626e-16` в `z`; это машинное округление, а не clipping.

## Sampling uncertainty before LB

Использован наиболее близкий доступный clean historical proxy: четыре
forward-OOF фолда (`2025-09-04` ... `2025-10-16`), composition-matched
production baseline для `SUBMIT_ORTH_ALPHA`, frozen joint EXP075 correction и
production-style clipping `z >= 0`. Для level-направления использован ровно
`u=1`. В каждом фолде correction был спроецирован вне константы, поэтому его
среднее до clipping численно равно нулю (`|mean| <= 1.27e-17`).

С seed `84002026` выполнено 2,000 exact simple-random-without-replacement
разбиений по 20% на каждом фолде, всего 8,000 splits. Исторический public имел
37,704–39,476 строк на фолд.

| Величина | Оценка |
| --- | ---: |
| reference residual SD | 1.754691151641 |
| SD(`r_public - r_full`) на historical 20% splits | 0.007949866810 |
| empirical 95% interval ошибки на historical splits | [-0.015172672736, +0.015727434802] |
| SD, нормированная к 50,000 из 250,000 | **0.007018778644** |
| 95% half-width для public ≈ 50k | **0.013756553358** |

На исторических proxy-фолдах смен знака не было: `0/8000` между public и full
и `0/8000` между public и private; односторонняя 95% upper bound для этой
условной частоты — примерно `0.000375`. Это обусловлено тем, что минимальный
исторический `|full mean residual|` был `0.03617`, то есть далеко от нуля, и не
является безусловной гарантией для текущего TEST.

Для ориентира при том же sampling SD условная вероятность смены знака public
относительно full/private составляет приблизительно `7.71%`, если истинный
`|r_bar| = 0.01`; `0.219%` при `0.02`; `0.000959%` при `0.03`. Поэтому значение
вблизи нуля надо читать вместе с CI, а `|r_bar| >= 0.03` устойчиво к одной лишь
случайности public-subset.

## Probe artifact

```text
C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP084_LEVEL_PROBE_P020.csv
SHA256 747544497c40fea8687ca2f36a494ba4b2413fedc0f85c204ed5e4e48c8db091
```

Full-file checks:

| Проверка | Значение |
| --- | ---: |
| rows | 250,000 |
| columns | `user_id,predict` |
| unique `user_id` | PASS |
| same row order as anchor | PASS |
| finite | PASS |
| `predict >= 0` | PASS |
| min `predict` | 0.0202013400267558 |
| max `predict` | 8966.5864076514 |
| zero count | 0 |
| mean `log1p(predict)` | 2.3499078326300307 |
| population std `log1p(predict)` | 1.6231233786527937 |
| `RMS(z_probe - z_anchor)` | 0.019999999999999928 |

Probe не отправлялся автоматически.

## Frozen interpretation rule

После получения `R1` интерпретация будет ровно такой:

| `|r_bar|` | Интерпретация |
| ---: | --- |
| `< 0.01` | global-level axis практически закрыта |
| `0.01 <= |r_bar| < 0.03` | небольшой signal |
| `0.03 <= |r_bar| < 0.05` | значимый level signal |
| `>= 0.05` | крупная находка, способная объяснить существенную часть LB gap |

Даже при сильном public residual автоматически создавать correction
`c_public_opt` нельзя. Сначала отдельно оцениваются перенос public → private и
private-safe shrinkage. После декодирования одного `R1` эксперимент
останавливается.
