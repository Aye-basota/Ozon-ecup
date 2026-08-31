# EXP079 — EXP075 A040

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp079_exp075_a040`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP079_EXP075_A040`
- **Original source:** `research/new_directions/EXP079_EXP075_A040`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** Unknown / not recoverable from repository history
- **Features:** recency, gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** weighted Delta RMSLE = -9.6967654023e-06
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** | max abs log-space difference | 2.6207559e-08 |
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 5 files, 1 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP079 — EXP075 A040

## Verdict

**NO_GO.** Амплитуда была зафиксирована как `a = 0.40` до расчётов и нигде не
подбиралась. Exact EXP075 parity прошёл, recency-weighted clean-SBVC результат
имеет слабый отрицательный знак, но кандидат не проходит два обязательных gate:

- latest clean fold: `Delta MSE = +9.0626712e-05 > 0`;
- bootstrap `P(Delta MSE < 0) = 0.620 < 0.95`.

Кандидат был собран и полностью проверен, затем удалён по правилу NO_GO. Никакой
submission не отправлялся.

## Parity

Использованы точные исходные артефакты:

- `C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv`, SHA256
  `9a8adb83e7b34bb6c12b7eb51584d1bf9a93825945d285258d4e1dd991f4b838`;
- `JOINT_A1_365_A2_TEST_PERP.npy`, SHA256
  `e3667884a661adf64a6ce5f231956bab18e45a7e6f017e453506f5e93d3045da`,
  shape `(250000,)`, dtype `float32`;
- отправленный `SUBMIT_EXP075_JOINT_A1_365_A2.csv`, SHA256
  `d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901`.

Проверка `z_full = max(log1p(alpha_predict) + D_perp, 0)`:

| check | result |
| --- | ---: |
| rows / unique IDs | 250,000 / 250,000 |
| order vs `sample_submit` | identical |
| clip rows rebuilt / sent | 1,219 / 1,219 |
| clip set | identical |
| RMS stored `D_perp` | 0.0584339123245 |
| RMS applied, float32 rebuild | 0.0581029213298 |
| RMS applied, sent CSV | 0.0581029213313 |
| corr(applied rebuild, sent) | 0.9999999999999998 |
| max abs log-space difference | 2.6207559e-08 |
| permitted bound: half float32 ULP + `%.10f` | 5.9655648e-08 |

**Parity PASS.** Побитовый SHA пересериализованного rebuild не обязан совпадать:
исходный submission был создан из `float64 perp`, а сохранённый после этого
artifact — `float32`; CSV также записан через `%.10f`. Rebuild SHA
`85cf5e517625d0f81fb9ec0107536fdf36b050bc7cf2f4cb21a2a604ddba7b43`
отличается, но наблюдаемый gap полностью находится внутри вычисленной
float32/serialization границы. SHA самого отправленного файла точно совпадает с
EXP075/EXP076 audit.

## Why 0.40

`0.40` взят только из уже существующего EXP076 evidence и зафиксирован промптом:
SBVC prior-only optimum `0.201–0.380`, posterior optimum `0.362–0.471`, public-
implied optimum `0.508` — только диагностический. В EXP079 не было ни нового fit,
ни amplitude sweep, ни обучения, ни LB-настройки.

## Clean SBVC

Повторён механизм EXP076: frozen EXP075 joint coefficients
`[0.7462560853, 0.6466415685]`, затем fold-wise проекция correction из полного
composition-matched SBVC span и применение только фиксированного множителя
`0.40`. Восстановленные `b/G` совпали с `s12_sbvc_folds.csv` с max abs error
`6.25e-15`.

| clean fold | Delta MSE | Delta RMSLE | sign |
| --- | ---: | ---: | --- |
| 2025-09-04 | -2.9017215e-05 | -8.1876817e-06 | gain |
| 2025-09-18 | -2.3485310e-04 | -6.6438563e-05 | gain |
| 2025-10-02 | -1.8612713e-04 | -5.3140848e-05 | gain |
| 2025-10-16 (latest) | **+9.0626712e-05** | **+2.6022090e-05** | **loss** |

Recency weights `1:2:4:8`:

```text
weighted Delta MSE   = -3.4547881105e-05
weighted Delta RMSLE = -9.6967654023e-06
folds not worse      = 3/4
```

Poisson user-cluster bootstrap, 1,000 replicates, один multiplier пользователя
для всех его fold-строк:

```text
Delta MSE 95% CI   = [-2.7168041e-04, +1.9313316e-04]
Delta RMSLE 95% CI = [-7.7725320e-05, +5.5601009e-05]
P(Delta MSE < 0)   = 0.620
```

Средний знак отрицательный, но эффект статистически неустойчив и latest fold
хуже baseline. Поэтому production-like SBVC не подтверждает A040 по заданным
критериям.

## Risk

Для фактически применённого TEST correction после clipping:

```text
RMS(delta_A040)              = 0.02328145755
break-even rho A040          = 0.00707144543
rho, при котором A040 optimal = 0.01414289086
```

| rho scenario | rho | expected Delta MSE | expected Delta RMSLE |
| --- | ---: | ---: | ---: |
| SBVC post-projection prior | 0.00709189 | -1.5672992e-06 | -4.7604720e-07 |
| SBVC min-projection prior | 0.01339783 | -4.8491745e-04 | -1.4729407e-04 |
| posterior lower estimate | 0.01277880 | -4.3746836e-04 | -1.3288079e-04 |
| posterior central estimate | 0.01481937 | -5.9387850e-04 | -1.8039290e-04 |
| realised public rho, diagnostic only | 0.01793073 | -8.3236382e-04 | -2.5283930e-04 |
| no signal | 0 | +5.4202627e-04 | +1.6462532e-04 |

Математический posterior-central robust uplift равен примерно
`-0.0001804 RMSLE`; нижняя posterior оценка даёт `-0.0001329`. Это ожидание не
перевешивает failed forward/bootstrap gates.

No-signal downside:

| amplitude | Delta MSE | Delta RMSLE |
| --- | ---: | ---: |
| A1.00 | +0.0033759495 | +0.0010250824 |
| A0.40 | +0.0005420263 | +0.0001646253 |

A040 оставляет `16.06%` MSE-downside A1.0, то есть уменьшает его на `83.94%`.
До clipping `RMS(delta_A040) / RMS(delta_A100) = 0.4000000000000001`,
`corr(delta_A040, delta_A100) = 0.9999999999999999`.

## Output

Проверенный кандидат имел:

```text
path   = C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP079_EXP075_A040.csv
SHA256 = af1ef0dc5e00d46d5abe3c968b92a3a163c79e7d29e5c0f93f22fe974c9743b8
```

Format PASS: 250,000 строк, columns `user_id,predict`, 250,000 unique IDs,
sample order identical, все значения finite и nonnegative. Дополнительные
статистики до удаления:

```text
zero count            = 800
min / max predict     = 0 / 4523.2595411205
mean / std log1p      = 2.3297307607 / 1.6225015548
RMS(z_new-z_alpha)    = 0.02328145755
corr(z_new,z_alpha)   = 0.99989704901
```

Поскольку verdict — NO_GO, файл по указанному пути **удалён**; SHA оставлен для
аудита и воспроизводимости. Подробные машинные результаты находятся в
`audit.json`, fold table — в `clean_sbvc_a040.csv`, bootstrap draws — в
`bootstrap_a040.npz`.

## Final conclusion

**NO_GO:** exact EXP075 parity подтверждён, A040 существенно ограничивает
no-signal downside и имеет небольшой отрицательный weighted результат, но latest
clean fold отрицательной forward-проверки не выдержал, а `P(gain)=0.62` далеко от
порога `0.95`. Candidate удалён, автоматической отправки не было. Scale-axis
закрыт; другой amplitude не исследуется независимо от будущего leaderboard score.
