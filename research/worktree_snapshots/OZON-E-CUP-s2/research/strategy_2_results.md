# Strategy 2 — результаты реализации

> Ветка: `team-a-strategy-2-impl` (от `team-a-strategy-research`, которая ответвлена
> от `team-a`). План: `research/strategy_2.md`.

## 1. Что реализовано

- Единый leakage-safe `build_features(cutoff_date)` с фиксированной историей 180 дней,
  базовыми агрегатами и point-process признаками `k_i`, `buygap_*`, `hazard_proxy`,
  `rec_over_buygap`, `n_expected_L`, `lgmv_*`.
- LightGBM Poisson для числа покупательных дней с `init_score=log(n_expected_L)`.
- Hurdle-классификатор `P(n>0)` и перенормированное положительное Poisson-распределение.
- Empirical Bayes LogNormal value-компонент с сеткой `K`.
- Вычисление `E[log1p(S)]`: Sobol QMC lookup при `n≤4`, Fenton–Wilkinson +
  Gauss–Hermite при `n≥5`.
- Вложенная train-only калибровка `sigma_scale`/`mu_shift`.
- Сезонная компонентная поправка `lambda *= 1.1628**alpha`,
  `mu += alpha*0.0804`, кросс-фолдовый бленд и финальная сборка submission.

## 2. Валидация

Основной CV использует принятый протокол проекта:

```
V = 2025-09-04, 2025-09-18, 2025-10-02, 2025-10-16
train: недельные cutoff'ы 2025-04-03..2025-10-16, только T+30<=V
панель train: 1 блок; validation: 3 блока
features: только (T-180,T]; target: (T,T+30]
калибровка: последний доступный train-cutoff, модель для него обучена ещё раньше
```

`src/config.py` и `src/validation.py` не менялись; seed импортируется из config.

## 3. Эксперименты

| блок | результат | решение |
|------|-----------|---------|
| FW correctness | raw max error 0.10027 | заменить малые n на QMC |
| hybrid correctness | max error 0.00549; QN11≈QN21 | KEEP |
| plain vs offset | deviance 1.46310→1.46290; RMSLE без улучшения | offset, эффект нейтрален |
| zero mass | Poisson gap −6.90 п.п.; hurdle −0.81 п.п. | KEEP hurdle |
| hurdle ablation | 1.76579→**1.75749** на 2025-10-16 | основной прирост S2 |
| K grid (2 folds) | K=5: 1.76492; K=8: 1.76485 | K=5, плато |
| sigma ablation | 1.76492→1.76481, выбор нестабилен | выбирать train-only |
| calendar alpha | 1.96787→1.95729 при alpha=0.5 | alpha=0.5 по правилу стратегии |
| blend with S1 | corr 0.9937, median weight 0.025 | REJECT |

## 4. S2-BEST

```
count       LightGBM Poisson, offset n_expected_L, 600 rounds
zero mass   LightGBM binary P(n>0), hurdle
value       EB LogNormal, K=5
aggregation QMC n<=4; FW + GH(11) n>=5; NMAX=30
calibration train-only sigma/mu; final sigma=0.9, mu=-0.1
season      alpha=0.5: lambda factor 1.07833, mu shift +0.0402
submission  standalone S2 (blend rejected), global level 2.3293
```

Full CV: **1.76831 ± 0.00967**; OOF **1.76817**.

| fold | S2-BEST | S1-BEST | delta |
|------|---------|---------|-------|
| 2025-09-04 | 1.78113 | 1.77449 | +0.00664 |
| 2025-09-18 | 1.77351 | 1.76484 | +0.00867 |
| 2025-10-02 | 1.76246 | 1.75170 | +0.01077 |
| 2025-10-16 | 1.75612 | 1.74443 | +0.01169 |

Strategy 2 не обогнала прямой GBM, но честно прошла свой критерий продолжения
(разрыв <0.02). Ансамблевый критерий не пройден, поэтому blend не включён.

## 5. Error analysis

- `y=0`: S2 1.90122 против S1 1.92665 — **лучше на 0.02542**.
- `y>0`: S2 1.67720 против S1 1.64187 — **хуже на 0.03533**.
- По всем бакетам `w180_days_buy` S2 хуже на 0.0058–0.0118.
- Residual correlation с S1 = 0.99369.

Hurdle действительно решил часть задачи нулей. Проигрыш сосредоточен в value-компоненте
на покупающих, поэтому следующий осмысленный шаг — train-only GBM-поправка к EB `mu_i`,
а не дальнейший тюнинг count-модели.

## 6. Финальный файл

`submissions/submission_strategy_2.csv`:

- 250 000 строк, колонки `user_id,predict`;
- порядок полностью совпадает с `sample_submit.csv`;
- 250 000 уникальных пользователей;
- NaN/inf/negative: 0/0/0;
- min 0.0, max 9194.7254;
- `mean(log1p(predict)) = 2.3293`.

**Public LB: 1.6619324597771563.**

| сабмит | ветка | LB (public) | к S1-BEST |
|--------|-------|-------------|-----------|
| `submission_strategy_1.csv` (S1-BEST) | `team-a-strategy-1-impl` | **1.6512803** | — |
| `submission_strategy_2.csv` (S2-BEST) | `team-a-strategy-2-impl` | 1.6619325 | +0.01065 |
| `experimental_submission_1.csv` (EXP-MIN) | `team-a-strategy-1-impl` | 1.6674246 | +0.01614 |
| `experimental_submission_2.csv` (EXP-SIM) | `team-a-strategy-1-impl` | 1.6682180 | +0.01694 |

Разрыв S2 к S1 на LB (**+0.01065**) близок к локальному разрыву по CV (+0.00945) и
подтверждает вывод §4: структурная модель не обгоняет прямой GBM, но остаётся в
пределах порога продолжения 0.02. При этом S2 — лучший из всех альтернативных
сабмитов команды: он на 0.00549 лучше EXP-MIN и на 0.00629 лучше EXP-SIM.
Оба уровня прогноза одинаковы (`2.3293`), поэтому сравнение не искажено калибровкой.

Важные артефакты: `artifacts/s2_oof_best.npz`, `s2_final.json`,
`s2_error_analysis.json`, модели count/hurdle и логи прогонов.

## 7. Воспроизведение

Полная проверка идей:

```powershell
python src/strategy_2.py aggregation --samples 10000
python src/strategy_2.py count-screen --fold 2025-10-16
python src/strategy_2.py cv --folds 2025-09-18 2025-10-16 --mode offset `
  --ks 1 2 3 5 8 15 --sigma-scales 1.0 --hurdle --output s2_k_grid
python src/strategy_2.py cv --folds 2025-09-04 2025-09-18 2025-10-02 2025-10-16 `
  --mode offset --ks 5 --sigma-scales 0.8 0.9 1.0 --hurdle --output s2_oof_best
python src/strategy_2.py season --mode offset --k 5 --mu-shift -0.1 --sigma-scale 1 --hurdle
python src/strategy_2_analysis.py
```

Минимальная команда финального переобучения и submission:

```powershell
python src/strategy_2.py final --mode offset --k 5 --sigma-scales 0.8 0.9 1.0 `
  --hurdle --calendar-alpha 0.5 --structural-weight 1.0 --level 2.3293
```
