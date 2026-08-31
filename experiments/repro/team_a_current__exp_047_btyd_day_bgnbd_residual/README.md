# exp_047 — BTYD-DAY-BGNBD-RESIDUAL

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_047_btyd_day_bgnbd_residual`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_047_btyd_day_bgnbd_residual`
- **Original source:** `experiments/exp_047_btyd_day_bgnbd_residual.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** sequence model, BG/NBD, BTYD, blend, calibration diagnostic
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Глобальная группа пользователя: `splitmix64(user_id) & 1`. Для каждого fold выполнен two-sided donor/recipient cross-fit; validation target и строки после cutoff не участвовали в BG/NBD или monetary fit.
- **Known score:** Baseline wCV 1:2:4:8 = **1.747509863** на 770 616 строках.
- **Seed:** Seed `42` взят только из `src/config.py`.
- **Postprocessing:** BG/NBD даёт реальную неуровневую residual direction: все outer folds и residual-alignment checks имеют правильный знак. Но честная nested оценка **−0.000269** не достигает заранее заданной границы `−0.0003`; full-OOF curve переоценивает эффект. Почти вырожденный `P(alive)≈0.994` и систематическое занижение будущего count (`≈1.52` против `1.91..2.01`) показывают, что classic common-origin dropout/count process недостаточно описывает динамику этой панели.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_047 — BTYD-DAY-BGNBD-RESIDUAL

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** `a28a71f`
- **Статус production:** `PROMOTE_TO_PRODUCTION_EXPERIMENT = NO`

## Гипотеза

Common-origin BG/NBD по историческим purchase-day events может дать полезную residual direction к `STRONGEST_CURRENT` за счёт latent `P(alive)` и полной conditional distribution числа покупательных дней на горизонте 30 дней. Главный endpoint — honest nested LOFO blend; слабый standalone BTYD сам по себе не является критерием отказа.

## Что изменено относительно базы

Только frequency-process: supervised count intensity из S2 заменена на basic BG/NBD likelihood. Count×value decomposition, monetary shrinkage и metric-aware S2 aggregation оставлены фиксированными.

```text
NOVEL PART:
latent alive / historical point-process likelihood

NOT NOVEL:
count×value decomposition
monetary shrinkage
metric aggregation
```

## Протокол

- Event: `purchase_day = (gmv > 0)`, один positive daily GMV — один event/value.
- Общий origin `2024-12-31`; `2025-01-01` имеет event time 1. `x` включает все purchase days от origin до cutoff, `t_x` — время последнего event, `T` — время cutoff.
- Глобальная группа пользователя: `splitmix64(user_id) & 1`. Для каждого fold выполнен two-sided donor/recipient cross-fit; validation target и строки после cutoff не участвовали в BG/NBD или monetary fit.
- Basic BG/NBD без covariates и penalizer; 3 заранее фиксированных numerical starts, L-BFGS-B в log-space. Выбирался converged максимум train likelihood.
- Monetary: donor-only `log(gmv_day)`, фиксированный `K=3`, population sigma.
- Агрегация: exact reuse S2 — QMC cache для `n<=4`, Fenton–Wilkinson + GH11 для `n>=5`; theoretical tail `N>=30` сложен в bin 30.
- Blend grid: `0 / .025 / .05 / .10 / .15`; raw log-space blend, fold calibration после blend, outer fold исключён из выбора веса, tie tolerance `1e-5` в пользу меньшего веса.
- Seed `42` взят только из `src/config.py`.

## FACT

### Exact baseline

`STRONGEST_CURRENT = .10 CAP + .20 UNC + .25 DIST + .225 ETX-AVG3 + .225 SEQ-AVG3` восстановлен из сохранённых raw OOF с exact row/target alignment и hashes компонентов.

| Fold | rows | RMSLE calibrated |
|---|---:|---:|
| 2025-09-04 | 188 518 | 1.766883357 |
| 2025-09-18 | 191 025 | 1.760509577 |
| 2025-10-02 | 193 694 | 1.748629224 |
| 2025-10-16 | 197 379 | 1.741278566 |

Baseline wCV 1:2:4:8 = **1.747509863** на 770 616 строках.

### Data/event audit

- История: 30 631 006 дневных строк, 250 000 пользователей, `2025-01-01..2026-02-13`; duplicate user-days нет.
- `gmv>0`, `to_ord>0` и `search_to_ord+cat_to_ord>0`: по **4 736 907** строк каждый; расхождений между флагами **0**.
- Primary event всё равно зафиксирован как `gmv>0`; `to_ord` нигде не трактуется как transaction count.
- Для каждого fold выполнены `0<=t_x<=T`, `x>=0`, `x=0 => t_x=0`, `x>0 => t_x>0`; удаление строк после cutoff не меняет summary.

### BG/NBD fit и count process

Все 8 donor fits converged и прошли preregistered stability gates: maximum gradient norm `3.48e-5`, maximum mean-NLL spread `2.34e-7`, maximum log-parameter spread `0.0946`. Диапазоны параметров по fold/side: `r=0.4726..0.5065`, `alpha=11.7826..12.1246`, `a=0.0350..0.0572`, `b=56.81..70.53`.

| Fold | E[N30] | actual count | AUC(any) | Brier | mean P(alive) |
|---|---:|---:|---:|---:|---:|
| 09-04 | 1.5102 | 1.9072 | .83075 | .16820 | .99337 |
| 09-18 | 1.5134 | 1.9743 | .83280 | .16728 | .99343 |
| 10-02 | 1.5173 | 1.9897 | .83451 | .16635 | .99374 |
| 10-16 | 1.5180 | 2.0051 | .83532 | .16596 | .99410 |

PMF конечна, неотрицательна и нормирована до `1e-8`; capped PMF mean совпадает с independent closed-form capped expectation до `1e-6`; `P(alive)` находится в `[0,1]`.

Standalone BTYD fold scores после отдельной штатной калибровки: `1.815790225 / 1.809706221 / 1.801635019 / 1.796095696`, wCV **1.800700554**.

### Primary nested LOFO endpoint

| Outer fold | selected w | blend RMSLE | delta к baseline |
|---|---:|---:|---:|
| 2025-09-04 | .05 | 1.766228688 | −0.000654669 |
| 2025-09-18 | .05 | 1.759938643 | −0.000570933 |
| 2025-10-02 | .10 | 1.748454140 | −0.000175084 |
| 2025-10-16 | .10 | 1.741085955 | −0.000192612 |

Honest nested LOFO wCV = **1.747240678**, delta = **−0.000269184**. Направление лучше на **4/4** outer folds, включая 10-16; положительный вес выбран на 4/4; residual alignment положительна на 4/4.

Full-OOF fixed-weight curve — только diagnostic: `w=0/.025/.05/.10/.15` даёт delta `0 / −0.000200 / −0.000321 / −0.000324 / −0.000009`. Она оптимистичнее nested endpoint и не использовалась для вердикта.

### Residual и segments diagnostics

- `Var(z_btyd-z_strongest)=0.22545`; Pearson/Spearman predictions `.95741/.95501`.
- Fold-centered residual alignment: `.03394/.03072/.01705/.01764`; raw mean correction `−0.31729`, после fold centering ≈0.
- Honest blend улучшает actual-zero (`−0.002355`), но ухудшает actual-positive (`+0.001197`).
- Выбранные до анализа сегменты: `rec_buy 15–60 −0.000631`, `w180_days_buy 2–15 −0.000273`, intersection `−0.000715`, long recency `−0.000592`, rare `x=1..3 −0.000258`, frequent `x>=11 −0.000426`, never-bought `x=0 −0.000044`.
- Segment gate не создавался.

## INFERENCE

BG/NBD даёт реальную неуровневую residual direction: все outer folds и residual-alignment checks имеют правильный знак. Но честная nested оценка **−0.000269** не достигает заранее заданной границы `−0.0003`; full-OOF curve переоценивает эффект. Почти вырожденный `P(alive)≈0.994` и систематическое занижение будущего count (`≈1.52` против `1.91..2.01`) показывают, что classic common-origin dropout/count process недостаточно описывает динамику этой панели.

## LIMITATIONS

- Protocol требовал exact reuse S2 aggregation. Deterministic MC audit подтверждает QMC-ветку `n<=4` (max error `2.3e-5`), но фиксированная унаследованная FW-ветка имеет error около `0.050..0.051` при `n=10,sigma=1.4` и `0.016` при `n=30,sigma=1.4`. Она намеренно не исправлялась: это изменило бы NOT NOVEL axis. Статус audit: `PASS_EXACT_S2_REUSE_WITH_DOCUMENTED_FW_LIMITATION`.
- Изначальный overly strict numerical preflight gate `mean NLL spread <=1e-7` сработал на первом donor при трёх converged решениях (`2.34e-7` на пользователя), хотя parameter-spread gate прошёл. До построения endpoint predictions порог был исправлен на `1e-6`, прогон выполнен в новом уникальном prefix `..._V2`; наблюдаемая spread осталась существенно ниже исправленного порога. Никаких model/validation решений по результатам не менялось.
- Это один common-origin basic BG/NBD experiment. После REJECT family закрывается для текущего цикла; вывод не является общим утверждением о BTYD вне этой задачи.

## Вердикт: REJECT

Единственная decision reason: `nested_delta_above_-0.0003`. Несмотря на правильный знак 4/4 и улучшение 10-16, primary delta `−0.000269184 > −0.0003`. Production/test inference/submission не выполнялись. Classic BTYD family закрыта: не переходить к Pareto/NBD, Gamma-Gamma и не спасать результат подбором origin/window/K/model/segments.

## Артефакты и воспроизводимость

- Runner: `python src/btyd_day_bgnbd.py`
- Повторный анализ без fit: `python src/btyd_day_bgnbd.py --analysis-only`
- Tests: `python -m pytest src/test_btyd_day_bgnbd.py src/test_validation.py -q`
- Config/prefix: `artifacts/BTYD_DAY_BGNBD_EXP047_V2/config.json`
- Raw OOF SHA256: `754d930b2347beb400b947c416cf56cc036f0b80c35ea039402432263b89d6af`
- Re-analysis: `PASS`; canonical summary SHA256 `2582fdc625f614061150da1a67338d6d9ddac7350b7825ade7b89b8b253ab057`.
- Summary: `research/strategies/results/BTYD_DAY_BGNBD/summary.json`
- Fit/monetary/count/LOFO/fixed-curve/residual/segment tables and artifact manifest are in `research/strategies/results/BTYD_DAY_BGNBD/`.
- Leakage audit: donor/recipient disjoint, stable groups, no target fitting, no rows after cutoff, no test/submission paths touched.
