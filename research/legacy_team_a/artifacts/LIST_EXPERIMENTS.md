Experimental roadmap: E-CUP 2026 LTV / GMV-30

0. Главный принцип исследования

Цель — не построить одну максимально сложную модель, а получить несколько моделей, которые смотрят на задачу принципиально по-разному:

GBDT / direct regression — извлекает нелинейности из агрегированных признаков.

Hurdle / zero-inflated architecture — отдельно моделирует вероятность покупки и размер GMV.

Autoregressive model — моделирует пользователя как временной ряд 30-дневных состояний.

Channel decomposition — отдельно прогнозирует Search и Catalog.

BTYD / probabilistic behavioral model — моделирует propensity, частоту и «живость» пользователя.

Sequential neural model — учится напрямую на последовательности пользовательских событий.

Lifecycle / Mixture-of-Experts — допускает, что dormant, active и new users требуют разных моделей.

Final stacking — объединяет только модели с различающимися ошибками.

Главный артефакт каждого эксперимента — не leaderboard score, а:

temporal OOF RMSLE;

RMSLE по пользовательским сегментам;

OOF predictions;

OOF residuals;

correlation residuals с остальными моделями.

Именно это позволит впоследствии построить сильный ансамбль.

I. Общая инфраструктура для всех экспериментов

До сравнения архитектур необходимо зафиксировать единый экспериментальный протокол.

EXP-00. Temporal snapshot dataset

Гипотеза

Одна строка на пользователя на последнюю дату недостаточна для обучения хорошей модели. Нужно превратить историю в множество задач вида:

history <= cutoff → GMV следующих 30 дней


Построение train

Для пользователя u и cutoff t:

X(u,t) = признаки, рассчитанные только по данным <= t
y(u,t) = sum(GMV за t+1 ... t+30)
z(u,t) = log1p(y(u,t))


Cutoff генерируются, например, каждые 7 или 14 дней.

Необязательно использовать абсолютно все возможные даты.

Хороший старт:

step = 14 days


После стабилизации pipeline:

step = 7 days


Temporal CV

Рекомендуемая схема:

Fold 1
validation target:
октябрь → ноябрь 2025

Fold 2
ноябрь → декабрь

Fold 3
декабрь → январь

Fold 4
январь → февраль 2026


Последний fold должен максимально имитировать production/test regime.

Purging

Если validation target начинается в T, training snapshot нельзя использовать, если его target пересекается с validation.

Иначе возникает temporal leakage.

Что сохраняем

Для каждого fold:

user_id
cutoff
target_gmv
target_log_gmv
prediction


В дальнейшем все модели должны использовать одни и те же folds.

II. Архитектура A — Direct LightGBM

EXP-01. Strong tabular baseline

Цель

Построить максимально сильный чистый GBDT baseline, относительно которого оцениваются все остальные архитектуры.

Target

target = log1p(gmv_next_30)


Loss:

RMSE


Финальный prediction:

predict = expm1(z_pred)


Feature block A1 — multi-window aggregates

Для каждого сигнала:

GMV
orders
order_days
carts
searches
active_days


окна:

1
3
7
14
30
60
90
180
365 days


Например:

gmv_7
gmv_30
gmv_90
orders_7
orders_30
searches_14
cart_30
...


Feature block A2 — recency

days_since_last_activity
days_since_last_search
days_since_last_cart
days_since_last_order

days_since_2nd_last_order
days_since_3rd_last_order
days_since_5th_last_order


Feature block A3 — gaps

last_order_gap
mean_last_3_order_gaps
mean_last_5_order_gaps

median_order_gap
std_order_gap
max_order_gap_90

current_inactivity_streak
max_inactivity_streak_90


Feature block A4 — ratios

orders / searches
orders / carts
carts / searches

gmv / orders
gmv / active_day

recent_orders / historical_orders
recent_gmv / historical_gmv


Все ratios должны иметь smoothing:

(a + alpha) / (b + beta)


EXP-02. LightGBM + 30-day block representation

Это отдельный ablation внутри архитектуры LightGBM.

Гипотеза

Так как target имеет горизонт 30 дней, модель должна видеть историю в той же временной гранулярности.

Для каждого пользователя относительно cutoff:

M1 = GMV[-30:]
M2 = GMV[-60:-30]
M3 = GMV[-90:-60]
...
M12


Но перед подачей модели:

log1p(M1)
...
log1p(M12)


Аналогично:

orders_month_1 ... orders_month_12
search_month_1 ...
cart_month_1 ...
active_days_month_1 ...


Дополнительные признаки

mean_3m
mean_6m
mean_12m

median_6m
std_6m

month_1 - month_2
month_1 - mean_3m

slope_3m
slope_6m

zero_month_fraction
positive_month_count

EWMA


Что сравниваем

EXP-01:
обычные rolling windows

EXP-02:
rolling windows
+
30-day blocks


Success criterion

Продолжаем использовать block features, если:

Δ RMSLE <= -0.002


стабильно хотя бы на 3 temporal folds.

EXP-03. LightGBM + Year-over-Year architecture

Это один из самых важных экспериментов.

Гипотеза

Для финального прогнозируемого периода известен аналогичный период прошлого года.

Нужно моделировать:

seasonality
+
personal seasonality
+
current deviation from personal seasonality


YoY 365

Для каждого snapshot t считаем значения, соответствующие будущему target-периоду, но год назад.

Например:

yoy_future_gmv_365
yoy_future_orders_365
yoy_future_cart_365
yoy_future_searches_365
yoy_future_active_days_365


YoY 364

Отдельно строим weekday-aligned версию:

yoy_future_gmv_364
...


Relative YoY

Особенно важные признаки:

recent_30_gmv / yoy_recent_30_gmv

recent_30_orders / yoy_recent_30_orders

yoy_future_gmv / yoy_previous_30_gmv


И:

current_vs_yoy_activity
current_vs_yoy_orders
current_vs_yoy_search


Основной вопрос эксперимента

Помогает ли прошлогоднее состояние пользователя прогнозировать будущий период сверх recent behavior?

Ablation

A: baseline
B: +365
C: +364
D: +364 +365
E: +364 +365 + relative YoY


III. Архитектура B — CatBoost

CatBoost нужно рассматривать независимо от LightGBM.

Причина — другие splits и другой inductive bias.

EXP-04. Direct CatBoost

Target

log1p(GMV_next30)


Features

Идентичный feature store EXP-03.

То есть:

RFM
rolling windows
30-day blocks
YoY
gaps
ratios
lifecycle


Зачем нужен эксперимент

Даже если:

CatBoost = 1.655
LightGBM = 1.650


CatBoost может оказаться крайне полезен для ensemble, если residual correlation недостаточно высокая.

Смотрим

corr(error_lgbm, error_cat)
corr(pred_lgbm, pred_cat)


EXP-05. CatBoost Hurdle Model

Это уже принципиально другая архитектура.

Stage 1 — transaction probability

Target:

has_gmv_30 = GMV_next30 > 0


Model:

CatBoostClassifier


Получаем:

p = P(GMV_next30 > 0)


Stage 2 — conditional GMV

Train только:

GMV_next30 > 0


Target:

z_positive = log1p(GMV_next30)


Model:

CatBoostRegressor


Получаем:

mu = E[
    log1p(GMV)
    | GMV > 0
]


Combination

Поскольку:

log1p(0) = 0


получаем:

z_pred = p * mu


И:

gmv_pred = expm1(z_pred)


Ablation

Проверить:

EXP-05A
pure hurdle

EXP-05B
direct CatBoost

EXP-05C
blend:
0.5 direct +
0.5 hurdle


Вес обязательно оптимизировать в OOF.

Что особенно анализировать

Hurdle должен оцениваться отдельно для:

no recent purchase
purchase <7d
purchase 7-30d
purchase 30-90d
purchase >90d


Если основной gain возникает у dormant users — архитектура делает именно то, что требуется.

IV. Архитектура C — Autoregressive model

EXP-06. Monthly autoregressive Ridge / ElasticNet

Это намеренно простая модель.

Она нужна не для победы сама по себе, а чтобы создать модель с принципиально другим bias.

Representation

z1 = log1p(GMV за последние 30 дней)
z2 = log1p(GMV за предыдущие 30)
...
z12


Дополнительно:

orders_1 ... orders_12
cart_1 ... cart_12
search_1 ... search_12
active_days_1 ... active_days_12


Model

Начать с:

Ridge


Затем:

ElasticNet


Target

log1p(GMV_next30)


Пример

future_z =
β0
+ β1*z_1
+ β2*z_2
...
+ β12*z_12
+ seasonal terms


Почему это важно

Если модель получает, например:

1.68 standalone


но residual correlation с LightGBM значительно ниже CatBoost, она может быть очень ценным ensemble component.

EXP-07. Autoregressive nonlinear MLP

Следующий challenger:

12 monthly states
↓
small MLP
↓
future log GMV


Например:

input
→ Linear(256)
→ GELU
→ Linear(128)
→ GELU
→ Linear(1)


Задача эксперимента:

понять, нужен ли nonlinear temporal mapping без полноценного Transformer.

V. Архитектура D — Search / Catalog decomposition

EXP-08. Channel-specific models

Гипотеза

Search и Catalog могут иметь различные паттерны:

purchase frequency
AOV
conversion
recency
seasonality


Поэтому общий GMV может скрывать две разные динамики.

Model S

Target:

log1p(Search_GMV_next30)


Features:

Search history
+
general user features


Model C

Target:

log1p(Catalog_GMV_next30)


Features:

Catalog history
+
general user features


Важно

Не использовать автоматически:

expm1(z_search)
+
expm1(z_catalog)


как финальный прогноз.

Лучше OOF meta model:

total_z =
f(
    direct_total_z,
    search_z,
    catalog_z
)


Первый вариант:

Ridge


Дополнительные interaction features

search_share_gmv
catalog_share_gmv

search_to_catalog_transition

recent_search_share
historical_search_share

channel_switch_indicator


VI. Архитектура E — Order frequency × basket model

EXP-09. Transactions × AOV decomposition

Это ещё одна независимая стратегия.

Model 1

Предсказываем:

orders_next30


Например:

log1p(orders_next30)


или classification:

0
1
2
3+


Model 2

Предсказываем conditional average order value:

AOV_next30


Reconstruction

GMV ≈ expected_orders × expected_AOV


Этот прогноз не обязан быть лучшим standalone.

Главная цель — получить ещё одну ортогональную оценку будущего GMV.

Auxiliary outputs

Особенно полезно добавить в final model:

expected_orders
prob_any_order
expected_AOV


как meta-features.

VII. Архитектура F — BTYD / probabilistic behavioral model

EXP-10. BTYD

Начальные кандидаты:

BG/NBD
Pareto/NBD


Для monetary component:

Gamma-Gamma


Input

Для каждого пользователя:

frequency
recency
T
monetary_value


Outputs

prob_alive
expected_transactions_30
expected_AOV
expected_GMV_30


EXP-10A

Использовать BTYD напрямую.

Оценить standalone RMSLE.

EXP-10B

Использовать BTYD outputs как features для LightGBM/CatBoost:

prob_alive
expected_transactions_30
expected_GMV_30


Это потенциально более перспективный вариант.

Особое внимание

BTYD score считать отдельно для:

1 historical order
2-3 orders
4-10 orders
10+ orders


Гипотеза:

наибольшая польза BTYD будет именно на sparse users.

VIII. Архитектура G — Sparse Event Transformer

EXP-11. Transformer

Это должна быть полностью независимая ветка.

Не добавляем Transformer поверх LightGBM.

Сначала проверяем, способен ли sequence model извлекать signal сам.

Representation

Не создаём отсутствующие дни.

Каждая существующая daily record становится token.

Token:

log1p(searches)
log1p(search_to_cart)
log1p(cat_to_cart)

log1p(search_to_order)
log1p(cat_to_order)

log1p(search_gmv)
log1p(cat_gmv)

delta_days_since_previous_event

day_of_week
day_of_month
month
day_of_year


Дополнительно binary:

has_order
has_cart
has_search


Sequence

Используем последние:

128


или:

256


активных дней.

Это важно:

128 events ≠ 128 calendar days


Временные gaps кодируются через:

delta_days


Baseline architecture

numeric token projection
+
calendar embeddings
+
delta-time embedding

↓

Transformer Encoder
3-4 layers
d_model ≈ 128
4 heads

↓

attention pooling / CLS

↓

MLP

↓

log1p(GMV_next30)


EXP-11A — single-task

Loss:

MSE(log1p(GMV))


EXP-11B — multi-task

Heads:

Head 1:
log1p(total GMV)

Head 2:
P(any order)

Head 3:
log1p(number of orders)

Head 4:
log1p(search GMV)

Head 5:
log1p(catalog GMV)


Loss:

L =
L_gmv
+ λ1 L_purchase
+ λ2 L_orders
+ λ3 L_search
+ λ4 L_catalog


Цель auxiliary tasks:

заставить embedding пользователя понимать его behavioral state.

EXP-11C — Transformer + handcrafted context

После Transformer pooling:

sequence_embedding
+
RFM features
+
YoY features
+
lifecycle features


Далее:

MLP
→ prediction


Это наиболее перспективная neural architecture.

Sampling во время обучения

Snapshot можно генерировать online:

1. выбрать user
2. выбрать random valid cutoff
3. взять events <= cutoff
4. взять target GMV следующих 30 дней


Это позволит генерировать огромное число различных состояний пользователя без материализации snapshot dataset.

IX. Архитектура H — TCN / GRU challenger

EXP-12. Lightweight sequence model

Transformer может оказаться неоптимальным для такого объёма данных.

Поэтому нужен более дешёвый challenger.

Кандидаты:

GRU
Temporal Convolutional Network


Input тот же, что у Transformer.

Почему эксперимент важен

Если:

Transformer OOF = 1.64
GRU OOF = 1.645


но GRU обучается в 5 раз быстрее, становится возможным:

больше seeds
больше snapshots
больше ablations


И итоговый ensemble может оказаться сильнее.

X. Архитектура I — lifecycle segmentation / Mixture of Experts

EXP-13. User lifecycle model

Гипотеза

Следующие пользователи имеют принципиально разные процессы:

A. new
B. active buyer
C. occasional buyer
D. declining
E. dormant
F. reactivated


Одна модель вынуждена аппроксимировать все режимы одновременно.

Stage 1 — lifecycle definition

Не обязательно ML.

Например:

Active

order <= 30 days


Warm

31-90 days


Dormant

>90 days


Never buyer

0 historical orders


Reactivated

long gap
+
recent search/cart/order


Stage 2 — experts

Например:

Expert 1:
active users CatBoost

Expert 2:
dormant users hurdle CatBoost

Expert 3:
new/sparse users BTYD + GBDT


EXP-13A — hard routing

segment → model


EXP-13B — soft routing

Gate model:

P(active regime)
P(dormant regime)
P(new regime)


Final:

z =
p1*z1
+
p2*z2
+
p3*z3


Caveat

Этот эксперимент запускать только после сильных standalone моделей.

Mixture-of-Experts очень легко переобучить.

XI. Архитектура J — user propensity / empirical Bayes

EXP-14. Persistent user value

Для каждого пользователя считаем исторические 30-дневные GMV:

z1
z2
...
zk


где:

zi = log1p(GMV_i)


Из них:

mean_user_z
median_user_z
std_user_z

positive_month_fraction
zero_month_fraction


Shrinkage

Для sparse users:

shrunk_user_value =
n/(n+k) * user_mean
+
k/(n+k) * population_mean


Проверяем два варианта

A:
standalone empirical Bayes prediction

B:
EB features → LightGBM


Этот блок должен хорошо оценивать постоянную latent покупательскую propensity.

XII. Финальный ensemble experiment

EXP-15. OOF model selection

Теперь у нас потенциально есть:

M1 LightGBM direct
M2 CatBoost direct
M3 CatBoost hurdle
M4 AR Ridge
M5 channel model
M6 order × AOV
M7 BTYD
M8 Transformer
M9 GRU/TCN
M10 Mixture-of-Experts


Нельзя просто смешивать все модели.

Шаг 1. Standalone leaderboard

Строим таблицу:

ModelOOF RMSLELast foldZero targetActiveDormant











LGBM











Cat











Hurdle











AR











Transformer











Шаг 2. Residual correlation

Для каждой модели:

error_i =
true_log_gmv - predicted_log_gmv


Строим correlation matrix.

Особенно интересны модели:

хороший RMSLE
+
низкая residual correlation


Шаг 3. Greedy ensemble selection

Начинаем с лучшей модели.

Например:

ensemble = LGBM


Последовательно проверяем добавление:

+ CatBoost hurdle
+ Transformer
+ AR
+ channel model
...


Модель остаётся в ensemble только если улучшает OOF.

XIII. Stacking

EXP-16. Log-space stacking

Все predictions переводим в:

z = log1p(prediction)


Meta features:

z_lgbm
z_cat
z_hurdle
z_transformer
z_ar
z_search
z_catalog
prob_purchase
expected_orders


Первый meta-model:

Ridge


Это предпочтительнее сложного meta-GBDT.

Ограничение

Meta-model обучается исключительно на OOF predictions.

Никогда:

train-model prediction
→ meta-model


иначе будет leakage.

XIV. Calibration

EXP-17. Global calibration

На OOF обучаем:

z_true = a + b*z_pred


и применяем:

z_calibrated =
a + b*z


EXP-18. Segment calibration

Если видим систематические различия:

active
dormant
new


можно использовать разные:

a_segment
b_segment


Но только для нескольких крупных сегментов.

XV. Итоговый Winning Pipeline

После экспериментов финальная архитектура должна быть не «самой сложной», а комбинацией лучших ортогональных моделей.

Предполагаемый вариант:

                    RAW DAILY DATA
                          │
              ┌───────────┴───────────┐
              │                       │
       TABULAR FEATURE STORE      EVENT SEQUENCES
              │                       │
       ┌──────┼────────┐         ┌────┴─────┐
       │      │        │         │          │
      RFM   Blocks    YoY    Transformer   GRU
       │      │        │         │
       └──────┼────────┘         │
              │                  │
       ┌──────┴──────┐           │
       │             │           │
     LGBM       CatBoost         │
       │             │           │
       │       ┌─────┴──────┐    │
       │       │            │    │
       │   Direct Cat    Hurdle  │
       │                         │
       ├────────────┬────────────┤
       │            │            │
      AR       Channel model  Transformer
       │            │            │
       └────────────┼────────────┘
                    │
              OOF PREDICTIONS
                    │
             Residual analysis
                    │
             Greedy selection
                    │
             LOG-SPACE STACK
                    │
               Calibration
                    │
                  expm1
                    │
             FINAL SUBMISSION


XVI. Каким я ожидаю реальный final ensemble

Наиболее вероятный кандидат:

35–50%
LightGBM direct

20–30%
CatBoost hurdle

15–25%
Sparse Transformer

5–15%
AR / monthly model

0–15%
channel decomposition


Это не стартовые веса.

Настоящие веса должны определяться исключительно OOF.

BTYD скорее всего войдёт не как самостоятельный член ensemble, а как feature provider.

XVII. Приоритет экспериментов

Я бы запускал исследования именно в таком порядке.

Phase 1 — validation

E00

Temporal snapshots + purged CV.

Без этого нельзя доверять остальным экспериментам.

Phase 2 — максимизация GBDT

E01

Strong LightGBM.

E02

30-day blocks.

E03

YoY 364/365.

E04

Direct CatBoost.

E05

CatBoost hurdle.

После этой фазы должен появиться очень сильный tabular baseline.

Phase 3 — diversity

E06

AR Ridge.

E08

Search/Catalog.

E09

Orders × AOV.

E10

BTYD.

Цель фазы не обязательно улучшить standalone score.

Цель — найти модели с отличающимися residuals.

Phase 4 — neural

E11A

Transformer single-task.

E11B

Transformer multi-task.

E11C

Transformer + tabular context.

E12

GRU/TCN.

Phase 5 — advanced behavioral modeling

E13

Lifecycle experts.

E14

Empirical Bayes user propensity.

Phase 6 — final optimization

E15

Residual-based model selection.

E16

Stacking.

E17

Calibration.

XVIII. Правило принятия эксперимента

Чтобы избежать endless feature engineering, для каждого эксперимента заранее задаём decision rule.

Feature experiment

Оставляем feature block, если:

mean ΔOOF <= -0.002


или:

он значительно улучшает важный сегмент
и не портит общий score.


New architecture

Оставляем standalone модель, если выполнено хотя бы одно:

Условие A

Она улучшает лучший OOF.

или

Условие B

Она немного хуже:

до +0.01 RMSLE


но имеет заметно более низкую residual correlation.

или

Условие C

Она сильно улучшает конкретный проблемный сегмент:

dormant
new
zero/nonzero
high-value


XIX. Таблица экспериментов для MLflow / W&B

Для каждого run сохранять:

experiment_id

architecture

feature_version

snapshot_version

fold

random_seed

target_type

parameters

OOF_RMSLE

last_fold_RMSLE

zero_target_RMSLE

positive_target_RMSLE

active_RMSLE

dormant_RMSLE

new_user_RMSLE

residual_std

prediction_mean

prediction_p50
prediction_p90
prediction_p99


И обязательно:

OOF prediction file


XX. Самая важная ablation matrix

В первую очередь я бы хотел получить такую таблицу:

ExperimentOOF RMSLEΔLast foldDecision









Current LGBM



—



baseline

+ temporal snapshots









+ 30d blocks









+ YoY365









+ YoY364









+ recency/gaps









Direct CatBoost









CatBoost Hurdle









AR Ridge









Search/Catalog









Transformer









Transformer multitask









Final ensemble









XXI. Что считать успешным исследованием

Не ставил бы задачу:

найти архитектуру с RMSLE < 1.64


Гораздо правильнее поставить три цели.

Goal 1

Получить максимально сильную tree model:

LGBM/CatBoost
+
snapshots
+
30-day blocks
+
YoY
+
behavioral features


Goal 2

Получить минимум одну качественно отличающуюся модель:

Transformer
или
AR
или
hurdle


с достаточно низкой residual correlation.

Goal 3

Получить ensemble, который стабильно улучшает каждый temporal fold, а не только один leaderboard submission.

Финальная исследовательская гипотеза

Наиболее вероятный путь к выигрышному решению выглядит не как:

очень большой Transformer


и не как:

ещё более затюненный LightGBM


а как сочетание трёх разных представлений пользователя:

1. Current behavioral state
   → GBDT

2. Purchase probability / dormant activation
   → Hurdle

3. Temporal behavioral trajectory
   → Transformer

4. Persistent long-term propensity
   → AR / empirical Bayes


которые затем объединяются через:

OOF log-space stacking
+
calibration.
