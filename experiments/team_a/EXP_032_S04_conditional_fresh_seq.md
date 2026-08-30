# EXP-032 — S04: Conditional Fresh Supervision for SEQ

## Статус

**PROPOSED / HIGH-UPSIDE**

Цель — безопасно использовать поздние наблюдаемые cutoff'ы `2025-10-22..2026-01-14`,
которые нельзя добавлять в обычное обучение `P(y>0)` / direct GMV из-за selection leakage,
но потенциально можно использовать для **условной интенсивности покупки**.

---

## Основание

Аудит `exp_028 FRESH-DIST-MIX` показал:

- все cutoff'ы `T >= 2025-10-17` загрязнены условием отбора пользователей по будущей активности;
- поэтому их нельзя использовать для полной `direct`, `dist` или activity/extensive модели;
- при этом поздние target'ы физически наблюдаемы;
- допустимая отдельная гипотеза — использовать их только для модели
  `E[log1p(GMV_30) | y_30 > 0]`.

Текущий SEQ уже даёт полезную ортогональную ошибку относительно табличной смеси,
поэтому свежую conditional supervision имеет смысл проверить именно поверх sequence encoder.

---

## Гипотеза

Поздние `EXTRA` cutoff'ы содержат более актуальную информацию о **величине покупки**
и пользовательском spending regime ближе к test-периоду.

Если использовать их только среди строк `y30 > 0`, не обучая на них вероятность покупки,
можно улучшить оценку:

`mu_pos = E[log1p(GMV_30) | y30 > 0]`

без внесения selection leakage в extensive-компонент.

---

## CLEAN / EXTRA

### CLEAN

Штатный чистый train-коридор проекта:

`2025-04-03 .. 2025-10-16`

с текущими правилами панели и cutoff-safe target.

### EXTRA

Физически наблюдаемые поздние cutoff'ы:

`2025-10-22, 10-29, 11-05, 11-12, 11-19, 11-26, 12-03, 12-10, 12-17, 12-24, 12-31, 2026-01-07, 2026-01-14`

Использовать **только строки с `y30 > 0` и только для conditional intensity head**.

EXTRA запрещено использовать для:

- `P(y30 > 0)`;
- общего `z30`;
- dist zero-bin;
- activity auxiliary losses;
- validation;
- калибровки уровня.

---

## Первый безопасный pilot

Проверить сначала только fold `2025-10-16`, seed 42.

### Stage 1 — CLEAN encoder

Обучить текущий SEQ encoder стандартным способом **только на CLEAN**.

Не менять:

- sequence representation;
- TCN architecture;
- optimizer;
- epochs;
- channels;
- validation;
- test depth policy.

Если на момент запуска EXP-030/D3A уже подтверждён multi-seed результатом,
можно использовать D3A как encoder baseline. Иначе использовать текущий стабильный SEQ baseline.

### Stage 2 — conditional head

Заморозить encoder.

Обучить отдельную голову:

`mu_pos = E[z30 | y30 > 0]`

на embeddings из:

- CLEAN positive rows;
- CLEAN + EXTRA positive rows.

Получить два строго сравнимых варианта:

1. `COND-CLEAN` — intensity head только CLEAN;
2. `COND-FRESH` — intensity head CLEAN + EXTRA.

Encoder в обоих вариантах должен быть **одинаковым и frozen**.

---

## Почему encoder сначала frozen

Если разрешить EXTRA обновлять shared encoder, selection-contaminated population
может косвенно изменить representation, которое затем использует extensive часть.

Первый experiment должен изолировать только вопрос:

> даёт ли свежая conditional supervision дополнительный signal по величине покупки?

Если frozen-вариант выигрывает, отдельным следующим экспериментом можно проверять
частичный fine-tune encoder с маленьким LR.

---

## Итоговый прогноз pilot

Extensive часть брать **только из CLEAN-trained модели**.

Новый conditional head не должен самостоятельно решать вероятность покупки.

При наличии текущего activity probability `p_buy`:

`z_new = combine(p_buy_clean, mu_pos_fresh)`

Использовать существующий корректный способ two-part / conditional composition,
если он уже реализован в репозитории. Не изобретать новую формулу без отдельного контроля.

Обязателен `COND-CLEAN` control, чтобы измерять выигрыш именно от EXTRA,
а не от самой смены головы.

---

## Validation

Только штатная clean validation проекта:

- `2025-09-04`
- `2025-09-18`
- `2025-10-02`
- `2025-10-16`
- calibrated RMSLE
- wCV weights `1:2:4:8`

Поздние cutoff'ы нельзя использовать как pseudo-validation.

Для первого gate достаточно `2025-10-16`, seed 42.

---

## Diagnostics

Сравнить `COND-FRESH` против `COND-CLEAN`:

- RMSLE_cal;
- RMSLE только по `y>0`;
- общий RMSLE после composition;
- bias / optimal shift;
- `Var(Δz)`;
- residual correlation;
- сегменты по purchase frequency и `rec_buy`;
- отдельно проверить, что `P(y>0)` / activity ranking не меняются из-за EXTRA;
- проверить отсутствие leakage и отсутствие EXTRA-gradient в frozen encoder.

---

## Gate

### CONTINUE

Продолжать на 4 folds / 3 seeds, если на fold `2025-10-16`:

- общий RMSLE не ухудшается;
- conditional RMSLE на `y>0` улучшается;
- итоговый общий gain желательно `<= -0.0005`;
- extensive/activity predictions остаются неизменными;
- leakage audit проходит.

### REJECT

Остановить, если:

- выигрыш есть только на contaminated late diagnostics;
- `COND-FRESH` ухудшает clean `10-16`;
- improvement исчезает относительно `COND-CLEAN`;
- EXTRA влияет на extensive часть;
- frozen-head схема не даёт измеримого сигнала.

---

## Если pilot проходит

Следующая лестница:

1. `COND-CLEAN` vs `COND-FRESH`, frozen encoder;
2. 4 folds × 3 seeds;
3. только после подтверждения — low-LR fine-tune encoder на conditional loss;
4. затем объединить с лучшим SEQ encoder (`SEQ-D3A`, если EXP-030 подтвердится);
5. финальный вклад проверять через LOFO ensemble с сохранённым ненулевым `CAP`.

---

## Что не делать

- не обучать full/direct SEQ на EXTRA;
- не добавлять EXTRA в activity head;
- не использовать поздние cutoff'ы как validation;
- не повторять `FRESH-DIST-MIX`;
- не смешивать одновременно новую architecture, supervision и blending;
- не делать LB submission по одному fold.

---

## Ожидаемая ценность

Это high-upside эксперимент, потому что в отличие от дальнейшего тюнинга TCN
он добавляет **новую, более свежую supervision**, при этом пытается использовать её
только в той части target, где selection leakage не должен напрямую определять
сам факт будущей активности.

Главный вопрос EXP-032:

> Улучшает ли CLEAN+EXTRA conditional supervision оценку величины будущего GMV
> среди покупателей, не затрагивая extensive-компонент?

---

## Результат пилота

Выполнено 2026-08-19, карточка — `experiments/exp_032_s04_cond_fresh_pilot.md`,
код `src/seq_cond.py`. Gate пройден: `COND-FRESH − COND-CLEAN = −0.00145`
калиброванного RMSLE на фолде `2025-10-16` (группа A), контроль объёма
`COND-VOL` даёт лишь −0.00013, диагностика отравления интенсива +0.0057 при
пороге 0.03. Дальше — 4 фолда × 3 сида головы (пункт 2 лестницы выше).

## Результат шага 2 лестницы (замена экстенсива)

Выполнено 2026-08-19, карточка — `experiments/exp_032b_prod_extensive.md`, код
`src/dist_pact.py` + `python -m src.seq_cond prod`. Пункт «При наличии текущего
activity probability `p_buy`: `z_new = combine(p_buy_clean, mu_pos_fresh)`»
закрыт: `p_buy` взят как `1 − p0` боевой головы `S1-DIST`, композиция —
существующая двухчастная (`ẑ = p̂·μ̂`), без новой формулы.

Гейт пройден: `FRESH − CLEAN` = −0.00101 (4/4), налог схемы +0.00088 → −0.00146,
итог против одноголовой базы −0.00247 (4/4). Ограничение найдено там, где спека
его не предполагала: собранный член теряет 2.7× разнообразия и в боевой смеси
при фиксированных весах хуже на 0/4. Поэтому шаг 4 лестницы («объединить с
лучшим SEQ encoder») выполняется НЕ через двухчастную пересборку, а через
шаг 3 — low-LR fine-tune энкодера на conditional loss.
