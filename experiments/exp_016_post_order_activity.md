# exp_016 — post-order activity features

- **Дата:** 2026-08-15
- **Автор:** B1
- **Коммит:** 0366bd9

## Гипотеза

Пользователи, которые после последней покупки продолжают искать/класть в корзину, отличаются от тех, кто купил и пропал. Добавляем признаки активности после последнего заказа, чтобы модель различала эти сценарии без жёсткого classifier gate.

## Что изменено относительно базы

Добавлен feature set `long_buy_post_order`: это `long_buy` + новые post-order колонки. Ансамбль сравнивает текущий `recency + long_buy` с вариантом `recency + long_buy_post_order`.

## Результат

- CV по фолдам: `2025-12-15 -> 2026-01-14`, `2025-11-15 -> 2025-12-15`.
- Baseline exp_011-like `recency + long_buy`, `w_rec=0.5`, `scale=1.20`: mean RMSLE `1.709007`.
- Новый `recency + long_buy_post_order`, `w_rec=0.5`, `scale=1.20`: mean RMSLE `1.708883`.
- По фолдам:
  - fold 1: `1.673083 -> 1.672956`
  - fold 2: `1.744931 -> 1.744811`
- Submit-кандидат: `submissions/exp_016_post_order_wrec050_scale120.csv`.
- LB: `1.6547788658437297`.

## Вердикт и вывод

Успех: новый лучший LB `1.6547788658437297`. Улучшение небольшое, но оно есть и на обоих CV-фолдах, и на LB. Идея с “активен после последней покупки / ушёл после покупки” полезнее как фичи, чем как hard-zero classifier gate.

## Конфиг прогона

Модель: LightGBM `recency` + LightGBM `long_buy_post_order`, log-space blend `0.5/0.5`, component scales `0.64/0.62`, global scale `1.20`, dense8 train cutoffs для submit `2025-08-28..2025-10-16`, seed из `config.py`.
