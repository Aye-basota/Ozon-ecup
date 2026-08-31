# exp_065 — FINAL-INTEGRATION

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** a28a71f + working tree

## Гипотеза

После отклонения EXP-061..064 legitimate champion должен остаться exact exp_037, а финальная production-сборка обязана воспроизводиться из primitive components без скрытого state. Вторым кандидатом должен быть заранее подтверждённый независимый BTYD05 hedge, а не public-LB-calibrated teammate blend или соседний weight/level probe.

## Что изменено относительно базы

Предсказания не менялись: independently rebuilt exp_037 и byte-copy verified exp_051 упакованы как ровно два canonical CSV; добавлены provenance и OOF→test regime audits.

## Результат

- Candidate A rebuilt из 9 component arrays byte-identical зарегистрированному exp_037: SHA256 `abc2218b...e04bda`, wCV reference 1.747509862, mean log1p on disk 2.329321370.
- Critical `Var(ETX-AVG3−SEQ-AVG3)` test/OOF = **0.7750**, PASS `[0.6,1.2]`; независимый canonical verifier PASS, max reconstruction error `4.97e-7`.
- Candidate B byte-identical exp_051: SHA256 `c3cfb4d...c2932`; fixed OOF delta **−0.000321, 4/4**, production correction variance ratio **1.1734**, support PASS, mean log1p 2.329300000.
- Оба файла: 250,000 rows/unique users, exact sample order, finite, nonnegative, anchor-band PASS. Leaderboard upload не выполнялся.

## Вердикт и вывод

**ACCEPT final package; strongest unchanged.** Ни одна новая session hypothesis не прошла development gate, поэтому validation-first champion остаётся exp_037. Public-score leader teammate pipeline не заменяет его: финальная смесь там public-LB calibrated, а clean-fold occurrence candidates хуже exp_037 на всех четырёх absolute folds.

## Конфиг прогона

Training NONE. A recipe `.10 CAP + .20 UNC + .25 DIST + .075×3 SEQ clip289 + .075×3 ETX DCW`, fixed level 2.3293. B recipe `.95 strong + .05 BTYD`, без изменений exp_051. Запуск `python src/final_integration.py`; manifest SHA256 `442ba96c...ae5e9`; independent ETX/test and disk-recipe verifier PASS; focused tests 11 passed.
