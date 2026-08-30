# STATE — читать целиком перед каждой задачей (лимит ~80 строк)

## Текущий лучший

**`SEQ-01-MIX`** (`exp_025`) — **LB public 1.6501764**, wCV **1.74834**: веса
0.15·NORM + 0.20·UNC + 0.10·CAP + 0.25·DIST + 0.30·SEQ-01, level 2.3293,
`submissions/submission_SEQ01_mix.csv`. Предыдущий `S1-DIST-MIX`: LB 1.6507774.

## Метрика и валидация — ЕДИНАЯ схема, `exp_016`

Метрика — RMSLE в `log1p`; главная **wCV**: калиброванные fold RMSLE с весами
1:2:4:8 (`exp_016`). Фолды 09-04/09-18/10-02/10-16; val-панель b3, чистый train
04-03..10-16 шаг 7, `T+30≤V`. Reports/OOF — `artifacts/`, журнал — `log.csv`.
**Пороги:** Δ≤−0.0020 → отправлять; −0.0020…−0.0005 → разработка; иначе шум;
обязательно ≥3/4, включая 10-16. Пол разрешения 0.0005; смеси только LOFO с
E03a=0.10. Seed floor: `Var(Δ)=0.00712`, corr=0.99885 (`exp_018`).

## Не повторять (провалы; список только растёт, ничего не удалять)

- **Cutoff 2026-01-14 и любой T≥2025-10-17** отравлены гарантированным окном панели (`e08`).
- **3-блочная train-панель** (`S1-E01`) хуже +0.00102, 4/4; val-панель обязана быть 3-блочной.
- **×1.16** (`e05`), **user_id** (`e04`), **сырые недельные лаги** (`e15`), **минус длинные фичи** (`S1-E04`) — ноль/хуже.
- **L=90** (`E03c`) и **строгий коридор L=180** (`E03b`): выбор L по минимуму bias ведёт к худшей модели.
- **EXP-MIN** (+0.01614) и **EXP-SIM** (+0.01694): низкая corr прогнозов ≠ полезная декорреляция ошибок.
- **Дельты со слабой базы не переносятся**: `DIST` дал −0.0040 против 3 cutoff'ов и −0.0015 против плотной.
- **Только выборка последнего fold** (`DIST-F4`): LB +0.00042; CV 24 и 29 cutoff'ов одинаков, локальный валидатор слеп.
- **«Асимметрия 0.64×»** отменена (`exp_016`): артефакт OOF cal; wCV даёт 0.975, но только на крупных дельтах.
- **Rounds≥450 у direct** (`exp_017`): 600 переобучает; 1600 стоят +0.00287, 0/4. Capacity зависит от объёма train.
- **Подбор весов ради 4-го знака** (`MIX-E11`): wCV −0.00038, LOFO −0.00036, LB +0.00023; E03a нельзя обнулять.
- **Gap-axis k=5/k=11 как selection gate** (`exp_019`): зарегистрированный bias slope не воспроизведён, вывод зависит от k; только диагностика.
- **`train_blocks=0`** (`exp_020`): avg3 +0.000113, 2/4 и последний хуже; +9.23% строк без выигрыша.
- **Постобработка `ẑ`**: affine −0.00027, segment −0.00017, правила неустойчивы; закрыто, включая `EXP-17/18`.
- **Личное время** (`exp_021`): 30 `pt_*` дали −0.000006 wCV, ΔAUC≈0; shuffled-ρ не хуже; `Var(Δ)=0.00270` < seed floor.
- **Dense supervision равнообъёмно** (`exp_022`): +0.001263 wCV, 0/4; AUC −0.000416; gate провален.
- **Multi-horizon hazard + count** (`exp_024`): +0.00286, 0/4; FULL=SELF. Стекинг мерить от SELF; `C/C2` закрыты.
- **Простой TCN как ранжирование** (`exp_025`): standalone +0.00322, AUC −0.00129; полезен только в смеси.
- **Renewal Clock** (`exp_027`): AUC 0.84106 < 0.84552 на 4/4; LOFO −0.000416 < gate, несмотря на diversity 8.67×. **STOP**.
- **DOMAIN importance weighting** (`exp_028`): shift почти весь technical depth (AUC 0.986); behavioral weighting ухудшило ordinary/weighted wCV на +0.000087/+0.000086, LOFO +0.000010; не повторять.
- **Calendar/YoY specialist по fixed-L180 drift** (`exp_029`): AUC real 0.644 следует placebo gap-curve (ожидание 0.6455); exact YoY state не имеет L180/3-block support, direction score не даёт component win. **STOP-CALENDAR**, не продолжать без новых полных годовых данных.

## Последние эксперименты (макс 10 строк, переполнение → HISTORY.md)

| ID | Дата | Автор | Гипотеза | wCV | Вердикт |
|----|------|-------|----------|-----|---------|
| CALENDAR-PLACEBO-01 | 2026-08-13 | Codex | fixed-L180 historical domain placebos + signed direction/YoY/error gates | — | **STOP-CALENDAR** (`exp_029`) |
| DOMAIN-01 | 2026-08-13 | Codex | grouped adversarial validation + test-like weighted CV/training | 1.75116; Δ+0.000087 | **STOP** (`exp_028`) |
| RENEWAL-01 | 2026-08-13 | Codex | recurrent-event Clock: KM shrinkage + timing-only GBDT | 1.75969; LOFO Δ−0.000416 | **STOP** (`exp_027`) |
| SEQ-01 | 2026-08-13 | A1 | S_10 B: dilated TCN на сырой дневной истории 365д вместо 227 агрегатов | 1.75270 | **CONTINUE** (`exp_025`) |
| MHZ-FULL | 2026-08-13 | A1 | S_03: multi-horizon hazard + счёт как супервизия | 1.75234 | **REJECT** (`exp_024`) |
| HOLIDAY-YOY | 2026-08-12 | A1 | персональная holiday-response 2025→2026 + placebo | 1.74958 (+0.00009) | **SEND_HIGH_RISK** (`exp_023`) |
| S1-DIST-MIX | 2026-08-11 | A1 | смесь с головой распределения | **1.74948** | **LB 1.6507774** |
| S1-ROUNDS | 2026-08-12 | A1 | S_05 A: кривая по раундам `direct`, 25..1600 | **1.75108** | в разработку; 600 = переобучение |
| S1-SEEDAVG5 | 2026-08-12 | A1 | S_05 B: усреднение 5 сидов при 300 раундах | **1.75037** | в разработку; брать 3 сида |
| S1-GAPAXIS | 2026-08-12 | A1 | S_01: gap-axis k=5 + k=11 control | gCV 1.75637 | REJECT |

LB public: SEQ-01-MIX 1.6501764 < DIST-MIX 1.6507774 < MIX-E11 1.6510029 < f4 1.6512012 < S1-BEST 1.6512803
< S2-BEST 1.6619325 < MIN 1.6674246 < SIM 1.6682180. Уровень `L*=2.3293`.

## Backlog

- [ ] **S10/SEQ** (`exp_025`): Var(Δ) 5.3× floor, LOFO −0.00106 4/4 при E03a=0.10; дальше positional supervision.
- [ ] Strategy index/Tier A report: S05 A+B PARTIAL; S01/S02A/S03 REJECT; S02B/S08 FAIL; осталась S04.
- [ ] **ETX → DEPRIORITIZED** (`exp_021`): attention по всем парам событий ещё не проверялся.
- [ ] **SEQ-DEPTH-AUG-01** (`exp_027`/`exp_029`): random input-depth cropping для existing sequence encoder; единственный следующий эксперимент после STOP-CALENDAR.
- [ ] Диагноз: 74% дисперсии — purchase/no-purchase, AUC≈0.844; остались sequence representation или новые данные/S04.
