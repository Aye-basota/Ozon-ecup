# exp_050 — BTYD05 production resolution

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_050_btyd05_production_resolution`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_050_btyd05_production_resolution`
- **Original source:** `experiments/exp_050_btyd05_production_resolution.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** dilated TCN, sequence model, BTYD
- **Features:** freshness/conditional features, history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** ## VALIDATION STATUS
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** остановлен; D3A/seed rescue, fold averaging и OOF→test carry не запускались.
- **Postprocessing:** None documented
- **Submission:** submissions/submission_STRONGEST_CURRENT.csv
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_050 — BTYD05 production resolution

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево
- **Код:** `src/btyd05_production.py`
- **Запуск:** `python src/btyd05_production.py`
- **Тип:** production-resolution only; neural/research training = **NO**

## Гипотеза

Проверить, можно ли без изменения зарегистрированных `exp_040/047/049`
semantics собрать `BTYD05_FRESH1`, а при провале exact FRESH parity — только
разрешённый fallback `BTYD05`.

## Что изменено относительно базы

Нового research axis нет. Для cutoff `2026-02-13` разрешён exact two-sided
BTYD cross-fit: donor A scores B, donor B scores A; состав смеси фиксирован.

## VALIDATION STATUS

**PREFERRED FROM `exp_049`; NO NEW VALIDATION.** Fixed combined endpoint имел
conditional k>0 delta `−0.000551`, 3/3. Веса и model family не менялись.

## PRODUCTION PARITY STATUS

### FRESH — FAIL

`exp_040` использовал frozen `SEQ-D3A-BASE` TCN, seed 42, 4 эпохи, 8 блоков,
hidden 64, `aug=none`, внешний `depth_clip=289`, pooled embedding
`[last,mean,max]`; donor split — `splitmix64(user_id)&1`, two-sided A/B.
Требуемая обработка: raw `z_FRESH-z_CLEAN`, donor-safe 0.5/99.5%
winsorization, GLOBAL, centering, alpha=1.

Все четыре fold checkpoint существуют и зафиксированы hashes:

```text
V0904 cc63eb41a893f8cd24b77b6b7b129a503f1e37001c1eae42b09c7398b1704dd6
V0918 19147af98a944c2d403b68e0383c0f857cab0e5d08564a7ce918d058f9ee85bb
V1002 d6c29829bad07e5998627fd6882c96e4efa49012cabe7100aa81a3bf7aad48b3
V1016 e5493a0f704944fdbc00551ddd38a3009e7cf93ac14dc86c6e9ad50d2c198a66
```

Но exact production checkpoint
`artifacts/model_SEQ-D3A-BASE-S42-TEST.pt` отсутствует. Сохранены только
`SEQ-C289` test checkpoints seed 43/44 — это другая model identity/trajectory.
Conditional-head weights `exp_040` также не сохранялись (их разрешалось бы
переобучить только при наличии exact production encoder). Поэтому FRESH
остановлен; D3A/seed rescue, fold averaging и OOF→test carry не запускались.

### BTYD — FAIL_UNSTABLE_MLE

Точный `exp_047` preprocessing прошёл: raw event history ограничена
`event_date <= 2026-02-13`, common origin `2024-12-31`, event=`gmv>0`
purchase day, sample/raw universe exact 250,000 users, cutoff rebuild exact.

Первый обязательный population fit (donor group 0, 124,787 users, T=409)
провалил неизменённые numerical stability gates:

| diagnostic | production | gate |
|---|---:|---:|
| mean-NLL spread между 3 starts | `1.62433e-5` | `<=1e-6` |
| max log-parameter spread | `1.18876` | `<=0.10` |
| max gradient norm | `0.002274` | `<=0.001` |

Все starts сообщили convergence, но dropout-параметры неидентифицированы:
`a=0.1028/0.1347/0.3084`, `b=240.7/330.0/790.3`. Лучший likelihood start
дал `r=0.612536, alpha=13.134504, a=0.308351, b=790.336256`, но выбирать его
вопреки gate запрещено. Fail-fast сработал до scoring group 1; параметры не
усреднялись, gates не ослаблялись, другой family/refit не запускался.

## TEST SUPPORT STATUS

**NOT REACHED.** Из-за нестабильного обязательного BTYD fit `z_BTYD` не создан,
поэтому test correction quantiles и `Var(c_test)/Var(c_oof)` не определены.
Post-hoc rescale/tuning не выполнялись.

## SUBMISSION STATUS

**NOT CREATED (CASE C).** Ни `submission_BTYD05_FRESH1.csv`, ни
`submission_BTYD05.csv` не существуют. Leaderboard не трогался.

## Exact artifacts and hashes

```text
src/btyd05_production.py
  4be1d59b0bd9293bfa92f8c366783ac56a47cd2ca9757d8ea8fbeba8449a62ed
research/strategies/results/BTYD05_PROD_EXP050/fresh_parity_audit.json
  496b445462314ca33f512d89438816d78079bcd4775b572e33eef7bf3c6c92ce
research/strategies/results/BTYD05_PROD_EXP050/production_support.json
  7c0f934501a99e92b2e2f6041823af81baebd26eaa19b0e9d9c22d0b20a21e98
research/strategies/results/BTYD05_PROD_EXP050/summary.json
  5ca059f6fa87cefff5924d34cbeefcda41decf70122eba808b8e1fd1d3f15cac
artifacts/BTYD_DAY_BGNBD_EXP047_V2/oof_raw.npz
  754d930b2347beb400b947c416cf56cc036f0b80c35ea039402432263b89d6af
data/raw/train.parquet
  5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0
data/raw/sample_submit.csv
  06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1
submissions/submission_STRONGEST_CURRENT.csv
  abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda
```

## Вердикт и вывод

**BLOCK / CASE C.** Exact FRESH production parity отсутствует; exact BTYD
production fit независимо провалил собственные stability gates. Уже
подтверждённый OOF candidate нельзя точно и воспроизводимо превратить в test
submission без запрещённого изменения execution semantics.
