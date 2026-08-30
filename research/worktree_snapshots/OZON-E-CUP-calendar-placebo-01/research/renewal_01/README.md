# RENEWAL-01 — итоговый отчёт

Полная experiment card и численные выводы: [`experiments/exp_027_next_purchase_clock.md`](../../experiments/exp_027_next_purchase_clock.md).

## Короткий ответ

1. **Вероятность покупки:** нет; AUC 0.84106 против 0.84552 у existing `b30_p`,
   хуже 4/4 и во всех cold/regularity/recency-normalized сегментах.
2. **Standalone RMSLE:** нет; two-part Clock 1.75969 против production 1.74834.
3. **Ансамбль:** нет по правилам проекта; лучший LOFO −0.000416 при gate −0.0005,
   meta −0.000266, собственный вклад сверх SELF −0.000111.
4. **Ортогональность:** высокая, но неполезная: Var=0.06173 = 8.67× seed floor,
   corr остатков 0.99007.
5. **Где работает:** слабый ensemble-эффект распределён по 4 folds; по
   классификации Clock не выигрывает ни одной проверенной группы. Replacement
   чуть сильнее у history=1, regular и normalized recency 0.5–1.5, но абсолютная
   дельта RMSLE не превышает 0.00075 и не образует отдельной победившей страты.
6. **Вердикт:** `STOP`; submission не готовился.
7. **Следующий эксперимент — ровно один:** `SEQ-POS-01`, dense positional
   supervision внутри causal sequence-encoder после AVGSEQ3, без renewal/hazard
   auxiliary heads.

## Навигация по артефактам

| Артефакт | Содержание |
|---|---|
| `artifacts/oof_RENEWAL-01.npz` | OOF `p_clock_30`, R0/R1, labels, base/amount/meta, segment clocks |
| `artifacts/test_RENEWAL-01.npz` | test probabilities и confidence clocks, порядок sample_submit |
| `artifacts/report_RENEWAL-01.json` | единый project RMSLE report + summary |
| `artifacts/renewal_01_metrics.json` | итоговые classification/RMSLE/diversity числа |
| `artifacts/renewal_01_fold_classification.csv` | ROC/PR/logloss/Brier/precision/recall по fold |
| `artifacts/renewal_01_calibration_bins.csv` | 10-bin calibration по fold и OOF |
| `artifacts/renewal_01_fold_rmsle.csv` | RMSLE всех режимов по fold |
| `artifacts/renewal_01_segments.csv` | cold start, regularity, dormancy, frequency, recency bands |
| `artifacts/renewal_01_{prediction,residual}_correlations.csv` | correlation matrices |
| `artifacts/renewal_01_replacement_grid.csv` | small replacement grids при CAP=0.10 |
| `artifacts/renewal_01_lofo.csv` | held-out fold weights и deltas |
| `artifacts/renewal_01_meta.csv` | alpha sensitivity и SELF-control |
| `artifacts/renewal_01_sensitivity.csv` | R0 shrinkage и R1 seed sensitivity |
| `artifacts/renewal_01_clock_profile.csv` | доли gaps около 7/14/30/60/90 |

## Воспроизведение

```powershell
$env:LGB_THREADS='3'
python src/renewal.py --baseline-artifacts artifacts
python -m pytest src/test_renewal.py -q
```

Раннер поддерживает `--stage cache|cv|test|evaluate` и resume по сохранённым
fold-моделям. Targeted suite: 9/9 passed. Полный suite: 93 passed и один
pre-existing fail в `test_calval.py` на cutoff `2025-08-08`; Clock-файлы к нему
не относятся. Общие `src/config.py` и `src/validation.py` не изменялись.
