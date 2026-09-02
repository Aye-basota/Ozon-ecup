# exp_021 — submission-level blends around exp019

- **Дата:** 2026-08-26
- **Автор:** B1
- **Коммит:** рабочее дерево

## Гипотеза

Готовые сабмиты `exp019`, `exp018`, `exp017`, `exp020` и `exp011` очень близки, но
часть моделей построена разными способами: behavior_v1 dist-head, CatBoost blend и
старый dense ensemble. Небольшое усреднение в `log1p` вокруг лучшего `exp019` может
снять часть дисперсии предсказаний без переобучения и без заметного сдвига уровня.

## Что изменено относительно базы

Добавлен только post-hoc blend уже готовых submission CSV в log-space; модели и фичи
не переобучались.

## Результат

- CV по фолдам: не считали, это submission-level эксперимент без OOF.
- CV mean: нет; лучший на момент — `exp_019`, mean RMSLE 1.707699.
- LB: не отправляли.

Сгенерированные кандидаты:

| файл | веса | mean log1p |
|------|------|------------|
| `exp_021_blend_e19_e18_9010.csv` | 0.90 exp019 + 0.10 exp018 | 2.369862 |
| `exp_021_blend_e19_e18_9010_level_e19.csv` | 0.90 exp019 + 0.10 exp018, level exp019 | 2.370966 |
| `exp_021_blend_e19_e18_8020.csv` | 0.80 exp019 + 0.20 exp018 | 2.368759 |
| `exp_021_blend_e19_e18_8020_level_e19.csv` | 0.80 exp019 + 0.20 exp018, level exp019 | 2.370966 |
| `exp_021_blend_e19_e11_9505.csv` | 0.95 exp019 + 0.05 exp011 | 2.369791 |
| `exp_021_blend_e19_e18_e11_851005.csv` | 0.85 exp019 + 0.10 exp018 + 0.05 exp011 | 2.368687 |
| `exp_021_blend_e19_e18_e11_851005_level_e19.csv` | 0.85 exp019 + 0.10 exp018 + 0.05 exp011, level exp019 | 2.370966 |

Диагностика разнообразия на test predictions:

- `Var(log exp019 - log exp018) = 0.002834`
- `Var(log exp019 - log exp011) = 0.004730`
- Корреляции log-предсказаний все выше 0.999, поэтому веса взяты маленькие и
консервативные.

## Вердикт и вывод

**SUBMIT-CANDIDATES.** Основной кандидат на один LB submit — `exp_021_blend_e19_e18_9010_level_e19.csv`:
он ближе всех к текущему champion `exp019`, добавляет 10% CatBoost-компоненты и
сохраняет уровень `mean(log1p)` как у exp019.
Если первый кандидат улучшит LB, можно пробовать более сильный `8020`; если ухудшит,
направление post-hoc blends вокруг этих файлов, вероятно, слабое.

## Конфиг прогона

```text
python src/submission_blend.py stats --files \
  exp_017_dist_post_order_wrec050_scale120.csv \
  exp_018_catboost_blend_wcat020_scale120.csv \
  exp_019_behavior_v1_dist_wrec050_scale120.csv \
  exp_020_behavior_v1_slim_dist_wrec050_scale120.csv \
  exp_011_dense8_logens_scale120.csv

python src/submission_blend.py blend --files exp019 exp018 --weights 0.9 0.1 --out exp_021_blend_e19_e18_9010.csv
python src/submission_blend.py blend --files exp019 exp018 --weights 0.8 0.2 --out exp_021_blend_e19_e18_8020.csv
python src/submission_blend.py blend --files exp019 exp011 --weights 0.95 0.05 --out exp_021_blend_e19_e11_9505.csv
python src/submission_blend.py blend --files exp019 exp018 exp011 --weights 0.85 0.10 0.05 --out exp_021_blend_e19_e18_e11_851005.csv
```
