# Team-B handoff model

Нужные файлы для интеграции:

```text
src/features.py
src/train.py
src/predict.py
requirements.txt
```

## Что это за модель

Финальная tabular-модель Team-B — ensemble из 5 компонентов:

```text
0.25 recency LightGBM regressor
0.10 post_order LightGBM dist-head
0.20 behavior_v1 LightGBM dist-head
0.25 behavior_v1 XGBoost regressor
0.20 behavior_v1 CatBoost regressor
```

Все компоненты предсказывают `z = log1p(gmv_next_30d)`. Финальный blend делается
в `log1p`-пространстве, затем применяется level alignment к `mean(log1p)=2.370966`.

## Python API

```python
import pandas as pd

from src.train import train_models
from src.predict import predict_log, predict_gmv

models, meta = train_models()

z_pred = predict_log(models, meta)
gmv_pred = predict_gmv(models, meta)
```

`train_models()` принимает аргументы `df_raw`, `cutoff_date`, `target_days` для
совместимости с интерфейсом `team-b-B2`, но текущий пайплайн читает данные из
`data/raw/train.parquet` внутри `src/features.py`.

## CLI

```bash
python src/predict.py --handoff --output exp_024_handoff_level_e19.csv
```

Эта команда обучает все компоненты на последних 8 clean weekly cutoff-ах
`2025-08-28..2025-10-16` и пишет submit в `submissions/`.

## Готовый test prediction

Если нужен только готовый CSV для blend с DL-моделью:

```text
submissions/exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_level_e19.csv
```

## Локальная метрика

```text
exp019 behavior_v1 dist:      1.707699
exp023 + XGBoost:             1.707119
exp024 + XGBoost + CatBoost:  1.706955
```

LB для exp024 пока pending.
