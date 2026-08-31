# TEAM_B_B2_EXP069

Честное сравнение pinned team source `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf` с `STRONGEST-CURRENT` на общей четырёхфолдовой S1 validation.

- `metrics.json` — scores, optimum/LOFO weights, OOF→TEST regime и production manifest.
- `weight_curve.csv` — wCV на сетке веса our `0.00..1.00` с шагом `0.01`.
- `artifacts/TEAM_B_B2_EXP069/` — сохранённые модели, OOF и TEST predictions (gitignored).
- `submissions/submission_TEAM_B_B2_OPTIMAL_ENSEMBLE.csv` — проверенный production-файл (gitignored).

Воспроизведение без переобучения при наличии artifacts: `python src/team_b_b2_ensemble.py`. Если artifacts отсутствуют, та же команда обучит пять комплектов моделей и сохранит их пофолдово.
