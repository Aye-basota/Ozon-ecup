"""Единая схема валидации и метрика.

Out-of-time cutoff-схема. Никаких случайных сплитов: одна строка = один
пользователь на одном cutoff'е, и порядок во времени обязателен.

Три сценария (research/strategy_1.md §3.2):
  S1  rolling out-of-time — 4 фолда в конце чистого коридора, train: T <= V-30;
  S2  max-gap            — train до 2025-06-15, val 2025-10-16 (разрыв 123 дня
                           ~ тестовому разрыву 120 дней); измеряет дрейф;
  S3  сезонный           — val 2025-02-13 (календарный аналог теста), 1-блочная
                           панель; используется ТОЛЬКО для оценки поправки.

**Главная метрика проекта — `wcv`** (exp_016): взвешенное среднее пофолдовых
RMSLE *после оптимального лог-сдвига*, веса `FOLD_WEIGHTS_S1 = 1:2:4:8`.
Калибровка обязательна, потому что уровень сабмита ставится по измеренному на LB
якорю: ошибка уровня на тесте равна нулю по построению и не должна участвовать в
сравнении моделей. Веса — потому что поздние фолды ближе к тесту и коэффициент
переноса на LB у них устойчивее. Полный набор метрик собирает `src/report.py`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from src.config import (CORRIDOR_END, CUTOFF_STEP, FOLD_WEIGHTS_S1, HISTORY_L, S2_TRAIN_END,
                        S2_VAL, S3_VAL, TARGET_DAYS, VAL_FOLDS_S1, cutoff_grid)


# --------------------------------------------------------------------------- метрика
def rmsle(y_true, y_pred) -> float:
    """Метрика соревнования: RMSLE с log1p и клиппингом отрицательных предсказаний."""
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(np.maximum(y_pred, 0.0))) ** 2)))


def rmsle_z(y_true, z_pred) -> float:
    """То же, но предсказание уже в лог-пространстве z = log1p(pred)."""
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.maximum(z_pred, 0.0)) ** 2)))


def bias_z(y_true, z_pred) -> float:
    """mean(log1p(y)) - mean(z). Отрицательный = модель перепрогнозирует."""
    return float(np.log1p(y_true).mean() - np.asarray(z_pred).mean())


def best_offset(y_true, z_pred, lo: float = -0.6, hi: float = 0.6, n: int = 241):
    """Оптимальный глобальный сдвиг в лог-пространстве и скор после него (по сетке)."""
    ly = np.log1p(y_true)
    grid = np.linspace(lo, hi, n)
    sc = np.array([np.sqrt(np.mean((ly - np.maximum(z_pred + d, 0.0)) ** 2)) for d in grid])
    i = int(sc.argmin())
    return float(grid[i]), float(sc[i])


def calibrate(y_true, z_pred, iters: int = 25):
    """Оптимальный лог-сдвиг и скор после него: (delta, RMSLE).

    Без клиппинга оптимум был бы ровно `d = mean(ly - z) = bias`. Но метрика
    клипует `z + d` нулём, и строки, ушедшие под ноль, перестают зависеть от `d`:

        MSE(d) = mean_{z+d>0} (ly - z - d)^2 + mean_{z+d<=0} ly^2
        dMSE/dd = 0   <=>   d = mean(ly - z) ПО АКТИВНОМУ МНОЖЕСТВУ {z + d > 0}

    Это неподвижная точка, и итерация от `bias` сходится за единицы шагов.
    На боевых прогонах поправка нулевая (предсказания уже неотрицательны, сдвиги
    порядка 0.05), но на сильно смещённых прогнозах сетка из `best_offset` и
    наивный `bias` расходятся с истинным минимумом в 4-м знаке.
    """
    ly = np.log1p(y_true)
    z = np.asarray(z_pred, dtype=float)
    d = float((ly - z).mean())
    for _ in range(iters):
        act = z + d > 0
        if not act.any():
            break
        d_new = float((ly[act] - z[act]).mean())
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    return d, rmsle_z(y_true, z + d)


def wcv(fold_scores, weights=None) -> float:
    """Главная метрика: взвешенное среднее пофолдовых скоров, веса 1:2:4:8.

    Порядок `fold_scores` обязан совпадать с `VAL_FOLDS_S1` (от раннего к позднему).
    """
    s = np.asarray(fold_scores, float)
    w = np.asarray(FOLD_WEIGHTS_S1 if weights is None else weights, float)
    # усечение весов под неполный набор фолдов дало бы правдоподобное, но чужое число
    assert len(w) == len(s), (
        f"скоров {len(s)}, весов {len(w)}: wCV определён только на полной схеме S1")
    return float(np.dot(w, s) / w.sum())


# --------------------------------------------------------------------------- фолды
def get_folds(min_history: int = HISTORY_L, step: int = CUTOFF_STEP,
              vals: list[dt.date] | None = None) -> list[tuple[list[dt.date], dt.date]]:
    """S1: список (train_cutoffs, val_cutoff).

    Train-cutoff допускается только если его target-окно полностью в прошлом
    относительно val-cutoff'а: T + TARGET_DAYS <= V. Это исключает пересечение
    обучающего таргета с признаковым окном валидации.
    """
    vals = vals or VAL_FOLDS_S1
    grid = cutoff_grid(min_history, step)
    out = []
    for V in vals:
        tr = [T for T in grid if T + dt.timedelta(days=TARGET_DAYS) <= V]
        out.append((tr, V))
    return out


def gap_curve_folds(min_history: int = HISTORY_L, step: int = CUTOFF_STEP,
                    val: dt.date = S2_VAL, n_train: int = 4):
    """S2: одна val-точка, несколько train-блоков на растущем удалении от неё.

    Даёт зависимость bias(gap). Реальный разрыв «последний чистый cutoff -> тест»
    равен 120 дням и локально недостижим, поэтому bias на нём экстраполируется
    по этой кривой (research/strategy_1.md §10, Эксперимент 5).
    """
    grid = [T for T in cutoff_grid(min_history, step) if T + dt.timedelta(days=TARGET_DAYS) <= val]
    out = []
    for end_i in range(n_train - 1, len(grid)):
        tr = grid[max(0, end_i - n_train + 1):end_i + 1]
        out.append((tr, val, (val - tr[-1]).days))
    return out


def s3_fold(step: int = CUTOFF_STEP):
    """Сезонный фолд: 1-блочная панель, val = 2025-02-13 (YoY-аналог теста).

    Train — cutoff'ы весны-лета с той же (усечённой) глубиной истории, что и у val.
    Возвращает (train_cutoffs, val_cutoff, L_eff): на 2025-02-13 доступно всего
    43 дня истории, поэтому L для этого сценария принудительно мал.
    """
    L_eff = 43
    grid = [dt.date(2025, 4, 15) + dt.timedelta(days=step * k) for k in range(14)]
    grid = [T for T in grid if T <= CORRIDOR_END]
    return grid, S3_VAL, L_eff


def describe_folds(folds) -> str:
    rows = []
    for tr, V in folds:
        gap = (V - max(tr)).days
        rows.append(f"  val {V}  train {min(tr)}..{max(tr)} ({len(tr)} cutoffs, gap {gap}d)")
    return "\n".join(rows)
