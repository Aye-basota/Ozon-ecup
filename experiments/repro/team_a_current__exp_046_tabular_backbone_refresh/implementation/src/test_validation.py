"""Проверки валидатора: метрика, калибровка, веса фолдов, состав отчёта.

Валидатор — единственный инструмент, по которому принимаются решения без сабмита.
Молчаливая ошибка здесь стоила бы не одного эксперимента, а всех сразу.

Запуск: python -m src.test_validation
"""
from __future__ import annotations

import numpy as np

from src.config import FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.report import evaluate, format_report
from src.validation import bias_z, calibrate, rmsle_z, wcv

RNG = np.random.default_rng(0)
FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]


def ok(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}", flush=True)
    assert cond, msg


def _synthetic(n_per_fold=4000, shift=0.0):
    ys, zs, cs = [], [], []
    for i, c in enumerate(FOLDS):
        y = np.expm1(np.maximum(RNG.normal(2.5, 1.2, n_per_fold), 0.0))
        z = np.maximum(np.log1p(y) + RNG.normal(shift, 0.4 + 0.05 * i, n_per_fold), 0.0)
        ys.append(y); zs.append(z); cs.append(np.full(n_per_fold, c, dtype="U10"))
    return np.concatenate(ys), np.concatenate(zs), np.concatenate(cs)


def test_calibrate_finds_the_exact_optimal_offset():
    """По сетке лучше не бывает — иначе неподвижная точка найдена неверно."""
    for shift in (0.0, 0.3, -0.3):
        y, z, _ = _synthetic(3000, shift=shift)
        d, sc = calibrate(y, z)
        grid = np.linspace(d - 0.2, d + 0.2, 401)
        best = min(rmsle_z(y, z + g) for g in grid)
        ok(sc <= best + 1e-9,
           f"shift={shift:+.1f}: скор после сдвига {sc:.6f} <= сеточного {best:.6f}")
        ok(sc <= rmsle_z(y, z) + 1e-12, f"shift={shift:+.1f}: калибровка не ухудшает скор")


def test_calibrate_equals_bias_when_clipping_does_not_bind():
    """Если после сдвига всё ещё z > 0, оптимум ровно аналитический."""
    y = np.expm1(RNG.uniform(1.5, 5.0, 3000))              # log1p(y) >= 1.5, ноль недостижим
    z = np.log1p(y) + 0.5 + RNG.normal(0, 0.2, len(y))
    d, _ = calibrate(y, z)
    ok(abs(d - bias_z(y, z)) < 1e-9, f"сдвиг {d:+.5f} == bias {bias_z(y, z):+.5f}")


def test_calibrated_score_ignores_a_pure_level_error():
    """Смысл калибровки: уровень сабмита ставится якорём, его ошибка не наша."""
    y, z, _ = _synthetic(3000)
    _, a = calibrate(y, z)
    _, b = calibrate(y, z + 0.25)
    ok(abs(a - b) < 1e-6, f"сдвиг всего прогноза на 0.25 не меняет скор ({a:.6f} vs {b:.6f})")
    ok(rmsle_z(y, z + 0.25) > rmsle_z(y, z), "некалиброванный скор от сдвига ухудшается")


def test_wcv_weights_are_normalized_and_favour_late_folds():
    ok(abs(wcv([1.0, 1.0, 1.0, 1.0]) - 1.0) < 1e-12, "веса нормированы: wCV(константа) = константа")
    early = wcv([1.0, 2.0, 2.0, 2.0])
    late = wcv([2.0, 2.0, 2.0, 1.0])
    ok(late < early, f"улучшение позднего фолда весит больше раннего ({late:.4f} < {early:.4f})")
    w = np.array(FOLD_WEIGHTS_S1)
    ok(float(w[-1] / w.sum()) > 0.5, f"на последний фолд приходится {w[-1] / w.sum():.0%} веса")
    ok(list(w) == sorted(w), "веса не убывают по времени")


def test_wcv_rejects_mismatched_number_of_folds():
    try:
        wcv([1.0, 1.0])
    except AssertionError:
        ok(True, "wCV на неполном наборе фолдов отвергнут явно")
        return
    ok(False, "wCV молча посчитался на двух фолдах из четырёх")


def test_evaluate_returns_every_metric_the_validator_promises():
    y, z, c = _synthetic()
    r = evaluate(y, z, c)
    need = ["fold_scores", "fold_cal", "fold_bias", "fold_mean_z", "wcv", "cv_mean", "cv_std",
            "oof_rmsle", "oof_bias", "oof_offset", "oof_cal", "mean_z", "per_fold", "n"]
    miss = [k for k in need if k not in r]
    ok(not miss, f"в отчёте есть все обязательные поля (нет: {miss})")
    ok(r["folds"] == FOLDS, f"фолды упорядочены по времени: {r['folds']}")
    ok(abs(r["wcv"] - wcv(r["fold_cal"])) < 1e-12, "wCV считается по КАЛИБРОВАННЫМ скорам")
    ok(abs(r["cv_mean"] - float(np.mean(r["fold_scores"]))) < 1e-12,
       "CV mean — среднее по фолдам, а не по строкам")
    ok(abs(r["mean_z"] - float(z.mean())) < 1e-9, "mean(log1p(pred)) считается по всем строкам")
    ok(all(a <= b + 1e-12 for a, b in zip(r["fold_cal"], r["fold_scores"])),
       "калиброванный скор фолда не хуже сырого")


def test_evaluate_is_invariant_to_row_order():
    """Склейка пофолдовых прогонов идёт в произвольном порядке — числа не должны плыть."""
    y, z, c = _synthetic(2000)
    p = RNG.permutation(len(y))
    a, b = evaluate(y, z, c), evaluate(y[p], z[p], c[p])
    for k in ("wcv", "oof_cal", "cv_mean", "mean_z"):
        ok(abs(a[k] - b[k]) < 1e-12, f"{k} не зависит от порядка строк")


def test_partial_fold_set_reports_no_wcv():
    """Пофолдовый прогон обязан не выдавать wCV: веса определены только на всей схеме."""
    y, z, c = _synthetic(1500)
    m = np.isin(c, FOLDS[:2])
    r = evaluate(y[m], z[m], c[m])
    ok(r["wcv"] is None and r["partial"], "на подмножестве фолдов wCV = None, partial = True")
    ok(len(r["per_fold"]) == 2, "остальные метрики при этом считаются")


def test_report_reproduces_the_published_numbers_of_the_best_submission():
    """Смесь S1-DIST-MIX обязана воспроизводить опубликованные 1.75645 / 1.74948."""
    from src.tracking import load_oof
    try:
        ds = [load_oof(e) for e in ("S1-E10", "S1-E02", "S1-E03a", "S1-DIST")]
    except FileNotFoundError:
        print("  [skip] OOF лучшей смеси не найдены — проверка пропущена")
        return
    keys = [np.char.add(np.asarray(d["cutoff"], dtype="U10"),
                        np.asarray(d["user_id"]).astype("U20")) for d in ds]
    o0 = np.argsort(keys[0])
    Z = [d["z"][np.argsort(k)] for d, k in zip(ds, keys)]
    z = np.average(np.vstack(Z), axis=0, weights=[0.15, 0.30, 0.10, 0.45])
    r = evaluate(ds[0]["y"][o0], z, np.asarray(ds[0]["cutoff"], dtype="U10")[o0])
    ok(abs(r["oof_cal"] - 1.75645) < 5e-5, f"OOF калибр. {r['oof_cal']:.5f} == 1.75645")
    ok(abs(r["wcv"] - 1.74948) < 5e-5, f"wCV {r['wcv']:.5f} == 1.74948")


def test_format_report_aligns_deltas_by_fold_date():
    """Пофолдовый прогон против полного опорного: дельты по датам, а не по позициям."""
    y, z, c = _synthetic(1200)
    full = evaluate(y, z, c)
    m = c == FOLDS[3]
    part = evaluate(y[m], z[m] - 0.3, c[m])
    txt = format_report(part, full)
    ok("лучше опорного на" in txt and "из 1" in txt,
       "сравнение идёт по одному общему фолду, а не по четырём")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"== валидатор: {len(tests)} проверок ==")
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print("\nвсе проверки пройдены")
