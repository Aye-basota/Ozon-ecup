"""Единый отчёт валидации: один вызов `evaluate` — все метрики эксперимента.

Зачем отдельный модуль. До exp_016 каждый раннер (`train.py`, `merge_oof.py`,
`blend.py`, разовые скрипты) считал свой набор чисел своим кодом, и сравнивать
эксперименты между собой можно было только вручную. Теперь метрики считаются
здесь и больше нигде, поэтому любые два прогона сопоставимы по определению.

Что считается (все семь величин из постановки задачи валидатора):
  * `fold_scores`   — RMSLE каждого фолда, как есть;
  * `wcv`           — ГЛАВНАЯ метрика: взвешенное 1:2:4:8 среднее пофолдовых
                      скоров ПОСЛЕ оптимального лог-сдвига;
  * `oof_rmsle`     — RMSLE на склеенном OOF;
  * `bias`          — mean(log1p y) − mean z, пофолдово и общий;
  * `oof_cal`       — OOF после оптимального лог-сдвига (и сам сдвиг);
  * `mean_z`        — mean(log1p(pred)), пофолдово и общий;
  * OOF-предсказания сохраняются отдельно (`tracking.save_oof`) для блендинга.

Почему главная метрика именно `wcv` — `experiments/exp_016_validator_calibration.md`:
уровень сабмита ставится по измеренному якорю, поэтому сравнивать модели нужно
по скору БЕЗ ошибки уровня; а поздние фолды ближе к тесту, и взвешивание
1:2:4:8 делает перенос локальной дельты на LB равным 1.00 ± 0.06.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, VAL_FOLDS_S1
from src.validation import bias_z, calibrate, rmsle_z, wcv

FOLDS_S1 = [d.isoformat() for d in VAL_FOLDS_S1]


def evaluate(y, z, cutoff, folds: list[str] | None = None) -> dict:
    """Полный набор метрик по OOF-предсказаниям одного эксперимента.

    `cutoff` — val-cutoff каждой строки; фолды упорядочиваются по времени, чтобы
    веса `wcv` всегда ложились на одни и те же даты независимо от порядка склейки.
    """
    y, z = np.asarray(y, float), np.asarray(z, float)
    cutoff = np.asarray(cutoff, dtype="U10")
    present = sorted(set(cutoff.tolist()))
    folds = folds or [f for f in FOLDS_S1 if f in present] + \
        [f for f in present if f not in FOLDS_S1]

    per = []
    for c in folds:
        m = cutoff == c
        assert m.any(), f"фолд {c} пуст"
        d, sc_cal = calibrate(y[m], z[m])
        per.append(dict(cutoff=c, n=int(m.sum()), rmsle=rmsle_z(y[m], z[m]),
                        bias=bias_z(y[m], z[m]), offset=d, rmsle_cal=sc_cal,
                        mean_z=float(z[m].mean()), mean_ly=float(np.log1p(y[m]).mean())))

    fs = [p["rmsle"] for p in per]
    fc = [p["rmsle_cal"] for p in per]
    d_all, oof_cal = calibrate(y, z)
    weighted = set(folds) == set(FOLDS_S1)      # веса определены только для полной схемы S1
    return dict(
        n=int(len(y)), folds=folds, fold_sizes=[p["n"] for p in per], per_fold=per,
        fold_scores=fs, fold_cal=fc, fold_bias=[p["bias"] for p in per],
        fold_mean_z=[p["mean_z"] for p in per],
        wcv=wcv(fc) if weighted else None,
        wcv_raw=wcv(fs) if weighted else None,
        cv_mean=float(np.mean(fs)), cv_std=float(np.std(fs)), cv_cal_mean=float(np.mean(fc)),
        oof_rmsle=rmsle_z(y, z), oof_bias=bias_z(y, z), oof_offset=d_all, oof_cal=oof_cal,
        mean_z=float(z.mean()), mean_ly=float(np.log1p(y).mean()),
        partial=not weighted,
    )


def format_report(r: dict, ref: dict | None = None) -> str:
    """Человекочитаемый блок; при наличии `ref` — ещё и дельты к опорному эксперименту.

    Дельты сопоставляются ПО ДАТЕ фолда: пофолдовые прогоны считают подмножество
    схемы, и выравнивание по позиции сравнивало бы разные фолды.
    """
    rf = {p["cutoff"]: p["rmsle_cal"] for p in ref["per_fold"]} if ref else {}
    L = ["", f"{'фолд':>12} {'n':>9} {'RMSLE':>9} {'bias':>8} {'сдвиг':>7} "
             f"{'после сдвига':>13} {'mean z':>8}" + ("   дельта" if ref else "")]
    wins = 0
    for p in r["per_fold"]:
        d = ""
        if p["cutoff"] in rf:
            d = f"   {p['rmsle_cal'] - rf[p['cutoff']]:+.5f}"
            wins += p["rmsle_cal"] < rf[p["cutoff"]]
        L.append(f"{p['cutoff']:>12} {p['n']:>9,} {p['rmsle']:>9.5f} {p['bias']:>+8.4f} "
                 f"{p['offset']:>+7.3f} {p['rmsle_cal']:>13.5f} {p['mean_z']:>8.4f}{d}")
    w = f"{r['wcv']:.5f}" if r["wcv"] is not None else "н/д (неполная схема)"
    dw = (f"   дельта {r['wcv'] - ref['wcv']:+.5f}"
          if ref and r["wcv"] is not None and ref.get("wcv") is not None else "")
    L.append(f"\n  wCV (1:2:4:8, калибр.) = {w}{dw}")
    L.append(f"  CV mean={r['cv_mean']:.5f} std={r['cv_std']:.5f} | калибр. CV={r['cv_cal_mean']:.5f}")
    L.append(f"  OOF={r['oof_rmsle']:.5f} bias={r['oof_bias']:+.4f} | "
             f"после сдвига {r['oof_offset']:+.3f}: {r['oof_cal']:.5f}")
    L.append(f"  mean(log1p(pred))={r['mean_z']:.4f}  против таргета {r['mean_ly']:.4f}")
    if ref:
        n = sum(p["cutoff"] in rf for p in r["per_fold"])
        L.append(f"  лучше опорного на {wins} фолдах из {n}")
    return "\n".join(L)


def save_report(exp_id: str, r: dict, extra: dict | None = None) -> Path:
    p = ARTIFACTS / f"report_{exp_id}.json"
    p.write_text(json.dumps({**r, **(extra or {})}, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p


def load_report(exp_id: str) -> dict:
    return json.loads((ARTIFACTS / f"report_{exp_id}.json").read_text(encoding="utf-8"))


def from_oof(exp_id: str) -> dict:
    """Метрики уже сохранённого эксперимента — без переобучения."""
    from src.tracking import load_oof
    d = load_oof(exp_id)
    return evaluate(d["y"], d["z"], d["cutoff"])
