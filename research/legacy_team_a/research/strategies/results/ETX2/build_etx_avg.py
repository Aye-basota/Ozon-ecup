"""ETX-AVG3: усреднение сидов 42/43/44 `ETX-01` В ЛОГ-ПРОСТРАНСТВЕ (EXP-037).

Ровно та же операция, что `MIX9/build_d3a_avg3.py` делает для `SEQ-D3A`, и это
принципиально: `SEQ-AVG3`, с которым сравнивается результат, собран этой же
функцией, поэтому разница объектов не может прийти из способа усреднения.

Зачем усреднение. `exp_036` оставил ETX за бортом сабмита с измеренным блокером:
`Var(z_ETX − z_SEQ-AVG3)` на тестовом cutoff'е 0.08844 против 0.02743 на OOF
(3.22x), тогда как все пары без ETX лежат в 0.63–1.11x. Там же померено, что
разброс ДВУХ прогонов ETX (`Var = 0.0376`) сопоставим со всем провалидированным
расхождением ETX↔TCN (0.0274) — то есть значительная часть аномалии может быть
шумом одиночной модели, а не режимом. Три сида отвечают на это прямо.

Сиды 43/44 считались на A10 (`--compile`), сид 42 — локально eager (`exp_036`).
Расхождение машин проект уже мерил (`exp_030c`: 0.0010–0.0015 wCV), и оно же
сидит в `SEQ-AVG3` (сид 42 локальный, 43/44 арендованные) — обе стороны
сравнения устроены одинаково.

Запуск: PYTHONPATH=. python research/strategies/results/ETX2/build_etx_avg.py
"""
from __future__ import annotations

import sys

import numpy as np

from src.config import ARTIFACTS, VAL_FOLDS_S1
from src.merge_oof import auc_positive, load_parts, merge_arrays
from src.report import format_report, save_report
from src.tracking import load_oof, save_oof

VARIANTS = {
    "ETX-AVG3": ["ETX-01-S42", "ETX-01-S43", "ETX-01-S44"],
    "ETX-AVG2": ["ETX-01-S42", "ETX-01-S43"],      # промежуточный контроль
}


def tags() -> list[str]:
    return [f"V{d.strftime('%m%d')}" for d in VAL_FOLDS_S1]


def have(part: str) -> bool:
    return (ARTIFACTS / f"oof_{part}.npz").exists()


def build(out: str, src: list[str]) -> bool:
    miss = [f"{p}-{t}" for p in src for t in tags() if not have(f"{p}-{t}")]
    if miss:
        print(f"{out}: НЕТ {len(miss)} частей, пропуск ({miss[0]} ...)")
        return False
    for t in tags():
        ds = [load_oof(f"{p}-{t}") for p in src]
        u0 = np.asarray(ds[0]["user_id"])
        for d in ds[1:]:
            assert np.array_equal(np.asarray(d["user_id"]), u0), f"{t}: разные строки"
            assert np.allclose(d["y"], ds[0]["y"]), f"{t}: разные таргеты"
        z = np.mean([np.asarray(d["z"], float) for d in ds], axis=0)
        save_oof(f"{out}-{t}", u0, ds[0]["cutoff"], z, ds[0]["y"])
        spread = float(np.mean([np.var(np.asarray(d["z"], float) - z) for d in ds]))
        pair = [float(np.var(np.asarray(ds[i]["z"], float) - np.asarray(ds[j]["z"], float)))
                for i in range(len(ds)) for j in range(i + 1, len(ds))]
        print(f"  {t}: {len(ds)} сидов, Var(z_i − z_avg) = {spread:.5f}, "
              f"попарно " + " ".join(f"{v:.5f}" for v in pair))
    parts = [f"{out}-{t}" for t in tags()]
    uid, cut, z, y = load_parts(parts)
    rep = merge_arrays(uid, cut, z, y)
    print(format_report(rep))
    print(f"  AUC(1[y>0]) = {auc_positive(y, z):.5f}")
    save_oof(out, uid, cut, z, y)
    save_report(out, rep, extra=dict(description="EXP-037: log-space avg сидов ETX-01",
                                     parts=parts, seeds=src))
    return True


def merge_seed(name: str) -> None:
    if any(not have(f"{name}-{t}") for t in tags()):
        print(f"{name}: не все фолды посчитаны, пропуск")
        return
    uid, cut, z, y = load_parts([f"{name}-{t}" for t in tags()])
    rep = merge_arrays(uid, cut, z, y)
    save_oof(name, uid, cut, z, y)
    save_report(name, rep, extra=dict(description="EXP-037: склейка 4 фолдов",
                                      parts=[f"{name}-{t}" for t in tags()]))
    print(f"{name}: wCV={rep['wcv']:.5f}  AUC={auc_positive(y, z):.5f}")


def main() -> None:
    todo = sys.argv[1:] or ["ETX-01-S43", "ETX-01-S44", "ETX-AVG2", "ETX-AVG3"]
    for name in todo:
        if name in VARIANTS:
            print(f"\n===== {name} =====")
            build(name, VARIANTS[name])
        else:
            merge_seed(name)


if __name__ == "__main__":
    main()
