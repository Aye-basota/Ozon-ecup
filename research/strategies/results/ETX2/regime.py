"""ETX2 — гейт РЕЖИМА на тестовом cutoff'е (EXP-037, этап 4).

Обязательная проверка перед сборкой любого сабмита с ETX. Логика взята из
`exp_036` и расширена: там мерилось расхождение ПАРЫ ЧЛЕНОВ, здесь — ещё и
расхождение готового КАНДИДАТА от отправленного чемпиона, то есть ровно того
объекта, который поедет на LB.

Что меряется и почему именно так:

* `Var(z_A - z_B)` на OOF и на тесте и их ОТНОШЕНИЕ. Тестовых меток нет, качество
  на тесте не измеримо; но если две модели на тестовой панели расходятся в разы
  сильнее, чем в любой точке, где эта разница валидировалась, значит валидация
  ничего не говорит о том, что произойдёт на LB. Все пары БЕЗ ETX в `exp_036`
  лежали в 0.63-1.11x, все пары С ETX — в 2.3-3.7x;
* то же по полосам АКТИВНОСТИ (число событийных дней в окне 289): в `exp_036`
  отношение росло монотонно 0.48x -> 5.70x, максимум там, где масса GMV;
* моменты и хвосты `dz`: mean/std/квантили, доля |dz| > 0.05 / 0.10 / 0.20,
  отдельно на самых активных. Дисперсия одна не различает «равномерно чуть шире»
  и «взрыв на 5% пользователей», а это разные риски.

Метки теста не используются нигде: их нет. Это диагностика сдвига распределения,
а не отбор — веса выбираются только LOFO (`lofo2.py`, `segblend.py`).

Запуск: PYTHONPATH=. python research/strategies/results/ETX2/regime.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, "research/strategies/results/ETX2")
import common  # noqa: E402

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, VAL_FOLDS_S1  # noqa: E402
from src.tracking import load_oof  # noqa: E402

OUT = "research/strategies/results/ETX2"

# OOF-имя -> список ztest-имён, лог-среднее которых и есть тестовая сторона члена.
# `ETX2_DC=1` — тестовая сторона ETX берётся с СОГЛАСОВАННОЙ статик-глубиной
# (`ztest_*-DC.npy`, `depth_fix.py`), `ETX2_DC=0` — как в `exp_036` (статик 365).
# "DCW" (боевая) — глубина 289 + dow четверг; "1"/"DC" — только глубина;
# "0" — как в `exp_036`, сырой статик 365.
_M = {"0": "", "1": "-DC", "DC": "-DC", "DCW": "-DCW"}
SFX = _M[os.environ.get("ETX2_DC", "DCW")]
TEST_OF = {
    "S1-E10": ["S1-NORM"], "S1-E02": ["S1-UNC"], "S1-E03a": ["S1-CAP"],
    "S1-DIST": ["S1-DIST"], "SEQ-01-S42": ["SEQ-01"],
    "SEQ-AVG3": ["SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"],
    "ETX-01-S42": [f"ETX-01-S42{SFX}"], "ETX-01-S43": [f"ETX-01-S43{SFX}"],
    "ETX-01-S44": [f"ETX-01-S44{SFX}"],
    "ETX-AVG2": [f"ETX-01-S4{i}{SFX}" for i in (2, 3)],
    "ETX-AVG3": [f"ETX-01-S4{i}{SFX}" for i in (2, 3, 4)],
}
CHAMP = {"S1-E10": 0.15, "S1-E02": 0.20, "S1-E03a": 0.10, "S1-DIST": 0.25,
         "SEQ-01-S42": 0.30}                      # отправленный SEQ-01-MIX, LB 1.6501764


def have_test(name: str) -> bool:
    return all((ARTIFACTS / f"ztest_{t}.npy").exists() for t in TEST_OF.get(name, []))


class Side:
    """Пара «OOF / тест» для произвольной линейной комбинации членов в лог-пространстве."""

    def __init__(self, names):
        self.oof = {}
        d0 = load_oof(names[0])
        k0 = np.char.add(np.asarray(d0["cutoff"], "U10"),
                         np.asarray(d0["user_id"]).astype("U20"))
        o0 = np.argsort(k0)
        self.cut = np.asarray(d0["cutoff"], "U10")[o0]
        self.uid = np.asarray(d0["user_id"])[o0]
        self.y = np.asarray(d0["y"], float)[o0]
        for n in names:
            d = load_oof(n)
            k = np.char.add(np.asarray(d["cutoff"], "U10"),
                            np.asarray(d["user_id"]).astype("U20"))
            o = np.argsort(k)
            assert np.array_equal(k[o], k0[o0]), f"{n}: другой набор строк OOF"
            self.oof[n] = np.asarray(d["z"], float)[o]
        self.tuid = common.test_uid()
        self.test = {}
        for n in names:
            if have_test(n):
                self.test[n] = np.mean([common.ztest(t) for t in TEST_OF[n]], axis=0)

    def mix(self, w: dict[str, float], where: str) -> np.ndarray | None:
        src = self.oof if where == "oof" else self.test
        if any(n not in src for n in w):
            return None
        acc = np.zeros(len(src[next(iter(w))]))
        for n, v in w.items():
            acc += v * src[n]
        return acc / sum(w.values())


def tail_stats(d: np.ndarray) -> dict:
    q = np.quantile(d, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return dict(var=float(np.var(d)), mean=float(d.mean()), std=float(d.std()),
                q01=float(q[0]), q05=float(q[1]), q25=float(q[2]), q50=float(q[3]),
                q75=float(q[4]), q95=float(q[5]), q99=float(q[6]),
                p05=float((np.abs(d) > 0.05).mean()), p10=float((np.abs(d) > 0.10).mean()),
                p20=float((np.abs(d) > 0.20).mean()))


def main() -> None:
    members = [n for n in TEST_OF if (ARTIFACTS / f"oof_{n}.npz").exists()]
    S = Side(members)
    print(f"OOF {len(S.y):,} строк, тест {len(S.tuid):,}; "
          f"члены с тестовой стороной: {sorted(S.test)}")

    # активность: число событийных дней в окне 289 — одно определение на обеих сторонах
    a_test = common.act_bin(common.event_counts(common.TEST_T, S.tuid))
    a_oof = np.empty(len(S.uid), np.int64)
    for c in sorted(set(S.cut.tolist())):
        m = S.cut == c
        a_oof[m] = common.act_bin(common.event_counts(dt.date.fromisoformat(c), S.uid[m]))
    f1016 = S.cut == VAL_FOLDS_S1[-1].isoformat()

    # ------------------------------------------------------------------ пары членов
    pairs = [("SEQ-01-S42", "SEQ-AVG3", "TCN vs TCN-avg3"),
             ("S1-E02", "S1-DIST", "таблица vs таблица"),
             ("SEQ-01-S42", "S1-DIST", "TCN vs таблица"),
             ("ETX-01-S42", "SEQ-AVG3", "ETX одиночный vs TCN-avg3 (exp_036: 3.22x)"),
             ("ETX-AVG2", "SEQ-AVG3", "ETX-AVG2 vs TCN-avg3"),
             ("ETX-AVG3", "SEQ-AVG3", "ETX-AVG3 vs TCN-avg3 — ЧЛЕН СЛОТА"),
             ("ETX-AVG3", "S1-DIST", "ETX-AVG3 vs таблица"),
             ("ETX-01-S43", "ETX-01-S42", "два сида ETX"),
             ("ETX-01-S44", "ETX-01-S42", "два сида ETX"),
             ("ETX-01-S44", "ETX-01-S43", "два сида ETX")]
    rows = []
    print(f"\n{'пара':<46}{'OOF':>9}{'тест':>9}{'отн':>7}{'10-16':>9}{'отн':>7}")
    for a, b, kind in pairs:
        if a not in S.oof or b not in S.oof or a not in S.test or b not in S.test:
            continue
        do, dt_ = S.oof[a] - S.oof[b], S.test[a] - S.test[b]
        vo, vt = float(np.var(do)), float(np.var(dt_))
        vf = float(np.var(do[f1016]))
        print(f"{a + ' - ' + b:<46}{vo:>9.5f}{vt:>9.5f}{vt / vo:>7.2f}{vf:>9.5f}"
              f"{vt / vf:>7.2f}")
        rows.append(dict(pair=f"{a} - {b}", kind=kind, oof=vo, test=vt,
                         ratio=vt / vo, oof_1016=vf, ratio_1016=vt / vf,
                         **{f"t_{k}": v for k, v in tail_stats(dt_).items()},
                         **{f"o_{k}": v for k, v in tail_stats(do).items()}))

    # ---------------------------------------------------------- кандидаты как смеси
    slot = {}
    for name, w in [("SEQ-AVG3", {"SEQ-AVG3": 1.0}),
                    ("0.5ETX-AVG3+0.5SEQ-AVG3", {"ETX-AVG3": 0.5, "SEQ-AVG3": 0.5}),
                    ("0.25ETX-AVG3+0.75SEQ-AVG3", {"ETX-AVG3": 0.25, "SEQ-AVG3": 0.75}),
                    ("0.5ETX-S42+0.5SEQ-AVG3", {"ETX-01-S42": 0.5, "SEQ-AVG3": 0.5})]:
        slot[name] = w
    cands = {}
    for name, w in slot.items():
        full = {"S1-E03a": 0.10, "S1-E02": 0.20, "S1-DIST": 0.25}
        for k, v in w.items():
            full[k] = full.get(k, 0.0) + 0.45 * v
        cands[f"CAP+UNC+DIST + слот({name})"] = full

    champ_o, champ_t = S.mix(CHAMP, "oof"), S.mix(CHAMP, "test")
    print(f"\n{'кандидат против отправленного SEQ-01-MIX':<44}{'OOF':>9}{'тест':>9}"
          f"{'отн':>7}{'|d|>.05':>9}{'|d|>.10':>9}{'|d|>.20':>9}")
    crows = []
    for name, w in cands.items():
        zo, zt = S.mix(w, "oof"), S.mix(w, "test")
        if zo is None or zt is None:
            continue
        do, dt_ = zo - champ_o, zt - champ_t
        ts, os_ = tail_stats(dt_), tail_stats(do)
        print(f"{name:<44}{os_['var']:>9.5f}{ts['var']:>9.5f}"
              f"{ts['var'] / os_['var']:>7.2f}{ts['p05']:>9.3%}{ts['p10']:>9.3%}"
              f"{ts['p20']:>9.3%}")
        by = []
        for bi, bn in enumerate(common.ACT_NAMES):
            mo, mt = a_oof == bi, a_test == bi
            if mt.sum() < 100 or mo.sum() < 100:
                by.append(dict(bin=bn, share_test=float(mt.mean()), oof=None, test=None,
                               ratio=None))
                continue
            vo, vt_ = float(np.var(do[mo])), float(np.var(dt_[mt]))
            by.append(dict(bin=bn, share_test=float(mt.mean()), oof=vo, test=vt_,
                           ratio=vt_ / vo if vo > 0 else None))
        crows.append(dict(candidate=name, weights=w, oof=os_, test=ts,
                          ratio=ts["var"] / os_["var"], by_activity=by,
                          cap_weight=w.get("S1-E03a", 0.0),
                          mean_z_test=float(zt.mean()), mean_z_oof=float(zo.mean())))

    print("\nпо полосам активности (Var(dz) тест / OOF), доля теста в скобках:")
    hdr = "".join(f"{b:>12}" for b in common.ACT_NAMES)
    print(f"{'кандидат':<44}{hdr}")
    for r in crows:
        line = "".join(f"{(f'{b['ratio']:.2f}' if b['ratio'] else '-'):>12}"
                       for b in r["by_activity"])
        print(f"{r['candidate']:<44}{line}")
    print(f"{'доля теста':<44}"
          + "".join(f"{b['share_test']:>12.3f}" for b in crows[0]["by_activity"]))

    # то же для ПАРЫ членов слота, как в exp_036
    print("\nпо полосам активности для пары членов (как exp_036):")
    for a, b, _ in pairs:
        if a not in S.test or b not in S.test or not a.startswith("ETX"):
            continue
        do, dt_ = S.oof[a] - S.oof[b], S.test[a] - S.test[b]
        line = []
        for bi in range(len(common.ACT_NAMES)):
            mo, mt = a_oof == bi, a_test == bi
            if mt.sum() < 100 or mo.sum() < 100:
                line.append("-")
                continue
            line.append(f"{np.var(dt_[mt]) / np.var(do[mo]):.2f}")
        print(f"  {a + ' - ' + b:<44}" + "".join(f"{v:>12}" for v in line))

    with open(f"{OUT}/regime_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "kind", "oof", "test", "ratio", "oof_1016", "ratio_1016",
                    "test_mean", "test_std", "test_p05", "test_p10", "test_p20"])
        for r in rows:
            w.writerow([r["pair"], r["kind"], f"{r['oof']:.5f}", f"{r['test']:.5f}",
                        f"{r['ratio']:.2f}", f"{r['oof_1016']:.5f}",
                        f"{r['ratio_1016']:.2f}", f"{r['t_mean']:+.5f}",
                        f"{r['t_std']:.5f}", f"{r['t_p05']:.4f}", f"{r['t_p10']:.4f}",
                        f"{r['t_p20']:.4f}"])
    (ARTIFACTS / "ETX2_regime.json").write_text(
        json.dumps(dict(pairs=rows, candidates=crows, act_names=common.ACT_NAMES),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаписано: {OUT}/regime_pairs.csv, artifacts/ETX2_regime.json")


if __name__ == "__main__":
    main()
