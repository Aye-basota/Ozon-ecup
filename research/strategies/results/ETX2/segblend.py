"""ETX2 — сегментный гейт ETX/TCN в слоте SEQ (EXP-037, этап 3).

`exp_036` показал, что ETX и TCN сильны в РАЗНЫХ частях популяции: событийное
представление выигрывает на краях активности (никогда не покупал: dAUC +0.0276;
`w180_days_buy` 16+: -0.00708 против -0.00619 у трёхсидового TCN), а в середине
(`w180_days_buy` 2-15, `rec_buy` 15-60 — там 35.3% MSE) сильнее остаётся TCN.
Глобальная доля 0.5 — компромисс между этими режимами. Вопрос этапа: покупает ли
что-нибудь ЯВНОЕ разделение, и покупает ли оно больше, чем стоит по степеням
свободы.

Ограничения приняты ДО замера и не двигались:

* гейт-признаки — только уже существующие cutoff-safe (`w180_days_buy`,
  `rec_buy`), никакой мета-модели;
* сегменты — 3 (или 4) КРУПНЫХ, границы взяты из таблицы сегментов `exp_036`,
  а не подобраны;
* сетка веса ETX внутри сегмента грубая: 0 / 0.25 / 0.5 / 0.75 / 1.0;
* отбор — тот же честный LOFO: и alpha, и веса смеси выбираются на ТРЁХ фолдах и
  проверяются на ЧЕТВЁРТОМ. Ничего не выбирается на тесте.

Считается через моменты, а не перебором векторов: сегменты образуют РАЗБИЕНИЕ,
поэтому `Var(ly - B(w) - w_seq * sum_s alpha_s * D_s)` раскладывается по
сегментным суммам, и 125 (или 625) комбинаций alpha стоят столько же, сколько
одна. Правильность разложения проверяется прямым пересчётом (`assert` ниже).

Запуск: PYTHONPATH=. python research/strategies/results/ETX2/segblend.py
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "research/strategies/results/ETX2")
import common  # noqa: E402
from lofo2 import CAP, DIST, E02, REF_EXPS, REF_W, narrow_grid  # noqa: E402

from src.blend import aligned, fold_masks, shifted_rmsle  # noqa: E402
from src.config import ARTIFACTS, FOLD_WEIGHTS_S1  # noqa: E402
from src.tracking import load_oof  # noqa: E402

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
# имена членов можно подменить для контроля (например, одиночным ETX-01-S42)
ETX_NAME = os.environ.get("ETX2_ETX", "ETX-AVG3")
TCN_NAME = os.environ.get("ETX2_TCN", "SEQ-AVG3")


def seg_labels(cut: np.ndarray, uid: np.ndarray, kind: str) -> np.ndarray:
    """Метки сегмента для строк OOF: признаки берутся на СВОЁМ cutoff'е строки."""
    out = np.empty(len(cut), np.int8)
    for c in sorted(set(cut.tolist())):
        m = cut == c
        f = common.feats(dt.date.fromisoformat(c), uid[m])
        out[m] = common.seg3(f) if kind == "seg3" else common.seg4(f)
    return out


def moments(u: np.ndarray, D: list[np.ndarray]):
    """Всё, что нужно для Var(u - w * sum_s alpha_s D_s) при ДИЗЪЮНКТНЫХ D_s."""
    return dict(n=len(u), su=u.sum(), su2=float((u * u).sum()),
                sd=np.array([d.sum() for d in D]),
                sd2=np.array([float((d * d).sum()) for d in D]),
                sud=np.array([float((u * d).sum()) for d in D]))


def var_from(mo, w: float, a: np.ndarray) -> float:
    n = mo["n"]
    e1 = (mo["su"] - w * float((a * mo["sd"]).sum())) / n
    e2 = (mo["su2"] - 2 * w * float((a * mo["sud"]).sum())
          + w * w * float((a * a * mo["sd2"]).sum())) / n
    return float(e2 - e1 * e1)


def main() -> None:
    for n in (ETX_NAME, TCN_NAME):
        if not (ARTIFACTS / f"oof_{n}.npz").exists():
            print(f"нет oof_{n}.npz — этап 3 ещё не запускается")
            return
    kinds = sys.argv[1:] or ["seg3", "seg4"]
    base = sorted({*REF_EXPS, CAP, E02, DIST, ETX_NAME, TCN_NAME})
    Z, y, cut = aligned(base)
    idx = {e: i for i, e in enumerate(base)}
    ly = np.log1p(y)
    folds, masks = fold_masks(cut)
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()
    ref_fc = np.array([shifted_rmsle(ly[m], np.average(Z[[idx[e] for e in REF_EXPS]][:, m],
                                                       axis=0, weights=REF_W))
                       for m in masks])
    ref_wcv = float(w_f @ ref_fc)

    # порядок строк `aligned` — сортировка по (cutoff, user_id как строка)
    d0 = load_oof(base[0])
    k = np.char.add(np.asarray(d0["cutoff"], "U10"), np.asarray(d0["user_id"]).astype("U20"))
    uid = np.asarray(d0["user_id"])[np.argsort(k)]

    grid = narrow_grid()
    zt, ze = Z[idx[TCN_NAME]], Z[idx[ETX_NAME]]
    dz = ze - zt
    print(f"n = {len(y):,}, опора SEQ-01-MIX wCV={ref_wcv:.5f}")
    print(f"слот: {TCN_NAME} + a_s*({ETX_NAME} - {TCN_NAME}), a из {ALPHA_GRID}\n")

    results = {}
    for kind in kinds:
        seg = seg_labels(cut, uid, kind)
        S = int(seg.max()) + 1
        names = common.SEG_NAMES[kind][:S]
        D = [np.where(seg == s, dz, 0.0) for s in range(S)]
        print(f"===== {kind}: {S} сегментов =====")
        for s in range(S):
            m = seg == s
            print(f"  {s} {names[s]:<24} доля {m.mean():.3f}  "
                  f"Var(z_ETX - z_TCN) = {np.var(dz[m]):.5f}")

        MO = [[moments(ly[m] - np.average(np.vstack([Z[idx[CAP]], Z[idx[E02]],
                                                     Z[idx[DIST]], zt])[:, m],
                                          axis=0, weights=w),
                       [d[m] for d in D]) for m in masks] for w in grid]
        alphas = [np.array(a) for a in itertools.product(ALPHA_GRID, repeat=S)]

        # контроль разложения: прямой пересчёт одной случайной пары (alpha, w)
        rng = np.random.default_rng(0)
        ai = int(rng.integers(len(alphas)))
        wi = int(rng.integers(len(grid)))
        fi = int(rng.integers(len(folds)))
        a_, w_, m_ = alphas[ai], grid[wi], masks[fi]
        z_slot = zt + sum(a_[s] * D[s] for s in range(S))
        zmix = np.average(np.vstack([Z[idx[CAP]], Z[idx[E02]], Z[idx[DIST]], z_slot])[:, m_],
                          axis=0, weights=w_)
        direct = shifted_rmsle(ly[m_], zmix)
        fast = float(np.sqrt(var_from(MO[wi][fi], w_[3], a_)))
        assert abs(direct - fast) < 1e-8, f"разложение неверно: {direct} vs {fast}"

        FC = np.empty((len(alphas), len(grid), len(folds)))
        for ai_, a in enumerate(alphas):
            for wi_, w in enumerate(grid):
                for fi_ in range(len(folds)):
                    FC[ai_, wi_, fi_] = np.sqrt(var_from(MO[wi_][fi_], w[3], a))
        flat = FC.reshape(-1, len(folds))
        combos = [(a, w) for a in alphas for w in grid]

        held, chosen = np.zeros(len(folds)), []
        for h in range(len(folds)):
            keep = [i for i in range(len(folds)) if i != h]
            wh = w_f[keep] / w_f[keep].sum()
            b = int(np.argmin(flat[:, keep] @ wh))
            held[h] = flat[b, h]
            chosen.append(combos[b])
        d_lofo = held - ref_fc
        ins = int(np.argmin(flat @ w_f))
        print(f"\n  честный LOFO: {float(w_f @ d_lofo):+.5f}  "
              f"фолдов лучше {int((d_lofo < 0).sum())}/4   пофолдово "
              + " ".join(f"{v:+.5f}" for v in d_lofo))
        for h, f_ in enumerate(folds):
            a, w = chosen[h]
            print(f"    держим {f_}: a={list(np.round(a, 2))} веса={list(w)}")
        a_ins, w_ins = combos[ins]
        print(f"  в выборке оптимум: a={list(np.round(a_ins, 2))} веса={list(w_ins)}  "
              f"{float(flat[ins] @ w_f) - ref_wcv:+.5f}")
        stable = all(np.array_equal(chosen[0][0], c[0]) for c in chosen)
        print(f"  alpha устойчив по held-out: {'ДА' if stable else 'НЕТ'}")

        print("\n  глобальные alpha (одна степень свободы), честный LOFO:")
        glob = {}
        for g in ALPHA_GRID:
            a = np.full(S, g)
            gi = [i for i, (aa, _) in enumerate(combos) if np.array_equal(aa, a)]
            sub = flat[gi]
            hh = np.zeros(len(folds))
            for h in range(len(folds)):
                keep = [i for i in range(len(folds)) if i != h]
                wh = w_f[keep] / w_f[keep].sum()
                hh[h] = sub[int(np.argmin(sub[:, keep] @ wh)), h]
            dd = hh - ref_fc
            glob[g] = dict(lofo=float(w_f @ dd), folds=int((dd < 0).sum()),
                           per_fold=dd.tolist())
            print(f"    a={g:<5} LOFO {float(w_f @ dd):+.5f}  {int((dd < 0).sum())}/4")

        results[kind] = dict(
            n_seg=S, names=names, shares=[float((seg == s).mean()) for s in range(S)],
            lofo=float(w_f @ d_lofo), folds=int((d_lofo < 0).sum()),
            per_fold=d_lofo.tolist(),
            chosen=[[list(np.round(a, 3)), list(w)] for a, w in chosen],
            stable=bool(stable), insample_alpha=[float(v) for v in a_ins],
            insample_w=list(w_ins), insample_delta=float(flat[ins] @ w_f) - ref_wcv,
            global_alpha={str(k_): v for k_, v in glob.items()})
        print()

    (ARTIFACTS / "ETX2_segblend.json").write_text(
        json.dumps(dict(ref_wcv=ref_wcv, folds=folds, alpha_grid=ALPHA_GRID,
                        etx=ETX_NAME, tcn=TCN_NAME, results=results),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("записано: artifacts/ETX2_segblend.json")


if __name__ == "__main__":
    main()
