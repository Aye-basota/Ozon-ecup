"""EXP-032B — извлечение боевого экстенсивного сигнала `P(y>0) = 1 - p0` из головы `S1-DIST`.

Зачем отдельный скрипт. `exp_014` показал, что экстенсив сети надо читать через
`p_0` головы распределения: `AUC(1 - p0) = 0.84689` против `0.84368` у самого
`ẑ` и `0.8457` у специально обученного бинарного классификатора. `S1-DIST` —
член боевой смеси (`0.25` в `SEQ-01-MIX`), то есть этот сигнал уже прошёл LB.
Но `artifacts/oof_S1-DIST.npz` хранит только `ẑ = Σ p_k·m_k`: вектор
вероятностей по бинам в момент прогона никуда не записывался, и восстановить
`p0` из `ẑ` нельзя.

Поэтому голова переобучается ТЕМ ЖЕ кодом и ТЕМ ЖЕ конфигом (`exp_014` §Конфиг)
и на валидации сохраняется `p0`. Обучение не переизобретается: используются
`Setup`/`assemble`/`fit_free` из `src.train`, то есть боевой путь целиком.
Единственное отличие от `python -m src.train --model dist` — вместо
`models.predict_dist` (который сразу сворачивает распределение в среднее)
берётся сырая матрица вероятностей, и её нулевой столбец пишется на диск.

**Контроль подлинности.** Прогон обязан воспроизвести `oof_S1-DIST.npz`: `ẑ`
пересчитывается как `Σ p_k·m_k` и сверяется с сохранённым OOF построчно. Если
расхождение больше `--tol`, скрипт падает — тогда это не боевой сигнал, а его
пересборка, и использовать её как «production-safe extensive» нельзя.

Никаких EXTRA-cutoff'ов: сетка обучения — штатная `T + 30 <= V` внутри чистого
коридора, ровно как у `S1-DIST`.

Запуск:
  python -m src.dist_pact --val 2025-10-16
  python -m src.dist_pact                       # все четыре фолда подряд
  python -m src.dist_pact --check               # только сверка уже посчитанного
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import time

import numpy as np

from src.config import ARTIFACTS, SEED, VAL_FOLDS_S1
from src.data import load
from src.features import feature_names, to_np
from src.train import Setup, assemble, fit_free, xy

# Конфиг `S1-DIST` из exp_014 — менять здесь нечего, иначе сигнал перестанет
# быть боевым: L=None (--L 0), min_history=90, norm_long, 3-блочная val-панель,
# 1-блочная train-панель, шаг 7, 250 раундов, seed 42.
DIST_ROUNDS = 250
NS = "PACT"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def production_setup(vals: list[dt.date]) -> Setup:
    """`Setup`, побитово повторяющий команду `S1-DIST` из exp_014."""
    return Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                 model="dist", rounds=DIST_ROUNDS, params={"seed": SEED},
                 norm_long=True, vals=vals)


def path(V: dt.date):
    return ARTIFACTS / f"{NS}_dist_{V.isoformat()}.npz"


def reference_z(V: dt.date):
    """`ẑ` боевого OOF `S1-DIST` на этом фолде, выровненный по user_id панели."""
    d = np.load(ARTIFACTS / "oof_S1-DIST.npz", allow_pickle=True)
    m = d["cutoff"] == V.isoformat()
    return d["user_id"][m], d["z"][m].astype(np.float64), d["y"][m].astype(np.float64)


def run_fold(V: dt.date, tol: float) -> dict:
    s = production_setup([V])
    tr = s.train_cutoffs(V)
    assert tr, f"{V}: пустая сетка обучения"
    assert max(tr) + dt.timedelta(days=30) <= V, "нарушен зазор T+30<=V"
    feats = feature_names(xy(V, s)[0])
    log(f"фолд {V}: train {len(tr)} cutoff'ов {tr[0]}..{tr[-1]}, "
        f"зазор {(V - max(tr)).days}д, признаков {len(feats)}")

    t = time.time()
    Xtr, ytr, _ = assemble(tr, s, feats, V)
    n_tr = Xtr.shape[0]
    box = [Xtr]
    del Xtr
    booster, cent = fit_free(s, box, ytr, None)
    log(f"  обучено: {n_tr:,} строк x {len(feats)}, {DIST_ROUNDS} раундов "
        f"[{time.time() - t:.0f}s]")

    Xv, yv = xy(V, s)
    uid = Xv["user_id"].to_numpy()
    Av = to_np(Xv, feats)
    P = booster.predict(Av, num_iteration=DIST_ROUNDS)
    p0 = P[:, 0].astype(np.float64)
    z = np.maximum(P @ np.asarray(cent), 0.0)
    del Av, P, booster, box, ytr
    gc.collect()

    uid_ref, z_ref, y_ref = reference_z(V)
    assert np.array_equal(uid, uid_ref), "порядок строк панели разошёлся с боевым OOF"
    assert np.allclose(yv, y_ref), "таргет разошёлся с боевым OOF"
    dz = float(np.max(np.abs(z - z_ref)))
    log(f"  сверка с oof_S1-DIST: max|Δẑ| = {dz:.3e} (порог {tol:g})")
    assert dz <= tol, (f"голова не воспроизвела боевой OOF (max|Δẑ| = {dz:.3e}) — "
                       "это пересборка, а не боевой сигнал")

    pact = 1.0 - p0
    np.savez_compressed(path(V), user_id=uid, y=yv, p0=p0, p_act=pact, z=z,
                        z_ref=z_ref, max_abs_dz=dz, rounds=DIST_ROUNDS,
                        cuts=np.array([T.isoformat() for T in tr], dtype="U10"))
    log(f"  записано: artifacts/{path(V).name}; mean P(y>0) = {pact.mean():.4f} "
        f"против факта {float((yv > 0).mean()):.4f}")
    return dict(val=V.isoformat(), n=int(len(uid)), max_abs_dz=dz,
                mean_p=float(pact.mean()), rate=float((yv > 0).mean()))


def check(vals: list[dt.date]):
    for V in vals:
        p = path(V)
        if not p.exists():
            log(f"{V}: нет {p.name}")
            continue
        d = np.load(p)
        uid_ref, z_ref, _ = reference_z(V)
        ok = np.array_equal(d["user_id"], uid_ref)
        log(f"{V}: n={len(d['user_id']):,} max|Δẑ|={float(d['max_abs_dz']):.3e} "
            f"mean P(y>0)={float(d['p_act'].mean()):.4f} порядок строк {'OK' if ok else 'РАЗОШЁЛСЯ'}")


def main():
    ap = argparse.ArgumentParser(description="боевой экстенсив 1-p0 из головы S1-DIST")
    ap.add_argument("--val", nargs="*", default=None, help="фолды YYYY-MM-DD")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="допуск сверки ẑ с oof_S1-DIST")
    ap.add_argument("--check", action="store_true", help="только сверка готовых файлов")
    a = ap.parse_args()
    vals = [dt.date.fromisoformat(v) for v in a.val] if a.val else list(VAL_FOLDS_S1)
    if a.check:
        check(vals)
        return
    load()
    out = [run_fold(V, a.tol) for V in vals]
    log("итог: " + "; ".join(f"{r['val']} max|Δẑ|={r['max_abs_dz']:.1e} "
                             f"p̄={r['mean_p']:.4f}/{r['rate']:.4f}" for r in out))


if __name__ == "__main__":
    main()
