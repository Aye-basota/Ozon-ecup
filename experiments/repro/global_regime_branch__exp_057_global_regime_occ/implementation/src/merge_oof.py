"""Сборка одного эксперимента из пофолдовых прогонов.

Нужна потому, что фолды выгодно считать РАЗНЫМИ процессами: LightGBM плохо
масштабируется по потокам (замер — research/compute_profile.md), и два процесса
по 6 потоков проходят 4 фолда быстрее, чем один процесс на 12 потоках.

Пофолдовые прогоны запускаются с `--val <дата> --no-log`, здесь их OOF
склеиваются в один `oof_<out>.npz` и пишется ОДНА строка в experiments/log.csv.

Запуск:
  python -m src.merge_oof --out S1-DIST --parts S1-DIST-V0904 S1-DIST-V0918 \
                          S1-DIST-V1002 S1-DIST-V1016 --ref S1-E10 \
                          --desc "E0: голова распределения" --model dist
"""
from __future__ import annotations

import argparse

import numpy as np

from src.report import evaluate, format_report, save_report
from src.tracking import load_oof, log_from_report, save_oof
from src.validation import rmsle_z


def merge_arrays(user_id, cutoff, z, y) -> dict:
    """Метрики склеенного OOF. Пара (cutoff, user_id) обязана быть уникальной.

    Считает ровно тот же отчёт, что и `train.run`: пофолдовые прогоны и цельный
    прогон обязаны давать побитово одинаковые числа, иначе их нельзя сравнивать.
    """
    cutoff = np.asarray(cutoff, dtype="U10")
    keys = np.char.add(cutoff, np.asarray(user_id).astype("U20"))
    assert len(np.unique(keys)) == len(keys), (
        "повтор пары (cutoff, user_id): один и тот же фолд склеен дважды")
    return evaluate(y, z, cutoff)


def diversity(z, y, z_ref) -> dict:
    """Разнообразие относительно опорной модели: то, ради чего нужен ансамбль."""
    r, r_ref = np.log1p(y) - z, np.log1p(y) - z_ref
    return dict(var_delta=float(np.var(z - z_ref)),
                corr_resid=float(np.corrcoef(r, r_ref)[0, 1]),
                corr_pred=float(np.corrcoef(z, z_ref)[0, 1]))


def auc_positive(y, z) -> float:
    """AUC экстенсива: насколько прогноз ранжирует «купит ли вообще»."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score((y > 0).astype(np.int8), z))


def load_parts(parts: list[str]):
    ds = [load_oof(p) for p in parts]
    return (np.concatenate([d["user_id"] for d in ds]),
            np.concatenate([np.asarray(d["cutoff"], dtype="U10") for d in ds]),
            np.concatenate([d["z"] for d in ds]), np.concatenate([d["y"] for d in ds]))


def aligned_ref(user_id, cutoff, ref: str):
    """z опорного эксперимента, выстроенный в том же порядке строк."""
    d = load_oof(ref)
    k_ref = np.char.add(np.asarray(d["cutoff"], dtype="U10"),
                        np.asarray(d["user_id"]).astype("U20"))
    k = np.char.add(cutoff, np.asarray(user_id).astype("U20"))
    idx = {key: i for i, key in enumerate(k_ref)}
    miss = [key for key in k if key not in idx]
    assert not miss, f"{ref}: нет {len(miss)} строк из {len(k)} (разные наборы фолдов?)"
    return d["z"][np.array([idx[key] for key in k])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="опорный эксперимент для Var(z - z_ref)")
    ap.add_argument("--desc", default="")
    ap.add_argument("--model", default="dist")
    ap.add_argument("--params", default="")
    ap.add_argument("--n-features", type=int, default=0)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    uid, cut, z, y = load_parts(a.parts)
    m = merge_arrays(uid, cut, z, y)
    print(f"склеено {len(a.parts)} прогонов -> {m['n']:,} строк OOF")

    mr = None
    if a.ref:
        z_ref = aligned_ref(uid, cut, a.ref)
        mr = merge_arrays(uid, cut, z_ref, y)
    print(format_report(m, mr))
    print(f"  AUC(1[y>0]) = {auc_positive(y, z):.5f}")

    if mr is not None:
        d = diversity(z, y, z_ref)
        w0, w1 = mr["wcv"], m["wcv"]
        wtxt = (f"wCV {w0:.5f} -> {w1:.5f} ({w1 - w0:+.5f})" if w0 and w1 else "wCV н/д")
        print(f"\nпротив {a.ref}: {wtxt}, OOF калибр. {mr['oof_cal']:.5f} -> {m['oof_cal']:.5f} "
              f"({m['oof_cal'] - mr['oof_cal']:+.5f})")
        print(f"  Var(z - z_ref) = {d['var_delta']:.5f}  (ориентир разнообразия >= 0.10)")
        print(f"  corr предсказаний {d['corr_pred']:.5f}, corr остатков {d['corr_resid']:.5f}")
        print(f"  AUC опорной = {auc_positive(y, z_ref):.5f}")

    save_oof(a.out, uid, cut, z, y)
    save_report(a.out, m, extra=dict(description=a.desc, parts=a.parts, ref=a.ref))
    print(f"\nOOF сохранён: artifacts/oof_{a.out}.npz, отчёт: artifacts/report_{a.out}.json")
    if not a.no_log:
        log_from_report(a.out, a.desc, m, scenario="S1", n_features=a.n_features,
                        model=a.model, params=a.params)
        print("строка записана в experiments/log.csv")


if __name__ == "__main__":
    main()
