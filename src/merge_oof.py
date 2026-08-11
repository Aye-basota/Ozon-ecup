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

from src.tracking import load_oof, log_experiment, save_oof
from src.validation import best_offset, bias_z, rmsle_z


def merge_arrays(user_id, cutoff, z, y) -> dict:
    """Метрики склеенного OOF. Пара (cutoff, user_id) обязана быть уникальной."""
    cutoff = np.asarray(cutoff, dtype="U10")
    keys = np.char.add(cutoff, np.asarray(user_id).astype("U20"))
    assert len(np.unique(keys)) == len(keys), (
        "повтор пары (cutoff, user_id): один и тот же фолд склеен дважды")
    folds = sorted(np.unique(cutoff).tolist())
    scores, biases, offs, sizes = [], [], [], []
    for c in folds:
        m = cutoff == c
        scores.append(rmsle_z(y[m], z[m]))
        biases.append(bias_z(y[m], z[m]))
        offs.append(best_offset(y[m], z[m]))
        sizes.append(int(m.sum()))
    o_all, oof_cal = best_offset(y, z)
    return dict(n=len(y), folds=folds, sizes=sizes, fold_scores=scores, fold_biases=biases,
                fold_offsets=[o for o, _ in offs], fold_calib=[s for _, s in offs],
                cv_mean=float(np.mean(scores)), cv_std=float(np.std(scores)),
                bias_mean=float(np.mean(biases)), oof=rmsle_z(y, z),
                best_offset=o_all, oof_calib=oof_cal)


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
    print(f"склеено {len(a.parts)} прогонов -> {m['n']:,} строк OOF\n")
    print(f"{'фолд':>12} {'n':>9} {'RMSLE':>9} {'bias':>8} {'после сдвига':>13}")
    for c, n, sc, b, o, cal in zip(m["folds"], m["sizes"], m["fold_scores"], m["fold_biases"],
                                   m["fold_offsets"], m["fold_calib"]):
        print(f"{c:>12} {n:>9,} {sc:>9.5f} {b:>+8.4f} {cal:>13.5f}  (сдвиг {o:+.3f})")
    print(f"\nCV mean={m['cv_mean']:.5f} std={m['cv_std']:.5f} | OOF={m['oof']:.5f} "
          f"bias={m['bias_mean']:+.4f} | после сдвига {m['best_offset']:+.3f}: {m['oof_calib']:.5f}")
    print(f"AUC(1[y>0]) = {auc_positive(y, z):.5f}")

    if a.ref:
        z_ref = aligned_ref(uid, cut, a.ref)
        d = diversity(z, y, z_ref)
        mr = merge_arrays(uid, cut, z_ref, y)
        print(f"\nпротив {a.ref}: OOF {mr['oof_calib']:.5f} -> {m['oof_calib']:.5f} "
              f"({m['oof_calib'] - mr['oof_calib']:+.5f} калиброванного)")
        wins = sum(s < r for s, r in zip(m["fold_calib"], mr["fold_calib"]))
        print(f"  лучше на {wins} фолдах из {len(m['folds'])}: "
              + " ".join(f"{s - r:+.5f}" for s, r in zip(m["fold_calib"], mr["fold_calib"])))
        print(f"  Var(z - z_ref) = {d['var_delta']:.5f}  (ориентир разнообразия >= 0.10)")
        print(f"  corr предсказаний {d['corr_pred']:.5f}, corr остатков {d['corr_resid']:.5f}")
        print(f"  AUC опорной = {auc_positive(y, z_ref):.5f}")

    save_oof(a.out, uid, cut, z, y)
    print(f"\nOOF сохранён: artifacts/oof_{a.out}.npz")
    if not a.no_log:
        log_experiment(exp_id=a.out, description=a.desc, scenario="S1",
                       n_features=a.n_features, model=a.model, params=a.params,
                       fold_scores=m["fold_scores"], cv_mean=m["cv_mean"], cv_std=m["cv_std"],
                       bias_mean=m["bias_mean"], best_offset=m["best_offset"],
                       cv_mean_calib=m["oof_calib"])
        print("строка записана в experiments/log.csv")


if __name__ == "__main__":
    main()
