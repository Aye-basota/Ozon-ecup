"""Сегментная оценка эксперимента: AUC активности и RMSLE там, где сидит ошибка.

Зачем отдельно от `src/report.py`. `report.evaluate` отвечает на вопрос «лучше ли
модель в среднем», а `research/rmsle_diagnostics` показал, что средним здесь
ничего не решается: 74% остаточной дисперсии — бернуллиевский член «попадёт ли
покупка в окно», 70.6% ошибки лежит в полосе 2–15 покупательных дней за 180
дней, а максимум RMSLE по свежести — интервал `rec_buy` 15–60 дней, где
решается «обычная пауза или уход». Обменный курс оттуда же: +0.001 AUC ≈
−0.0006…−0.0019 wCV, то есть содержательный выигрыш требует **+0.003…+0.008 AUC**,
и движение в четвёртом знаке подтверждением гипотезы не считается.

Сегменты читаются из НЕнормированного кэша `feat_<T>_LNone.parquet` — того же,
что и в `research/rmsle_diagnostics/build_frame.py`, чтобы границы полос
(`rec_buy`, `w180_days_buy`) совпадали с уже опубликованными числами.

Все RMSLE — после ПОФОЛДОВОГО оптимального сдвига, посчитанного на полном фолде,
а не внутри сегмента: сегментная перекалибровка закрыта измерением
(`rmsle_diagnostics` §3), и давать её каждому варианту заново значило бы
сравнивать модели вместе с бесплатной подгонкой уровня.

Запуск:
  python -m src.ptime_eval --exps S1-SEEDAVG3 PT-OD-AVG3 PT-FULL-AVG3 PT-SHUF-AVG3 \
                           --base S1-SEEDAVG3 --out research/strategies/results/STRATEGY_08
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from src.blend import aligned
from src.config import ARTIFACTS, DATA_PROCESSED, FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.tracking import load_oof
from src.validation import calibrate, rmsle_z

SEG_COLS = ["rec_buy", "w180_days_buy", "w180_gmv"]
FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]
W = np.asarray(FOLD_WEIGHTS_S1, float)
W = W / W.sum()

# боевая смесь S1-DIST-MIX (STATE.md «Текущий лучший», exp_014)
MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}


def auc(y_pos: np.ndarray, score: np.ndarray) -> float:
    """AUC активности: то же определение, что и в `merge_oof.auc_positive`."""
    from sklearn.metrics import roc_auc_score
    n1 = int(y_pos.sum())
    if n1 == 0 or n1 == len(y_pos):
        return float("nan")
    return float(roc_auc_score(y_pos.astype(np.int8), score))


def order_of(exp: str):
    """Перестановка строк, которую применяет `blend.aligned` (ключ cutoff+user_id)."""
    d = load_oof(exp)
    cut = np.asarray(d["cutoff"], dtype="U10")
    k = np.char.add(cut, np.asarray(d["user_id"]).astype("U20"))
    o = np.argsort(k)
    return d["user_id"][o], cut[o]


def seg_frame() -> pl.DataFrame:
    parts = []
    for T in VAL_FOLDS_S1:
        f = pl.read_parquet(DATA_PROCESSED / f"feat_{T.strftime('%Y%m%d')}_LNone.parquet",
                            columns=["user_id"] + SEG_COLS)
        parts.append(f.with_columns(cutoff=pl.lit(T.isoformat())))
    return pl.concat(parts)


def calibrated(Z: np.ndarray, y: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Пофолдовый оптимальный сдвиг — ровно то, что меряет wCV (exp_016 §6)."""
    out = np.empty_like(Z)
    for i in range(len(Z)):
        z = Z[i].astype(float)
        for c in FOLDS:
            m = cut == c
            d, _ = calibrate(y[m], z[m])
            out[i, m] = np.maximum(z[m] + d, 0.0)
    return out


def segments(df: pl.DataFrame) -> dict[str, np.ndarray]:
    rb, nb = df["rec_buy"].to_numpy(), df["w180_days_buy"].to_numpy()
    rb_known = ~np.isnan(rb)
    band_rec = rb_known & (rb >= 15) & (rb <= 60)
    band_freq = (nb >= 2) & (nb <= 15)
    seg = {
        "ВСЕ": np.ones(len(rb), bool),
        "rec_buy 15-60": band_rec,
        "w180_days_buy 2-15": band_freq,
        "пересечение 2-15 x 15-60": band_rec & band_freq,
        "rec_buy 0-7": rb_known & (rb <= 7),
        "rec_buy 8-14": rb_known & (rb >= 8) & (rb <= 14),
        "rec_buy 15-30": rb_known & (rb >= 15) & (rb <= 30),
        "rec_buy 31-60": rb_known & (rb >= 31) & (rb <= 60),
        "rec_buy 61-180": rb_known & (rb >= 61) & (rb <= 180),
        "rec_buy 180+": rb_known & (rb > 180),
        "никогда не покупал": ~rb_known,
        "w180_days_buy 0-1": nb <= 1,
        "w180_days_buy 2-3": (nb >= 2) & (nb <= 3),
        "w180_days_buy 4-7": (nb >= 4) & (nb <= 7),
        "w180_days_buy 8-15": (nb >= 8) & (nb <= 15),
        "w180_days_buy 16+": nb >= 16,
    }
    return seg


def seg_metrics(name: str, mask: np.ndarray, Zc: np.ndarray, y: np.ndarray,
                cut: np.ndarray, exps: list[str]) -> list[dict]:
    """Пофолдово взвешенные 1:2:4:8 RMSLE и AUC внутри сегмента."""
    rows = []
    ly = np.log1p(y)
    pos = y > 0
    for i, e in enumerate(exps):
        rs, au, ns = [], [], []
        for c in FOLDS:
            m = mask & (cut == c)
            ns.append(int(m.sum()))
            rs.append(rmsle_z(y[m], Zc[i, m]) if m.any() else np.nan)
            au.append(auc(pos[m], Zc[i, m]) if m.any() else np.nan)
        mse_share = float(np.sum(((ly - Zc[i]) ** 2)[mask]) / np.sum((ly - Zc[i]) ** 2))
        rows.append(dict(segment=name, exp=e, n=int(mask.sum()),
                         share=float(mask.mean()), mse_share=mse_share,
                         rmsle=float(np.dot(W, rs)), auc=float(np.dot(W, au)),
                         rmsle_folds=[round(v, 5) for v in rs],
                         auc_folds=[round(v, 5) for v in au]))
    return rows


def mix_z(cut: np.ndarray, uid: np.ndarray) -> np.ndarray | None:
    """z боевой смеси, выстроенный в том же порядке строк."""
    if not all((ARTIFACTS / f"oof_{e}.npz").exists() for e in MIX):
        return None
    Z, _, cut_m = aligned(list(MIX))
    assert np.array_equal(cut_m, cut), "порядок строк смеси разошёлся с оценкой"
    return np.average(Z, axis=0, weights=[MIX[e] for e in MIX])


def fixed_mix(exps: list[str], y: np.ndarray, cut: np.ndarray, Z: np.ndarray) -> list[dict]:
    """Кандидат подставляется в слот `S1-E10` боевой смеси; веса НЕ подбираются.

    Ровно та же процедура, что и в Tier A checkpoint (`tier_a_fixed_mix_oof.csv`):
    перебор весов отвергнут (`MIX-E11`, LB +0.00023), поэтому ансамблевая ценность
    измеряется подстановкой при фиксированных 0.15/0.30/0.10/0.45.
    """
    if not all((ARTIFACTS / f"oof_{e}.npz").exists() for e in MIX):
        return []
    Zm, ym, cm = aligned(list(MIX))
    assert np.array_equal(cm, cut) and np.allclose(ym, y), "порядок строк смеси разошёлся"
    w = np.array([MIX[e] for e in MIX], float)
    slot = list(MIX).index("S1-E10")
    out = []
    for name, zc in [("S1-DIST-MIX (база)", None)] + [(e, Z[i]) for i, e in enumerate(exps)]:
        Zx = Zm.copy()
        if zc is not None:
            Zx[slot] = zc
        z = np.average(Zx, axis=0, weights=w)
        fc = [calibrate(y[cut == c], z[cut == c])[1] for c in FOLDS]
        out.append(dict(config=name, wcv=float(np.dot(W, fc)),
                        **{f"fold_{c[5:7]}{c[8:10]}": v for c, v in zip(FOLDS, fc)}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="опорный эксперимент для дельт")
    ap.add_argument("--out", default=None, help="каталог для CSV")
    a = ap.parse_args()

    exps = list(dict.fromkeys([a.base] + a.exps))
    Z, y, cut = aligned(exps)
    uid, cut2 = order_of(exps[0])
    assert np.array_equal(cut, cut2)
    print(f"строк OOF {len(y):,}, моделей {len(exps)}")

    df = (pl.DataFrame({"cutoff": cut, "user_id": uid})
          .join(seg_frame(), on=["cutoff", "user_id"], how="left"))
    assert df.height == len(y)
    Zc = calibrated(Z, y, cut)
    ib = exps.index(a.base)

    # --- 1. общий скор, AUC и стабильность по фолдам ---------------------------
    print(f"\n{'модель':>14} {'wCV':>9} {'дельта':>9} {'AUC(y>0)':>9} {'дельта':>9}"
          f"  {'AUC pooled':>10}  фолдов лучше   пофолдовый RMSLE_cal")
    base_rows = seg_metrics("ВСЕ", np.ones(len(y), bool), Zc, y, cut, exps)
    # pooled AUC — по всем 770k строкам сразу; именно его печатают
    # `merge_oof` и `rmsle_diagnostics` §4 (0.8423 у боевой смеси).
    # Взвешенный 1:2:4:8 выше на ~0.001: pooled смешивает фолды с разной базовой
    # ставкой. Сравнивать варианты между собой можно по любому, но не смешивая их.
    pooled = [auc(y > 0, Zc[i]) for i in range(len(exps))]
    fold_tbl = []
    for i, e in enumerate(exps):
        r, rb = base_rows[i], base_rows[ib]
        better = sum(x < b - 1e-12 for x, b in zip(r["rmsle_folds"], rb["rmsle_folds"]))
        print(f"{e:>14} {r['rmsle']:>9.5f} {r['rmsle'] - rb['rmsle']:>+9.5f} "
              f"{r['auc']:>9.5f} {r['auc'] - rb['auc']:>+9.5f}  {pooled[i]:>10.5f}  "
              f"{better:>10}/4   " + " ".join(f"{v:.5f}" for v in r["rmsle_folds"]))
        r["auc_pooled"] = pooled[i]
        for c, sc, ac in zip(FOLDS, r["rmsle_folds"], r["auc_folds"]):
            fold_tbl.append(dict(exp=e, cutoff=c, rmsle_cal=sc, auc=ac))

    # --- 2. сегменты ----------------------------------------------------------
    seg = segments(df)
    rows = []
    for name, m in seg.items():
        rows += seg_metrics(name, m, Zc, y, cut, exps)
    print(f"\n{'сегмент':>26} {'доля':>6} {'доля MSE':>9} | "
          + " ".join(f"{e[:13]:>13}" for e in exps))
    for name in seg:
        rs = [r for r in rows if r["segment"] == name]
        r0 = rs[ib]
        print(f"{name:>26} {r0['share']:>6.1%} {r0['mse_share']:>9.1%} | RMSLE "
              + " ".join(f"{r['rmsle']:>13.5f}" for r in rs))
        print(f"{'':>26} {'':>6} {'':>9} |   AUC "
              + " ".join(f"{r['auc']:>13.5f}" for r in rs))
        print(f"{'':>26} {'':>6} {'':>9} | дельта "
              + " ".join(f"{r['rmsle'] - r0['rmsle']:>+6.5f}/{r['auc'] - r0['auc']:>+6.5f}"
                         for r in rs))

    # --- 3. разнообразие и корреляция остатков --------------------------------
    ly = np.log1p(y)
    zmix = mix_z(cut, uid)
    print(f"\n{'модель':>14} {'Var(z-z_base)':>14} {'corr ост. base':>15} "
          + (f"{'Var(z-z_mix)':>13} {'corr ост. mix':>14}" if zmix is not None else ""))
    div = []
    for i, e in enumerate(exps):
        d = dict(exp=e, var_vs_base=float(np.var(Z[i] - Z[ib])),
                 corr_resid_base=float(np.corrcoef(ly - Z[i], ly - Z[ib])[0, 1]))
        line = f"{e:>14} {d['var_vs_base']:>14.5f} {d['corr_resid_base']:>15.5f} "
        if zmix is not None:
            d["var_vs_mix"] = float(np.var(Z[i] - zmix))
            d["corr_resid_mix"] = float(np.corrcoef(ly - Z[i], ly - zmix)[0, 1])
            line += f"{d['var_vs_mix']:>13.5f} {d['corr_resid_mix']:>14.5f}"
        print(line)
        div.append(d)
    print("  пол разнообразия (два сида одной конфигурации): Var=0.00712, corr=0.99885 "
          "(exp_018)")

    # --- 4. ансамблевая ценность при ФИКСИРОВАННЫХ весах ----------------------
    fm = fixed_mix(exps, y, cut, Z)
    if fm:
        print(f"\nподстановка в слот S1-E10 боевой смеси (веса 0.15/0.30/0.10/0.45, не подбираются):")
        print(f"{'конфигурация':>20} {'wCV':>10} {'дельта':>10}   пофолдово")
        w0 = fm[0]["wcv"]
        for r in fm:
            print(f"{r['config']:>20} {r['wcv']:>10.6f} {r['wcv'] - w0:>+10.6f}   "
                  + " ".join(f"{v:.5f}" for k, v in r.items() if k.startswith("fold_")))

    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        if fm:
            pl.DataFrame(fm).write_csv(o / "fixed_mix.csv")
        pl.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in r.items()}
                      for r in rows]).write_csv(o / "segment_metrics.csv")
        pl.DataFrame(fold_tbl).write_csv(o / "fold_stability.csv")
        pl.DataFrame(div).write_csv(o / "diversity.csv")
        print(f"\nзаписано: {o}/segment_metrics.csv, fold_stability.csv, diversity.csv")


if __name__ == "__main__":
    main()
