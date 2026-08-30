"""Сборка единого кадра для диагностики RMSLE — только из СОХРАНЁННЫХ артефактов.

Ничего не переобучает. Берёт OOF четырёх членов боевой смеси `S1-DIST-MIX`
(`exp_014`), складывает их в лог-пространстве с боевыми весами, калибрует
пофолдовым сдвигом (это ровно то, что меряет wCV) и приклеивает сегментные
признаки с ТОГО ЖЕ cutoff'а — то есть без единого бита будущего.

Выход:
  fold_predictions.parquet  — 770 616 строк OOF x 4 фолда: y, z членов, z смеси,
                              z после пофолдовой калибровки, сегменты;
  test_predictions.parquet  — 250 000 строк тестового cutoff'а: z смеси после
                              уровня 2.3293 и те же сегменты.

Запуск: python research/rmsle_diagnostics/build_frame.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl

from src.config import ARTIFACTS, CUTOFF_TEST, VAL_FOLDS_S1
from src.blend import aligned
from src.validation import calibrate

OUT = Path(__file__).resolve().parent

# боевая смесь S1-DIST-MIX (exp_014 §Смесь, STATE.md «Текущий лучший»)
MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}
# те же четыре модели, обученные на всём коридоре, — предсказания на тесте
MIX_TEST = {"S1-NORM": 0.15, "S1-UNC": 0.30, "S1-CAP": 0.10, "S1-DIST": 0.45}
LEVEL = 2.3293                      # измеренный оптимум уровня (exp_006/exp_016)

# сегменты берутся из некалиброванного кэша LNone: w365_* там сырые суммы,
# а не годовые ставки, поэтому наивные бейзлайны считаются без пересчёта
SEG = ["rec_buy", "rec_any", "rec_cart", "w30_gmv", "w60_gmv", "w90_gmv", "w180_gmv",
       "w365_gmv", "all_gmv", "w30_days_buy", "w90_days_buy", "w180_days_buy",
       "w365_days_buy", "w30_days_present", "w90_days_present", "w180_lgmv_mean",
       "buygap_mean", "tenure", "w30_searches", "w90_searches"]


def seg_frame(T) -> pl.DataFrame:
    tag = T.strftime("%Y%m%d")
    f = pl.read_parquet(ROOT / "data" / "processed" / f"feat_{tag}_LNone.parquet",
                        columns=["user_id"] + SEG)
    return f.with_columns([pl.col(c).cast(pl.Float32) for c in SEG])


def build_oof() -> pl.DataFrame:
    exps = list(MIX)
    Z, y, cut = aligned(exps)
    uid = np.asarray(np.load(ARTIFACTS / f"oof_{exps[0]}.npz")["user_id"])
    # aligned() сортирует по ключу cutoff+user_id; повторяем ту же перестановку
    d0 = np.load(ARTIFACTS / f"oof_{exps[0]}.npz")
    key = np.char.add(np.asarray(d0["cutoff"], dtype="U10"),
                      np.asarray(d0["user_id"]).astype("U20"))
    order = np.argsort(key)
    uid = uid[order]
    assert np.array_equal(np.asarray(d0["y"])[order], y), "порядок строк разошёлся"

    w = np.array([MIX[e] for e in exps], float)
    z_mix = np.average(Z, axis=0, weights=w)

    df = pl.DataFrame({"cutoff": cut, "user_id": uid, "y": y, "z_mix": z_mix.astype(np.float32)}
                      | {f"z_{e}": Z[i].astype(np.float32) for i, e in enumerate(exps)})

    # пофолдовая калибровка — определение wCV (exp_016 §6)
    parts = []
    for T in VAL_FOLDS_S1:
        c = T.isoformat()
        m = df.filter(pl.col("cutoff") == c)
        d, _ = calibrate(m["y"].to_numpy(), m["z_mix"].to_numpy())
        m = m.with_columns(
            z_cal=pl.max_horizontal(pl.col("z_mix") + d, pl.lit(0.0)).cast(pl.Float32),
            fold_delta=pl.lit(d, dtype=pl.Float32))
        parts.append(m.join(seg_frame(T), on="user_id", how="left"))
        print(f"  {c}: n={m.height:,}  delta={d:+.4f}")
    return pl.concat(parts).with_columns(ly=pl.col("y").log1p().cast(pl.Float32))


def build_test() -> pl.DataFrame:
    Z = np.vstack([np.load(ARTIFACTS / f"ztest_{n}.npy") for n in MIX_TEST])
    uid = np.load(ARTIFACTS / f"uid_{list(MIX_TEST)[0]}.npy")
    w = np.array(list(MIX_TEST.values()), float)
    z = np.average(Z, axis=0, weights=w)
    delta = LEVEL - float(z.mean())
    print(f"  тест: n={len(z):,}  mean(z)={z.mean():.4f}  delta={delta:+.4f} -> {LEVEL}")
    df = pl.DataFrame({"user_id": uid, "z_mix": z.astype(np.float32),
                       "z_cal": np.maximum(z + delta, 0.0).astype(np.float32)}
                      | {f"z_{e}": Z[i].astype(np.float32) for i, e in enumerate(MIX_TEST)})
    return df.join(seg_frame(CUTOFF_TEST), on="user_id", how="left")


if __name__ == "__main__":
    print("OOF (4 фолда):")
    oof = build_oof()
    oof.write_parquet(OUT / "fold_predictions.parquet")
    print(f"  записано {oof.height:,} x {oof.width} -> fold_predictions.parquet")
    print("тест:")
    te = build_test()
    te.write_parquet(OUT / "test_predictions.parquet")
    print(f"  записано {te.height:,} x {te.width} -> test_predictions.parquet")
