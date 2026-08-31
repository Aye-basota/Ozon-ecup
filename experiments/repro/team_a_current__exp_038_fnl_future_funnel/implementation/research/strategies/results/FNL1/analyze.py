"""EXP-038 (FNL) — диагностика пилота: BASE / BUYCTRL / CART / FUNNEL на 2025-10-16.

Что здесь считается и почему именно это.

1. `folds.csv` — калиброванный RMSLE и дельты. Главная дельта эксперимента —
   `d_cal_vs_BUYCTRL` при СОВПАДАЮЩЕЙ lambda, а не `d_cal_vs_BASE`: `BUYCTRL`
   несёт вторую голову без нового источника информации (роль контроля `SELF` из
   `exp_024`), поэтому только разница с ним отделяет новый сигнал от эффекта
   multi-task регуляризации и от того, что глобальный `clip_grad_norm_` при
   любой второй голове чуть перераспределяет шаг энкодера.
   Отдельно считается `rmsle_pos` — RMSLE на строках `y > 0`: вместе с
   `auc_y30_pos` он отвечает на вопрос «улучшение идёт через РАНЖИРОВАНИЕ
   активности или через ВЕЛИЧИНУ GMV».

2. `segments.csv` — там, где сидит ошибка (`rmsle_diagnostics`): `rec_buy 15-60`
   (максимум RMSLE по свежести), `w180_days_buy 2-15` (70.6% ошибки), их
   пересечение, «никогда не покупал» и `w180_days_buy 16+` (высокая активность).
   Калибровка — ПОФОЛДОВАЯ, посчитанная на полном фолде: сегментная
   перекалибровка закрыта измерением (`rmsle_diagnostics` §3), и давать её
   каждому варианту заново значило бы сравнивать модели вместе с бесплатной
   подгонкой уровня.

3. `aux_auc.csv` — самая информативная таблица при отрицательном результате.
   Для каждой арки и каждой метки воронки считаются ДВА числа: AUC собственной
   вспомогательной головы и AUC той же метки, ранжированной ГЛАВНОЙ головой `z`.
   Если второе почти догоняет первое, значит точечный прогноз GMV уже содержит
   всё, что несёт метка воронки, — ровно тот вывод, которым `exp_024` закрыл
   multi-horizon hazard. Метки для этой таблицы пересчитываются один раз на всю
   валидационную панель, поэтому арки без соответствующей головы (включая BASE)
   всё равно участвуют.

4. `diversity.csv` — `Var(z - z_BASE)`, корреляция прогнозов и корреляция
   ОСТАТКОВ. Пол сидов TCN: `Var(dz)` между сидами = 0.008 (`exp_036`).

5. `curves.csv` — поэпохные `train_mse`, `train_aux`, средняя норма градиента ДО
   клиппинга и валидационный RMSLE.

Запуск: PYTHONIOENCODING=utf-8 python research/strategies/results/FNL1/analyze.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS, DATA_PROCESSED  # noqa: E402
from src.fnl import ARMS  # noqa: E402
from src.validation import bias_z, calibrate, rmsle_z  # noqa: E402

OUT = Path(__file__).resolve().parent
V = dt.date(2025, 10, 16)
TAG = "V1016"

# (exp_id, арка, lambda). BASE — общая опора для всех.
RUNS = [("FNL-BASE-L00-S42", "BASE", 0.0),
        ("FNL-FUNNEL-L30-S42", "FUNNEL", 0.3),
        ("FNL-BUYCTRL-L30-S42", "BUYCTRL", 0.3),
        ("FNL-CART-L30-S42", "CART", 0.3),
        ("FNL-FUNNEL-L10-S42", "FUNNEL", 0.1),
        ("FNL-BUYCTRL-L10-S42", "BUYCTRL", 0.1),
        ("FNL-CART-L10-S42", "CART", 0.1),
        # Контроль шума: повтор BASE тем же сидом в другом процессе. Новой
        # информации нет по построению, поэтому его дельта — цена самого прогона.
        ("FNL-BASER2-L00-S42", "BASE-R2", 0.0)]
CTRL = "FNL-BASER2-L00-S42"

# Полный набор меток воронки + buy30: считается один раз, используется всеми арками.
ALL_HEADS = ARMS["BUYCTRL"] + ARMS["FUNNEL"]


def auc(pos: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(pos):
        return float("nan")
    return float(roc_auc_score(pos.astype(np.int8), score))


def load_run(exp: str):
    """(user_id, z, y, aux_pred, head_names) одного прогона."""
    o = np.load(ARTIFACTS / f"oof_{exp}-{TAG}.npz", allow_pickle=False)
    f = np.load(ARTIFACTS / f"fnl_{exp}-{TAG}.npz", allow_pickle=False)
    assert np.array_equal(o["user_id"], f["user_id"]), f"{exp}: разошлись user_id"
    return (o["user_id"], o["z"].astype(np.float64), o["y"].astype(np.float64),
            f["aux_pred"], [str(s) for s in f["head_names"]])


def seg_masks(user_id: np.ndarray) -> dict[str, np.ndarray]:
    """Сегменты из НЕнормированного кэша фичей — те же границы, что в публикациях."""
    f = pl.read_parquet(DATA_PROCESSED / f"feat_{V.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id", "rec_buy", "w180_days_buy"])
    j = pl.DataFrame(dict(user_id=user_id)).join(f, on="user_id", how="left")
    assert j.height == len(user_id)
    rb = j["rec_buy"].to_numpy().astype(np.float64)
    nb = j["w180_days_buy"].to_numpy().astype(np.float64)
    known = ~np.isnan(rb)
    band_rec = known & (rb >= 15) & (rb <= 60)
    band_freq = nb >= 2
    band_freq &= nb <= 15
    return {
        "ВСЕ": np.ones(len(rb), bool),
        "rec_buy 15-60": band_rec,
        "w180_days_buy 2-15": band_freq,
        "пересечение 2-15 x 15-60": band_rec & band_freq,
        "никогда не покупал": ~known,
        "rec_buy 0-7": known & (rb <= 7),
        "rec_buy 61-180": known & (rb >= 61) & (rb <= 180),
        "w180_days_buy 0-1": nb <= 1,
        "w180_days_buy 16+": nb >= 16,
    }


def main() -> None:
    missing = [e for e, _, _ in RUNS if not (ARTIFACTS / f"oof_{e}-{TAG}.npz").exists()]
    if missing:
        print("нет OOF для прогонов:", ", ".join(missing))
        print("готовые считаются всё равно")
    runs = [(e, a, l) for e, a, l in RUNS if (ARTIFACTS / f"oof_{e}-{TAG}.npz").exists()]
    assert runs, "нет ни одного готового прогона"

    data = {e: load_run(e) for e, _, _ in runs}
    uid0, _, y, _, _ = data[runs[0][0]]
    for e, _, _ in runs:
        assert np.array_equal(data[e][0], uid0), f"{e}: другой порядок пользователей"
    pos = y > 0
    ly = np.log1p(y)

    # --- метки воронки на валидационной панели, общие для всех арок.
    # Берутся ИЗ АРТЕФАКТОВ прогонов (`fnl_*.npz` хранит `aux_true` своей арки),
    # а не пересчитываются: это ровно те метки, на которых шло обучение, и при
    # этом не нужно поднимать плотную панель на 2.9 ГБ рядом с идущим обучением.
    hname, cols = [], []
    for e, _, _ in runs:
        _, _, _, _, hn = data[e]
        at = np.load(ARTIFACTS / f"fnl_{e}-{TAG}.npz", allow_pickle=False)["aux_true"]
        for j, nm in enumerate(hn):
            if nm not in hname:
                hname.append(nm)
                cols.append(at[:, j])
    A_true = (np.column_stack(cols) if cols
              else np.zeros((len(uid0), 0), np.float32))
    kind_of = {h.name: h.kind for h in ALL_HEADS}
    missing = [h.name for h in ALL_HEADS if h.name not in hname]
    if missing:
        print("метки без прогона-источника (пропущены в aux_auc):", ", ".join(missing))

    # --- калибровка: пофолдовый оптимальный сдвиг на ПОЛНОМ фолде
    zc, off = {}, {}
    for e, _, _ in runs:
        z = data[e][1]
        d, _ = calibrate(y, z)
        off[e] = d
        zc[e] = np.maximum(z + d, 0.0)

    base = "FNL-BASE-L00-S42"
    have_base = base in zc

    # ---------------------------------------------------------------- folds.csv
    rows_f = []
    for e, arm, lam in runs:
        z, zk = data[e][1], zc[e]
        buy = f"FNL-BUYCTRL-L{int(round(lam * 100)):02d}-S42"
        rows_f.append(dict(
            exp=e, arm=arm, lam=lam, n=len(y),
            rmsle=rmsle_z(y, z), rmsle_cal=rmsle_z(y, zk), offset=off[e],
            bias=bias_z(y, z), mean_z=float(z.mean()), std_z=float(z.std()),
            rmsle_pos=rmsle_z(y[pos], zk[pos]),
            rmsle_zero=rmsle_z(y[~pos], zk[~pos]),
            auc_y30_pos=auc(pos, z),
            d_cal_vs_BASE=(rmsle_z(y, zk) - rmsle_z(y, zc[base])) if have_base else None,
            d_cal_vs_BUYCTRL=(rmsle_z(y, zk) - rmsle_z(y, zc[buy]))
            if buy in zc and arm != "BUYCTRL" else None,
            d_auc_vs_BASE=(auc(pos, z) - auc(pos, data[base][1])) if have_base else None))
    # Каждая дельта — в единицах ЦЕНЫ ПРОГОНА. `noise` = |RMSLE(BASER2) − RMSLE(BASE)|:
    # дельта арки, не превосходящая её по модулю, результатом не считается.
    noise = None
    if have_base and CTRL in zc:
        noise = abs(rmsle_z(y, zc[CTRL]) - rmsle_z(y, zc[base]))
    for r in rows_f:
        r["noise_floor"] = noise
        r["d_over_noise"] = (r["d_cal_vs_BASE"] / noise
                             if noise and r["d_cal_vs_BASE"] is not None and noise > 0
                             else None)
    pl.DataFrame(rows_f).write_csv(OUT / "folds.csv")

    # ------------------------------------------------------------- segments.csv
    masks = seg_masks(uid0)
    rows_s = []
    for name, m in masks.items():
        if not m.any():
            continue
        for e, arm, lam in runs:
            zk = zc[e]
            buy = f"FNL-BUYCTRL-L{int(round(lam * 100)):02d}-S42"
            r = rmsle_z(y[m], zk[m])
            rows_s.append(dict(
                segment=name, exp=e, arm=arm, lam=lam, n=int(m.sum()),
                share=float(m.mean()),
                mse_share=float(((ly - zk) ** 2)[m].sum() / ((ly - zk) ** 2).sum()),
                rmsle=r, auc=auc(pos[m], data[e][1][m]),
                d_vs_BASE=(r - rmsle_z(y[m], zc[base][m])) if have_base else None,
                d_vs_BUYCTRL=(r - rmsle_z(y[m], zc[buy][m]))
                if buy in zc and arm != "BUYCTRL" else None,
                d_auc_vs_BASE=(auc(pos[m], data[e][1][m])
                               - auc(pos[m], data[base][1][m])) if have_base else None))
    pl.DataFrame(rows_s).write_csv(OUT / "segments.csv")

    # -------------------------------------------------------------- aux_auc.csv
    rows_a = []
    for e, arm, lam in runs:
        z = data[e][1]
        ap, hn = data[e][3], data[e][4]
        for j, nm in enumerate(hname):
            t = A_true[:, j]
            kind = kind_of[nm]
            own = None
            if nm in hn:
                p = ap[:, hn.index(nm)].astype(np.float64)
                own = (auc(t > 0.5, p) if kind == "bin"
                       else float(np.corrcoef(p, t)[0, 1]))
            main = (auc(t > 0.5, z) if kind == "bin"
                    else float(np.corrcoef(z, t)[0, 1]))
            rows_a.append(dict(exp=e, arm=arm, lam=lam, head=nm, kind=kind,
                               trained_on=nm in hn, prevalence=float(t.mean()),
                               own_head=own, main_z_head=main,
                               gap=(own - main) if own is not None else None))
    pl.DataFrame(rows_a).write_csv(OUT / "aux_auc.csv")

    # ------------------------------------------------------------ diversity.csv
    rows_d = []
    if have_base:
        rb = ly - zc[base]
        for e, arm, lam in runs:
            d = data[e][1] - data[base][1]
            r = ly - zc[e]
            rows_d.append(dict(exp=e, arm=arm, lam=lam,
                               var_dz_vs_BASE=float(np.var(d)),
                               corr_z_vs_BASE=float(np.corrcoef(data[e][1],
                                                                data[base][1])[0, 1])
                               if e != base else 1.0,
                               corr_resid_vs_BASE=float(np.corrcoef(r, rb)[0, 1])
                               if e != base else 1.0))
    pl.DataFrame(rows_d).write_csv(OUT / "diversity.csv")

    # --------------------------------------------------------------- curves.csv
    rows_c = []
    for e, arm, lam in runs:
        p = ARTIFACTS / f"curve_{e}-{TAG}.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8"))
        for h in c["hist"]:
            rows_c.append(dict(exp=e, arm=arm, lam=lam, **h))
    pl.DataFrame(rows_c).write_csv(OUT / "curves.csv")

    # ------------------------------------------------------------------- печать
    with pl.Config(tbl_rows=60, tbl_cols=20, fmt_str_lengths=40, tbl_width_chars=200):
        print("\n=== ФОЛД 2025-10-16 ===")
        print(pl.DataFrame(rows_f).select(
            "arm", "lam", "rmsle_cal", "d_cal_vs_BASE", "d_over_noise",
            "d_cal_vs_BUYCTRL", "auc_y30_pos", "d_auc_vs_BASE", "rmsle_pos"))
        if noise:
            print(f"\nцена прогона (|BASER2 − BASE|) = {noise:.5f}; "
                  f"дельты в её единицах — колонка d_over_noise")
        print("\n=== СЕГМЕНТЫ (главные) ===")
        print(pl.DataFrame(rows_s).filter(
            pl.col("segment").is_in(["rec_buy 15-60", "w180_days_buy 2-15",
                                     "пересечение 2-15 x 15-60",
                                     "никогда не покупал", "w180_days_buy 16+"]))
              .select("segment", "arm", "lam", "rmsle", "d_vs_BASE", "d_vs_BUYCTRL",
                      "auc", "d_auc_vs_BASE"))
        print("\n=== AUC ВСПОМОГАТЕЛЬНЫХ МЕТОК: своя голова против главной z ===")
        print(pl.DataFrame(rows_a).filter(pl.col("trained_on"))
              .select("arm", "lam", "head", "prevalence", "own_head", "main_z_head", "gap"))
        print("\n=== РАЗНООБРАЗИЕ ===")
        print(pl.DataFrame(rows_d))
    print(f"\nзаписано в {OUT}")


if __name__ == "__main__":
    main()
