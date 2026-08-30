"""Диагностика RMSLE боевой смеси `S1-DIST-MIX` — на сохранённых OOF, без обучения.

Работает поверх `fold_predictions.parquet` (`build_frame.py`). Все числа считаются
в лог-пространстве после ПОФОЛДОВОЙ калибровки — это ровно то, что меряет главная
метрика проекта wCV (`exp_016` §6): уровень сабмита ставится по измеренному якорю,
поэтому ошибка уровня в сравнение не входит.

Разделы = разделы README.md:
  1 error decomposition   2 calibration   3 zero/non-zero
  4 temporal stability    5 predictability baselines   6 test prediction shift

Запуск: python research/rmsle_diagnostics/diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl

from src.config import FOLD_WEIGHTS_S1, VAL_FOLDS_S1
from src.validation import calibrate, rmsle_z, wcv

HERE = Path(__file__).resolve().parent
DIAG = HERE / "diagnostics"
DIAG.mkdir(exist_ok=True)
FOLDS = [d.isoformat() for d in VAL_FOLDS_S1]
FW = np.asarray(FOLD_WEIGHTS_S1, float)
FW = FW / FW.sum()


def emit(name: str, df: pl.DataFrame) -> None:
    df.write_csv(DIAG / f"{name}.csv", float_precision=6)
    print(f"    -> diagnostics/{name}.csv  ({df.height} строк)")


# ------------------------------------------------------------------ сегментация
def bucket(v: np.ndarray, edges: list[float], labels: list[str],
           na_label: str | None = None) -> np.ndarray:
    """Метка бакета по возрастающим ПРАВЫМ границам.

    `labels` длиннее `edges` ровно на 1: последняя метка — переполнение.
    NaN получает `na_label` (или метку переполнения, если он не задан).
    """
    assert len(labels) == len(edges) + 1, "меток должно быть на одну больше, чем границ"
    out = np.full(len(v), labels[-1], dtype=object)
    for lab, hi in list(zip(labels[:-1], edges))[::-1]:
        out[v <= hi] = lab
    out[np.isnan(v)] = na_label if na_label is not None else labels[-1]
    return out


def deciles(v: np.ndarray, k: int = 10) -> np.ndarray:
    q = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    idx = np.searchsorted(q, v, "right")
    return np.array([f"D{i + 1:02d}" for i in idx], dtype=object)


def segments(f: pl.DataFrame) -> dict[str, np.ndarray]:
    """Разбиения популяции. Всё, кроме `outcome`/`gmv_bucket`, известно ДО cutoff'а."""
    y = f["y"].to_numpy()
    return {
        "outcome": np.where(y > 0, "y>0", "y=0"),
        "gmv_bucket": bucket(y, [0.0, 500, 2000, 5000, 15000, 50000],
                             ["y=0", "0-500", "500-2k", "2k-5k", "5k-15k", "15k-50k", "50k+"]),
        "recency": bucket(f["rec_buy"].to_numpy(), [7, 14, 30, 60, 90, 180],
                          ["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181+"],
                          na_label="никогда не покупал"),
        "pred_decile": deciles(f["z_cal"].to_numpy()),
        "lifecycle": bucket(f["w180_days_buy"].to_numpy(), [0, 1, 3, 7, 15, 30],
                            ["0 дней покупок", "1", "2-3", "4-7", "8-15", "16-30", "31+"]),
        "recent_spend": bucket(f["w30_gmv"].to_numpy(), [0.0, 1000, 5000, 20000],
                               ["нет трат за 30д", "0-1k", "1k-5k", "5k-20k", "20k+"]),
    } | {"lifecycle x outcome": np.char.add(
        np.char.add(bucket(f["w180_days_buy"].to_numpy(), [0, 1, 3, 7, 15, 30],
                           ["0 дней покупок", "1", "2-3", "4-7", "8-15", "16-30", "31+"]
                           ).astype("U16"), " | "),
        np.where(f["y"].to_numpy() > 0, "y>0", "y=0")).astype(object)}


# ------------------------------------------------------- 1. error decomposition
def decompose(f: pl.DataFrame, name: str, seg: np.ndarray) -> pl.DataFrame:
    """users% | error_share | RMSLE | mean_actual | mean_pred | log_bias по фолдам и в сумме.

    error_share — доля группы в СУММЕ квадратов лог-ошибки фолда: именно она,
    а не локальный RMSLE, говорит, сколько метрики стоит группа.
    """
    ly, z, cut = f["ly"].to_numpy(), f["z_cal"].to_numpy(), f["cutoff"].to_numpy()
    y = f["y"].to_numpy()
    e = (ly - z) ** 2
    rows = []
    for grp in sorted(set(seg.tolist())):
        gm = seg == grp
        acc = {}
        for c in FOLDS:
            m = gm & (cut == c)
            fm = cut == c
            acc[c] = dict(users=m.sum() / fm.sum(), share=e[m].sum() / e[fm].sum(),
                          rmsle=np.sqrt(e[m].mean()) if m.any() else np.nan)
        rows.append(dict(
            segmentation=name, segment=str(grp),
            users_pct=float(np.dot(FW, [acc[c]["users"] for c in FOLDS])),
            error_share=float(np.dot(FW, [acc[c]["share"] for c in FOLDS])),
            rmsle=float(np.sqrt(e[gm].mean())),
            mean_actual_gmv=float(y[gm].mean()), mean_actual_log=float(ly[gm].mean()),
            mean_pred_log=float(z[gm].mean()), log_bias=float((ly[gm] - z[gm]).mean()),
            **{f"share_{c[5:]}": acc[c]["share"] for c in FOLDS},
            **{f"rmsle_{c[5:]}": acc[c]["rmsle"] for c in FOLDS}))
    return pl.DataFrame(rows).sort("error_share", descending=True)


# ------------------------------------------------------------------ 2. calibration
def murphy(ly: np.ndarray, z: np.ndarray, k: int = 20) -> dict:
    """Разложение MSE = uncertainty − resolution + reliability по бинам прогноза.

    reliability — то, что снимает ЛЮБАЯ монотонная 1D-перекалибровка (её потолок);
    resolution — сколько модель уже различает; uncertainty — дисперсия таргета.
    """
    q = np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])
    b = np.searchsorted(q, z, "right")
    n = len(ly)
    rel = res = 0.0
    ybar = ly.mean()
    for i in range(k):
        m = b == i
        if not m.any():
            continue
        rel += m.sum() * (ly[m].mean() - z[m].mean()) ** 2
        res += m.sum() * (ly[m].mean() - ybar) ** 2
    return dict(mse=float(((ly - z) ** 2).mean()), uncertainty=float(ly.var()),
                resolution=float(res / n), reliability=float(rel / n))


def affine_fit(ly: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """z_cal = a*z + b по МНК — оптимальная аффинная перекалибровка в лог-пространстве."""
    a, b = np.polyfit(z, ly, 1)
    return float(a), float(b)


def isotonic_ceiling(ly: np.ndarray, z: np.ndarray, k: int = 200) -> float:
    """Потолок ЛЮБОЙ монотонной 1D-перекалибровки: среднее ly в 200 бинах z (in-sample)."""
    q = np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])
    b = np.searchsorted(q, z, "right")
    out = np.empty_like(z)
    for i in range(k):
        m = b == i
        if m.any():
            out[m] = ly[m].mean()
    return float(np.sqrt(((ly - out) ** 2).mean()))


def temporal_safe_calibration(f: pl.DataFrame) -> pl.DataFrame:
    """Аффинная и посегментная калибровка, обученная ТОЛЬКО на прошлых фолдах."""
    rows = []
    for i, c in enumerate(FOLDS):
        cur = f.filter(pl.col("cutoff") == c)
        ly_v, z_v = cur["ly"].to_numpy(), cur["z_mix"].to_numpy()
        base = calibrate(cur["y"].to_numpy(), z_v)[1]          # текущая схема: только сдвиг
        r = dict(fold=c, n=cur.height, base_shift_only=base)
        if i == 0:
            r |= dict(a=np.nan, b=np.nan, affine=np.nan, d_affine=np.nan,
                      seg4=np.nan, d_seg4=np.nan)
            rows.append(r)
            continue
        past = f.filter(pl.col("cutoff").is_in(FOLDS[:i]))
        ly_p, z_p = past["ly"].to_numpy(), past["z_mix"].to_numpy()
        a, b = affine_fit(ly_p, z_p)
        # уровень всё равно ставится якорем -> после аффинного сжатия снимаем сдвиг заново
        za = np.maximum(a * z_v + b, 0.0)
        aff = calibrate(cur["y"].to_numpy(), za)[1]
        # 4 крупных сегмента: сдвиг внутри сегмента, оценённый на прошлых фолдах
        sp, sv = seg4(past), seg4(cur)
        zs = z_v.copy()
        for s in np.unique(sv):
            mp, mv = sp == s, sv == s
            zs[mv] = z_v[mv] + (ly_p[mp] - z_p[mp]).mean() if mp.any() else z_v[mv]
        seg = calibrate(cur["y"].to_numpy(), np.maximum(zs, 0.0))[1]
        rows.append(r | dict(a=a, b=b, affine=aff, d_affine=aff - base,
                             seg4=seg, d_seg4=seg - base))
    return pl.DataFrame(rows)


def seg4(f: pl.DataFrame) -> np.ndarray:
    """Четыре крупных устойчивых сегмента по частоте покупок за 180 дней."""
    db = np.nan_to_num(f["w180_days_buy"].to_numpy(), nan=0.0)
    return bucket(db, [1, 4, 12], ["0-1", "2-4", "5-12", "13+"])


# ------------------------------------------------------------- 5. baselines
def naive_baselines(f: pl.DataFrame) -> pl.DataFrame:
    """Простые временные предикторы, калиброванные ТАК ЖЕ, как модель (пофолдовый сдвиг)."""
    g30, g60, g90 = (f[c].to_numpy() for c in ("w30_gmv", "w60_gmv", "w90_gmv"))
    g180, g365 = f["w180_gmv"].to_numpy(), f["w365_gmv"].to_numpy()
    b30 = g30                                  # последний 30-дневный блок
    b60 = np.maximum(g60 - g30, 0.0)           # предыдущий
    b90 = np.maximum(g90 - g60, 0.0)           # позапрошлый
    ew = 0.5 * b30 + 0.3 * b60 + 0.2 * b90     # EWMA по трём 30-дневным блокам
    cand = {
        "константа (оптимальная)": np.zeros(f.height),
        "персистентность w30_gmv": np.log1p(b30),
        "среднее 2 блоков по 30д": np.log1p(0.5 * (b30 + b60)),
        "среднее 3 блоков по 30д": np.log1p(g90 / 3.0),
        "EWMA 0.5/0.3/0.2": np.log1p(ew),
        "w180_gmv / 6": np.log1p(g180 / 6.0),
        "w365_gmv / 12.17": np.log1p(g365 / 12.17),
        "смесь S1-DIST-MIX": f["z_mix"].to_numpy(),
    }
    y, cut = f["y"].to_numpy(), f["cutoff"].to_numpy()
    rows = []
    for name, z in cand.items():
        sc = [calibrate(y[cut == c], z[cut == c])[1] for c in FOLDS]
        rows.append(dict(baseline=name, wcv=wcv(sc),
                         **{f"fold_{c[5:]}": s for c, s in zip(FOLDS, sc)}))
    return pl.DataFrame(rows).sort("wcv")


def persistence(f: pl.DataFrame) -> pl.DataFrame:
    """Переносится ли ошибка ОДНОГО пользователя между непересекающимися окнами таргета.

    Окна: фолд 09-04 -> (09-05..10-04], фолд 10-16 -> (10-17..11-15]. Пересечения нет.
    """
    a = f.filter(pl.col("cutoff") == "2025-09-04").select(
        ["user_id", "ly", "z_cal", "w30_gmv"])
    b = f.filter(pl.col("cutoff") == "2025-10-16").select(
        ["user_id", pl.col("ly").alias("ly_b"), pl.col("z_cal").alias("z_b")])
    j = a.join(b, on="user_id", how="inner")
    ra = (j["ly"] - j["z_cal"]).to_numpy()
    rb = (j["ly_b"] - j["z_b"]).to_numpy()
    return pl.DataFrame([dict(
        pair="2025-09-04 -> 2025-10-16", n=j.height,
        corr_residual=float(np.corrcoef(ra, rb)[0, 1]),
        corr_target=float(np.corrcoef(j["ly"].to_numpy(), j["ly_b"].to_numpy())[0, 1]),
        corr_w30gmv_next_target=float(np.corrcoef(
            np.log1p(j["w30_gmv"].to_numpy()), j["ly"].to_numpy())[0, 1]))])


# --------------------------------------------------------------------- главный ход
def main() -> None:
    f = pl.read_parquet(HERE / "fold_predictions.parquet")
    y, ly, z, cut = (f["y"].to_numpy(), f["ly"].to_numpy(),
                     f["z_cal"].to_numpy(), f["cutoff"].to_numpy())

    print("\n=== 0. воспроизведение боевой точки ===")
    fold_cal = [calibrate(y[cut == c], f["z_mix"].to_numpy()[cut == c])[1] for c in FOLDS]
    print(f"  wCV = {wcv(fold_cal):.5f}   (STATE.md: 1.74948)")
    print("  пофолдово: " + " ".join(f"{s:.5f}" for s in fold_cal))

    print("\n=== 1. error decomposition ===")
    segs = segments(f)
    dec = pl.concat([decompose(f, k, v) for k, v in segs.items()])
    emit("error_decomposition", dec)
    top = dec.filter(pl.col("segmentation") != "pred_decile").sort("error_share",
                                                                   descending=True)
    print(top.select(["segmentation", "segment", "users_pct", "error_share",
                      "rmsle", "log_bias"]).head(8))

    print("\n=== 2. calibration ===")
    mur = pl.DataFrame([dict(fold=c, **murphy(ly[cut == c], z[cut == c])) for c in FOLDS]
                       + [dict(fold="pooled", **murphy(ly, z))])
    mur = mur.with_columns(reliability_share=pl.col("reliability") / pl.col("mse"))
    emit("murphy_decomposition", mur)
    print(mur)
    cal = temporal_safe_calibration(f)
    emit("calibration_temporal_safe", cal)
    print(cal)
    iso = [dict(fold=c, shift_only=calibrate(y[cut == c], f["z_mix"].to_numpy()[cut == c])[1],
                isotonic_ceiling_insample=isotonic_ceiling(ly[cut == c], z[cut == c]))
           for c in FOLDS]
    iso = pl.DataFrame(iso).with_columns(gain=pl.col("isotonic_ceiling_insample")
                                         - pl.col("shift_only"))
    emit("calibration_isotonic_ceiling", iso)
    print(iso)

    # калибровочная кривая по децилям прогноза
    curve = decompose(f, "pred_decile", segs["pred_decile"]).sort("segment")
    emit("calibration_curve", curve.select(["segment", "users_pct", "error_share", "rmsle",
                                            "mean_pred_log", "mean_actual_log", "log_bias"]))

    print("\n=== 3. zero / non-zero ===")
    rows = []
    for c in FOLDS + ["pooled"]:
        m = np.ones(len(y), bool) if c == "pooled" else cut == c
        z0 = z[m][y[m] == 0]
        e = (ly[m] - z[m]) ** 2
        rows.append(dict(fold=c, n=int(m.sum()), zero_rate=float((y[m] == 0).mean()),
                         zero_error_share=float(e[y[m] == 0].sum() / e.sum()),
                         mean_pred_on_zeros=float(z0.mean()),
                         rmsle_zeros=float(np.sqrt(e[y[m] == 0].mean())),
                         rmsle_pos=float(np.sqrt(e[y[m] > 0].mean())),
                         auc_zero=auc(y[m] > 0, z[m])))
    zt = pl.DataFrame(rows)
    emit("zero_nonzero", zt)
    print(zt)

    print("\n  оракулы (сколько метрики лежит в каждой части задачи):")
    orc = oracles(f)
    emit("oracle_bounds", orc)
    print(orc)

    print("\n  правила неактивности и положительный пол (temporal-safe):")
    rules = inactivity_rules(f)
    emit("inactivity_rules", rules)
    print(rules)

    print("\n  из чего состоит остаточная дисперсия (активность против величины):")
    vs = variance_split(f)
    emit("variance_split", vs)
    print(vs)

    print("\n  остаточная линейная структура по признакам:")
    rs = residual_structure(f)
    emit("residual_structure", rs)
    print(rs)

    print("\n=== 4. temporal stability ===")
    st = stability(f)
    emit("fold_stability", st)
    print(st)

    print("\n=== 5. predictability baselines ===")
    nb = naive_baselines(f)
    emit("naive_baselines", nb)
    print(nb)
    pr = persistence(f)
    emit("persistence", pr)
    print(pr)

    print("\n=== 6. test prediction shift ===")
    ts = test_shift(f)
    emit("test_shift", ts)
    print(ts)


def auc(pos: np.ndarray, s: np.ndarray) -> float:
    """AUC через ранги — без sklearn."""
    r = np.empty(len(s), float)
    o = np.argsort(s, kind="stable")
    r[o] = np.arange(1, len(s) + 1)
    # средние ранги для связок
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n1 = int(pos.sum())
    n0 = len(s) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oracles(f: pl.DataFrame) -> pl.DataFrame:
    """Верхние границы: что осталось бы от ошибки при идеальном знании части задачи."""
    rows = []
    for c in FOLDS:
        m = f.filter(pl.col("cutoff") == c)
        y, ly, z = m["y"].to_numpy(), m["ly"].to_numpy(), m["z_cal"].to_numpy()
        pos = y > 0
        # A. идеальный классификатор: нулям ставим 0, остальным — текущий прогноз,
        #    перекалиброванный внутри положительных
        za = z.copy()
        za[~pos] = 0.0
        za[pos] = z[pos] + (ly[pos] - z[pos]).mean()
        # B. идеальная величина: знаем ly для покупающих, классификатор текущий
        zb = z.copy()
        zb[pos] = ly[pos]
        rows.append(dict(fold=c, current=rmsle_z(y, z),
                         oracle_zero=rmsle_z(y, np.maximum(za, 0)),
                         oracle_magnitude=rmsle_z(y, np.maximum(zb, 0))))
    d = pl.DataFrame(rows)
    return d.with_columns(gain_zero=pl.col("oracle_zero") - pl.col("current"),
                          gain_magnitude=pl.col("oracle_magnitude") - pl.col("current"))


def inactivity_rules(f: pl.DataFrame) -> pl.DataFrame:
    """Простые правила: обнулить/поджать давно неактивных; пол снизу. Обучены на прошлых фолдах."""
    rows = []
    for thr in (90.0, 120.0, 180.0):
        for i, c in enumerate(FOLDS):
            if i == 0:
                continue
            cur = f.filter(pl.col("cutoff") == c)
            past = f.filter(pl.col("cutoff").is_in(FOLDS[:i]))
            rec_c = np.nan_to_num(cur["rec_buy"].to_numpy(), nan=9999.0)
            rec_p = np.nan_to_num(past["rec_buy"].to_numpy(), nan=9999.0)
            zc, yc = cur["z_mix"].to_numpy(), cur["y"].to_numpy()
            base = calibrate(yc, zc)[1]
            mp = rec_p > thr
            # оптимальный сдвиг ВНУТРИ сегмента, оценённый на прошлых фолдах
            d = float((past["ly"].to_numpy()[mp] - past["z_mix"].to_numpy()[mp]).mean()) \
                if mp.any() else 0.0
            zz = zc.copy()
            zz[rec_c > thr] += d
            rows.append(dict(rule=f"rec_buy>{thr:.0f}: сдвиг из прошлых фолдов",
                             fold=c, seg_share=float((rec_c > thr).mean()), delta=d,
                             base=base, after=calibrate(yc, np.maximum(zz, 0))[1],
                             gain=calibrate(yc, np.maximum(zz, 0))[1] - base))
    for floor in (0.05, 0.10, 0.20):
        for c in FOLDS:
            cur = f.filter(pl.col("cutoff") == c)
            zc, yc = cur["z_mix"].to_numpy(), cur["y"].to_numpy()
            base = calibrate(yc, zc)[1]
            after = calibrate(yc, np.maximum(zc, floor))[1]
            rows.append(dict(rule=f"пол z >= {floor}", fold=c,
                             seg_share=float((zc < floor).mean()), delta=floor,
                             base=base, after=after, gain=after - base))
    return pl.DataFrame(rows)


def variance_split(f: pl.DataFrame, k: int = 50) -> pl.DataFrame:
    """Из чего состоит остаточная дисперсия внутри бина прогноза.

    Внутри узкого бина прогноза `ly` — смесь точечной массы в нуле и положительной
    величины, поэтому

        Var(ly) = p(1-p)*mu^2  +  p*sigma^2
                  ^ «купит или нет»   ^ «сколько именно»

    Первое слагаемое — цена незнания АКТИВНОСТИ, второе — незнания ВЕЛИЧИНЫ.
    Их отношение говорит, в какой из двух подзадач лежит остаток ошибки.
    """
    rows = []
    for c in FOLDS:
        m = f.filter(pl.col("cutoff") == c)
        ly, z, y = m["ly"].to_numpy(), m["z_cal"].to_numpy(), m["y"].to_numpy()
        q = np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])
        b = np.searchsorted(q, z, "right")
        posv = y > 0
        act = mag = tot = 0.0
        wp = wm = nm = 0.0          # для доли объяснённой дисперсии подзадач
        for i in range(k):
            s = b == i
            if s.sum() < 50:
                continue
            pos = y[s] > 0
            p = pos.mean()
            mu = ly[s][pos].mean() if pos.any() else 0.0
            sg = ly[s][pos].var() if pos.sum() > 1 else 0.0
            act += s.sum() * p * (1 - p) * mu ** 2
            mag += s.sum() * p * sg
            tot += s.sum() * ly[s].var()
            wp += s.sum() * p * (1 - p)
            if pos.sum() > 1:
                wm += pos.sum() * sg
                nm += pos.sum()
        pbar = posv.mean()
        rows.append(dict(fold=c, bins=k, var_within=tot / len(ly),
                         var_activity=act / len(ly), var_magnitude=mag / len(ly),
                         share_activity=act / tot, share_magnitude=mag / tot,
                         explained_activity=1 - (wp / len(ly)) / (pbar * (1 - pbar)),
                         explained_magnitude=1 - (wm / nm) / ly[posv].var(),
                         explained_total=1 - ((ly - z) ** 2).mean() / ly.var(),
                         sigma_pos_within=float(np.sqrt(wm / nm)),
                         sigma_pos_uncond=float(ly[posv].std())))
    return pl.DataFrame(rows)


def residual_structure(f: pl.DataFrame) -> pl.DataFrame:
    """Осталась ли в остатке ЛИНЕЙНАЯ связь с признаками — грубая проба на недоучёт."""
    cols = ["rec_buy", "rec_any", "w30_gmv", "w90_gmv", "w180_gmv", "w365_gmv",
            "w30_days_buy", "w90_days_buy", "w180_days_buy", "w365_days_buy",
            "w30_days_present", "w90_days_present", "buygap_mean", "tenure",
            "w30_searches", "w90_searches", "w180_lgmv_mean"]
    rows = []
    for c in FOLDS:
        m = f.filter(pl.col("cutoff") == c)
        r = (m["ly"] - m["z_cal"]).to_numpy()
        out = dict(fold=c)
        for col in cols:
            v = m[col].to_numpy().astype(float)
            v = np.log1p(np.nan_to_num(v, nan=np.nanmax(v[np.isfinite(v)]) if
                                       np.isnan(v).any() else 0.0))
            out[col] = float(np.corrcoef(v, r)[0, 1])
        rows.append(out)
    return pl.DataFrame(rows)


def stability(f: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for c in FOLDS:
        m = f.filter(pl.col("cutoff") == c)
        y, ly = m["y"].to_numpy(), m["ly"].to_numpy()
        zm, zc = m["z_mix"].to_numpy(), m["z_cal"].to_numpy()
        d, sc = calibrate(y, zm)
        rows.append(dict(fold=c, n=m.height, rmsle_raw=rmsle_z(y, zm), rmsle_cal=sc,
                         cal_delta=d, zero_rate=float((y == 0).mean()),
                         mean_ly=float(ly.mean()), std_ly=float(ly.std()),
                         mean_z=float(zm.mean()), std_z=float(zc.std()),
                         p10_z=float(np.quantile(zc, .1)), p50_z=float(np.quantile(zc, .5)),
                         p90_z=float(np.quantile(zc, .9)),
                         log_bias=float(ly.mean() - zm.mean()),
                         spread_ratio=float(zc.std() / ly.std()),
                         auc_zero=auc(y > 0, zm)))
    return pl.DataFrame(rows)


def test_shift(f: pl.DataFrame) -> pl.DataFrame:
    t = pl.read_parquet(HERE / "test_predictions.parquet")

    def row(name, z_raw, z_cal, d, ly=None, y=None):
        """`persist` — mean(log1p(gmv за 30 дней ДО cutoff'а)): наблюдаемый и на тесте,
        поэтому это единственный мост между популяциями, не требующий таргета."""
        return dict(
            source=name, n=len(z_cal), mean_z_raw=float(z_raw.mean()),
            mean_z=float(z_cal.mean()), std_z=float(z_cal.std()),
            **{f"p{q}": float(np.quantile(z_cal, q / 100))
               for q in (10, 25, 50, 75, 90, 99)},
            share_z_lt_05=float((z_cal < 0.5).mean()),
            med_rec_buy=float(np.nanmedian(d["rec_buy"].to_numpy())),
            med_w180_days_buy=float(np.nanmedian(d["w180_days_buy"].to_numpy())),
            persist_level=float(np.log1p(d["w30_gmv"].to_numpy()).mean()),
            mean_ly=float(ly.mean()) if ly is not None else np.nan,
            target_over_persist=float(ly.mean() / np.log1p(d["w30_gmv"].to_numpy()).mean())
            if ly is not None else np.nan,
            zero_rate=float((y == 0).mean()) if y is not None else np.nan)

    rows = [row("test 2026-02-13", t["z_mix"].to_numpy(), t["z_cal"].to_numpy(), t)]
    for c in FOLDS:
        m = f.filter(pl.col("cutoff") == c)
        rows.append(row(f"OOF {c}", m["z_mix"].to_numpy(), m["z_cal"].to_numpy(), m,
                        m["ly"].to_numpy(), m["y"].to_numpy()))
    return pl.DataFrame(rows)


if __name__ == "__main__":
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_float("full")
    main()
