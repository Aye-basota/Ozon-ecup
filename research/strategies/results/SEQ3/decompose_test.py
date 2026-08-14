"""Этап 1 — разложение провалившегося сабмита по ОДНОЙ оси за раз.

`submission_SEQAVG3_mix.csv` дал LB **1.6553136** против **1.6501764** у
`submission_SEQ01_mix.csv`: **+0.0051372**, то есть 20 парных SE (0.00025) в
худшую сторону. Между двумя сабмитами изменились ТРИ вещи сразу:

  1. усреднение сидов        `SEQ-01-S42` -> `mean(42, 43, 44)`;
  2. веса смеси              0.15/0.20/0.10/0.25/**0.30** -> 0.10/0.15/0.10/0.20/**0.45**;
  3. политика глубины        `--depth-clip 289` -> полные 365.

Тестовых меток нет, поэтому RMSLE каждого варианта посчитать нельзя. Но можно
посчитать, НАСКОЛЬКО каждая ось двигает сам прогноз, и перевести это в ожидаемый
ущерб. Уровень сабмита фиксируется якорем `L* = 2.3293`, поэтому сдвиг среднего
роли не играет и всё решает `Var(Δz)`:

    RMSLE² = E[(ly − z)²];   z' = z + d,  E[d] снимается якорем
    ΔRMSLE ≈ (Var(d) + 2·Cov(d, z − ly)) / (2·RMSLE)

Если возмущение `d` не несёт сигнала (не коррелирует с ошибкой), ущерб равен
`Var(d) / (2·RMSLE)` — это ВЕРХНЯЯ оценка вреда и одновременно точная оценка для
чистого шума. Сравнение этой оценки с фактическими +0.0051372 и есть проверка
гипотезы «виновата глубина».

Реконструкция вариантов. Точно восстанавливаются оба отправленных сабмита и
любой вариант на полной глубине. Для `AVG3` при обрезке 289 нет тестовых
прогнозов сидов 43/44 (сохранены только `FULL`), а переобучать финальные модели
запрещено, поэтому используется допущение

    Δглубины одинаково у сидов:   z_i,full − z_i,clip ≈ z_42,full − z_42,clip

Оно не постулируется, а ПРОВЕРЯЕТСЯ на кросс-фолдовом стресс-тесте
(`xdepth_*_on1016.npz`, три сида на одной панели) — см. `--check-seed-common`.

Запуск: PYTHONPATH=. python research/strategies/results/SEQ3/decompose_test.py
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS, CUTOFF_TEST, DATA_START
from src.features import features_cached

OUT = Path(__file__).parent
LEVEL = 2.3293                    # якорь уровня, одинаковый у обоих сабмитов
LB = {"SEQ01_MIX": 1.650176372731295, "AVG3_MIX": 1.6553135958569027}
RMSLE0 = LB["SEQ01_MIX"]
SEQ_L, CLIP = 365, 289

# веса GBDT-части: (S1-NORM, S1-UNC, S1-CAP, S1-DIST), доля SEQ = 1 − сумма
W_OLD = dict(zip(["S1-NORM", "S1-UNC", "S1-CAP", "S1-DIST"], [0.15, 0.20, 0.10, 0.25]))
W_NEW = dict(zip(["S1-NORM", "S1-UNC", "S1-CAP", "S1-DIST"], [0.10, 0.15, 0.10, 0.20]))


def z(name: str) -> np.ndarray:
    return np.load(ARTIFACTS / f"ztest_{name}.npy")


def mix(w: dict, z_seq: np.ndarray) -> np.ndarray:
    w_seq = 1.0 - sum(w.values())
    out = w_seq * z_seq
    for n, v in w.items():
        out = out + v * z(n)
    return out


def anchored(zz: np.ndarray) -> np.ndarray:
    """То, что реально уезжает в сабмит: `max(z + δ, 0)` при δ = L* − mean(z)."""
    return np.maximum(zz + (LEVEL - float(zz.mean())), 0.0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pair_stats(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """Всё про переход a -> b в лог-пространстве (после якоря — как в сабмите)."""
    d = b - a
    q = np.quantile(d, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return dict(pair=name, var=float(np.var(d)), mean=float(d.mean()), std=float(d.std()),
                pearson=float(np.corrcoef(a, b)[0, 1]), spearman=spearman(a, b),
                p01=q[0], p05=q[1], p25=q[2], p50=q[3], p75=q[4], p95=q[5], p99=q[6],
                sh_010=float((np.abs(d) > 0.10).mean()), sh_025=float((np.abs(d) > 0.25).mean()),
                sh_050=float((np.abs(d) > 0.50).mean()),
                exp_drmsle=float(np.var(d) / (2 * RMSLE0)))


def segments(uid: np.ndarray) -> dict[str, np.ndarray]:
    """Сегменты диагностики проекта на ТЕСТОВОМ cutoff'е + «есть ли активность в
    76 добавляемых старых днях» — то есть ровно у кого 289->365 вообще что-то меняет."""
    f = features_cached(CUTOFF_TEST, None, True)
    f = pl.DataFrame({"user_id": uid}).join(f, on="user_id", how="left")
    rec = f["rec_buy"].to_numpy()
    d180 = f["w180_days_buy"].to_numpy()
    ten = f["tenure"].to_numpy() if "tenure" in f.columns else np.full(len(uid), np.nan)

    # активность в (T-365, T-289] — по плотной панели, без агрегатов
    from src.seq import day_index, panel, user_rows
    p, _, _, _ = panel()
    dT = day_index(CUTOFF_TEST)
    lo, hi = dT - SEQ_L + 1, dT - CLIP + 1          # добавляемый кусок, 76 дней
    rows = user_rows(uid)
    from src.seq import CHANNELS
    old = p[rows, lo:hi, :][:, :, [CHANNELS.index("present"), CHANNELS.index("buy")]]
    n_old_days = old[:, :, 0].astype(np.float32).sum(1)
    n_old_buys = old[:, :, 1].astype(np.float32).sum(1)

    s = {
        "ВСЕ": np.ones(len(uid), bool),
        "rec_buy 15-60": (rec >= 15) & (rec <= 60),
        "полоса 2-15 (w180_days_buy)": (d180 >= 2) & (d180 <= 15),
        "никогда не покупал": ~np.isfinite(rec) | (rec > 400),
        "старый кусок: ПУСТ": n_old_days == 0,
        "старый кусок: есть дни": n_old_days > 0,
        "старый кусок: есть покупки": n_old_buys > 0,
    }
    if np.isfinite(ten).any():
        s["tenure < 180"] = ten < 180
        s["tenure >= 300"] = ten >= 300
    return s, n_old_days, n_old_buys


def check_seed_common() -> None:
    """Проверка допущения «Δ глубины одинаково у сидов» на кросс-фолдовом стрессе."""
    files = sorted(ARTIFACTS.glob("xdepth_*-V0904_on1016.npz"))
    if len(files) < 2:
        print("\n[проверка допущения] нет двух сидов кросс-фолдового стресса — пропуск")
        return
    ds, names = [], []
    for f in files:
        d = np.load(f)
        dep = list(d["depths"])
        i_lo, i_hi = dep.index(int(d["train_max"])), dep.index(int(d["avail"]))
        ds.append(d["z"][i_hi] - d["z"][i_lo])
        names.append(f.stem.replace("xdepth_", "").replace("_on1016", ""))
    print("\n[проверка допущения] Δ(глубина 212->289) на ОДНОЙ панели 10-16, разные сиды:")
    for i in range(len(ds)):
        print(f"  {names[i]:>22}: Var(Δ) = {np.var(ds[i]):.5f}, mean {ds[i].mean():+.4f}")
    rows = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            v_dd = float(np.var(ds[i] - ds[j]))
            v_m = float(np.mean([np.var(ds[i]), np.var(ds[j])]))
            rows.append(dict(a=names[i], b=names[j], corr=float(np.corrcoef(ds[i], ds[j])[0, 1]),
                             var_of_diff=v_dd, var_mean=v_m, share_seed_specific=v_dd / (2 * v_m)))
            print(f"  {names[i]} vs {names[j]}: corr(Δ_i, Δ_j) = {rows[-1]['corr']:.4f}, "
                  f"Var(Δ_i − Δ_j) = {v_dd:.5f} = {v_dd / (2 * v_m):.1%} от Var(Δ) "
                  f"(доля сид-специфичного)")
    pl.DataFrame(rows).write_csv(OUT / "seed_common_depth.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    uid = np.load(ARTIFACTS / "uid_SEQ-01.npy")

    z42c, z42f = z("SEQ-01"), z("SEQ-01-FULL")
    z43f, z44f = z("SEQ-S43-FULL"), z("SEQ-S44-FULL")
    d_depth = z42f - z42c                      # измеренный сдвиг от политики глубины
    avg3f = (z42f + z43f + z44f) / 3.0
    avg3c = avg3f - d_depth                    # допущение, проверяется check_seed_common

    print(f"тестовых строк {len(uid):,}; уровень якоря L* = {LEVEL}")
    print(f"  сид 42: mean z clip289 {z42c.mean():.4f} -> full365 {z42f.mean():.4f} "
          f"({z42f.mean() - z42c.mean():+.4f}), Var(Δ) = {np.var(d_depth):.5f}, "
          f"corr = {np.corrcoef(z42c, z42f)[0, 1]:.5f}")
    print(f"  mean z сидов на полной глубине: 42 {z42f.mean():.4f}  43 {z43f.mean():.4f}  "
          f"44 {z44f.mean():.4f}  -> AVG3 {avg3f.mean():.4f}")

    V = {
        "1_SEQ01_MIX_clip":  anchored(mix(W_OLD, z42c)),      # ТОЧНО = отправленный SEQ-01-MIX
        "2_oldW_AVG3_clip":  anchored(mix(W_OLD, avg3c)),
        "3_newW_AVG3_clip":  anchored(mix(W_NEW, avg3c)),
        "4_oldW_AVG3_full":  anchored(mix(W_OLD, avg3f)),
        "5_newW_AVG3_full":  anchored(mix(W_NEW, avg3f)),     # ТОЧНО = submission_SEQAVG3_mix
        "1b_oldW_S42_full":  anchored(mix(W_OLD, z42f)),      # только глубина, при w_SEQ=0.30
    }

    # --- сверка с реальными файлами сабмитов: реконструкция обязана быть точной ---
    from src.config import SUBMISSIONS
    for tag, f in [("1_SEQ01_MIX_clip", "submission_SEQ01_mix.csv"),
                   ("5_newW_AVG3_full", "submission_SEQAVG3_mix.csv")]:
        p = SUBMISSIONS / f
        if not p.exists():
            print(f"  [сверка] нет {f}")
            continue
        sub = pl.read_csv(p)
        got = (pl.DataFrame({"user_id": uid, "z": V[tag]})
               .join(sub, on="user_id", how="inner").sort("user_id"))
        err = float(np.abs(np.log1p(got["predict"].to_numpy()) - got["z"].to_numpy()).max())
        print(f"  [сверка] {f}: max|log1p(файл) − реконструкция| = {err:.2e} "
              f"{'OK' if err < 2e-5 else '!!! РАСХОЖДЕНИЕ'}")

    # --- пары: ровно одна ось за раз ------------------------------------------
    # Путь 1 -> 1b -> 4 -> 5 проходит ТОЛЬКО через точно восстановимые варианты
    # (никакого допущения о сидах): сначала глубина при одном сиде, потом
    # усреднение уже на полной глубине, потом веса. Пары A/B/C с приставкой
    # «прибл.» используют реконструкцию `AVG3 @ clip` и даны для полноты.
    pairs = [
        ("ТОЧНО C''. глубина  (1 -> 1b)  старые веса, один сид", "1_SEQ01_MIX_clip",
         "1b_oldW_S42_full"),
        ("ТОЧНО A'.  усреднение (1b -> 4) full, старые веса", "1b_oldW_S42_full",
         "4_oldW_AVG3_full"),
        ("ТОЧНО D.   веса      (4 -> 5)  full, AVG3", "4_oldW_AVG3_full", "5_newW_AVG3_full"),
        ("ТОЧНО ИТОГО  отправленный -> провалившийся", "1_SEQ01_MIX_clip", "5_newW_AVG3_full"),
        ("прибл. A.  усреднение (1 -> 2)  clip, старые веса", "1_SEQ01_MIX_clip",
         "2_oldW_AVG3_clip"),
        ("прибл. B.  веса      (2 -> 3)  clip, AVG3", "2_oldW_AVG3_clip", "3_newW_AVG3_clip"),
        ("прибл. C.  ГЛУБИНА   (3 -> 5)  новые веса, AVG3", "3_newW_AVG3_clip",
         "5_newW_AVG3_full"),
        ("прибл. C'. ГЛУБИНА   (2 -> 4)  старые веса, AVG3", "2_oldW_AVG3_clip",
         "4_oldW_AVG3_full"),
    ]
    rows = [dict(pair_stats(V[b], V[c], name)) for name, b, c in pairs]
    print(f"\n{'ось':<52}{'Var(Δz)':>10}{'ожид. ΔRMSLE':>14}{'corr':>9}{'|Δ|>0.25':>10}")
    for r in rows:
        print(f"{r['pair']:<52}{r['var']:>10.5f}{r['exp_drmsle']:>+14.5f}"
              f"{r['pearson']:>9.5f}{r['sh_025']:>10.2%}")
    tot = [r for r in rows if "ИТОГО" in r["pair"]][0]
    print(f"\nфакт LB: {LB['SEQ01_MIX']:.7f} -> {LB['AVG3_MIX']:.7f} = "
          f"{LB['AVG3_MIX'] - LB['SEQ01_MIX']:+.7f}")
    print(f"верхняя оценка вреда по Var(Δ) всей перестройки: "
          f"{tot['exp_drmsle']:+.5f} (если бы Δ не нёс никакого сигнала)")
    print(f"  доля факта, объяснённая чистым шумом: "
          f"{(LB['AVG3_MIX'] - LB['SEQ01_MIX']) / tot['exp_drmsle']:.0%}")

    print(f"\n{'ось':<48}{'p01':>8}{'p05':>8}{'p50':>8}{'p95':>8}{'p99':>8}"
          f"{'mean':>8}{'>0.1':>8}{'>0.5':>8}")
    for r in rows:
        print(f"{r['pair']:<52}{r['p01']:>8.3f}{r['p05']:>8.3f}{r['p50']:>8.3f}"
              f"{r['p95']:>8.3f}{r['p99']:>8.3f}{r['mean']:>+8.3f}"
              f"{r['sh_010']:>8.1%}{r['sh_050']:>8.1%}")
    pl.DataFrame(rows).write_csv(OUT / "test_axes.csv")

    # --- по сегментам: где именно глубина двигает прогноз ----------------------
    segs, n_old_days, n_old_buys = segments(uid)
    print(f"\nдобавляемые 76 дней (T−365, T−289]: у {float((n_old_days > 0).mean()):.1%} "
          f"пользователей там есть хотя бы один день лога, "
          f"у {float((n_old_buys > 0).mean()):.1%} — покупка; "
          f"в среднем {n_old_days.mean():.1f} дней")
    srows = []
    print(f"\n{'сегмент':<30}{'доля':>7}" + "".join(f"{n[:11]:>13}" for n, _, _ in pairs[:4]))
    for sname, m in segs.items():
        r = dict(segment=sname, share=float(m.mean()))
        for name, b, c in pairs[:4]:
            r[name.split(".")[0].replace("ТОЧНО ","")] = float(np.var((V[c] - V[b])[m]))
        srows.append(r)
        print(f"{sname:<30}{r['share']:>7.1%}"
              + "".join(f"{r[n.split('.')[0].replace("ТОЧНО ","")]:>13.5f}" for n, _, _ in pairs[:4]))
    pl.DataFrame(srows).write_csv(OUT / "test_segments.csv")
    print(f"\n(в клетках — Var(Δz) внутри сегмента для соответствующей оси)")

    if not a.no_check:
        check_seed_common()
    print(f"\nзаписано: {OUT}/test_axes.csv, test_segments.csv, seed_common_depth.csv")


if __name__ == "__main__":
    main()
