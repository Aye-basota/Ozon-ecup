"""Важность признаков лучшего решения (S1-BEST -> submissions/submission_strategy_1.csv).

Локальный анализ, ничего не коммитится: и скрипт, и результат лежат в artifacts/
(каталог в .gitignore). Модель не переобучается «по-новому»: воспроизводится ровно
та конфигурация, из которой собран сабмит (experiments/exp_006):

    0.45 * S1-NORM  (L=None, norm_long, 227 призн.)
    0.45 * S1-UNC   (L=None,            236 призн.)
    0.10 * S1-CAP   (L=180,             195 призн.)

Три меры важности:
  gain    — суммарное уменьшение лосса по сплитам признака (доля от суммы по модели);
  split   — число сплитов;
  shap    — точный TreeSHAP (LightGBM pred_contrib) на выборке строк ТЕСТОВОГО
            cutoff'а 2026-02-13, в единицах z = log1p(pred).

Смесь усредняется в лог-пространстве, поэтому SHAP смеси считается точно:
    z = Σ w_i·z_i = Σ w_i·base_i + Σ_f (Σ_i w_i·shap_{i,f})
то есть построчная взвешенная сумма вкладов. Признак, которого в наборе модели нет,
даёт по ней вклад 0.

Запуск (из корня репозитория):  python artifacts/importance_best.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTIFACTS, CUTOFF_TEST, SEED  # noqa: E402
from src.data import load  # noqa: E402
from src.features import feature_names, make_xy, to_np  # noqa: E402
from src.predict import train_full  # noqa: E402
from src.train import GROUPS, Setup, select_features  # noqa: E402

N_SHAP = 20_000            # строк тестовой панели под TreeSHAP
TOP = 40

VARIANTS = [               # имя, вес в сабмите, аргументы Setup
    ("S1-NORM", 0.45, dict(L=0, norm_long=True)),
    ("S1-UNC", 0.45, dict(L=0, norm_long=False)),
    ("S1-CAP", 0.10, dict(L=180, norm_long=False)),
]

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def group_of(feat: str) -> str:
    hits = [g for g, f in GROUPS.items() if f(feat)]
    return "+".join(hits) if hits else "-"


def main():
    load()
    rng = np.random.default_rng(SEED)

    order = None           # порядок user_id тестовой панели, общий для всех вариантов
    idx = None             # индексы строк под SHAP
    gains, splits, shaps, bases, feat_sets = {}, {}, {}, {}, {}

    for name, _w, kw in VARIANTS:
        s = Setup(min_history=90, step=7, rounds=600, train_blocks=1, **kw)
        s.params = dict(s.params, seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

        Xt, _ = make_xy(CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
        uid = Xt["user_id"].to_numpy()
        if order is None:
            order = uid
            idx = np.sort(rng.choice(len(uid), size=min(N_SHAP, len(uid)), replace=False))
        assert np.array_equal(uid, order), f"{name}: другой порядок user_id на тесте"

        feats = select_features(feature_names(Xt), s.drop_groups, None)
        At = to_np(Xt, feats)[idx]
        del Xt
        gc.collect()
        log(f"{name}: {len(feats)} признаков, SHAP-выборка {At.shape[0]:,} строк")

        m = train_full(s, feats, ["direct"])["direct"][1]
        feat_sets[name] = feats
        gains[name] = m.feature_importance("gain").astype(np.float64)
        splits[name] = m.feature_importance("split").astype(np.float64)

        t = time.time()
        contrib = m.predict(At, pred_contrib=True)          # (n, nfeat + 1), последний — base
        shaps[name] = contrib[:, :-1]
        bases[name] = float(contrib[0, -1])
        log(f"{name}: TreeSHAP за {time.time() - t:.0f}s, base={bases[name]:.4f}, "
            f"mean z={contrib.sum(1).mean():.4f}")
        del m, At, contrib
        gc.collect()

    # --- сведение по объединению признаков -------------------------------------
    union = sorted({f for fs in feat_sets.values() for f in fs})
    pos = {f: i for i, f in enumerate(union)}
    n = len(union)

    gain_share = {}                                        # доля gain внутри своей модели
    split_cnt = {}
    shap_blend = np.zeros((len(idx), n), np.float64)
    shap_own = {}
    for name, w, _kw in VARIANTS:
        g = np.zeros(n); sp = np.zeros(n); sh = np.zeros((len(idx), n))
        cols = [pos[f] for f in feat_sets[name]]
        g[cols] = gains[name] / gains[name].sum()
        sp[cols] = splits[name]
        sh[:, cols] = shaps[name]
        gain_share[name] = g
        split_cnt[name] = sp
        shap_own[name] = np.abs(sh).mean(0)
        shap_blend += w * sh

    blend_gain = sum(w * gain_share[name] for name, w, _ in VARIANTS)
    blend_split = sum(w * split_cnt[name] for name, w, _ in VARIANTS)
    blend_shap = np.abs(shap_blend).mean(0)
    blend_shap_signed = shap_blend.mean(0)

    df = pl.DataFrame({
        "feature": union,
        "group": [group_of(f) for f in union],
        "shap_abs_mean": blend_shap,
        "shap_share": blend_shap / blend_shap.sum(),
        "shap_signed_mean": blend_shap_signed,
        "gain_share": blend_gain / blend_gain.sum(),
        "split_weighted": blend_split,
        **{f"shap_{name}": shap_own[name] for name, _, _ in VARIANTS},
        **{f"gain_{name}": gain_share[name] for name, _, _ in VARIANTS},
        "in_models": ["+".join(nm for nm, _, _ in VARIANTS if f in feat_sets[nm]) for f in union],
    }).sort("shap_abs_mean", descending=True)

    out_csv = ARTIFACTS / "importance_S1-BEST.csv"
    df.write_csv(out_csv, float_precision=6)
    log(f"CSV: {out_csv}")

    # --- отчёт ------------------------------------------------------------------
    grp = (df.group_by("group")
             .agg(pl.col("shap_share").sum().alias("shap_share"),
                  pl.col("gain_share").sum().alias("gain_share"),
                  pl.len().alias("n_feats"))
             .sort("shap_share", descending=True))

    L = []
    L.append("# Важность признаков — S1-BEST (`submissions/submission_strategy_1.csv`)")
    L.append("")
    L.append(f"Посчитано {time.strftime('%Y-%m-%d %H:%M')}, локальный файл, в git не попадает.")
    L.append("")
    L.append("Модели переобучены ровно как в `experiments/exp_006`: 29 cutoff'ов "
             "2025-04-03..2025-10-16, 1-блочная панель на обучении, LightGBM 600 раундов, "
             f"seed {SEED}. Смесь в лог-пространстве:")
    L.append("")
    L.append("| вариант | вес | L | norm_long | признаков | base (SHAP) |")
    L.append("|---|---|---|---|---|---|")
    for name, w, kw in VARIANTS:
        L.append(f"| {name} | {w:.2f} | {kw['L'] or 'None'} | {kw['norm_long']} | "
                 f"{len(feat_sets[name])} | {bases[name]:.4f} |")
    L.append("")
    L.append(f"SHAP — точный TreeSHAP на {len(idx):,} случайных пользователях тестовой панели "
             f"(cutoff {CUTOFF_TEST}), единицы — `z = log1p(pred)` ДО калибровочного сдвига "
             "δ = −0.1486. `shap_share` — доля признака в сумме средних |вкладов|; "
             "`gain_share` — взвешенная доля gain.")
    L.append("")
    L.append(f"## Топ-{TOP} по |SHAP| смеси")
    L.append("")
    L.append("| # | признак | группа | mean\\|SHAP\\| | доля | знак (mean SHAP) | gain доля | модели |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(df.head(TOP).iter_rows(named=True), 1):
        L.append(f"| {i} | `{r['feature']}` | {r['group']} | {r['shap_abs_mean']:.4f} | "
                 f"{r['shap_share']:.2%} | {r['shap_signed_mean']:+.4f} | "
                 f"{r['gain_share']:.2%} | {r['in_models']} |")
    L.append("")
    L.append(f"## Топ-{TOP} по gain")
    L.append("")
    L.append("| # | признак | группа | gain доля | mean\\|SHAP\\| | сплитов (взв.) |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(df.sort("gain_share", descending=True).head(TOP).iter_rows(named=True), 1):
        L.append(f"| {i} | `{r['feature']}` | {r['group']} | {r['gain_share']:.2%} | "
                 f"{r['shap_abs_mean']:.4f} | {r['split_weighted']:.0f} |")
    L.append("")
    L.append("## По группам признаков (`src/train.py: GROUPS`)")
    L.append("")
    L.append("| группа | признаков | доля SHAP | доля gain |")
    L.append("|---|---|---|---|")
    for r in grp.iter_rows(named=True):
        L.append(f"| {r['group']} | {r['n_feats']} | {r['shap_share']:.2%} | {r['gain_share']:.2%} |")
    L.append("")
    L.append("## Топ-15 по каждой модели отдельно (|SHAP|)")
    L.append("")
    for name, _w, _kw in VARIANTS:
        L.append(f"**{name}**")
        L.append("")
        L.append("| # | признак | mean\\|SHAP\\| | gain доля |")
        L.append("|---|---|---|---|")
        d = df.select("feature", f"shap_{name}", f"gain_{name}").sort(f"shap_{name}", descending=True)
        for i, r in enumerate(d.head(15).iter_rows(named=True), 1):
            L.append(f"| {i} | `{r['feature']}` | {r[f'shap_{name}']:.4f} | {r[f'gain_{name}']:.2%} |")
        L.append("")
    L.append(f"Полная таблица по всем {n} признакам: `artifacts/importance_S1-BEST.csv`.")
    L.append("")

    out_md = ARTIFACTS / "importance_S1-BEST.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    log(f"отчёт: {out_md}")


if __name__ == "__main__":
    main()
