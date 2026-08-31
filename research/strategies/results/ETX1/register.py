"""Регистрация гейта ETX-01 в `experiments/log.csv` по правилам `AGENTS.md`.

Числа берутся ТОЛЬКО из уже записанных артефактов (`oof_*`, `curve_*`,
`fold_V1016.csv`) — руками сюда ничего не вписывается, иначе журнал и файлы
разойдутся. `wcv` не заполняется: схема неполная (один фолд из четырёх), а
`validation.wcv` определён только на полной S1 — ровно так же зарегистрирован
`SEQ-AVAIL-AUG` (`exp_029`).

Запуск: PYTHONPATH=. python research/strategies/results/ETX1/register.py
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.config import ARTIFACTS
from src.tracking import log_experiment
from src.validation import calibrate, rmsle_z

EXP = "ETX-01-S42"
TAG = "V1016"
FOLD = "2025-10-16"
BASE = "SEQ-D3A-S42"


def main() -> None:
    t = pl.read_csv(f"research/strategies/results/ETX1/fold_{TAG}.csv")
    row = {r["exp"]: r for r in t.to_dicts()}
    c = json.loads((ARTIFACTS / f"curve_{EXP}-{TAG}.json").read_text(encoding="utf-8"))
    d = np.load(ARTIFACTS / f"oof_{EXP}-{TAG}.npz")
    y, z = np.asarray(d["y"], float), np.asarray(d["z"], float)
    off, cal = calibrate(y, z)
    me = row[EXP]
    assert abs(cal - me["rmsle_cal"]) < 1e-9, "diag и OOF разошлись"

    cfg = c["cfg"]
    params = dict(
        arch="sparse event transformer (causal SDPA + time-ALiBi)",
        d_model=cfg["d_model"], blocks=cfg["blocks"], heads=cfg["heads"],
        head_dim=cfg["head_dim"], ffn=cfg["ffn"], dropout=cfg["dropout"],
        n_tok=cfg["n_tok"], tok_features=22, static_features=6,
        batch=cfg["batch"], lr=cfg["lr"], wd=cfg["wd"], epochs=cfg["epochs"],
        warmup=cfg["warmup"], opt="AdamW cosine, lr x10 для log_m", precision="bf16",
        params_n=c["n_params"], seed=cfg["seed"], fold=FOLD,
        supervision="один forward = один пример (история<=T -> y30(T))",
        peak_vram_gb=round(c["peak_vram_gb"], 2),
        rows_s=round(c["throughput_rows_s"]),
        tau_final_min=min(c["tau_final"]), tau_final_max=max(c["tau_final"]),
        epoch_cal=[round(h["rmsle_cal"], 5) for h in c["hist"]],
    )
    concl = (
        f"exp_036 гейт STRATEGY_13 вариант B на фолде {FOLD}, сид 42, "
        f"{c['n_params']:,} параметров. Калибр. RMSLE {cal:.5f} против "
        f"{row[BASE]['rmsle_cal']:.5f} у {BASE} (Δ {me['d_rmsle_cal']:+.5f}) и "
        f"{row['SEQ-D3A-BASE-S42']['rmsle_cal']:.5f} у SEQ-D3A-BASE-S42 без depth-curriculum. "
        f"AUC(y>0) {me['auc']:.5f} (Δ {me['d_auc']:+.5f}). "
        f"Var(z-z_SEQ)={me['var_vs_base']:.5f}, Var(z-z_DIST_MIX)={me['var_vs_distmix']:.5f}, "
        f"corr остатков с SEQ {me['corr_resid']:.5f}, с DIST-MIX {me['corr_resid_distmix']:.5f}. "
        f"tau_h разошлись {min(c['tau_final']):.1f}..{max(c['tau_final']):.1f} д. "
        f"{c['runtime_s'] / 60:.0f} мин, пик VRAM {c['peak_vram_gb']:.2f} ГБ, "
        f"{c['throughput_rows_s']:,.0f} примеров/с на RTX 4060 Ti."
    )
    log_experiment(
        exp_id=EXP, description=(
            "exp_036 ETX-01: sparse event transformer (токен = реальный день лога, "
            "<=192 события + query, causal SDPA, ALiBi в календарном времени); "
            "быстрый gate на фолде 2025-10-16, seed 42"),
        scenario="S1", n_features=22, model="event-transformer",
        params=params, cutoffs="24 @ step 7", panel_blocks=3,
        fold_scores=[round(rmsle_z(y, z), 5)], cv_mean=round(rmsle_z(y, z), 5), cv_std=0.0,
        bias_mean=round(float(np.log1p(y).mean() - z.mean()), 5), best_offset=round(off, 5),
        cv_mean_calib=round(cal, 5), delta_vs_b0=round(me["d_rmsle_cal"], 5),
        runtime_s=round(c["runtime_s"], 1), verdict=VERDICT, conclusion=concl,
        fold_cal=[round(cal, 5)], mean_z=round(float(z.mean()), 5))
    print(f"зарегистрировано: {EXP}  {cal:.5f}  Δ {me['d_rmsle_cal']:+.5f}  -> {VERDICT}")


VERDICT = "CONTINUE"

if __name__ == "__main__":
    main()
