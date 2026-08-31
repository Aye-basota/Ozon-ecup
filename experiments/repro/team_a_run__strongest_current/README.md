# Logged run — STRONGEST-CURRENT

## Catalogue metadata

- **Catalogue ID:** `team_a_run__strongest_current`
- **Namespace:** `team_a_run`
- **Experiment ID:** `STRONGEST-CURRENT`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** sequence model, blend
- **Features:** history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** cutoffs:** 4 фолда S1
- **Known score:** verdict:** ОТПРАВЛЕН, LB 1.6496571 — новый чемпион
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — STRONGEST-CURRENT

This run was recovered from `experiments/log.csv`.

- **exp_id:** STRONGEST-CURRENT
- **timestamp:** 2026-08-20T12:35:00
- **commit:** a28a71f
- **description:** exp_037 EXP-037: боевая смесь CAP/UNC/DIST/слот = 0.10/0.20/0.25/0.45, слот = 0.5*ETX-AVG3 + 0.5*SEQ-AVG3; тест ETX при DCW (depth-clip 289 + статик query в обученном диапазоне)
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** blend
- **params:** {"seq_member": "0.5*ETX-AVG3 + 0.5*SEQ-AVG3", "test_policy_etx": "DCW: depth-clip 289 + depth_cap 289 + cdow=четверг", "test_policy_seq": "clip289", "test_z": ["S1-CAP", "S1-UNC", "S1-DIST", "SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44", "ETX-01-S42-DCW", "ETX-01-S43-DCW", "ETX-01-S44-DCW"], "weights": {"S1-E03a": 0.1, "S1-E02": 0.2, "S1-DIST": 0.25, "ETX-AVG3": 0.225, "SEQ-AVG3": 0.225}, "lofo": -0.00092, "lofo_folds": 4, "alpha_curve": {"0.0": -0.00055, "0.25": -0.00084, "0.5": -0.00092, "0.75": -0.00083, "1.0": -0.00053}, "regime_pair_ratio": 0.78, "regime_mix_ratio": 0.94, "sha256": "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"}
- **cutoffs:** 4 фолда S1
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77078, 1.76227, 1.74876, 1.74164]
- **cv_mean:** 1.75586
- **cv_std:** 0.01136
- **bias_mean:** -0.06256
- **best_offset:** -0.06256
- **cv_mean_calib:** 1.75433
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ОТПРАВЛЕН, LB 1.6496571 — новый чемпион
- **conclusion:** ОТПРАВЛЕН, LB public 1.6496571 (было 1.6501764, дельта -0.0005193 = 2.1 парной SE). Перенос wCV->LB 0.564 при ожидавшихся 0.57 — третье независимое подтверждение коэффициента; прогноз 1.64966 против факта 1.6496571. exp_037 ЭТАПЫ 2-5. Честный LOFO -0.00092 на 4/4 включая 10-16 (-0.00073/-0.00060/-0.00094/-0.00102) — ЛУЧШИЙ ЧЛЕН СЛОТА SEQ ЗА ПРОЕКТ (прежний рекорд -0.00061). wCV смеси 1.74751 против 1.74777 у SEQAVG3-CLIP-MIX и 1.74834 у отправленного SEQ-01-MIX. Максимум по alpha ровно в 0.5, замена (alpha=1) проваливает гейт 2/4. СЕГМЕНТНЫЙ ГЕЙТ ОТКЛОНЁН: seg3 -0.00090 (хуже глобального), seg4 -0.00094 при sd разности 0.00027 и 3 лишних степенях свободы, alpha неустойчив по held-out. ГЕЙТ РЕЖИМА ПРОЙДЕН: пара ETX-AVG3 vs SEQ-AVG3 на тесте 0.01797 против 0.02318 на OOF = 0.78x (полоса всех пар проекта 0.63..1.11x), кандидат против чемпиона 0.94x, по полосам активности 0.61..1.07 без роста на high-activity. Файл: 250000 строк, mean log1p 2.329321, delta -0.13536, нулей 0.109%, max 2641.4, реконструкция 4.97e-07. Ожидание LB при переносе 0.57: ~1.64966 (1.6493..1.6500).
- **wcv:** 1.74751
- **fold_cal:** [1.76688, 1.76051, 1.74863, 1.74128]
- **mean_z:** 2.68819
- **lb_public:** 1.6496571
