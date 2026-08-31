# Logged run — SEQ-AVG3-CLIP-MIX

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_avg3_clip_mix`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-AVG3-CLIP-MIX`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** sequence model, blend
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_035 MIX9: смесь CAP/E02/DIST/SEQ-AVG3 = 0.10/0.20/0.25/0.45, тест при --depth-clip 289. Честный LOFO -0.00055 к отправленному SEQ-01-MIX (1.74834), 4/4 фолда включая 10-16. Альтернатива SEQ-D3A-AVG3 дала -0.00061 при sd разности 0.00027 - неразрешимо, взят SEQ-AVG3. Сабмит собран и проверен, НЕ отправлен.
- **Known score:** conclusion:** полоса «в разработку» (-0.0020..-0.0005); при переносе 0.57 ожидание LB ~1.64986 = ~1.3 парной SE
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** description:** exp_035 MIX9: смесь CAP/E02/DIST/SEQ-AVG3 = 0.10/0.20/0.25/0.45, тест при --depth-clip 289. Честный LOFO -0.00055 к отправленному SEQ-01-MIX (1.74834), 4/4 фолда включая 10-16. Альтернатива SEQ-D3A-AVG3 дала -0.00061 при sd разности 0.00027 - неразрешимо, взят SEQ-AVG3. Сабмит собран и проверен, НЕ отправлен.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-AVG3-CLIP-MIX

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-AVG3-CLIP-MIX
- **timestamp:** 2026-08-19T23:25:50
- **commit:** a28a71f
- **description:** exp_035 MIX9: смесь CAP/E02/DIST/SEQ-AVG3 = 0.10/0.20/0.25/0.45, тест при --depth-clip 289. Честный LOFO -0.00055 к отправленному SEQ-01-MIX (1.74834), 4/4 фолда включая 10-16. Альтернатива SEQ-D3A-AVG3 дала -0.00061 при sd разности 0.00027 - неразрешимо, взят SEQ-AVG3. Сабмит собран и проверен, НЕ отправлен.
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** blend
- **params:** {"seq_member": "SEQ-AVG3 (\u0441\u0438\u0434\u044b 42/43/44, \u043b\u043e\u0433-\u0441\u0440\u0435\u0434\u043d\u0435\u0435)", "test_depth": "clip289", "test_z": ["S1-CAP", "S1-UNC", "S1-DIST", "SEQ-01", "SEQ-C289-S43", "SEQ-C289-S44"], "weights": {"S1-DIST": 0.25, "S1-E02": 0.2, "S1-E03a": 0.1, "SEQ-AVG3": 0.45}}
- **cutoffs:** 4 фолда S1
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76915, 1.76142, 1.74881, 1.74189]
- **cv_mean:** 1.75532
- **cv_std:** 0.01062
- **bias_mean:** -0.04139
- **best_offset:** -0.04139
- **cv_mean_calib:** 1.75468
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ACCEPT кандидат, НЕ ОТПРАВЛЕН
- **conclusion:** полоса «в разработку» (-0.0020..-0.0005); при переносе 0.57 ожидание LB ~1.64986 = ~1.3 парной SE
- **wcv:** 1.74777
- **fold_cal:** [1.76675, 1.76047, 1.74879, 1.74171]
- **mean_z:** 2.66701
- **lb_public:** Unknown / not recoverable from repository history
