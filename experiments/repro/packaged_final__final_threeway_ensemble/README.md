# final_threeway_ensemble

## Catalogue metadata

- **Catalogue ID:** `packaged_final__final_threeway_ensemble`
- **Namespace:** `packaged_final`
- **Experiment ID:** `final_threeway_ensemble`
- **Original source:** `research/FINAL_THREEWAY_ENSEMBLE.json`
- **Source ref:** `origin/team-a final/research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** three-way final-candidate ensemble
- **Model:** ensemble, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** "team_b_validation": {
- **Known score:** "current_rmsle": 1.7959069747011052,
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** "submission_STRONGEST_CURRENT.csv": 1.6496571902356205,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the three named source submissions are present
- **Notes:** Reported leaderboard results and forecasts are kept distinct exactly as in the preserved source.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# final_threeway_ensemble

```json
{
  "output": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\submissions\\SUBMIT_FINAL_3WAY_V1.csv",
  "output_sha256": "9ee91292e95bfb76d8012dfa225633d4ed4e796182d79167064db23517029828",
  "rows": 250000,
  "columns": [
    "user_id",
    "predict"
  ],
  "dtypes": {
    "user_id": "int64",
    "predict": "float64"
  },
  "nan_count": 0,
  "finite": true,
  "nonnegative": true,
  "same_order_as_sample": true,
  "unique_user_id": 250000,
  "zeros": 0,
  "min_predict": 0.0016419861717479245,
  "max_predict": 4635.998017624503,
  "mean_predict": 40.43306803271054,
  "mean_log1p": 2.329754704595576,
  "std_log1p": 1.6201419134534747,
  "blend_space": "log1p",
  "weights": {
    "local_cap_unc_dist_seq_etx": 0.05,
    "joint_v2": 0.9,
    "team_b_final": 0.05
  },
  "reference_level": 2.3295556048710435,
  "team_b_level_shift": -0.12166701846543782,
  "team_b_internal_current_log_scale": 1.12,
  "sources": {
    "local_cap_unc_dist_seq_etx": {
      "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\submissions\\submission_STRONGEST_CURRENT.csv",
      "sha256": "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda",
      "weight": 0.05,
      "mean_log1p_raw": 2.3293213699010047,
      "mean_log1p_used": 2.3293213699010047,
      "std_log1p_used": 1.5720141493966198,
      "zeros_raw": 273
    },
    "joint_v2": {
      "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\submissions\\SUBMIT_JOINT_V2.csv",
      "sha256": "211879cb1c79bbbde93d451fca5b61c521b523f989ce42bab62cd3ab87233cba",
      "weight": 0.9,
      "mean_log1p_raw": 2.3297898398410823,
      "mean_log1p_used": 2.3297898398410823,
      "std_log1p_used": 1.6225290168435618,
      "zeros_raw": 360
    },
    "team_b_final": {
      "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\team-b-final\\submissions\\final_classic_ml.csv",
      "sha256": "4ed2916baca85c13d51dcfc4f99877b5d06c03abce90ea0c1aae8c0506d44aba",
      "weight": 0.05,
      "mean_log1p_raw": 2.4512064373460523,
      "mean_log1p_used": 2.3295556048710435,
      "std_log1p_used": 1.634115179510716,
      "zeros_raw": 3
    }
  },
  "prediction_correlations_after_level_alignment": {
    "local_cap_unc_dist_seq_etx": {
      "local_cap_unc_dist_seq_etx": 1.0,
      "joint_v2": 0.9980468812562504,
      "team_b_final": 0.9960477127961432
    },
    "joint_v2": {
      "local_cap_unc_dist_seq_etx": 0.9980468812562504,
      "joint_v2": 1.0,
      "team_b_final": 0.9961070163779306
    },
    "team_b_final": {
      "local_cap_unc_dist_seq_etx": 0.9960477127961432,
      "joint_v2": 0.9961070163779306,
      "team_b_final": 1.0
    }
  },
  "team_b_validation": {
    "folds": [
      {
        "train_cutoff": "2025-11-15",
        "val_cutoff": "2025-12-15",
        "rows": 250000,
        "target_mean_log": 2.4272974674339434,
        "current_rmsle": 1.7959069747011052,
        "team_rmsle": 1.7410721743833586,
        "final_rmsle": 1.7505298076188436,
        "final_shifted_rmsle": 1.7414156708431467,
        "current_bias": 0.39475839969420157,
        "team_bias": -0.05633146743394379,
        "final_bias": 0.19176795948653633,
        "final_shifted_bias": 0.07035458372175597,
        "current_team_prediction_corr": 0.9975735329107247,
        "current_team_error_corr": 0.9845913500657568
      },
      {
        "train_cutoff": "2025-12-15",
        "val_cutoff": "2026-01-14",
        "rows": 250000,
        "target_mean_log": 2.2413663239108015,
        "current_rmsle": 1.7585346147057634,
        "team_rmsle": 1.6768702874591008,
        "final_rmsle": 1.706190516149757,
        "final_shifted_rmsle": 1.6865167736034226,
        "current_bias": 0.5041018626164196,
        "team_bias": 0.12959967608919873,
        "final_bias": 0.3355758786791704,
        "final_shifted_bias": 0.21416311365608987,
        "current_team_prediction_corr": 0.9967649462983331,
        "current_team_error_corr": 0.9853991149183449
      }
    ],
    "mean": {
      "current_rmsle": 1.7772207947034344,
      "team_rmsle": 1.7089712309212297,
      "final_rmsle": 1.7283601618843003,
      "final_shifted_rmsle": 1.7139662222232848,
      "current_bias": 0.4494301311553106,
      "team_bias": 0.03663410432762747,
      "final_bias": 0.26367191908285337,
      "final_shifted_bias": 0.1422588486889229,
      "current_team_prediction_corr": 0.9971692396045289,
      "current_team_error_corr": 0.9849952324920508
    },
    "production_level_shift": 0.12141659750496991,
    "current_weight": 0.55,
    "team_weight": 0.45
  },
  "known_public_lb": {
    "submission_STRONGEST_CURRENT.csv": 1.6496571902356205,
    "SUBMIT_JOINT_V2.csv": 1.6459363044782171,
    "team_b_final": null
  },
  "offline_evidence": {
    "local_cap_unc_dist_seq_etx": {
      "validation": "canonical four-fold OOF with per-fold log-offset calibration",
      "fold_rmsle": [
        1.7668833567997195,
        1.7605095767798136,
        1.748629223964952,
        1.7412785664479717
      ],
      "fold_weights": [
        1.0,
        2.0,
        4.0,
        8.0
      ],
      "weighted_cv": 1.7475098625201952,
      "public_geometry_optimal_weight_from_joint_v2": 0.010101354671869523
    },
    "joint_v2": {
      "exact_common_oof_score": null,
      "predecessor_exp075_nested_delta_rmsle": -0.00125067,
      "predecessor_all_four_fold_signs_positive": true,
      "joint_v2_plane_only_delta_vs_exp075_oof": 0.0003779151228466837,
      "out_of_plane_local_value": null
    },
    "team_b_final": {
      "two_fold_mean_raw_rmsle": 1.7283601618843003,
      "two_fold_mean_level_corrected_rmsle": 1.7139662222232848,
      "two_fold_mean_team_only_rmsle": 1.7089712309212297,
      "two_fold_mean_current_team_error_corr": 0.9849952324920508,
      "scored_span_optimal_weight_range_rcond_1e_4_to_1e_6": [
        0.08558791561384882,
        0.10781536701752063
      ],
      "chosen_weight_after_local_risk_shrinkage": 0.05
    },
    "common_oof_final_threeway_rmsle": null,
    "common_oof_limitation": "JOINT_V2 and team-b-final do not have predictions on the canonical same-row OOF panel; no synthetic final CV was reported"
  },
  "test_geometry": {
    "raw_prediction_correlations": {
      "local_vs_joint_v2": 0.9980468812562309,
      "local_vs_team_b_final": 0.9960477127961432,
      "joint_v2_vs_team_b_final": 0.9961066418444048
    },
    "local_vs_joint_v2_post_scored_span_rms_fraction": 0.00034304601106224206,
    "local_vs_joint_v2_post_scored_span_energy_fraction": 1.176805657057159e-07,
    "team_b_final_vs_joint_v2_post_scored_span_rms_fraction": 0.29450682186259836,
    "team_b_final_vs_joint_v2_post_scored_span_energy_fraction": 0.08673426812360824
  },
  "weight_rationale": {
    "joint_v2": "dominant anchor: best exact LB and already absorbs nearly all scored-span information",
    "local_cap_unc_dist_seq_etx": "small robustness reserve: strongest canonical OOF evidence, but almost entirely duplicated by the scored span",
    "team_b_final": "half-shrunk scored-span optimum: real test novelty, but fixed blend loses on both OOT folds and the novel residual has no target-alignment evidence"
  },
  "public_lb_forecast": {
    "point": 1.6459,
    "reasonable_range": [
      1.64586,
      1.64605
    ],
    "status": "estimate, not fact",
    "assumption": "50% shrinkage of the stable scored-span optimum for team-b-final; no value assigned to its unvalidated post-span residual"
  }
}
```
