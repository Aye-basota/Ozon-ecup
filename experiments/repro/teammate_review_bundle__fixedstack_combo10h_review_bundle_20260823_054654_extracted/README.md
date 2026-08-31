# fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted

## Catalogue metadata

- **Catalogue ID:** `teammate_review_bundle__fixedstack_combo10h_review_bundle_20260823_054654_extracted`
- **Namespace:** `teammate_review_bundle`
- **Experiment ID:** `fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/RUN_MANIFEST.json`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** completed teammate review run
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_combo_10h\\submissions\\submission_combo10h_candidate_4_ridge_core_plus_recent_dist_s075__slotbeta875.csv",
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted

```json
{
  "version": "fixedstack_combo_10h_2026-08-23_001",
  "started": "2026-08-23T04:18:50",
  "fixed_parent": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\run_best_bas_fixedstack_14h_v2.py",
  "previous_runner": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\run_best_bas_research_23h.py",
  "work": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_research",
  "package": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\submission_STRONGEST_CURRENT",
  "loaded_existing_oof": [
    "recent_hurdle",
    "multiscale_direct",
    "recent_direct",
    "recent_dist"
  ],
  "args": {
    "max_hours": 9.5,
    "threads": 8,
    "child_threads": 5,
    "reuse_work_dir": null,
    "no_install": false,
    "preflight_only": false,
    "self_test": false,
    "child_final": null
  },
  "friend_rebuild_error": 4.96611577169119e-07,
  "finished": "2026-08-23T05:46:54",
  "runtime_hours": 1.470208673675855,
  "remaining_hours": 8.029791326324144,
  "selection": [
    {
      "rank": 1,
      "name": "ridge_drop_recent_hurdle_stable18_s075",
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_combo_10h\\submissions\\submission_combo10h_candidate_1_ridge_drop_recent_hurdle_stable18_s075.csv",
      "delta_table": -0.0015469442204357087,
      "latest_delta": -0.0016618744898704296,
      "wins_recent": 3,
      "family": "ridge_subset",
      "friend_corr": 0.9997312220683123,
      "friend_std_dz": 0.038379258764679215,
      "friend_pct05": 0.170936,
      "var_ratio": 1.2091884042525611
    },
    {
      "rank": 2,
      "name": "ridge_core_plus_recent_dist_s075",
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_combo_10h\\submissions\\submission_combo10h_candidate_2_ridge_core_plus_recent_dist_s075.csv",
      "delta_table": -0.0014613670390685248,
      "latest_delta": -0.0015291237256671586,
      "wins_recent": 3,
      "family": "ridge_subset",
      "friend_corr": 0.9997565700594275,
      "friend_std_dz": 0.036492968719565984,
      "friend_pct05": 0.154044,
      "var_ratio": 1.1201008650295392
    },
    {
      "rank": 3,
      "name": "ridge_drop_recent_hurdle_stable18_s075__slotbeta875",
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_combo_10h\\submissions\\submission_combo10h_candidate_3_ridge_drop_recent_hurdle_stable18_s075__slotbeta875.csv",
      "delta_table": -0.0015038034540948728,
      "latest_delta": -0.0016065329991197252,
      "wins_recent": 3,
      "family": "ridge_subset_slotstrength",
      "friend_corr": 0.9997938138526201,
      "friend_std_dz": 0.033588089741865716,
      "friend_pct05": 0.12396,
      "var_ratio": 1.2091884042525616
    },
    {
      "rank": 4,
      "name": "ridge_core_plus_recent_dist_s075__slotbeta875",
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_combo_10h\\submissions\\submission_combo10h_candidate_4_ridge_core_plus_recent_dist_s075__slotbeta875.csv",
      "delta_table": -0.0014245196784167468,
      "latest_delta": -0.0014841313416025237,
      "wins_recent": 3,
      "family": "ridge_subset_slotstrength",
      "friend_corr": 0.9998132666772231,
      "friend_std_dz": 0.031938683912605896,
      "friend_pct05": 0.109872,
      "var_ratio": 1.1201008650295394
    }
  ],
  "finalizable_experts": [
    "cap",
    "unc",
    "dist",
    "hurdle",
    "multiscale_direct",
    "recent_direct",
    "recent_dist",
    "recent_hurdle_fast12",
    "recent_hurdle_stable18"
  ]
}
```
