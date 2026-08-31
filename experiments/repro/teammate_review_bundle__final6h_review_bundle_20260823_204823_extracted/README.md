# final6h_REVIEW_BUNDLE_20260823_204823_extracted

## Catalogue metadata

- **Catalogue ID:** `teammate_review_bundle__final6h_review_bundle_20260823_204823_extracted`
- **Namespace:** `teammate_review_bundle`
- **Experiment ID:** `final6h_REVIEW_BUNDLE_20260823_204823_extracted`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/RUN_MANIFEST.json`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** completed teammate review run
- **Model:** Ridge, two-part / hurdle, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_final6h\\submissions\\submission_final6h_B_metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv",
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# final6h_REVIEW_BUNDLE_20260823_204823_extracted

```json
{
  "version": "final6h_fixedfriend_2026-08-23_001",
  "started": "2026-08-23T16:18:39",
  "combo_parent": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\continue_fixedstack_combo_10h.py",
  "fixed_parent": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\run_best_bas_fixedstack_14h_v2.py",
  "previous_runner": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\run_best_bas_research_23h.py",
  "work": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_research",
  "package": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\submission_STRONGEST_CURRENT",
  "args": {
    "max_hours": 6.0,
    "threads": 8,
    "child_threads": 6,
    "reuse_work_dir": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_research",
    "no_install": false,
    "preflight_only": false,
    "self_test": false,
    "child_occ": null,
    "child_hurdle": null,
    "child_fold": "TEST"
  },
  "friend_rebuild_error": 4.96611577169119e-07,
  "known_friend_public": 1.6496571,
  "known_ridge_submission_public": 1.6492897556391737,
  "finished": "2026-08-23T20:48:23",
  "runtime_hours": 4.496829305820995,
  "remaining_hours": 1.5031706941790048,
  "completed_occurrence_families": [
    "occ_r10_fast",
    "occ_r16_bal",
    "occ_r22_stable",
    "occ_r14_multiscale",
    "occ_r18_wide",
    "occ_r24_multiscale",
    "occ_r12_wide",
    "occ_r20_shallow"
  ],
  "branch_A": {
    "branch": "A",
    "name": "blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
    "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_final6h\\submissions\\submission_final6h_A_blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv",
    "delta_table": -0.0016507644461350188,
    "latest_delta": -0.0018111533415456904,
    "family": "adaptive_blend",
    "corr": 0.999722108135685,
    "std": 0.03837392554198951,
    "mae": 0.028848508201543544,
    "pct02": 0.54634,
    "pct05": 0.168408,
    "pct10": 0.0167
  },
  "branch_B": {
    "branch": "B",
    "name": "metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
    "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_final6h\\submissions\\submission_final6h_B_metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv",
    "delta_table": -0.0017665443820891783,
    "latest_delta": -0.0020268591063281605,
    "family": "occurrence_meta_risk",
    "corr": 0.9996277554755861,
    "std": 0.04536330866017492,
    "mae": 0.03437858891234656,
    "pct02": 0.613636,
    "pct05": 0.2373,
    "pct10": 0.034112
  }
}
```
