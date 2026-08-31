# extra90_REVIEW_BUNDLE_20260823_222555_extracted

## Catalogue metadata

- **Catalogue ID:** `teammate_review_bundle__extra90_review_bundle_20260823_222555_extracted`
- **Namespace:** `teammate_review_bundle`
- **Experiment ID:** `extra90_REVIEW_BUNDLE_20260823_222555_extracted`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/RUN_MANIFEST.json`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** completed teammate review run
- **Model:** Ridge, blend
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_extra90m\\submissions\\submission_extra90_4_hier_trust_bias_ridge_recentpow1p7_s075.csv",
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# extra90_REVIEW_BUNDLE_20260823_222555_extracted

```json
{
  "version": "extra90m_cached_meta_2026-08-23_001",
  "finished": "2026-08-23T22:25:55",
  "runtime_minutes": 31.111259865760804,
  "work": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_research",
  "occ_names": [
    "occ_r10_fast",
    "occ_r16_bal",
    "occ_r22_stable",
    "occ_r14_multiscale",
    "occ_r18_wide",
    "occ_r24_multiscale",
    "occ_r12_wide",
    "occ_r20_shallow"
  ],
  "selected": [
    {
      "rank": 1,
      "name": "blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
      "family": "adaptive_blend",
      "delta": -0.0016507644461350188,
      "wins_recent": 3,
      "latest_delta": -0.0018111533415456904,
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_extra90m\\submissions\\submission_extra90_1_blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv",
      "corr": 0.999722108135685,
      "std": 0.03837392554198951,
      "mae": 0.028848508201543544,
      "pct02": 0.54634,
      "pct05": 0.168408,
      "pct10": 0.0167
    },
    {
      "rank": 2,
      "name": "xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
      "family": "xmeta_risk",
      "delta": -0.0018206636174889232,
      "wins_recent": 3,
      "latest_delta": -0.0020723716645272283,
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_extra90m\\submissions\\submission_extra90_2_xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv",
      "corr": 0.9996313803551532,
      "std": 0.04514010386400232,
      "mae": 0.03415787760187457,
      "pct02": 0.611248,
      "pct05": 0.234076,
      "pct10": 0.0332
    },
    {
      "rank": 3,
      "name": "xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
      "family": "raw_occ_extra",
      "delta": -0.0016246012273087196,
      "wins_recent": 3,
      "latest_delta": -0.0019148299253186618,
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_extra90m\\submissions\\submission_extra90_3_xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv",
      "corr": 0.9996601138284242,
      "std": 0.042902163680878555,
      "mae": 0.032615393235133135,
      "pct02": 0.596476,
      "pct05": 0.21638,
      "pct10": 0.026476
    },
    {
      "rank": 4,
      "name": "hier_trust_bias_ridge_recentpow1p7_s075",
      "family": "hierarchical",
      "delta": -0.0013961545468362739,
      "wins_recent": 3,
      "latest_delta": -0.0018024423141691504,
      "file": "C:\\Users\\Dimentiy\\repoVScode\\Ozon-ecup\\src\\DL\\best_bas\\_best_bas_extra90m\\submissions\\submission_extra90_4_hier_trust_bias_ridge_recentpow1p7_s075.csv",
      "corr": 0.9996206214622866,
      "std": 0.045932767039829035,
      "mae": 0.034541864347786985,
      "pct02": 0.613808,
      "pct05": 0.236324,
      "pct10": 0.0374
    }
  ],
  "friend_rebuild_error": 4.96611577169119e-07
}
```
