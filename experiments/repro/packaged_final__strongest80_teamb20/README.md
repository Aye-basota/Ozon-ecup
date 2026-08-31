# strongest80_teamb20

## Catalogue metadata

- **Catalogue ID:** `packaged_final__strongest80_teamb20`
- **Namespace:** `packaged_final`
- **Experiment ID:** `strongest80_teamb20`
- **Original source:** `research/STRONGEST80_TEAMB20.json`
- **Source ref:** `origin/team-a final/research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late pair-blend candidate
- **Model:** blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** "submission_STRONGEST_CURRENT": 0.8,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the two named source submissions are present
- **Notes:** Reported leaderboard results and forecasts are kept distinct exactly as in the preserved source.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# strongest80_teamb20

```json
{
  "output": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\submissions\\SUBMIT_STRONGEST80_TEAMB20.csv",
  "sha256": "bd5f364a8552ef6faf49004386e10ddd51806c739c43879eda3f83a3c6601491",
  "blend_space": "log1p",
  "weights": {
    "submission_STRONGEST_CURRENT": 0.8,
    "team_b_final": 0.2
  },
  "team_b_internal_current_log_scale": 1.12,
  "team_b_level_shift": -0.12190138468055683,
  "source_correlation": 0.9960480463024701,
  "centered_source_difference_rms": 0.15544676868244206,
  "rows": 250000,
  "unique_user_id": 250000,
  "same_order_as_sample": true,
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
  "zeros": 76,
  "min_predict": 0.0,
  "max_predict": 2874.8793885096798,
  "mean_predict": 37.76442183465612,
  "mean_log1p": 2.329321369901005,
  "std_log1p": 1.5834087267226964,
  "known_strongest_public_lb": 1.6496571902356205,
  "public_lb_forecast": {
    "point": 1.64875,
    "reasonable_range": [
      1.64855,
      1.64915
    ],
    "status": "estimate, not fact"
  }
}
```
