# Reproduce `SUBMIT_JOINT86_TEAMB14.csv`

## Catalogue metadata

- **Catalogue ID:** `packaged_final__submit_joint86_teamb14`
- **Namespace:** `packaged_final`
- **Experiment ID:** `submit_joint86_teamb14`
- **Original source:** `reproducibility/SUBMIT_JOINT86_TEAMB14/README.md`
- **Source ref:** `origin/team-a final/research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** packaged final submission; exact outer blend, frozen upstream anchor
- **Model:** LightGBM, CatBoost, XGBoost, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** training code, validation reports and the reference submission.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from frozen inputs; raw-to-JOINT_V2 remains explicitly PROVENANCE_INCOMPLETE
- **Notes:** Reported leaderboard results and forecasts are kept distinct exactly as in the preserved source.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Reproduce `SUBMIT_JOINT86_TEAMB14.csv`

This package contains the exact final builder, frozen components, Team-B
training code, validation reports and the reference submission.

## Exact formula

The blend is not a raw-space average.

1. `z_a = log1p(SUBMIT_JOINT_V2.predict)`.
2. `z_b_raw = log1p(final_classic_ml.predict)`.
3. Find scalar `s` by 100 bisection iterations such that
   `mean(max(z_b_raw+s,0)) = mean(z_a)`. The realized shift is
   `-0.1214326530964569`.
4. `z = 0.86*z_a + 0.14*max(z_b_raw+s,0)`.
5. `predict = max(expm1(z),0)`.

The Team-A 86% vector is exactly `inputs/SUBMIT_JOINT_V2.csv`, SHA
`211879cb1c79bbbde93d451fca5b61c521b523f989ce42bab62cd3ab87233cba`.
The Team-B 14% vector is `team-b-final/submissions/final_classic_ml.csv`, SHA
`4ed2916baca85c13d51dcfc4f99877b5d06c03abce90ea0c1aae8c0506d44aba`.
Inside Team-B, `CURRENT_LOG_SCALE=1.12` remains unchanged.

## Exact rebuild

From the repository root:

```powershell
python scripts/reproduce_final.py --solution SUBMIT_JOINT86_TEAMB14 --from-precomputed
```

Or directly:

```powershell
python reproducibility/SUBMIT_JOINT86_TEAMB14/build_submit.py
```

Expected output SHA256:

```text
85d9cd645e14a7895da9ad8cc89065714606266be588c762d37487d2b4edac02
```

## Raw Team-B retraining

Use Python 3.11 and the pinned `requirements.txt`:

```powershell
python scripts/reproduce_final.py --solution SUBMIT_JOINT86_TEAMB14 --from-raw `
  --raw-data data/raw/train.parquet `
  --team-b-python .venv-team-b/Scripts/python.exe
```

This executes raw → Team-B features → LightGBM/XGBoost/CatBoost training →
Team-B prediction → 86/14 blend. The JOINT_V2 anchor remains frozen because its
primary upstream generation script was not found.

## Public leaderboard provenance

The teammate reproduction request records:

- `SUBMIT_JOINT_V2`: 1.6459363044782171;
- `SUBMIT_JOINT86_TEAMB14`: 1.6458200196207617.

The copied source is
`research/provenance/downloads/TEAMMATE_REPRO_REQUEST.md`. The 1.64582 value is a
reported score, not a forecast. The Team-B component itself was not separately
submitted in the found evidence.

## Provenance boundary

Geometry, ORTH, A1/A2, EXP075 and EXP089 reports establish the research lineage
leading toward JOINT_V2 and audit its out-of-plane structure. They do not contain
a script that emits the exact JOINT_V2 SHA. Therefore:

```text
SUBMIT_JOINT_V2 upstream: PROVENANCE_INCOMPLETE
JOINT86/TEAMB14 outer blend: COMPLETE, BYTE-EXACT
Team-B training: COMPLETE, previously BYTE-EXACT in pinned environment
```

No reverse-engineered formula is presented as primary provenance.

## Package map

- `COMPONENT_MANIFEST.json` — sources, weights, clipping, SHA and score evidence;
- `inputs/` — frozen 86% Team-A anchor;
- `team-b-final/src/` — training/features/inference code;
- `team-b-final/data/sample_submit.csv` — exact user order;
- `team-b-final/submissions/` — frozen 14% component;
- `reference/` — exact final CSV;
- `reports/` — original blend report;
- `MANIFEST.sha256` — checksums of committed package files.
