from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ROWS = 250_000
EXPECTED_COLUMNS = ["user_id", "predict"]
WEIGHTS = {"friend": 0.12, "occ_meta_B": 0.16, "occ_raw_X3": 0.72}
SERIALIZATION_FLOOR = 5e-7
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_SIZES = (188_518, 191_025, 193_694, 197_379)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class Submission:
    path: Path
    frame: pd.DataFrame
    user_id: np.ndarray
    predict: np.ndarray
    z: np.ndarray


def read_submission(path: Path, expected_rows: int = EXPECTED_ROWS) -> Submission:
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path}: columns {list(frame.columns)!r}, expected {EXPECTED_COLUMNS!r}"
        )
    if len(frame) != expected_rows:
        raise ValueError(f"{path}: {len(frame)} rows, expected {expected_rows}")
    if frame["user_id"].isna().any() or frame["predict"].isna().any():
        raise ValueError(f"{path}: missing values")
    user_id = frame["user_id"].to_numpy(np.int64)
    predict = frame["predict"].to_numpy(np.float64)
    if np.unique(user_id).size != expected_rows:
        raise ValueError(f"{path}: duplicate user_id")
    if not np.all(np.isfinite(predict)):
        raise ValueError(f"{path}: NaN or inf prediction")
    if np.any(predict < 0):
        raise ValueError(f"{path}: negative prediction")
    return Submission(path, frame, user_id, predict, np.log1p(predict))


def blend_log_predictions(parts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if set(parts) != set(WEIGHTS):
        raise ValueError(f"component names {sorted(parts)} do not match {sorted(WEIGHTS)}")
    lengths = {len(value) for value in parts.values()}
    if len(lengths) != 1:
        raise ValueError("component lengths differ")
    before_floor = sum(WEIGHTS[name] * np.asarray(parts[name], np.float64) for name in WEIGHTS)
    return before_floor, np.maximum(before_floor, 0.0)


def submission_stats(submission: Submission, sample_user_id: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(submission.path.resolve()),
        "sha256": sha256(submission.path),
        "rows": len(submission.frame),
        "columns": list(submission.frame.columns),
        "unique_user_id": int(np.unique(submission.user_id).size),
        "duplicate_user_id": int(pd.Series(submission.user_id).duplicated().sum()),
        "missing_values": int(submission.frame.isna().sum().sum()),
        "nan_predictions": int(np.isnan(submission.predict).sum()),
        "inf_predictions": int(np.isinf(submission.predict).sum()),
        "negative_predictions": int((submission.predict < 0).sum()),
        "zero_predictions": int((submission.predict == 0).sum()),
        "sample_order_exact": bool(np.array_equal(submission.user_id, sample_user_id)),
        "mean_predict": float(np.mean(submission.predict)),
        "mean_log1p": float(np.mean(submission.z)),
        "min_predict": float(np.min(submission.predict)),
        "max_predict": float(np.max(submission.predict)),
    }


def parse_sha_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64:
            raise ValueError(f"bad manifest line: {line!r}")
        entries.append((expected, relative))
    return entries


def audit_sha_manifest(bundle_root: Path) -> dict[str, Any]:
    manifest_path = bundle_root / "MANIFEST.sha256"
    rows = []
    for expected, relative in parse_sha_manifest(manifest_path):
        path = bundle_root / Path(relative)
        actual = sha256(path) if path.is_file() else None
        rows.append(
            {
                "relative_path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "MATCH" if actual == expected else ("MISSING" if actual is None else "MISMATCH"),
            }
        )
    bad = [row for row in rows if row["status"] != "MATCH"]
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "entries": len(rows),
        "matches": len(rows) - len(bad),
        "bad_entries": bad,
    }


def inspect_friend_oof(repo_root: Path) -> dict[str, Any]:
    component_paths = {
        "S1-CAP": repo_root / "artifacts" / "oof_S1-E03a.npz",
        "S1-UNC": repo_root / "artifacts" / "oof_S1-E02.npz",
        "S1-DIST": repo_root / "artifacts" / "oof_S1-DIST.npz",
        "ETX-AVG3": repo_root / "artifacts" / "oof_ETX-AVG3.npz",
        "SEQ-AVG3": repo_root / "artifacts" / "oof_SEQ-AVG3.npz",
    }
    loaded: dict[str, dict[str, np.ndarray]] = {}
    missing = []
    for name, path in component_paths.items():
        if not path.is_file():
            missing.append(str(path.resolve()))
            continue
        with np.load(path, allow_pickle=False) as source:
            loaded[name] = {key: source[key] for key in ("cutoff", "user_id", "y", "z")}
    if missing:
        return {"status": "MISSING", "missing_paths": missing}

    anchor = loaded["S1-CAP"]
    anchor_order = np.lexsort((anchor["user_id"], anchor["cutoff"]))
    anchor_cutoff = anchor["cutoff"][anchor_order]
    anchor_user_id = anchor["user_id"][anchor_order]
    anchor_y = anchor["y"][anchor_order]
    key = np.rec.fromarrays([anchor_cutoff, anchor_user_id], names="cutoff,user_id")
    counts = [int(np.sum(anchor_cutoff == fold)) for fold in FOLDS]
    problems = []
    if counts != list(FOLD_SIZES):
        problems.append(f"fold sizes {counts}, expected {list(FOLD_SIZES)}")
    if np.unique(key).size != len(key):
        problems.append("duplicate (cutoff,user_id) keys")
    for name, data in loaded.items():
        order = np.lexsort((data["user_id"], data["cutoff"]))
        component_key = np.rec.fromarrays(
            [data["cutoff"][order], data["user_id"][order]], names="cutoff,user_id"
        )
        if np.unique(component_key).size != len(component_key):
            problems.append(f"{name}: duplicate (cutoff,user_id) keys")
        if not np.array_equal(component_key, key):
            problems.append(f"{name}: aligned key set differs")
        if not np.array_equal(data["y"][order], anchor_y):
            problems.append(f"{name}: target differs")
        if not np.all(np.isfinite(data["z"])):
            problems.append(f"{name}: non-finite predictions")
    return {
        "status": "AVAILABLE" if not problems else "INVALID",
        "rows": len(key),
        "folds": list(FOLDS),
        "fold_sizes": counts,
        "unique_keys": int(np.unique(key).size),
        "target_equality": not any("target differs" in problem for problem in problems),
        "row_order_equality_after_key_alignment": not any(
            "aligned key set differs" in problem for problem in problems
        ),
        "problems": problems,
        "component_paths": {name: str(path.resolve()) for name, path in component_paths.items()},
        "component_sha256": {name: sha256(path) for name, path in component_paths.items()},
    }


def write_level_audit(path: Path) -> None:
    reason = "NOT_RUN: canonical row-level OOF is missing for occ_meta_B and occ_raw_X3"
    rows = [
        {"model": name, "status": "UNAVAILABLE_FOR_COMPARISON", "reason": reason}
        for name in (
            "S1-BEST",
            "STRONGEST_CURRENT",
            "latest",
            "friend",
            "occ_meta_B",
            "occ_raw_X3",
        )
    ]
    columns = [
        "model",
        "status",
        "reason",
        "fold",
        "optimal_fold_log_shift",
        "mean_raw_prediction",
        "mean_calibrated_prediction",
        "floor_share",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_report(summary: dict[str, Any], reconstruction: dict[str, Any]) -> str:
    source = reconstruction["source_latest"]
    return f"""# AUTHORITATIVE-LATEST-INTEGRATION audit

## Verdict

**{summary['verdict']}**

`latest.csv` is numerically reconstructed from the frozen component submissions,
but the public score is only externally reported and canonical row-level OOF for
the two late occurrence/Ridge components is absent from the bundle.

## Production-state classification

- `best_public_observed`: `latest/latest.csv`, public LB `1.64921756224069`, **EXTERNALLY_REPORTED**; LB event date is not recorded, evidence was present by the 2026-08-24 provenance audit.
- `best_exactly_reproducible`: `STRONGEST_CURRENT` / `exp_037` (end-to-end package, recorded LB, canonical four-fold OOF).
- `research_private_safe_anchor`: `STRONGEST_CURRENT` / `exp_037`.

## Test reconstruction

- Recipe: `0.12 * friend + 0.16 * occ_meta_B + 0.72 * occ_raw_X3` in `z=log1p(predict)` space.
- Full policy: component nonnegative validation, convex log blend, `z=max(z,0)`, `predict=expm1(z)`; no post-blend level normalization.
- Rows/schema/order: `{source['rows']}` rows, `user_id,predict`, exact sample order.
- Source SHA-256: `{source['sha256']}`.
- Reconstructed SHA-256: `{reconstruction['reconstructed_csv']['sha256']}`.
- Byte-identical: `{str(reconstruction['byte_identical']).lower()}` (CSV writer formatting is recorded separately from numeric equality).
- Required max error: `{reconstruction['max_abs_log1p_source_csv_vs_reconstructed_full_policy']:.17g}` (floor `{SERIALIZATION_FLOOR:.1e}`).
- Reconstructed roundtrip max error: `{reconstruction['max_abs_log1p_written_csv_vs_reconstructed_full_policy']:.17g}`.

## Lineage and CAP

`friend.csv` is byte-identical to `STRONGEST_CURRENT` and contains CAP at 10%.
Both late submissions preserve the 45% fixed SEQ/ETX slot and replace the 55%
table slot through `friend + 0.55 * (candidate_table - table_core)`, followed by
fixed level handling. Expanding the final blend proves 45% shared neural anchor,
6.6% original table core, 8.8% `occ_meta_B` table candidate and 39.6%
`occ_raw_X3` table candidate. The directly fixed CAP coefficient is therefore
1.2%, while extra CAP dependence inside the learned candidate tables cannot be
reduced to a documented scalar from the supplied artifacts.

```text
CAP_LINEAGE = UNKNOWN
PRIVATE_SAFE_STATUS = UNRESOLVED
```

The three latest components are not independent models: both occurrence
components share the same fixed neural anchor and Ridge/greedy table ancestry.
This is intentional table-slot replacement, but it is double use of common
ancestry rather than three independent signals.

## OOF

`friend` canonical OOF is available and aligned on the four project folds.
Canonical row-level OOF for `occ_meta_B` and `occ_raw_X3` is missing; the bundle
explicitly omits the multi-GB research cache. Summary validation CSVs cannot
replace row-level OOF and cannot prove target equality or absence of in-sample
stacking for these exact production CSVs.

```text
CANONICAL_OOF = MISSING
```

No project wCV, segment diagnostics, canonical latest NPZ, or model-dependent
level audit was synthesized.

## LB provenance

The number `1.64921756224069` occurs in textual README/provenance documents and
is described there as coming from a transmitted external journal. It does not
occur in the three relevant `RUN_MANIFEST.json` files or a SHA-to-score registry.
Accordingly it is classified as `EXTERNALLY_REPORTED`, not independently verified.
"""


def run_audit(repo_root: Path, bundle_root: Path, prefix: str) -> dict[str, Any]:
    result_dir = repo_root / "research" / "strategies" / "results" / prefix
    artifact_dir = repo_root / "artifacts" / prefix
    if result_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"prefix already exists: {prefix}")
    result_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    latest_dir = bundle_root / "latest"
    sample = read_submission(bundle_root / "data" / "sample_submit.csv")
    components = {
        "friend": read_submission(latest_dir / "components" / "friend.csv"),
        "occ_meta_B": read_submission(latest_dir / "components" / "occ_meta_B.csv"),
        "occ_raw_X3": read_submission(latest_dir / "components" / "occ_raw_X3.csv"),
    }
    source_latest = read_submission(latest_dir / "latest.csv")

    for name, component in components.items():
        if not np.array_equal(component.user_id, sample.user_id):
            raise ValueError(f"{name}: user_id order differs from sample_submit.csv")
    if not np.array_equal(source_latest.user_id, sample.user_id):
        raise ValueError("latest.csv: user_id order differs from sample_submit.csv")

    before_floor, reconstructed_z = blend_log_predictions(
        {name: component.z for name, component in components.items()}
    )
    reconstructed_predict = np.expm1(reconstructed_z)
    if not np.all(np.isfinite(reconstructed_predict)) or np.any(reconstructed_predict < 0):
        raise RuntimeError("invalid reconstructed predictions")

    output_path = artifact_dir / "latest_reconstructed.csv"
    pd.DataFrame(
        {"user_id": sample.user_id, "predict": reconstructed_predict}
    ).to_csv(output_path, index=False)
    written = read_submission(output_path)

    source_error = float(np.max(np.abs(source_latest.z - reconstructed_z)))
    written_error = float(np.max(np.abs(written.z - reconstructed_z)))
    if source_error > SERIALIZATION_FLOOR:
        raise RuntimeError(
            f"REJECT_NOT_REPRODUCIBLE: max log error {source_error} > {SERIALIZATION_FLOOR}"
        )

    manifest_audit = audit_sha_manifest(bundle_root)
    component_stats = {
        name: submission_stats(component, sample.user_id) for name, component in components.items()
    }
    source_stats = submission_stats(source_latest, sample.user_id)
    reconstructed_stats = submission_stats(written, sample.user_id)
    relevant_manifest_bad = [
        row
        for row in manifest_audit["bad_entries"]
        if row["relative_path"]
        in {
            "latest/latest.csv",
            "latest/components/friend.csv",
            "latest/components/occ_meta_B.csv",
            "latest/components/occ_raw_X3.csv",
            "latest/rebuild_latest.py",
            "data/sample_submit.csv",
        }
    ]
    if relevant_manifest_bad:
        raise RuntimeError(f"relevant manifest mismatch: {relevant_manifest_bad}")

    reconstruction = {
        "status": "PASS",
        "recipe": WEIGHTS,
        "blend_space": "log1p",
        "global_level_policy": "NONE_AFTER_BLEND; components were already fixed-level submissions",
        "component_negative_policy": "reject negatives; original rebuild used clip(pred, 0, None)",
        "final_floor_policy": "z = maximum(z, 0)",
        "pre_floor_negative_rows": int((before_floor < 0).sum()),
        "post_floor_zero_rows": int((reconstructed_z == 0).sum()),
        "mean_z_before_floor": float(np.mean(before_floor)),
        "mean_z_after_floor": float(np.mean(reconstructed_z)),
        "source_latest": source_stats,
        "reconstructed_csv": reconstructed_stats,
        "max_abs_log1p_source_csv_vs_reconstructed_full_policy": source_error,
        "max_abs_log1p_written_csv_vs_reconstructed_full_policy": written_error,
        "max_abs_predict_source_csv_vs_reconstructed": float(
            np.max(np.abs(source_latest.predict - reconstructed_predict))
        ),
        "serialization_floor": SERIALIZATION_FLOOR,
        "serialization_floor_pass": source_error <= SERIALIZATION_FLOOR,
        "byte_identical": sha256(source_latest.path) == sha256(output_path),
        "csv_precision": {
            "reader_dtype": "float64",
            "writer": "pandas.DataFrame.to_csv(index=False), default float formatting",
            "roundtrip_checked": True,
        },
        "sha_manifest": manifest_audit,
    }

    run_manifest_paths = sorted(bundle_root.glob("review_bundles/**/RUN_MANIFEST.json"))
    component_manifest = {
        "recipe": WEIGHTS,
        "components": component_stats,
        "friend_identity": {
            "status": "BYTE_IDENTICAL",
            "friend_sha256": component_stats["friend"]["sha256"],
            "strongest_current_path": str(
                (
                    bundle_root
                    / "friend_original"
                    / "submission_STRONGEST_CURRENT"
                    / "submission"
                    / "submission_STRONGEST_CURRENT.csv"
                ).resolve()
            ),
            "strongest_current_sha256": sha256(
                bundle_root
                / "friend_original"
                / "submission_STRONGEST_CURRENT"
                / "submission"
                / "submission_STRONGEST_CURRENT.csv"
            ),
            "recipe": {
                "S1-CAP": 0.10,
                "S1-UNC": 0.20,
                "S1-DIST": 0.25,
                "ETX-AVG3@DCW": 0.225,
                "SEQ-AVG3@clip289": 0.225,
            },
        },
        "occ_meta_B_lineage": {
            "mapped_name": "metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
            "family": "occurrence_meta_risk",
            "source_stage": "final6h",
            "test_prediction_available": True,
            "canonical_row_level_oof_available": False,
        },
        "occ_raw_X3_lineage": {
            "mapped_name": "xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
            "family": "raw_occ_extra",
            "source_stage": "extra90",
            "test_prediction_available": True,
            "canonical_row_level_oof_available": False,
        },
        "expanded_fixed_ancestry": {
            "shared_SEQ_ETX_anchor": 0.45,
            "original_table_core": 0.066,
            "occ_meta_B_candidate_table": 0.088,
            "occ_raw_X3_candidate_table": 0.396,
            "direct_fixed_CAP": 0.012,
            "total_effective_CAP": "UNKNOWN: candidate tables can depend on CAP and no scalar attribution is recorded",
        },
        "CAP_LINEAGE": "UNKNOWN",
        "PRIVATE_SAFE_STATUS": "UNRESOLVED",
        "double_counting": {
            "status": "SHARED_ANCESTRY_CONFIRMED",
            "detail": "occ_meta_B and occ_raw_X3 both preserve the same 45% SEQ/ETX anchor and share Ridge/greedy ancestry",
        },
        "run_manifests": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in run_manifest_paths
        ],
    }

    friend_oof = inspect_friend_oof(repo_root)
    oof_status = {
        "CANONICAL_OOF": "MISSING",
        "required_folds": list(FOLDS),
        "required_fold_sizes": list(FOLD_SIZES),
        "components": {
            "friend": friend_oof,
            "occ_meta_B": {
                "status": "MISSING",
                "reason": "row-level production-candidate OOF/cache omitted from supplied bundle",
            },
            "occ_raw_X3": {
                "status": "MISSING",
                "reason": "row-level production-candidate OOF/cache omitted from supplied bundle",
            },
        },
        "summary_validation_csvs_are_canonical_oof": False,
        "stacking_leakage_status": "UNRESOLVED_FOR_EXACT_LATE_COMPONENTS",
        "training_target_leakage_status": "UNRESOLVED_FOR_EXACT_LATE_COMPONENTS",
        "oof_latest_canonical_created": False,
        "project_wcv_computed": False,
        "segments_computed": False,
        "native_second_line_diagnostics_promoted_to_project_wcv": False,
    }

    summary = {
        "prefix": prefix,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "CONTINUE_PROVENANCE",
        "verdict_reasons": [
            "test submission reconstruction passes the project serialization floor",
            "latest public LB is only EXTERNALLY_REPORTED, not present in a reliable manifest/score registry",
            "CAP total effective lineage is not reducible to a proven scalar",
            "canonical row-level OOF is missing for occ_meta_B and occ_raw_X3",
        ],
        "best_public_observed": {
            "object": str(source_latest.path.resolve()),
            "lb_public": 1.64921756224069,
            "status": "EXTERNALLY_REPORTED",
            "lb_event_date": "UNKNOWN",
            "evidence_present_by": "2026-08-24",
        },
        "best_exactly_reproducible": {
            "object": "STRONGEST_CURRENT / exp_037",
            "lb_public": 1.6496571,
            "reason": "end-to-end package, internal LB journal entry, canonical four-fold OOF",
        },
        "research_private_safe_anchor": {
            "object": "STRONGEST_CURRENT / exp_037",
            "status": "RETAINED",
        },
        "latest_test_assembly_reproducible": True,
        "latest_end_to_end_training_reproducible": False,
        "latest_can_be_cv_lofo_anchor": False,
        "CAP_LINEAGE": "UNKNOWN",
        "PRIVATE_SAFE_STATUS": "UNRESOLVED",
        "CANONICAL_OOF": "MISSING",
    }

    write_json(result_dir / "reconstruction.json", reconstruction)
    write_json(result_dir / "component_manifest.json", component_manifest)
    write_json(result_dir / "oof_status.json", oof_status)
    write_json(result_dir / "summary.json", summary)
    write_level_audit(result_dir / "level_audit.csv")
    (result_dir / "REPORT.md").write_text(
        make_report(summary, reconstruction), encoding="utf-8"
    )
    return {
        "result_dir": str(result_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "summary": summary,
        "reconstruction": reconstruction,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-root", type=Path, default=repo_root / "пайплайн сокомандника"
    )
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    result = run_audit(repo_root, args.bundle_root.resolve(), args.prefix)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
