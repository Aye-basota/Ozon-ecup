from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()


def scalar(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "unknown"
        return f"{value:.12g}"
    if isinstance(value, str):
        return value.strip() or "unknown"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def flatten(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if depth > 8:
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (dict, list)):
                out.update(flatten(child, path, depth + 1))
            else:
                out[path] = child
    elif isinstance(value, list):
        if len(value) <= 20 and all(not isinstance(x, (dict, list)) for x in value):
            out[prefix] = value
        else:
            for index, child in enumerate(value[:100]):
                path = f"{prefix}[{index}]"
                if isinstance(child, (dict, list)):
                    out.update(flatten(child, path, depth + 1))
                else:
                    out[path] = child
    return out


def pick(flat: dict[str, Any], exact: tuple[str, ...], suffixes: tuple[str, ...] = ()) -> tuple[str, str]:
    for key in exact:
        if key in flat:
            return scalar(flat[key]), key
    for suffix in suffixes:
        candidates = sorted(key for key in flat if key.lower().endswith(suffix.lower()))
        if candidates:
            key = candidates[0]
            return scalar(flat[key]), key
    return "unknown", "unknown"


def metric_candidates(flat: dict[str, Any]) -> dict[str, Any]:
    keep: dict[str, Any] = {}
    pattern = re.compile(r"(?i)(rmsle|wcv|cv_mean|cv_score|fold_score|delta|gain|lb|leaderboard|runtime|verdict|score)")
    for key, value in flat.items():
        if pattern.search(key) and isinstance(value, (str, int, float, bool, list)):
            keep[key] = value
        if len(keep) >= 250:
            keep["__truncated__"] = True
            break
    return keep


def infer_run_id(path: Path) -> str:
    name = path.stem
    for prefix in ("report_", "curve_", "summary_", "validation_", "metrics_"):
        if name.lower().startswith(prefix):
            return name[len(prefix):]
    if name.lower() in {"summary", "validation", "standalone_metrics", "reconstruction", "audit"}:
        return path.parent.name
    return name


def associations(relative: str) -> str:
    ids: list[str] = []
    for match in re.finditer(r"(?i)(?:^|[/_\\-])EXP[_-]?0*(\d{1,3})(?:\D|$)", relative):
        ids.append(f"team_a:exp_{int(match.group(1)):03d}")
    directory_map = {
        "BLOCK4_SAF": 39, "FRESH_CONTRAST": 40, "BTYD_DAY_BGNBD": 47,
        "BTYD05_PROD_EXP050": 50, "BTYD_STABLE_EXP051": 51,
        "CHANNEL_SHAPLEY_SPLIT": 52, "RESIDUAL_SIGNAL_DISCOVERY": 53,
        "BURST_GAP_EXP054": 54, "LANDMARK_MEMORY_EXP055": 55,
        "LATE_SSL_EXP056": 56, "STATE_REWEIGHT_EXP057": 57,
        "FINGERPRINT_EXP058": 58, "LEVEL_MINUS_006_EXP060": 60,
        "OPEN_FUNNEL_EXP061": 61, "PLATFORM_DETREND_EXP062": 62,
        "OCCURRENCE_REVISIT_EXP063": 63, "EVENT_ORDER_EXP064": 64,
        "FINAL_INTEGRATION_EXP065": 65, "LATEST_DELTA_COMPAT": 66,
        "AUTHORITATIVE_LATEST": 67, "RECENCY_RIDGE": 68,
    }
    upper = relative.upper()
    for token, number in directory_map.items():
        if token in upper:
            ids.append(f"team_a:exp_{number:03d}")
    return ";".join(sorted(set(ids))) or "unknown"


def evidence_record(
    path: Path,
    value: Any,
    *,
    relative_override: str | None = None,
    association_override: str | None = None,
    source_kind: str = "direct_run_metric_json",
) -> dict | None:
    if not isinstance(value, (dict, list)):
        return None
    flat = flatten(value)
    candidates = metric_candidates(flat)
    if not candidates:
        return None
    relative = relative_override or path.relative_to(SOURCE).as_posix()
    cv, cv_key = pick(flat, ("cv_mean", "metrics.cv_mean", "validation.cv_mean", "base.cv_mean"), (".cv_mean", ".mean_cv"))
    wcv, wcv_key = pick(flat, ("wcv", "base_wcv", "metrics.wcv", "validation.wcv"), (".wcv", ".weighted_cv"))
    oof, oof_key = pick(flat, ("oof_rmsle", "oof_score", "metrics.oof_rmsle"), (".oof_rmsle", ".oof_score"))
    delta, delta_key = pick(
        flat,
        ("delta", "delta_vs_b0", "delta_wcv", "nested_delta", "delta_vs_base", "metrics.delta"),
        (".delta_wcv", ".nested_delta", ".delta_vs_base", ".delta_vs_b0"),
    )
    lb, lb_key = pick(flat, ("lb_public", "lb_score", "leaderboard_score", "public_lb"), (".lb_public", ".lb_score"))
    runtime, runtime_key = pick(flat, ("runtime_s", "runtime", "elapsed_s"), (".runtime_s", ".elapsed_s"))
    verdict, verdict_key = pick(flat, ("verdict", "status", "decision"), (".verdict", ".decision"))
    folds, folds_key = pick(flat, ("fold_scores", "fold_cal", "folds", "validation.fold_scores"), (".fold_scores", ".fold_cal"))
    model, model_key = pick(flat, ("model", "model_family", "params.model", "config.model"), (".model_family",))
    seed, seed_key = pick(flat, ("seed", "params.params.seed", "params.seed", "config.seed"), (".seed",))
    params, params_key = pick(flat, ("params", "config", "model_config"))
    description, description_key = pick(flat, ("description", "name", "experiment", "hypothesis"), (".description",))
    return {
        "metric_record_id": f"json:{relative}",
        "run_id": infer_run_id(path),
        "experiment_association": association_override or associations(relative),
        "source_path": relative,
        "source_kind": source_kind,
        "evidence_tier": "2_run_saved_metric",
        "cv_score": cv,
        "cv_key": cv_key,
        "weighted_cv": wcv,
        "weighted_cv_key": wcv_key,
        "oof_score": oof,
        "oof_key": oof_key,
        "delta": delta,
        "delta_key": delta_key,
        "lb_score": lb,
        "lb_key": lb_key,
        "folds_or_scores": folds,
        "folds_key": folds_key,
        "runtime": runtime,
        "runtime_key": runtime_key,
        "verdict": verdict,
        "verdict_key": verdict_key,
        "model": model,
        "model_key": model_key,
        "seed": seed,
        "seed_key": seed_key,
        "params": params,
        "params_key": params_key,
        "description": description,
        "description_key": description_key,
        "all_metric_candidates": json.dumps(candidates, ensure_ascii=False, separators=(",", ":")),
    }


def manifest_records(path: Path, namespace: str, id_column: str) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            run_id = row.get(id_column) or row.get("exp_id") or row.get("id") or f"row_{index}"
            records.append({
                "metric_record_id": f"manifest:{namespace}:{run_id}:{index}",
                "run_id": run_id,
                "experiment_association": f"{namespace}:{run_id}",
                "source_path": path.relative_to(DEST).as_posix(),
                "source_kind": "experiment_manifest_row",
                "evidence_tier": "3_experiment_manifest",
                "cv_score": row.get("cv_mean") or "unknown",
                "cv_key": "cv_mean" if row.get("cv_mean") else "unknown",
                "weighted_cv": row.get("wcv") or "unknown",
                "weighted_cv_key": "wcv" if row.get("wcv") else "unknown",
                "oof_score": row.get("oof") or "unknown",
                "oof_key": "oof" if row.get("oof") else "unknown",
                "delta": row.get("delta_vs_b0") or row.get("delta") or "unknown",
                "delta_key": "delta_vs_b0" if row.get("delta_vs_b0") else ("delta" if row.get("delta") else "unknown"),
                "lb_score": row.get("lb_public") or "unknown",
                "lb_key": "lb_public" if row.get("lb_public") else "unknown",
                "folds_or_scores": row.get("fold_scores") or "unknown",
                "folds_key": "fold_scores" if row.get("fold_scores") else "unknown",
                "runtime": row.get("runtime_s") or "unknown",
                "runtime_key": "runtime_s" if row.get("runtime_s") else "unknown",
                "verdict": row.get("verdict") or "unknown",
                "verdict_key": "verdict" if row.get("verdict") else "unknown",
                "model": row.get("model") or "unknown",
                "model_key": "model" if row.get("model") else "unknown",
                "seed": "unknown",
                "seed_key": "unknown",
                "params": row.get("params") or "unknown",
                "params_key": "params" if row.get("params") else "unknown",
                "description": row.get("description") or "unknown",
                "description_key": "description" if row.get("description") else "unknown",
                "all_metric_candidates": json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            })
    return records


FIELDS = [
    "metric_record_id", "run_id", "experiment_association", "source_path", "source_kind", "evidence_tier",
    "cv_score", "cv_key", "weighted_cv", "weighted_cv_key", "oof_score", "oof_key", "delta", "delta_key",
    "lb_score", "lb_key", "folds_or_scores", "folds_key", "runtime", "runtime_key", "verdict", "verdict_key",
    "model", "model_key", "seed", "seed_key", "params", "params_key", "description", "description_key",
    "all_metric_candidates",
]


def main() -> None:
    records: list[dict] = []
    errors: list[dict] = []
    for path in SOURCE.rglob("*.json"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.stat().st_size > 100_000_000:
            continue
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
            record = evidence_record(path, value)
            if record:
                records.append(record)
        except Exception as exc:
            errors.append({"source_path": path.relative_to(SOURCE).as_posix(), "error": f"{type(exc).__name__}: {exc}"})

    git_artifact_root = DEST / "evidence" / "git_machine_artifacts"
    if git_artifact_root.exists():
        for path in git_artifact_root.rglob("*.json"):
            relative = path.relative_to(DEST).as_posix()
            namespace = path.relative_to(git_artifact_root).parts[0]
            association = {
                "independent_domain": "independent_domain:exp_028",
                "independent_calendar": "independent_calendar:exp_029",
            }.get(namespace, "unknown")
            try:
                with path.open("r", encoding="utf-8-sig") as handle:
                    value = json.load(handle)
                record = evidence_record(
                    path,
                    value,
                    relative_override=relative,
                    association_override=association,
                    source_kind="git_committed_run_metric_json",
                )
                if record:
                    records.append(record)
            except Exception as exc:
                errors.append({"source_path": relative, "error": f"{type(exc).__name__}: {exc}"})

    worktree_artifact_root = DEST / "evidence" / "worktree_artifacts"
    if worktree_artifact_root.exists():
        association_map = {
            "independent_anniversary": "independent_anniversary:exp_058",
            "independent_calendar": "independent_calendar:exp_029",
            "independent_domain": "independent_domain:exp_028",
            "independent_global_regime": "independent_global_regime:exp_057",
            "independent_renewal": "independent_renewal:exp_027",
            "team_a_s2": "team_a_s2:exp_012",
        }
        for path in worktree_artifact_root.rglob("*.json"):
            relative = path.relative_to(DEST).as_posix()
            namespace = path.relative_to(worktree_artifact_root).parts[0]
            try:
                with path.open("r", encoding="utf-8-sig") as handle:
                    value = json.load(handle)
                record = evidence_record(
                    path,
                    value,
                    relative_override=relative,
                    association_override=association_map.get(namespace, "unknown"),
                    source_kind="linked_worktree_run_metric_json",
                )
                if record:
                    records.append(record)
            except Exception as exc:
                errors.append({"source_path": relative, "error": f"{type(exc).__name__}: {exc}"})

    manifests = DEST / "evidence" / "machine_manifests"
    current_log = manifests / "team_a_current__log.csv"
    if current_log.exists():
        records.extend(manifest_records(current_log, "team_a_current_run", "exp_id"))
    s2_log = manifests / "team_a_s2__log.csv"
    if s2_log.exists():
        records.extend(manifest_records(s2_log, "team_a_s2_run", "id"))

    output = DEST / "registry" / "run_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (DEST / "registry" / "run_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (DEST / "contradictions" / "json_parse_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "error"])
        writer.writeheader()
        writer.writerows(errors)
    print(json.dumps({
        "run_metric_records": len(records),
        "direct_json_records": sum(r["source_kind"] in {"direct_run_metric_json", "git_committed_run_metric_json", "linked_worktree_run_metric_json"} for r in records),
        "manifest_rows": sum(r["source_kind"] == "experiment_manifest_row" for r in records),
        "json_parse_errors": len(errors),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
