from __future__ import annotations

import csv
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - inventory still works without parquet metadata
    pq = None


SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()


DIR_EXPERIMENT_MAP = {
    "BLOCK4_SAF": "team_a:exp_039",
    "FRESH_CONTRAST": "team_a:exp_040",
    "BTYD_DAY_BGNBD": "team_a:exp_047",
    "BTYD_DAY_BGNBD_EXP047_V2": "team_a:exp_047",
    "BTYD05_PROD_EXP050": "team_a:exp_050",
    "BTYD_STABLE_EXP051": "team_a:exp_051",
    "CHANNEL_SHAPLEY_SPLIT": "team_a:exp_052",
    "CHANNEL_SHAPLEY_EXP052": "team_a:exp_052",
    "RESIDUAL_SIGNAL_DISCOVERY": "team_a:exp_053",
    "RESDISC_053": "team_a:exp_053",
    "BURST_GAP_EXP054": "team_a:exp_054",
    "LANDMARK_MEMORY_EXP055": "team_a:exp_055",
    "LATE_SSL_EXP056": "team_a:exp_056",
    "STATE_REWEIGHT_EXP057": "team_a:exp_057",
    "FINGERPRINT_EXP058": "team_a:exp_058",
    "LEVEL_MINUS_006_EXP060": "team_a:exp_060",
    "OPEN_FUNNEL_EXP061": "team_a:exp_061",
    "PLATFORM_DETREND_EXP062": "team_a:exp_062",
    "OCCURRENCE_REVISIT_EXP063": "team_a:exp_063",
    "EVENT_ORDER_EXP064": "team_a:exp_064",
    "FINAL_INTEGRATION_EXP065": "team_a:exp_065",
    "LATEST_DELTA_COMPAT_EXP066_A1": "team_a:exp_066",
    "AUTHORITATIVE_LATEST_AUDIT_20260825_160144": "team_a:exp_067",
    "AUTHORITATIVE_LATEST_AUDIT_20260825_160144_V2": "team_a:exp_067",
    "RECENCY_RIDGE_PRED_EXP068_A1": "team_a:exp_068",
    "ETX1": "team_a:exp_036",
    "ETX2": "team_a:exp_037",
    "FNL1": "team_a:exp_038",
    "MIX9": "team_a:exp_035",
    "STRATEGY_01": "team_a:exp_019",
    "STRATEGY_02": "team_a:exp_020",
    "STRATEGY_08": "team_a:exp_021",
    "HOLIDAY-YOY": "team_a:exp_023",
    "MHZ": "team_a:exp_024",
    "SEQ": "team_a:exp_025",
    "SEQ2": "team_a:exp_026",
    "SEQ3": "team_a:exp_028",
    "SEQ4": "team_a:exp_029",
}


def load_checksums() -> dict[str, str]:
    path = DEST / "inventory" / "source_checksums_sha256.csv"
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["relative_path"].replace("\\", "/")] = row["sha256"]
    return result


def rel(path: Path) -> str:
    return path.relative_to(SOURCE).as_posix()


def text_probe(path: Path, limit: int = 262_144) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def csv_schema(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            sample = handle.read(131_072)
        lines = sample.splitlines()
        if not lines:
            return "empty"
        try:
            dialect = csv.Sniffer().sniff(sample[:65_536], delimiters=",;\t")
            columns = next(csv.reader([lines[0]], dialect=dialect))
        except Exception:
            columns = lines[0].split(",")
        return "columns=" + "|".join(columns[:80])
    except Exception as exc:
        return f"schema_error={type(exc).__name__}"


def parquet_schema(path: Path) -> str:
    if pq is None:
        return "pyarrow_unavailable"
    try:
        meta = pq.ParquetFile(path).metadata
        names = list(meta.schema.names)
        return f"rows={meta.num_rows};row_groups={meta.num_row_groups};columns=" + "|".join(names[:120])
    except Exception as exc:
        return f"schema_error={type(exc).__name__}"


def npy_header(stream) -> tuple[tuple[int, ...], str]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    return shape, str(dtype)


def numpy_schema(path: Path) -> str:
    try:
        if path.suffix.lower() == ".npy":
            with path.open("rb") as handle:
                shape, dtype = npy_header(handle)
            return f"array={path.stem}:{shape}:{dtype}"
        arrays: list[str] = []
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".npy"):
                    continue
                with archive.open(info) as handle:
                    shape, dtype = npy_header(handle)
                arrays.append(f"{Path(info.filename).stem}:{shape}:{dtype}")
                if len(arrays) >= 80:
                    arrays.append("...")
                    break
        return "arrays=" + "|".join(arrays)
    except Exception as exc:
        return f"schema_error={type(exc).__name__}"


def json_schema(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            return "keys=" + "|".join(list(value)[:100])
        if isinstance(value, list):
            return f"list_length={len(value)}"
        return f"scalar={type(value).__name__}"
    except Exception as exc:
        return f"schema_error={type(exc).__name__}"


def inspect_schema(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return csv_schema(path)
    if suffix == ".parquet":
        return parquet_schema(path)
    if suffix in {".npz", ".npy"}:
        return numpy_schema(path)
    if suffix == ".json" and path.stat().st_size <= 50_000_000:
        return json_schema(path)
    return "unknown"


def associations(relative: str) -> list[str]:
    found: list[str] = []
    upper = relative.upper()
    for dirname, exp_id in DIR_EXPERIMENT_MAP.items():
        if dirname.upper() in upper:
            found.append(exp_id)
    for match in re.finditer(r"(?i)(?:^|[/_\\-])EXP[_-]?0*(\d{1,3})(?:\D|$)", relative):
        found.append(f"team_a:exp_{int(match.group(1)):03d}")
    for match in re.finditer(r"(?i)\b(S1-(?:B0|E\d+[A-Z]?|DIST(?:-F4)?|MIX-E11|SEEDAVG\d+|ROUNDS|SAMPLE-[AB]|GAPAXIS|VAL-W))\b", relative):
        found.append("team_a_run:" + match.group(1).upper())
    for token in ("SEQ", "ETX", "FNL", "GAP", "PT", "MHZ", "BTYD", "RIDGE", "ZERO", "FRESH", "MIX"):
        if re.search(rf"(?i)(?:^|[/_\\-]){token}(?:[/_\\-]|\d|$)", relative):
            found.append("component_family:" + token.lower())
    return sorted(set(found)) or ["unknown"]


def classify(path: Path, relative: str, probe: str) -> tuple[str, str]:
    lower = relative.lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    roles: list[str] = []

    if lower.startswith("data/") or "/data/" in lower:
        roles.append("dataset")
    if "submission" in name or lower.startswith("submissions/") or "/submission/" in lower:
        roles.append("submission")
    if re.search(r"(^|[/_\\-])oof([/_\\.-]|$)", lower):
        roles.append("oof_prediction")
    if ("test" in name and suffix in {".npy", ".npz", ".parquet", ".csv"}) or "test_pred" in lower:
        roles.append("test_prediction")
    if suffix in {".pt", ".pth", ".ckpt", ".bin"} or name.startswith("model_"):
        roles.append("checkpoint_or_model")
    if "fold" in name and suffix in {".csv", ".json", ".npz", ".parquet"}:
        roles.append("fold_artifact")
    if "manifest" in name or name in {"log.csv", "submissions.csv", "run_start.json"}:
        roles.append("manifest")
    if any(word in name for word in ("metric", "curve", "diagnostic", "summary", "score", "validation")) and suffix in {".csv", ".json", ".txt", ".md"}:
        roles.append("metrics_or_diagnostics")
    if "lb" in name or "leaderboard" in name or "lb_public" in probe[:50_000]:
        roles.append("leaderboard_evidence_candidate")
    if suffix in {".py", ".sh", ".mjs", ".ipynb"}:
        roles.append("code")
        if re.search(r"(?i)(train|fit|run)", name) or re.search(r"(?i)(\.fit\(|lgb\.train|torch\.optim|backward\()", probe):
            roles.append("training_script")
        if re.search(r"(?i)(predict|inference|submit|make_submission|test_model)", name) or re.search(r"(?i)(\.predict\(|submission)", probe):
            roles.append("inference_script")
        if re.search(r"(?i)(ensemble|blend|stack|lofo|weight|merge)", name) or re.search(r"(?i)(blend|ensemble|log-space|log_space)", probe):
            roles.append("ensemble_script")
        if re.search(r"(?i)(feature|build_frame|\bfe\b)", name) or "build_features" in probe:
            roles.append("feature_generation_script")
        if re.search(r"(?i)(calibr|postprocess|shrink|detrend|shift)", name):
            roles.append("postprocessing_script")
        if "config" in name or re.search(r"(?m)^[A-Z][A-Z0-9_]+\s*=", probe):
            roles.append("config_or_parameter_source")
    if suffix == ".ipynb":
        roles.append("notebook")
    if suffix == ".md" and re.search(r"(?i)(^|/)(exp[_-]?\d+|report|hypothesis_card)[^/]*\.md$", lower):
        roles.append("primary_experiment_report_candidate")
    if not roles:
        roles.append("other")

    if "dataset" in roles:
        purpose = "source_or_derived_dataset"
    elif "submission" in roles:
        purpose = "competition_submission_or_candidate"
    elif "oof_prediction" in roles:
        purpose = "out_of_fold_prediction"
    elif "test_prediction" in roles:
        purpose = "test_prediction"
    elif "checkpoint_or_model" in roles:
        purpose = "trained_model_state"
    elif "metrics_or_diagnostics" in roles:
        purpose = "run_metrics_or_diagnostics"
    elif "code" in roles:
        purpose = "experiment_or_pipeline_code"
    else:
        purpose = "unclassified"
    return ";".join(sorted(set(roles))), purpose


def excluded_reason(relative: str) -> str | None:
    lower = relative.lower()
    name = Path(lower).name
    if name in {"agents.md", "agent.md", "state.md", "history.md", "claude.md"}:
        return "explicitly_excluded_agent_or_state_document"
    if name in {"probably_exp.md", "independent_stage1_findings.md"}:
        return "secondary_project_state_or_findings_summary"
    if name in {"readme.md", "discription.md"} and "/results/" not in lower:
        return "navigation_or_project_summary_not_used_as_fact_source"
    if suffix_of(relative) in {".md", ".txt", ".rst"} and any(token in name for token in ("roadmap", "todo", "master_summary", "executive_summary", "strategies_index", "project_state", "plan")):
        return "instruction_roadmap_or_secondary_summary"
    if lower.startswith("research/strategy") and suffix_of(relative) == ".md" and "/results/" not in lower:
        return "secondary_strategy_document"
    if lower.startswith("docs/") and suffix_of(relative) in {".md", ".txt"}:
        return "instruction_or_plan_document"
    return None


def suffix_of(relative: str) -> str:
    return Path(relative).suffix.lower()


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checksums = load_checksums()
    files = [
        p
        for p in SOURCE.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
    ]
    records: list[dict] = []
    excluded: list[dict] = []
    for index, path in enumerate(sorted(files), 1):
        relative = rel(path)
        probe = text_probe(path) if path.suffix.lower() in {".py", ".sh", ".mjs", ".md", ".txt", ".json"} and path.stat().st_size <= 5_000_000 else ""
        roles, purpose = classify(path, relative, probe)
        reason = excluded_reason(relative)
        if reason:
            excluded.append({"original_path": relative, "reason": reason, "used_for_facts": "no"})
        record = {
            "original_path": relative,
            "filename": path.name,
            "extension": path.suffix.lower() or "[none]",
            "type": roles,
            "purpose": purpose,
            "size_bytes": path.stat().st_size,
            "sha256": checksums.get(relative, "unknown"),
            "experiment_association": ";".join(associations(relative)),
            "schema_or_arrays": inspect_schema(path),
            "last_write_utc": path.stat().st_mtime_ns,
            "copied": "no",
            "evidence_excluded": "yes" if reason else "no",
        }
        records.append(record)
        if index % 250 == 0:
            print(f"inspected {index}/{len(files)}", flush=True)

    fields = [
        "original_path", "filename", "extension", "type", "purpose", "size_bytes", "sha256",
        "experiment_association", "schema_or_arrays", "last_write_utc", "copied", "evidence_excluded",
    ]
    write_csv(DEST / "inventory" / "files.csv", records, fields)
    write_csv(
        DEST / "artifacts" / "manifest.csv",
        [r for r in records if any(x in r["type"] for x in ("prediction", "submission", "checkpoint", "fold_artifact", "manifest", "metrics", "dataset"))],
        fields,
    )
    write_csv(
        DEST / "code_index" / "scripts.csv",
        [r for r in records if "code" in r["type"]],
        fields,
    )
    write_csv(
        DEST / "inventory" / "datasets.csv",
        [r for r in records if "dataset" in r["type"]],
        fields,
    )
    write_csv(
        DEST / "inventory" / "report_candidates.csv",
        [r for r in records if "report" in r["type"]],
        fields,
    )
    write_csv(
        DEST / "inventory" / "excluded_interpretive_documents.csv",
        excluded,
        ["original_path", "reason", "used_for_facts"],
    )

    by_top: dict[str, dict[str, int]] = {}
    for row in records:
        top = row["original_path"].split("/", 1)[0]
        item = by_top.setdefault(top, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(row["size_bytes"])
    write_csv(
        DEST / "inventory" / "source_tree_summary.csv",
        ({"top_level": k, **v} for k, v in sorted(by_top.items())),
        ["top_level", "files", "bytes"],
    )

    role_counts: Counter[str] = Counter()
    for row in records:
        role_counts.update(row["type"].split(";"))
    print(json.dumps({"files": len(records), "excluded_docs": len(excluded), "roles": role_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
