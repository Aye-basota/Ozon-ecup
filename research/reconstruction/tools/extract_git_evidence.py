from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()

SELECTED_REFS = {
    "team_a_s2": "team-a-strategy-2-impl",
    "team_b_core": "origin/team-b-B2",
    "team_b_alt": "origin/team-b-strategy-1-impl",
    "independent_renewal": "codex/renewal-01",
    "independent_domain": "codex/domain-01",
    "independent_calendar": "codex/calendar-placebo-01",
    "independent_global_regime": "exp/057-global-regime-occ",
    "independent_anniversary": "exp/058-exact-anniversary",
}

BRANCH_ONLY_ALLOW = {
    "team_a_s2": re.compile(r"experiments/exp_0(?:09|10|11|12)_", re.I),
    "team_b_core": re.compile(r"experiments/exp_\d+_", re.I),
    "team_b_alt": re.compile(r"experiments/exp_\d+_", re.I),
    "independent_renewal": re.compile(r"experiments/exp_027_next_purchase_clock\.md$", re.I),
    "independent_domain": re.compile(r"experiments/exp_028_domain_shift\.md$", re.I),
    "independent_calendar": re.compile(r"experiments/exp_029_calendar_placebo\.md$", re.I),
    "independent_global_regime": re.compile(r"experiments/exp_057_global_regime_occ\.md$", re.I),
    "independent_anniversary": re.compile(r"experiments/exp_058_exact_anniversary\.md$", re.I),
}


def git(*args: str, check: bool = True) -> bytes:
    result = subprocess.run(["git", *args], cwd=SOURCE, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def section(text: str, titles: tuple[str, ...]) -> str:
    escaped = "|".join(re.escape(t) for t in titles)
    pattern = re.compile(rf"(?ims)^##\s+(?:{escaped})\s*$\n(.*?)(?=^##\s+|\Z)")
    match = pattern.search(text)
    return match.group(1).strip() if match else "unknown"


def first_number(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.M)
        if match:
            try:
                return f"{float(match.group(1)):.12g}"
            except ValueError:
                return match.group(1)
    return "unknown"


def infer_family(title: str, hypothesis: str, change: str) -> str:
    value = f"{title} {hypothesis} {change}".lower()
    rules = [
        ("production_integration_and_provenance", ("rebuild", "provenance", "production", "submission", "latest")),
        ("ensembles_and_stacking", ("ensemble", "blend", "mix", "weight grid", "stack")),
        ("neural_sequence_models", ("sequence", "tcn", "seq", "transformer", "encoder")),
        ("target_decomposition_and_hurdle", ("hurdle", "two-part", "two part", "classifier", "dist head", "distribution head", "delta model")),
        ("temporal_and_calendar", ("calendar", "season", "holiday", "temporal", "recency", "ewm", "trend", "anniversary", "renewal")),
        ("train_example_construction", ("cutoff", "panel", "sample", "train block", "dense")),
        ("calibration_and_postprocessing", ("calibr", "scale", "clip", "shrink", "offset")),
        ("domain_dataset_and_unlabeled", ("domain", "fingerprint", "unlabeled", "test-like", "placebo")),
        ("behavioral_and_btyd", ("bgnbd", "btyd", "purchase clock", "funnel", "occurrence")),
        ("feature_representation", ("feature", "aggregation", "personal time", "history depth", "search catalog", "aov")),
        ("tabular_model_and_hyperparameters", ("lightgbm", "lgbm", "num_leaves", "min_leaf", "learning rate", "tweedie", "huber", "ridge")),
        ("validation_and_diagnostics", ("validation", "diagnostic", "alignment", "variance predictability")),
    ]
    for family, needles in rules:
        if any(needle in value for needle in needles):
            return family
    return "other_or_unknown"


def parse_report(namespace: str, source_ref: str, source_path: str, text: str, sha256: str, clean_path: str) -> dict:
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    title = title_match.group(1).strip() if title_match else Path(source_path).stem
    date_match = re.search(r"(?im)^[-*]\s*\*\*Дата:\*\*\s*([^\r\n]+)", text)
    date = date_match.group(1).strip() if date_match else "unknown"
    id_match = re.search(r"(?i)exp[_-]?(\d{1,3}[a-z]?)", Path(source_path).stem)
    local_id = f"exp_{id_match.group(1).lower()}" if id_match else Path(source_path).stem
    hypothesis = section(text, ("Гипотеза", "Hypothesis"))
    change = section(text, ("Что изменено относительно базы", "Изменение относительно baseline", "Change"))
    facts = section(text, ("Результат", "Результаты", "Results", "FACTS"))
    interpretation = section(text, ("Вердикт и вывод", "Вердикт", "Вывод", "INTERPRETATION"))
    config = section(text, ("Конфиг прогона", "Конфигурация", "Config", "Reproducibility"))
    cv_score = first_number([
        r"CV\s+mean(?:\([^\)]*\))?\s*[:=]\s*[*`]*([0-9]+\.[0-9]+)",
        r"mean\s+RMSLE\s*[:=]\s*[*`]*([0-9]+\.[0-9]+)",
        r"wCV\s*[:=]\s*[*`]*([0-9]+\.[0-9]+)",
    ], facts)
    lb_score = first_number([
        r"LB(?:\s*\(public\)|\s+public)?\s*[:=]\s*[*`]*([0-9]+\.[0-9]+)",
        r"public\s+LB\s*[:=]\s*[*`]*([0-9]+\.[0-9]+)",
    ], facts)
    delta = first_number([
        r"(?:delta|дельта|разница|Δ)[^\r\n]{0,30}?([+-][0-9]+\.[0-9]+)",
        r"(?:лучше|хуже)[^\r\n]{0,40}?([+-]?[0-9]+\.[0-9]+)",
    ], facts)
    runtime = first_number([r"runtime(?:_s)?\s*[:=]\s*[*`]*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*(?:секунд|seconds|sec)"], text)
    verdict_token = "unknown"
    if interpretation != "unknown":
        match = re.search(r"(?i)\b(accept|accepted|keep|pass|success|успех|reject|rejected|fail|blocked|inconclusive|neutral|нейтрально)\b", interpretation)
        if match:
            verdict_token = match.group(1).lower()
    return {
        "experiment_id": f"{namespace}:{local_id}",
        "namespace": namespace,
        "local_id": local_id,
        "canonical_name": title,
        "family_inferred": infer_family(title, hypothesis, change),
        "date_reported": date,
        "hypothesis_reported": hypothesis,
        "change_reported": change,
        "facts_reported": facts,
        "interpretation_reported": interpretation,
        "config_reported": config,
        "cv_candidate": cv_score,
        "delta_candidate": delta,
        "lb_candidate": lb_score,
        "runtime_candidate": runtime,
        "verdict_reported": verdict_token,
        "source_ref": source_ref,
        "source_path": source_path,
        "clean_evidence_path": clean_path,
        "sha256": sha256,
        "evidence_tier": "7_primary_experiment_report",
    }


def export_report(namespace: str, source_ref: str, source_path: str, raw: bytes, occurrence_rows: list[dict], parsed_rows: list[dict]) -> None:
    sha = hashlib.sha256(raw).hexdigest()
    output = DEST / "experiments" / namespace / Path(source_path).name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    occurrence_rows.append({
        "namespace": namespace,
        "source_ref": source_ref,
        "source_path": source_path,
        "sha256": sha,
        "size_bytes": len(raw),
        "clean_evidence_path": output.relative_to(DEST).as_posix(),
    })
    text = raw.decode("utf-8", errors="replace")
    parsed_rows.append(parse_report(namespace, source_ref, source_path, text, sha, output.relative_to(DEST).as_posix()))


def main() -> None:
    refs_raw = git("for-each-ref", "--format=%(refname:short)%09%(objectname)%09%(creatordate:iso-strict)", "refs/heads", "refs/remotes")
    ref_rows: list[dict] = []
    for line in refs_raw.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            ref_rows.append({"ref": parts[0], "commit": parts[1], "commit_date": parts[2]})
    write_csv(DEST / "inventory" / "git_refs.csv", ref_rows, ["ref", "commit", "commit_date"])

    occurrence_rows: list[dict] = []
    parsed_rows: list[dict] = []

    for path in sorted((SOURCE / "experiments").glob("*.md")):
        if path.name.startswith("_") or not re.match(r"(?i)^exp[_-]?\d+", path.name):
            continue
        export_report("team_a_current", "WORKTREE", f"experiments/{path.name}", path.read_bytes(), occurrence_rows, parsed_rows)

    anniversary_report = SOURCE.parent / "exp058-exact-anniversary" / "experiments" / "exp_058_exact_anniversary.md"
    if anniversary_report.exists():
        export_report(
            "independent_anniversary",
            "LINKED_WORKTREE:exp/058-exact-anniversary",
            "experiments/exp_058_exact_anniversary.md",
            anniversary_report.read_bytes(),
            occurrence_rows,
            parsed_rows,
        )

    for namespace, ref_name in SELECTED_REFS.items():
        paths = git("ls-tree", "-r", "--name-only", ref_name, "--", "experiments").decode("utf-8", errors="replace").splitlines()
        allow = BRANCH_ONLY_ALLOW[namespace]
        for source_path in sorted(paths):
            if not allow.search(source_path) or not source_path.lower().endswith(".md"):
                continue
            raw = git("show", f"{ref_name}:{source_path}")
            export_report(namespace, ref_name, source_path, raw, occurrence_rows, parsed_rows)

    id_counts = Counter(row["experiment_id"] for row in parsed_rows)
    for row in parsed_rows:
        if id_counts[row["experiment_id"]] > 1:
            row["experiment_id"] += ":" + Path(row["source_path"]).stem.lower()

    occurrence_fields = ["namespace", "source_ref", "source_path", "sha256", "size_bytes", "clean_evidence_path"]
    write_csv(DEST / "inventory" / "git_report_occurrences.csv", occurrence_rows, occurrence_fields)
    parsed_fields = [
        "experiment_id", "namespace", "local_id", "canonical_name", "family_inferred", "date_reported",
        "hypothesis_reported", "change_reported", "facts_reported", "interpretation_reported", "config_reported",
        "cv_candidate", "delta_candidate", "lb_candidate", "runtime_candidate", "verdict_reported", "source_ref",
        "source_path", "clean_evidence_path", "sha256", "evidence_tier",
    ]
    write_csv(DEST / "registry" / "report_catalog.csv", parsed_rows, parsed_fields)
    with (DEST / "registry" / "report_catalog.jsonl").open("w", encoding="utf-8") as handle:
        for row in parsed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Preserve direct machine manifests from the worktree and the distinct Strategy-2 branch.
    manifest_dir = DEST / "evidence" / "machine_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for source_name in ("log.csv", "submissions.csv"):
        source_path = SOURCE / "experiments" / source_name
        if source_path.exists():
            (manifest_dir / f"team_a_current__{source_name}").write_bytes(source_path.read_bytes())
    for source_name in ("log.csv", "submissions.csv"):
        raw = git("show", f"team-a-strategy-2-impl:experiments/{source_name}", check=False)
        if raw:
            (manifest_dir / f"team_a_s2__{source_name}").write_bytes(raw)

    sha_counts: dict[str, int] = {}
    for row in occurrence_rows:
        sha_counts[row["sha256"]] = sha_counts.get(row["sha256"], 0) + 1
    duplicate_contents = sum(1 for count in sha_counts.values() if count > 1)
    print(json.dumps({
        "git_refs": len(ref_rows),
        "exported_reports": len(occurrence_rows),
        "unique_report_contents": len(sha_counts),
        "duplicate_content_groups": duplicate_contents,
        "namespaces": {name: sum(1 for row in occurrence_rows if row["namespace"] == name) for name in sorted({r["namespace"] for r in occurrence_rows})},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
