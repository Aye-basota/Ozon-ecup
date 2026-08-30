"""Generate experiments/README.md from preserved experiment reports."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EARLY = ROOT / "experiments" / "team_a"
NEW = ROOT / "research" / "new_directions"


def clean(value: str, limit: int = 180) -> str:
    value = re.sub(r"[`*_#]", "", value)
    value = re.sub(r"\s+", " ", value).strip().replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return clean(line[2:], 120)
    return fallback


def section_line(text: str, names: tuple[str, ...]) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        normalized = clean(line).lower()
        if line.lstrip().startswith("#") and any(name in normalized for name in names):
            for candidate in lines[index + 1 : index + 12]:
                value = clean(candidate)
                if value and not candidate.lstrip().startswith("#"):
                    return value
    return ""


def matching_line(text: str, patterns: tuple[str, ...]) -> str:
    for line in text.splitlines():
        value = clean(line)
        lower = value.lower()
        if value and any(pattern in lower for pattern in patterns):
            return value
    return ""


def parse(path: Path, experiment_id: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    verdict = section_line(text, ("вердикт", "verdict", "итог")) or matching_line(
        text, ("verdict", "accept", "reject", "no_go", "blocked", "prepared")
    )
    metric = matching_line(text, ("wcv", "rmsle", "public lb", "delta", "δ"))
    result = section_line(text, ("результат", "results", "result", "итог")) or metric
    idea = section_line(text, ("гипотез", "цель", "idea", "goal", "scope"))
    if not idea:
        paragraphs = [clean(line) for line in text.splitlines() if clean(line) and not line.startswith("#")]
        idea = paragraphs[0] if paragraphs else "См. исходный отчёт."
    used = "YES" if (
        experiment_id.lower().startswith("exp_037")
        or experiment_id.startswith("EXP075")
        or experiment_id.startswith("EXP089")
        or experiment_id.startswith("EXP090")
    ) else "NO"
    return {
        "id": experiment_id,
        "name": title(text, path.stem),
        "idea": clean(idea),
        "result": clean(result or "См. отчёт."),
        "verdict": clean(verdict or "Не выделен отдельно; исходный verdict сохранён в отчёте."),
        "metric": clean(metric or "—"),
        "used": used,
        "location": path.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(EARLY.glob("exp_*.md"), key=lambda item: item.name.lower()):
        match = re.match(r"(?i)(exp_\d{3}[a-z]?)", path.name)
        if match:
            rows.append(parse(path, match.group(1)))
    for path in sorted(EARLY.glob("EXP_*.md"), key=lambda item: item.name.lower()):
        if not any(row["location"] == path.relative_to(ROOT).as_posix() for row in rows):
            match = re.match(r"(?i)(EXP_\d{3}[A-Z]?)", path.name)
            if match:
                rows.append(parse(path, match.group(1)))
    for directory in sorted(NEW.glob("EXP*"), key=lambda item: item.name.lower()):
        if not directory.is_dir():
            continue
        report = directory / "REPORT.md"
        if report.exists():
            rows.append(parse(report, directory.name))

    rows.extend(
        [
            {
                "id": "SUBMISSION_GEOMETRY",
                "name": "Submission geometry and shrinkage",
                "idea": "Reconstruct and shrink directions in the bank of scored submissions in log space.",
                "result": "Produced SUBMIT_v2_shrunk and SUBMIT_NEXT_BEST; later JOINT work extends this lineage.",
                "verdict": "ACCEPTED HISTORICAL LINEAGE; public-LB-fitted and not a canonical OOF experiment.",
                "metric": "Public LB 1.6467120 then 1.6466079.",
                "used": "YES",
                "location": "research/submission_geometry/submission_geometry/README.md",
            },
            {
                "id": "ORTH",
                "name": "Orthogonal public-geometry candidates",
                "idea": "Add controlled directions outside the incumbent scored span and audit public/private risk.",
                "result": "ORTH_ALPHA/ORTH_FINAL became audited ancestors of the later JOINT lineage.",
                "verdict": "HISTORICAL INPUT; upstream JOINT_V2 generation script remains missing.",
                "metric": "ORTH_ALPHA public LB 1.6461597403.",
                "used": "YES",
                "location": "research/provenance/downloads/SUBMIT_ORTH_FINAL_reasoning.md",
            },
        ]
    )

    header = """# Team-A experiment index

This index lists the experiment reports that were actually found. Original IDs,
results and verdicts are preserved; duplicate numeric IDs from independent
worktrees are disambiguated by their source directory rather than renumbered.
Binary artifacts remain external and are addressable through
`docs/TEAM_A_SOURCE_INVENTORY.csv`.

| ID | Name | Idea | Result | Verdict | Key metric | Used in final solution? | Location |
|---|---|---|---|---|---|---|---|
"""
    lines = [header]
    for row in rows:
        lines.append(
            "| {id} | {name} | {idea} | {result} | {verdict} | {metric} | {used} | [{location}](../{location}) |\n".format(
                **row
            )
        )
    output = ROOT / "experiments" / "README.md"
    output.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
