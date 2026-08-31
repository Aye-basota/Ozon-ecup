"""Materialize the complete hackathon experiment catalogue.

This is intentionally a repository-maintenance tool, not an experiment itself.
It inventories the current worktree, named git lineages and the teammate bundle,
then creates one self-describing folder per recovered experiment under
``experiments/repro``.  Historical implementation files are read directly from
git objects, so running this script does not require checking out old branches.

Run:
    python tools/materialize_experiment_catalog.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUT = EXPERIMENTS / "repro"
SNAPSHOTS = OUTPUT / "_snapshots"


@dataclass
class Entry:
    catalog_id: str
    namespace: str
    experiment_id: str
    title: str
    kind: str
    original_path: str
    source_ref: str
    source_commit: str
    source_files: list[str]
    commands: list[str]
    model: str
    features: str
    validation: str
    score: str
    submission: str
    seed: str
    preprocessing: str
    postprocessing: str
    external_data: str
    reproducibility: str
    notes: str
    preserved_document: str


def git(*args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, text=True,
        encoding="utf-8", errors="replace", capture_output=True,
    )
    if check and process.returncode:
        raise RuntimeError(process.stderr.strip() or "git command failed")
    return process.stdout


def safe_slug(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^a-z0-9а-я]+", "_", value, flags=re.IGNORECASE)
    return value.strip("_")[:110] or "unnamed"


def first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return fallback


def extract_commands(text: str) -> list[str]:
    commands: list[str] = []
    in_fence = False
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence and current:
                command = "\n".join(current).strip()
                if re.search(r"(^|\n)\s*(python|python3|pytest|bash|sh|PYTHONPATH=|LGB_THREADS=)", command):
                    commands.append(command)
            current = []
            in_fence = not in_fence
            continue
        if in_fence:
            current.append(line)
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if re.match(r"^(python|python3|pytest|bash|sh)\s+", stripped):
            commands.append(stripped)
    unique: list[str] = []
    for command in commands:
        if command not in unique:
            unique.append(command)
    return unique[:20]


def keyword_summary(text: str, mapping: list[tuple[str, str]], default: str) -> str:
    lowered = text.lower()
    values = [label for keyword, label in mapping if keyword in lowered]
    return ", ".join(dict.fromkeys(values)) if values else default


MODEL_KEYWORDS = [
    ("lightgbm", "LightGBM"), ("catboost", "CatBoost"),
    ("xgboost", "XGBoost"), ("transformer", "event Transformer / ETX"),
    ("tcn", "dilated TCN"), ("seq-", "sequence model"),
    ("bg/nbd", "BG/NBD"), ("btyd", "BTYD"), ("ridge", "Ridge"),
    ("hurdle", "two-part / hurdle"), ("distribution head", "distribution head"),
    ("голова распределения", "distribution head"), ("ensemble", "ensemble"),
    ("blend", "blend"), ("калибр", "calibration diagnostic"),
]
FEATURE_KEYWORDS = [
    ("personal time", "personal-time features"), ("личного времени", "personal-time features"),
    ("holiday", "holiday/YoY features"), ("calendar", "calendar features"),
    ("календар", "calendar features"), ("recency", "recency"),
    ("fresh", "freshness/conditional features"), ("funnel", "funnel features"),
    ("ворон", "funnel features"), ("occurrence", "occurrence features"),
    ("search/catalog", "Search/Catalog decomposition"), ("shapley", "channel Shapley"),
    ("gap", "gap/burst features"), ("depth", "history-depth features"),
    ("fingerprint", "dataset/user fingerprint"), ("ewm", "EWM aggregates"),
    ("агрегат", "window aggregates"), ("227", "227 tabular features"),
]


def metadata_from_text(text: str) -> dict[str, str]:
    model = keyword_summary(text, MODEL_KEYWORDS, "Unknown / not recoverable from repository history")
    features = keyword_summary(text, FEATURE_KEYWORDS, "See preserved experiment card and implementation")
    validation = "Unknown / not recoverable from repository history"
    score = "Unknown / not recoverable from repository history"
    submission = "None documented"
    seed = "Seed from src/config.py unless the preserved card explicitly states otherwise"
    preprocessing = "See preserved experiment card and frozen implementation"
    postprocessing = "None documented"
    external_data = "Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card"
    for line in text.splitlines():
        stripped = line.strip(" -*\t")
        lower = stripped.lower()
        if validation.startswith("Unknown") and any(token in lower for token in ("validation", "валидац", "фолд", "wcv", "cv mean")):
            validation = stripped[:500]
        if score.startswith("Unknown") and any(token in lower for token in ("lb ", "leaderboard", "wcv", "cv mean", "rmsle")) and re.search(r"\d[.,]\d", stripped):
            score = stripped[:500]
        if any(token in lower for token in ("submission_", "сабмит", "submission:")):
            submission = stripped[:500]
        if "seed" in lower or "сид" in lower:
            seed = stripped[:500]
        if any(token in lower for token in ("postprocess", "постпроцесс", "log-space", "лог-простран", "level ", "уров")):
            postprocessing = stripped[:500]
    return {
        "model": model, "features": features, "validation": validation,
        "score": score, "submission": submission, "seed": seed,
        "preprocessing": preprocessing, "postprocessing": postprocessing,
        "external_data": external_data,
    }


def source_candidates_for_current(path: Path, text: str) -> list[str]:
    stem = path.stem.lower()
    mapping = {
        "001": ["src/train.py", "src/models.py"],
        "002": ["src/train.py", "src/models.py"],
        "003": ["src/train.py", "src/models.py"],
        "004": ["src/train.py", "src/models.py"],
        "005": ["src/train.py", "src/models.py"],
        "006": ["src/blend.py", "src/predict.py", "src/submit.py"],
        "007": ["src/final_experiments.py"], "008": ["src/final_experiments.py"],
        "013": ["src/models.py", "src/train.py"],
        "014": ["src/models.py", "src/train.py", "src/predict.py"],
        "015": ["src/predict.py"], "016": ["src/report.py", "src/cv_lb.py"],
        "017": ["src/tune.py"], "018": ["src/seedavg.py"],
        "019": ["src/gapval.py"], "020": ["src/sampleval.py"],
        "021": ["src/ptime.py", "src/ptime_eval.py"],
        "022": ["src/densityval.py"], "023": ["src/holiday_yoy.py"],
        "024": ["src/mhz.py"], "025": ["src/seq.py"],
        "026": ["src/seq.py", "src/seedavg.py"],
        "027": ["src/seq.py", "research/strategies/results/SEQ3/stress.py", "research/strategies/results/SEQ4/crossdepth.py"],
        "028": ["src/dist_pact.py", "src/train.py"], "029": ["src/seq.py"],
        "030": ["src/seq.py"], "032": ["src/seq_cond.py", "src/seq.py"],
        "035": ["research/strategies/results/MIX9/lofo_mix.py", "research/strategies/results/MIX9/verify_submission.py"],
        "036": ["src/etx.py", "research/strategies/results/ETX1/lofo_etx.py"],
        "037": ["src/etx.py", "research/strategies/results/ETX2/depth_fix.py", "research/strategies/results/ETX2/lofo2.py"],
        "038": ["src/fnl.py"], "039": ["src/block4_saf.py"],
        "040": ["src/fresh_contrast.py"], "041": ["src/ridge15.py"],
        "042": ["src/zero2d_shrink.py"], "043": ["src/det_pair.py"],
        "044": ["src/fresh_cond_ft.py"], "045": ["src/buyctrl_det.py"],
        "046": ["src/tabular_backbone_refresh.py"], "047": ["src/btyd_day_bgnbd.py"],
        "048": ["src/selection_mismatch_cv.py"], "049": ["src/selection_mismatch_followup.py"],
        "050": ["src/btyd05_production.py"], "051": ["src/btyd_exp051.py", "src/btyd_stable_fit.py"],
        "052": ["src/channel_shapley_split.py"], "053": ["src/residual_signal_discovery.py"],
        "054": ["src/burst_gap_etx.py"], "055": ["src/landmark_memory_etx.py"],
        "056": ["src/late_unlabeled_etx.py"], "057": ["src/production_state_reweight.py"],
        "058": ["src/dataset_fingerprint.py"],
        "059": ["research/strategies/results/SEQ65_TEMPORAL_HEAVY/build_submission.py"],
        "060": ["src/submit.py"], "061": ["src/open_funnel.py"],
        "062": ["src/platform_detrend.py"], "063": ["src/occurrence_revisit.py"],
        "064": ["src/event_order.py"], "065": ["src/final_integration.py"],
        "066": ["src/authoritative_latest_audit.py"], "067": ["src/authoritative_latest_audit.py"],
        "068": ["src/recency_ridge_predictions.py"], "069": ["src/team_b_b2_ensemble.py"],
        "070": ["src/final_team_b_ensemble.py"], "071": ["src/final_team_b_ensemble.py"],
    }
    sources = ["src/config.py", "src/data.py", "src/features.py", "src/validation.py"]
    match = re.search(r"exp_(\d{3})", stem)
    if match:
        sources.extend(mapping.get(match.group(1), ["src/train.py", "src/predict.py", "src/models.py"]))
    else:
        sources.extend(["src/train.py", "src/predict.py", "src/models.py"])
    for candidate in re.findall(r"(?:src|research)/[A-Za-z0-9_./-]+\.py", text):
        sources.append(candidate)
    return [item for item in dict.fromkeys(sources) if (ROOT / item).exists()]


def historical_card_paths(ref: str, filters: Iterable[str] | None = None) -> list[str]:
    paths = [line.strip() for line in git("ls-tree", "-r", "--name-only", ref).splitlines()]
    cards = [path for path in paths if re.fullmatch(r"experiments/exp_[^/]+\.md", path, re.IGNORECASE)]
    if filters is not None:
        wanted = tuple(filters)
        cards = [path for path in cards if any(token in Path(path).name for token in wanted)]
    return sorted(cards)


def introduction_commit(ref: str, path: str) -> str:
    commits = git("log", ref, "--diff-filter=A", "--format=%H", "--", path).splitlines()
    return commits[0] if commits else git("rev-parse", ref).strip()


def snapshot_sources(commit: str) -> list[str]:
    files = [line for line in git("ls-tree", "-r", "--name-only", commit, "src").splitlines() if line.endswith(".py")]
    return sorted(files)


def write_historical_implementation(
    destination: Path, commit: str, source_ref: str, source_files: list[str],
) -> list[str]:
    provenance: dict[str, dict[str, str]] = {}
    fallbacks: list[str] = []
    for relative in source_files:
        content = git("show", f"{commit}:{relative}", check=False)
        if not content:
            continue
        selected = commit
        if relative.endswith(".py"):
            try:
                compile(content, relative, "exec")
            except SyntaxError:
                fallback = git("show", f"{source_ref}:{relative}", check=False)
                try:
                    if not fallback:
                        raise ValueError("fallback source is absent")
                    compile(fallback, relative, "exec")
                except (SyntaxError, ValueError):
                    pass
                else:
                    content = fallback
                    selected = source_ref
                    fallbacks.append(relative)
        output = destination / "implementation" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        provenance[relative] = {"card_introduction_commit": commit, "implementation_source": selected}
    (destination / "implementation" / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return fallbacks


def write_current_implementation(destination: Path, source_files: list[str]) -> None:
    for relative in source_files:
        source = ROOT / relative
        if source.is_file():
            relative_path = Path(relative)
            if relative_path.parts and relative_path.parts[0] == "пайплайн сокомандника":
                relative_path = Path("external_source") / relative_path.name
            output = destination / "implementation" / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)


RUNNER = '''"""Launcher for this archived experiment.

By default this prints provenance and recovered commands.  Pass ``--execute N``
to run command N from experiment.json, or append a replacement command after
``--command`` when the historical card did not preserve an executable command.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.repro.runner import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
'''


def build_entry(
    *, namespace: str, experiment_id: str, original_path: str, source_ref: str,
    source_commit: str, content: str, kind: str, source_files: list[str],
    reproducibility: str, notes: str = "",
) -> Entry:
    catalog_id = f"{safe_slug(namespace)}__{safe_slug(experiment_id)}"
    meta = metadata_from_text(content)
    return Entry(
        catalog_id=catalog_id, namespace=namespace, experiment_id=experiment_id,
        title=first_heading(content, experiment_id), kind=kind,
        original_path=original_path, source_ref=source_ref,
        source_commit=source_commit, source_files=source_files,
        commands=extract_commands(content), reproducibility=reproducibility,
        notes=notes, preserved_document="README.md", **meta,
    )


def current_card_entries() -> list[tuple[Entry, str, bool]]:
    entries: list[tuple[Entry, str, bool]] = []
    for path in sorted(EXPERIMENTS.glob("exp_*.md"), key=lambda item: item.name.lower()):
        content = path.read_text(encoding="utf-8")
        entry = build_entry(
            namespace="team_a_current", experiment_id=path.stem,
            original_path=str(path.relative_to(ROOT)).replace("\\", "/"),
            source_ref="working tree", source_commit=git("rev-parse", "HEAD").strip(),
            content=content, kind="experiment card",
            source_files=source_candidates_for_current(path, content),
            reproducibility="FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card",
        )
        entries.append((entry, content, False))
    duplicate = EXPERIMENTS / "EXP_032_S04_conditional_fresh_seq.md"
    if duplicate.exists():
        content = duplicate.read_text(encoding="utf-8")
        entry = build_entry(
            namespace="team_a_current", experiment_id=duplicate.stem,
            original_path=str(duplicate.relative_to(ROOT)).replace("\\", "/"),
            source_ref="working tree", source_commit=git("rev-parse", "HEAD").strip(),
            content=content, kind="experiment design/card",
            source_files=source_candidates_for_current(duplicate, content),
            reproducibility="FULL when required data/artifacts are present",
            notes="Alternative preserved card for EXP-032; kept because it contains additional implementation detail.",
        )
        entries.append((entry, content, False))
    return entries


def log_only_entries(existing: set[str]) -> list[tuple[Entry, str, bool]]:
    path = EXPERIMENTS / "log.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    entries: list[tuple[Entry, str, bool]] = []
    for row in rows:
        exp_id = (row.get("exp_id") or row.get("id") or "").strip()
        if not exp_id:
            continue
        catalog_id = f"team_a_run__{safe_slug(exp_id)}"
        if catalog_id in existing:
            continue
        body = [f"# Logged run — {exp_id}", "", "This run was recovered from `experiments/log.csv`.", ""]
        for key, value in row.items():
            body.append(f"- **{key}:** {value or 'Unknown / not recoverable from repository history'}")
        content = "\n".join(body) + "\n"
        entry = build_entry(
            namespace="team_a_run", experiment_id=exp_id,
            original_path="experiments/log.csv", source_ref="working tree",
            source_commit=git("rev-parse", "HEAD").strip(), content=content,
            kind="logged run / arm", source_files=["src/train.py", "src/predict.py", "src/features.py", "src/models.py"],
            reproducibility="PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored",
            notes="Run-level entry; a numbered experiment card may document the wider experiment family.",
        )
        entries.append((entry, content, False))
    return entries


def historical_entries() -> list[tuple[Entry, str, bool]]:
    specs = [
        ("team_b_b2", "88dc69163b1f39aaac55ddfbfc9986e2203cfbdf", None),
        ("team_b_strategy", "824f41575bc2fa4ae11b8f6f9dfd907571276d37", None),
        ("strategy_2", "3c1d86d836c7b73519abe99f94686431852187cc", ("s2_",)),
        ("renewal_branch", "bbae4b0c7a14f3aa42aedff05d8d02c2c8fffdba", ("next_purchase_clock",)),
        ("domain_branch", "c219dabe316982bb9de23e15c4f24d3e9eb16ae0", ("domain_shift",)),
        ("calendar_branch", "96271398780a33b5423a9971128d4bd946051f0f", ("calendar_placebo",)),
        ("global_regime_branch", "4003b6874f397fe48577b26118ae1d560a703419", ("global_regime_occ",)),
    ]
    result: list[tuple[Entry, str, bool]] = []
    for namespace, ref, filters in specs:
        for path in historical_card_paths(ref, filters):
            content = git("show", f"{ref}:{path}")
            commit = introduction_commit(ref, path)
            sources = snapshot_sources(commit)
            entry = build_entry(
                namespace=namespace, experiment_id=Path(path).stem,
                original_path=f"git:{ref[:12]}:{path}", source_ref=ref,
                source_commit=commit, content=content, kind="git-history experiment card",
                source_files=sources,
                reproducibility=(
                    "FULL from the frozen commit when the card contains a command; "
                    "PARTIAL when the card explicitly says the experimental code was rolled back"
                ),
                notes="Frozen implementation is copied from the commit that introduced this card.",
            )
            result.append((entry, content, True))
    return result


def historical_table_only_entries() -> list[tuple[Entry, str, bool]]:
    """Recover experiment-index rows that never received a standalone card."""
    specs = [
        ("team_b_b2", "88dc69163b1f39aaac55ddfbfc9986e2203cfbdf"),
        ("team_b_strategy", "824f41575bc2fa4ae11b8f6f9dfd907571276d37"),
    ]
    result: list[tuple[Entry, str, bool]] = []
    for namespace, ref in specs:
        index_text = git("show", f"{ref}:experiments.md", check=False)
        card_names = [Path(path).stem for path in historical_card_paths(ref)]
        for line in index_text.splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("exp_") or "|" not in stripped:
                continue
            columns = [column.strip() for column in stripped.split("|")]
            exp_id = columns[0]
            if any(name == exp_id or name.startswith(exp_id + "_") for name in card_names):
                continue
            content = "\n".join([
                f"# Historical index-only run — {exp_id}", "",
                "No standalone experiment card survived. The exact index row is preserved below.", "",
                "| Field | Recovered value |", "|---|---|",
            ] + [f"| column_{index + 1} | {value} |" for index, value in enumerate(columns)]) + "\n"
            entry = build_entry(
                namespace=namespace, experiment_id=exp_id,
                original_path=f"git:{ref[:12]}:experiments.md", source_ref=ref,
                source_commit=ref, content=content, kind="git-history index-only run",
                source_files=snapshot_sources(ref),
                reproducibility="PARTIAL: metrics and nearest source snapshot survive; exact rolled-back code is not recoverable",
                notes="Recovered because this run existed only as a row in experiments.md.",
            )
            result.append((entry, content, True))
    return result


def teammate_candidate_entries() -> list[tuple[Entry, str, bool]]:
    """Split aggregate teammate validation tables into one folder per candidate."""
    bundle = ROOT / "пайплайн сокомандника" / "review_bundles"
    evidence: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for path in sorted(bundle.glob("**/results/*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error):
            continue
        for row in rows:
            name = (row.get("name") or row.get("best_name") or "").strip()
            if not name:
                continue
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            evidence.setdefault(name, []).append((rel, row))

    runners = [
        "пайплайн сокомандника/research_scripts/continue_fixedstack_combo_10h.py",
        "пайплайн сокомандника/research_scripts/continue_best_bas_final6h.py",
        "пайплайн сокомандника/research_scripts/materialize_final6h_extra90m.py",
        "пайплайн сокомандника/research_scripts/run_best_bas_fixedstack_14h_v2.py",
    ]
    result: list[tuple[Entry, str, bool]] = []
    for name, rows in sorted(evidence.items()):
        documents = [f"# Teammate candidate — {name}", "",
                     "This candidate was recovered from completed review-bundle result tables.", ""]
        for index, (relative, row) in enumerate(rows, start=1):
            documents.extend([f"## Evidence row {index}", "", f"Source: `{relative}`", "",
                              "| Field | Value |", "|---|---|"])
            documents.extend(f"| {key} | {str(value).replace('|', '&#124;')} |" for key, value in row.items())
            documents.append("")
        content = "\n".join(documents) + "\n"
        sources = [path for path in runners if (ROOT / path).exists()]
        entry = build_entry(
            namespace="teammate_candidate", experiment_id=name,
            original_path=" | ".join(relative for relative, _ in rows),
            source_ref="external teammate review bundles", source_commit="NOT_IN_GIT_HISTORY",
            content=content, kind="completed teammate candidate/subrun",
            source_files=sources,
            reproducibility="PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external",
            notes=f"Recovered from {len(rows)} result-table row(s); no score or parameter was inferred.",
        )
        result.append((entry, content, False))
    return result


def new_direction_entries() -> list[tuple[Entry, str, bool]]:
    """Materialize every late research/new_directions package as one experiment.

    These packages landed on ``team-a`` after the first history pass.  Treat the
    directory as the unit of provenance: reports, launchers and frozen helper
    modules from one directory belong to the same research decision.
    """
    base = ROOT / "research" / "new_directions"
    if not base.exists():
        return []
    result: list[tuple[Entry, str, bool]] = []
    code_suffixes = {".py", ".sh", ".ps1"}
    for directory in sorted((path for path in base.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        reports = [
            path for path in files
            if path.suffix.lower() == ".md" and (
                "report" in path.name.lower()
                or path.name.lower() in {"analysis.md", "selected_experiment.md"}
            )
        ]
        if not reports:
            reports = [path for path in files if path.suffix.lower() == ".md"]
        document_parts = [path.read_text(encoding="utf-8", errors="replace") for path in reports]
        content = "\n\n---\n\n".join(document_parts)
        if not content:
            content = (
                f"# {directory.name}\n\n"
                "No narrative report survived in this package. See the frozen implementation files.\n"
            )
        source_paths = [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in files
            if path.suffix.lower() in code_suffixes or path in reports
        ]
        code_count = sum(path.suffix.lower() in code_suffixes for path in files)
        rel = str(directory.relative_to(ROOT)).replace("\\", "/")
        reproducibility = (
            "FULL when the data/frozen artifacts named by the report are present"
            if code_count
            else "PARTIAL: the report survives, but no experiment launcher was recoverable from this package"
        )
        entry = build_entry(
            namespace="new_direction", experiment_id=directory.name,
            original_path=rel, source_ref="origin/team-a late research package",
            source_commit=git("rev-parse", "HEAD").strip(), content=content,
            kind="late research direction / experiment package", source_files=source_paths,
            reproducibility=reproducibility,
            notes=(
                f"Directory-level audit unit: {len(files)} files, {code_count} launcher/helper scripts, "
                f"{len(reports)} preserved report documents. Numeric claims are copied from those reports."
            ),
        )
        result.append((entry, content, False))
    return result


def reconstructed_anniversary_entry() -> list[tuple[Entry, str, bool]]:
    """Recover the sole primary report missing from the normal source tree."""
    registry = ROOT / "research" / "reconstruction" / "registry" / "report_catalog.csv"
    if not registry.exists():
        return []
    with registry.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next(
        (item for item in rows if item.get("experiment_id") == "independent_anniversary:exp_058"),
        None,
    )
    if row is None:
        return []
    body = [
        "# Reconstructed primary report — exp_058 EXACT-ANNIVERSARY-WINDOW", "",
        "The original Markdown card is absent from the merged tree. The independent reconstruction "
        "preserved its normalized primary-report row verbatim; every field is reproduced below.", "",
        "| Registry field | Preserved value |", "|---|---|",
    ]
    for key, value in row.items():
        escaped = (value or "Unknown / not recoverable from repository history").replace("|", "&#124;").replace("\n", "<br>")
        body.append(f"| {key} | {escaped} |")
    content = "\n".join(body) + "\n"
    source_files = [
        "research/reconstruction/evidence/worktree_artifacts/independent_anniversary/src/exact_anniversary.py",
        "research/reconstruction/evidence/worktree_artifacts/independent_anniversary/src/test_exact_anniversary.py",
    ]
    entry = build_entry(
        namespace="independent_anniversary", experiment_id="exp_058_exact_anniversary",
        original_path=row.get("source_path") or "experiments/exp_058_exact_anniversary.md",
        source_ref=row.get("source_ref") or "reconstructed linked worktree",
        source_commit="PRIMARY_CARD_NOT_IN_MERGED_GIT; normalized row SHA256=" + (row.get("sha256") or "UNKNOWN"),
        content=content, kind="reconstructed primary experiment report", source_files=source_files,
        reproducibility=(
            "PARTIAL: normalized report fields, implementation and tests survive; "
            "the original Markdown bytes and ignored input artifacts do not"
        ),
        notes="Rejected experiment; no submission was created. No missing field was inferred.",
    )
    return [(entry, content, False)]


def packaged_final_entries() -> list[tuple[Entry, str, bool]]:
    """Create explicit catalogue units for late submission and blend pipelines."""
    specs = [
        (
            "submit_joint86_teamb14",
            "reproducibility/SUBMIT_JOINT86_TEAMB14/README.md",
            [
                "reproducibility/SUBMIT_JOINT86_TEAMB14/build_submit.py",
                "reproducibility/SUBMIT_JOINT86_TEAMB14/create_manifest.py",
                "scripts/reproduce_final.py",
                "scripts/build_optimized_pair_blends.py",
                "research/OPTIMIZED_PAIR_BLENDS.json",
            ],
            "packaged final submission; exact outer blend, frozen upstream anchor",
            "FULL from frozen inputs; raw-to-JOINT_V2 remains explicitly PROVENANCE_INCOMPLETE",
        ),
        (
            "submit_strongest55_teamb45",
            "reproducibility/SUBMIT_STRONGEST55_TEAMB45/README.md",
            [
                "reproducibility/SUBMIT_STRONGEST55_TEAMB45/build_submit.py",
                "reproducibility/SUBMIT_STRONGEST55_TEAMB45/verify.py",
                "scripts/reproduce_final.py",
                "scripts/build_optimized_pair_blends.py",
                "research/OPTIMIZED_PAIR_BLENDS.json",
            ],
            "packaged final candidate; exact outer blend, frozen upstream inputs",
            "FULL from frozen inputs; raw retraining has the limitations documented by the package",
        ),
        (
            "strongest80_teamb20",
            "research/STRONGEST80_TEAMB20.json",
            ["scripts/build_strongest80_teamb20.py", "research/STRONGEST80_TEAMB20.json"],
            "late pair-blend candidate",
            "FULL when the two named source submissions are present",
        ),
        (
            "optimized_pair_blends",
            "research/OPTIMIZED_PAIR_BLENDS.json",
            ["scripts/build_optimized_pair_blends.py", "research/OPTIMIZED_PAIR_BLENDS.json"],
            "pair-blend search and final-candidate generator",
            "FULL when the named source submissions are present",
        ),
        (
            "final_threeway_ensemble",
            "research/FINAL_THREEWAY_ENSEMBLE.json",
            ["scripts/build_final_threeway_ensemble.py", "research/FINAL_THREEWAY_ENSEMBLE.json"],
            "three-way final-candidate ensemble",
            "FULL when the three named source submissions are present",
        ),
        (
            "submit_orth_final",
            "research/SUBMIT_ORTH_FINAL_reasoning.md",
            [
                "research/SUBMIT_ORTH_FINAL_reasoning.md",
                "research/new_directions/EXP_ORTH_ROBUST_H12_INTERP/run_orth_h12_interp.py",
            ],
            "ORTH final submission lineage",
            "PARTIAL: reasoning and a later interpolation launcher survive; the exact ORTH_FINAL generator is absent",
        ),
        (
            "submission_geometry",
            "docs/EXPERIMENT_HISTORY.md",
            ["docs/EXPERIMENT_HISTORY.md"],
            "submission-geometry research lineage",
            "PARTIAL: scores and history survive, but the external geometry workspace scripts are not in this repository",
        ),
    ]
    result: list[tuple[Entry, str, bool]] = []
    for experiment_id, report_name, sources, kind, reproducibility in specs:
        report = ROOT / report_name
        if not report.exists():
            continue
        raw = report.read_text(encoding="utf-8", errors="replace")
        if report.suffix.lower() == ".json":
            try:
                raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
            content = f"# {experiment_id}\n\n```json\n{raw}\n```\n"
        else:
            content = raw
        existing_sources = [path for path in sources if (ROOT / path).is_file()]
        entry = build_entry(
            namespace="packaged_final", experiment_id=experiment_id,
            original_path=report_name, source_ref="origin/team-a final/research package",
            source_commit=git("rev-parse", "HEAD").strip(), content=content,
            kind=kind, source_files=existing_sources, reproducibility=reproducibility,
            notes="Reported leaderboard results and forecasts are kept distinct exactly as in the preserved source.",
        )
        result.append((entry, content, False))
    return result


def script_entries() -> list[tuple[Entry, str, bool]]:
    groups: list[tuple[str, Path, str]] = []
    for path in sorted((ROOT / "research" / "eda").glob("e*.py")):
        groups.append(("eda", path, "EDA experiment/script"))
    teammate = ROOT / "пайплайн сокомандника"
    for path in sorted((teammate / "research_scripts").glob("*.py")):
        groups.append(("teammate_research", path, "teammate research runner"))
    for path in sorted((teammate / "review_bundles").glob("**/RUN_MANIFEST.json")):
        groups.append(("teammate_review_bundle", path, "completed teammate review run"))
    manual = [
        ("teammate_final", teammate / "latest" / "rebuild_latest.py", "authoritative latest submission rebuild"),
        ("teammate_final", teammate / "friend_original" / "submission_STRONGEST_CURRENT" / "pipeline" / "build_submission.py", "STRONGEST_CURRENT rebuild"),
        ("team_b_final", ROOT / "src" / "team_b_b2_ensemble.py", "team-B B2 integration pipeline"),
        ("team_a_final", ROOT / "src" / "final_team_b_ensemble.py", "requested final partial-slot ensemble"),
    ]
    groups.extend(manual)
    entries: list[tuple[Entry, str, bool]] = []
    for namespace, path, kind in groups:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if path.suffix == ".json":
            try:
                pretty = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pretty = content
            document = f"# {path.parent.parent.name}\n\n```json\n{pretty}\n```\n"
        else:
            document = f"# {path.stem}\n\nOriginal script: `{rel}`\n\n```python\n{content}\n```\n"
        entry = build_entry(
            namespace=namespace, experiment_id=path.parent.parent.name if path.suffix == ".json" else path.stem,
            original_path=rel, source_ref="working tree", source_commit=git("rev-parse", "HEAD").strip(),
            content=content, kind=kind, source_files=[rel] if path.suffix == ".py" else [],
            reproducibility="FULL when the inputs/checkpoints named by the preserved runner are available",
        )
        entries.append((entry, document, False))
    return entries


def write_entry(entry: Entry, document: str, historical: bool) -> None:
    destination = OUTPUT / entry.catalog_id
    destination.mkdir(parents=True, exist_ok=True)
    implementation = destination / "implementation"
    if implementation.exists():
        if OUTPUT.resolve() not in implementation.resolve().parents:
            raise RuntimeError(f"refusing to clear unexpected path: {implementation}")
        shutil.rmtree(implementation)
    fallbacks: list[str] = []
    if historical:
        fallbacks = write_historical_implementation(
            destination, entry.source_commit, entry.source_ref, entry.source_files,
        )
    else:
        write_current_implementation(destination, entry.source_files)
    if fallbacks:
        fallback_note = (
            "The card-introduction commit contained syntactically incomplete WIP source; "
            f"the frozen implementation uses the nearest surviving branch-head version for: {', '.join(fallbacks)}. "
            "Exact provenance is in implementation/SOURCE_PROVENANCE.json."
        )
        entry.notes = f"{entry.notes} {fallback_note}".strip()
        entry.reproducibility = "PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback"
    header = [
        f"# {entry.title}", "",
        "## Catalogue metadata", "",
        f"- **Catalogue ID:** `{entry.catalog_id}`",
        f"- **Namespace:** `{entry.namespace}`",
        f"- **Experiment ID:** `{entry.experiment_id}`",
        f"- **Original source:** `{entry.original_path}`",
        f"- **Source ref:** `{entry.source_ref}`",
        f"- **Source commit:** `{entry.source_commit}`",
        f"- **Kind:** {entry.kind}",
        f"- **Model:** {entry.model}",
        f"- **Features:** {entry.features}",
        f"- **Preprocessing:** {entry.preprocessing}",
        f"- **Validation:** {entry.validation}",
        f"- **Known score:** {entry.score}",
        f"- **Seed:** {entry.seed}",
        f"- **Postprocessing:** {entry.postprocessing}",
        f"- **Submission:** {entry.submission}",
        f"- **External data/artifacts:** {entry.external_data}",
        f"- **Reproducibility:** {entry.reproducibility}",
    ]
    if entry.notes:
        header.append(f"- **Notes:** {entry.notes}")
    header.extend([
        "", "## Reproduction", "",
        "Run `python run.py` to inspect recovered commands and provenance. "
        "Use `python run.py --execute N` only after preparing the data/artifacts listed below.",
        "", "## Preserved original documentation", "",
    ])
    # The frozen implementation remains byte-faithful.  Normalize only the
    # generated Markdown envelope so exact source whitespace does not create
    # thousands of distracting `git diff --check` warnings.
    normalized_document = "\n".join(line.rstrip() for line in document.strip().splitlines())
    rendered = "\n".join(line.rstrip() for line in header).rstrip()
    if normalized_document:
        rendered += "\n" + normalized_document
    (destination / "README.md").write_text(rendered + "\n", encoding="utf-8")
    (destination / "experiment.json").write_text(
        json.dumps(asdict(entry), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (destination / "run.py").write_text(RUNNER, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    collected = current_card_entries()
    existing = {entry.catalog_id for entry, _, _ in collected}
    extras = log_only_entries(existing)
    collected.extend(extras)
    collected.extend(historical_entries())
    collected.extend(historical_table_only_entries())
    collected.extend(script_entries())
    collected.extend(teammate_candidate_entries())
    collected.extend(new_direction_entries())
    collected.extend(reconstructed_anniversary_entry())
    collected.extend(packaged_final_entries())

    unique: dict[str, tuple[Entry, str, bool]] = {}
    for item in collected:
        entry = item[0]
        if entry.catalog_id in unique:
            suffix = hashlib.sha1(entry.original_path.encode("utf-8")).hexdigest()[:8]
            entry.catalog_id = f"{entry.catalog_id}__{suffix}"
        unique[entry.catalog_id] = item

    entries = [item[0] for item in unique.values()]
    for entry, document, historical in unique.values():
        write_entry(entry, document, historical)

    manifest_json = OUTPUT / "catalog.json"
    manifest_csv = OUTPUT / "catalog.csv"
    manifest_json.write_text(
        json.dumps([asdict(entry) for entry in entries], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with manifest_csv.open("w", encoding="utf-8", newline="") as stream:
        fields = list(asdict(entries[0]).keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = asdict(entry)
            row["source_files"] = " | ".join(row["source_files"])
            row["commands"] = " | ".join(row["commands"])
            writer.writerow(row)

    print(f"materialized {len(entries)} catalogue entries in {OUTPUT}")


if __name__ == "__main__":
    main()
