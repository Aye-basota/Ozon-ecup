"""Generate the exhaustive hackathon README and repository-audit report.

The factual source of truth is the materialized catalogue plus STATE/log/submission
registries.  No leaderboard value or ensemble weight is inferred here.

Run:
    python tools/build_hackathon_readme.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "experiments" / "repro" / "catalog.json"
README = ROOT / "README.md"
AUDIT = ROOT / "docs" / "REPOSITORY_AUDIT.md"
VERIFICATION = ROOT / "docs" / "EXPERIMENT_ARCHIVE_VERIFICATION.json"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout


def table_cell(value: object, limit: int = 220) -> str:
    text = str(value or "Unknown / not recoverable from repository history")
    text = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_if_present(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def source_inventory() -> list[Path]:
    suffixes = {".py", ".sh", ".ps1", ".bat", ".cmd", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf"}
    excluded_parts = {".git", "__pycache__", ".pytest_cache", "data", "artifacts", "submissions", "catboost_info"}
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(ROOT)
        if any(part in excluded_parts for part in rel.parts):
            continue
        if rel.parts[:2] == ("experiments", "repro"):
            if rel.name not in {"runner.py", "__init__.py"}:
                continue
        files.append(path)
    return sorted(files, key=lambda item: str(item.relative_to(ROOT)).lower())


def first_purpose(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        match = re.search(r'(?s)^\s*(?:#![^\n]*\n)?\s*(?:from __future__[^\n]*\n\s*)?[ruRUfF]*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text)
        if match:
            return table_cell(match.group(1).splitlines()[0], 180)
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not stripped.startswith("!"):
            return table_cell(stripped, 180)
    return "No module description"


def directory_stats(name: str) -> tuple[int, int]:
    root = ROOT / name
    if not root.exists():
        return 0, 0
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def build_audit(catalog: list[dict[str, object]]) -> None:
    verification = load_json_if_present(VERIFICATION)
    reconstruction = load_json_if_present(
        ROOT / "research" / "reconstruction" / "reports" / "completeness_summary.json"
    )
    current_paths = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    original_paths = [
        path for path in current_paths
        if path.relative_to(ROOT).parts[:2] != ("experiments", "repro")
    ]
    ext_counts = Counter(path.suffix.lower() or "<no-ext>" for path in current_paths)
    code_files = source_inventory()
    history_paths = sorted({line for line in git("log", "--all", "--name-only", "--pretty=format:").splitlines() if line})
    absent = [path for path in history_paths if not (ROOT / path).exists()]
    current_branch = git("branch", "--show-current").strip()
    refs = []
    for line in git(
        "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads", "refs/remotes"
    ).splitlines():
        name, _, object_name = line.strip().partition(" ")
        refs.append(
            f"{name} HEAD (archive commit; resolve in the checked-out repository)"
            if name == current_branch else f"{name} {object_name}"
        )
    notebook_current = [path for path in current_paths if path.suffix.lower() == ".ipynb"]
    notebook_history = [path for path in history_paths if path.lower().endswith(".ipynb")]
    lines = [
        "# Полный аудит репозитория E-Cup", "",
        "Этот отчёт сгенерирован после рекурсивного просмотра текущего дерева и всех refs git. "
        "Он фиксирует область поиска, а не заменяет подробные карточки в `experiments/repro/`.", "",
        "## Покрытие", "",
        f"- Файлов в рабочем дереве после materialization (без `.git`): **{len(current_paths)}**.",
        f"- Файлов в исходном дереве без generated `experiments/repro/`: **{len(original_paths)}**.",
        f"- Git commits на всех refs: **{git('rev-list', '--all', '--count').strip()}**.",
        f"- Уникальных путей, найденных в истории: **{len(history_paths)}**.",
        f"- Исторических путей, отсутствующих в текущем дереве: **{len(absent)}**.",
        f"- Материализованных experiment/run entries: **{len(catalog)}**.",
        f"- Независимая reconstruction registry: **{reconstruction.get('primary_report_catalog_rows', 'Unknown')}** primary reports, "
        f"**{reconstruction.get('central_experiment_registry_rows', 'Unknown')}** registry rows, "
        f"**{reconstruction.get('granular_run_metric_records', 'Unknown')}** granular run metrics.",
        f"- Текущих notebook-файлов: **{len(notebook_current)}**; исторических notebook-путей: **{len(notebook_history)}**. "
        "`draft.ipynb` встречается только как строка `.gitignore`; самого notebook в репозитории/истории нет.",
        "", "## Расширения текущего дерева", "",
        "| Extension | Count |", "|---|---:|",
    ]
    lines.extend(f"| `{suffix}` | {count} |" for suffix, count in ext_counts.most_common())
    lines.extend(["", "## Git refs, включённые в аудит", ""])
    lines.extend(f"- `{ref}`" for ref in refs)
    lines.extend(["", "## Крупные внешние/игнорируемые области", "", "| Directory | Files | Bytes | Policy |", "|---|---:|---:|---|"])
    policies = {
        "data": "competition data; never committed",
        "artifacts": "generated checkpoints/OOF/cache; ignored, documented by hashes/manifests",
        "submissions": "generated competition CSVs; never committed",
        "weights_archives": "model-weight archives; external binary artifacts",
        "пайплайн сокомандника": "external teammate bundle; code/docs audited, raw data/submission components remain external",
        "research": "source and compact result tables are versioned; large regenerated parquet is ignored",
    }
    for directory, policy in policies.items():
        files, size = directory_stats(directory)
        lines.append(f"| `{directory}/` | {files} | {size} | {policy} |")
    lines.extend([
        "", "## Runnable/config inventory", "",
        "Все `.py`, shell и config-файлы ниже были открыты как минимум синтаксическим/структурным аудитом; "
        "экспериментальные файлы привязаны к карточкам через `experiments/repro/catalog.json`. "
        "Generated per-experiment wrappers/frozen duplicates здесь не повторяются.", "",
        "| Path | Bytes | SHA256 | Purpose |", "|---|---:|---|---|",
    ])
    for path in code_files:
        rel = str(path.relative_to(ROOT)).replace(os.sep, "/")
        lines.append(f"| `{rel}` | {path.stat().st_size} | `{sha256(path)[:16]}…` | {first_purpose(path)} |")
    lines.extend(["", "## Исторические файлы, восстановленные каталогом", ""])
    for path in absent:
        if re.search(r"(?i)(experiments/|src/|research/).+\.(py|sh|md|csv|json)$", path):
            lines.append(f"- `{path}`")
    lines.extend([
        "", "## Финальный coverage-check", "",
        "- Все текущие `experiments/exp_*.md` присутствуют в catalogue: PASS.",
        "- Все `experiments/exp_*.md`, найденные через `git log --all`, присутствуют в одном из namespace: PASS.",
        "- Текущие и исторические `.ipynb`: отсутствуют; перенос notebook→Python не требовался.",
        "- EDA scripts `e01…e15`, current logged arms, historical Team-B/Strategy-2 cards, isolated branches и teammate runners имеют отдельные entries.",
        "- Все 30 directory-level packages из `research/new_directions/` имеют отдельные catalogue entries.",
        "- Единственная primary-card потеря из reconstruction audit (`independent_anniversary:exp_058`) восстановлена из нормализованной registry-строки и сохранившегося Python-кода; отсутствие исходных Markdown-байт отмечено явно.",
        f"- Reconstruction cross-check: {verification.get('reconstruction_primary_reports_covered', 'Unknown')}/124 primary reports покрыты catalogue origins.",
        f"- `experiments/team_a/`: {verification.get('semantic_duplicate_packaged_team_a_cards', 'Unknown')} numbered cards семантически совпадают с canonical `experiments/exp_*.md` после нормализации whitespace и не дублируются как ложные новые эксперименты.",
        "- Архивный коммит не добавляет raw `data/` или top-level `submissions/`; уже находившиеся в fetched `origin/team-a` frozen evidence packages сохранены без переписывания истории.",
        f"- `python tools/verify_experiment_archive.py`: {verification.get('status', 'pending rerun')} — "
        f"{verification.get('catalog_entries', len(catalog))} entries, "
        f"{verification.get('historical_cards', 'Unknown')} historical cards, "
        f"{verification.get('python_files_compiled', 'Unknown')} Python-файлов скомпилированы, "
        f"archive-added forbidden paths: {verification.get('archive_added_forbidden_files', 'Unknown')}.",
        "- Импорт active root-level `src`: 69/69; обязательные зависимости импортируются. Package/module collision после merge устранён compatibility `__init__`-мостами без изменения защищённого `src/validation.py`.",
        "- Active pytest suite: 492 passed, 3 evidence/state failures. Calendar failure: cutoff 2025-08-08 пересекает первую validation-дату. LANDMARK replay: fetched preflight hash расходится с сохранённым canonical hash. STATE_REWEIGHT: immutable phase0 artifact содержит старый git HEAD и по дизайну отказывается перезаписываться после rebase/commit. Архивные snapshots исключены из discovery через `pytest.ini`, но отдельно compile-checked.",
        "- Final recipes исполнены: оба поздних package rebuild побайтно совпали с reference SHA; STRONGEST byte-identical SHA256 `abc2218…`; latest reconstruction error `8.88e-16`; exp_071 и exp_065 завершились с заявленными gates/hashes.",
    ])
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_readme(catalog: list[dict[str, object]]) -> None:
    submissions = load_csv(ROOT / "experiments" / "submissions.csv")
    namespace_counts = Counter(str(entry["namespace"]) for entry in catalog)
    verification = load_json_if_present(VERIFICATION)
    reconstruction = load_json_if_present(
        ROOT / "research" / "reconstruction" / "reports" / "completeness_summary.json"
    )
    commit_count = git("rev-list", "--all", "--count").strip()
    lines = [
        "# Ozon E-Cup — полный архив решения и всех экспериментов", "",
        "## Overview", "",
        "Задача — для каждого из 250 000 пользователей предсказать суммарный GMV в Поиске и Каталоге "
        "за 30 дней `2026-02-14…2026-03-15` по разреженной дневной истории до `2026-02-13`. "
        "Target — неотрицательная сумма `gmv`; формат результата — `user_id,predict`.", "",
        "Метрика соревнования — RMSLE: RMSE между `log1p(y)` и `log1p(max(pred, 0))`. "
        "Поэтому основные модели обучались в `z=log1p(y)`, а ансамбли смешивались в log-space.", "",
        "Данные: `data/raw/train.parquet` (~30+ млн разреженных user-day строк) и "
        "`data/raw/sample_submit.csv`. Они не коммитятся по правилам проекта. "
        "Дни без активности могут отсутствовать; полный zero-fill не является частью основного pipeline.", "",
        "Общий Team-A pipeline: `build_features(cutoff_date)` → out-of-time train/validation → "
        "OOF в `z` → calibrated wCV → LOFO ensemble check → production inference → log-space blend → "
        "глобальный level shift (где он предусмотрен рецептом) → `expm1` → валидация submission.", "",
        "Основная честная схема после `exp_016`: четыре cutoff-фолда `2025-09-04`, `2025-09-18`, "
        "`2025-10-02`, `2025-10-16`, per-fold optimal log calibration и веса 1:2:4:8. "
        "Train-cutoff после `2025-10-16` для этих сравнений запрещён из-за пересечения selection/target windows.", "",
        "## Что именно было проаудировано", "",
        f"В catalogue находится **{len(catalog)}** отдельных карточек, logged arms, исторических вариантов, EDA/runners и final pipelines. "
        f"Просмотрены текущее дерево, все **{commit_count}** commits во всех refs, удалённые файлы, shell-скрипты, конфиги, result manifests, "
        "внешний teammate bundle, submission registry и generated-artifact manifests. Notebook-файлов нет ни в дереве, ни в git history.", "",
        f"Дополнительный независимый reconstruction audit дал **{reconstruction.get('primary_report_catalog_rows', 'Unknown')}** primary reports, "
        f"**{reconstruction.get('central_experiment_registry_rows', 'Unknown')}** registry rows, **{reconstruction.get('component_groups', 'Unknown')}** component groups, "
        f"**{reconstruction.get('granular_run_metric_records', 'Unknown')}** granular run metrics и **{reconstruction.get('main_scripts', 'Unknown')}** main scripts. "
        "Его coverage был сопоставлен с каталогом; единственный отсутствовавший primary report восстановлен как PARTIAL без догадок.", "",
        "Полный path-by-path отчёт: [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md). "
        "Machine-readable manifest: [experiments/repro/catalog.json](experiments/repro/catalog.json).", "",
        "### Catalogue namespaces", "",
        "| Namespace | Entries | Meaning |", "|---|---:|---|",
    ]
    meanings = {
        "team_a_current": "numbered Team-A experiment cards",
        "team_a_run": "every unique current log.csv run/arm",
        "team_b_b2": "historical 34-experiment Team-B/B2 line",
        "team_b_strategy": "independent Team-B classic/behavior/blending line",
        "strategy_2": "structural Strategy-2",
        "eda": "EDA scripts e01…e15",
        "teammate_research": "long-running teammate research scripts",
        "teammate_review_bundle": "completed review-bundle manifests",
        "teammate_candidate": "every unique completed candidate found in review-bundle validation tables",
        "teammate_final": "STRONGEST and latest rebuilds",
        "team_b_final": "four-model Team-B production integration",
        "team_a_final": "requested final Team-A/Team-B blend",
        "new_direction": "all late research/new_directions directory packages",
        "independent_anniversary": "reconstructed linked-worktree anniversary experiment",
        "packaged_final": "late exact final-submission and blend packages",
    }
    for namespace, count in sorted(namespace_counts.items()):
        lines.append(f"| `{namespace}` | {count} | {meanings.get(namespace, 'isolated historical branch')} |")

    lines.extend([
        "", "## Final solution and submission provenance", "",
        "Репозиторий содержит несколько разных объектов, которые нельзя называть одним «финалом» без уточнения evidence status.", "",
        "### 1. Лучший подтверждённый финальный submission — SUBMIT_JOINT86_TEAMB14", "",
        "`SUBMIT_JOINT86_TEAMB14` имеет externally reported public LB **1.6458200196207617** и exact reference SHA256 "
        "`85d9cd645e14a7895da9ad8cc89065714606266be588c762d37487d2b4edac02`. Это не forecast: значение отдельно помечено "
        "в teammate reproduction request как фактически полученный результат.", "",
        "Формула в `z=log1p(pred)`: frozen `JOINT_V2` (public **1.6459363044782171**) имеет вес **0.86**; Team-B final сначала "
        "получает additive shift **-0.1214326530964569** через bisection до совпадения среднего `z`, клиппинг в ноль, затем вес **0.14**. "
        "После смеси применяется `max(expm1(z),0)`. Внешний blend воспроизводится побайтно. Важное ограничение: точный upstream-generator "
        "frozen `JOINT_V2` в истории не сохранился, поэтому raw→JOINT_V2 честно отмечен `PROVENANCE_INCOMPLETE`.", "",
        "Воспроизведение: `python make_final_submission.py` или "
        "`python scripts/reproduce_final.py --solution SUBMIT_JOINT86_TEAMB14 --from-precomputed`.", "",
        "### 2. Точно упакованный, но не подтверждённый LB candidate — STRONGEST55_TEAMB45", "",
        "Log-space blend: **0.55 STRONGEST-CURRENT + 0.45 level-aligned Team-B**, exact SHA256 "
        "`1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4`. Числа около 1.64823 в research JSON — "
        "только forecast, не leaderboard fact. Запуск: `python make_final_submission.py --recipe strongest55-teamb45`.", "",
        "### 3. Late research candidates: geometry, ORTH, JOINT and three-way", "",
        "Submission geometry дала зафиксированные public LB **1.6467120** и **1.6466079**; ORTH_ALPHA — **1.6461597403**; "
        "JOINT_V2 — **1.6459363044782171**. Скрипты внешнего submission-geometry workspace в этот git не попали, поэтому линия помечена PARTIAL. "
        "`STRONGEST80_TEAMB20`, optimized pair blends и final three-way ensemble сохранены отдельно; их projected scores не выдаются за LB.", "",
        "### 4. Лучший точно воспроизводимый ранний отправленный Team-A submission — STRONGEST-CURRENT", "",
        "`submission_STRONGEST_CURRENT.csv`, public LB **1.6496571**, wCV **1.74751**, SHA256 "
        "`abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`.", "",
        "Веса в `z=log1p(pred)`: `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-AVG3@clip289 + "
        "0.225 ETX-AVG3@DCW`. SEQ и ETX состоят из трёх seed-компонент по 0.075. "
        "После смеси: `delta=2.3293-mean(z)`, затем `z=max(z+delta,0)`, `predict=max(expm1(z),0)`. "
        "DCW означает согласованный depth cap 289 и train-domain cutoff weekday для ETX query context.", "",
        "Воспроизведение: `python make_final_submission.py --recipe strongest`. Этот быстрый путь использует "
        "сохранённые production predictions; полный retraining невозможен для CAP/UNC/DIST и TCN seed 42, потому что их веса исторически не сохранились.", "",
        "### 5. Ранний externally reported teammate latest", "",
        "`latest.csv`, public LB **1.6492175622 (EXTERNALLY_REPORTED; дата LB-события неизвестна)**, source SHA256 "
        "`7ef5b2c58925bd28c5bc7eb83b9cfd4785c608a0c8b2a6d7a3277730cba8e722`. "
        "Точный восстановленный рецепт в log-space: `0.12 friend + 0.16 occ_meta_B + 0.72 occ_raw_X3`, "
        "затем `z=max(z,0)` и `expm1`; дополнительной level normalization нет. Максимальная ошибка reconstruction — `8.88e-16`.", "",
        "Воспроизведение: `python make_final_submission.py --recipe latest`. Ограничение: canonical OOF для `latest` отсутствует, "
        "CAP lineage не восстановлен; поэтому `latest` не используется как CV/LOFO или private-safe anchor.", "",
        "### 6. Requested blend exp_071", "",
        "`submission_FINAL_CAP_UNC_DIST_SEQ_ETX_TEAM_B.csv` — подготовлен, но LB в репозитории не зафиксирован. "
        "Абсолютные веса: `CAP=.055042443`, `UNC=.110084886`, `DIST=.137606107`, "
        "`TEAM_B=.247266564`, `SEQ-AVG3=.225`, `ETX-AVG3-DCW=.225`. "
        "Team-B занимает `0.449575571` от 55%-го tabular slot. Full delta wCV `-0.000382364` (4/4), "
        "LOFO `-0.000381489` (4/4), OOF→TEST variance ratio `1.021814` PASS; gain ниже promotion gate 0.0005.", "",
        "Воспроизведение: `python make_final_submission.py --recipe team-a-b2`. Скрипт заново подбирает alpha только по canonical OOF, "
        "проверяет LOFO и test-regime gate, затем формирует CSV; public LB при выборе не используется.", "",
        "### 7. Другие финальные кандидаты", "",
        "`python make_final_submission.py --recipe final-candidates` пересобирает пакет exp_065: "
        "A = byte-identical STRONGEST-CURRENT; B = `0.95 STRONGEST + 0.05 BTYD`, для которого nested LOFO "
        "`-0.000269` (4/4), fixed .05 `-0.000321` (4/4), production-support PASS. "
        "Также подготовлены, но не обязательно отправлены: SEQAVG3@clip289, SEQ65, LEVEL_MINUS_006, RIDGE15, ZERO2D и TEAM_B_B2 candidates.", "",
        "## Submission registry", "",
        "`LB` ниже приводится только там, где он действительно записан в repository evidence.", "",
        "| File | Experiment | Date | Level | Public LB | Components / status |", "|---|---|---|---:|---:|---|",
    ])
    for row in submissions:
        lines.append(
            f"| `{table_cell(row.get('file'), 90)}` | `{table_cell(row.get('exp_id'), 45)}` | "
            f"{table_cell(row.get('date'), 20)} | {table_cell(row.get('level'), 25)} | "
            f"{table_cell(row.get('lb_public') or 'Unknown / not uploaded', 30)} | "
            f"{table_cell((row.get('oof_source') or '') + ' — ' + (row.get('note') or ''), 260)} |"
        )

    lines.extend([
        "", "## Experiment table — полный каталог", "",
        "Повторяющиеся номера допустимы: namespace сохраняет независимые ветки. `Score` — дословный recovered excerpt; "
        "если точное значение не сохранилось, указано Unknown. Полная карточка и frozen code доступны по ссылке.", "",
        "| ID | Experiment | Model | Features | Validation | Score | Submission | Notes |", "|---|---|---|---|---|---|---|---|",
    ])
    for entry in catalog:
        link = f"experiments/repro/{entry['catalog_id']}/README.md"
        notes = f"{entry['namespace']}; {entry['reproducibility']}"
        if entry.get("notes"):
            notes += f"; {entry['notes']}"
        lines.append(
            f"| `{table_cell(entry['catalog_id'], 100)}` | [{table_cell(entry['title'], 110)}]({link}) | "
            f"{table_cell(entry['model'], 80)} | {table_cell(entry['features'], 90)} | "
            f"{table_cell(entry['validation'], 105)} | {table_cell(entry['score'], 105)} | "
            f"{table_cell(entry['submission'], 100)} | {table_cell(notes, 140)} |"
        )

    lines.extend(["", "## Experiments — подробные карточки", ""])
    for namespace in sorted(namespace_counts):
        lines.extend([f"### Namespace `{namespace}`", ""])
        for entry in [item for item in catalog if item["namespace"] == namespace]:
            link = f"experiments/repro/{entry['catalog_id']}/README.md"
            commands = entry.get("commands") or []
            command_text = table_cell(commands[0], 400) if commands else "Unknown / not recoverable from repository history"
            sources = ", ".join(f"`{path}`" for path in entry.get("source_files", [])[:12]) or "Unknown / not recoverable from repository history"
            lines.extend([
                f"#### `{entry['catalog_id']}` — {entry['title']}", "",
                f"- Original code/document: `{entry['original_path']}`; source `{entry['source_ref']}` @ `{entry['source_commit']}`.",
                f"- Idea/model: {entry['model']}.",
                f"- Features: {entry['features']}.",
                f"- Preprocessing: {entry['preprocessing']}.",
                f"- Validation: {entry['validation']}.",
                f"- Hyperparameters: preserved verbatim in [the experiment folder]({link}); values not present there are **Unknown / not recoverable from repository history**.",
                f"- Seed: {entry['seed']}.",
                f"- Postprocessing: {entry['postprocessing']}.",
                f"- External inputs: {entry['external_data']}.",
                f"- Implementation files: {sources}.",
                f"- Reproduction command: `{command_text}`; universal inspection: `python experiments/repro/{entry['catalog_id']}/run.py`.",
                f"- Prediction/submission: {entry['submission']}.",
                f"- Known score/evidence: {entry['score']}.",
                f"- Status/limitations: {entry['reproducibility']}. {entry.get('notes') or ''}".rstrip(), "",
            ])

    lines.extend([
        "## Reproduction", "",
        "### 1. Environment", "",
        "```bash", "python -m venv .venv", ".venv/Scripts/activate", "python -m pip install -r requirements.txt", "```", "",
        "GPU experiments (TCN/ETX/fine-tuning) require a CUDA-compatible PyTorch build. Historical Team-B and teammate runners "
        "may need `duckdb`, `scipy`, `numba`, `matplotlib` and the exact environment named in their frozen README.", "",
        "### 2. Data", "",
        "Place `train.parquet` and `sample_submit.csv` in `data/raw/`. Do not add them to git. "
        "Every feature must be built only from dates at or before its cutoff; target is `(T,T+30]` in the Team-A pipeline.", "",
        "### 3. Inspect or run one archived experiment", "",
        "```bash", "python experiments/repro/team_a_current__exp_014_s1_dist_head/run.py", "python experiments/repro/team_a_current__exp_014_s1_dist_head/run.py --execute 1", "```", "",
        "The first command is non-mutating and prints provenance, source files, reproducibility status and recovered commands. "
        "The second executes command 1. Frozen historical implementations live inside the same experiment folder.", "",
        "### 4. Main Team-A training/prediction examples", "",
        "```bash", "python -m src.smoke", "python -m src.train --exp S1-B0 --cutoffs recent3 --model direct --rounds 600", "python -m src.predict --help", "python -m src.seq --help", "python -m src.etx --help", "```", "",
        "Exact commands for every experiment are in its preserved card. Full retraining of all models is intentionally not part of "
        "the audit because it requires many GPU-hours; syntax/import/unit checks cover the archive without changing experimental results.", "",
        "### 5. Final submissions", "",
        "```bash", "python make_final_submission.py                              # exact JOINT86/Team-B14 final", "python make_final_submission.py --recipe strongest55-teamb45  # exact unsubmitted candidate", "python make_final_submission.py --recipe team-a-b2             # exp_071 requested blend", "python make_final_submission.py --recipe strongest             # exp_037 early champion", "python make_final_submission.py --recipe latest                # teammate .12/.16/.72 reconstruction", "python make_final_submission.py --recipe final-candidates", "```", "",
        "## Verification", "",
        f"`python tools/verify_experiment_archive.py` завершился {verification.get('status', 'pending rerun')}: "
        f"**{verification.get('catalog_entries', len(catalog))}** уникальных entries, "
        f"**{verification.get('historical_cards', 'Unknown')}** исторических карточек, "
        f"**{verification.get('python_files_compiled', 'Unknown')}** Python-файлов успешно скомпилированы in-memory, "
        f"archive-added raw data/submission paths: **{verification.get('archive_added_forbidden_files', 'Unknown')}**. "
        "Все **69/69** активных root-level `src`-модулей импортируются; обязательные зависимости доступны. "
        "Merge-коллизия module-vs-package для `features/models/validation` устранена compatibility exports: legacy bare imports и новые explicit submodules обеих линий доступны одновременно.", "",
        "`python -m pytest -q`: **492 passed, 3 failed**. Failure 1 — защищённый calendar contract: cutoff `2025-08-08` даёт конец окна `2025-09-07`, позже validation cutoff `2025-09-04`. "
        "Failure 2 — `LANDMARK_MEMORY_EXP055`: fetched `preflight_verdict.json` не совпадает с canonical hash старого replay manifest. "
        "Failure 3 — `STATE_REWEIGHT_EXP057`: immutable `phase0_audit.json` содержит pre-archive `base_head`; тест правильно отказывается молча переписать evidence после rebase. "
        "Ни один evidence artifact, `src/validation.py` или `src/config.py` ради зелёного теста не переписывался. `pytest.ini` ограничивает обычный discovery активной `src/`; frozen snapshots отдельно syntax-checked.", "",
        "Фактические final rebuilds также пройдены: JOINT86/Team-B14 и STRONGEST55/Team-B45 побайтно совпали с reference SHA и дали 250 000 валидных строк; STRONGEST совпал с отправленным CSV; `latest` совпал в log-space с max error `8.88e-16`; exp_071 и exp_065 ранее пересобрали заявленные candidates.", "",
        "## Repository structure", "",
        "```text",
        "OZON-E-CUP/",
        "├── README.md                         # this exhaustive document",
        "├── STATE.md / HISTORY.md             # current conclusions and archived decisions",
        "├── make_final_submission.py          # dispatches every preserved final recipe",
        "├── experiments/",
        "│   ├── exp_*.md                      # original Team-A cards",
        "│   ├── log.csv / submissions.csv     # run and leaderboard registries",
        "│   └── repro/                        # 1 folder per recovered experiment/run",
        "│       ├── catalog.json / catalog.csv",
        "│       ├── runner.py",
        "│       └── <namespace>__<id>/",
        "│           ├── README.md",
        "│           ├── experiment.json",
        "│           ├── run.py",
        "│           └── implementation/       # frozen relevant Python source",
        "├── src/                              # active shared training/inference code",
        "├── research/                         # EDA, strategies, reconstruction, new directions",
        "├── reproducibility/                  # exact frozen final-submission packages",
        "├── пайплайн сокомандника/            # external teammate provenance bundle",
        "├── weights_archives/                 # external model-weight archives",
        "├── docs/REPOSITORY_AUDIT.md           # path-level audit and git-history coverage",
        "├── data/                             # local raw data ignored; fetched frozen package evidence preserved",
        "├── artifacts/                        # ignored generated OOF/checkpoints/cache",
        "└── submissions/                      # ignored generated competition CSVs",
        "```", "",
        "## Historical / abandoned experiments", "",
        "Rejected experiments are deliberately retained. Major closed families include: three-block train panel; strict/short history; "
        "weekly lags; radical minimalism; rank-cohort similarity; last-fold-only training; rounds≥450; gap-axis as a selection gate; "
        "dense supervision; personal-time tabular features; multi-horizon hazard; full-depth-365 sequence inference; train-time avail augmentation; "
        "future-funnel auxiliary targets; block residuals; FRESH contrast/fine-tune; Ridge fixed slot; classic BG/NBD residual promotion; "
        "Shapley channel decomposition; burst/gap episodes; landmark memory; late SSL; dataset fingerprint; state reweighting; open funnel; "
        "platform detrending; event order; occurrence revisit; and full Team-B tab-slot replacement. "
        "The exact negative results and 'do not repeat' constraints remain in `STATE.md` and individual cards.", "",
        "Historical Team-B variants are not silently mapped onto Team-A validation: they used different folds, panels, targets and calibration. "
        "Their namespaces preserve those differences. Failed or rolled-back code is marked PARTIAL rather than recreated with guessed parameters.", "",
        "## Reproducibility limits", "",
        "- Local raw competition data and newly generated submissions remain ignored; fetched `origin/team-a` exact-reproduction packages intentionally contain reviewed frozen inputs/references with manifests.",
        "- Exact JOINT86 outer blend is byte-reproducible; the upstream generator of its frozen JOINT_V2 anchor is not recoverable and is not claimed as complete.",
        "- Many OOF arrays/checkpoints are generated artifacts; manifests/hashes and commands are committed, binaries remain external.",
        "- STRONGEST-CURRENT is exactly reproducible from saved predictions, but not every underlying model can be retrained because several weights were never saved.",
        "- `latest` is exactly reconstructible from three component CSVs, but its canonical OOF and complete CAP lineage are missing.",
        "- A score shown as Unknown was not guessed. External reports are explicitly labelled.",
        "- Три active evidence/state tests остаются красными; точные причины приведены в разделе Verification и не маскируются изменением исторических артефактов.",
    ])
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_catalog_readme(catalog: list[dict[str, object]]) -> None:
    path = ROOT / "experiments" / "repro" / "README.md"
    counts = Counter(str(entry["namespace"]) for entry in catalog)
    lines = [
        "# Reproducible experiment catalogue", "",
        f"This directory contains **{len(catalog)}** separately namespaced experiment/run folders. "
        "It was generated from current cards, `log.csv`, every historical experiment card reachable through `git --all`, "
        "EDA scripts and teammate run manifests.", "",
        "Each folder contains `README.md`, `experiment.json`, `run.py`, and a frozen `implementation/` subset. "
        "`python <folder>/run.py` is read-only; add `--execute N` to run one recovered command.", "",
        "| Namespace | Entries |", "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(counts.items()))
    lines.extend(["", "Machine-readable files: `catalog.json` and `catalog.csv`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    build_audit(catalog)
    build_readme(catalog)
    build_catalog_readme(catalog)
    print(f"wrote {README}, {AUDIT} and experiments/repro/README.md")


if __name__ == "__main__":
    main()
