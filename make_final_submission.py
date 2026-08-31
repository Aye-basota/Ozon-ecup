"""Reproduce any final E-Cup submission recipe preserved in this repository.

The default is the final explicitly requested Team-A/Team-B partial-slot blend
(``exp_071``).  Other recipes remain separate because they have different
evidence status: ``strongest`` is the best exactly reproducible submitted Team-A
model, while ``latest`` has the best externally reported public LB but lacks its
canonical OOF bank.

Examples:
    python make_final_submission.py
    python make_final_submission.py --recipe strongest
    python make_final_submission.py --recipe latest
    python make_final_submission.py --recipe final-candidates
"""
from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RECIPES = {
    "final-team-b": ROOT / "src" / "final_team_b_ensemble.py",
    "strongest": (
        ROOT / "пайплайн сокомандника" / "friend_original"
        / "submission_STRONGEST_CURRENT" / "pipeline" / "build_submission.py"
    ),
    "latest": ROOT / "пайплайн сокомандника" / "latest" / "rebuild_latest.py",
    "final-candidates": ROOT / "src" / "final_integration.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe", choices=tuple(RECIPES), default="final-team-b",
        help=(
            "final-team-b=exp_071 prepared blend; strongest=exp_037 submitted champion; "
            "latest=.12/.16/.72 teammate blend; final-candidates=exp_065 A/B package"
        ),
    )
    args = parser.parse_args()
    script = RECIPES[args.recipe]
    if not script.exists():
        raise FileNotFoundError(
            f"Recipe source is absent: {script}. Restore the external teammate bundle "
            "or select a recipe whose artifacts are available."
        )
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
