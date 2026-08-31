"""Reproduce any final E-Cup submission recipe preserved in this repository.

The default is the byte-exact packaged ``SUBMIT_JOINT86_TEAMB14`` submission:
it has the best externally reported public LB among the fully identified final
files in the merged Team-A evidence.  Its outer blend is exact; the frozen
``JOINT_V2`` input remains explicitly marked as upstream-provenance incomplete.

Examples:
    python make_final_submission.py
    python make_final_submission.py --recipe strongest55-teamb45
    python make_final_submission.py --recipe team-a-b2
    python make_final_submission.py --recipe strongest
    python make_final_submission.py --recipe latest
    python make_final_submission.py --recipe final-candidates
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RECIPES = {
    "joint86-teamb14": ROOT / "reproducibility" / "SUBMIT_JOINT86_TEAMB14" / "build_submit.py",
    "strongest55-teamb45": ROOT / "reproducibility" / "SUBMIT_STRONGEST55_TEAMB45" / "build_submit.py",
    "team-a-b2": ROOT / "src" / "final_team_b_ensemble.py",
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
        "--recipe", choices=tuple(RECIPES), default="joint86-teamb14",
        help=(
            "joint86-teamb14=reported 1.645820 final; strongest55-teamb45=unsubmitted exact candidate; "
            "team-a-b2=exp_071 prepared blend; strongest=exp_037 submitted champion; "
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
    # Do not leak this dispatcher's ``--recipe`` into the archived script's
    # independent argparse parser.
    previous_argv = sys.argv
    try:
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
