"""Run a standardized tabular challenger from an experiment config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.workflow import ExperimentSpec, run_cv, run_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test", action="store_true", help="also train the full model and write TEST predictions")
    args = parser.parse_args()
    config_path = args.config.resolve()
    spec = ExperimentSpec.from_yaml(config_path)
    result = run_cv(spec, config_path.parent)
    print(f"wCV={result['wCV']:.12f} delta={result['delta_wCV']:+.12f}")
    if args.test:
        path = run_test(spec, config_path.parent)
        print(f"TEST={path}")


if __name__ == "__main__":
    main()
