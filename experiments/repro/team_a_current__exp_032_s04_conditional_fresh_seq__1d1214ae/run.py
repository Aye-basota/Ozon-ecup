"""Launcher for this archived experiment.

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
