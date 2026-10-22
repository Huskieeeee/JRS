#!/usr/bin/env python3
"""Compatibility entry point; implementation: floor_tase.experiments.merge."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floor_tase.experiments.merge import *  # noqa: F403
from floor_tase.experiments.merge import main

if __name__ == "__main__":
    main()
