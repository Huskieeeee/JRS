"""Locate repository-owned experiment assets without depending on cwd."""
import os
from pathlib import Path


def project_root():
    configured = os.environ.get("FLOOR_TASE_ROOT")
    candidates = [Path(configured)] if configured else [Path(__file__).resolve().parents[3], Path.cwd(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs/v15_main_template.json").is_file():
            return candidate.resolve()
    raise RuntimeError("Experiment commands require a repository checkout. Install with pip install -e . or set FLOOR_TASE_ROOT.")


ROOT = project_root()
