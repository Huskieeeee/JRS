#!/usr/bin/env bash
# Codex Cloud install script. Run from the selected repository's checkout.
set -euo pipefail
if [[ -n "${PYTHON:-}" ]]; then
  task_python="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  task_python=python3.12
else
  task_python=python3
fi

task_project=$("$task_python" - "${FLOOR_TASE_ROOT:-$PWD}" <<'PY'
from pathlib import Path
import sys
import zipfile

if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required; select Python 3.12 in environment setup.")
base = Path(sys.argv[1]).resolve()
def locate():
    candidates = ([base / 'pyproject.toml'] if (base / 'pyproject.toml').is_file() else
                  [p for pattern in ('*/pyproject.toml', '*/*/pyproject.toml') for p in base.glob(pattern)])
    return sorted({p.parent for p in candidates if 'floor-tase-handoff' in p.read_text()})
projects = locate()
if not projects:
    archives = sorted(set(base.glob('FLOOR_MAS_TASE_GitHub_Codex_*.zip')) |
                      set(base.glob('FLOOR_MAS_TASE_Handoff_v1_6*.zip')))
    if len(archives) != 1:
        raise SystemExit('Expected extracted source or exactly one FLOOR-MAS v1.6/engineering ZIP in the repository.')
    with zipfile.ZipFile(archives[0]) as archive:
        for entry in archive.infolist():
            if not (base / entry.filename).resolve().is_relative_to(base):
                raise SystemExit('Archive contains a path outside the repository.')
        archive.extractall(base)
    projects = locate()
if len(projects) != 1:
    raise SystemExit('Run from the JRS checkout, or set FLOOR_TASE_ROOT to the project directory; expected one FLOOR-TASE project.')
print(projects[0])
PY
)
cd "$task_project"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_venv="$task_project/.venv"
"$task_python" -m venv "$task_venv"
"$task_venv/bin/python" -m pip install --disable-pip-version-check --retries 1 --timeout 30 -e .
if [[ -f scripts/dev/check.sh ]]; then
  PYTHON="$task_venv/bin/python" bash scripts/dev/check.sh
else
  "$task_venv/bin/python" -m compileall -q src scripts tests
  "$task_venv/bin/python" -m unittest discover -s tests -v
fi
printf '\nReady. Project: %s\nPython: %s/bin/python\n' "$task_project" "$task_venv"
