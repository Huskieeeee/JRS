#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -n "${PYTHON:-}" ]]; then
  task_python="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  task_python="$PWD/.venv/bin/python"
else
  task_python=python3
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
"$task_python" -m compileall -q src scripts tests
for task_script in scripts/*.sh scripts/dev/*.sh; do bash -n "$task_script"; done
"$task_python" -m unittest discover -s tests -v
"$task_python" -m floor_tase doctor
task_output=$(mktemp -d "${TMPDIR:-/tmp}/floor-tase-check.XXXXXXXX")
trap 'rm -rf "$task_output"' EXIT
"$task_python" -m floor_tase smoke --out "$task_output/run" --workers 1
