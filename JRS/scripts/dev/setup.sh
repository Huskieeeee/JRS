#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
task_python=${PYTHON:-python3}
if [[ "${FLOOR_TASE_OFFLINE:-0}" == 1 ]]; then
  "$task_python" -m pip install --no-build-isolation --no-deps -e .
else
  "$task_python" -m pip install -e .
fi
"$task_python" -m floor_tase doctor
