#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage=${1:-smoke}; workers=${2:-1}; device=${3:-cuda}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "$device" == cuda* ]]; then python scripts/check_gpu.py --require-cuda; fi
if [[ "$stage" == all ]]; then
  bash scripts/run_v16.sh smoke "$workers" "$device"
  python scripts/benchmark_v16.py --device "$device"
  bash scripts/run_v16.sh main "$workers" "$device"
  exit 0
fi
config=$(python scripts/prepare_v16.py "$stage" --device "$device")
label=${device//:/_}
python -u scripts/run_handoff_round.py --config "$config" --out "results/server_v16_${stage}_${label}" --workers "$workers"
python scripts/report_v16.py --run "results/server_v16_${stage}_${label}" --out "analysis/server_v16_${stage}_${label}"
