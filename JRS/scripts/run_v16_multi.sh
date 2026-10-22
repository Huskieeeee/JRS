#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage=${1:-main}; IFS=, read -r -a gpus <<< "${2:?Give GPU IDs, e.g. 4,5}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
pids=(); runs=()
for index in "${!gpus[@]}"; do
  config=$(python scripts/prepare_v16.py "$stage" --device cuda --shard "$index" --shards "${#gpus[@]}")
  run="results/server_v16_${stage}_cuda_shard_${index}of${#gpus[@]}"
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" python -u scripts/run_handoff_round.py --config "$config" --out "$run" --workers 1 > "logs/v16_${stage}_shard_${index}.log" 2>&1 &
  pids+=("$!"); runs+=("$run")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then echo 'A shard failed; inspect logs/v16_*_shard_*.log'; exit 1; fi
merged="results/server_v16_${stage}_cuda_merged"
if [[ ! -d "$merged" ]]; then python scripts/merge_v16.py --runs "${runs[@]}" --out "$merged"; fi
python scripts/report_v16.py --run "$merged" --out "analysis/server_v16_${stage}_cuda_merged"
