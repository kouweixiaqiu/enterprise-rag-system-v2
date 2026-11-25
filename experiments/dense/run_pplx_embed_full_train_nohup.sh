#!/usr/bin/env bash

set +e

project_root="/home/fuxuanyi/AI_projects/enterprise-rag-system-v2"
results_dir="$project_root/experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_train"
pid_file="$results_dir/experiment.pid"
exit_file="$results_dir/exit_code.txt"

cd "$project_root" || exit 200
mkdir -p "$results_dir"

printf 'START pplx-embed-v1-0.6b full TRAIN offline streaming batch_size=32\n'
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$project_root/.venv/bin/python" -u \
  "$project_root/experiments/dense/run_pplx_embed_full_train.py" &
experiment_pid=$!
printf '%s\n' "$experiment_pid" > "$pid_file"
printf 'EXPERIMENT_PID %s\n' "$experiment_pid"

wait "$experiment_pid"
run_code=$?
printf '%s\n' "$run_code" > "$exit_file"
printf 'EXIT_CODE %s\n' "$run_code"
exit "$run_code"
