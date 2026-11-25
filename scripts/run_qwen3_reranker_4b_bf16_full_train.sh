#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/fuxuanyi/AI_projects/enterprise-rag-system-v2
result_dir="$project_dir/experiments/dense/results/qwen3_reranker_4b_bf16_top20_fixed_600_0_train"
stamp=$(date +%Y%m%d_%H%M%S)
log_file="$result_dir/resume_${stamp}.log"
exit_file="$result_dir/resume_${stamp}.exit_code"

mkdir -p "$result_dir"
printf '%s\n' "$log_file" >"$result_dir/active_log.txt"
printf '%s\n' "$exit_file" >"$result_dir/active_exit_file.txt"

cd "$project_dir" || exit 2
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/usr/bin/time -v ./.venv/bin/python -u \
    experiments/dense/run_qwen3_reranker_4b_bf16_full_train.py \
    2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" >"$exit_file"
exit "$status"
