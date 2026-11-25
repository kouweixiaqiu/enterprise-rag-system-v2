#!/usr/bin/env bash
set -Eeuo pipefail

project_dir=/home/fuxuanyi/AI_projects/enterprise-rag-system-v2
python_bin="$project_dir/.venv/bin/python"
run_stamp=20261122_081203
interval_seconds=1800
rerank_dir="$project_dir/experiments/dense/results/qwen3_reranker_4b_gguf_top20_fixed_600_0_train"
run_log="$rerank_dir/run_${run_stamp}.log"
exit_file="$rerank_dir/exit_code_${run_stamp}.txt"
rankings_file="$rerank_dir/rankings.jsonl"
summary_file="$rerank_dir/summary.json"
task_dir="$project_dir/experiments/automation/results/qwen4b_${run_stamp}_then_generate"
metrics_file="$task_dir/qwen4b_metrics.json"
task_exit_file="$task_dir/exit_code.txt"
state_file="$task_dir/state.txt"
key_config=/home/fuxuanyi/.config/enterprise-rag/siliconflow.env

mkdir -p "$task_dir"

finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf '%s\n' "$status" >"$task_exit_file"
    if [[ "$status" -eq 0 ]]; then
        printf 'complete\n' >"$state_file"
    else
        printf 'failed exit_code=%s\n' "$status" >"$state_file"
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

printf 'waiting_for_qwen4b\n' >"$state_file"
log "bound to run stamp $run_stamp; checking every $interval_seconds seconds"

while [[ ! -f "$exit_file" ]]; do
    sleep "$interval_seconds"
    checkpoint_count=$(find "$rerank_dir/checkpoints" -maxdepth 1 -type f -name 'TRAIN_Q*.json' | wc -l)
    log "still waiting: checkpoints=$checkpoint_count/450"
done

run_exit=$(tr -d '[:space:]' <"$exit_file")
if [[ ! "$run_exit" =~ ^[0-9]+$ ]]; then
    log "invalid reranker exit-code file: $exit_file"
    exit 1
fi
if [[ "$run_exit" -ne 0 ]]; then
    log "reranker failed with exit code $run_exit; generation will not run"
    if [[ -f "$run_log" ]]; then
        log "last 80 reranker log lines follow"
        tail -n 80 "$run_log"
    fi
    exit 1
fi
if [[ ! -s "$rankings_file" || ! -s "$summary_file" ]]; then
    log "reranker reported success but final ranking or summary is missing"
    exit 1
fi

printf 'validating_qwen4b\n' >"$state_file"
log "reranker exited successfully; validating all 450 rankings and metrics"
"$python_bin" -u "$project_dir/scripts/summarize_qwen4b_train_completion.py" \
    --output "$metrics_file"

if [[ ! -f "$key_config" ]]; then
    log "SiliconFlow config is missing: $key_config"
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$key_config"
set +a
if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
    log "SILICONFLOW_API_KEY is missing or empty in the local config"
    exit 1
fi

printf 'generating_answers\n' >"$state_file"
log "starting fixed 20-question incremental generation"
"$python_bin" -u "$project_dir/scripts/generate_qwen4b_reranked_top5_seed42_20.py"
log "postprocessing and generation completed"
