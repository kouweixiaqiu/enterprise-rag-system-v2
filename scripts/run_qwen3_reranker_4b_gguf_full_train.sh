#!/usr/bin/env bash
set -Eeuo pipefail

project_dir=/home/fuxuanyi/AI_projects/enterprise-rag-system-v2
python_bin="$project_dir/.venv/bin/python"
evaluator="$project_dir/experiments/dense/run_qwen3_reranker_4b_gguf_full_train.py"
results_dir="$project_dir/experiments/dense/results/qwen3_reranker_4b_gguf_top20_fixed_600_0_train"
server_bin=/home/fuxuanyi/.local/opt/ollama-v0.32.5/lib/ollama/llama-server
model_path=/home/fuxuanyi/models/qwen3-reranker-4b/Qwen3-Reranker-4B-Q4_K_M.gguf
ollama_lib=/home/fuxuanyi/.local/opt/ollama-v0.32.5/lib/ollama
cuda_lib="$ollama_lib/cuda_v12"
host=127.0.0.1
port=18080
parallel=2
physical_batch=1408
stamp=$(date +%Y%m%d_%H%M%S)
server_log="$results_dir/server_${stamp}.log"
run_log="$results_dir/run_${stamp}.log"
exit_file="$results_dir/exit_code_${stamp}.txt"
preflight_log="$results_dir/preflight_${stamp}.log"
server_pid=

mkdir -p "$results_dir"

cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -TERM "$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ ! -f "$exit_file" ]]; then
        printf '%s\n' "$status" >"$exit_file"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [[ ! -x "$python_bin" || ! -f "$evaluator" || ! -x "$server_bin" || ! -f "$model_path" ]]; then
    echo "required executable or model is missing" >&2
    exit 2
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "CUDA is not available in this WSL session; refusing to start llama-server" >&2
    exit 2
fi

if curl --silent --fail "http://$host:$port/health" >/dev/null 2>&1; then
    echo "refusing to reuse or stop an existing service on $host:$port" >&2
    exit 2
fi

cd "$project_dir"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

"$python_bin" -u "$evaluator" --preflight 2>&1 | tee "$preflight_log"
slot_context=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["required_slot_context"])' "$results_dir/preflight.json")
total_context=$((slot_context * parallel))

echo "launching llama-server: parallel=$parallel slot_context=$slot_context total_context=$total_context physical_batch=$physical_batch" | tee -a "$run_log"
env \
    GGML_BACKEND_PATH="$cuda_lib/libggml-cuda.so" \
    LD_LIBRARY_PATH="$cuda_lib:$ollama_lib" \
    "$server_bin" \
    --model "$model_path" \
    --host "$host" \
    --port "$port" \
    --rerank \
    --pooling rank \
    --ctx-size "$total_context" \
    --parallel "$parallel" \
    --n-gpu-layers 99 \
    --batch-size "$slot_context" \
    --ubatch-size "$physical_batch" \
    --flash-attn on \
    --cache-ram 0 \
    --no-webui \
    -lv 4 \
    >"$server_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "llama-server exited before becoming ready; see $server_log" >&2
        exit 1
    fi
    if curl --silent --fail "http://$host:$port/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    echo "llama-server readiness timed out; see $server_log" >&2
    exit 1
fi

set +e
QWEN4B_RERANK_ENDPOINT="http://$host:$port/v1/rerank" \
    "$python_bin" -u "$evaluator" 2>&1 | tee -a "$run_log"
evaluation_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$evaluation_status" >"$exit_file"
exit "$evaluation_status"
