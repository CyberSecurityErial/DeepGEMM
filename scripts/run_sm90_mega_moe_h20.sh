#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python}
num_processes=${NUM_PROCESSES:-8}
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
output_dir=${OUTPUT_DIR:-"$repo_root/work/sm90-mega-moe/$run_id"}
jit_cache_dir=${DG_JIT_CACHE_DIR:-"$output_dir/jit-cache"}

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

usage() {
    cat <<EOF
Usage: $0 <env|build|smoke|bench|trace|all> [benchmark arguments...]

Environment variables:
  PYTHON_BIN       Python executable (default: python)
  NUM_PROCESSES    Number of local GPU processes (default: 8)
  OUTPUT_DIR       Log/result directory
  SMOKE_TOKENS     Smoke cases (default: "1 4 512")
  BENCH_TOKENS     Benchmark cases (default: "1 2 4 8 16 32 64 128 256 512")

Examples:
  $0 env
  $0 build
  $0 smoke
  $0 bench --num-tests 30
  OUTPUT_DIR=work/h20-run $0 all
EOF
}

command_name=${1:-}
if [[ -z "$command_name" ]]; then
    usage
    exit 1
fi
if [[ "$command_name" == "-h" || "$command_name" == "--help" || "$command_name" == "help" ]]; then
    usage
    exit 0
fi
shift

mkdir -p "$output_dir" "$jit_cache_dir"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export DG_JIT_CACHE_DIR="$jit_cache_dir"
export DG_JIT_USE_NVRTC=${DG_JIT_USE_NVRTC:-0}

run_logged() {
    local name=$1
    shift
    echo "[$(timestamp)] Running: $name"
    echo "Command: $*"
    "$@" 2>&1 | tee "$output_dir/$name.log"
}

check_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

collect_env() {
    check_command git
    check_command nvidia-smi
    check_command nvcc
    check_command "$python_bin"

    {
        echo "timestamp=$(timestamp)"
        echo "repo=$repo_root"
        echo "commit=$(git rev-parse HEAD)"
        echo "branch=$(git branch --show-current)"
        echo "python=$python_bin"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
        echo
        git status --short --branch
        echo
        nvidia-smi
        echo
        nvidia-smi topo -m
        echo
        nvcc --version
        echo
        "$python_bin" -m pip --version
    } 2>&1 | tee "$output_dir/environment.txt"

    run_logged env-check \
        "$python_bin" scripts/check_sm90_mega_moe_env.py \
        --num-gpus "$num_processes"
}

build_extension() {
    check_command "$python_bin"
    git submodule update --init --recursive

    ln -sfn "$repo_root/third-party/cutlass/include/cutlass" \
        "$repo_root/deep_gemm/include/cutlass"
    ln -sfn "$repo_root/third-party/cutlass/include/cute" \
        "$repo_root/deep_gemm/include/cute"

    rm -rf "$repo_root/build" "$repo_root/dist"
    find "$repo_root" -maxdepth 1 -name '*.egg-info' -type d -exec rm -rf {} +

    run_logged build "$python_bin" setup.py build

    local so_file
    so_file=$(find "$repo_root/build" -name '*.so' -type f | head -n 1)
    if [[ -z "$so_file" ]]; then
        echo "Build finished without producing a Python extension" >&2
        exit 1
    fi
    ln -sfn "../${so_file#"$repo_root/"}" "$repo_root/deep_gemm/$(basename "$so_file")"

    run_logged import-check "$python_bin" -c \
        "import deep_gemm, torch; print('deep_gemm', deep_gemm.__version__); print('torch', torch.__version__)"
}

run_smoke() {
    local smoke_tokens_text=${SMOKE_TOKENS:-"1 4 512"}
    read -r -a smoke_tokens <<<"$smoke_tokens_text"

    export DG_PRINT_CONFIGS=1
    export DG_JIT_PTXAS_VERBOSE=1
    export DG_JIT_PTXAS_CHECK=1
    export DG_JIT_PRINT_COMPILER_COMMAND=${DG_JIT_PRINT_COMPILER_COMMAND:-1}

    run_logged smoke \
        "$python_bin" tests/bench_mega_moe_sm90.py \
        --num-processes "$num_processes" \
        --num-max-tokens-per-rank 8192 \
        --tokens "${smoke_tokens[@]}" \
        --smoke-only \
        "$@"
}

run_bench() {
    local bench_tokens_text=${BENCH_TOKENS:-"1 2 4 8 16 32 64 128 256 512"}
    read -r -a bench_tokens <<<"$bench_tokens_text"

    export DG_PRINT_CONFIGS=${DG_PRINT_CONFIGS:-1}
    export DG_JIT_PTXAS_CHECK=${DG_JIT_PTXAS_CHECK:-1}

    run_logged benchmark \
        "$python_bin" tests/bench_mega_moe_sm90.py \
        --num-processes "$num_processes" \
        --num-max-tokens-per-rank 8192 \
        --tokens "${bench_tokens[@]}" \
        --num-tests 10 \
        "$@"
}

run_trace() {
    local trace_tokens_text=${TRACE_TOKENS:-"4 128 512"}
    read -r -a trace_tokens <<<"$trace_tokens_text"
    local trace_dir="$output_dir/traces"
    mkdir -p "$trace_dir"

    export DG_PRINT_CONFIGS=${DG_PRINT_CONFIGS:-1}
    run_logged trace \
        "$python_bin" tests/bench_mega_moe_sm90.py \
        --num-processes "$num_processes" \
        --num-max-tokens-per-rank 8192 \
        --tokens "${trace_tokens[@]}" \
        --num-tests 5 \
        --trace-dir "$trace_dir" \
        "$@"
}

echo "Output directory: $output_dir"
echo "JIT cache directory: $jit_cache_dir"
echo "Visible process count: $num_processes"

case "$command_name" in
    env)
        collect_env
        ;;
    build)
        build_extension
        ;;
    smoke)
        run_smoke "$@"
        ;;
    bench)
        run_bench "$@"
        ;;
    trace)
        run_trace "$@"
        ;;
    all)
        collect_env
        build_extension
        run_smoke "$@"
        run_bench "$@"
        ;;
    *)
        echo "Unknown command: $command_name" >&2
        usage
        exit 1
        ;;
esac

echo "[$(timestamp)] Completed: $command_name"
echo "Results: $output_dir"
