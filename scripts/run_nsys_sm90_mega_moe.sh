#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python}
num_processes=${NUM_PROCESSES:-8}
tokens=${TOKENS:-4}
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
output_dir=${OUTPUT_DIR:-"$repo_root/work/sm90-mega-moe-nsys/$run_id"}
jit_cache_dir=${DG_JIT_CACHE_DIR:-"$output_dir/jit-cache"}

mkdir -p "$output_dir" "$jit_cache_dir"

command -v nsys >/dev/null 2>&1 || {
    echo "Nsight Systems (nsys) is not available" >&2
    exit 1
}

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
export DG_USE_NVIDIA_TOOLS=1
export DG_JIT_WITH_LINEINFO=1
export DG_JIT_CACHE_DIR="$jit_cache_dir"

echo "Warm up the JIT cache"
DG_USE_NVIDIA_TOOLS=0 \
"$python_bin" tests/bench_mega_moe_sm90.py \
    --num-processes "$num_processes" \
    --num-max-tokens-per-rank 8192 \
    --tokens "$tokens" \
    --smoke-only \
    "$@" 2>&1 | tee "$output_dir/warmup.log"

nsys profile \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --output="$output_dir/sm90-mega-moe" \
    "$python_bin" tests/bench_mega_moe_sm90.py \
    --num-processes "$num_processes" \
    --num-max-tokens-per-rank 8192 \
    --tokens "$tokens" \
    --smoke-only \
    "$@" 2>&1 | tee "$output_dir/nsys.log"

echo "NSYS report written to $output_dir"
