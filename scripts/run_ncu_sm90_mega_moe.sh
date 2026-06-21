#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python}
num_processes=${NUM_PROCESSES:-8}
tokens=${TOKENS:-4}
run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
output_dir=${OUTPUT_DIR:-"$repo_root/work/sm90-mega-moe-ncu/$run_id"}
jit_cache_dir=${DG_JIT_CACHE_DIR:-"$output_dir/jit-cache"}

mkdir -p "$output_dir" "$jit_cache_dir"

command -v ncu >/dev/null 2>&1 || {
    echo "Nsight Compute (ncu) is not available" >&2
    exit 1
}

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
export DG_JIT_CACHE_DIR="$jit_cache_dir"
export DG_JIT_USE_NVRTC=0
export DG_JIT_WITH_LINEINFO=1
export DG_JIT_PTXAS_CHECK=1

python_args=(
    tests/bench_mega_moe_sm90.py
    --num-processes "$num_processes"
    --num-max-tokens-per-rank 8192
    --tokens "$tokens"
    --ncu-profile-only
)
python_args+=("$@")

echo "Output directory: $output_dir"
echo "Warm up the JIT cache"
"$python_bin" "${python_args[@]}" 2>&1 | tee "$output_dir/warmup.log"

ncu_args=(
    --config-file off
    --force-overwrite
    --kernel-name sm90_fp8_mega_moe_impl
    --import-source yes
    --replay-mode application
    --section LaunchStats
    --section Occupancy
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section SourceCounters
    --rule LocalMemoryUsage
    --launch-skip 0
    --launch-count 1
    --lockstep-kernel-launch
    --communicator tcp
    --communicator-tcp-num-peers "$num_processes"
    --clock-control none
    --kill yes
    --app-replay-buffer memory
)

pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for ((rank = 0; rank < num_processes; ++rank)); do
    (
        export DG_USE_NVIDIA_TOOLS=1
        ncu "${ncu_args[@]}" \
            -o "$output_dir/sm90-mega-moe-rank$rank" \
            "$python_bin" tests/bench_mega_moe_sm90.py \
            --num-processes "$num_processes" \
            --local-rank-idx "$rank" \
            --num-max-tokens-per-rank 8192 \
            --tokens "$tokens" \
            --ncu-profile-only \
            "$@"
    ) >"$output_dir/rank$rank.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
trap - INT TERM

if ((status != 0)); then
    echo "NCU profiling failed; inspect $output_dir/rank*.log" >&2
    exit "$status"
fi

echo "NCU reports written to $output_dir"
