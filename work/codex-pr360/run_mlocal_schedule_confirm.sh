#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/chen/workspace/source_code/DeepGEMM}
SITE=${SITE:-/tmp/codex-pr360-site-mgrouplocal-20260624-172521}
RUNS=${RUNS:-5}
NUM_TESTS=${NUM_TESTS:-7}
RUN_ROOT=${RUN_ROOT:-/tmp/codex-pr360-mlocal-confirm-$(date +%Y%m%d-%H%M%S)}

if [[ ! -d "$SITE/deep_gemm" ]]; then
  echo "Missing clean DeepGEMM install: $SITE" >&2
  echo "Rebuild the M-local /tmp install before running this script." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
source "$ROOT/work/codex-pr360/env.sh"
export PYTHONPATH="$SITE:$ROOT/work/codex-pr360/site${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

printf 'GPU process snapshot before benchmark:\n'
nvidia-smi pmon -c 1 || true
printf '\nRUN_ROOT=%s\n' "$RUN_ROOT"
printf 'RUNS=%s NUM_TESTS=%s SITE=%s\n' "$RUNS" "$NUM_TESTS" "$SITE"

run_case() {
  local model="$1" hidden="$2" ih="$3" topk="$4" variant="$5" extra_env="$6" iter="$7"
  local log="$RUN_ROOT/${model}_${variant}_run${iter}.log"
  echo "RUN model=$model variant=$variant run=$iter log=$log"
  (
    cd /tmp
    export DG_JIT_CACHE_DIR="$RUN_ROOT/jit_${model}_${variant}"
    export DG_SM90_MOE_KERNEL=cooperative
    unset DG_SM90_MOE_MLOCAL DG_SM90_MOE_L1_NMAJOR DG_SM90_MOE_EXPERT_LOCAL DG_SM90_MOE_NMAJOR DG_SM90_MOE_SFB_SMEM
    eval "$extra_env"
    python3 /tmp/codex-pr360-tests/bench_mega_moe_sm90.py \
      --num-processes 8 --batches 8192 \
      --hidden "$hidden" --intermediate-hidden "$ih" \
      --num-experts 384 --num-topk "$topk" --num-tests "$NUM_TESTS"
  ) >"$log" 2>&1
  python3 - <<PY_PARSE
import pathlib, re
log = pathlib.Path("$log")
text = log.read_text(errors="ignore")
ms = re.findall(r"tokens=\s*8192\s+recv=\s*(\d+)\s+experts=\s*(\d+)\s+([0-9.]+) us\s+([0-9.]+) TFLOPS\s+([0-9.]+) GB/s", text)
if not ms:
    print(f"RESULT model=$model variant=$variant run=$iter status=FAIL log={log}", flush=True)
else:
    recv, experts, us, tflops, gbs = ms[-1]
    print(f"RESULT model=$model variant=$variant run=$iter recv={recv} experts={experts} us={us} tflops={tflops} gbs={gbs}", flush=True)
PY_PARSE
}

for ((i = 1; i <= RUNS; ++i)); do
  run_case MiMoPro 6144 2048 8 base "" "$i"
  run_case MiMoPro 6144 2048 8 mlocal "export DG_SM90_MOE_MLOCAL=1" "$i"
  run_case V4Pro 7168 3072 6 base "" "$i"
  run_case V4Pro 7168 3072 6 mlocal "export DG_SM90_MOE_MLOCAL=1" "$i"
done

python3 - <<PY_SUMMARY
from pathlib import Path
import re, statistics as st
root = Path("$RUN_ROOT")
vals = {}
for log in sorted(root.glob("*.log")):
    m = re.match(r"(MiMoPro|V4Pro)_(base|mlocal)_run(\d+)\.log", log.name)
    if not m:
        continue
    text = log.read_text(errors="ignore")
    ms = re.findall(r"tokens=\s*8192\s+recv=\s*(\d+)\s+experts=\s*(\d+)\s+([0-9.]+) us", text)
    if ms:
        vals.setdefault((m.group(1), m.group(2)), []).append(float(ms[-1][2]))

print("\nSummary:")
for key in sorted(vals):
    data = vals[key]
    std = st.stdev(data) if len(data) > 1 else 0.0
    print(f"SUMMARY {key[0]} {key[1]} n={len(data)} mean={st.mean(data):.1f} std={std:.1f} values=" + ",".join(f"{x:.1f}" for x in data))
for model in ("MiMoPro", "V4Pro"):
    base = vals.get((model, "base"), [])
    opt = vals.get((model, "mlocal"), [])
    if len(base) == len(opt) and base:
        wins = sum(o < b for b, o in zip(base, opt))
        print(f"SPEEDUP {model} mlocal {(st.mean(base) / st.mean(opt) - 1) * 100:.2f}% wins={wins}/{len(base)}")
    else:
        print(f"SPEEDUP {model} mlocal incomplete base_n={len(base)} opt_n={len(opt)}")
print(f"RUN_ROOT {root}")
PY_SUMMARY
