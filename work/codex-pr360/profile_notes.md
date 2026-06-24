# SM90 MegaMoE Profiling Notes

Date: 2026-06-23 UTC

## Context

- Repo workspace: `/home/chen/workspace/source_code/DeepGEMM`
- Active PR code: upstream PR #360, commit `ec757bd`
- Main branch in workspace: `work/sm90-megamoe-pr360`
- Isolated worktree: `/home/chen/workspace/source_code/DeepGEMM/work/codex-pr360/src`
- Isolated Python/CUDA env: `/home/chen/workspace/source_code/DeepGEMM/work/codex-pr360`
- GPU: 8 x NVIDIA L20X, SM90
- Host NVCC: 12.8. DeepGEMM warns that NVCC 12.9+ is preferred, but tests and benchmarks run.

Helper scripts:

- `work/codex-pr360/env.sh`: sets isolated `PYTHONPATH`, `CUDA_HOME`, `LD_LIBRARY_PATH`, `DG_JIT_CACHE_DIR`, and NCCL/NVSHMEM env vars.
- `work/codex-pr360/profile_sm90_megamoe_ncu.sh`: NCU wrapper for one profiled rank plus peer ranks.

## Log Style

This note is meant to teach the optimization process, not just archive numbers.
Each round should stay short and follow this shape:

- Command: exact command or script.
- Guess: what bottleneck I am testing, in plain words.
- Result: only key numbers/counters.
- Lesson: what this teaches about the kernel.
- Next: the next test or code change.

Rule of thumb:

- If a change wins, record why I thought it might win.
- If a change loses, record what bottleneck guess was wrong.
- Avoid long theory unless it changes the next experiment.

## Profiling Pass 1: Kernel Selection Sweep

Goal: establish whether the first optimization target should be the auto routing heuristic or the SM90 kernel body.

Common benchmark command shape:

```bash
source /home/chen/workspace/source_code/DeepGEMM/work/codex-pr360/env.sh
cd /home/chen/workspace/source_code/DeepGEMM/work/codex-pr360/src
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=<unique-port>
export DG_SM90_MOE_KERNEL=<auto|pingpong|cooperative>
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 1 2 4 8 16 32 64 128 256 512 1024 \
  --num-tests 5
```

Auto routing details from source:

- `fp8_mega_moe` uses pingpong when `num_tokens < DG_SM90_MOE_COOPERATIVE_THRESHOLD`.
- Default threshold is `256`.
- Cooperative starts at `num_tokens >= 256`.
- L2 N-major scheduling starts when `tokens_per_expert >= 256`, so for default shape it first turns on at `num_tokens=1024`.

Results:

| tokens | auto us | pingpong us | cooperative us | current auto choice |
| ---: | ---: | ---: | ---: | --- |
| 1 | 148.7 | 148.8 | 314.0 | pingpong |
| 2 | 216.2 | 216.3 | 271.6 | pingpong |
| 4 | 308.1 | 306.9 | 399.1 | pingpong |
| 8 | 372.2 | 373.6 | 474.7 | pingpong |
| 16 | ~397 | 397.3 | 503.3 | pingpong |
| 32 | ~402 | 402.5 | 515.5 | pingpong |
| 64 | 408.2 | 408.2 | 525.1 | pingpong |
| 128 | 417.6 | 420.2 | 536.6 | pingpong |
| 256 | 555.7 | 612.4 | 555.7 | cooperative |
| 512 | 816.9 | 883.9 | 817.6 | cooperative |
| 1024 | 1310.0 | 1378.3 | 1308.3 | cooperative + N-major |

Analysis:

- The default `256` cooperative threshold looks reasonable for the default shape.
- Pingpong is clearly better through `128` tokens.
- Cooperative becomes better at `256+` tokens.
- This pass does not expose an easy threshold-only optimization for the default shape.
- Next useful step is NCU on representative points:
  - `tokens=128`, pingpong: low-latency regime just before switch.
  - `tokens=256`, cooperative: first cooperative regime.
  - `tokens=1024`, cooperative + N-major: large-token throughput regime.

Feedback / next action:

- Do kernel-level NCU profiling on the three representative points above.
- Use the NCU results to decide whether the bottleneck is dispatch/barrier, TMA/global memory, scheduler stalls, local memory, or math pipe utilization.

## Correction: NCU Wrapper

Command:

```bash
MASTER_PORT=29141 DG_SM90_MOE_KERNEL=pingpong \
  work/codex-pr360/profile_sm90_megamoe_ncu.sh \
  --output work/codex-pr360/profiles/ncu-t128-pingpong \
  --batch 128 --kernel pingpong --master-port 29141
```

Result:

- Warmup OK: `tokens=128`, pingpong, about `421 us`.
- NCU failed: current NCU does not support `--lockstep-kernel-launch`.
- Peer ranks stayed alive; killed only this failed run's processes.

Why this point:

- Existing script assumed another NCU CLI.
- Profiling result is invalid until the launcher is fixed.

Fix:

- Changed wrapper to use `ncu --target-processes all` around the 8-rank spawn path.
- Removed unsupported lockstep/communicator args.

Next:

- Rerun `tokens=128` pingpong NCU with the fixed wrapper.

## Correction: Full NCU Replay Too Heavy

Command:

```bash
MASTER_PORT=29142 DG_SM90_MOE_KERNEL=pingpong \
  work/codex-pr360/profile_sm90_megamoe_ncu.sh \
  --output work/codex-pr360/profiles/ncu-t128-pingpong \
  --batch 128 --kernel pingpong --master-port 29142
```

Result:

- Warmup OK: `tokens=128`, pingpong, about `422 us`.
- NCU found `sm90_fp8_mega_moe_pingpong_impl`.
- Application replay stayed on pass 1 too long; stopped it.
- No leftover processes.

Why this point:

- Full sections + application replay are too heavy for the first signal on this multi-rank kernel.

Next:

- Use lighter profiling first: benchmark timing, config toggles, and only small NCU sections if needed.

## Profiling Pass 2: N-major Toggle

Command:

```bash
DG_SM90_MOE_KERNEL=cooperative DG_SM90_MOE_NMAJOR=<0|1> \
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 256 512 1024 --num-tests 7
```

Result:

| tokens | N-major off us | N-major on us | take |
| ---: | ---: | ---: | --- |
| 256 | 554.5 | 559.1 | off slightly better |
| 512 | 818.8 | 819.6 | same |
| 1024 | 1313.8 | 1317.2 | off slightly better in this run |

Why this point:

- Auto only enables N-major at 1024.
- HBM GB/s drops at 512+, so weight scheduling was a plausible quick win.

Next:

- Do not change N-major heuristic yet.
- Test `num_experts_per_wave`; it controls wave balance and tail work.

## Profiling Pass 3: Pingpong Experts Per Wave

Command:

```bash
DG_SM90_MOE_KERNEL=pingpong DG_SM90_MOE_EXPERTS_PER_WAVE=<auto|4|8|16|32> \
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 64 128 --num-tests 7
```

Result:

| wave | 64 tok us | 128 tok us | take |
| ---: | ---: | ---: | --- |
| auto | 410.2 | 417.8 | current default |
| 4 | 414.2 | 420.5 | worse |
| 8 | 401.0 | 408.0 | best |
| 16 | 408.0 | 414.4 | close to auto |
| 32 | 411.3 | 421.3 | worse |

Why this point:

- Threshold and N-major did not give a win.
- `num_experts_per_wave` controls how much expert work is grouped per wave; too many waves add overhead, too few hurt balance/cache.
- The shared heuristic picked `16` here; `8` may reduce per-wave tail/pressure for pingpong.

Next:

- Test `wave=8` across the whole pingpong range: `1 2 4 8 16 32 64 128`.
- Only then change the default heuristic.

## Profiling Pass 4: Wave=8 Full Pingpong Range

Command:

```bash
DG_SM90_MOE_KERNEL=pingpong DG_SM90_MOE_EXPERTS_PER_WAVE=<auto|8> \
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 1 2 4 8 16 32 64 128 --num-tests 9
```

Result:

| tokens | auto us | wave=8 us | take |
| ---: | ---: | ---: | --- |
| 1 | 150.5 | 166.5 | worse |
| 2 | 215.3 | 230.0 | worse |
| 4 | 310.3 | 310.7 | same |
| 8 | 370.6 | 365.1 | better |
| 16 | 395.4 | 388.9 | better |
| 32 | 403.2 | 395.1 | better |
| 64 | 407.7 | 400.9 | better |
| 128 | 420.9 | 408.5 | better |

Why this point:

- Pass 3 showed `wave=8` helps at 64/128.
- Need to avoid hurting tiny-token latency.

Next:

- Tune pingpong only: use `min(auto, 8)` for `num_tokens >= 8`.
- Keep tiny tokens on old auto.
- Rebuild and retest correctness + perf.

## Optimization Highlight

Compared with upstream PR #360 baseline, Optimization 1 improves the default auto path for `8-128` tokens, while leaving `1-4` and `256+` effectively unchanged.

Measured speedup on the confirmation sweep:

| tokens | before us | after us | speedup |
| ---: | ---: | ---: | ---: |
| 8 | 370.6 | 365.9 | 1.28% |
| 16 | 395.4 | 388.7 | 1.72% |
| 32 | 403.2 | 395.2 | 2.02% |
| 64 | 407.7 | 401.3 | 1.59% |
| 128 | 420.9 | 409.8 | 2.71% |

Average speedup over `8-128`: about `1.86%`.

How it was found:

1. Auto vs pingpong/cooperative showed the 256 switch point was already reasonable.
2. N-major on/off showed no real win.
3. `num_experts_per_wave` sweep showed `wave=8` beat auto at 64/128.
4. Full pingpong-range sweep showed `wave=8` hurts 1/2 tokens but helps 8-128.
5. Final change: only cap pingpong wave to 8 when `num_tokens >= 8`.

This is useful because the gain comes from host heuristic tuning, not kernel-body risk. It is a good base for further profiling.

## Optimization 1: Pingpong Wave Cap

Change:

- File: `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp`
- Pingpong only: when `num_tokens >= 8`, use `min(auto_wave, 8)`.
- Tiny tokens keep old auto wave.
- Added `DG_SM90_MOE_EXPERTS_PER_WAVE` override for future tuning.

Config check:

| tokens | wave after change |
| ---: | ---: |
| 1 | 32 |
| 8 | 8 |
| 128 | 8 |

Correctness:

- `tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast`
- Result: `PASSED all 32 scenarios`, all `diff=0.0000`.

Perf after change, default auto:

| tokens | before us | after us | take |
| ---: | ---: | ---: | --- |
| 1 | 150.5 | 149.6 | same |
| 2 | 215.3 | 217.6 | same/noise |
| 4 | 310.3 | 310.4 | same |
| 8 | 370.6 | 365.9 | +1.3% |
| 16 | 395.4 | 388.7 | +1.7% |
| 32 | 403.2 | 395.2 | +2.0% |
| 64 | 407.7 | 401.3 | +1.6% |
| 128 | 420.9 | 409.8 | +2.6% |
| 256 | 555.7 | 557.2 | same |
| 512 | 816.9 | 817.9 | same |
| 1024 | 1310.0 | 1314.2 | same |

Why this optimization:

- Threshold and N-major were not wins.
- Wave sweep showed `wave=8` improves 8-128 tokens but hurts 1-2 tokens.
- So the fix is conditional, not a global wave=8.

Next:

- Check cooperative wave tuning separately.

## Validation Plan: Repeat A/B for Optimization 1

Concern:

- `1%-3%` is small enough to be noise-sensitive.

Plan:

- Use the optimized build for both sides.
- `old`: set `DG_SM90_MOE_EXPERTS_PER_WAVE=16` to mimic PR #360 pingpong wave for 8-128.
- `new`: unset override, using tuned default.
- Run multiple alternating rounds on `8 16 32 64 128`.
- Compare averages.

## Bigger Ideas Beyond Heuristics

Heuristic tuning is the low-risk first step. It proves PR #360 has headroom, but it is not the main path to fully squeeze performance.

Non-trivial next ideas:

1. Add lightweight device-side phase timers. Split time into dispatch, math loop, L1 epilogue, L2 epilogue, combine/reduce. This tells us where to cut.
2. Reduce dispatch/barrier overhead for small tokens. 1-16 token cases are latency dominated; fewer waves or fewer global barriers may matter more than math throughput.
3. Rework pingpong epilogue pressure. The wave=8 win suggests the old wave shape may create scheduler/register/barrier pressure, not pure math shortage.
4. Cooperative weight reuse/TMA multicast. For large tokens, cooperative is DRAM/weight-reuse sensitive; cluster/TMA multicast could be a real win, but needs correct cross-CTA amax/SF handling.
5. L2 scheduling by real routed tokens, not only expected tokens. Current heuristic uses expected distribution; actual topk imbalance may choose bad waves.

Current direction:

- Finish repeated A/B to prove Optimization 1 is real.
- Then add phase timers and use them to pick the first kernel-body change.

## Validation Result: 5-round A/B for Optimization 1

Command:

```bash
old: DG_SM90_MOE_EXPERTS_PER_WAVE=16
new: unset DG_SM90_MOE_EXPERTS_PER_WAVE
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 8 16 32 64 128 --num-tests 9
```

Result, 5 alternating rounds:

| tokens | old avg us | new avg us | speedup |
| ---: | ---: | ---: | ---: |
| 8 | 372.84 | 366.18 | 1.82% |
| 16 | 396.66 | 390.72 | 1.52% |
| 32 | 402.98 | 396.60 | 1.61% |
| 64 | 407.86 | 401.56 | 1.57% |
| 128 | 416.72 | 411.14 | 1.36% |

Average speedup: `1.57%`.

Take:

- The gain is real across repeated runs.
- It is small; keep it, but bigger work must come from kernel internals.

## Candidate 2: Optional Stats Counter Overhead

Observation:

- Python wrappers default `cumulative_local_expert_recv_stats=None`.
- The SM90 bench always passes a stats tensor.
- Kernel then does `red_add(cumulative_local_expert_recv_stats + i, num_recv_tokens)` once per local expert.
- Normal perf path does not use this tensor unless profiling/debugging.

Why this matters:

- This is real kernel work: extra atomic/global writes in dispatch cleanup.
- If benchmark always passes it, profiling includes optional stats overhead, not pure fast path.

Test:

- Change bench to pass `None` unless phase profiling is explicitly enabled.
- Measure default performance again.

## Candidate 2 Result: Optional Stats Counter

Change tested:

- Bench passed `None` for `cumulative_local_expert_recv_stats` unless phase profiling is enabled.

Result:

- No meaningful speedup. Example after no-stats: `8=364.4us`, `128=409.7us`, `1024=1312.0us`, roughly same as optimized build with stats.

Take:

- Counter overhead is not the main bottleneck.
- Keep this as profiling hygiene only, not a performance win.

## Tool Result: Lightweight NCU Still Hangs

Command:

```bash
ncu --target-processes all \
  --kernel-name regex:^sm90_fp8_mega_moe_.* \
  --launch-count 8 \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python3 tests/bench_mega_moe_sm90.py --num-processes 8 --ncu-profile-only --batches 128
```

Result:

- NCU attaches to all 8 ranks and finds `sm90_fp8_mega_moe_pingpong_impl`.
- It hangs at profiling 0%; stopped manually.
- No leftover processes.

Take:

- Current NCU CLI/replay mode is not reliable for this multi-rank in-kernel-barrier workload.
- Next profiling should use in-kernel lightweight timers or code-structured A/B, not more NCU retries.

## Candidate 3: Fold Topk Weight Into Quant Scale

Observation:

- L1 epilogue computes `swiglu[]`, finds amax, then loops over `swiglu[] *= topk_weight`.
- Later quantization loops over the same `swiglu[]` and multiplies by `sf_inv`.

Idea:

- Keep amax logic unchanged: `amax *= abs(weight)`.
- Replace two-step multiply with `quant_scale = weight * sf_inv`.
- Quantize as `swiglu * quant_scale`.
- This removes the separate loop that mutates `swiglu[]`.

Why this is non-trivial:

- It changes actual kernel L1 epilogue instruction/register pressure, not host heuristic.
- Applies to both pingpong and cooperative.

Next:

- Run correctness with a fresh JIT cache.
- Then benchmark 8-128 and 256+ separately.

## Roofline / Theoretical Limit Plan

Add a ceiling model to every serious perf table.

For each case:

```text
compute_lower_bound = FLOPs / peak_fp8_tensor_tflops
hbm_lower_bound     = HBM_bytes / peak_hbm_GBps
comm_lower_bound    = cross_rank_bytes / peak_link_GBps
latency_lower_bound = launch + grid/NVLink barrier + unavoidable dispatch floor
theory_lower_bound  = max(all lower bounds)
headroom            = measured_time / theory_lower_bound
```

Table columns to add:

| tokens | measured us | compute lb | HBM lb | comm lb | latency lb | tightest | headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |

Notes:

- Compute and HBM estimates can use the benchmark's existing FLOPs/bytes model first.
- Peak FP8/HBM/link numbers must be either measured locally or written as assumptions. Do not hide assumptions.
- If the tightest lower bound is far below measured time, the missing gap is likely scheduler/barrier/epilogue/register pressure, not pure math or HBM.

## Candidate 3 Result: Fold Topk Weight Into Quant Scale

Correctness:

- Fresh JIT cache.
- `tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast`
- Result: `PASSED all 32 scenarios`.

Perf A/B, 3 alternating rounds, large-token cooperative range:

| tokens | old avg us | new avg us | speedup |
| ---: | ---: | ---: | ---: |
| 256 | 549.47 | 550.10 | -0.12% |
| 512 | 808.40 | 810.03 | -0.20% |
| 1024 | 1297.30 | 1295.43 | 0.14% |

Average: `-0.06%`.

Take:

- The algebra is correct, but performance gain does not survive A/B.
- Do not keep this as an optimization. Revert it.

Why no improvement:

1. The removed loop is small: `kNumPairs=8`, so only a handful of FP32 multiplies per row fragment.
2. The L1 epilogue also has SiLU `exp`, amax reduction, FP8 conversion, TMA store, and arrival signaling. Those likely dominate this local multiply.
3. The compiler may already schedule the original multiply loop into idle slots, so removing it does not reduce the critical path.
4. Large-token cases are mostly cooperative/L2/weight-reuse dominated; L1 quant math is not the tightest bound there.

Conclusion:

- This points us away from tiny algebraic cleanups. Next kernel-body work should target phase-level bottlenecks: dispatch/barrier/TMA/combine, or cooperative weight reuse.

## Bottleneck Hypotheses and Cases

Do not guess blindly. Use cases that separate bottlenecks.

| hypothesis | expected symptom | validation case |
| --- | --- | --- |
| HBM / weight traffic bound | time tracks touched experts / estimated HBM bytes | vary `num_experts` and `masked_ratio` |
| compute bound | time tracks FLOPs when H/IH changes | vary `hidden` / `intermediate_hidden` |
| dispatch/barrier latency bound | time flat even when FLOPs/routed tokens shrink | compare tiny tokens and high `masked_ratio` |
| topk/combine bound | time grows with topk beyond math bytes | vary `num_topk` |
| L1 epilogue / activation bound | `fast_math=0/1` or activation shape changes matter | vary `fast_math`, keep routing fixed |

Next cases:

1. `masked_ratio`: 0.0 vs 0.7 at tokens 128/512.
2. `num_experts`: 128 vs 256 at tokens 128/512.
3. `topk`: 1/2/4/8 at tokens 128.
4. `hidden/IH`: smaller H/IH at tokens 128/512.

## Bottleneck Validation: Designed Cases

Command log:

- `work/codex-pr360/profiles/bottleneck-cases-20260623-020018.log`

Guess:

- If runtime follows `recv` tokens, the bottleneck is mostly math/activation work.
- If runtime follows `experts`, the bottleneck is per-expert scheduling, weight traffic, or workspace/barrier cleanup.
- If runtime barely moves when useful work shrinks, the bottleneck is fixed dispatch/barrier/combine cost.

Result:

| case | tokens | recv | touched experts | time us | lesson |
| --- | ---: | ---: | ---: | ---: | --- |
| mask 0.0 | 128 | 1022 | 32 | 406.3 | baseline |
| mask 0.7 | 128 | 318 | 32 | 396.3 | `recv` -69%, time only -2.5%: fixed/per-expert cost dominates small batch |
| mask 0.0 | 512 | 4113 | 32 | 803.2 | baseline |
| mask 0.7 | 512 | 1246 | 32 | 534.8 | bigger batch benefits from less real work: math/traffic matters here |
| experts 128 | 128 | 988 | 16 | 322.2 | fewer local experts is much faster |
| experts 256 | 128 | 1022 | 32 | 406.5 | default |
| experts 512 | 128 | 1102 | 64 | 749.6 | many experts hurts badly: wave/workspace/weight working set pressure |
| topk 1 | 128 | 129 | 31 | 383.6 | almost all experts still touched, so time stays high |
| topk 8 | 128 | 1022 | 32 | 411.5 | 8x recv only costs +7.3% at 128 tokens |
| default shape | 512 | 4113 | 32 | 802.7 | default H/IH |
| half H/IH | 512 | 4042 | 32 | 268.0 | matrix size still matters a lot once work is large |

Lesson:

- For `8-128` tokens, do not think only in FLOPs. Most time is fixed path: dispatch counts, per-expert waves, barriers, combine, and touching many experts.
- For `512+` tokens, matrix size and weight/GEMM traffic matter strongly.
- This explains why Optimization 1 is small but real: changing wave shape trims fixed per-wave pressure, but it does not remove the deeper fixed path.

Next:

- Small batch: inspect/measure dispatch + combine + per-expert cleanup before doing more math micro-optimizations.
- Large batch: focus on cooperative scheduling, weight reuse, and whether L2/SM utilization is below the H200-style bound.

## Theory Bound: H200-style Lower Bound

Assumption:

- User confirmed the L20X parameter set should be treated the same as H200 here.
- I use `1979 TFLOPS` FP8 tensor peak and `4800 GB/s` HBM as the simple device bounds.
- Latency floor is taken from the best tiny-token measurement: about `149 us`.
- This is a lower bound, not a prediction. If measured time is much higher, the missing part is scheduler/barrier/communication/epilogue overhead or poor utilization.

| tokens | measured us | compute lb us | HBM lb us | latency lb us | tightest | headroom |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 149.6 | 0.3 | 64.3 | 149.0 | latency 149.0 us | 1.00x |
| 8 | 365.9 | 2.8 | 238.9 | 149.0 | HBM 238.9 us | 1.53x |
| 128 | 409.8 | 43.9 | 298.9 | 149.0 | HBM 298.9 us | 1.37x |
| 256 | 557.2 | 92.3 | 304.6 | 149.0 | HBM 304.6 us | 1.83x |
| 512 | 817.9 | 182.1 | 315.4 | 149.0 | HBM 315.4 us | 2.59x |
| 1024 | 1314.2 | 364.0 | 337.3 | 149.0 | compute 364.0 us | 3.61x |

How to read this:

- `1 token`: already at the measured latency floor; little room unless we reduce fixed kernel path.
- `8-128 tokens`: the benchmark byte model says HBM is the tightest simple bound, but measured is still `1.37x-1.53x` above it. That gap is where scheduling/barrier/combine work hides.
- `512-1024 tokens`: headroom grows to `2.6x-3.6x`, so the large-batch path is not just raw HBM bandwidth. Cooperative scheduling, weight reuse, and math-pipe utilization need direct measurement.

Important correction:

- The failed topk-weight-fold optimization targeted a few FP32 multiplies in L1 epilogue. These tables explain why it did not move performance: that work is not the tightest bound.

## Optimization 2: Cooperative Large-token Wave

Why I looked here:

- The theory table showed `512-1024` tokens still have large headroom.
- Bottleneck cases showed large batches care about matrix/weight work, not only fixed dispatch.
- Auto config print showed cooperative uses `wave=8` at `1024` tokens on the default 32-local-expert shape.

Probe:

```bash
DG_SM90_MOE_KERNEL=cooperative \
DG_SM90_MOE_EXPERTS_PER_WAVE=<auto|4|8|16|32> \
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 256 512 1024 --num-tests 7
```

First signal:

| wave | 256 us | 512 us | 1024 us | lesson |
| ---: | ---: | ---: | ---: | --- |
| auto | 555.0 | 815.5 | 1305.4 | baseline |
| 4 | 571.6 | 844.0 | 1361.0 | too many small waves hurts |
| 8 | 555.3 | 834.7 | 1312.3 | OK at 256, worse at 512 |
| 16 | 556.2 | 815.6 | 1289.2 | close to auto, faster at 1024 in this run |
| 32 | 557.8 | 815.1 | 1277.4 | best 1024 in this run |

Guess:

- At `1024` tokens, each expert has enough blocks that splitting the 32 local experts into smaller waves creates extra wave transition/barrier/scheduler pressure.
- One wave over the whole local expert set is better for this default H200/L20X shape.

Correction:

- My first A/B launcher had a bad bash port expression and stopped after the first old run. Fixed the script and reran.
- A config-print run made one 512-token timing huge due JIT/printing. I used it only to confirm config, not as perf data.

Final A/B after rebuilding:

- `old`: force `DG_SM90_MOE_EXPERTS_PER_WAVE=8`, matching old auto at 1024.
- `new`: unset override, using the new default heuristic.
- 5 alternating rounds, `--batches 1024 --num-tests 9`.

| tokens | old avg us | new avg us | speedup |
| ---: | ---: | ---: | ---: |
| 1024 | 1325.62 | 1310.68 | 1.14% |

Change kept:

- File: `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp`
- Cooperative only.
- If `num_tokens >= 1024` and `num_experts_per_rank == 32`, use `num_experts_per_wave=32`.
- This is deliberately narrow: it matches the measured H200/L20X default topology and avoids guessing for untested expert counts.

Lesson:

- This is still a heuristic optimization, but it came from a bottleneck hypothesis: large-token cooperative had scheduler/weight-reuse headroom.
- The win is small but stable. Bigger wins likely need a real kernel feature: phase timers first, then weight-reuse/scheduler changes.

## Final Result After Two Kept Optimizations

Correctness:

- `tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast`
- Result: `PASSED all 32 scenarios`, all `diff=0.0000`.

Final default sweep:

- Log: `work/codex-pr360/profiles/final-default-20260623-022818.log`
- Command: default auto path, `--batches 1 2 4 8 16 32 64 128 256 512 1024 --num-tests 9`.

| tokens | PR360 us | final us | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 148.7 | 148.4 | 0.20% |
| 2 | 216.2 | 216.0 | 0.09% |
| 4 | 308.1 | 306.3 | 0.59% |
| 8 | 372.2 | 367.5 | 1.28% |
| 16 | 397.3 | 388.0 | 2.40% |
| 32 | 402.5 | 396.3 | 1.56% |
| 64 | 408.2 | 402.1 | 1.52% |
| 128 | 417.6 | 407.2 | 2.55% |
| 256 | 555.7 | 555.5 | 0.04% |
| 512 | 816.9 | 816.7 | 0.02% |
| 1024 | 1310.0 | 1267.7 | 3.34% |

What to learn from this:

- Optimization 1 helps the small/medium pingpong band: `8-128` tokens.
- Optimization 2 helps the large cooperative point: `1024` tokens.
- `256/512` did not move; that is useful information. The next real work should target cooperative internals, not more host thresholds.

## Long-sequence Exploration Start

Why `4096` is not enough:

- For this MegaMoE shape, `4096` is only a transition point. It can show cooperative behavior, but not enough to expose true long-sequence scaling.
- Long sequence should test whether the kernel becomes compute-bound, HBM/weight-traffic-bound, or scheduler/arrival-mask-bound when each expert has many M blocks.

Plan:

- Probe `8192 / 16384 / 32768 / 65536` tokens first.
- Do not start with `131072`: the symmetric buffer alone is too close to full-card memory.
- Use low repeat count first to find stable capacity, then repeat interesting points.

Symmetric buffer size from `_C.get_symm_buffer_size_for_mega_moe`, default H200/L20X shape:

| tokens | symm buffer GiB | take |
| ---: | ---: | --- |
| 8192 | 8.34 | safe |
| 16384 | 16.59 | safe |
| 32768 | 33.10 | safe |
| 65536 | 66.11 | likely safe with 143 GiB free |
| 131072 | 132.14 | too close; skip until needed |

Teaching point:

- Before profiling long sequence, first check memory formula. Otherwise an OOM tells you nothing about performance.
- Here `65536` is the practical first upper point; `131072` would mix kernel behavior with memory pressure.

### Long-sequence Correction: One Length Per Process

Mistake:

```bash
python3 tests/bench_mega_moe_sm90.py \
  --batches 8192 16384 32768 65536 --num-tests 3
```

Result:

- `8192` ran: `8626.0 us`, `665.3 TFLOPS`.
- Then the process hit CUDA OOM while moving to the next length.

Why:

- The benchmark sets `num_max_tokens_per_rank = max(batches)`.
- So even the `8192` case allocated the `65536` symmetric buffer.
- In the same spawned process, allocator/symmetric-memory cleanup did not return enough memory before the next config allocation.

Fix:

- Run each long length in a separate Python process.
- This is also cleaner: `num_max_tokens_per_rank` then matches the tested length.

Teaching point:

- For long sequence, benchmark harness behavior matters. If capacity is wrong, OOM is not a kernel result.

### Long-sequence Baseline: Single Process Per Length

Command log:

- `work/codex-pr360/profiles/longseq-single-20260623-024054.log`

Command shape:

```bash
for b in 8192 16384 32768 65536; do
  python3 tests/bench_mega_moe_sm90.py     --num-processes 8 --batches $b --num-tests 3
done
```

Result:

| tokens | recv | time us | TFLOPS | model GB/s | compute lb us | headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 65155 | 8538.0 | 672.2 | 360 | 2900.1 | 2.94x |
| 16384 | 130971 | 17474.3 | 660.2 | 273 | 5829.5 | 3.00x |
| 32768 | 262357 | 35621.3 | 648.7 | 228 | 11676.4 | 3.05x |
| 65536 | 524339 | 74363.2 | 621.1 | 199 | 23338.5 | 3.19x |

Lesson:

- Runtime scales roughly linearly with token count, but throughput gets worse: `672 TFLOPS` at 8k drops to `621 TFLOPS` at 64k.
- The simple model GB/s also drops, so the benchmark byte model is missing real traffic or stalls.
- A likely reason is repeated weight tile loading and scheduler order: long sequence has many M blocks per expert, so block order decides whether weight tiles are reused or reloaded.

Next test:

- Toggle L2 N-major scheduling on long sequence.
- If N-major matters, then block order / weight reuse is a real long-sequence lever, and L1 scheduling becomes a reasonable feature candidate.

### Why Pingpong Is Slower on Long Sequence

Data:

| tokens | pingpong us | cooperative us | cooperative faster |
| ---: | ---: | ---: | ---: |
| 8192 | 9852.1 | 8752.8 | 12.6% |
| 32768 | 40291.9 | 35451.5 | 13.7% |

Why I think this happens:

1. `BLOCK_M` is smaller in pingpong.
   - Pingpong uses `BLOCK_M=64`.
   - Cooperative uses `BLOCK_M=128` by splitting one tile across two math warpgroups.
   - For long sequence, each expert has many M blocks, so pingpong creates about 2x as many M tiles.

2. Cooperative shares the B/weight tile across two warpgroups.
   - In cooperative, two WGs work on the same `(expert, m128, n128)` tile, each owning 64 rows.
   - The B tile is loaded once and consumed by both WGs.
   - In pingpong, the two WGs work on different `m64` tiles, so the same B tile is effectively loaded for each M slice.
   - Long sequence has many M slices, so this repeated weight traffic becomes expensive.

3. More tiles also means more scheduler/epilogue/arrival work.
   - More L1 tiles: more SwiGLU + FP8 stores + L2 arrival-mask updates.
   - More L2 tiles: more BF16 scatter work and tile-boundary synchronization.
   - Combine is mostly the same for both kernels, so the slowdown is likely before combine.

Concrete scale example at 32768 tokens:

- Rank0 received `262357` routed rows, about `8199` rows per local expert.
- M blocks per expert:
  - pingpong: `ceil(8199 / 64) = 129`
  - cooperative: `ceil(8199 / 128) = 65`
- N blocks per expert: `32` for L1 and `56` for L2.
- Rough tile count per expert:
  - pingpong: `129 * (32 + 56) = 11352`
  - cooperative: `65 * (32 + 56) = 5720`

Lesson:

- Pingpong is a latency/overlap design: good when there are not many M blocks and epilogue overlap matters.
- Long sequence is a throughput/reuse problem: cooperative wins because it reduces tile count and reuses B/weight tiles better.
- So for long sequence, optimizing pingpong is probably the wrong target. The useful target is cooperative: reduce per-tile overhead, improve weight reuse, or reduce combine/scatter cost.

### Correction: Does PR360 Use Pingpong for Long Sequence?

Important correction:

- PR360 default auto routing does **not** use pingpong for long sequence.
- The default threshold is `256` tokens:
  - `<256`: pingpong
  - `>=256`: cooperative

So the right conclusion is:

- Pingpong has a real scalability weakness for long sequence.
- But PR360 mostly avoids that weakness through the cooperative threshold.
- If someone forces pingpong, or if the threshold is moved too high, long-sequence performance drops about `12%-14%` in my tests.

Better wording for the PR360 critique:

- Not: "PR360 uses pingpong for long sequence."
- Yes: "PR360's pingpong design is not suitable for long sequence; the long-sequence path must stay cooperative, and further optimization should target cooperative internals."

This matters because a wrong diagnosis would push us to optimize the wrong kernel.

## Optimization Scope Going Forward

User intent:

- The target is not just to run PR360 or tune a few thresholds.
- The target is to make the SM90 MegaMoE kernel better than PR360 where possible.
- Long sequence is the hard case and should be treated as a main optimization target.

Directions to actively consider:

1. Length-specific strategy.
   - Small/medium tokens and long tokens should not be forced into one heuristic story.
   - Long sequence needs its own cooperative-path analysis.

2. Cooperative kernel internals.
   - Scheduler order, per-tile overhead, L1/L2 epilogue, scatter, and combine need phase evidence.
   - Do not optimize pingpong for long sequence unless data contradicts the current result.

3. Quantization / scale representation.
   - Current SM90 path uses FP8 e4m3 with float scale factors.
   - Potential experiments: different scale granularity, cheaper scale path, or reducing scale traffic.
   - Any quantization change must pass correctness first, then A/B perf.

4. Failed ideas are still useful.
   - If an idea does not improve performance, record which bottleneck guess was wrong.

Rule:

- Prefer feature experiments that test a bottleneck hypothesis over small parameter sweeps.

### Phase Profiling: Coarse Then Tile-Level

I first made a mistake while testing the new phase profiler: running the main repo's `tests/bench_mega_moe_sm90.py` imported the source `deep_gemm/` package instead of the isolated installed wheel, so `_C.so` was missing. The fix was to run the isolated script under `work/codex-pr360/src/tests/...`, whose `_C.so` points to `work/codex-pr360/site`.

Coarse phase data said the long path is not combine-bound:

| tokens | dispatch_pull cycles | math_loop cycles | combine_reduce cycles |
| ---: | ---: | ---: | ---: |
| 8192 | 2.91M | 12.30M | 0.54M |
| 32768 | 10.98M | 48.02M | 1.78M |

Then I split `math_loop` into representative L1/L2 tile timing:

| tokens | L1 tiles | L1 avg cycles | L2 tiles | L2 avg cycles |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 128 | 55.9k | 222 | 21.6k |
| 32768 | 501 | 55.4k | 876 | 21.9k |

Lesson: L1 is heavier per tile because it has the long `K=7168` GEMM plus SwiGLU/FP8 quant. Dispatch pull is still large enough to optimize, especially for long sequence.

### Failed Feature: Fuse Topk Weight Into L1 SwiGLU

Hypothesis: L1 epilogue computes `silu(g) * u`, stores it, then runs a second loop to multiply topk weight. I tried fusing the weight into the first SwiGLU loop to remove that second pass.

A/B result:

| tokens | old us | fused us | result |
| ---: | ---: | ---: | ---: |
| 8192 | 8585.6 | 8665.5 | -0.93% |
| 32768 | 35417.5 | 35422.2 | ~0.00% |

Phase check at 8192 showed why: L1 tile avg went from `55.7k` to `56.5k` cycles. The removed multiply loop was not the bottleneck; moving weight earlier added pressure in the hot epilogue path. I left this behind an env switch, default off: `DG_SM90_MOE_FUSE_TOPK_WEIGHT=0`.

### Winning Feature: Fast Rank Select In Dispatch Pull

Hypothesis: dispatch pull spends time mapping each received token back to `(source rank, token index)` with the full round-robin algorithm. For `kNumRanks <= 32`, most tokens are in the first balanced round of each expert. So I precompute per-expert active-rank mask, active count, and first-round length, then select rank in O(1) for that common case. Tail tokens still use the original algorithm, so the mapping is exact.

A/B result:

| tokens | old us | fast_rank us | improvement |
| ---: | ---: | ---: | ---: |
| 8192 | 8730.9 | 8584.6 | +1.70% |
| 32768 | 35478.0 | 34964.9 | +1.47% |
| 65536 | 74780.0 | 73683.8 | +1.49% |

Short/medium sweep sanity:

| tokens | old us | fast_rank us | note |
| ---: | ---: | ---: | --- |
| 256 | 557.3 | 548.8 | +1.55%, cooperative path |
| 512 | 818.7 | 810.4 | +1.02%, cooperative path |
| 1024 | 1265.6 | 1248.9 | +1.34%, cooperative path |

For `<256` tokens, auto uses pingpong, so the small differences there are measurement noise, not this feature.

One caution: the phase profiler is useful for finding large bottlenecks, but a single phase-profile run did not reliably show `dispatch_pull` getting smaller for this feature. I treat the repeated end-to-end A/B above as the stronger evidence here.

Correctness:

- `tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast --num-tests 1`
- Passed all 32 scenarios, all reported `diff=0.0000`.

### Rough Theory Limit Check

This is a teaching estimate, not a precise hardware proof. I used two simple lower bounds:

1. Compute lower bound: FP8 work divided by H200/L20X-class dense FP8 peak.
2. HBM lower bound: the benchmark's simple byte model divided by about `4.8 TB/s` HBM bandwidth.

The stricter one is compute in all long cases:

| tokens | measured fast_rank us | compute lb us | HBM-model lb us | stricter lb | headroom |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 8584.6 | 2900.1 | ~641 | compute | 2.96x |
| 32768 | 34964.9 | 11676.4 | ~1695 | compute | 2.99x |
| 65536 | 73683.8 | 23338.5 | ~3093 | compute | 3.16x |

Interpretation:

- The simple HBM byte model is not the tightest limit.
- The gap to compute lower bound is still about `3x`, so the missing performance is likely from scheduling, synchronization, cross-rank movement, TMA/epilogue overhead, and imperfect tensor-core occupancy.
- The fast-rank feature helps because it removes work from dispatch, but the larger remaining target is still cooperative math-loop efficiency.



### Phase Split: Epilogue Is Not The Main Long-Sequence L1 Bottleneck

I split the tile profiler one level further: `l1_tile_loop/l2_tile_loop` now include the whole tile, and `l1_epilogue/l2_epilogue` isolate the post-WGMMA epilogue part.

| tokens | L1 tile avg | L1 epilogue avg | L2 tile avg | L2 epilogue avg |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 55.6k | 3.4k | 21.7k | 4.9k |
| 32768 | 55.7k | 3.6k | 22.3k | 5.4k |

What this teaches me:

- L1 epilogue is only about `6-7%` of an L1 tile. So the failed topk-weight fusion makes sense: I was optimizing a small tail and added pressure to a hot path.
- L2 epilogue is a bigger fraction of L2 tile, around `23-24%`, but L2 tiles are much cheaper than L1 tiles overall.
- For long sequence, the bigger target is still L1 mainloop/scheduling/imbalance, not just the activation/quant epilogue.

### Failed Feature: L1 N-Major Scheduling

Hypothesis: L1 has very large `K=7168`, and the L1 weight tile is reused across many token blocks. I added an experimental scheduler path `DG_SM90_MOE_L1_NMAJOR=1` that sweeps M for the same L1 N block, similar in spirit to the existing L2 N-major path. The default remains off.

A/B result with fast-rank on and topk fusion off:

| tokens | L1 M-major us | L1 N-major us | result |
| ---: | ---: | ---: | ---: |
| 8192 | 8461.4 | 8428.7 | +0.39% |
| 32768 | 35762.6 | 37765.3 | -5.60% |

The first guess was wrong for the important long case. Phase profile at `32768` showed why:

| mode | time us | math_loop | combine_barrier | L1 tile avg | L2 tile avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 M-major | 34449.3 | 48.42M | 0.16M | 55.7k | 22.0k |
| L1 N-major | 37475.1 | 48.32M | 5.08M | 55.8k | 21.9k |

The single-tile cost did not meaningfully change. The slowdown appeared as a much larger wait before combine. My read: L1 N-major changes the global completion shape and makes long-sequence work finish less evenly across ranks/SMs. So this is not a keeper as an automatic long-sequence policy. It stays as a diagnostic switch only.


### Failed Feature: Direct L2 Scatter From Registers

Hypothesis: L2 epilogue first packs BF16 into `smem_cd_l2`, syncs, then reads `uint4` back for NVLink scatter. Since phase profiling showed L2 epilogue is about `5k cycles` per tile, I tried a temporary local switch, `DG_SM90_MOE_DIRECT_L2_SCATTER=1`: use warp shuffles to gather the four WGMMA col-lanes for each row and scatter `uint4` directly from registers. This removes the shared-memory staging path and its cross-WG hazard barrier.

Correctness smoke passed (`diff=0.0000`), but performance was much worse:

| tokens | default us | direct scatter us | result |
| ---: | ---: | ---: | ---: |
| 8192 | 8438.3 | 10423.9 | -23.5% |
| 32768 | 35496.4 | 42039.6 | -18.4% |

Phase profile at `32768` explains it:

| mode | time us | math_loop | L2 tile avg | L2 epilogue avg | combine_barrier |
| --- | ---: | ---: | ---: | ---: | ---: |
| default SMEM staging | 34624.4 | 48.01M | 21.7k | 5.1k | 1.85M |
| direct register scatter | 42085.0 | 65.29M | 37.0k | 18.3k | 2.12M |

The idea failed because the SMEM path is not just a useless detour. It cheaply rearranges WGMMA's per-lane register layout into contiguous `uint4` chunks, then lets many lanes issue simple vector writes. My direct path paid many shuffle instructions and reduced useful scatter parallelism. Lesson: if I want to improve L2 scatter, I need a better layout-aware scatter pattern, not simply bypass SMEM.


### Default Path Check After Failed Experiments

After adding the diagnostic switches above, I rechecked the normal path with both failed features off:

- `DG_SM90_MOE_L1_NMAJOR=0`
- `DG_SM90_MOE_FAST_RANK_SELECT=1`
- `DG_SM90_MOE_FUSE_TOPK_WEIGHT=0`

Correctness: all 32 scenarios passed, all `diff=0.0000`.

Sanity benchmark:

| tokens | time us | TFLOPS |
| ---: | ---: | ---: |
| 8192 | 8525.4 | 673.1 |
| 32768 | 35647.1 | 648.0 |

This is within the normal run-to-run band of the fast-rank default path. The failed L1 N-major experiment is not enabled by default; the direct-scatter code was not kept after profiling showed it was clearly worse.


## 2026-06-23: L1 weight scale pointer hoist, failed but useful

### Why I tried it
From the phase split, long sequence time is dominated by `math_loop`; L1 tiles are much heavier than L2 tiles, while L1 epilogue is only a small fraction. In the L1 K loop we repeatedly recompute:

- `gate_n = n_block_idx / 2`
- `up_n = kL1SFGateBlks + gate_n`
- `l1_sf_base + n * kL1SFKBlocks + k`

Hypothesis: hoist the gate/up scale base pointers once per tile, then only add `next_k` in the loop. If integer address arithmetic is visible in the hot path, this should help more at 32768/65536 than at short lengths.

### Correction during measurement
The first 8-rank rerun was not clean: `nvidia-smi pmon` showed other jobs on GPUs 0-4, so full 8-GPU timing could not be trusted. I did not kill those jobs. Instead, I used free GPUs 5/6 with `CUDA_VISIBLE_DEVICES=5,6`, `--num-processes 2`, and moved JIT caches to `/tmp` because `/home` was over quota.

### Result
Correctness smoke passed: `L1.smoke diff=0`.

2-rank A/B on GPUs 5/6, same code except the one header:

| batch | clean, repeat=20 | hoist, repeat=20 | delta |
| --- | ---: | ---: | ---: |
| 32768 | 40049.5 us | 40475.0 us | -1.06% |
| 65536 | 76353.8 us | 76950.0 us | -0.78% |

Earlier repeat=5 samples were noisy and even briefly favored hoist, but the higher-repeat long-seq run says the change is not a reliable win.

### Conclusion
I reverted it. The useful lesson is negative: this L1 hot path is not mainly limited by the small integer address calculation for weight scales. The stricter bottleneck is more likely WGMMA/TMA/scale-load scheduling, register pressure, or tail/rank imbalance. This also explains why a seemingly harmless pointer hoist can lose: it may extend pointer live ranges and make scheduling/register allocation slightly worse.


## 2026-06-23: Correction - "WGMMA slow" is not proven yet

I kept saying the bottleneck is probably WGMMA. That needs a sharper wording.

What the current data proves:

- Long sequence time is dominated by the GEMM mainloop.
- L1 tile time is much larger than L2 tile time.
- L1/L2 epilogues are small compared with tile time.
- Removing small address arithmetic did not help.
- Direct L2 scatter made the mainloop and epilogue worse, so layout/register pressure matters.

What this does **not** prove:

- It does not prove the WGMMA instruction itself is slow.
- It does not prove tensor core utilization is low.
- It does not distinguish WGMMA compute limit from scale-load latency, TMA/SMEM wait, register pressure, or scheduler tail imbalance.

Better mental model:

The bottleneck is currently "inside the WGMMA mainloop region", not necessarily "WGMMA instruction throughput". On SM90, accumulator registers, scale values, TMA/SMEM staging, and WGMMA issue all live in the same tight loop. A small feature can lose if it increases live ranges or scheduling pressure.

How to verify next:

Use NCU on one long case, and look for:

- tensor pipe active / tensor pipe utilization: proves whether tensor cores are saturated.
- eligible warps and issue stalls: tells whether warps are ready but not issuing.
- long scoreboard / memory dependency stalls: points to scale/TMA/SMEM latency.
- register count and occupancy: tells whether extra feature code is choking scheduling.
- L1 vs L2 phase counters: keeps the metric tied back to MegaMoE phases.

So the next optimization should not start with "make WGMMA faster". It should start with: identify which part of the mainloop prevents WGMMA from being fed or overlapped.


## 2026-06-23: Expert-local L1->L2 scheduling experiment

### Why I tried it
The scheduler had a TODO about better block swizzle inside expert waves. Current cooperative order is wave-level:

1. run L1 for every expert in the wave;
2. then run L2 for every expert in the wave.

For long sequences, one expert has many M blocks. I tested a different phase order:

1. run one expert's L1 blocks;
2. immediately run that expert's L2 blocks;
3. then advance to the next expert.

The bet: L2 may see fresher L1 output / metadata and shorter tail waiting. The risk: some CTAs may enter L2 early and spin on that expert's L1 arrival mask, wasting work that could have gone to later experts.

### Implementation
Added a default-off switch:

- `DG_SM90_MOE_EXPERT_LOCAL=0`: current wave-level order.
- `DG_SM90_MOE_EXPERT_LOCAL=1`: experimental expert-local L1->L2 order.

This is not a parameter tweak; it changes the persistent scheduler's flattened block stream. The kernel math/epilogue code is unchanged.

### Correctness
2-rank correctness with expert-local enabled:

- L1 smoke: passed, `diff=0`.
- L1-L4 full set: 32/32 passed, all `diff=0`.

### Measurement caveat
The machine was not clean: another python process was attached to all 8 GPUs, and some GPUs had high memory use. I used physical GPUs 0/7 for 2-rank A/B and `/tmp` JIT caches. These numbers are useful for direction, not final 8-rank proof.

### Result
2-rank, `tokens=32768`:

| run | default | expert-local | delta |
| --- | ---: | ---: | ---: |
| repeat=10 | 35969.9 us | 35844.3 us | +0.35% |
| repeat=20 | 37141.7 us | 37415.9 us | -0.74% |
| repeat=10 | 35801.3 us | 35744.4 us | +0.16% |

Average across these noisy runs: default 36304.3 us, expert-local 36334.9 us, about **-0.08%**.

2-rank, `tokens=65536`:

| default | expert-local | delta |
| ---: | ---: | ---: |
| 68912.0 us | 69225.5 us | -0.45% |

Phase profile for `tokens=32768` showed a useful signal:

| metric | default | expert-local |
| --- | ---: | ---: |
| time | 36275.1 us | 35863.6 us |
| math_loop | 52.405M cycles | 52.062M cycles |
| combine_barrier | 2.439M cycles | 0.077M cycles |
| l1_tile avg | 55.5k | 54.7k |
| l2_tile avg | 24.3k | 24.2k |

### Conclusion
Do not enable this by default now. End-to-end long-seq speed did not improve reliably, and 65536 was slightly worse.

The useful lesson: phase order can affect tail/barrier time, not just cache locality. Expert-local reduced `combine_barrier` in one profiled run, but that did not convert into stable wall-clock gain. My current guess is that earlier L2 starts sometimes reduce final tail, but sometimes make CTAs wait on per-pool L1 arrival masks and disturb global load balance.

Next time this is worth revisiting only with a cleaner 8-GPU run and perhaps a less extreme variant: not per-expert L1->L2, but smaller waves or a limited interleave after a group of experts.


## 2026-06-23: NCU correction - mainloop is not WGMMA-throughput bound

### Why I ran NCU
I had been saying "probably WGMMA" too loosely. Phase counters only told me the time sits in the GEMM mainloop region. They did not tell me whether tensor cores are full, whether we are waiting on memory, or whether barriers/tail dominate.

### Profiling correction
Trying NCU on the 2-rank distributed harness with application replay got stuck in replay pass 1. I interrupted only my own NCU process. No report was produced.

Then I switched to a 1-rank long case to remove cross-process replay complexity:

- `num_processes=1`
- `tokens=32768`
- cooperative kernel, default scheduler
- `DG_JIT_WITH_LINEINFO=1`
- report: `/tmp/codex-pr360-ncu-1r-32768/sm90-megamoe-default-1r-32768.ncu-rep`

This is not final 8-rank performance evidence, but it is good enough to identify the kernel's single-SM mainloop behavior.

### Key NCU numbers

| metric | value | what it says |
| --- | ---: | --- |
| Compute / tensor pipe active | 40.31% | tensor pipe is not saturated |
| L2 throughput | 62.21% | L2 path is busier than tensor pipe |
| DRAM throughput | 30.20% | not pure HBM bandwidth bound |
| One or more eligible warps | 36.45% | schedulers often have no ready warp |
| No eligible | 63.55% | strong wait/dependency signal |
| Active warps / scheduler | 2.96 | low, close to launch-limited occupancy |
| Eligible warps / scheduler | 0.53 | not enough ready warps to feed issue slots |
| Issued warp / scheduler | 0.36 | low issue utilization |
| Registers / thread | 168 | occupancy/register pressure is real |

Warp stall top entries:

| stall | value |
| --- | ---: |
| Barrier | 2.63 |
| Long scoreboard | 1.87 |
| Wait | 0.82 |
| Dispatch stall | 0.33 |
| Branch resolving | 0.32 |
| Short scoreboard | 0.16 |
| GMMA | 0.13 |
| MIO throttle | 0.13 |
| Math pipe throttle | 0.09 |

### Conclusion
This corrects my earlier mental model. The kernel is not obviously WGMMA-throughput bound. Tensor pipe active is only ~40%, and GMMA stall is small compared with barrier and long scoreboard stalls.

Better bottleneck guess now:

1. barriers / phase handoff / persistent scheduler ordering;
2. long scoreboard from TMA/global loads, likely A/SFA/B/SF or remote/scatter-dependent data;
3. low eligible warps caused by register + smem limited occupancy;
4. tensor cores are underfed rather than intrinsically slow.

So the next useful optimization should target one of these:

- reduce cross-warpgroup or per-tile barriers;
- make more independent work available while waiting on L1/L2 arrival or TMA;
- reduce register/live-range pressure so active/eligible warps improve;
- reduce long-scoreboard global scale/load dependencies.

This also explains why tiny address hoists did not help: the problem is not a few integer instructions; it is waiting and limited ready work.


### Wave-size correction
The first 1-rank NCU compiled with `num_experts_per_wave=2`, which is not the 8-rank long-seq shape. I reran 1-rank NCU with `DG_SM90_MOE_EXPERTS_PER_WAVE=32` to better match the target cooperative long-seq schedule.

| metric | wave=2 | wave=32 |
| --- | ---: | ---: |
| duration | 32.21 ms | 33.32 ms |
| tensor pipe active | 40.31% | 40.69% |
| L2 throughput | 62.21% | 65.35% |
| eligible warps / scheduler | 0.533 | 0.537 |
| active warps / scheduler | 2.963 | 2.962 |
| issued warp / scheduler | 0.36 | 0.37 |
| registers / thread | 168 | 168 |
| stall barrier | 2.63 | 2.77 |
| stall long scoreboard | 1.87 | 1.73 |
| stall wait | 0.82 | 0.82 |
| stall GMMA | 0.13 | 0.12 |

The correction matters methodologically, but not for the conclusion. Wave=32 still shows underfed tensor cores, low eligible warps, high barrier/scoreboard stalls, and small GMMA stall.


## 2026-06-24: SFB-in-SMEM - useful, but only in the right length band

### Why I tried it
NCU said the math warpgroup is not WGMMA-throughput bound: tensor pipe is only ~40% active, and stalls include barrier + long scoreboard. That made me look for tiny global-load dependencies inside the math loop.

The weight scale factor (SFB) is a good candidate: each GEMM tile needs only 1-2 floats, but the math warps load it from global. I tried moving those loads to the B-loader warp, storing 2 floats per pipeline stage in SMEM, then letting math warps read from SMEM after the full barrier.

### What changed
- Added `DG_SM90_MOE_SFB_SMEM`.
- `1` forces SFB staging in SMEM.
- `0` forces old direct global SFB loads.
- default `-1/auto` enables it only when `tokens_per_expert` is in `[32, 4096]`.

The auto band is important. I first thought this might be a general long-seq win. It is not.

### Correctness
Built isolated wheel/site:

- wheel: `/tmp/codex-pr360-dist-sfb-auto/deep_gemm-2.5.0+local-cp312-cp312-linux_x86_64.whl`
- site: `/tmp/codex-pr360-site-sfb-auto`

Correctness passed:

| mode | test | result |
| --- | --- | --- |
| forced off | L1 smoke | pass, diff=0 |
| forced on | L1 smoke | pass, diff=0 |
| forced on | L2/L3/L4, 31 cases | pass, diff=0 |
| auto default | L1/L2/L3/L4, 32 cases | pass, diff=0 |

### Performance result
2 ranks, H200 shape, `hidden=7168`, `intermediate_hidden=2048`, `experts=256`, `topk=8`, cooperative kernel.

| tokens | forced off | forced on | speedup | decision |
| ---: | ---: | ---: | ---: | --- |
| 8192 | 10178.1 us | 9775.9 us | +4.11% | enable |
| 16384 | 18603.4 us | 18386.2 us | +1.18% | enable |
| 32768 | 35654.4 us avg | 35035.7 us avg | +1.77% | enable |
| 65536 | 72114.7 us avg | 71463.6 us avg | +0.91% | enable |
| 131072 | 146262.0 us avg | 147038.0 us avg | -0.53% | disable |

Auto run confirmed the heuristic:

| tokens | auto `sfb_in_smem` | auto time |
| ---: | ---: | ---: |
| 8192 | 1 | 9786.8 us |
| 16384 | 1 | 18314.0 us |
| 32768 | 1 | 35748.6 us |
| 65536 | 1 | 71349.4 us |
| 131072 | 0 | 146001.3 us |

### What phase profiling taught
Phase profiling is noisy in absolute time, but it shows the direction.

| tokens | metric | off | on | read |
| ---: | --- | ---: | ---: | --- |
| 32768 | `math_loop` | 52.316M | 50.434M | better |
| 32768 | `l1_tile_loop avg` | 55.1k | 50.8k | better |
| 32768 | `l2_tile_loop avg` | 24.5k | 25.2k | slightly worse |
| 131072 | `math_loop` | 194.020M | 187.451M | better |
| 131072 | `l1_tile_loop avg` | 54.5k | 50.6k | better |
| 131072 | `l2_tile_loop avg` | 22.7k | 23.1k | slightly worse |

So the feature does what it was designed to do: L1 tile loop gets cheaper. The miss was assuming that would always win end-to-end.

### Why 131k regressed
This is an Amdahl problem plus producer pressure.

SFB is tiny: only 1-2 floats per tile. Moving it out of math can remove a scoreboard dependency, but it also adds producer-side global loads, SMEM stores, and 128B/stage more shared memory. At 131k, the whole run is dominated by a much larger steady-state math/dispatch body. The saved math-side scale latency is too small, while the producer and barrier side still pay the staging cost.

Practical lesson: a micro-feature can be real and still need a length gate. Here the correct optimization is not "turn it on", but "turn it on only where measurement says the saved dependency is visible".

### 8-rank interval correction
The user pointed out the real use case: each rank usually handles a not-too-long token slice, so I should not overfit to 131072. I first reran one 8-rank H200-like shape and then corrected the test to the PR360 standard model shapes.

First 8-rank H200-like shape:

```bash
DG_SM90_MOE_KERNEL=cooperative DG_SM90_MOE_SFB_SMEM=<0|1> \
python3 /tmp/codex-pr360-tests/bench_mega_moe_sm90.py \
  --num-processes 8 --hidden 7168 --intermediate-hidden 2048 \
  --num-experts 256 --num-topk 8 --num-tests 10 --batches <case>
```

For this shape, `tokens_per_expert = tokens_per_rank / 4`.

| tokens/rank | tokens/expert | off avg | on avg | speedup | read |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 128 | 805.9 us | 799.9 us | +0.76% | weak |
| 1024 | 256 | 1286.0 us | 1256.0 us | +2.38% | useful |
| 2048 | 512 | 2268.3 us | 2202.7 us | +2.98% | useful |
| 4096 | 1024 | 4418.5 us | 4244.5 us | +4.10% | useful |
| 8192 | 2048 | 8736.2 us | 8462.2 us | +3.24% | useful |
| 16384 | 4096 | 17382.0 us | 17281.0 us | +0.58% | small |
| 32768 | 8192 | 35605.4 us | 35692.0 us | -0.24% | disable |

Correction inside this pass: grouped `1024` had one bad on-run (`1447 us`), which made the mean look negative. I reran `1024` alone five times and got stable `1252-1262 us` with SFB-SMEM on. This is why repeated runs matter before changing a heuristic.

### PR360-standard SOTA comparison
The user then asked to align with the main PR test standard. I used the PR360 8-rank token list `16 64 256 512 1024 4096 8192` and four model shapes:

| model | hidden | intermediate | experts | topk |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-V4 Flash | 4096 | 2048 | 256 | 6 |
| DeepSeek-V4 Pro | 7168 | 3072 | 384 | 6 |
| MiMo-V2.5 | 4096 | 2048 | 256 | 8 |
| MiMo-V2.5-Pro | 6144 | 2048 | 384 | 8 |

Here "SOTA off" means the current PR360-style path with direct global SFB loads (`DG_SM90_MOE_SFB_SMEM=0`). "SFB on" means the new staging feature forced on. I repeated the main sweep twice, then reran `256/512/1024` once more because the lower bound changed the heuristic.

| model | tokens/rank | tokens/expert | SOTA off | SFB on | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash | 256 | 48.0 | 326.3 us | 324.3 us | +0.61% |
| Flash | 512 | 96.0 | 345.7 us | 338.0 us | +2.28% |
| Flash | 1024 | 192.0 | 597.0 us | 582.2 us | +2.55% |
| Flash | 4096 | 768.0 | 1807.2 us | 1748.7 us | +3.35% |
| Flash | 8192 | 1536.0 | 3546.7 us | 3412.8 us | +3.92% |
| V4 Pro | 256 | 32.0 | 1051.1 us | 1035.2 us | +1.54% |
| V4 Pro | 512 | 64.0 | 1097.1 us | 1083.0 us | +1.30% |
| V4 Pro | 1024 | 128.0 | 1695.8 us | 1675.6 us | +1.20% |
| V4 Pro | 4096 | 512.0 | 5307.1 us | 5115.4 us | +3.75% |
| V4 Pro | 8192 | 1024.0 | 10677.6 us | 10653.0 us | +0.23% |
| MiMo | 256 | 64.0 | 333.9 us | 327.3 us | +2.01% |
| MiMo | 512 | 128.0 | 491.3 us | 476.1 us | +3.19% |
| MiMo | 1024 | 256.0 | 758.4 us | 734.7 us | +3.23% |
| MiMo | 4096 | 1024.0 | 2362.8 us | 2278.9 us | +3.68% |
| MiMo | 8192 | 2048.0 | 4799.2 us | 4649.6 us | +3.22% |
| MiMo-Pro | 256 | 42.7 | 663.0 us | 652.9 us | +1.56% |
| MiMo-Pro | 512 | 85.3 | 701.1 us | 690.5 us | +1.54% |
| MiMo-Pro | 1024 | 170.7 | 1269.9 us | 1255.0 us | +1.19% |
| MiMo-Pro | 4096 | 682.7 | 3912.9 us | 3908.1 us | +0.13% |
| MiMo-Pro | 8192 | 1365.3 | 7983.5 us | 7905.3 us | +0.99% |

Important correction: my earlier lower bound `tokens_per_expert >= 256` was too conservative for the PR360 standard. The four-model sweep shows that `tokens/rank=256` is already mostly positive, and the lowest useful PR-standard point is V4 Pro with `tokens_per_expert=32`. So the auto lower bound should be `32`, not `256`.

### Current decision
Keep SFB-in-SMEM, but auto-enable only for `tokens_per_expert` in `[32, 4096]`.

Why this range:

- Below PR360's cooperative entry point, there is no strong evidence to pay SFB staging cost.
- From `32` through the common PR360 token range, most standard model cases improve, commonly `1-4%`.
- At `4096`, the win can be small but is still non-negative in the tested standard shapes.
- At `8192`, the earlier long-shape test regressed; the producer-side load/store and extra SMEM traffic can outweigh saving a tiny math-side global scale load.

Baseline note: I did not complete the DeepEP/TileLang unfused baseline table in this pass. The benchmark's `--baseline-version both` path is DeepEP dispatch/combine + TileLang SwiGLU/FP8 + DeepGEMM grouped GEMMs, split into V1 contiguous, V1 low-latency, and V2 ElasticBuffer. I tried a minimal `/tmp` install instead of changing the main environment. TileLang without dependencies needed `apache-tvm-ffi`, `torch-c-dlpack-ext`, `z3-solver`, `cloudpickle`, and `ml-dtypes`; after pinning `z3-solver==4.15.4.0`, importing `tilelang.profiler.bench` with the existing torch env still aborted inside TVM FFI: `TypeAttr __ffi_repr__ is already registered`. DeepEP's PyPI build also requires an older NVSHMEM layout with `libnvshmem.a`, while the installed NVIDIA NVSHMEM wheel provides `libnvshmem_device.a` and `libnvshmem_host.so.3`. The user asked not to make large environment changes, so I stopped there. This does not affect the SFB conclusion above, because the SFB table compares against the current PR360/SOTA fused path with only this feature toggled.

Next useful idea: for longer sequences, SFB is not the right lever once the producer side dominates. The next feature should target larger long-seq costs: scheduler tail, combine/dispatch overlap, or reducing producer/barrier pressure rather than moving a 1-2 float scale load.


## 2026-06-24: 384-expert wave scheduling - useful only for the long typical cases

Why I looked here:

After the first SFB-SMEM pass, the PR360-standard table had a pattern: 256-expert models got clear SFB wins, but the 384-expert models were still slow, especially at `8192 tokens/rank`. That says the next bottleneck is probably not the tiny SFB load anymore. A better guess is scheduling overhead/tail behavior with `384 / 8 = 48` local experts.

Key insight:

`num_experts_per_wave` decides how many local experts one scheduling wave covers. Too small means many waves and more scheduler/barrier/tail overhead. Too large can reduce balance or hurt locality. So I first swept the wave value instead of changing the kernel body.

Command shape:

```bash
DG_SM90_MOE_SFB_SMEM=-1 DG_SM90_MOE_EXPERTS_PER_WAVE=<1|2|4|8|16|24|48> \
python3 /tmp/codex-pr360-tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 1024 4096 8192 \
  --hidden <model-hidden> --intermediate-hidden <model-intermediate> \
  --num-experts 384 --num-topk <model-topk> --num-tests 7
```

Correction while testing:

I initially thought one broad 48-local-expert rule was enough. That was wrong. With SFB enabled, the best wave depends on the model shape, and `4096 tokens/rank` is noisy enough that the mean does not clearly justify changing it. I also hit a build-cache trap: `pip install --force-reinstall` reused `work/codex-pr360/src/build/python_api.o`, so the new heuristic did not enter `_C`. I fixed this by deleting `work/codex-pr360/src/build` before rebuilding. Lesson: when a C++ header-only heuristic changes, clean the extension build cache.

Exploration result under the current SFB-on code:

| model | tokens/rank | old auto wave | tested best wave | read |
| --- | ---: | ---: | ---: | --- |
| V4 Pro | 4096 | 2 | 16 sometimes, 2 sometimes | not stable; keep old auto |
| V4 Pro | 8192 | 1 | 16 | stable win |
| MiMo-Pro | 4096 | 2 | 24/48 sometimes | not stable; keep old auto |
| MiMo-Pro | 8192 | 1 | 24 | stable win |

Final rule:

- V4 Pro 384-expert shape (`hidden=7168`, `intermediate=3072`, `topk=6`): use `wave=16` only when `tokens_per_expert >= 1024`.
- MiMo-Pro 384-expert shape (`hidden=6144`, `intermediate=2048`, `topk=8`): use `wave=24` only when `tokens_per_expert >= 1024`.
- Keep `4096 tokens/rank` on old auto. This is important: avoiding a weak/unstable win is also an optimization decision.

Why the timings move by hundreds of us:

The reported line is already an inner average from `bench_kineto`, but each outer process still samples routing and runs distributed rendezvous/NVSHMEM work. Small changes in received-token distribution and communication timing can move a multi-rank MoE run by hundreds of microseconds. So I used repeated outer runs and reported mean/std, not the single best number.

Final 8192-token repeated test (`5` valid outer runs, each with `--num-tests 7`):

| model | tokens/rank | tokens/expert | old wave | old mean | old std | new wave | new mean | new std | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 Pro (best wave=16) | 8192 | 1024.0 | 1 | 10560.5 us | 75.5 | 16 | 9808.0 us | 89.8 | +7.67% |
| MiMo-Pro (best wave=24) | 8192 | 1365.3 | 1 | 7797.7 us | 39.9 | 24 | 7518.1 us | 53.2 | +3.72% |

Rough comparison to the earlier PR360 direct-SFB table:

| model | old PR360-style fused path | final 8192 mean | rough gain |
| --- | ---: | ---: | ---: |
| V4 Pro | 10677.6 us | 9808.0 us | +8.87% |
| MiMo-Pro | 7983.5 us | 7518.1 us | +6.19% |

Takeaway:

For 384 experts, the useful feature is not a new math trick. It is making the wave size match the long-token scheduling problem. The important part was the correction: `4096` looked tempting in single runs, but repeated tests showed it was not reliable. The final change only touches the long typical cases where the signal survives averaging.


## 2026-06-24: Combined result table with baseline column

What this table means:

- `PR360-style original` is the earlier fused path before my current changes for that row.
- `Current final` is the final heuristic stack that applies to the row: low-token pingpong wave tuning, SFB-in-SMEM auto range, and the 384-expert long-token wave rule.
- `Official baseline` follows the benchmark's own `--baseline --baseline-version both` path. In code this prints `baseline[v1-contig]`, `baseline[v1-ll]`, and `baseline[v2]`. I am leaving it `n/a` because I did not find public numeric rows for these exact cases, and the local baseline path still failed to load cleanly in this environment.

Do not read the empty baseline cells as zero. They mean "not measured yet".

Overall read:

| group | cases | combined speedup read |
| --- | ---: | --- |
| PR360-standard model/token rows | 20 | mean `+2.86%`, range `+0.12%` to `+9.98%` |
| Flash | 5 | mean `+2.54%`, range `+0.62%` to `+3.92%` |
| V4 Pro | 5 | mean `+3.55%`, range `+1.21%` to `+9.98%` |
| MiMo | 5 | mean `+3.07%`, range `+2.02%` to `+3.68%` |
| MiMo-Pro | 5 | mean `+2.27%`, range `+0.12%` to `+6.96%` |
| low-token pingpong sanity set | 8-128 tokens/rank | mean `+1.88%` |

Detailed table:

| model | tokens/rank | tokens/expert | PR360-style original | current final | combined speedup | official baseline |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Flash | 256 | 48.0 | 326.3 us | 324.3 us | +0.62% | n/a |
| Flash | 512 | 96.0 | 345.7 us | 338.0 us | +2.28% | n/a |
| Flash | 1024 | 192.0 | 597.0 us | 582.2 us | +2.54% | n/a |
| Flash | 4096 | 768.0 | 1807.2 us | 1748.7 us | +3.35% | n/a |
| Flash | 8192 | 1536.0 | 3546.7 us | 3412.8 us | +3.92% | n/a |
| V4 Pro | 256 | 32.0 | 1051.1 us | 1035.2 us | +1.54% | n/a |
| V4 Pro | 512 | 64.0 | 1097.1 us | 1083.0 us | +1.30% | n/a |
| V4 Pro | 1024 | 128.0 | 1695.8 us | 1675.6 us | +1.21% | n/a |
| V4 Pro | 4096 | 512.0 | 5307.1 us | 5115.4 us | +3.75% | n/a |
| V4 Pro | 8192 | 1024.0 | 10677.6 us | 9708.6 us | +9.98% | n/a |
| MiMo | 256 | 64.0 | 333.9 us | 327.3 us | +2.02% | n/a |
| MiMo | 512 | 128.0 | 491.3 us | 476.1 us | +3.19% | n/a |
| MiMo | 1024 | 256.0 | 758.4 us | 734.7 us | +3.23% | n/a |
| MiMo | 4096 | 1024.0 | 2362.8 us | 2278.9 us | +3.68% | n/a |
| MiMo | 8192 | 2048.0 | 4799.2 us | 4649.6 us | +3.22% | n/a |
| MiMo-Pro | 256 | 42.7 | 663.0 us | 652.9 us | +1.55% | n/a |
| MiMo-Pro | 512 | 85.3 | 701.1 us | 690.5 us | +1.54% | n/a |
| MiMo-Pro | 1024 | 170.7 | 1269.9 us | 1255.0 us | +1.19% | n/a |
| MiMo-Pro | 4096 | 682.7 | 3912.9 us | 3908.1 us | +0.12% | n/a |
| MiMo-Pro | 8192 | 1365.3 | 7983.5 us | 7463.6 us | +6.96% | n/a |

Teaching point:

The big lesson from the total table is that SFB-in-SMEM is a broad small win, while the 384-expert wave rule is a narrow bigger win. If I only looked at average speedup, the work would look modest. If I look by model shape, the useful next direction becomes clearer: 384-expert long-token scheduling is still where the largest remaining opportunity is.


## 2026-06-24: 384-expert L1 schedule exploration - promising but not confirmed

Why I looked here:

After wave tuning, the remaining 384-expert long-token cases still looked scheduling-sensitive. The next non-trivial idea was not another wave value, but changing block order:

- Default L1 order is M-major: keep the same activation M block while sweeping N blocks.
- `DG_SM90_MOE_L1_NMAJOR=1` flips L1 to N-major: keep one L1 weight N tile hot while sweeping M blocks.
- `DG_SM90_MOE_EXPERT_LOCAL=1` changes phase order: do L1 and L2 per expert instead of doing L1 for a wave, then L2 for that wave.

Hypothesis:

If long 384-expert cases are losing time to L1 weight locality or wave tail behavior, L1 N-major or expert-local scheduling could help. If activation reuse is still more valuable, it will hurt. This is a real feature test because the scheduler state machine changes; it is not just changing one threshold.

Quick sweep (`8192 tokens/rank`, `--num-tests 5`, same clean `/tmp` install):

| model | variant | time | vs base | read |
| --- | --- | ---: | ---: | --- |
| MiMo-Pro | base | 7338.2 us | 0.00% | current final |
| MiMo-Pro | L1 N-major | 7183.8 us | +2.15% | promising |
| MiMo-Pro | expert-local | 7610.9 us | -3.58% | reject alone |
| MiMo-Pro | L1 N-major + expert-local | 7791.1 us | -5.81% | reject |
| MiMo-Pro | L2 N-major off | 7329.7 us | +0.12% | basically neutral |
| V4 Pro | base | 9713.5 us | 0.00% | current final |
| V4 Pro | L1 N-major | 10088.0 us | -3.71% | reject alone |
| V4 Pro | expert-local | 10275.2 us | -5.47% | reject alone |
| V4 Pro | L1 N-major + expert-local | 9523.8 us | +1.99% | promising but suspicious |
| V4 Pro | L2 N-major off | 9776.3 us | -0.64% | keep default |

What I learned:

MiMo-Pro and V4 Pro do not want the same scheduler. MiMo-Pro likes L1 N-major in the quick sweep, but V4 Pro hates it unless it is combined with expert-local. That means a single "384 experts => L1 N-major" rule would be wrong.

Correction:

I started a 5-run repeat, but the numbers inflated from the 7-10 ms range to 14-23 ms. `nvidia-smi pmon` showed another 8-GPU LLaMA training job using all GPUs (`torchrun_main.py`, PIDs 4035998-4036005). I did not kill it because it is not this benchmark. This repeat is invalid and should not drive a heuristic change.

Current decision:

Do not enable this by default yet. The next valid step, when GPUs are free, is to repeat only two candidates:

- MiMo-Pro 384, `8192 tokens/rank`: base vs `DG_SM90_MOE_L1_NMAJOR=1`.
- V4 Pro 384, `8192 tokens/rank`: base vs `DG_SM90_MOE_L1_NMAJOR=1 DG_SM90_MOE_EXPERT_LOCAL=1`.

Teaching point:

This is exactly why repeated measurements matter. A quick sweep is allowed to generate hypotheses. A default heuristic needs repeatable mean/std under a clean GPU. The useful optimization idea here is "model-specific L1 scheduling for long 384-expert cases", but it is not proven yet.


## 2026-06-24: L1 schedule confirm script prepared

Why no new timing yet:

I checked again and all 8 GPUs were still occupied by another `torchrun_main.py` LLaMA training job (PIDs `4113728-4113735`, about `97-99%` SM utilization). I did not kill it. Running the 384-expert confirmation now would only reproduce the polluted 14-23 ms numbers from the invalid repeat.

What I did instead:

I added `work/codex-pr360/run_l1_schedule_confirm.sh`. It fixes the exact next clean-GPU experiment:

- MiMo-Pro 384, `8192 tokens/rank`: base vs `DG_SM90_MOE_L1_NMAJOR=1`.
- V4 Pro 384, `8192 tokens/rank`: base vs `DG_SM90_MOE_L1_NMAJOR=1 DG_SM90_MOE_EXPERT_LOCAL=1`.
- Default repeat shape: `RUNS=5`, `NUM_TESTS=7`.
- It reuses the clean current-branch install at `/tmp/codex-pr360-site-current-20260624-140358`, uses separate JIT caches, prints each run, then prints mean/std and speedup.

Dry run:

```bash
RUNS=0 work/codex-pr360/run_l1_schedule_confirm.sh
```

This passed and only printed the GPU snapshot plus empty summary. No benchmark was launched.

Next action when GPUs are free:

```bash
work/codex-pr360/run_l1_schedule_confirm.sh
```

If MiMo-Pro and V4 Pro both keep positive mean speedup under a clean repeat, then I can safely turn the model-specific L1 schedule rule into a default heuristic. If either one loses the signal, the lesson is still useful: quick sweeps can identify locality hypotheses, but only clean repeated measurements can justify a default.


## 2026-06-24: L1 schedule heuristic enabled after clean repeat

Clean repeat setup:

The GPUs became idle again, so I ran the prepared script:

```bash
work/codex-pr360/run_l1_schedule_confirm.sh
```

This used the clean current-branch install `/tmp/codex-pr360-site-current-20260624-140358`, `RUNS=5`, `NUM_TESTS=7`, and separate JIT caches. The run root was `/tmp/codex-pr360-l1-confirm-20260624-162902`.

Repeat result:

| model | candidate | base mean | candidate mean | base std | candidate std | speedup | stability read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| MiMo-Pro 384, 8192 tokens/rank | L1 N-major | 7577.5 us | 7463.6 us | 84.2 | 397.5 | +1.53% | wins 4/5; one slow candidate outlier |
| V4 Pro 384, 8192 tokens/rank | L1 N-major + expert-local | 9883.7 us | 9708.6 us | 64.7 | 54.1 | +1.80% | wins 5/5 |

Raw values:

| model | base values | candidate values |
| --- | --- | --- |
| MiMo-Pro | 7547.6, 7501.0, 7541.7, 7577.1, 7720.1 | 7260.1, 7306.7, 8173.7, 7270.6, 7306.8 |
| V4 Pro | 9897.0, 9802.5, 9833.3, 9928.2, 9957.7 | 9705.6, 9771.5, 9738.0, 9625.9, 9701.9 |

Correction / interpretation:

MiMo-Pro has one candidate outlier (`8173.7 us`). I kept it in the mean instead of deleting it, so `+1.53%` is conservative. The median story is stronger (`7547.6 -> 7306.7 us`, about `+3.30%`), but mean is the safer headline. V4 Pro is cleaner: every repeated pair wins and the candidate std is not inflated.

Implementation decision:

Enable only the measured long-token 384-expert cases:

- MiMo-Pro 384 (`hidden=6144`, `intermediate=2048`, `topk=8`) with `tokens_per_expert >= 1024`: auto-enable `l1_nmajor_schedule`.
- V4 Pro 384 (`hidden=7168`, `intermediate=3072`, `topk=6`) with `tokens_per_expert >= 1024`: auto-enable both `l1_nmajor_schedule` and `expert_local_schedule`.
- Keep 4096-token cases unchanged. Earlier quick sweep said 4096 was noisy, and the clean config check confirms the new auto rule still leaves 4096 off.

Post-implementation config check:

I rebuilt a clean install at `/tmp/codex-pr360-site-l1auto-20260624-164016` and ran `DG_PRINT_CONFIGS=1` for `4096` and `8192` tokens/rank.

| model | tokens/rank | l1_nmajor | expert_local | read |
| --- | ---: | ---: | ---: | --- |
| MiMo-Pro | 4096 | 0 | 0 | unchanged |
| MiMo-Pro | 8192 | 1 | 0 | expected auto candidate |
| V4 Pro | 4096 | 0 | 0 | unchanged |
| V4 Pro | 8192 | 1 | 1 | expected auto candidate |

Teaching point:

This is the first scheduling feature after wave tuning that survived a clean repeated test. The important trick was not "turn on L1 N-major everywhere". The data says the right move is model-specific: MiMo-Pro wants L1 weight locality, while V4 Pro only benefits when the phase order also becomes expert-local. That is why the heuristic is narrow.


## 2026-06-24: M-local scheduler hypothesis prepared, timing blocked by busy GPUs

Why I looked here:

The current long-token wins came from changing schedule order, not from tiny arithmetic cleanup. After L1 N-major, the next real schedule question is whether L2 should wait for the whole wave's L1, or start as soon as one M block has all its L1 N blocks ready.

Hypothesis:

Default wave-level order keeps L1 broad and simple: run L1 for the wave, then L2 for the wave. The new `DG_SM90_MOE_MLOCAL=1` experiment changes the order inside each expert to:

1. pick one M block,
2. run all L1 N blocks for that M block,
3. then run all L2 N blocks for the same M block.

If the bottleneck is L2 input residency / phase distance, this could help. If the cost is mainly waiting for L1 completion or losing weight locality, it will hurt. This is a real scheduler feature, not a wave-value sweep.

What I implemented:

- Added `mlocal_schedule` to `MegaMoESM90Config`.
- Added `DG_SM90_MOE_MLOCAL=1` for the cooperative SM90 path, default off.
- Added `get_next_block_m_local()` in `sm90_mega_moe.cuh`.
- When M-local is enabled, it overrides the normal expert-local/default scheduler path for that experiment only.

Validation so far:

| check | result |
| --- | --- |
| clean wheel build | passed, installed at `/tmp/codex-pr360-site-mlocal-20260624-170911` |
| installed header contains M-local path | passed |
| offline NVCC compile, `kMLocalSchedule=true` V4 Pro 384-like instance | passed, cubin `/tmp/codex_mlocal_compile_check.cubin` |
| offline NVCC compile, `kMLocalSchedule=false` default-like instance | passed, cubin `/tmp/codex_default_compile_check.cubin` |
| 8-GPU timing | not run yet |

Correction:

I did not run benchmark timing because all 8 GPUs were occupied by another LLaMA training job (`torchrun_main.py`, PIDs `229149-229156`, about `68-85%` GPU utilization and `~9 GB` memory per GPU). Running MegaMoE now would produce polluted numbers. This is exactly the kind of measurement I should reject before it becomes a fake optimization.

Next clean test:

Use the new site only when GPUs are idle:

```bash
SITE=/tmp/codex-pr360-site-mlocal-20260624-170911
DG_SM90_MOE_KERNEL=cooperative DG_SM90_MOE_MLOCAL=<0|1> \
PYTHONPATH=$SITE:/home/chen/workspace/source_code/DeepGEMM/work/codex-pr360/site \
python3 /tmp/codex-pr360-tests/bench_mega_moe_sm90.py \
  --num-processes 8 --batches 8192 \
  --hidden 7168 --intermediate-hidden 3072 \
  --num-experts 384 --num-topk 6 --num-tests 7
```

Then repeat MiMo-Pro too (`hidden=6144`, `intermediate=2048`, `topk=8`). If M-local loses, the lesson is still useful: shortening L1-to-L2 phase distance is not free because early L2 blocks can stall on the arrival mask.
