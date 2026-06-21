# SM90 FP8 MegaMoE — 8×H20 测试手册

本文档用于让没有参与开发的测试人员，在单机 8×H20 环境中拉取代码、构建、跑通 SM90 FP8 MegaMoE，并回传足够的诊断信息。

测试目标分为两级：

1. 跑通验证：JIT 编译成功，8 个 rank 正常完成通信，输出全部为 finite，PTXAS 没有 local-memory spill。
2. 性能验证：采集不同 token 数的 kernel latency，并按需生成 Kineto、NSYS 和 NCU 报告。

当前 smoke test 不等价于完整数值正确性测试。它检查调用链、通信同步和基本输出有效性，但不会与一个独立 MoE reference 做逐元素误差比较。

## 1. 机器要求

- 单机 8 张 NVIDIA H20，所有卡对当前容器可见。
- 8 卡之间支持 CUDA P2P；推荐 NVSwitch/NVLink 全互联。
- Linux x86_64。
- CUDA Toolkit 12.3 以上，推荐 12.9 以上。
- PyTorch 2.9 以上，并带 NCCL 和 `torch.distributed._symmetric_memory`。
- Python 3.8 以上。
- 支持 C++20 的主机编译器。
- 可选：Nsight Systems 和 Nsight Compute。

不要在 MIG 模式下运行。测试期间不要让其他任务占用这些 GPU。

## 2. 拉取指定分支

```bash
git clone --recursive \
  --branch echo/support_hopper_megamoe \
  https://github.com/CyberSecurityErial/DeepGEMM.git

cd DeepGEMM
git rev-parse HEAD
git status --short --branch
```

把 `git rev-parse HEAD` 的输出保存在最终测试报告中。

如果仓库已经存在：

```bash
git fetch origin
git checkout echo/support_hopper_megamoe
git pull --ff-only origin echo/support_hopper_megamoe
git submodule update --init --recursive
```

## 3. 一键执行推荐流程

建议固定一个输出目录，环境信息、构建日志、smoke 日志和 benchmark 日志都会写入其中：

```bash
export OUTPUT_DIR="$PWD/work/h20-sm90-megamoe"
export NUM_PROCESSES=8
export PYTHON_BIN=python

bash scripts/run_sm90_mega_moe_h20.sh all
```

这条命令依次执行：

1. GPU、CUDA、PyTorch、NCCL、symmetric memory 和 P2P 检查。
2. 构建 `deep_gemm._C`。
3. 运行 token 数为 `1 / 4 / 512` 的 smoke cases。
4. 运行默认 token sweep benchmark。

如果集群只提供 `python3`：

```bash
export PYTHON_BIN=python3
```

## 4. 分步骤执行

### 4.1 环境检查

```bash
export OUTPUT_DIR="$PWD/work/h20-sm90-megamoe"
bash scripts/run_sm90_mega_moe_h20.sh env
```

成功标志：

```text
PASS: host satisfies the SM90 MegaMoE prerequisites
```

另外检查：

- `Visible GPUs: 8`
- 每张卡 capability 的 major 为 `9`
- `NCCL available: True`
- `nvidia-smi topo -m` 中没有异常的跨卡 `SYS` 路径
- 没有 CUDA peer-access failure

### 4.2 构建

```bash
bash scripts/run_sm90_mega_moe_h20.sh build
```

成功标志：

- `setup.py build` 返回 0。
- `import deep_gemm` 成功。
- 输出中能看到 DeepGEMM 和 PyTorch 版本。

### 4.3 Smoke test

```bash
bash scripts/run_sm90_mega_moe_h20.sh smoke
```

默认 shape：

```text
EP=8
experts=384
topk=6
hidden=7168
intermediate_hidden=3072
num_max_tokens_per_rank=8192
tokens=1,4,512
```

三个 token case 用于覆盖不同 kernel 路径：

| tokens | 预期主要路径 |
|---:|---|
| 1 | decode，BLOCK_M=64，BLOCK_N=128 |
| 4 | decode split-N，BLOCK_M=64，BLOCK_N=256，L2 arrival counter |
| 512 | split-MN，BLOCK_M=128，BLOCK_N=256 |

成功标志：

```text
Smoke passed: tokens=1, output is finite
Smoke passed: tokens=4, output is finite
Smoke passed: tokens=512, output is finite
```

脚本会设置：

```text
DG_PRINT_CONFIGS=1
DG_JIT_PTXAS_VERBOSE=1
DG_JIT_PTXAS_CHECK=1
```

因此以下情况都应视为失败：

- JIT/NVCC/PTXAS 编译错误。
- PTXAS 报告 local-memory spill，并触发 `DG_JIT_PTXAS_CHECK`。
- NVLink barrier timeout 或 device trap。
- 任意 rank 非零退出。
- 输出包含 NaN 或 Inf。

### 4.4 Benchmark

```bash
bash scripts/run_sm90_mega_moe_h20.sh bench --num-tests 30
```

每个 case 会打印：

```text
tokens=... recv=... experts=... latency=... us compute=... TFLOPS
```

第一轮建议只记录 latency，不要立即以 TFLOPS 判断 kernel 好坏。这里的 TFLOPS 按接收 token 数估算，路由不均衡、通信和 combine 都会影响最终值。

完整 sweep：

```bash
export BENCH_TOKENS="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192"
bash scripts/run_sm90_mega_moe_h20.sh bench --num-tests 30
```

如果只是排障，希望缩短运行时间：

```bash
export BENCH_TOKENS="1 4 128 512"
bash scripts/run_sm90_mega_moe_h20.sh bench --num-tests 5 --flush-l2 0
```

正式性能数据应保留 `--flush-l2 1`。

## 5. Timeline 和 profiler

### 5.1 Kineto trace

```bash
export OUTPUT_DIR="$PWD/work/h20-sm90-megamoe"
bash scripts/run_sm90_mega_moe_h20.sh trace
```

Trace 默认生成在：

```text
$OUTPUT_DIR/traces/
```

每个 token case、每个 rank 都有独立 JSON，不会相互覆盖。

### 5.2 Nsight Systems

先完成 build 和 smoke，再运行：

```bash
export OUTPUT_DIR="$PWD/work/h20-sm90-megamoe-nsys"
export TOKENS=4
bash scripts/run_nsys_sm90_mega_moe.sh
```

重点检查：

- 8 个 rank 是否都启动了 `sm90_fp8_mega_moe_impl`。
- 各 rank kernel 启动时间是否严重错位。
- 是否存在某个 rank 提前退出，导致其他 rank 卡在 barrier。
- kernel 前后是否有意外的大规模同步或 allocator 开销。

### 5.3 Nsight Compute

MegaMoE kernel 内含跨 rank barrier。NCU 必须使用 application replay，并让 8 个 profiler 进程 lockstep 运行。不要单独 profile 一个 rank，也不要改成 kernel replay。

```bash
export OUTPUT_DIR="$PWD/work/h20-sm90-megamoe-ncu"
export TOKENS=4
bash scripts/run_ncu_sm90_mega_moe.sh
```

脚本会：

1. 先用 8 卡运行一次，填充 JIT cache。
2. 启动 8 个 NCU 进程。
3. 通过 TCP communicator 做 lockstep application replay。
4. 每个 rank 输出一个 `.ncu-rep` 和一个日志。

建议分别采集三条路径：

```bash
TOKENS=1   OUTPUT_DIR="$PWD/work/ncu-t1"   bash scripts/run_ncu_sm90_mega_moe.sh
TOKENS=4   OUTPUT_DIR="$PWD/work/ncu-t4"   bash scripts/run_ncu_sm90_mega_moe.sh
TOKENS=512 OUTPUT_DIR="$PWD/work/ncu-t512" bash scripts/run_ncu_sm90_mega_moe.sh
```

优先关注：

- Registers Per Thread。
- Local Memory Per Thread，应为 0。
- Achieved Occupancy。
- Warp Stall Reasons。
- Tensor Core / SM throughput。
- HBM throughput。
- L1/L2 hit rate。
- SourceCounters 中 WGMMA、TMA load 和 epilogue 区域的热点。

## 6. 常见失败

### PyTorch symmetric memory 不存在

典型报错：

```text
torch.distributed._symmetric_memory ...
```

处理：升级到支持 symmetric memory 的 PyTorch 2.9+ build。

### 不是 SM90

环境检查会打印 capability。H20 应为 major `9`。如果不是，确认容器映射的 GPU 是否正确。

### CUDA peer access 失败

先检查：

```bash
nvidia-smi topo -m
nvidia-smi -q | grep -i -A3 mig
```

确认没有 MIG，容器没有禁用 P2P，8 卡位于支持当前通信方式的同一节点。

### Barrier timeout/device trap

常见原因：

- 某个 rank 更早发生 JIT、OOM 或非法访存错误。
- rank 数和可见 GPU 数不一致。
- 某个 rank 被调度器或 profiler 杀死。
- P2P/NVLink 不可用。

必须从 `rank0` 到 `rank7` 的日志中找到最先报错的进程；其他 rank 的 barrier trap 通常只是后果。

### OOM

先减少测试池和 shape：

```bash
bash scripts/run_sm90_mega_moe_h20.sh smoke \
  --num-max-tokens-per-rank 512 \
  --hidden 4096 \
  --intermediate-hidden 2048 \
  --num-experts 64 \
  --num-topk 4
```

这只能用于排障。最终仍需回到默认模型 shape。

### NCU 找不到 kernel

确认 kernel filter 是：

```text
sm90_fp8_mega_moe_impl
```

并确认 warmup 阶段已经成功完成 JIT。

## 7. 需要回传的结果

最少回传以下文件：

```text
environment.txt
env-check.log
build.log
import-check.log
smoke.log
benchmark.log
```

同时提供：

```bash
git rev-parse HEAD
git status --short --branch
```

如果 profiler 已运行，再提供：

- `traces/*.json`
- NSYS `.nsys-rep`
- 8 个 NCU `.ncu-rep`
- `rank0.log` 到 `rank7.log`

可以打包普通日志，排除较大的 JIT cache：

```bash
tar \
  --exclude='jit-cache' \
  -czf h20-sm90-megamoe-logs.tar.gz \
  work/h20-sm90-megamoe
```

## 8. 测试报告模板

```text
Commit:
GPU:
Driver:
CUDA Toolkit:
PyTorch:
CUDA_VISIBLE_DEVICES:
Topology:

Environment check: PASS / FAIL
Build: PASS / FAIL
Smoke tokens=1: PASS / FAIL
Smoke tokens=4: PASS / FAIL
Smoke tokens=512: PASS / FAIL
PTXAS local memory: 0 / nonzero

Latency:
  tokens=1:
  tokens=4:
  tokens=8:
  tokens=16:
  tokens=32:
  tokens=64:
  tokens=128:
  tokens=256:
  tokens=512:

First failing rank:
First error:
Attached logs/reports:
```
