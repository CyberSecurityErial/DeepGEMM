既然你明确说 **要长期改，并且要在 DeepGEMM 里支持 MegaMoE 的 Hopper 版本**，那我的建议很明确：

> **fork DeepGEMM，然后在自己的 fork 里维护一个长期 integration branch，再用多个 feature branch 分阶段开发。不要直接在 main 上改。**

DeepGEMM 现在已经是一个持续活跃的上游项目，README 里也明确它包含 FP8/FP4 GEMM、Mega MoE、JIT 编译等能力；issues/PR 也很多，说明 upstream 会持续变化，你长期改必须有一套能稳定同步 upstream 的分支策略。([GitHub][1])

---

## 推荐仓库策略

### 1. 先 fork，不要只 clone 后本地开分支

你要长期支持 Hopper MegaMoE，最好走：

```text
upstream/deepseek-ai/DeepGEMM
        ↓
your-fork/DeepGEMM
        ↓
echo/hopper-megamoe
        ↓
feature/*
```

原因：

1. **你可以长期保存自己的改动**；
2. **以后可以向 upstream 提 PR**；
3. **可以持续 rebase / merge upstream/main**；
4. **可以给领导、同事、Codex 一个稳定仓库地址**；
5. **不会被 upstream main 的频繁变化打乱实验记录**。

---

## 推荐分支结构

```text
main
  只跟 upstream/main 同步，不直接改

echo/hopper-megamoe
  你的长期集成分支，代表“当前可工作的 Hopper MegaMoE 版本”

feature/hopper-megamoe-harness
  算子验证 pipeline、case、bench、profile

feature/sm90-fp8-local-compute
  单卡 local fused MoE compute

feature/sm90-fp8-api-dispatch
  Python API / capability dispatch / fallback

feature/deepep-sm90-megamoe-baseline
  DeepEP dispatch + candidate compute + combine

feature/sm90-symm-buffer
  如果后面要动 symmetric buffer / communication overlap

feature/h200-bench-report
  benchmark/report/profile 相关脚本
```

你长期工作的主线应该是：

```text
feature/*  ->  echo/hopper-megamoe  ->  optional PR to upstream
```

不要直接在 `echo/hopper-megamoe` 上乱改。这个分支应该尽量保持：

```text
能安装
能跑测试
能跑 benchmark
能 profile
```

---

## 初始化命令

```bash
# 1. fork 后 clone 你自己的 fork
git clone git@github.com:<your-name>/DeepGEMM.git
cd DeepGEMM

# 2. 添加 upstream
git remote add upstream https://github.com/deepseek-ai/DeepGEMM.git

# 3. 拉 upstream
git fetch upstream

# 4. 让自己的 main 对齐 upstream/main
git checkout main
git reset --hard upstream/main
git push origin main --force-with-lease

# 5. 创建长期集成分支
git checkout -b echo/hopper-megamoe
git push -u origin echo/hopper-megamoe

# 6. 从长期分支开第一个 feature
git checkout -b feature/hopper-megamoe-harness
```

之后每个阶段都这样：

```bash
git checkout echo/hopper-megamoe
git pull origin echo/hopper-megamoe

git checkout -b feature/xxx
# 开发
git add .
git commit -m "feat: add xxx for Hopper MegaMoE"
git push -u origin feature/xxx
```

然后在 GitHub 上开 PR：

```text
feature/xxx -> echo/hopper-megamoe
```

---

## 同步 upstream 的方式

DeepGEMM 现在还在持续变化，比如近期 issues 里有 MegaMoE 的 FP4 activation dispatch、JIT cache 多进程同步、CUDA/NVRTC 兼容等问题，这些都可能影响你后面的 H200 多进程测试和 MegaMoE 开发。([GitHub][2])

所以建议每周同步一次 upstream：

```bash
git fetch upstream

git checkout main
git reset --hard upstream/main
git push origin main --force-with-lease

git checkout echo/hopper-megamoe
git merge main
# 或者如果你更熟悉 rebase：
# git rebase main

git push origin echo/hopper-megamoe
```

如果你还不熟 rebase，**先用 merge**，更安全、可追溯。

---

## 代码改动原则：不要把 Hopper 写成一堆 ifdef

你要支持的是：

```text
MegaMoE Hopper version
```

不是：

```text
在 Blackwell MegaMoE 里到处塞 sm90 if/else
```

所以建议目录和命名上明确隔离：

```text
deep_gemm/
  jit/
  testing/

csrc/
  kernels/
    sm90_fp8_gemm/
    sm90_fp8_megamoe/        # 新增：Hopper MegaMoE
    sm100_fp8_fp4_megamoe/   # 原有/Blackwell 路径
  apis/
    fp8_mega_moe_sm90.cuh
    fp8_fp4_mega_moe_sm100.cuh
```

Python API 也建议做 sibling，而不是强行复用一个名字：

```python
deep_gemm.fp8_mega_moe_sm90(...)
deep_gemm.fp8_fp4_mega_moe_sm100(...)
```

然后再提供统一 dispatch：

```python
deep_gemm.mega_moe(...)
```

里面根据硬件和 dtype 选择：

```python
if is_sm100 and weight_dtype == fp4:
    return fp8_fp4_mega_moe_sm100(...)
elif is_sm90 and weight_dtype == fp8:
    return fp8_mega_moe_sm90(...)
else:
    return grouped_gemm_fallback(...)
```

你之前文档里也已经把 H200 路线判断清楚了：Hopper/H200 不应该直接照搬 Blackwell FP4 路径，而应该优先做 FP8 MegaMoE / W4A16 / fallback 路径。

---

## 开发顺序建议

你现在这个长期分支可以按 5 个 PR 做，不要一个巨型 PR。

### PR 1：验证与 benchmark harness

分支：

```text
feature/hopper-megamoe-harness
```

做：

```text
tests/test_megamoe_sm90_ref.py
tests/test_megamoe_sm90_local.py
bench/run_megamoe_sm90_bench.py
scripts/run_ncu_megamoe_sm90.sh
scripts/run_nsys_megamoe_sm90.sh
```

这一步不一定写真正 kernel，先把 reference、case、CSV、profile 通道搭起来。

---

### PR 2：SM90 local compute kernel skeleton

分支：

```text
feature/sm90-fp8-local-compute
```

做：

```text
input grouped by expert
GEMM1
SwiGLU
GEMM2
local combine
```

先不接 DeepEP，不做 overlap。

---

### PR 3：Python API + fallback dispatch

分支：

```text
feature/sm90-megamoe-api
```

做：

```python
deep_gemm.fp8_mega_moe_sm90(...)
deep_gemm.mega_moe(...)
```

并加：

```text
unsupported hardware -> fallback
unsupported dtype -> fallback
unsupported shape -> fallback
```

---

### PR 4：DeepEP dispatch/combine 接入

分支：

```text
feature/sm90-megamoe-deepep
```

做：

```text
DeepEP dispatch
    ↓
SM90 local MegaMoE compute
    ↓
DeepEP combine
```

DeepEP 本身就是面向 MoE expert parallel dispatch/combine 的库，它支持 high-throughput/low-latency all-to-all GPU kernels，也支持 FP8 dispatch 场景，所以这一步是你从单卡算子进入 8 卡 EP 算子层的关键。([GitHub][3])

---

### PR 5：overlap / persistent / symm buffer 优化

分支：

```text
feature/sm90-megamoe-overlap
```

这一步再考虑：

```text
communication-compute overlap
symmetric buffer
persistent scheduling
arrival metadata
per-expert ready queue
double buffer
```

不要一开始就做这步。

---

## commit 命名建议

长期项目一定要让 commit 能读：

```bash
git commit -m "test: add SM90 MegaMoE torch reference"
git commit -m "bench: add local MegaMoE shape sweep"
git commit -m "feat: add SM90 FP8 MegaMoE API stub"
git commit -m "feat: add SM90 local fused MoE compute kernel"
git commit -m "perf: add NCU profile script for MegaMoE SM90"
git commit -m "fix: handle empty experts in SM90 MegaMoE"
```

不要写：

```text
update
fix bug
test
tmp
```

因为你以后要向领导解释、向 upstream 提 PR、让 Codex 接着做，commit 信息会很重要。

---

## 我建议你现在就这么定

```text
fork DeepGEMM
  ↓
main 跟 upstream/main 对齐
  ↓
echo/hopper-megamoe 作为长期集成分支
  ↓
feature/hopper-megamoe-harness 作为第一个开发分支
```

第一个目标不要写 kernel，而是先让 DeepGEMM 里出现一条清晰的 H200 MegaMoE 验证入口：

```bash
python -m pytest tests/test_megamoe_sm90_ref.py -s
python bench/run_megamoe_sm90_bench.py --case toy
bash scripts/run_ncu_megamoe_sm90.sh
```

等这个 pipeline 稳了，再开始把 Blackwell MegaMoE 的结构拆出来，做 SM90 FP8 sibling。

[1]: https://github.com/deepseek-ai/DeepGEMM?utm_source=chatgpt.com "DeepGEMM: clean and efficient FP8 GEMM kernels ..."
[2]: https://github.com/deepseek-ai/DeepGEMM/issues?utm_source=chatgpt.com "Issues · deepseek-ai/DeepGEMM"
[3]: https://github.com/deepseek-ai/DeepGEMM/issues/338?utm_source=chatgpt.com "[RFC] Support FP4 (E2M1) activation dispatch in MegaMoE"
