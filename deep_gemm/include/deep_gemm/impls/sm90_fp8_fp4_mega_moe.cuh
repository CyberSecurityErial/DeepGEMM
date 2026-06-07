#pragma once

#include <cstdint>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumMMAThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumThreads = kNumDispatchThreads + kNumMMAThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_fp4_mega_moe_impl(void* y,
                           int* cumulative_local_expert_recv_stats,
                           const uint32_t num_tokens,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {
    (void) y;
    (void) tensor_map_l1_acts;
    (void) tensor_map_l1_acts_sf;
    (void) tensor_map_l1_weights;
    (void) tensor_map_l1_weights_sf;
    (void) tensor_map_l1_output;
    (void) tensor_map_l2_acts;
    (void) tensor_map_l2_acts_sf;
    (void) tensor_map_l2_weights;
    (void) tensor_map_l2_weights_sf;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000)) || defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumDispatchThreads == kNumDispatchWarps * 32,
                     "kNumDispatchThreads must match dispatch warp participants");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
    DG_STATIC_ASSERT(kNumTokensPerWarp * kNumTopk <= 32,
                     "One dispatch warp can only cover at most 32 token-topk entries");

    // Thread indices
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Workspaces
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    // Token and buffer layouts
    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Registered inputs
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF and its buffer configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");

    // UTCCP 4x32 transpose index mapping within each 128-element group.
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 dispatch buffers
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // Task scheduler is still useful for dispatch-side expert counts and pool offsets.
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);

    // [Dispatch M2 scratch]: requires dynamic shared memory >= kNumExperts * sizeof(uint32_t).
    // The same array intentionally changes meaning:
    // A. count phase: this CTA's local send count per global expert.
    // B. offset phase: this CTA's base slot in the source-rank send queue.
    // C. scatter phase: block-local allocator returning final dst slots.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    constexpr uint32_t kDispatchSmemExpertCountBytes = kNumExperts * sizeof(uint32_t);

    constexpr uint32_t kDispatchBarrierIdx = 0;

    // Dispatch warps only. MMA, epilogue, L2, and combine state intentionally stay out
    // of this scaffold while the Hopper dispatch path is being written.
    if (warp_idx < kNumDispatchWarps) {
        const uint32_t dispatch_warp_idx = warp_idx;

        // [Dispatch M0]: external input ready.
        // The input_* buffers are already populated by the caller before this kernel starts.

        // [Dispatch M1]: read routing metadata; each active lane owns one flattened token-topk entry.
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t token_base_idx =
                     (sm_idx * kNumDispatchWarps + dispatch_warp_idx) * kNumTokensPerWarp;
                 token_base_idx < num_tokens;
                 token_base_idx += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                const uint32_t token_idx = token_base_idx + lane_idx / kNumTopk;
                const uint32_t topk_slot = lane_idx % kNumTopk;
                if (lane_idx < kNumActivateLanes and token_idx < num_tokens) {
                    const uint32_t token_topk_idx = token_idx * kNumTopk + topk_slot;
                    const int expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_topk_idx));
                    if (expert_idx >= 0)
                        process(token_topk_idx, static_cast<uint32_t>(expert_idx));
                }
                __syncwarp();
            }
        };

        // [Dispatch M2]: set up dispatch queue slots for each token-topk entry.
        // [Dispatch M2/A]: clear per-CTA expert counters. Use a regular SM90-safe clear here;
        // the SM100 path keeps the bulk-zero variant.
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
            smem_expert_count[i] = 0;
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // [Dispatch M2/A]: count local contributions to every global expert.
        read_topk_idx([&](const uint32_t& token_topk_idx, const uint32_t& expert_idx) {
            (void) token_topk_idx;
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // [Dispatch M2/B]: convert per-CTA counts into source-rank queue base slots.
        // high32 counts producer arrivals, so zero-count experts still publish completion.
        // low32 accumulates token-topk sends for this source rank and becomes the base slot.
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // [Dispatch M2/C]: publish source token-topk indices into the destination rank's
        // per-expert, per-source-rank queue. get_expert_send_count_ptr() is local to this
        // source rank's workspace; M3 later copies its low32 count to
        // recv_count[src_rank, local_expert], matching src_token_topk_idx's slot dimension.
        read_topk_idx([&](const uint32_t& token_topk_idx, const uint32_t& expert_idx) {
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
            const uint32_t dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                local_expert_idx, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });
        // M3 grid/NVLink ordering is intentionally not implemented in this scaffold; no
        // downstream reader may assume the remote metadata is visible until that barrier lands.

        (void) cumulative_local_expert_recv_stats;
        (void) input_token_buffer;
        (void) input_sf_buffer;
        (void) input_topk_weights_buffer;
        (void) kDispatchSmemExpertCountBytes;
        (void) transform_sf_token_idx;
        (void) l1_token_buffer;
        (void) l1_sf_buffer;
        (void) l1_topk_weights_buffer;
        (void) scheduler;
    }
#endif
}

} // namespace deep_gemm
