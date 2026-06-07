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

    // SM90 M0-M3 keeps the SF buffers in the shared workspace layout so host-side
    // slicing remains compatible. Hopper-specific SF copy/reorder is intentionally
    // deferred; do not inherit SM100 UTCCP/SFB token-index transforms here.

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
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;

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

        // [Dispatch M3/A]: all SMs on this source rank must finish src_token_topk_idx
        // stores before SM 0 publishes counts to remote ranks. This preserves the
        // downstream invariant that a visible recv count implies a complete source list.
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
        // cuda kernel not support grid sync, so we use a gmem memory to achieve grid sync.

        // [Dispatch M3/B]: finalize the destination rank's receive metadata.
        // expert_send_count lives in this source rank's workspace; low32 is the number
        // of slots written for this source rank, high32 is the producer-arrival count.
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t dst_rank_idx = i / kNumExpertsPerRank;
                const uint32_t dst_local_expert_idx = i % kNumExpertsPerRank;
                const uint64_t expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffffull;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // [Dispatch M3/C]: publish the metadata phase across ranks. M4/M5 must only read
        // src_token_topk_idx, expert_recv_count, and expert_recv_count_sum after this barrier.
        // cta sync & grid sync < nvlink sync, so nvlink barrier's lambda func body is cta sync, and need a template parameter named kDispatchGridSyncIndex
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, only SM 0 writes receive metadata */ false,
            /* Later M4/M5 readers need a post-barrier grid sync */ true
        );

        // [Dispatch M4]: enumerate the finalized local expert pool. This maps each
        // local pool entry to the source rank/token/topk slot that M5 will pull.
        // Pool destinations are block-major, not a plain expert prefix sum:
        //   pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert.
        const auto enumerate_local_expert_pool = [&](const auto& process) {
            // Cache expert token counts in registers (same pattern as the GEMM scheduler).
            scheduler.fetch_expert_recv_count();

            // Per-rank counts for current expert (re-loaded when expert changes).
            constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
            int current_expert_idx = -1;
            uint32_t stored_rank_count[kNumRanksPerLane] = {};
            uint32_t expert_start_idx = 0, expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
            for (uint32_t token_idx = sm_idx * kNumDispatchWarps + dispatch_warp_idx; ;
                 token_idx += kNumGlobalWarps) {
                // Advance expert until token_idx lands in the current expert receive range.
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;

                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }

                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t rank_idx = i * 32 + lane_idx;
                        stored_rank_count[i] = rank_idx < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(rank_idx, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection via iterative min-peeling. This matches
                // the SM100 dispatch order and converts token_idx_in_expert to a
                // per-source-rank slot in src_token_topk_idx.
                uint32_t current_rank_in_expert_idx = 0;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];

                uint32_t offset = 0;
                const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank = 0;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffffu;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = remaining[i] < min_in_lane ? remaining[i] : min_in_lane;
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and
                                slot_idx_in_round < num_seen_ranks + num_active_lanes) {
                                current_rank_in_expert_idx =
                                    i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            }
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }

                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= remaining[i] < length ? remaining[i] : length;
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;
                const uint32_t pool_token_idx =
                    expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

                process(pool_token_idx,
                        static_cast<uint32_t>(current_expert_idx),
                        token_idx_in_expert,
                        current_rank_in_expert_idx,
                        src_token_idx,
                        src_topk_idx,
                        src_token_topk_idx);
            }
        };

        enumerate_local_expert_pool([&](const uint32_t& pool_token_idx,
                                        const uint32_t& local_expert_idx,
                                        const uint32_t& token_idx_in_expert,
                                        const uint32_t& src_rank_idx,
                                        const uint32_t& src_token_idx,
                                        const uint32_t& src_topk_idx,
                                        const uint32_t& src_token_topk_idx) {
            (void) pool_token_idx;
            (void) local_expert_idx;
            (void) token_idx_in_expert;
            (void) src_rank_idx;
            (void) src_token_idx;
            (void) src_topk_idx;
            (void) src_token_topk_idx;
        });

        // M5+ is intentionally out of scope for this scaffold.

        (void) cumulative_local_expert_recv_stats;
        (void) input_token_buffer;
        (void) input_sf_buffer;
        (void) input_topk_weights_buffer;
        (void) kDispatchSmemExpertCountBytes;
        (void) l1_token_buffer;
        (void) l1_sf_buffer;
        (void) l1_topk_weights_buffer;
    }
#endif
}

} // namespace deep_gemm
