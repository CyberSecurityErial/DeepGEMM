#pragma once

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm90.hpp"

// SM90 MegaMoE heuristics live in their own translation-unit header so the
// SM100 path (`mega_moe.hpp`) stays untouched. We include `mega_moe.hpp` only
// to reuse the shared `get_num_experts_per_wave_for_mega_moe` wave search.
#include "mega_moe.hpp"

namespace deep_gemm {

// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3, scales are per-128 channel float.
//   - No 2-CTA cluster MMA: TMA multicast cluster=2 may still be used.
//   - SF for activations is float (not UE8M0 int) and per-128 (not per-32).
// `get_mega_moe_config_sm90` drives the pingpong kernel (BLOCK_M=64) and
// `get_mega_moe_cooperative_config_sm90` the cooperative kernel (BLOCK_M=128);
// this config is what the SM90 host runtimes read.
// ============================================================================

struct MegaMoESM90Config {
    // Block tiling (no STORE_BLOCK_M / SF_BLOCK_M concept on SM90)
    int block_m, block_n, block_k;

    // Cluster size for TMA multicast (1 or 2). Multicast is on A.
    int cluster_size;

    // Pool capacity and SF-padded token count (SF is per-128 float on SM90)
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors (acts/weights). Both are 128B on FP8 K-major.
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // L2 GEMM uses N-major block scheduling (weight stays resident in L2 while
    // sweeping m) — enabled at large tokens-per-expert to cut weight L2 thrash.
    bool l2_nmajor_schedule;

    // Experimental L1 N-major scheduling. This trades input activation reuse for
    // potential L1 weight reuse and is controlled separately from L2.
    bool l1_nmajor_schedule;

    // Experimental expert-local phase order: for each expert, run its L1 blocks
    // then its L2 blocks before advancing to the next expert in the wave.
    bool expert_local_schedule;

    // Experimental M-local phase order: for each expert, run one M block's L1
    // blocks, then that same M block's L2 blocks before advancing in M.
    bool mlocal_schedule;

    // Experimental path: B-loader warp loads weight scale factors into SMEM
    // per pipeline stage so math warpgroups do not issue global SF loads.
    bool sfb_in_smem;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout: dispatch + non-epilogue (TMA) + epilogue (math)
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", l2_nmajor_schedule=" << config.l2_nmajor_schedule
           << ", l1_nmajor_schedule=" << config.l1_nmajor_schedule
           << ", expert_local_schedule=" << config.expert_local_schedule
           << ", mlocal_schedule=" << config.mlocal_schedule
           << ", sfb_in_smem=" << config.sfb_in_smem
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static std::tuple<int, int> get_block_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    // Pingpong: fixed block_m=64 (one m64 WGMMA per tile) with exactly 2 math
    // warpgroups. The two WGs process DIFFERENT tiles, alternating via an
    // OrderedSequenceBarrier so one WG's MMA overlaps the other's epilogue.
    //
    //   12 warps = 384 threads:
    //     HW WG0: 2 dispatch + TMA A + TMA B  → dealloc<48>
    //     HW WG1: Math WG0 (leader)           → alloc<224>
    //     HW WG2: Math WG1 (follower)         → alloc<224>
    //     Register budget: 128×48 + 256×224 = 6144 + 57344 = 63488 ≤ 64512
    constexpr int block_m                = 64;
    constexpr int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static std::tuple<int, int> get_block_config_for_mega_moe_cooperative_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    // cooperative: fixed block_m=128 (M-split across two math warpgroups, each
    // owning a WG_BLOCK_M=64 band) with exactly 2 math warpgroups. Single-launch
    // fused kernel like pingpong; the only block-shape difference is block_m=128.
    constexpr int block_m                = 128;
    constexpr int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    // Reuse SM100 logic; the block-shape units are different but the wave-balancing
    // intent is identical.
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& sfb_in_smem = false) {
    constexpr int kSmemAlignment = 1024;

    // Dispatch region (same as SM100)
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (single-buffered, BLOCK_N/2 post-SwiGLU)
    // and L2 BF16, then 1024-byte aligned (matches kernel's SMEM_CD_SIZE).
    // The tile covers the full BLOCK_M rows; each warpgroup writes its own
    // WG_BLOCK_M-row slice within a single staging tile.
    const int smem_cd_l1 = block_m * (block_n / 2);  // 1 byte/elem (FP8)
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // SF on SM90:
    //   * SFA per stage must hold the larger of L1 (BLOCK_M floats, per-128 K)
    //     and L2 (2 * BLOCK_M floats, per-64 K), aligned to 128 bytes
    //   * SFB can optionally be staged by the B-loader warp into SMEM
    //     (2 floats per stage covers L1 gate/up; L2 uses only slot 0).
    const int smem_sfa_per_stage = align(2 * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = sfb_in_smem ? align(2 * static_cast<int>(sizeof(float)), 128) : 0;

    // Per-stage: A tile + B tile + SFA tile + SFB tile
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    // Barriers (8 bytes each):
    //   * dispatch: num_dispatch_warps
    //   * GEMM full + empty: 2 * num_stages
    //   * combine: 2 * num_epilogue_warps
    //   * pingpong order: 4 (OrderedSequenceBarrier<2,2>)
    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps + 4) * 8;
    const int smem_barriers_per_stage = 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    // Select max num_stages
    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int block_n = 128;
    const int block_k = 128;
    // NOTES: cluster_size=1 for SM90 in this initial implementation. Cluster=2
    // multicast on A is feasible (each pair of CTAs shares m_block, splits N),
    // but the SwiGLU/FP8-quantize epilogue would then need cross-CTA amax
    // reduction so that one per-128 SF correctly covers both 64-col halves.
    // We defer that optimisation; cluster=1 is correct and self-contained.
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave_auto = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
    // Pingpong benefits from smaller waves once enough tokens exist to amortize
    // the extra waves; keep tiny-token latency on the original heuristic.
    const int num_experts_per_wave_tuned = num_tokens >= 8
                                                ? std::min(num_experts_per_wave_auto, 8)
                                                : num_experts_per_wave_auto;
    const int num_experts_per_wave_override = get_env<int>("DG_SM90_MOE_EXPERTS_PER_WAVE", 0);
    const int num_experts_per_wave = num_experts_per_wave_override > 0
                                         ? std::min(num_experts_per_wave_override, num_experts_per_rank)
                                         : num_experts_per_wave_tuned;

    const int num_dispatch_threads = 64;
    const int num_non_epilogue_threads = 64;

    // L2 N-major scheduling: enable at large tokens-per-expert, where the expert
    // weight (L2 B operand) is large and low-reuse and the M-major order thrashes
    // L2 (measured 47% hit / 97% L2 busy). N-major keeps each weight N-column
    // resident while sweeping m. Threshold matches the megamoe_sm90 branch.
    const float tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    // DIAGNOSTIC: DG_SM90_MOE_NMAJOR overrides the N-major heuristic (-1=auto, 0=off, 1=on).
    const int nmajor_override = get_env<int>("DG_SM90_MOE_NMAJOR", -1);
    const bool l2_nmajor_schedule = nmajor_override < 0
                                        ? (tokens_per_expert >= 256.0f)
                                        : (nmajor_override != 0);
    const bool l1_nmajor_schedule = false;
    const bool expert_local_schedule = false;
    const bool mlocal_schedule = false;
    const bool sfb_in_smem = false;

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        sfb_in_smem);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        l2_nmajor_schedule, l1_nmajor_schedule, expert_local_schedule, mlocal_schedule, sfb_in_smem,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

// cooperative config: identical to `get_mega_moe_config_sm90` but forces
// block_m=128 (M-split fused kernel). Single-launch fused like pingpong.
static MegaMoESM90Config get_mega_moe_cooperative_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_cooperative_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int block_n = 128;
    const int block_k = 128;
    // NOTES: cluster_size=1 for SM90 in this initial implementation. Cluster=2
    // multicast on A is feasible (each pair of CTAs shares m_block, splits N),
    // but the SwiGLU/FP8-quantize epilogue would then need cross-CTA amax
    // reduction so that one per-128 SF correctly covers both 64-col halves.
    // We defer that optimisation; cluster=1 is correct and self-contained.
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave_auto = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
    const float tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    // On the H200/L20X default shape (32 local experts), large-token cooperative
    // runs are more stable when the whole local expert set is one wave.
    // For 384-expert PR360 models (48 local experts), the shared auto rule
    // picks tiny waves at long tokens/rank. Current SFB-on testing showed the
    // 4096-token point is noisy/neutral, while 8192 is consistently faster with
    // model-specific larger waves, so gate this tuning by tokens/expert >= 1024.
    const bool is_v4_pro_384 = (num_experts_per_rank == 48 and hidden == 7168 and
                                intermediate_hidden == 3072 and num_topk == 6);
    const bool is_mimo_pro_384 = (num_experts_per_rank == 48 and hidden == 6144 and
                                  intermediate_hidden == 2048 and num_topk == 8);
    const int num_experts_per_wave_tuned = (is_v4_pro_384 and tokens_per_expert >= 1024.0f)
                                               ? 16
                                               : (is_mimo_pro_384 and tokens_per_expert >= 1024.0f)
                                                     ? 24
                                                     : (num_experts_per_rank == 48 and tokens_per_expert >= 1024.0f)
                                                           ? 24
                                                           : (num_tokens >= 1024 and num_experts_per_rank == 32)
                                                                 ? 32
                                                                 : num_experts_per_wave_auto;
    const int num_experts_per_wave_override = get_env<int>("DG_SM90_MOE_EXPERTS_PER_WAVE", 0);
    const int num_experts_per_wave = num_experts_per_wave_override > 0
                                         ? std::min(num_experts_per_wave_override, num_experts_per_rank)
                                         : num_experts_per_wave_tuned;

    const int num_dispatch_threads = 64;
    const int num_non_epilogue_threads = 64;

    // L2 N-major scheduling: enable at large tokens-per-expert, where the expert
    // weight (L2 B operand) is large and low-reuse and the M-major order thrashes
    // L2 (measured 47% hit / 97% L2 busy). N-major keeps each weight N-column
    // resident while sweeping m. Threshold matches the megamoe_sm90 branch.
    // DIAGNOSTIC: DG_SM90_MOE_NMAJOR overrides the N-major heuristic (-1=auto, 0=off, 1=on).
    const int nmajor_override = get_env<int>("DG_SM90_MOE_NMAJOR", -1);
    const bool l2_nmajor_schedule = nmajor_override < 0
                                        ? (tokens_per_expert >= 256.0f)
                                        : (nmajor_override != 0);
    // L1 scheduling is model-specific for the long 384-expert PR360 cases.
    // MiMo-Pro benefits from keeping each L1 weight N tile hot while sweeping M.
    // V4 Pro only won when L1 N-major was paired with expert-local L1->L2
    // scheduling, so keep both decisions tied to the measured model shape.
    const bool auto_mimo_pro_l1_nmajor = is_mimo_pro_384 and tokens_per_expert >= 1024.0f;
    const bool auto_v4_pro_expert_local = is_v4_pro_384 and tokens_per_expert >= 1024.0f;
    // DIAGNOSTIC: DG_SM90_MOE_L1_NMAJOR overrides the experimental L1 order (-1=auto, 0=off, 1=on).
    const int l1_nmajor_override = get_env<int>("DG_SM90_MOE_L1_NMAJOR", -1);
    const bool l1_nmajor_schedule = l1_nmajor_override < 0
                                       ? (auto_mimo_pro_l1_nmajor or auto_v4_pro_expert_local)
                                       : (l1_nmajor_override != 0);
    // DIAGNOSTIC: DG_SM90_MOE_EXPERT_LOCAL overrides per-expert L1->L2 scheduling (-1=auto, 0=off, 1=on).
    const int expert_local_override = get_env<int>("DG_SM90_MOE_EXPERT_LOCAL", -1);
    const bool expert_local_schedule = expert_local_override < 0
                                           ? auto_v4_pro_expert_local
                                           : (expert_local_override != 0);
    // DIAGNOSTIC: DG_SM90_MOE_MLOCAL tests per-M-block L1->L2 scheduling.
    const bool mlocal_schedule = get_env<int>("DG_SM90_MOE_MLOCAL", 0) != 0;
    // DG_SM90_MOE_SFB_SMEM controls staging weight SF through SMEM:
    //   -1/unspecified: auto, enabled only in the measured medium-long band.
    //    0: force off, 1: force on.
    // PR360-standard 8-rank testing showed stable wins from the cooperative
    // entry band (tokens_per_expert around 32) through 4096 and a regression at
    // 8192, where producer-side work dominates the tiny scale-load saving.
    const int sfb_smem_override = get_env<int>("DG_SM90_MOE_SFB_SMEM", -1);
    const bool sfb_in_smem = sfb_smem_override < 0
                                  ? (tokens_per_expert >= 32.0f and tokens_per_expert <= 4096.0f)
                                  : (sfb_smem_override != 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        sfb_in_smem);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        l2_nmajor_schedule, l1_nmajor_schedule, expert_local_schedule, mlocal_schedule, sfb_in_smem,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(cooperative, num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
