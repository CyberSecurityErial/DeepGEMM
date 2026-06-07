#pragma once

#include <ostream>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM90MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // Scale-factor tiles
    int sf_block_m, sf_block_n;

    // Pool capacity and SF-padded token count
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // TMA swizzle modes
    int swizzle_acts_mode, swizzle_weights_mode;

    // Expert wave scheduling
    int num_experts_per_wave;

    // Pipeline and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_mma_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const SM90MegaMoEConfig& config) {
        os << "SM90MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode
           << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_mma_threads=" << config.num_mma_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static SM90MegaMoEConfig get_sm90_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    (void) num_ranks;
    (void) num_experts;
    (void) num_experts_per_rank;
    (void) num_max_tokens_per_rank;
    (void) num_tokens;
    (void) num_topk;
    (void) hidden;
    (void) intermediate_hidden;
    (void) num_padded_sf_pool_tokens;

    DG_HOST_UNREACHABLE("SM90 MegaMoE heuristic scaffold is not implemented yet");
    return {};
}

} // namespace deep_gemm
