import argparse
import math
import os
import random
from typing import Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import bench_kineto
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather


KERNEL_NAME = 'sm90_fp8_mega_moe_impl'


def quantize_grouped_fp8_weights(
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, shape_n, shape_k = weights.shape
    assert shape_n % 128 == 0 and shape_k % 128 == 0

    blocked = weights.view(
        num_groups, shape_n // 128, 128, shape_k // 128, 128
    ).float()
    scales = blocked.abs().amax(dim=(2, 4)).clamp_min(1e-4) / 448.0
    quantized = (blocked / scales[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.view_as(weights).contiguous(), scales.contiguous()


def run_case(
    args: argparse.Namespace,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    num_tokens: int,
) -> None:
    num_experts_per_rank = args.num_experts // num_ranks
    assert args.num_experts % num_ranks == 0
    assert num_tokens <= args.num_max_tokens_per_rank
    assert args.hidden % 128 == 0
    assert args.intermediate_hidden % 128 == 0
    assert args.intermediate_hidden <= 4096

    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        args.num_experts,
        args.num_max_tokens_per_rank,
        args.num_topk,
        args.hidden,
        args.intermediate_hidden,
    )

    x = torch.randn(
        (num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda'
    )
    l1_weights = torch.randn(
        (num_experts_per_rank, args.intermediate_hidden * 2, args.hidden),
        dtype=torch.bfloat16,
        device='cuda',
    ) * 0.05
    l2_weights = torch.randn(
        (num_experts_per_rank, args.hidden, args.intermediate_hidden),
        dtype=torch.bfloat16,
        device='cuda',
    ) * 0.05
    scores = torch.randn(
        (num_tokens, args.num_experts), dtype=torch.float, device='cuda'
    )
    topk_weights, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, largest=True, sorted=False
    )

    x_fp8 = per_token_cast_to_fp8(
        x, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )
    l1_weights = quantize_grouped_fp8_weights(l1_weights)
    l2_weights = quantize_grouped_fp8_weights(l2_weights)
    l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights
    )

    y = torch.empty(
        (num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda'
    )
    recv_stats = torch.zeros(
        (num_experts_per_rank,), dtype=torch.int, device='cuda'
    )
    activation_clamp = (
        args.activation_clamp
        if math.isfinite(args.activation_clamp)
        else None
    )

    def run_fused():
        sym_buffer.x[:num_tokens].copy_(x_fp8[0])
        sym_buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats=recv_stats,
            activation_clamp=activation_clamp,
            fast_math=bool(args.fast_math),
        )
        return y

    run_fused()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    dist.barrier()

    if args.ncu_profile_only:
        dist_print(
            f'NCU case: tokens={num_tokens}, hidden={args.hidden}, '
            f'intermediate={args.intermediate_hidden}',
            once_in_node=True,
        )
        sym_buffer.destroy()
        return

    trace_path = (
        os.path.join(args.trace_dir, f'sm90_mega_moe_rank{rank_idx}.json')
        if args.trace_dir
        else None
    )
    latency = bench_kineto(
        run_fused,
        KERNEL_NAME,
        num_tests=args.num_tests,
        suppress_kineto_output=True,
        trace_path=trace_path,
        barrier=dist.barrier,
    )

    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    first_local_expert = rank_idx * num_experts_per_rank
    last_local_expert = first_local_expert + num_experts_per_rank
    gathered_topk_idx[
        (gathered_topk_idx < first_local_expert)
        | (gathered_topk_idx >= last_local_expert)
    ] = -1
    local_experts = gathered_topk_idx[gathered_topk_idx >= 0]
    num_recv_tokens = int(local_experts.numel())
    num_touched_experts = int(torch.unique(local_experts).numel())

    tflops = (
        2
        * num_recv_tokens
        * args.hidden
        * args.intermediate_hidden
        * 3
        / 1e12
        / latency
    )
    dist_print(
        f'tokens={num_tokens:4d} recv={num_recv_tokens:5d} '
        f'experts={num_touched_experts:3d} latency={latency * 1e6:8.1f} us '
        f'compute={tflops:7.1f} TFLOPS',
        once_in_node=True,
    )

    dist.barrier()
    sym_buffer.destroy()


def worker(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    capability = torch.cuda.get_device_capability()
    assert capability[0] == 9, f'SM90 is required, got {capability}'

    for num_tokens in args.tokens:
        run_case(args, rank_idx, num_ranks, group, num_tokens)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 FP8 MegaMoE benchmark')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--tokens', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=3072)
    parser.add_argument('--num-experts', type=int, default=384)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--trace-dir', type=str, default='')
    parser.add_argument('--ncu-profile-only', action='store_true')
    args = parser.parse_args()

    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    if args.local_rank_idx is None:
        torch.multiprocessing.spawn(
            worker,
            args=(args.num_processes, args),
            nprocs=args.num_processes,
        )
    else:
        worker(args.local_rank_idx, args.num_processes, args)
