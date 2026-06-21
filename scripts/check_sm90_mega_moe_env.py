#!/usr/bin/env python3

import argparse
import re
import sys


def parse_version(value: str):
    match = re.match(r'(\d+)\.(\d+)', value)
    return tuple(map(int, match.groups())) if match else (0, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate an SM90 MegaMoE host')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--allow-no-peer', action='store_true')
    args = parser.parse_args()

    errors = []
    warnings = []

    try:
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:
        print(f'FAIL: unable to import the required PyTorch modules: {exc}')
        return 1

    print(f'Python: {sys.version.split()[0]}')
    print(f'PyTorch: {torch.__version__}')
    print(f'PyTorch CUDA: {torch.version.cuda}')
    print(f'NCCL available: {dist.is_nccl_available()}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    if parse_version(torch.__version__) < (2, 9):
        errors.append('PyTorch >= 2.9 is required for symmetric memory')
    if parse_version(torch.version.cuda or '0.0') < (12, 3):
        errors.append('PyTorch must be built with CUDA >= 12.3')
    if not dist.is_nccl_available():
        errors.append('PyTorch NCCL support is unavailable')
    if not torch.cuda.is_available():
        errors.append('CUDA is unavailable')
    if not hasattr(symm_mem, 'empty') or not hasattr(symm_mem, 'rendezvous'):
        errors.append('torch.distributed._symmetric_memory lacks empty/rendezvous')

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f'Visible GPUs: {device_count}')
        if device_count < args.num_gpus:
            errors.append(
                f'{args.num_gpus} visible GPUs are required, found {device_count}'
            )

        selected = min(device_count, args.num_gpus)
        for device_idx in range(selected):
            props = torch.cuda.get_device_properties(device_idx)
            capability = torch.cuda.get_device_capability(device_idx)
            memory_gib = props.total_memory / 2**30
            print(
                f'GPU {device_idx}: {props.name}, capability={capability}, '
                f'memory={memory_gib:.1f} GiB'
            )
            if capability[0] != 9:
                errors.append(
                    f'GPU {device_idx} is not SM90: capability={capability}'
                )

        no_peer_pairs = []
        for src in range(selected):
            for dst in range(selected):
                if src != dst and not torch.cuda.can_device_access_peer(src, dst):
                    no_peer_pairs.append((src, dst))
        if no_peer_pairs:
            message = (
                'CUDA peer access is unavailable for: '
                + ', '.join(f'{src}->{dst}' for src, dst in no_peer_pairs[:16])
            )
            if args.allow_no_peer:
                warnings.append(message)
            else:
                errors.append(message)

        print(f'PyTorch arch list: {torch.cuda.get_arch_list()}')

    for warning in warnings:
        print(f'WARN: {warning}')
    for error in errors:
        print(f'FAIL: {error}')

    if errors:
        return 1

    print('PASS: host satisfies the SM90 MegaMoE prerequisites')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
