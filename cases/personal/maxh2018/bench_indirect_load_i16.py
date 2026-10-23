#!/usr/bin/env python3
"""Unmasked int16 gather guard for Ascend950PR; no compiler modifications.

Source: smallq_transfer_b2adede7a708bd8c,
indirect_smallq_supplement_20261020/holdout_acceptance.json (frozen validation).
One logical program, E16, shape=(16,), W2, int64 unique indices in [0,256).
Dependencies: numpy, torch, torch_npu and compatible Triton-Ascend.
Auto must select a real legal route; the script neither overrides guards nor
supplies a prediction. Compare compiler-generated reports with profiler timing.
"""

import argparse
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver
from profile_utils import profile_kernel


@triton.jit
def indirect_load_i16_kernel(X, Index, Output, E: tl.constexpr, SHAPE: tl.constexpr):
    logical = tl.program_id(0) * E + tl.arange(0, E)
    offset = tl.reshape(tl.load(Index + logical), SHAPE)
    value = tl.load(X + offset)
    tl.store(Output + logical, tl.reshape(value, (E,)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('auto', 'simd', 'simt_only'), required=True)
    parser.add_argument('--factor', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--seed', type=int, default=90521000)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be nonnegative; repeats must be positive')

    e, programs, domain = 16, 1, 256
    indices = np.array(random.Random(args.seed).sample(range(domain), e * programs), dtype=np.int64)
    source = (np.arange(domain + 64) % 97 + 1).astype(np.int16)
    x = torch.from_numpy(source).npu()
    index = torch.from_numpy(indices).npu()
    output = torch.full((e * programs + 64,), 23, dtype=torch.int16, device='npu')
    cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())['num_vectorcore'])

    def launch():
        indirect_load_i16_kernel[(programs,)](
            x[32:-32], index, output[32:-32], E=e, SHAPE=(16,), num_warps=2,
            compile_mode='simd_simt' if args.route == 'auto' else args.route,
            superblock_factor=args.factor,
            logical_program_count_hint=programs, physical_vector_core_count_hint=cores,
            auto_simt_scope_mode='auto' if args.route == 'auto' else 'off',
            auto_simt_scope_dump=str(args.report) if args.route == 'auto' else '',
            compile_on_910_95=True, multibuffer=False, enable_dynamic_cv_pipeline=False)

    def check():
        observed = output.cpu().numpy()
        np.testing.assert_array_equal(observed[32:-32], source[32 + indices])
        assert np.all(observed[:32] == 23) and np.all(observed[-32:] == 23), 'canary changed'
        np.testing.assert_array_equal(x.cpu().numpy(), source)

    launch()
    torch.npu.synchronize()
    check()
    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000)
    profiler = profile_kernel(launch, args.profile_dir, 'indirect_load_i16_kernel') if args.profile_dir else None
    torch.npu.synchronize()
    check()
    print(json.dumps(dict(case='maxh2018_indirect_load_i16_n16_w2', correct=True,
                          elapsed_us=dict(median=statistics.median(samples)), profiler=profiler,
                          seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
