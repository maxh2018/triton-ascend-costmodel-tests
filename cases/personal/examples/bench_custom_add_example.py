#!/usr/bin/env python3
"""Minimal custom case: float32 vector addition on an Ascend NPU.

Source: standalone example for this test repository. Dependencies: torch,
torch_npu, Triton-Ascend with PR #2528 route support; default profile: 950PR.
"""

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver

from profile_utils import profile_kernel


@triton.jit
def custom_add_kernel(x, y, output, n, n_blocks, BLOCK: tl.constexpr):
    for block in range(tl.program_id(0), n_blocks, tl.num_programs(0)):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        value = tl.load(x + offsets, mask=mask, other=0) + tl.load(y + offsets, mask=mask, other=0)
        tl.store(output + offsets, value, mask=mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--n-elements", type=int, default=1024)
    args = parser.parse_args()
    if args.n_elements < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("n-elements/repeats must be positive; warmup must be nonnegative")

    # 1. Prepare input and reference result.
    torch.manual_seed(0)
    x = torch.randn(args.n_elements, device="npu", dtype=torch.float32)
    y = torch.randn_like(x)
    output = torch.empty_like(x)
    n_blocks = triton.cdiv(args.n_elements, 256)
    cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])

    # 2. Preserve the same computation while selecting the requested route.
    def launch():
        custom_add_kernel[(min(n_blocks, cores), )](
            x, y, output, args.n_elements, n_blocks, BLOCK=256, num_warps=4,
            compile_mode="simd_simt" if args.route == "auto" else args.route,
            superblock_factor=args.factor,
            logical_program_count_hint=n_blocks,
            physical_vector_core_count_hint=cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.route == "auto" else "",
        )

    # 3. Fail immediately if the computed result is wrong.
    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(output, x + y, rtol=1e-5, atol=1e-5)

    # 4. Warm up, measure, and print the result expected by the collector.
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
        samples.append(float(begin.elapsed_time(end)) * 1000.0)
    profiler = profile_kernel(launch, args.profile_dir, "custom_add_kernel") if args.profile_dir else None
    print(json.dumps({
        "case": f"custom_add_f32_n{args.n_elements}",
        "correct": True,
        "elapsed_us": {"median": statistics.median(samples)},
        "profiler": profiler,
    }, indent=2))


if __name__ == "__main__":
    main()
