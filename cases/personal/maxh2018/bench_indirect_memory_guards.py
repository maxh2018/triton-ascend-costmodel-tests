#!/usr/bin/env python3
"""29 validation-derived indirect memory guards for Ascend950PR.

Source geometry/IDs are frozen in indirect_memory_guards.json. Requires NumPy,
torch_npu and compatible Triton-Ascend/CANN/BiShengIR. Uses normal Auto or the
requested explicit route; never overrides compiler guards, core count or costs.
Eight deterministic inputs are checked; timing uses the first input only.
Store timing uses distinct, unfilled destinations, with protected boundary
canaries. Current prediction failures remain failures, not skipped tests.
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
def indirect_guard(X, Index, Mask, Values, Output, E: tl.constexpr,
                   SHAPE: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr,
                   STORE: tl.constexpr, FLAT: tl.constexpr):
    lane = tl.arange(0, E)
    logical = tl.program_id(0) * E + lane
    offset = tl.reshape(tl.load(Index + logical), SHAPE)
    if KIND == 'prefix':
        active = tl.reshape(lane < K, SHAPE) if FLAT else tl.reshape(
            lane, SHAPE) < K
    elif KIND == 'loaded':
        active = (tl.reshape(tl.load(Mask + logical) != 0, SHAPE)
                  if FLAT else tl.reshape(tl.load(Mask + logical), SHAPE) != 0)
    else:
        active = tl.full(SHAPE, True, tl.int1)
    if STORE:
        value = tl.reshape(tl.load(Values + logical), SHAPE)
        tl.store(X + offset, value, mask=active)
    else:
        value = tl.load(X + offset, mask=active, other=0)
        if FLAT:
            tl.store(Output + logical, tl.reshape(value, (E, )))
        else:
            tl.store(Output + tl.reshape(logical, SHAPE), value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard-id', required=True)
    parser.add_argument('--route',
                        choices=('auto', 'simd', 'simt_only'),
                        required=True)
    parser.add_argument('--factor', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--seed', type=int, default=90521000)
    args = parser.parse_args()
    cases = json.loads(
        Path(__file__).with_name(
            'indirect_memory_guards.json').read_text())['cases']
    case = next((c for c in cases if c['guard_id'] == args.guard_id), None)
    if case is None or args.warmup < 0 or args.repeats < 1:
        parser.error('unknown guard or invalid warmup/repeats')
    c = case
    dtype = np.dtype(c['dtype'])
    unsigned = c['dtype'].startswith('uint')
    storage = np.dtype('int' + str(c['bits'])) if unsigned else dtype
    n = c['E'] * c['F']
    domain = max(c['address_domain_elements'], 4 * n)
    contexts = []

    def payload(array):
        base = torch.from_numpy(
            np.ascontiguousarray(array).view(storage)).npu()
        return view(base)

    def view(base):
        if c.get('signature') == 'bf16':
            return base.view(torch.bfloat16)
        # Native Torch views preserve bits and expose size() to the profiler;
        # Triton TensorWrapper does not implement that profiler API.
        return base.view(getattr(torch, c['dtype'])) if unsigned else base

    def raw(value):
        return value.view(getattr(torch, str(storage)))

    cores = int(
        driver.active.utils.get_device_properties(
            torch.npu.current_device())['num_vectorcore'])
    options = dict(
        num_warps=c['W'],
        compile_mode='simd_simt' if args.route == 'auto' else args.route,
        superblock_factor=args.factor,
        logical_program_count_hint=c['F'],
        physical_vector_core_count_hint=cores,
        auto_simt_scope_mode='auto' if args.route == 'auto' else 'off',
        auto_simt_scope_dump=str(args.report) if args.route == 'auto' else '',
        compile_on_910_95=True,
        multibuffer=False,
        enable_dynamic_cv_pipeline=False)

    def invoke(context, target=None):
        x, index, mask, values, output, *_ = context
        indirect_guard[(c['F'], )](
            view(raw(x)[32:-32]) if target is None else view(target),
            index,
            mask,
            values,
            view(raw(output)[32:-32]),
            E=c['E'],
            SHAPE=tuple(c['shape']),
            K=c['K'],
            KIND=c['kind'],
            STORE=c['op'] == 'store',
            FLAT=c['mode'] == 'simt',
            **options)

    for offset in range(8):
        rng = random.Random(args.seed + offset)
        indices = np.array(
            rng.sample(range(domain), n),
            dtype=np.int32 if c['index_bits'] == 32 else np.int64)
        active = np.ones(n, dtype=np.uint8)
        if c['kind'] != 'none':
            active[:] = 0
            for pid in range(c['F']):
                positions = list(range(c['E']))
                if c['kind'] == 'loaded':
                    rng.shuffle(positions)
                active[pid * c['E'] +
                       np.array(positions[:c['K']], dtype=np.int64)] = 1
        source = (np.arange(domain + 64) % 97 + 1).astype(dtype)
        values = (np.arange(n) % 31 + 2).astype(dtype)
        context = (payload(source.copy()), torch.from_numpy(indices).npu(),
                   torch.from_numpy(active).npu(), payload(values),
                   payload(np.full(n + 64, 23,
                                   dtype=dtype)), indices, active, values)
        invoke(context)
        torch.npu.synchronize()
        expected = source.copy()
        if c['op'] == 'store':
            expected[32 + indices[active.astype(bool)]] = values[active.astype(
                bool)]
        np.testing.assert_array_equal(
            raw(context[0]).cpu().numpy().view(dtype), expected)
        observed = raw(context[4]).cpu().numpy().view(dtype)
        assert np.all(observed[:32] == 23) and np.all(
            observed[-32:] == 23), 'output canary'
        if c['op'] == 'load':
            np.testing.assert_array_equal(
                observed[32:-32],
                np.where(active, source[32 + indices], 0).astype(dtype))
        contexts.append(context)

    slots = args.warmup + args.repeats + (28 if args.profile_dir else 0)
    arena = torch.empty(
        (slots, domain + 64), dtype=getattr(
            torch, str(storage)), device='npu') if c['op'] == 'store' else None
    if arena is not None:
        arena[:, :32].fill_(23)
        arena[:, -32:].fill_(23)
    calls = 0

    def launch():
        nonlocal calls
        invoke(contexts[0], arena[calls,
                                  32:-32] if arena is not None else None)
        calls += 1

    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True)
        begin.record()
        launch()
        end.record()
        torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000)
    profiler = profile_kernel(launch, args.profile_dir,
                              'indirect_guard') if args.profile_dir else None
    torch.npu.synchronize()
    if arena is not None:
        observed = arena.cpu().numpy().view(dtype)
        assert np.all(observed[:, :32] == 23) and np.all(
            observed[:, -32:] == 23), 'store canary'
        indices, active, values = contexts[0][-3:]
        np.testing.assert_array_equal(
            observed[:, 32 + indices[active.astype(bool)]],
            np.broadcast_to(values[active.astype(bool)],
                            (slots, int(active.sum()))))
    assert calls == slots, 'profiler launch contract changed'
    print(
        json.dumps(dict(case='maxh2018_' + args.guard_id,
                        correct=True,
                        correctness_inputs=8,
                        timed_seed=args.seed,
                        elapsed_us=dict(median=statistics.median(samples)),
                        profiler=profiler),
                   indent=2))


if __name__ == '__main__':
    main()
