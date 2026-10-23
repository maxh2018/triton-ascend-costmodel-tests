"""NPU launch instrumentation for the pytest CostModel adapter.

Compile before profiling and execute each original launch exactly once. Never
replay a kernel: atomic/in-place kernels may change the test's inputs or outputs.
"""
from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path


def resolve_grid(kernel, args, kwargs, grid):
    signature = getattr(kernel, "signature", None) or inspect.signature(kernel.fn)
    bound = signature.bind_partial(*args, **{k: v for k, v in kwargs.items() if k in signature.parameters})
    bound.apply_defaults()
    meta = {**kwargs, **bound.arguments}
    dimensions = tuple(grid(meta) if callable(grid) else grid)
    if not 1 <= len(dimensions) <= 3 or any(int(n) != n or n <= 0 for n in dimensions):
        raise ValueError(f"invalid runtime grid: {dimensions!r}")
    return tuple(int(n) for n in dimensions)


def freeze(value):
    # Distinguish launch specializations, not tensor contents or allocation IDs.
    if hasattr(value, "data_ptr") and hasattr(value, "shape"):
        return (str(value.dtype), tuple(value.shape), tuple(value.stride()),
                str(value.device), int(value.data_ptr()) % 16)
    if isinstance(value, dict):
        return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    return (type(value).__qualname__, repr(value))


class NpuBackend:
    def __init__(self):
        import torch
        import torch_npu
        from triton.runtime import driver
        from triton.runtime.jit import JITFunction

        self.torch = torch
        self.profiler = torch_npu.profiler
        self.driver = driver
        self.jit_class = JITFunction

    def vector_cores(self):
        device = self.torch.npu.current_device()
        return int(self.driver.active.utils.get_device_properties(device)["num_vectorcore"])

    def measure(self, launch, output, kernel_name):
        torch, profiler = self.torch, self.profiler
        torch.npu.synchronize()
        config = profiler._ExperimentalConfig(
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            profiler_level=profiler.ProfilerLevel.Level1,
        )
        with profiler.profile(
            activities=[profiler.ProfilerActivity.NPU], record_shapes=False,
            profile_memory=False, with_stack=False,
            on_trace_ready=profiler.tensorboard_trace_handler(str(output)),
            experimental_config=config,
        ):
            result = launch()
            torch.npu.synchronize()
        rows = []
        for path in output.rglob("op_statistic.csv"):
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows.extend(csv.DictReader(stream))
        matches = [r for r in rows if kernel_name in r.get("OP Type", "")]
        if len(matches) != 1:
            raise RuntimeError(f"expected one profiler row for {kernel_name!r}, got {len(matches)}")
        measured = float(matches[0]["Avg Time(us)"])
        if not math.isfinite(measured) or measured <= 0:
            raise ValueError(f"invalid profiler time: {measured}")
        return result, {"average_us": measured, "profile_dir": str(output),
                        "op_type": matches[0]["OP Type"], "count": int(matches[0]["Count"])}


class LaunchCapture:
    def __init__(self, directory, profile, backend):
        self.directory, self.profile, self.backend = directory, profile, backend
        self.records = []
        self.by_key = {}

    def run(self, original, kernel, args, grid, warmup, kwargs):
        if warmup:
            return original(kernel, *args, grid=grid, warmup=warmup, **kwargs)
        dimensions = resolve_grid(kernel, args, kwargs, grid)
        key = (id(kernel), dimensions, freeze(args), freeze(kwargs))
        previous = self.by_key.get(key)
        if previous is not None:
            previous["launch_count"] += 1
            return original(kernel, *args, grid=dimensions, warmup=False, **previous["_options"])
        directory = self.directory / f"launch_{len(self.records) + 1:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        report = directory / "route.json"
        options = dict(kwargs)
        options.update(compile_mode="simd_simt", auto_simt_scope_mode="auto",
                       auto_simt_scope_dump=str(report), auto_simt_model_profile=str(self.profile))
        options.setdefault("superblock_factor", 1)
        options["logical_program_count_hint"] = options.get("logical_program_count_hint") or math.prod(dimensions)
        options["physical_vector_core_count_hint"] = (
            options.get("physical_vector_core_count_hint") or self.backend.vector_cores())
        record = {"kernel": kernel.fn.__name__, "grid": dimensions, "launch_count": 1,
                  "route_report": str(report), "predicted_us": None, "measured_us": None,
                  "relative_error": None, "pass": False, "failure_reasons": [], "_options": options}
        self.records.append(record)
        self.by_key[key] = record
        try:
            compiled = original(kernel, *args, grid=dimensions, warmup=True, **options)
            if hasattr(compiled, "result"):
                compiled = compiled.result()
            name = getattr(compiled, "name", None) or record["kernel"]
            result, measurement = self.backend.measure(
                lambda: original(kernel, *args, grid=dimensions, warmup=False, **options),
                directory / "profiler", name,
            )
            record["profiler"] = measurement
            record["measured_us"] = measurement["average_us"]
            try:
                route = json.loads(report.read_text(encoding="utf-8"))
                prediction = route.get("selected_route_elapsed_prediction") or {}
                predicted = float(prediction["predicted_microseconds"]) if prediction.get("applied") else None
                if predicted is None or not math.isfinite(predicted) or predicted <= 0:
                    raise ValueError("missing or invalid selected-route prediction")
                record["predicted_us"] = predicted
                record["relative_error"] = predicted / record["measured_us"] - 1
                record["pass"] = abs(record["relative_error"]) <= 0.10
                if not record["pass"]:
                    record["failure_reasons"].append("prediction_error_exceeds_10_percent")
            except (OSError, ValueError, TypeError, KeyError) as error:
                record["failure_reasons"].append(f"missing_or_invalid_prediction: {error}")
            return result
        except Exception as error:
            record["failure_reasons"].append(f"launch_or_profiler_error: {error}")
            raise

    def public_records(self):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.records]
