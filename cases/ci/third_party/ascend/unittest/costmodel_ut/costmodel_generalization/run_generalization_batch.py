#!/usr/bin/env python3
"""Run manifest entries concurrently, retry only failures, and audit final attempts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SINGLE_CASE_SCRIPT = HERE / "run_generalization_case.py"
MATRIX_SCRIPT = HERE / "run_generalization_matrix.py"
ROOT = HERE.parents[6]
sys.path.insert(0, str(ROOT))
from costmodel_tests.manifests import load_manifest, select_cases
from costmodel_tests.profiles import prepare_profile, profile_report_lines
from costmodel_tests.comparison_modes import comparison_modes, matrix_mode_flags
from costmodel_tests.pytest_runner import run_pytest_case


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_batch_timing_source(requested: str, measurement_count: int, profiler_limit: int) -> str:
    if requested == "auto":
        return "profiler" if measurement_count <= profiler_limit else "event"
    return requested


def select_case_batch_size(requested: int, timing_source: str, retries: int,
                           benchmark_case_count: int, jobs: int) -> int:
    if requested:
        return requested
    if timing_source == "event" and retries == 0 and benchmark_case_count > jobs:
        # Two waves amortize fixed matrix/toolchain startup while retaining
        # enough tasks for dynamic load balancing across devices.
        return (benchmark_case_count + 2 * jobs - 1) // (2 * jobs)
    return 1


def parse_busy_devices(npu_smi_output: str) -> set[str]:
    """Return physical device IDs that have a process in `npu-smi info`."""
    busy = set()
    for line in npu_smi_output.splitlines():
        match = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match:
            busy.add(match.group(1))
    return busy


def require_idle_devices(devices: list[str]) -> None:
    completed = subprocess.run(
        ["npu-smi", "info"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot verify that requested devices are idle: {completed.stdout.strip()}"
        )
    busy = parse_busy_devices(completed.stdout)
    selected_busy = [device for device in devices if device in busy]
    if selected_busy:
        raise RuntimeError(
            "refusing to use busy NPU device(s): " + ",".join(selected_busy)
        )


def partition_cases(cases: list[dict], batch_size: int) -> list[list[dict]]:
    """Group benchmark cases while keeping pytest entries isolated."""
    groups = []
    pending = []
    for case in cases:
        if case.get("runner", "benchmark") != "benchmark" or batch_size == 1:
            if pending:
                groups.append(pending)
                pending = []
            groups.append([case])
            continue
        pending.append(case)
        if len(pending) == batch_size:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    return groups


def case_record_count(case: dict, validation_scope: str) -> int:
    if validation_scope == "compare":
        return len(comparison_modes(case))
    return 1 if validation_scope == "auto" else 5 + len(case.get("extra_modes", []))


def precision_from_records(records: list[dict]) -> bool:
    try:
        evidence = [read_json(Path(record["evidence_dir"]) / "evidence.json") for record in records]
        return bool(evidence) and all(
            (entry.get("benchmark") or {}).get("correct") is True for entry in evidence
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_case_batch(cases: list[dict], batch_index: int, args: argparse.Namespace,
                   env: dict, device: str | None) -> list[dict]:
    """Run several cases in one matrix process to amortize toolchain/NPU startup."""
    output = args.output_root / "batches" / f"batch_{batch_index:03d}" / "attempt_01"
    output.mkdir(parents=True, exist_ok=True)
    batch_manifest = {**args.manifest_data, "cases": cases}
    manifest_path = output / "batch_manifest.json"
    write_json(manifest_path, batch_manifest)
    log = output.parent / "attempt_01.log"
    command = [
        sys.executable, str(MATRIX_SCRIPT),
        "--manifest", str(manifest_path),
        *matrix_mode_flags(args.validation_scope),
        "--profile", str(args.output_root / "costmodel_profile.json"),
        "--expected-bishengir", str(args.expected_bishengir),
        "--output-root", str(output),
        "--warmup", str(args.warmup), "--repeats", str(args.repeats),
        "--measurement-processes", str(args.measurement_processes),
        "--timing-source", args.effective_timing_source,
        "--compile-workers", str(args.compile_workers),
        "--triton-root", str(args.triton_root.resolve()),
        "--npuir-root", str(args.npuir_root.resolve()),
    ]
    child_env = env.copy()
    child_env.setdefault("TRITON_DEBUG", "0")
    child_env.setdefault("TRITON_DISABLE_LINE_INFO", "1")
    if device is not None:
        child_env["ASCEND_RT_VISIBLE_DEVICES"] = device
    names = [case["name"] for case in cases]
    print(f"START_BATCH {batch_index:03d} cases={len(cases)} device={device or 'inherited'} "
          f"first={names[0]} last={names[-1]}", flush=True)
    started = time.monotonic()
    process_status = 1
    process_error = None
    with log.open("w", encoding="utf-8") as stream:
        try:
            process_status = subprocess.run(
                command, env=child_env, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        except OSError as error:
            process_error = str(error)
            stream.write(process_error + "\n")
    try:
        matrix = read_json(output / "matrix_summary.json")
    except (OSError, ValueError, TypeError):
        matrix = {}
    records_by_case = {name: [] for name in names}
    for record in matrix.get("records", []):
        base = record.get("name", "").split("__", 1)[0]
        if base in records_by_case:
            records_by_case[base].append(record)
    stable = (matrix.get("source_stable_during_run") is True
              and matrix.get("compiler_stable_during_run") is True)
    elapsed = time.monotonic() - started
    results = []
    for case in cases:
        name = case["name"]
        records = records_by_case[name]
        expected = case_record_count(case, args.validation_scope)
        precision_pass = precision_from_records(records)
        passed = (stable and len(records) == expected and precision_pass
                  and all(record.get("returncode") == 0 for record in records))
        failure = None
        if not passed:
            failure = process_error or (
                "batch_matrix_failed" if process_status else "missing_or_invalid_case_result"
            )
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name} batch={batch_index:03d}", flush=True)
        results.append({
            "case": name,
            "runner": "benchmark",
            "status": status,
            "device": device,
            "attempt_count": 1,
            "final_output_root": str(output),
            "precision_pass": precision_pass,
            "timing_pass": None,
            "audit_results": [],
            "attempts": [{
                "attempt": 1,
                "returncode": 0 if passed else (process_status or 1),
                "status": status,
                "failure_reason": failure,
                "elapsed_seconds": elapsed,
                "output_root": str(output),
                "log": str(log),
                "batch_index": batch_index,
            }],
        })
    return results


def run_case(case: dict, args: argparse.Namespace, env: dict, device: str | None) -> dict:
    """One case owns an isolated directory; successful cases are never rerun."""
    if case.get("runner", "benchmark") == "pytest":
        return run_pytest_case(case, args, env, device)
    attempts = []
    expected = case_record_count(case, args.validation_scope)
    for number in range(1, args.retries + 2):
        output = args.output_root / "cases" / case["name"] / f"attempt_{number:02d}"
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / f"attempt_{number:02d}.log"
        command = [
            sys.executable, str(SINGLE_CASE_SCRIPT), case["name"],
            "--manifest", str(args.output_root / "generalization_matrix.json"),
            "--validation-scope", args.validation_scope,
            "--profile", str(args.output_root / "costmodel_profile.json"),
            "--expected-bishengir", str(args.expected_bishengir),
            "--output-root", str(output),
            "--triton-root", str(args.triton_root.resolve()),
            "--npuir-root", str(args.npuir_root.resolve()),
            "--warmup", str(args.warmup), "--repeats", str(args.repeats),
            "--measurement-processes", str(args.measurement_processes),
            "--timing-source", args.effective_timing_source,
        ]
        child_env = env.copy()
        if device is not None:
            child_env["ASCEND_RT_VISIBLE_DEVICES"] = device
        print(f"START {case['name']} attempt={number}/{args.retries + 1} device={device or 'inherited'}", flush=True)
        started = time.monotonic()
        failure = None
        with log.open("w", encoding="utf-8") as stream:
            try:
                status = subprocess.run(command, env=child_env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False).returncode
            except OSError as error:
                status = 1
                failure = str(error)
                stream.write(failure + "\n")
        audit, matrix = {}, {}
        try:
            summary = read_json(output / "single_case_summary.json")
            audit = read_json(output / "generalization_audit.json")
            matrix = read_json(output / "matrix_summary.json")
            passed = (
                status == 0 and summary.get("status") == "PASS" and audit.get("all_pass") is True
                and audit.get("expected_record_count") == expected
                and audit.get("audited_record_count") == expected
                and len(matrix.get("records", [])) == expected
                and all(r.get("returncode") == 0 for r in matrix["records"])
            )
            if not passed:
                failure = failure or "runner_or_case_audit_failed"
        except (OSError, ValueError, TypeError, KeyError) as error:
            passed = False
            failure = f"missing_or_invalid_result: {error}"
        attempts.append({
            "attempt": number, "returncode": status, "status": "PASS" if passed else "FAIL",
            "failure_reason": failure, "elapsed_seconds": time.monotonic() - started,
            "output_root": str(output), "log": str(log),
        })
        print(f"{'PASS' if passed else 'FAIL'} {case['name']} attempt={number}", flush=True)
        if passed:
            break
        if number <= args.retries:
            print(f"RETRY {case['name']} ({number}/{args.retries})", flush=True)
    precision_pass = precision_from_records(matrix.get("records", []))
    audit_results = audit.get("results", [])
    gated_results = [result for result in audit_results if result.get("timing_gate_applied") is True]
    timing_pass = (all(result.get("pass") is True for result in gated_results) if gated_results else None)
    return {"case": case["name"], "runner": "benchmark", "status": attempts[-1]["status"], "device": device,
            "attempt_count": len(attempts), "final_output_root": attempts[-1]["output_root"],
            "precision_pass": precision_pass, "timing_pass": timing_pass,
            "audit_results": audit_results, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "cases/ci/manifest.json")
    parser.add_argument("--case", action="append", default=[], help="exact case name; repeat to select a batch")
    parser.add_argument("--validation-scope", choices=("auto", "all", "compare"), default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="maximum concurrent cases (routes within a case stay serial)")
    parser.add_argument("--retries", type=int, default=3, help="extra attempts after failure; 0 disables retries")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", help="shared ASCEND_RT_VISIBLE_DEVICES for all workers")
    devices.add_argument("--devices", help=argparse.SUPPRESS)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-bishengir", type=Path)
    parser.add_argument("--triton-root", type=Path, default=ROOT)
    parser.add_argument("--npuir-root", type=Path, default=ROOT / "third_party/ascend/AscendNPU-IR")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measurement-processes", type=int)
    parser.add_argument(
        "--compile-workers", type=int, default=8,
        help="parallel prepare/compiler processes inside each batched matrix worker",
    )
    parser.add_argument("--timing-source", choices=("auto", "profiler", "event"), default="auto")
    parser.add_argument("--profiler-max-measurements", type=int, default=64)
    parser.add_argument(
        "--require-idle-devices", action="store_true",
        help="query npu-smi once and refuse to start if a selected physical device is busy",
    )
    parser.add_argument(
        "--case-batch-size", type=int, default=0,
        help=("benchmark cases per persistent matrix process; 0 automatically uses 10 for "
              "large event-only runs without retries"),
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.retries < 0:
        parser.error("--jobs must be positive and --retries must be nonnegative")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("invalid warmup/repeats")
    if args.measurement_processes is not None and args.measurement_processes < 1:
        parser.error("--measurement-processes must be positive")
    if args.compile_workers < 1:
        parser.error("--compile-workers must be positive")
    if args.profiler_max_measurements < 0:
        parser.error("--profiler-max-measurements may not be negative")
    if args.case_batch_size < 0:
        parser.error("--case-batch-size may not be negative")
    pool = [d.strip() for d in args.devices.split(",")] if args.devices is not None else None
    if pool is not None and (not all(pool) or len(set(pool)) != len(pool) or args.jobs > len(pool)):
        parser.error("--devices must be unique/nonempty and contain at least --jobs devices")
    try:
        manifest = load_manifest(args.manifest)
        selected = select_cases(manifest, args.case)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if not selected:
        parser.error("manifest cases must be nonempty")
    measurement_count = sum(
        0 if case.get("runner", "benchmark") == "pytest" else
        case_record_count(case, args.validation_scope)
        for case in selected
    )
    benchmark_case_count = sum(case.get("runner", "benchmark") == "benchmark"
                               for case in selected)
    args.effective_timing_source = select_batch_timing_source(
        args.timing_source, measurement_count, args.profiler_max_measurements)
    if args.measurement_processes is None:
        # Device-event benchmarks already collect several event samples in one
        # process and synchronize once after the repeat group.  Extra fresh
        # processes add startup/compile overhead without improving that timing
        # protocol.  Profiler measurements retain the three-process median.
        args.measurement_processes = 1 if args.effective_timing_source == "event" else 3
    if args.case_batch_size > 1 and args.retries:
        parser.error("--case-batch-size > 1 requires --retries 0")
    args.effective_case_batch_size = select_case_batch_size(
        args.case_batch_size, args.effective_timing_source, args.retries,
        benchmark_case_count, args.jobs,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(ROOT),
            str(ROOT / "costmodel_tests/benchmarks"),
            str(HERE),
            env.get("PYTHONPATH", ""),
        )
        if path
    )
    if env.get("TRITON_ASCEND_COMPILE_MODE"):
        parser.error("unset TRITON_ASCEND_COMPILE_MODE so it cannot override tested routes")
    if args.device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    requested_devices = pool if pool is not None else ([args.device] if args.device is not None else [])
    if args.require_idle_devices:
        if not requested_devices:
            parser.error("--require-idle-devices requires --device or --devices")
        try:
            require_idle_devices(requested_devices)
        except RuntimeError as error:
            parser.error(str(error))
    compiler = args.expected_bishengir or shutil.which("bishengir-compile")
    if any(c["runner"] == "benchmark" for c in selected) and (not compiler or not Path(compiler).is_file()):
        parser.error("bishengir-compile was not found; load your toolchain or pass --expected-bishengir")
    args.expected_bishengir = Path(compiler).resolve() if compiler else None
    args.output_root = args.output_root.resolve()
    if args.output_root.exists() and (not args.output_root.is_dir() or any(args.output_root.iterdir())):
        parser.error(f"output directory must be new or empty: {args.output_root}")
    functional_only = all(c["runner"] == "pytest" and
                          "--costmodel-functional-only" in c.get("arguments", []) for c in selected)
    try:
        prepare_profile(args.profile, args.output_root / "costmodel_profile.json", env,
                        devices=requested_devices, mode="functional" if functional_only else "costmodel")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    manifest["cases"] = selected
    args.manifest_data = manifest
    write_json(args.output_root / "manifest.json", manifest)
    write_json(args.output_root / "generalization_matrix.json", manifest)
    results = {}
    slots = Queue()
    for device in (pool or [None] * args.jobs):
        slots.put(device)

    def batch_summary():
        functional_only = all(c["runner"] == "pytest" and
                              "--costmodel-functional-only" in c.get("arguments", []) for c in selected)
        return {"schema": "costmodel-batch-run-v1", "jobs": args.jobs, "retries": args.retries,
                "profile_metadata": str(args.output_root / "profile_metadata.json"),
                "validation_scope": "functional" if functional_only else args.validation_scope,
                "timing_source_requested": args.timing_source,
                "timing_source_effective": args.effective_timing_source,
                "measurement_count": measurement_count,
                "measurement_processes": args.measurement_processes,
                "compile_workers": args.compile_workers,
                "case_batch_size": args.effective_case_batch_size,
                "required_idle_devices": args.require_idle_devices,
                "case_count": len(selected), "completed_case_count": len(results),
                "cases": [results[c["name"]] for c in selected if c["name"] in results]}

    groups = partition_cases(selected, args.effective_case_batch_size)
    print(f"Running {len(selected)} case(s) in {len(groups)} task(s), jobs={args.jobs}, "
          f"retries={args.retries}, timing={args.effective_timing_source}, "
          f"measurement_processes={args.measurement_processes}, "
          f"compile_workers={args.compile_workers}, "
          f"case_batch_size={args.effective_case_batch_size}", flush=True)

    def task_worker(task):
        batch_index, cases = task
        device = slots.get()
        try:
            if len(cases) > 1:
                return run_case_batch(cases, batch_index, args, env, device)
            return [run_case(cases[0], args, env, device)]
        finally:
            slots.put(device)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(task_worker, task): task for task in enumerate(groups)}
        for future in as_completed(futures):
            _, cases = futures[future]
            try:
                for result in future.result():
                    results[result["case"]] = result
            except Exception as error:
                for case in cases:
                    results[case["name"]] = {
                        "case": case["name"], "status": "FAIL", "attempts": [],
                        "attempt_count": 0, "final_output_root": None, "error": str(error),
                    }
            write_json(args.output_root / "batch_summary.partial.json", batch_summary())
    # Only each case's final attempt enters the aggregate audit. Earlier failures stay in attempts/ logs.
    records_by_name = {}
    visited_roots = set()
    for case in selected:
        result = results[case["name"]]
        if case["runner"] != "benchmark" or result["final_output_root"] is None:
            continue
        result_root = Path(result["final_output_root"])
        if result_root in visited_roots:
            continue
        visited_roots.add(result_root)
        try:
            matrix = read_json(result_root / "matrix_summary.json")
            for record in matrix["records"]:
                if record["name"] in records_by_name:
                    raise ValueError(f"duplicate final matrix record: {record['name']}")
                records_by_name[record["name"]] = record
        except (OSError, ValueError, KeyError, TypeError):
            pass  # The audit reports every missing route as FAIL.
    records = list(records_by_name.values())
    write_json(args.output_root / "matrix_summary.json", {
        "schema": "triton-costmodel-generalization-batch-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.output_root / "generalization_matrix.json"), "records": records,
        "validation_scope": args.validation_scope,
        "batch_summary": str(args.output_root / "batch_summary.json"),
    })
    benchmark_manifest = {**manifest, "cases": [c for c in selected if c["runner"] == "benchmark"]}
    audit_status = 0
    if benchmark_manifest["cases"]:
        write_json(args.output_root / "benchmark_manifest.json", benchmark_manifest)
        audit_command = [sys.executable, str(HERE / "audit_generalization_matrix.py"),
                         "--validation-scope", args.validation_scope,
                         "--evidence-root", str(args.output_root),
                         "--manifest", str(args.output_root / "benchmark_manifest.json"),
                         "--profile", str(args.output_root / "costmodel_profile.json"),
                         "--output-json", str(args.output_root / "generalization_audit.json"),
                         "--output-markdown", str(args.output_root / "generalization_audit.md")]
        with (args.output_root / "audit.log").open("w", encoding="utf-8") as stream:
            audit_status = subprocess.run(audit_command, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, check=False).returncode
    try:
        aggregate_audit = read_json(args.output_root / "generalization_audit.json")
        audit_by_case = {}
        for row in aggregate_audit.get("results", []):
            audit_by_case.setdefault(row["case"], []).append(row)
    except (OSError, ValueError, TypeError, KeyError):
        audit_by_case = {}
    for case in benchmark_manifest["cases"]:
        result = results[case["name"]]
        rows = audit_by_case.get(case["name"], [])
        expected = case_record_count(case, args.validation_scope)
        result["audit_results"] = rows
        gated = [row for row in rows if row.get("timing_gate_applied") is True]
        result["timing_pass"] = all(row.get("pass") is True for row in gated) if gated else None
        if len(rows) != expected or not all(row.get("pass") is True for row in rows):
            result["status"] = "FAIL"
            if result.get("attempts"):
                result["attempts"][-1]["status"] = "FAIL"
                result["attempts"][-1]["failure_reason"] = "aggregate_audit_failed"
    summary = batch_summary()
    passed = audit_status == 0 and all(r["status"] == "PASS" for r in results.values())
    summary.update(status="PASS" if passed else "FAIL", audit_exit_status=audit_status,
                   failed_cases=[c["name"] for c in selected if results[c["name"]]["status"] != "PASS"])
    write_json(args.output_root / "batch_summary.json", summary)
    print(f"Batch {summary['status']}; failed_cases={summary['failed_cases']}", flush=True)
    write_json(args.output_root / "report.json", summary)
    lines = ["# CostModel test report", "", f"Status: {summary['status']}", "",
             "| Case | Precision | Timing | Attempts | Status |", "|---|---|---|---:|---|"]
    lines[4:4] = profile_report_lines(args.output_root)
    for result in summary["cases"]:
        timing = result.get("timing_pass", False)
        if timing is None:
            timing = "Not checked"
        lines.append(f"| {result['case']} | {result.get('precision_pass', False)} | "
                     f"{timing} | {result['attempt_count']} | {result['status']} |")
    lines += ["", "## Timing details", "",
              "| Case / node | Kernel / route | Predicted (us) | Measured (us) | Error | Gate |",
              "|---|---|---:|---:|---:|---|"]
    for result in summary["cases"]:
        details = []
        if result.get("runner") == "pytest":
            for node in result.get("pytest_results", []):
                for launch in node.get("launches", []):
                    details.append({**launch, "case": node["nodeid"], "label": launch["kernel"]})
        else:
            details = [{**r, "label": r.get("mode", "auto")} for r in result.get("audit_results", [])]
        for detail in details:
            predicted = detail.get("predicted_us")
            measured = detail.get("measured_us")
            error = detail.get("relative_error")
            values = [f"{v:.3f}" if v is not None else "N/A" for v in (predicted, measured)]
            error_text = f"{error:+.2%}" if error is not None else "N/A"
            name = detail.get("case", result["case"]).replace("|", "\\|").replace("\n", " ")
            gate = ("PASS" if detail.get("pass") else "FAIL") if detail.get(
                "timing_gate_applied", True) else ("RECORDED" if detail.get("pass") else "FAIL")
            lines.append(f"| {name} | {detail['label']} | {values[0]} | {values[1]} | {error_text} | {gate} |")
    (args.output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.output_root / 'report.md'}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
