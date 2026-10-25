#!/usr/bin/env python3
"""Run the selected manifest; default: cases/ci/manifest.json (PR 100 cases)."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERAL = ROOT / "cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization"
sys.path.insert(0, str(ROOT))
from costmodel_tests.manifests import load_manifest, select_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run", "list"), default="run",
                        help="run the manifest (default), or list its entries")
    parser.add_argument("--manifest", type=Path,
                        help="highest priority: execute this manifest; may reference any case directories")
    parser.add_argument("--category", metavar="{ci,personal,operator_repos}",
                        help="only without --manifest: choose cases/<category>/manifest.json (default: ci)")
    parser.add_argument("--case", action="append", default=[], help="select an exact manifest name; repeat for several")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", help="one shared NPU device")
    devices.add_argument("--devices", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=int, default=1, help="concurrent entries (default: 1)")
    parser.add_argument("--retries", type=int, default=3, help="extra attempts after failure (default: 3)")
    parser.add_argument("--output-root", type=Path, help="new or empty result directory")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-bishengir", type=Path)
    parser.add_argument("--triton-root", type=Path, default=ROOT)
    parser.add_argument("--npuir-root", type=Path, default=ROOT / "third_party/ascend/AscendNPU-IR")
    parser.add_argument("--validation-scope", choices=("auto", "all"), default="auto",
                        help="default Auto; all adds explicit benchmark routes (pytest stays Auto)")
    parser.add_argument("--timing-source", choices=("auto", "profiler", "event"), default="auto",
                        help="auto uses device events for large selections")
    parser.add_argument("--profiler-max-measurements", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measurement-processes", type=int)
    parser.add_argument("--compile-workers", type=int, default=8,
                        help="parallel prepare/compiler processes per batched worker")
    parser.add_argument("--case-batch-size", type=int, default=0,
                        help="benchmark cases per persistent worker; 0 selects a safe automatic value")
    parser.add_argument("--require-idle-devices", action="store_true",
                        help="refuse to start if npu-smi reports a process on a selected device")
    parser.add_argument("--dry-run", action="store_true", help="show the execution command without running tests")
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
    # An explicit manifest is authoritative. Category only supplies a fallback
    # location, so it is neither validated nor applied in the explicit branch.
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        manifest_source, default_category = "explicit", None
    else:
        default_category = args.category if args.category is not None else "ci"
        if default_category not in ("ci", "personal", "operator_repos"):
            parser.error("--category must be ci, personal or operator_repos when --manifest is omitted")
        manifest_path = (ROOT / "cases" / default_category / "manifest.json").resolve()
        manifest_source = "category_default"
    try:
        manifest = load_manifest(manifest_path)
        cases = select_cases(manifest, args.case)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Manifest source: {manifest_source}", flush=True)
    if default_category is not None:
        print(f"Category: {default_category}", flush=True)
    print(f"Cases: {len(cases)}", flush=True)
    if manifest.get("validation_status") == "unverified":
        print("Validation: UNVERIFIED", flush=True)
    if args.action == "list":
        for case in cases:
            print(f"{case['name']}\t{case['runner']}\t{case['driver']}")
        return
    if not cases:
        parser.error("manifest is empty; register cases before running")
    output = (args.output_root or ROOT / "costmodel_test_results" / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output directory must be new or empty: {output}")
    command = [sys.executable, str(GENERAL / "run_generalization_batch.py"),
               "--manifest", str(manifest_path), "--output-root", str(output),
               "--jobs", str(args.jobs), "--retries", str(args.retries),
               "--validation-scope", args.validation_scope,
               "--timing-source", args.timing_source,
               "--profiler-max-measurements", str(args.profiler_max_measurements),
               "--case-batch-size", str(args.case_batch_size),
               "--compile-workers", str(args.compile_workers),
               "--warmup", str(args.warmup), "--repeats", str(args.repeats),
               "--triton-root", str(args.triton_root.resolve()), "--npuir-root", str(args.npuir_root.resolve())]
    if args.measurement_processes is not None:
        command += ["--measurement-processes", str(args.measurement_processes)]
    if args.require_idle_devices:
        command.append("--require-idle-devices")
    for name in args.case:
        command += ["--case", name]
    for option, value in (("--device", args.device), ("--devices", args.devices)):
        if value is not None:
            command += [option, value]
    for option, value in (("--profile", args.profile), ("--expected-bishengir", args.expected_bishengir)):
        if value is not None:
            command += [option, str(value.resolve())]
    print(f"Output: {output}", flush=True)
    if args.dry_run:
        print("run: " + json.dumps(command, ensure_ascii=False))
        return
    completed = subprocess.run(command, cwd=ROOT, check=False)
    try:
        batch = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
        passed = completed.returncode == 0 and batch["status"] == "PASS"
    except (OSError, ValueError, TypeError, KeyError):
        passed = False
    # Save launcher failures too, even if the worker could not start its run.
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "PASS" if passed else "FAIL", "category": default_category,
               "manifest_source": manifest_source,
               "manifest": str(manifest_path), "case_count": len(cases),
               "returncode": completed.returncode, "command": command,
               "batch_summary": str(output / "batch_summary.json")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['status']}: {output / 'summary.json'}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
