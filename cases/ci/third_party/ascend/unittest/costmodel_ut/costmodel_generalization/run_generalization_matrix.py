#!/usr/bin/env python3
"""Run the cross-operator cost-model matrix with isolated compiler evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[6]))
from costmodel_tests.profiles import prepare_profile
from costmodel_tests.comparison_modes import comparison_modes


def capture(command: list[str], cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"command": command, "returncode": completed.returncode, "output": completed.stdout}


def git_state(path: Path) -> dict:
    if not path.is_dir():
        return {"root": str(path), "sha": None, "status_porcelain": None}
    revision = capture(["git", "rev-parse", "HEAD"], cwd=path)
    status = capture(["git", "status", "--porcelain=v1"], cwd=path)
    diff = capture(["git", "diff", "--binary", "HEAD", "--"], cwd=path)
    revision_text = revision["output"].strip() if revision["returncode"] == 0 else None
    status_lines = (status["output"].splitlines() if status["returncode"] == 0 else None)
    state_material = json.dumps(
        {
            "revision": revision_text,
            "status": status_lines,
            "diff": diff["output"] if diff["returncode"] == 0 else None,
        },
        sort_keys=True,
    ).encode("utf-8")
    return {
        "root": str(path),
        "sha": revision_text,
        "status_porcelain": status_lines,
        "diff_sha256": hashlib.sha256((diff["output"] if diff["returncode"] == 0 else "").encode("utf-8")).hexdigest(),
        "source_state_sha256": hashlib.sha256(state_material).hexdigest(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compiler_state(expected: Path) -> dict:
    resolved_expected = expected.resolve()
    selected_text = shutil.which("bishengir-compile")
    if selected_text is None:
        raise RuntimeError("bishengir-compile is not on PATH")
    selected = Path(selected_text).resolve()
    if selected != resolved_expected:
        raise RuntimeError("wrong bishengir-compile selected: "
                           f"expected {resolved_expected}, got {selected}; "
                           "load the intended compiler environment before running the matrix")
    version = capture([str(selected), "--version"])
    if version["returncode"] != 0:
        raise RuntimeError(f"failed to query {selected}: {version['output']}")
    return {
        "path": str(selected),
        "version": version["output"].strip(),
        "size_bytes": selected.stat().st_size,
        "mtime_ns": selected.stat().st_mtime_ns,
    }


def select_timing_source(requested: str, measurement_count: int, limit: int) -> str:
    if requested == "auto":
        return "profiler" if measurement_count <= limit else "event"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE.parents[4] / "manifest.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="required for execution; omitted only with --list-cases",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list every case in manifest order, then exit without requiring an NPU",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="CASE_NAME",
        help="run only this exact manifest case; repeat to select multiple cases",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        help=("optional profiler scratch root; compilation evidence and caches "
              "remain under --output-root"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="explicit baseline; otherwise use the profile shipped by the installed TA",
    )
    parser.add_argument(
        "--expected-bishengir",
        type=Path,
        default=Path(shutil.which("bishengir-compile") or "bishengir-compile"),
        help="fail before compiling if PATH resolves a different compiler",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--triton-root", type=Path, default=REPO_ROOT,
                        help="source checkout of the compiler under test (for evidence)")
    parser.add_argument("--npuir-root", type=Path, default=REPO_ROOT / "third_party/ascend/AscendNPU-IR",
                        help="AscendNPU-IR source checkout (for evidence)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--measurement-processes",
        type=int,
        default=3,
        help="fresh processes used for the median profiler measurement",
    )
    parser.add_argument(
        "--measurement-selection",
        choices=("minimum", "median", "minimum-profiler", "median-profiler"),
        default="median",
    )
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel compiler subprocesses; NPU measurement remains serialized",
    )
    parser.add_argument(
        "--compile-batch-size",
        type=int,
        default=0,
        help="prepared caches retained at once; 0 uses --compile-workers",
    )
    parser.add_argument(
        "--timing-source",
        choices=("auto", "profiler", "event"),
        default="auto",
        help="auto uses profiler for small selections and device events for large matrices",
    )
    parser.add_argument(
        "--profiler-max-measurements",
        type=int,
        default=64,
        help="maximum selected route measurements for auto profiler mode",
    )
    parser.add_argument(
        "--disable-profiler",
        action="store_true",
        help="deprecated alias for --timing-source event",
    )
    parser.add_argument("--include-explicit", action="store_true")
    parser.add_argument("--comparison-modes", action="store_true",
                        help="Auto/SIMD/SIMT F1 and manifest-declared Mix references")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="ROUTE[:FACTOR]",
        help=("run only the requested route/factor pair; repeat for multiple pairs "
              "(for example --mode simd or --mode simt_only:4)"),
    )
    parser.add_argument("--only-family", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= --shard-index < --shard-count")
    if args.measurement_processes <= 0:
        parser.error("--measurement-processes must be positive")
    if args.compile_workers <= 0:
        parser.error("--compile-workers must be positive")
    if args.compile_batch_size < 0:
        parser.error("--compile-batch-size may not be negative")
    if args.profiler_max_measurements < 0:
        parser.error("--profiler-max-measurements may not be negative")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        case["driver"] = str((args.manifest.resolve().parent / case["driver"]).resolve())
    cases_by_name = {case["name"]: case for case in manifest["cases"]}
    if len(cases_by_name) != len(manifest["cases"]):
        parser.error("manifest contains duplicate case names")
    unknown_cases = sorted(set(args.case) - set(cases_by_name))
    if unknown_cases:
        parser.error(f"unknown --case value(s): {', '.join(unknown_cases)}")
    if args.list_cases:
        for index, case in enumerate(manifest["cases"]):
            print(f"{index:03d}\t{case['name']}\t{case['family']}\t"
                  f"{case['split']}\t{case['driver']}")
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --list-cases is used")
    selected_cases = set(args.case)
    profile_snapshot = args.output_root / "costmodel_profile.json"
    try:
        args.profile = prepare_profile(args.profile, profile_snapshot, os.environ)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.profile_root:
        args.profile_root.mkdir(parents=True, exist_ok=True)
    # The compiler and audit must use the same profile, even with --profile.
    os.environ["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(args.profile.resolve())
    os.environ["PYTHONPATH"] = str(HERE) + (os.pathsep + os.environ["PYTHONPATH"]
                                         if os.environ.get("PYTHONPATH") else "")
    manifest_snapshot = args.output_root / "generalization_matrix.json"
    manifest_snapshot.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Freeze and record the two debug-affecting values once for all routes.
    os.environ.setdefault("TRITON_DEBUG", "0")
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "0")
    run_started_utc = datetime.now(timezone.utc).isoformat()
    toolchain = {
        "triton_ascend": git_state(args.triton_root),
        "npuir": git_state(args.npuir_root),
        "bishengir_compile": compiler_state(args.expected_bishengir),
    }
    device_snapshot = capture(["npu-smi", "info"])

    modes = []
    for mode_spec in args.mode:
        route, separator, factor_text = mode_spec.partition(":")
        if route not in {"auto", "simd", "simt_only", "mixed"}:
            parser.error(f"unsupported route in --mode: {route}")
        try:
            factor = int(factor_text) if separator else 1
        except ValueError:
            parser.error(f"invalid factor in --mode: {mode_spec}")
        if factor <= 0:
            parser.error(f"factor must be positive in --mode: {mode_spec}")
        modes.append((route, factor))
    if args.comparison_modes and (args.mode or args.include_explicit):
        parser.error("--comparison-modes cannot be combined with --mode/--include-explicit")
    if not modes:
        modes = [("auto", 1)]
    if args.include_explicit and not args.mode:
        modes += [("simd", 1), ("simt_only", 1), ("simt_only", 2), ("simt_only", 4)]

    global_measurement_count = 0
    for case in manifest["cases"]:
        if selected_cases and case["name"] not in selected_cases:
            continue
        if args.only_family and case["family"] not in args.only_family:
            continue
        global_measurement_count += len(comparison_modes(case)) if args.comparison_modes else len(modes)
        if args.include_explicit and not args.mode:
            global_measurement_count += len(case.get("extra_modes", []))

    work_items = []
    for case_index, case in enumerate(manifest["cases"]):
        if selected_cases and case["name"] not in selected_cases:
            continue
        if args.only_family and case["family"] not in args.only_family:
            continue
        if case_index % args.shard_count != args.shard_index:
            continue
        case_modes = comparison_modes(case) if args.comparison_modes else list(modes)
        if args.include_explicit and not args.mode:
            case_modes += [
                (entry["route"], int(entry.get("factor", 1)))
                for entry in case.get("extra_modes", [])
            ]
        for route, factor in case_modes:
            name = f"{case['name']}__{route}_f{factor}"
            report = args.output_root / f"{name}.route.json"
            report.unlink(missing_ok=True)
            driver = Path(case["driver"])
            factor_flag = case.get("factor_argument", "--factor")
            command = [
                sys.executable,
                str(driver),
                "--route",
                route,
                factor_flag,
                str(factor),
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--report",
                str(report),
                *case.get("arguments", []),
            ]
            work_items.append(
                {
                    "name": name,
                    "family": case["family"],
                    "split": case["split"],
                    "route": route,
                    "factor": factor,
                    "report": report,
                    "command": command,
                    "profiler_enabled": bool(case.get("profiler", True)),
                }
            )
    if not work_items:
        parser.error("no matrix measurements selected")

    requested_timing_source = "event" if args.disable_profiler else args.timing_source
    timing_source = select_timing_source(
        requested_timing_source,
        global_measurement_count,
        args.profiler_max_measurements,
    )
    profile_parent = args.profile_root or args.output_root
    for item in work_items:
        item["timing_source"] = (
            "profiler"
            if timing_source == "profiler" and item["profiler_enabled"]
            else "event"
        )
        if item["timing_source"] == "profiler":
            item["command"] += [
                "--profile-dir",
                str(profile_parent / item["name"] / "profiler"),
            ]

    compile_batch_size = args.compile_batch_size or args.compile_workers
    records = []
    summary_context = {
        "schema": "triton-costmodel-generalization-run-v1",
        "started_utc": run_started_utc,
        "device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "environment": {
            "TRITON_DEBUG": os.environ.get("TRITON_DEBUG"),
            "TRITON_DISABLE_LINE_INFO": os.environ.get("TRITON_DISABLE_LINE_INFO"),
        },
        "toolchain": toolchain,
        "device_snapshot": device_snapshot,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "manifest": str(args.manifest),
        "manifest_snapshot": str(manifest_snapshot.resolve()),
        "manifest_sha256": sha256(manifest_snapshot),
        "selected_cases": list(args.case),
        "profile_snapshot": str(profile_snapshot.resolve()),
        "profile_metadata": str(args.output_root / "profile_metadata.json"),
        "profile_sha256": sha256(profile_snapshot),
        "execution": {
            "compile_workers": args.compile_workers,
            "compile_batch_size": compile_batch_size,
            "measurement_serialized": True,
            "timing_source_requested": requested_timing_source,
            "timing_source_effective": timing_source,
            "profiler_max_measurements": args.profiler_max_measurements,
            "selected_measurement_count": len(work_items),
            "global_selected_measurement_count": global_measurement_count,
        },
    }

    def collector_command(item: dict, phase: str) -> list[str]:
        collector = [
            sys.executable,
            str(HERE / "collect_backend_evidence.py"),
            "--name",
            item["name"],
            "--output-root",
            str(args.output_root),
            "--route-report",
            str(item["report"]),
            "--timing-source",
            item["timing_source"],
        ]
        if phase == "prepare":
            collector += ["--prepare-only", "--prepare-without-profiler"]
        elif phase == "measure":
            collector += [
                "--reuse-prepared",
                "--measure-in-fresh-process",
                "--measurement-processes",
                str(args.measurement_processes),
                "--measurement-selection",
                args.measurement_selection,
            ]
        else:
            raise ValueError(f"unknown collector phase: {phase}")
        return collector + ["--", *item["command"]]

    def prepare(item: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            collector_command(item, "prepare"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    prepared_count = 0
    total_count = len(work_items)
    for batch_start in range(0, total_count, compile_batch_size):
        batch = work_items[batch_start:batch_start + compile_batch_size]
        prepare_results = {}
        with ThreadPoolExecutor(max_workers=args.compile_workers) as executor:
            futures = {executor.submit(prepare, item): item for item in batch}
            for future in as_completed(futures):
                item = futures[future]
                completed = future.result()
                prepare_results[item["name"]] = completed
                prepared_count += 1
                status = "PASS" if completed.returncode == 0 else "FAIL"
                print(
                    f"[compile {prepared_count}/{total_count}] {item['name']}: {status}",
                    flush=True,
                )
                if completed.returncode:
                    print(completed.stdout, file=sys.stderr, flush=True)

        # Compilation is parallel, but NPU measurement stays serialized.
        for item in batch:
            prepared = prepare_results[item["name"]]
            if prepared.returncode:
                returncode = prepared.returncode
            else:
                completed = subprocess.run(
                    collector_command(item, "measure"), check=False
                )
                returncode = completed.returncode
            shutil.rmtree(
                args.output_root / item["name"] / "cache", ignore_errors=True
            )
            records.append({
                "name": item["name"],
                "family": item["family"],
                "split": item["split"],
                "route": item["route"],
                "factor": item["factor"],
                "timing_source": item["timing_source"],
                "prepare_returncode": prepared.returncode,
                "returncode": returncode,
                "evidence_dir": str((args.output_root / item["name"]).resolve()),
            })
            (args.output_root / "matrix_summary.partial.json").write_text(
                json.dumps(
                    {**summary_context, "records": records},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    toolchain_end = {
        "triton_ascend": git_state(args.triton_root),
        "npuir": git_state(args.npuir_root),
        "bishengir_compile": compiler_state(args.expected_bishengir),
    }
    source_stable = all(toolchain[name].get("source_state_sha256") == toolchain_end[name].get("source_state_sha256")
                        for name in ("triton_ascend", "npuir"))
    compiler_stable = (toolchain["bishengir_compile"] == toolchain_end["bishengir_compile"])
    summary = {
        **summary_context,
        "toolchain_end": toolchain_end,
        "source_stable_during_run": source_stable,
        "compiler_stable_during_run": compiler_stable,
        "records": records,
    }
    (args.output_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(1 if any(row["returncode"] for row in records) or not source_stable or not compiler_stable else 0)


if __name__ == "__main__":
    main()
