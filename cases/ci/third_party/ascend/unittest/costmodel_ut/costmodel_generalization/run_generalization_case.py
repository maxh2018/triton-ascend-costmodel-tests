#!/usr/bin/env python3
"""Run and audit one manifest case (Auto selected route by default).

Load the CANN/Triton/bishengir environment before running this script on an NPU
host. Example: python run_generalization_case.py softmax_original --device 0
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[6]))
from costmodel_tests.profiles import prepare_profile


def run_logged(command: list[str], env: dict[str, str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="exact case name from cases/ci/manifest.json")
    parser.add_argument("--device", help="ASCEND_RT_VISIBLE_DEVICES value")
    parser.add_argument("--manifest", type=Path, default=HERE.parents[4] / "manifest.json")
    parser.add_argument("--validation-scope", choices=("auto", "all"), default="auto")
    parser.add_argument("--output-root", type=Path, help="new or empty output directory")
    parser.add_argument(
        "--profile", type=Path,
        help="explicit baseline; otherwise use the profile shipped by the installed TA",
    )
    parser.add_argument("--expected-bishengir", type=Path, help="defaults to bishengir-compile on PATH")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measurement-processes", type=int, default=3)
    parser.add_argument("--timing-source", choices=("auto", "profiler", "event"), default="auto")
    parser.add_argument("--triton-root", type=Path, default=HERE.parents[4])
    parser.add_argument("--npuir-root", type=Path,
                        default=HERE.parents[4] / "third_party/ascend/AscendNPU-IR")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        if "driver" in case:
            case["driver"] = str((args.manifest.resolve().parent / case["driver"]).resolve())
    selected = [case for case in manifest["cases"] if case["name"] == args.case]
    if len(selected) != 1:
        parser.error(
            f"expected exactly one manifest case named {args.case!r}; "
            "use run_generalization_matrix.py --list-cases"
        )
    if args.warmup < 0 or args.repeats <= 0 or args.measurement_processes <= 0:
        parser.error("warmup must be nonnegative; repeats and measurement-processes must be positive")

    env = os.environ.copy()
    benchmark_dir = HERE.parents[6] / "costmodel_tests/benchmarks"
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(benchmark_dir), str(HERE), env.get("PYTHONPATH", "")) if path
    )
    if env.get("TRITON_ASCEND_COMPILE_MODE"):
        parser.error("unset TRITON_ASCEND_COMPILE_MODE so it cannot override the tested routes")
    if args.device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    env.setdefault("TRITON_DEBUG", "0")
    env.setdefault("TRITON_DISABLE_LINE_INFO", "1")
    compiler = args.expected_bishengir or shutil.which("bishengir-compile")
    if compiler is None:
        parser.error("bishengir-compile is not on PATH; load your toolchain environment first")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_root or HERE / "single_case_results" / f"{args.case}_{timestamp}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output directory must be new or empty: {output}")
    try:
        profile = prepare_profile(args.profile, output / "costmodel_profile.json", env)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    # Keep the original manifest untouched. Both runner and audit use this subset.
    manifest["cases"] = selected
    subset = output / "single_case_manifest.json"
    subset.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    env["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(profile)
    expected_count = 1 if args.validation_scope == "auto" else 5 + len(selected[0].get("extra_modes", []))
    print(f"Case: {args.case}; expected route records: {expected_count}", flush=True)
    print(f"Output: {output}", flush=True)
    print("[1/2] Running correctness checks and profiling; see runner.log", flush=True)
    runner_status = run_logged([
        sys.executable, str(HERE / "run_generalization_matrix.py"),
        "--manifest", str(subset),
        *(["--include-explicit"] if args.validation_scope == "all" else []),
        "--profile", str(profile),
        "--expected-bishengir", str(Path(compiler).resolve()),
        "--output-root", str(output),
        "--warmup", str(args.warmup), "--repeats", str(args.repeats),
        "--measurement-processes", str(args.measurement_processes),
        "--timing-source", args.timing_source,
        "--triton-root", str(args.triton_root.resolve()),
        "--npuir-root", str(args.npuir_root.resolve()),
    ], env, output / "runner.log")

    audit_json = output / "generalization_audit.json"
    audit_markdown = output / "generalization_audit.md"
    audit = None
    audit_status = None
    prerequisites = [output / name for name in (
        "matrix_summary.json", "generalization_matrix.json", "costmodel_profile.json"
    )]
    if all(path.is_file() for path in prerequisites):
        # Audit partial results too, so failed or missing routes are explained.
        print("[2/2] Auditing prediction errors (maximum absolute error: 10%)", flush=True)
        audit_status = run_logged([
            sys.executable, str(HERE / "audit_generalization_matrix.py"),
            "--evidence-root", str(output),
            "--manifest", str(output / "generalization_matrix.json"),
            "--profile", str(output / "costmodel_profile.json"),
            "--validation-scope", args.validation_scope,
            "--output-json", str(audit_json),
            "--output-markdown", str(audit_markdown),
        ], env, output / "audit.log")
        if audit_json.is_file():
            audit = json.loads(audit_json.read_text(encoding="utf-8"))
    else:
        print("[2/2] Audit unavailable: runner did not produce the required files", flush=True)

    passed = (
        runner_status == 0 and audit_status == 0 and audit is not None
        and audit.get("all_pass") is True
        and audit.get("expected_record_count") == expected_count
        and audit.get("audited_record_count") == expected_count
    )
    result = {
        "case": args.case, "status": "PASS" if passed else "FAIL",
        "validation_scope": args.validation_scope,
        "profile_metadata": str(output / "profile_metadata.json"),
        "runner_exit_status": runner_status, "audit_exit_status": audit_status,
        "audit_all_pass": audit.get("all_pass") if audit else None,
        "expected_record_count": expected_count,
        "audited_record_count": audit.get("audited_record_count") if audit else None,
        "output_root": str(output),
    }
    (output / "single_case_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"{args.case}: {result['status']}", flush=True)
    if audit_markdown.is_file():
        print(audit_markdown.read_text(encoding="utf-8"), flush=True)
    else:
        print(f"Inspect logs in {output}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
