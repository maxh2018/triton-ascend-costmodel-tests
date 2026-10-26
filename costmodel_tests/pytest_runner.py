"""Run one manifest pytest entry, retrying only its failed parameterized nodes."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_pytest_case(case, args, env, device):
    directory = args.output_root / "cases" / case["name"]
    directory.mkdir(parents=True, exist_ok=True)
    targets = [case["driver"]]
    merged, attempts = {}, []
    child_env = env.copy()
    if device is not None:
        child_env["ASCEND_RT_VISIBLE_DEVICES"] = device
    for number in range(1, args.retries + 2):
        output = directory / f"attempt_{number:02d}"
        log = directory / f"attempt_{number:02d}.log"
        command = [sys.executable, "-m", "pytest", *targets,
                   "-p", "costmodel_tests.pytest_costmodel",
                   "--costmodel-output", str(output / "pytest_costmodel"),
                   f"--junitxml={output / 'pytest.xml'}", "-v"]
        if (args.output_root / "costmodel_profile.json").is_file():
            command += ["--costmodel-profile", str(args.output_root / "costmodel_profile.json")]
        if args.expected_bishengir is not None:
            command += ["--costmodel-expected-bishengir", str(args.expected_bishengir)]
        command += case.get("arguments", [])
        print(f"START {case['name']} attempt={number}/{args.retries + 1} device={device or 'inherited'}", flush=True)
        started = time.monotonic()
        failure = None
        with log.open("w", encoding="utf-8") as stream:
            try:
                status = subprocess.run(command, cwd=ROOT, env=child_env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False).returncode
            except OSError as error:
                status, failure = 1, str(error)
                stream.write(failure + "\n")
        try:
            report = json.loads((output / "pytest_costmodel/summary.json").read_text(encoding="utf-8"))
            rows = report["results"]
            if not isinstance(rows, list):
                raise ValueError("invalid pytest results")
            for row in rows:
                merged[row["nodeid"]] = row
            passed = (status == 0 and report["status"] == "PASS" and bool(merged)
                      and any(r["status"] == "PASS" for r in merged.values())
                      and all(r["status"] in ("PASS", "SKIP") for r in merged.values()))
            failure = failure or (None if passed else "pytest_or_timing_audit_failed")
        except (OSError, ValueError, KeyError, TypeError) as error:
            passed, failure = False, f"missing_or_invalid_result: {error}"
        attempts.append({"attempt": number, "returncode": status,
                         "status": "PASS" if passed else "FAIL", "failure_reason": failure,
                         "elapsed_seconds": time.monotonic() - started,
                         "output_root": str(output), "log": str(log), "targets": targets})
        print(f"{'PASS' if passed else 'FAIL'} {case['name']} attempt={number}", flush=True)
        if passed:
            break
        failed = [r for r in merged.values() if r["status"] not in ("PASS", "SKIP")]
        if failed:
            # nodeids are relative to pytest's rootdir. Use the original absolute
            # file plus its ::selector so retries do not depend on that rootdir.
            selectors = [r["nodeid"].partition("::") for r in failed]
            if any(not separator or not selector for _, separator, selector in selectors):
                break
            targets = [case["driver"] + "::" + selector for _, _, selector in selectors]
        elif merged:
            # Infrastructure error without an identifiable failed node: do not
            # replay successful tests just to retry the whole file.
            break
        if number <= args.retries:
            print(f"RETRY {case['name']} ({number}/{args.retries})", flush=True)
    results = list(merged.values())
    active = [r for r in results if r["status"] != "SKIP"]
    functional_only = "--costmodel-functional-only" in case.get("arguments", [])
    final = {"case": case["name"], "runner": "pytest", "status": attempts[-1]["status"],
             "validation_scope": "functional" if functional_only else "auto",
             "device": device, "attempt_count": len(attempts), "attempts": attempts,
             "final_output_root": attempts[-1]["output_root"],
             "precision_pass": bool(active) and all(r.get("precision_pass") is True for r in active),
             "timing_pass": None if functional_only else
                            bool(active) and all(r.get("timing_pass") is True for r in active),
             "pytest_results": results}
    (directory / "pytest_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final
