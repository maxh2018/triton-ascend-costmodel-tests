#!/usr/bin/env python3
"""Run the CI 100-case nightly acceptance and always generate the HTML dashboard."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from costmodel_tests.dashboard import DEFAULT_OUTPUT, generate, write_json
from costmodel_tests.manifests import load_manifest

TZ = timezone(timedelta(hours=8))


def timestamp():
    return datetime.now(TZ).isoformat(timespec="seconds")


def git_state():

    def query(*args):
        try:
            p = subprocess.run(["git", *args],
                               cwd=ROOT,
                               capture_output=True,
                               text=True,
                               timeout=10)
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    return {
        "sha": query("rev-parse", "HEAD"),
        "dirty": bool(query("status", "--porcelain"))
    }


def check_device(device):
    executable = shutil.which("npu-smi")
    if not executable:
        return {
            "status": "Unavailable",
            "ok": False,
            "output": "npu-smi 不在 PATH，请先加载 NPU 环境"
        }
    try:
        p = subprocess.run([executable, "info", "-t", "health", "-i", device],
                           capture_output=True,
                           text=True,
                           timeout=15)
        output = p.stdout + p.stderr
        match = re.search(r"Health Status\s*:\s*([^\r\n]+)", output)
        status = match.group(1).strip() if match else "Unknown"
        return {
            "status":
            status,
            "ok":
            p.returncode == 0
            and status.lower() not in ("critical", "unknown"),
            "output":
            output
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "Unavailable", "ok": False, "output": str(error)}


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


@contextmanager
def output_lock(output):
    """Avoid overlapping nightly runs sharing a history directory; crash-safe OS lock."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".nightly.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            acquire = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.
                                          LOCK_NB)
            release = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        else:
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        try:
            acquire()
        except OSError as error:
            raise ValueError("该输出目录已有夜间测试在运行，请等待结束") from error
        try:
            yield
        finally:
            release()


def execute(args, runner=None, device_check=None):
    """The injectable runner/device probe support real CPU-only integration tests."""
    with output_lock(args.output_root.resolve()):
        return _execute(args, runner, device_check)


def _execute(args, runner, device_check):
    manifest = load_manifest(args.manifest)
    if not manifest["cases"]:
        raise ValueError("manifest is empty")
    if any(c["runner"] != "benchmark" for c in manifest["cases"]):
        raise ValueError(
            "nightly acceptance currently requires benchmark entries (the CI 100-case manifest)"
        )
    output = args.output_root.resolve()
    run = output / "runs" / datetime.now(TZ).strftime("%Y-%m-%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "nightly_manifest.json"
    write_json(snapshot, manifest)
    command = [
        sys.executable,
        str(runner or ROOT / "costmodel_tests/run.py"), "--manifest",
        str(snapshot), "--device", args.device, "--output-root",
        str(run / "raw"), "--validation-scope", "auto", "--timing-source",
        "profiler", "--measurement-processes", "3", "--warmup", "3",
        "--repeats", "7", "--jobs", "1", "--compile-workers", "1",
        "--case-batch-size", "1", "--require-idle-devices", "--retries",
        str(args.retries)
    ]
    for option, value in (("--profile", args.profile),
                          ("--expected-bishengir", args.expected_bishengir),
                          ("--triton-root", args.triton_root),
                          ("--npuir-root", args.npuir_root)):
        if value is not None:
            command += [option, str(value.resolve())]
    metadata = {
        "schema": "costmodel-nightly-v1",
        "started_at": timestamp(),
        "state": "RUNNING",
        "finished_at": None,
        "device": args.device,
        "label": args.label,
        "raw_root": str(run / "raw"),
        "manifest_source": str(args.manifest.resolve()),
        "manifest_sha256":
        hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "git": git_state(),
        "command": command,
        "returncode": None,
        "error": None,
        "policy": {
            "timing_source": "profiler",
            "validation_scope": "auto",
            "measurement_processes": 3,
            "maximum_ape_percent": 10,
            "jobs": 1
        }
    }
    write_json(run / "run.json", metadata)
    generate(output)
    print(f"Run: {run}", flush=True)
    print(f"Dashboard: {output / 'index.html'}", flush=True)
    process = None
    try:
        health = (device_check or check_device)(args.device)
        metadata["health"] = health
        write_json(run / "device_health.json", health)
        if not health["ok"]:
            metadata.update(state="BLOCKED",
                            returncode=2,
                            error="NPU 预检未通过：" + health["output"])
        else:
            print(
                "Running CI acceptance: Auto / Profiler / 3 measurements / APE <= 10%",
                flush=True)
            with (run / "nightly.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix")
                deadline = time.monotonic() + args.timeout_hours * 3600
                while True:
                    try:
                        metadata["returncode"] = process.wait(timeout=min(
                            20, max(.01, deadline - time.monotonic())))
                        metadata["state"] = "COMPLETE"
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            stop_process(process)
                            metadata.update(
                                state="INCOMPLETE",
                                returncode=124,
                                error=
                                f"超过总时限 {args.timeout_hours:g} 小时，已停止本次测试进程")
                            break
                        write_json(run / "run.json", metadata)
                        generate(output)
    except KeyboardInterrupt:
        if process:
            stop_process(process)
        metadata.update(state="INCOMPLETE",
                        returncode=130,
                        error="测试被中断，未完成全部验收")
    except (OSError, ValueError) as error:
        if process:
            stop_process(process)
        metadata.update(state="BLOCKED", returncode=2, error=str(error))
    finally:
        metadata["finished_at"] = timestamp()
        write_json(run / "run.json", metadata)
        reports = generate(output)
    report = next(r for r in reports if r["id"] == run.name)
    passed = report["status"] == "PASS"
    print(
        f"Nightly: {report['status']} ({report['counts'].get('PASS', 0)}/{report['total']} accepted)",
        flush=True)
    print(f"Report: {run / 'index.html'}", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",
                        required=True,
                        help="single physical NPU ID")
    parser.add_argument("--output-root",
                        type=Path,
                        default=DEFAULT_OUTPUT,
                        help="persistent dashboard/history directory")
    parser.add_argument("--manifest",
                        type=Path,
                        default=ROOT / "cases/ci/manifest.json")
    parser.add_argument("--retries",
                        type=int,
                        default=3,
                        help="extra attempts for failed cases only")
    parser.add_argument("--timeout-hours",
                        type=float,
                        default=8,
                        help="maximum duration of the complete nightly run")
    parser.add_argument("--label",
                        default="",
                        help="optional build or experiment label")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-bishengir", type=Path)
    parser.add_argument("--triton-root", type=Path)
    parser.add_argument("--npuir-root", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+", args.device):
        parser.error("--device must be one numeric NPU ID")
    if args.retries < 0 or not (0 < args.timeout_hours <= 168):
        parser.error(
            "--retries must be nonnegative; --timeout-hours must be in (0, 168]"
        )
    try:
        status = execute(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
