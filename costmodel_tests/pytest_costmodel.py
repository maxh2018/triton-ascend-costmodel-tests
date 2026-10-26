"""Pytest adapter: original assertions + Auto-route prediction error <= 10%.

Loaded by costmodel_tests/run.py. Tests without a measured runtime kernel are
UNSUPPORTED (a failing audit), never silently accepted as timing PASS.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import pytest

from .pytest_npu import LaunchCapture, NpuBackend
from .profiles import prepare_profile, profile_report_lines


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pytest_addoption(parser):
    group = parser.getgroup("costmodel")
    group.addoption("--costmodel-output", type=Path, help="enable strict Auto-route timing audit; write evidence here")
    group.addoption("--costmodel-profile", type=Path, help="profile used for compilation and timing audit")
    group.addoption("--costmodel-expected-bishengir", type=Path, help="expected compiler executable on PATH")
    group.addoption("--costmodel-functional-only", action="store_true",
                    help="run original pytest assertions without launch instrumentation or timing gates")


def pytest_configure(config):
    output = config.getoption("costmodel_output")
    if output is None:
        return
    if getattr(config.option, "numprocesses", None):
        raise pytest.UsageError("CostModel pytest profiling is serial; do not use pytest-xdist -n")
    if os.environ.get("TRITON_ASCEND_COMPILE_MODE"):
        raise pytest.UsageError("unset TRITON_ASCEND_COMPILE_MODE before CostModel pytest")
    expected = config.getoption("costmodel_expected_bishengir")
    if expected is not None:
        compiler = shutil.which("bishengir-compile")
        if compiler is None or Path(compiler).resolve() != expected.resolve():
            raise pytest.UsageError("bishengir-compile on PATH does not match --expected-bishengir")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise pytest.UsageError(f"CostModel output must be new or empty: {output}")
    functional_only = config.getoption("costmodel_functional_only")
    mode = "functional" if functional_only else "collect_only" if config.option.collectonly else "costmodel"
    env = os.environ.copy()
    try:
        snapshot = prepare_profile(config.getoption("costmodel_profile"), output / "costmodel_profile.json", env, mode=mode)
    except (OSError, ValueError) as error:
        raise pytest.UsageError(str(error)) from error
    plugin = CostModelPytest(output, snapshot, functional_only=functional_only)
    config.pluginmanager.register(plugin, "costmodel-audit")
    config._costmodel_adapter = plugin


class CostModelPytest:
    def __init__(self, output, profile, backend_factory=NpuBackend, functional_only=False):
        self.output, self.profile = output, profile
        self.functional_only = functional_only
        self.backend_factory = backend_factory
        self.backend = self.original = self.active = None
        self.items = {}
        self.current = None
        self.environment = {}

    def install(self):
        if self.backend is not None:
            return
        self.backend = self.backend_factory()
        self.original = self.backend.jit_class.run
        for key, value in {"TRITON_ASCEND_AUTO_SIMT_PROFILE": str(self.profile),
                           "TRITON_CACHE_DIR": str(self.output / "cache")}.items():
            self.environment[key] = os.environ.get(key)
            os.environ[key] = value
        original = self.original

        @wraps(original)
        def run(kernel, *args, grid, warmup, **kwargs):
            if self.active is None:
                return original(kernel, *args, grid=grid, warmup=warmup, **kwargs)
            return self.active.run(original, kernel, args, grid, warmup, kwargs)

        self.backend.jit_class.run = run

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.nodeid)[-80:]
        suffix = hashlib.sha256(item.nodeid.encode()).hexdigest()[:12]
        directory = self.output / "cases" / f"{name}_{suffix}"
        directory.mkdir(parents=True, exist_ok=True)
        self.current = {"nodeid": item.nodeid, "status": "PENDING", "precision_pass": None,
                        "timing_pass": None if self.functional_only else False,
                        "failure_reasons": [], "evidence_dir": str(directory)}
        self.items[item.nodeid] = self.current
        if not self.functional_only:
            self.install()
            self.active = LaunchCapture(directory, self.profile, self.backend)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        record = self.items.get(item.nodeid)
        if record is None:
            # A different setup hook may fail/skip before ours runs.
            record = {"nodeid": item.nodeid, "status": "PENDING", "precision_pass": None,
                      "timing_pass": None if self.functional_only else False,
                      "failure_reasons": [], "evidence_dir": None}
            self.items[item.nodeid] = record
        capture = self.active if self.current is record else None
        launches = capture.public_records() if capture else []
        record["launches"] = launches
        if report.skipped:
            record.update(status="SKIP", skip_reason=str(report.longrepr))
        elif report.when == "call":
            record["precision_pass"] = report.passed
            record["timing_pass"] = (None if self.functional_only else
                                     bool(launches) and all(r["pass"] for r in launches))
            if report.passed:
                if self.functional_only:
                    record["status"] = "PASS"
                elif not launches:
                    record.update(status="UNSUPPORTED", failure_reasons=["no_runtime_jit_kernel_captured"])
                elif not record["timing_pass"]:
                    record.update(status="FAIL", failure_reasons=["kernel_timing_audit_failed"])
                else:
                    record["status"] = "PASS"
                if record["status"] != "PASS":
                    report.outcome = "failed"
                    report.longrepr = (f"CostModel {record['status']}: {', '.join(record['failure_reasons'])}\n"
                                       f"Evidence: {record['evidence_dir']}")
            else:
                record.update(status="FAIL", failure_reasons=["original_test_failed"])
        elif (report.when == "teardown" and report.passed and record["status"] == "PASS"
              and not self.functional_only):
            record["timing_pass"] = bool(launches) and all(r["pass"] for r in launches)
            if not record["timing_pass"]:
                record.update(status="FAIL", failure_reasons=["teardown_kernel_timing_audit_failed"])
                report.outcome = "failed"
                report.longrepr = f"CostModel FAIL: teardown kernel timing audit; evidence: {record['evidence_dir']}"
        elif report.failed:
            record["status"] = "FAIL"
            record["failure_reasons"].append(f"{report.when}_failed")
        if record["evidence_dir"]:
            write_json(Path(record["evidence_dir"]) / "result.json", record)
        if report.when == "teardown":
            self.active = self.current = None

    def pytest_collection_finish(self, session):
        write_json(self.output / "collected_cases.json", {"cases": [
            {"nodeid": item.nodeid, "source": str(item.path), "line": item.location[1] + 1}
            for item in session.items]})

    def pytest_sessionfinish(self, session, exitstatus):
        results = list(self.items.values())
        counts = {status: sum(r["status"] == status for r in results)
                  for status in ("PASS", "FAIL", "UNSUPPORTED", "SKIP", "PENDING")}
        if session.config.option.collectonly:
            status = "NOT_RUN"
        else:
            passed = exitstatus == 0 and counts["PASS"] > 0 and not any(
                counts[s] for s in ("FAIL", "UNSUPPORTED", "PENDING"))
            status = "PASS" if passed else "FAIL"
            if not passed and session.exitstatus == 0:
                session.exitstatus = 1
        summary = {"status": status, "validation_scope": "functional" if self.functional_only else "auto",
                   "profile_metadata": str(self.output / "profile_metadata.json"),
                   "maximum_absolute_relative_error": None if self.functional_only else 0.10,
                   "collected_case_count": len(session.items), "counts": counts, "results": results}
        write_json(self.output / "summary.json", summary)
        lines = ["# Pytest CostModel audit", "", f"Status: {status}", "",
                 "| Test | Precision | Timing | Status |", "|---|---|---|---|"]
        lines[4:4] = profile_report_lines(self.output)
        for r in results:
            node = r["nodeid"].replace("|", "\\|").replace("\n", " ")
            timing = "Not checked" if self.functional_only else r['timing_pass']
            lines.append(f"| {node} | {r['precision_pass']} | {timing} | {r['status']} |")
        lines += ["", "## Kernel timing", "",
                  "| Test | Kernel | Predicted (us) | Measured (us) | Error | Gate |",
                  "|---|---|---:|---:|---:|---|"]
        for r in results:
            node = r["nodeid"].replace("|", "\\|").replace("\n", " ")
            for launch in r.get("launches", []):
                predicted = launch["predicted_us"]
                measured = launch["measured_us"]
                error = launch["relative_error"]
                values = [f"{v:.3f}" if v is not None else "N/A" for v in (predicted, measured)]
                error_text = f"{error:+.2%}" if error is not None else "N/A"
                gate = "PASS" if launch["pass"] else "FAIL: " + ", ".join(launch["failure_reasons"])
                gate = gate.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {node} | {launch['kernel']} | {values[0]} | {values[1]} | {error_text} | {gate} |")
        (self.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def pytest_terminal_summary(self, terminalreporter):
        if self.functional_only:
            terminalreporter.write_sep("=", "Pytest functionality: original assertions; timing not checked")
            terminalreporter.write_line(f"Report: {self.output / 'report.md'}")
            return
        terminalreporter.write_sep("=", "CostModel pytest: precision + Auto timing error <= 10%")
        terminalreporter.write_line(f"Report: {self.output / 'report.md'}")
        terminalreporter.write_line("UNSUPPORTED means no runtime kernel was measured; it is not a timing PASS.")

    def pytest_unconfigure(self, config):
        if self.original is not None:
            self.backend.jit_class.run = self.original
        for key, value in self.environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
