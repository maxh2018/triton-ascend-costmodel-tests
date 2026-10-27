#!/usr/bin/env python3
"""Build an offline HTML dashboard from nightly runs; no NPU imports required."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "costmodel_test_results/nightly"
LABELS = {
    "PASS": "通过",
    "TIME_FAIL": "时间误差超标",
    "PRECISION_FAIL": "精度失败",
    "EXECUTION_FAIL": "执行异常",
    "BLOCKED": "环境阻塞",
    "NOT_RUN": "未执行",
    "UNVERIFIED": "未完成时间验收",
    "RUNNING": "运行中",
    "FAIL": "未通过",
    "INCOMPLETE": "结果不完整",
}


def read_json(path, default=None):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(
            data, dict) else ({} if default is None else default)
    except (OSError, ValueError, TypeError):
        return {} if default is None else default


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def artifact(run, value, metadata):
    """Resolve copied Linux evidence on any host, and refuse outside paths."""
    if not value:
        return None
    text = str(value).replace("\\", "/")
    original = str(metadata.get("raw_root", "")).replace("\\", "/").rstrip("/")
    if original and text.startswith(original + "/"):
        candidate = run / "raw" / text[len(original) + 1:]
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = run / candidate
    candidate = candidate.resolve()
    return candidate if candidate.is_relative_to(run.resolve()) else None


def link(run, path):
    if path is None or not path.is_file():
        return None
    return quote(path.relative_to(run).as_posix(), safe="/")


def excerpt(paths):
    # Only known log locations; never traverse profiler dumps or external files.
    pieces = []
    for path in paths:
        if path and path.is_file():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                pieces.append(stream.read().decode("utf-8", errors="replace"))
    return "\n".join(pieces)[-16000:]


def case_report(run, case, result, audits, matrix, metadata):
    name = case["name"]
    row = next(
        (r
         for r in result.get("audit_results", audits) if r.get("case") == name
         and r.get("mode") == "auto" and r.get("factor") == 1), {})
    record = next((r for r in matrix if r.get("name") == name + "__auto_f1"),
                  {})
    directory = artifact(run, record.get("evidence_dir"), metadata)
    evidence = read_json(directory / "evidence.json") if directory else {}
    benchmark = evidence.get("benchmark") or {}
    precision = benchmark.get("correct")
    if precision is None and result.get("precision_pass") is True:
        precision = True
    attempts = result.get("attempts") or []
    logs = [artifact(run, attempts[-1].get("log"), metadata)
            ] if attempts else []
    logs += ([
        directory / "stderr.log", directory / "compile_stderr.log", directory /
        "stdout.log"
    ] if directory else [])
    text = excerpt(logs)
    predicted, actual = number(row.get("predicted_us")), number(
        row.get("measured_us"))
    ape = abs(
        predicted - actual
    ) / actual * 100 if predicted is not None and actual and actual > 0 else None
    source = row.get("measurement_source")
    profiler = source == "npu_profiler_kernel_duration_average_us" and row.get(
        "timing_gate_applied") is True
    reasons = row.get("failure_reasons") or []
    detail = "; ".join(str(v) for v in reasons)
    device_error = any(s in text
                       for s in ("TsdOpen failed",
                                 "Failed to start the device", "507033"))
    accuracy_error = any(
        s in text
        for s in ("Tensor-likes are not close", "Mismatched elements:",
                  "Not equal to tolerance", "Greatest absolute difference:"))
    if not result and not row:
        status = "NOT_RUN"
    elif device_error:
        status, detail, precision = "BLOCKED", "NPU 初始化失败，尚未完成用例验收", None
    elif accuracy_error or benchmark.get("correct") is False:
        status, precision = "PRECISION_FAIL", False
    elif precision is not True:
        # Missing evidence / compile failure is not evidence of bad numerics.
        status = "EXECUTION_FAIL"
    elif not profiler or predicted is None or actual is None or actual <= 0:
        status = "UNVERIFIED"
        detail = detail or "需要 Profiler 实测、有效预测和已启用的时间验收"
    elif ape > 10 + 1e-9:
        status = "TIME_FAIL"
    elif row.get("pass") is True and result.get(
            "status") == "PASS" and not reasons:
        status = "PASS"
    else:
        status = "EXECUTION_FAIL"
    if status not in ("PASS", "TIME_FAIL") and not detail:
        detail = text[-2500:] or result.get("error") or "结果或验收记录不完整"
    last_log = logs[0] if attempts else None
    return {
        "name": name,
        "family": case.get("family", "custom"),
        "status": status,
        "precision": precision,
        "predicted_us": predicted,
        "actual_us": actual,
        "ape_percent": ape,
        "measurement_source": source,
        "attempts": result.get("attempt_count", len(attempts)),
        "detail": detail,
        "log": link(run, last_log),
        "evidence":
        link(run, directory / "evidence.json") if directory else None,
    }


def build_report(run):
    run = Path(run).resolve()
    metadata = read_json(run / "run.json")
    manifest = read_json(run / "nightly_manifest.json")
    batch = read_json(run / "raw/batch_summary.json")
    if not batch:
        batch = read_json(run / "raw/batch_summary.partial.json")
    audit = read_json(run / "raw/generalization_audit.json")
    matrix = read_json(run / "raw/matrix_summary.json").get("records", [])
    results = {r["case"]: r for r in batch.get("cases", [])}
    rows = [
        case_report(run, c, results.get(c["name"], {}),
                    audit.get("results", []), matrix, metadata)
        for c in manifest.get("cases", [])
    ]
    counts = dict(Counter(r["status"] for r in rows))
    status = metadata.get("state", "INCOMPLETE")
    if status not in ("RUNNING", "BLOCKED", "INCOMPLETE"):
        if rows and all(r["status"] == "PASS" for r in rows) and metadata.get(
                "returncode") == 0 and batch.get("status") == "PASS":
            status = "PASS"
        elif any(r["status"] in ("TIME_FAIL", "PRECISION_FAIL",
                                 "EXECUTION_FAIL", "UNVERIFIED")
                 for r in rows):
            status = "FAIL"
        elif metadata.get("returncode") not in (None, 0) and not batch:
            status = "BLOCKED"
        elif counts.get("BLOCKED"):
            status = "BLOCKED"
        else:
            status = "INCOMPLETE"
    profile = read_json(run / "raw/profile_metadata.json")
    runtime = (profile.get("runtimes") or [{}])[0]
    message = metadata.get("error") or ""
    if status == "BLOCKED" and not message:
        message = excerpt([run / "nightly.log"])[-3000:] or "环境预检或设备初始化失败"
    return {
        "id":
        run.name,
        "started_at":
        metadata.get("started_at", ""),
        "finished_at":
        metadata.get("finished_at"),
        "status":
        status,
        "base":
        "",
        "device":
        metadata.get("device"),
        "total":
        len(rows),
        "counts":
        counts,
        "rows":
        rows,
        "message":
        message,
        "label":
        metadata.get("label", ""),
        "git":
        metadata.get("git", {}),
        "health":
        metadata.get("health", {}),
        "manifest_sha256":
        metadata.get("manifest_sha256"),
        "ta_version":
        runtime.get("ta_distribution_version") or runtime.get("ta_version")
        or runtime.get("triton_version"),
        "profile_version":
        profile.get("profile_version"),
        "profile_sha256":
        profile.get("source_sha256"),
        "links": {
            "log": link(run, run / "nightly.log"),
            "metadata": link(run, run / "run.json"),
            "profile": link(run, run / "raw/profile_metadata.json")
        },
    }


def render(reports):
    template = Path(__file__).with_name("dashboard.html").read_text(
        encoding="utf-8")
    # A manifest name or log can contain </script>; keep embedded JSON inert.
    payload = json.dumps({
        "reports": reports,
        "labels": LABELS
    },
                         ensure_ascii=True,
                         allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace(
        "&", "\\u0026")
    return template.replace("__DASHBOARD_DATA__", payload)


def generate(output):
    output = Path(output).resolve()
    reports = []
    for run in sorted((output / "runs").glob("*"), reverse=True):
        if not run.is_dir() or not (run / "run.json").is_file():
            continue
        report = build_report(run)
        write_json(run / "dashboard.json", report)
        write_text(run / "index.html", render([report]))
        portal = deepcopy(report)
        portal["base"] = "runs/" + quote(run.name, safe="") + "/"
        reports.append(portal)
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / "index.html", render(reports))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reports = generate(args.output_root)
    print(
        f"Dashboard: {args.output_root.resolve() / 'index.html'} ({len(reports)} runs)"
    )


if __name__ == "__main__":
    main()
