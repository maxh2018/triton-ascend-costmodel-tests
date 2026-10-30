"""Host-only mode summaries and calendar-day APE trends."""
from datetime import datetime, timedelta, timezone
from statistics import mean, median
import math

MODES = ("auto", "simd", "simt_only", "mixed")
MODE_LABELS = {"auto": "Auto", "simd": "SIMD", "simt_only": "SIMT-only", "mixed": "Mix"}
TZ = timezone(timedelta(hours=8))
PROFILER = "npu_profiler_kernel_duration_average_us"
ROUTE_LABELS = {"all_simd": "SIMD", "all_simt_only": "SIMT-only", "mixed_simd_simt": "Mix"}


def actual_route(report):
    """Only the effective compiler decision identifies the route actually used."""
    decision = report.get("effective_decision_kind")
    factor = report.get("effective_runtime_factor")
    return {"actual_route": ROUTE_LABELS.get(decision, decision),
            "actual_route_kind": decision, "actual_factor": factor}


def valid_ape(row):
    return (row.get("precision") is True
            and row.get("measurement_source") == PROFILER
            and row.get("timing_gate_applied") is True
            and isinstance(row.get("ape_percent"), (float, int))
            and math.isfinite(row["ape_percent"]))


def primary_result(row, mode):
    results = [r for r in row.get("mode_results", []) if r["mode"] == mode]
    return min(results, key=lambda r: (r.get("status") in ("NOT_RUN", "NOT_APPLICABLE"),
                                      r["factor"] != 1, r["factor"])) if results else {}


def summarize_modes(rows):
    summaries = {}
    for mode in MODES:
        results = [primary_result(r, mode) for r in rows]
        values = [r["ape_percent"] for r in results if valid_ape(r)]
        summaries[mode] = {
            "valid_count": len(values), "total": len(rows),
            "mape_percent": mean(values) if values else None,
            "median_ape_percent": median(values) if values else None,
            "max_ape_percent": max(values) if values else None,
            "over_limit_count": sum(v > 10 + 1e-9 for v in values),
            "counts": {status: sum(r.get("status", "NOT_RUN") == status for r in results)
                       for status in ("PASS", "TIME_FAIL", "PRECISION_FAIL", "EXECUTION_FAIL",
                                      "BLOCKED", "UNVERIFIED", "NOT_RUN", "NOT_APPLICABLE")},
        }
    return summaries


def run_timestamp(report):
    try:
        stamp = datetime.fromisoformat(report.get("started_at", "").replace("Z", "+00:00"))
        return stamp.replace(tzinfo=TZ) if stamp.tzinfo is None else stamp.astimezone(TZ)
    except (ValueError, TypeError):
        return None


def run_day(report):
    stamp = run_timestamp(report)
    return stamp.date() if stamp else None


def ten_day_trend(reports, anchor):
    """Latest finished batch per Beijing day, for the same case set and jobs."""
    end = run_day(anchor) or datetime.now(TZ).date()
    start = end - timedelta(days=9)
    case_set = {r["name"] for r in anchor.get("rows", [])}
    daily = {}
    for report in sorted(reports, key=lambda r: run_timestamp(r) or datetime.min.replace(tzinfo=TZ)):
        day = run_day(report)
        if (not day or not start <= day <= end
                or report.get("status") not in ("PASS", "FAIL")
                or not report.get("finished_at")
                or report.get("jobs", 1) != anchor.get("jobs", 1)
                or {r["name"] for r in report.get("rows", [])} != case_set):
            continue
        daily[day] = report
    points = []
    for offset in range(10):
        day = start + timedelta(days=offset)
        report = daily.get(day)
        point = {"date": day.isoformat(), "run_id": None, "modes": {}, "cases": {}}
        if report:
            point.update(run_id=report["id"], status=report["status"],
                         modes=report.get("mode_summary") or summarize_modes(report["rows"]))
            point["cases"] = {r["name"]: {mode: primary_result(r, mode)["ape_percent"]
                                            for mode in MODES if valid_ape(primary_result(r, mode))}
                              for r in report["rows"]}
        points.append(point)
    return points
