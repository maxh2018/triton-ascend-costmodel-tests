"""Host-only tests for parallel compile and selectable timing sources."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
GENERAL = (
    ROOT /
    "cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization"
)
sys.path.insert(0, str(GENERAL))

from audit_generalization_matrix import measured_us, timing_gate_applies, timing_gate_passes  # noqa: E402
from run_generalization_batch import (  # noqa: E402
    partition_cases,
    parse_busy_devices,
    precision_from_records,
    select_batch_timing_source,
    select_case_batch_size,
)
from run_generalization_matrix import select_timing_source  # noqa: E402

COLLECTOR = GENERAL / "collect_backend_evidence.py"


class CollectorTwoPhaseTest(unittest.TestCase):

    def test_auto_timing_policy_uses_global_workload_size(self) -> None:
        self.assertEqual(select_timing_source("auto", 64, 64), "profiler")
        self.assertEqual(select_timing_source("auto", 65, 64), "event")
        self.assertEqual(select_timing_source("event", 1, 64), "event")
        self.assertEqual(select_batch_timing_source("auto", 100, 64), "event")
        self.assertEqual(select_batch_timing_source("auto", 1, 64), "profiler")

    def test_benchmark_cases_are_batched_but_pytest_cases_stay_isolated(self) -> None:
        cases = [
            {"name": "a", "runner": "benchmark"},
            {"name": "b", "runner": "benchmark"},
            {"name": "p", "runner": "pytest"},
            {"name": "c", "runner": "benchmark"},
            {"name": "d", "runner": "benchmark"},
            {"name": "e", "runner": "benchmark"},
        ]
        self.assertEqual(
            [[case["name"] for case in group] for group in partition_cases(cases, 2)],
            [["a", "b"], ["p"], ["c", "d"], ["e"]],
        )

    def test_large_event_run_defaults_to_ten_batches_of_ten(self) -> None:
        batch_size = select_case_batch_size(0, "event", 0, 100, 5)
        groups = partition_cases(
            [{"name": f"case_{index}", "runner": "benchmark"} for index in range(100)],
            batch_size,
        )
        self.assertEqual(batch_size, 10)
        self.assertEqual(len(groups), 10)
        self.assertEqual([len(group) for group in groups], [10] * 10)
        self.assertEqual(select_case_batch_size(0, "profiler", 0, 100, 5), 1)
        self.assertEqual(select_case_batch_size(0, "event", 1, 100, 5), 1)
        self.assertEqual(select_case_batch_size(0, "event", 0, 100, 3), 17)

    def test_precision_failure_survives_batch_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence_dir = Path(temporary)
            (evidence_dir / "evidence.json").write_text(
                json.dumps({"benchmark": {"correct": False}}), encoding="utf-8"
            )
            self.assertFalse(precision_from_records([{"evidence_dir": str(evidence_dir)}]))

    def test_busy_device_parser_only_reads_process_rows(self) -> None:
        output = """
| 0      | Ascend950PR      | OK            | 285.2 |
| 1      | Ascend950PR      | OK            | 266.1 |
| 0                         | 1579853       |       | 314 |
| No running processes found in NPU 1                   |
| 3                         | 1577161       |       | 1483 |
"""
        self.assertEqual(parse_busy_devices(output), {"0", "3"})

    def test_audit_obeys_explicit_event_target(self) -> None:
        evidence = {
            "benchmark": {
                "correct": True,
                "profiler": {
                    "average_us": 10.0
                },
                "elapsed_us": {
                    "median": 12.5
                },
            },
            "measurement_metrics": {
                "costmodel_target": "device_event_median_us"
            },
        }
        self.assertEqual(measured_us(evidence),
                         (12.5, "device_event_median_us"))
        self.assertFalse(timing_gate_applies("device_event_median_us"))
        self.assertTrue(timing_gate_applies("npu_profiler_kernel_duration_average_us"))
        self.assertTrue(timing_gate_passes("device_event_median_us", -0.45, 0.10))
        self.assertFalse(timing_gate_passes("npu_profiler_kernel_duration_average_us", -0.45, 0.10))

    def test_prepare_cache_then_measure_with_device_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            driver = root / "fake_benchmark.py"
            driver.write_text(
                textwrap.dedent("""
                    import argparse
                    import json
                    import os

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--profile-dir")
                    parser.add_argument("--report")
                    args = parser.parse_args()
                    result = {
                        "correct": True,
                        "elapsed_us": {"median": 12.5},
                        "profiler": (
                            {"average_us": 10.0} if args.profile_dir else None
                        ),
                        "always_compile": os.environ.get("TRITON_ALWAYS_COMPILE"),
                    }
                    print(json.dumps(result))
                    """),
                encoding="utf-8",
            )
            benchmark = [
                sys.executable,
                str(driver),
                "--profile-dir",
                str(root / "profiler"),
            ]
            common = [
                sys.executable,
                str(COLLECTOR),
                "--name",
                "fake__auto_f1",
                "--output-root",
                str(output_root),
                "--timing-source",
                "event",
            ]
            prepared = subprocess.run(
                [
                    *common,
                    "--prepare-only",
                    "--prepare-without-profiler",
                    "--",
                    *benchmark,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            case_dir = output_root / "fake__auto_f1"
            self.assertTrue((case_dir / "prepare.json").is_file())
            self.assertTrue((case_dir / "cache").is_dir())

            measured = subprocess.run(
                [
                    *common,
                    "--reuse-prepared",
                    "--measure-in-fresh-process",
                    "--measurement-processes",
                    "3",
                    "--measurement-selection",
                    "median",
                    "--",
                    *benchmark,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(measured.returncode, 0, measured.stderr)
            evidence = json.loads((case_dir / "evidence.json").read_text())
            self.assertEqual(evidence["measurement_process_count"], 3)
            self.assertEqual(evidence["measurement_source_requested"], "event")
            self.assertEqual(
                evidence["measurement_metrics"]["costmodel_target"],
                "device_event_median_us",
            )
            self.assertEqual(
                evidence["measurement_metrics"]["device_event_median_us"],
                12.5)
            self.assertIsNone(evidence["measurement_metrics"]
                              ["profiler_kernel_duration_average_us"])
            self.assertEqual(evidence["benchmark"]["always_compile"], "0")


if __name__ == "__main__":
    unittest.main()
