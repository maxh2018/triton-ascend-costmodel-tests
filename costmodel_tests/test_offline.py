"""CPU-only checks for corpus integrity, audit gates and the suite launcher.

Run: python -m unittest discover -s costmodel_tests -p test_offline.py -v
"""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GENERAL = ROOT / "cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization"


class OfflineTests(unittest.TestCase):
    def test_historical_matrix_is_complete(self):
        cases = json.loads((GENERAL / "generalization_matrix.json").read_text())["cases"]
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({c["name"] for c in cases}), 100)
        self.assertEqual(len({c["driver"] for c in cases}), 19)
        self.assertEqual(sum(5 + len(c.get("extra_modes", [])) for c in cases), 508)
        for case in cases:
            self.assertTrue((GENERAL / case["driver"]).is_file())
        sources = json.loads((ROOT / "costmodel_tests/sources.json").read_text())
        for path in sources["pytest_base_files"] + sources["pytest_added_from_pr"]:
            self.assertTrue((ROOT / "cases/ci" / path).is_file(), path)

    def test_category_defaults_follow_registered_manifests(self):
        for category in ("ci", "personal", "operator_repos"):
            count = len(json.loads((ROOT / "cases" / category / "manifest.json").read_text())["cases"])
            result = subprocess.run([sys.executable, str(ROOT / "costmodel_tests/run.py"),
                                     "list", "--category", category], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Category: {category}", result.stdout)
            self.assertIn(f"Cases: {count}", result.stdout)
        personal_example = ROOT / "cases/personal/examples/custom_manifest.example.json"
        example = json.loads(personal_example.read_text())
        self.assertTrue((personal_example.parent / example["cases"][0]["driver"]).is_file())

    def test_real_audit_rejects_wrong_results_errors_and_missing_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"name": "example", "family": "example", "split": "holdout"}
            (root / "manifest.json").write_text(json.dumps({"cases": [case]}))
            profile = {"kernel_timing": {
                "system_counter_frequency_mhz": 1,
                "boundary_residual_system_cycles_by_physical_program_count": {"1": 0, "2": 0}}}
            (root / "profile.json").write_text(json.dumps(profile))
            candidate = {"legal": True, "total_system_cycles": 10, "runtime_physical_program_count": 1}
            route = {"selected_route_elapsed_prediction": {"applied": True, "predicted_microseconds": 10},
                     "stage_model": {"routes": {"all_simd": candidate},
                                     "all_simt_factor_routes": {str(f): candidate for f in (1, 2, 4)}}}
            records = []
            for mode, factor in (("auto", 1), ("simd", 1), ("simt_only", 1), ("simt_only", 2), ("simt_only", 4)):
                name = f"example__{mode}_f{factor}"
                evidence = root / name
                evidence.mkdir()
                (evidence / "evidence.json").write_text(json.dumps({
                    "returncode": 0, "benchmark": {"correct": True, "profiler": {"average_us": 10}}}))
                if mode == "auto":
                    (evidence / "route.json").write_text(json.dumps(route))
                records.append({"name": name, "family": "example", "split": "holdout", "route": mode,
                                "factor": factor, "returncode": 0, "evidence_dir": str(evidence)})
            summary = root / "matrix_summary.json"
            summary.write_text(json.dumps({"records": records}))
            command = [sys.executable, str(GENERAL / "audit_generalization_matrix.py"),
                       "--validation-scope", "all",
                       "--evidence-root", str(root), "--manifest", str(root / "manifest.json"),
                       "--profile", str(root / "profile.json"), "--output-json", str(root / "audit.json"),
                       "--output-markdown", str(root / "audit.md")]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            first = root / records[0]["name"] / "evidence.json"
            for benchmark in ({"correct": False, "profiler": {"average_us": 10}},
                              {"correct": True, "profiler": {"average_us": 20}}):
                first.write_text(json.dumps({"returncode": 0, "benchmark": benchmark}))
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            summary.write_text(json.dumps({"records": records[:-1]}))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            report = json.loads((root / "audit.json").read_text())
            self.assertTrue(any("missing_matrix_record" in r["failure_reasons"] for r in report["results"]))

            # Auto-only accepts the selected route even if an unselected route fails its prediction gate.
            summary.write_text(json.dumps({"records": records}))
            first.write_text(json.dumps({"returncode": 0, "benchmark": {
                "correct": True, "profiler": {"average_us": 10}}}))
            explicit = root / records[1]["name"] / "evidence.json"
            explicit.write_text(json.dumps({"returncode": 0, "benchmark": {
                "correct": True, "profiler": {"average_us": 100}}}))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            auto_command = command.copy()
            del auto_command[2:4]  # Default audit scope is Auto, no explicit override.
            self.assertEqual(subprocess.run(auto_command, capture_output=True).returncode, 0)
            auto_report = json.loads((root / "audit.json").read_text())
            self.assertEqual(auto_report["validation_scope"], "auto")
            self.assertEqual(auto_report["expected_record_count"], 1)
            self.assertEqual(auto_report["audited_record_count"], 1)

    def test_launcher_propagates_failed_batch(self):
        spec = importlib.util.spec_from_file_location("suite_runner", ROOT / "costmodel_tests/run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"

            def fake_run(command, **kwargs):
                if Path(command[1]).name == "run_generalization_batch.py":
                    output.mkdir()
                    (output / "batch_summary.json").write_text('{"status": "FAIL"}')
                    return subprocess.CompletedProcess(command, 1)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, "argv", ["run.py", "--output-root", str(output)]), \
                    patch.object(module.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit) as stopped:
                    module.main()
            self.assertEqual(stopped.exception.code, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "FAIL")
            self.assertEqual(summary["returncode"], 1)
            self.assertEqual(summary["case_count"], 100)

    def test_custom_manifest_can_be_listed_and_forwarded_without_modifying_original(self):
        spec = importlib.util.spec_from_file_location("custom_suite_runner", ROOT / "costmodel_tests/run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = (GENERAL / "generalization_matrix.json").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "custom_manifest.json"
            manifest.write_text(json.dumps({"cases": [{"name": "custom_add", "family": "elementwise",
                                                       "driver": "bench_custom_add.py"}]}))
            (Path(directory) / "bench_custom_add.py").write_text("# driver\n")
            for suite in ("list", "run", "selected"):
                command = ["run.py", "list" if suite == "list" else "run", "--manifest", str(manifest)]
                if suite != "list":
                    command += ["--dry-run", "--output-root", str(Path(directory) / suite)]
                if suite == "selected":
                    command += ["--case", "custom_add"]
                stream = io.StringIO()
                with patch.object(sys, "argv", command), contextlib.redirect_stdout(stream):
                    module.main()
                if suite == "list":
                    self.assertIn("custom_add", stream.getvalue())
                    self.assertIn("Cases: 1", stream.getvalue())
                else:
                    arguments = json.loads(stream.getvalue().splitlines()[-1].split(": ", 1)[1])
                    self.assertEqual(arguments[arguments.index("--manifest") + 1], str(manifest.resolve()))
                    self.assertEqual(arguments[arguments.index("--validation-scope") + 1], "auto")
                    self.assertFalse((Path(directory) / suite).exists())
        self.assertEqual((GENERAL / "generalization_matrix.json").read_bytes(), original)

    def test_single_case_forwards_scope_to_runner_and_audit(self):
        spec = importlib.util.spec_from_file_location("single_case_scope", GENERAL / "run_generalization_case.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            (root / "drivers").mkdir()
            (root / "drivers/example.py").write_text("# personal driver\n")
            manifest.write_text(json.dumps({"cases": [{"name": "example", "driver": "drivers/example.py", "extra_modes": [
                {"route": "mixed", "factor": 4}]}]}))
            profile = root / "profile.json"
            profile.write_text("{}")
            for scope, count in (("auto", 1), ("all", 6)):
                output = root / scope

                def fake_run(command, env, log):
                    if Path(command[1]).name == "run_generalization_matrix.py":
                        self.assertEqual("--include-explicit" in command, scope == "all")
                        subset = json.loads(Path(command[command.index("--manifest") + 1]).read_text())
                        self.assertEqual(subset["cases"][0]["driver"], str((root / "drivers/example.py").resolve()))
                        for name in ("matrix_summary.json", "generalization_matrix.json", "costmodel_profile.json"):
                            (output / name).write_text("{}")
                    else:
                        self.assertEqual(command[command.index("--validation-scope") + 1], scope)
                        (output / "generalization_audit.json").write_text(json.dumps({
                            "all_pass": True, "expected_record_count": count, "audited_record_count": count}))
                    return 0

                command = ["case.py", "example", "--manifest", str(manifest), "--profile", str(profile),
                           "--expected-bishengir", sys.executable, "--output-root", str(output)]
                if scope == "all":
                    command += ["--validation-scope", "all"]
                with patch.object(sys, "argv", command), patch.object(module, "run_logged", side_effect=fake_run), \
                        patch.dict("os.environ", {"TRITON_ASCEND_COMPILE_MODE": ""}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        module.main()
                self.assertEqual(stopped.exception.code, 0)
                summary = json.loads((output / "single_case_summary.json").read_text())
                self.assertEqual(summary["validation_scope"], scope)
                self.assertEqual(summary["expected_record_count"], count)


if __name__ == "__main__":
    unittest.main()
