"""CPU-only subprocess tests for concurrency and selective retries, using the real audit."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from costmodel_tests.test_profile_resolution import fake_ta, write_test_profile

GENERAL = Path(__file__).resolve().parents[1] / "cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization"

FAKE_WORKER = r'''
import argparse, json, os
from pathlib import Path
import subprocess, sys, time

p = argparse.ArgumentParser()
p.add_argument('case')
p.add_argument('--output-root', type=Path)
p.add_argument('--manifest', type=Path)
p.add_argument('--profile', type=Path)
p.add_argument('--validation-scope', choices=('auto', 'all', 'compare'), default='auto')
a, unused = p.parse_known_args()
profile = json.loads(a.profile.read_text())
if profile.get('microbenchmark_profile'):
    assert (a.profile.parent / profile['microbenchmark_profile']).is_file()
output = a.output_root
output.mkdir()
number = int(output.name.split('_')[-1])
started = time.monotonic()
time.sleep(0.25)
mode = os.environ.get('BATCH_TEST_MODE', 'flaky')
if mode == 'missing':
    raise SystemExit(1)
cases = json.loads(a.manifest.read_text())
case = next(c for c in cases['cases'] if c['name'] == a.case)
assert Path(case['driver']).is_absolute() and Path(case['driver']).is_file()
assert str(Path(os.environ['BATCH_TEST_AUDIT']).parent) in os.environ['PYTHONPATH'].split(os.pathsep)
cases['cases'] = [case]
(output / 'manifest.json').write_text(json.dumps(cases))
fail = mode == 'always' or (a.case == 'flaky' and number == 1)
modes = [('auto', 1), ('simd', 1), ('simt_only', 1), ('simt_only', 2), ('simt_only', 4)]
modes += [(m['route'], m['factor']) for m in case.get('extra_modes', [])]
if a.validation_scope == 'auto':
    modes = [('auto', 1)]
elif a.validation_scope == 'compare':
    modes = [('auto', 1), ('simd', 1), ('simt_only', 1)]
    modes += [(m['route'], m['factor']) for m in case.get('extra_modes', []) if m['route']=='mixed']
candidate = {'legal': True, 'total_system_cycles': 10, 'runtime_physical_program_count': 1}
route = {'selected_route_elapsed_prediction': {'applied': True, 'predicted_microseconds': 10},
         'stage_model': {'routes': {'all_simd': candidate, 'mixed_simd_simt': candidate},
                         'all_simt_factor_routes': {str(f): candidate for f in (1, 2, 4)}}}
records = []
for mode_name, factor in modes:
    name = f'{a.case}__{mode_name}_f{factor}'
    directory = output / name
    directory.mkdir()
    (directory / 'evidence.json').write_text(json.dumps({
        'returncode': 0, 'benchmark': {'correct': True, 'profiler': {'average_us': 20 if fail else 10}}}))
    if mode_name == 'auto':
        (directory / 'route.json').write_text(json.dumps(route))
    records.append({'name': name, 'family': case['family'], 'split': case['split'],
                    'route': mode_name, 'factor': factor, 'returncode': 0, 'evidence_dir': str(directory)})
(output / 'matrix_summary.json').write_text(json.dumps({'records': records}))
result = subprocess.run([sys.executable, os.environ['BATCH_TEST_AUDIT'],
    '--validation-scope', a.validation_scope,
    '--evidence-root', str(output), '--manifest', str(output / 'manifest.json'),
    '--profile', str(a.profile), '--output-json', str(output / 'generalization_audit.json'),
    '--output-markdown', str(output / 'generalization_audit.md')], capture_output=True)
audit = json.loads((output / 'generalization_audit.json').read_text())
(output / 'single_case_summary.json').write_text(json.dumps({'status': 'PASS' if audit['all_pass'] else 'FAIL'}))
(output / 'timing.json').write_text(json.dumps({'start': started, 'end': time.monotonic(),
    'device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}))
raise SystemExit(result.returncode)
'''

FAKE_MATRIX = r'''
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path)
p.add_argument('--output-root', type=Path)
p.add_argument('--profile', type=Path)
a, unused = p.parse_known_args()
manifest = json.loads(a.manifest.read_text())
candidate = {'legal': True, 'total_system_cycles': 10,
             'runtime_physical_program_count': 1}
route = {'selected_route_elapsed_prediction': {
             'applied': True, 'predicted_microseconds': 10},
         'stage_model': {'routes': {
             'all_simd': candidate, 'mixed_simd_simt': candidate},
             'all_simt_factor_routes': {
                 str(factor): candidate for factor in (1, 2, 4)}}}
records = []
for case in manifest['cases']:
    name = f"{case['name']}__auto_f1"
    directory = a.output_root / name
    directory.mkdir()
    (directory / 'evidence.json').write_text(json.dumps({
        'returncode': 0,
        'benchmark': {'correct': True, 'elapsed_us': {'median': 10}},
        'measurement_metrics': {'costmodel_target': 'device_event_median_us'},
    }))
    (directory / 'route.json').write_text(json.dumps(route))
    records.append({
        'name': name, 'family': case['family'], 'split': case['split'],
        'route': 'auto', 'factor': 1, 'returncode': 0,
        'evidence_dir': str(directory),
    })
(a.output_root / 'matrix_summary.json').write_text(json.dumps({
    'source_stable_during_run': True,
    'compiler_stable_during_run': True,
    'records': records,
}))
'''


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment, _ = fake_ta(self.root)
        environment_patch = patch.dict(os.environ, environment)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        self.worker = self.root / "fake_worker.py"
        self.worker.write_text(FAKE_WORKER, encoding="utf-8")
        self.matrix_worker = self.root / "fake_matrix.py"
        self.matrix_worker.write_text(FAKE_MATRIX, encoding="utf-8")
        self.profile = self.root / "profile.json"
        write_test_profile(self.profile, {"kernel_timing": {"system_counter_frequency_mhz": 1,
            "boundary_residual_system_cycles_by_physical_program_count": {"1": 0, "2": 0}}})
        self.manifest = self.root / "manifest.json"
        (self.root / "drivers").mkdir()
        (self.root / "drivers/worker.py").write_text("# custom benchmark placeholder\n")
        self.manifest.write_text(json.dumps({"cases": [
            {"name": "stable", "family": "example", "split": "holdout",
             "driver": "drivers/worker.py",
             "extra_modes": [{"route": "mixed", "factor": 4}]},
            {"name": "flaky", "family": "example", "split": "holdout", "driver": "drivers/worker.py"}]}))
        spec = importlib.util.spec_from_file_location("batch_under_test", GENERAL / "run_generalization_batch.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def run_batch(self, *extra, mode="flaky", expected_status=0, scope="all"):
        output = self.root / "results"
        command = ["batch.py", "--manifest", str(self.manifest), "--profile", str(self.profile),
                   "--expected-bishengir", sys.executable, "--output-root", str(output), *extra]
        if scope is not None:
            command += ["--validation-scope", scope]
        with patch.object(sys, "argv", command), \
                patch.object(self.module, "SINGLE_CASE_SCRIPT", self.worker), \
                patch.object(self.module, "MATRIX_SCRIPT", self.matrix_worker), \
                patch.dict(os.environ, {"BATCH_TEST_MODE": mode,
                                        "BATCH_TEST_AUDIT": str(GENERAL / "audit_generalization_matrix.py"),
                                        "TRITON_ASCEND_COMPILE_MODE": ""}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                self.module.main()
        self.assertEqual(stopped.exception.code, expected_status)
        return output

    def test_concurrent_device_slots_and_retry_only_failed_case(self):
        output = self.run_batch("--jobs", "2", "--devices", "0,1")
        batch = json.loads((output / "batch_summary.json").read_text())
        stable, flaky = batch["cases"]
        self.assertEqual(batch["status"], "PASS")
        self.assertEqual([stable["attempt_count"], flaky["attempt_count"]], [1, 2])
        self.assertEqual([a["status"] for a in flaky["attempts"]], ["FAIL", "PASS"])
        self.assertNotEqual(stable["device"], flaky["device"])
        first = [json.loads((output / "cases" / name / "attempt_01/timing.json").read_text())
                 for name in ("stable", "flaky")]
        self.assertLess(max(r["start"] for r in first), min(r["end"] for r in first))
        self.assertFalse((output / "cases/stable/attempt_02").exists())
        records = json.loads((output / "matrix_summary.json").read_text())["records"]
        self.assertEqual(len(records), 11)  # Mixed record remains included.
        self.assertTrue(all("attempt_02" in r["evidence_dir"] for r in records if r["name"].startswith("flaky__")))
        self.assertTrue(json.loads((output / "generalization_audit.json").read_text())["all_pass"])

    def test_comparison_modes_run_only_declared_mix_references(self):
        output = self.run_batch("--jobs", "2", "--device", "1", "--retries", "1",
                                mode="stable", scope="compare")
        summary = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["measurement_count"], 7)
        records = json.loads((output / "matrix_summary.json").read_text())["records"]
        self.assertEqual(len(records), 7)
        self.assertEqual(sum(r['route']=='mixed' for r in records), 1)
        self.assertTrue(any(r['name']=='stable__mixed_f4' for r in records))
        self.assertFalse(any(r['route']=='simt_only' and r['factor']!=1 for r in records))

    def test_batch_profile_snapshot_keeps_shared_microbenchmark_dependency(self):
        dependency = self.root / "microbench/hardware.json"
        dependency.parent.mkdir()
        dependency.write_text('{"measurements": {}}')
        profile = json.loads(self.profile.read_text())
        profile["microbenchmark_profile"] = "microbench/hardware.json"
        write_test_profile(self.profile, profile)
        output = self.run_batch(mode="stable", scope="auto")
        snapshot = json.loads((output / "costmodel_profile.json").read_text())
        self.assertEqual((output / snapshot["microbenchmark_profile"]).read_bytes(), dependency.read_bytes())
        self.assertTrue(json.loads((output / "report.json").read_text())["status"] == "PASS")

    def test_default_three_retries_stop_after_four_attempts(self):
        output = self.run_batch("--case", "flaky", mode="always", expected_status=1)
        batch = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual(batch["cases"][0]["attempt_count"], 4)
        self.assertEqual(batch["failed_cases"], ["flaky"])
        self.assertFalse((output / "cases/stable").exists())
        self.assertFalse((output / "cases/flaky/attempt_05").exists())

    def test_default_auto_scope_runs_one_record_per_case_and_retries_only_failure(self):
        output = self.run_batch("--jobs", "2", scope=None)
        batch = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual(batch["validation_scope"], "auto")
        self.assertEqual([r["attempt_count"] for r in batch["cases"]], [1, 2])
        records = json.loads((output / "matrix_summary.json").read_text())["records"]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["route"] == "auto" for r in records))
        audit = json.loads((output / "generalization_audit.json").read_text())
        self.assertEqual(audit["expected_record_count"], 2)
        self.assertTrue(audit["all_pass"])

    def test_zero_retries_means_one_attempt(self):
        output = self.run_batch("--case", "flaky", "--retries", "0", expected_status=1)
        batch = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual(batch["cases"][0]["attempt_count"], 1)

    def test_event_batch_runs_two_cases_once_and_deduplicates_records(self):
        output = self.run_batch(
            "--jobs", "2", "--devices", "0,1", "--retries", "0",
            "--timing-source", "event", "--case-batch-size", "2",
            mode="stable", scope="auto",
        )
        batch = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual(batch["case_batch_size"], 2)
        self.assertEqual(batch["compile_workers"], 8)
        self.assertEqual(batch["completed_case_count"], 2)
        self.assertEqual([case["status"] for case in batch["cases"]], ["PASS", "PASS"])
        self.assertTrue((output / "batches/batch_000/attempt_01/matrix_summary.json").is_file())
        records = json.loads((output / "matrix_summary.json").read_text())["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual({record["name"] for record in records},
                         {"stable__auto_f1", "flaky__auto_f1"})
        audit = json.loads((output / "generalization_audit.json").read_text())
        self.assertTrue(audit["all_pass"])

    def test_missing_worker_results_are_failed_and_audited(self):
        output = self.run_batch("--case", "stable", "--retries", "1", mode="missing", expected_status=1)
        batch = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual(batch["cases"][0]["attempt_count"], 2)
        audit = json.loads((output / "generalization_audit.json").read_text())
        self.assertFalse(audit["all_pass"])
        self.assertTrue(all("missing_matrix_record" in r["failure_reasons"] for r in audit["results"]))

    def test_invalid_selection_and_options_fail_before_creating_output(self):
        for extra in (("--case", "unknown"), ("--jobs", "0"), ("--retries", "-1"),
                      ("--jobs", "2", "--devices", "0")):
            with self.subTest(extra=extra):
                output = self.run_batch(*extra, expected_status=2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
