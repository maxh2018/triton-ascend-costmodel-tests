"""Exercise nightly subprocesses and dashboard verdicts without NPU dependencies."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from costmodel_tests import dashboard, nightly

FAKE_RUNNER = r'''
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path);p.add_argument('--output-root',type=Path)
a,unused=p.parse_known_args()
mode=os.environ.get('NIGHTLY_FAKE_MODE','pass')
if mode=='preflight':print('Profile preflight failed: incompatible schema');raise SystemExit(2)
if mode=='timeout':time.sleep(30)
raw=a.output_root;raw.mkdir()
manifest=json.loads(a.manifest.read_text());cases=[];records=[];audits=[]
for c in manifest['cases']:
 name=c['name'];directory=raw/'cases'/name/'attempt_02';directory.mkdir(parents=True)
 evidence_dir=directory/(name+'__auto_f1');evidence_dir.mkdir()
 (directory.parent/'attempt_01.log').write_text('TsdOpen failed / 507033')
 log=directory.parent/'attempt_02.log';log.write_text('finished')
 correct=mode not in ('precision','compile','device')
 if mode=='precision':log.write_text('Tensor-likes are not close!')
 if mode=='compile':log.write_text('CompilationError: unsupported operation')
 if mode=='device':log.write_text('Failed to start the device. TsdOpen failed 507033')
 benchmark={'profiler':{'average_us':10}}
 if mode!='compile':benchmark['correct']=correct
 (evidence_dir/'evidence.json').write_text(json.dumps({'benchmark':benchmark,'returncode':0 if correct else 1}))
 event=mode=='event'
 row={'case':name,'mode':'auto','factor':1,'predicted_us':12 if mode=='time' else 10,'measured_us':10,
      'relative_error':.2 if mode=='time' else 0,'failure_reasons':[],
      'timing_gate_applied':not event,'pass':mode not in ('precision','compile','device'),
      'measurement_source':'device_event_median_us' if event else 'npu_profiler_kernel_duration_average_us'}
 cases.append({'case':name,'status':'PASS' if correct else 'FAIL','precision_pass':correct,
               'attempt_count':2,'attempts':[{'log':str(directory.parent/'attempt_01.log')},{'log':str(log)}],
               'audit_results':[row]})
 audits.append(row);records.append({'name':name+'__auto_f1','evidence_dir':str(evidence_dir)})
 if mode=='partial':break
(raw/'matrix_summary.json').write_text(json.dumps({'records':records}))
(raw/'generalization_audit.json').write_text(json.dumps({'results':audits}))
filename='batch_summary.partial.json' if mode=='partial' else 'batch_summary.json'
(raw/filename).write_text(json.dumps({'status':'PASS','cases':cases}))
raise SystemExit(1 if mode=='partial' else 0)
'''


class NightlyTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "driver.py").write_text("# placeholder")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps({
                "cases": [{
                    "name": n,
                    "driver": "driver.py",
                    "family": "elementwise"
                } for n in ("first", "second")]
            }))
        self.runner = self.root / "fake_runner.py"
        self.runner.write_text(FAKE_RUNNER)
        self.args = SimpleNamespace(manifest=self.manifest,
                                    output_root=self.root / "nightly",
                                    device="1",
                                    retries=3,
                                    timeout_hours=1,
                                    profile=None,
                                    expected_bishengir=None,
                                    triton_root=None,
                                    npuir_root=None,
                                    label="offline fixture")

    def run_job(self, mode="pass", health=None):
        with patch.dict(os.environ,
                        {"NIGHTLY_FAKE_MODE": mode}), redirect_stdout(
                            io.StringIO()):
            code = nightly.execute(self.args,
                                   runner=self.runner,
                                   device_check=lambda _: health or {
                                       "ok": True,
                                       "status": "OK",
                                       "output": "OK"
                                   })
        run = sorted((self.args.output_root / "runs").iterdir())[-1]
        return code, run, dashboard.read_json(run / "dashboard.json")

    def test_success_uses_profiler_serial_and_keeps_retry_history(self):
        code, run, report = self.run_job()
        self.assertEqual(code, 0)
        self.assertEqual(report["counts"], {"PASS": 2})
        command = dashboard.read_json(run / "run.json")["command"]
        for flag, value in [("--timing-source", "profiler"),
                            ("--measurement-processes", "3"), ("--jobs", "1"),
                            ("--repeats", "7"), ("--warmup", "3")]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--require-idle-devices", command)
        self.assertEqual(report["rows"][0]["attempts"], 2)
        self.assertTrue((self.args.output_root / "index.html").is_file())

    def test_event_only_pass_is_not_nightly_acceptance(self):
        code, _, report = self.run_job("event")
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"], {"UNVERIFIED": 2})

    def test_numeric_ape_rechecked_even_if_runner_claims_pass(self):
        code, _, report = self.run_job("time")
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"], {"TIME_FAIL": 2})
        self.assertEqual(report["rows"][0]["ape_percent"], 20)

    def test_precision_failure_is_separate(self):
        _, _, report = self.run_job("precision")
        self.assertEqual(report["counts"], {"PRECISION_FAIL": 2})

    def test_compile_error_is_not_precision_failure(self):
        _, _, report = self.run_job("compile")
        self.assertEqual(report["counts"], {"EXECUTION_FAIL": 2})
        self.assertIsNone(report["rows"][0]["precision"])

    def test_device_init_error_is_blocked(self):
        _, _, report = self.run_job("device")
        self.assertEqual(report["counts"], {"BLOCKED": 2})
        self.assertEqual(report["status"], "BLOCKED")

    def test_critical_health_generates_page_without_starting_runner(self):
        code, run, report = self.run_job(health={
            "ok": False,
            "status": "Critical",
            "output": "Critical 40F84E00"
        })
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["counts"], {"NOT_RUN": 2})
        self.assertFalse((run / "raw").exists())
        self.assertIn("40F84E00", report["message"])

    def test_overlapping_nightly_run_is_rejected(self):
        with nightly.output_lock(self.args.output_root):
            with self.assertRaisesRegex(ValueError, "已有夜间测试"):
                self.run_job()

    def test_profile_preflight_failure_still_generates_dashboard(self):
        _, run, report = self.run_job("preflight")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("incompatible schema", report["message"])
        self.assertTrue((run / "index.html").is_file())

    def test_partial_run_does_not_become_pass(self):
        code, _, report = self.run_job("partial")
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"], {"PASS": 1, "NOT_RUN": 1})
        self.assertNotEqual(report["status"], "PASS")

    def test_timeout_keeps_report(self):
        self.args.timeout_hours = .00003
        code, run, report = self.run_job("timeout")
        self.assertEqual(code, 1)
        self.assertEqual(
            dashboard.read_json(run / "run.json")["returncode"], 124)
        self.assertEqual(report["status"], "INCOMPLETE")

    def test_latest_blocked_run_does_not_show_older_green_result(self):
        self.run_job()
        self.run_job(health={
            "ok": False,
            "status": "Critical",
            "output": "Critical"
        })
        reports = dashboard.generate(self.args.output_root)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["status"], "BLOCKED")
        self.assertEqual(reports[1]["status"], "PASS")

    def test_render_escapes_embedded_json(self):
        report = {
            "rows": [{
                "name": '</script><script>alert("injection")</script>'
            }]
        }
        html = dashboard.render([report])
        self.assertNotIn(report["rows"][0]["name"], html)
        self.assertIn("\\u003c/script", html)

    def test_paths_from_copied_linux_evidence_resolve_locally(self):
        _, run, _ = self.run_job()
        metadata = dashboard.read_json(run / "run.json")
        original = metadata["raw_root"]
        metadata["raw_root"] = "/server/nightly/raw"
        dashboard.write_json(run / "run.json", metadata)
        for p in (run / "raw").glob("*.json"):
            p.write_text(p.read_text().replace(original.replace("\\", "\\\\"),
                                               "/server/nightly/raw"))
        report = dashboard.build_report(run)
        self.assertEqual(report["counts"], {"PASS": 2})
        self.assertIsNotNone(report["rows"][0]["evidence"])
        self.assertIsNone(
            dashboard.artifact(run, str(self.root / "outside.log"), metadata))


if __name__ == "__main__":
    unittest.main()
