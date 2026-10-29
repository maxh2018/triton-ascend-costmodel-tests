"""Exercise nightly subprocesses and dashboard verdicts without NPU dependencies."""
from contextlib import redirect_stdout
import hashlib
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
                                    require_idle_devices=False,
                                    label="offline fixture")

    def run_job(self, mode="pass", health=None, occupancy=None):
        with patch.dict(os.environ,
                        {"NIGHTLY_FAKE_MODE": mode}), redirect_stdout(
                            io.StringIO()):
            code = nightly.execute(self.args,
                                   runner=self.runner,
                                   device_check=lambda _: health or {
                                       "ok": True,
                                       "status": "OK",
                                       "output": "OK"
                                   },
                                   occupancy_check=lambda _: occupancy or {
                                       "status": "IDLE",
                                       "processes": [],
                                       "output": "idle"
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
        self.assertNotIn("--require-idle-devices", command)
        self.assertEqual(report["rows"][0]["attempts"], 2)
        self.assertTrue((self.args.output_root / "index.html").is_file())

    def test_busy_device_warns_and_runs_by_default(self):
        busy = {
            "status": "BUSY",
            "processes": [{
                "pid": 12345,
                "name": "python",
                "memory_mb": "512"
            }],
            "output": "busy"
        }
        code, run, report = self.run_job(occupancy=busy)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertIn("12345", report["warnings"][0])
        self.assertEqual(report["occupancy"]["processes"], busy["processes"])
        self.assertTrue((run / "raw/batch_summary.json").is_file())
        self.assertIn("WARNING",
                      (run / "nightly.log").read_text(encoding="utf-8"))
        self.assertEqual(dashboard.read_json(run / "device_occupancy.json"),
                         busy)

    def test_busy_device_only_blocks_in_explicit_strict_mode(self):
        self.args.require_idle_devices = True
        code, run, report = self.run_job(occupancy={
            "status": "BUSY",
            "processes": [{
                "pid": 12345
            }],
            "output": "busy"
        })
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse((run / "raw").exists())
        self.assertIn("--require-idle-devices",
                      dashboard.read_json(run / "run.json")["command"])

    def test_unknown_occupancy_warns_and_continues_by_default(self):
        code, _, report = self.run_job(occupancy={
            "status": "UNKNOWN",
            "processes": [],
            "output": "query timeout"
        })
        self.assertEqual(code, 0)
        self.assertIn("未能确认", report["warnings"][0])
        self.assertEqual(report["status"], "PASS")

    def test_unknown_occupancy_blocks_only_in_strict_mode(self):
        self.args.require_idle_devices = True
        code, _, report = self.run_job(occupancy={
            "status": "UNKNOWN",
            "processes": [],
            "output": "query timeout"
        })
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "BLOCKED")

    def test_busy_warning_does_not_relax_timing_acceptance(self):
        code, _, report = self.run_job("time",
                                       occupancy={
                                           "status": "BUSY",
                                           "processes": [{
                                               "pid": 12345
                                           }],
                                           "output": "busy"
                                       })
        self.assertEqual(code, 1)
        self.assertEqual(report["counts"], {"TIME_FAIL": 2})
        self.assertTrue(report["warnings"])

    def test_process_table_parser_supports_actual_four_column_output(self):
        text = '''
| NPU ID | Name | Health | Power(W) |
| 1 | Ascend950PR | OK | 278 |
| NPU ID | Process id | Process name | Process memory(MB) |
| No running processes found in NPU 0 |
| 1 | 1058350 | | 2084 |
| 1 | 1059767 | python | 1562 |
'''
        processes, known = nightly.parse_device_processes(text, "1")
        self.assertTrue(known)
        self.assertEqual([p["pid"] for p in processes], [1058350, 1059767])
        self.assertEqual(processes[0]["memory_mb"], "2084")
        self.assertEqual(processes[0]["name"], "")
        self.assertEqual(nightly.parse_device_processes(text, "0"), ([], True))
        self.assertEqual(
            nightly.parse_device_processes(
                "| No running processes found in NPU 10 |", "1"), ([], False))

    def test_process_table_parser_uses_pid_column_in_chip_format(self):
        text = '''
| NPU ID | Chip ID | Process id | Process name | Process memory(MB) |
| 0 | 0 | 1234 | python | 100 |
| 1 | 0 | 5678 | python | 200 |
'''
        processes, known = nightly.parse_device_processes(text, "1")
        self.assertTrue(known)
        self.assertEqual([p["pid"] for p in processes], [5678])

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

    def test_case_json_view_preserves_full_numbers_and_original_bytes(self):
        _, run, _ = self.run_job()
        matrix = dashboard.read_json(run / "raw/matrix_summary.json")
        directory = Path(matrix["records"][0]["evidence_dir"])
        original = b'{\r\n  "count": 18446744073709551615, "payload": "all_fields_visible"\r\n}\r\n'
        (directory / "route.json").write_bytes(original)
        reports = dashboard.generate(self.args.output_root)
        row = reports[0]["rows"][0]
        self.assertTrue((run / row["json_view"]).is_file())
        html = (run / row["json_view"]).read_text(encoding="utf-8")
        data = json.loads(
            html.split('<script id="case-json-data" type="application/json">')
            [1].split('</script>')[0])
        route = data["documents"][0]
        self.assertIn("route.json", route["name"])
        self.assertIn("18446744073709551615", route["text"])
        self.assertEqual(route["raw"].encode(), original)
        self.assertEqual(route["sha256"], hashlib.sha256(original).hexdigest())
        self.assertNotIn("all_fields_visible",
                         (self.args.output_root /
                          "index.html").read_text(encoding="utf-8"))

    def test_case_json_includes_previous_attempts_but_defaults_to_latest(self):
        _, run, _ = self.run_job()
        batch = dashboard.read_json(run / "raw/batch_summary.json")
        case = batch["cases"][0]
        directory = run / "raw/cases/first"
        for index in (1, 2):
            output = directory / f"attempt_{index:02d}"
            output.mkdir(exist_ok=True)
            case["attempts"][index - 1]["output_root"] = str(output)
            dashboard.write_json(output / "single_case_summary.json",
                                 {"attempt": index})
        case["final_output_root"] = str(directory / "attempt_02")
        dashboard.write_json(run / "raw/batch_summary.json", batch)
        report = dashboard.generate(self.args.output_root)[0]
        files = report["rows"][0]["json_files"]
        self.assertTrue(any("attempt_01" in x["path"] for x in files))
        self.assertIn("attempt_02", files[0]["path"])
        self.assertEqual(report["rows"][0]["status"], "PASS")

    def test_case_json_shows_malformed_file_without_crashing(self):
        _, run, _ = self.run_job()
        output = run / "raw/cases/first/attempt_02"
        (output / "single_case_summary.json").write_text('{"truncated":')
        report = dashboard.generate(self.args.output_root)[0]
        files = report["rows"][0]["json_files"]
        bad = next(x for x in files
                   if x["path"].endswith("single_case_summary.json"))
        self.assertIn("JSON 无法解析", bad["error"])

    def test_case_json_does_not_embed_scripts_or_read_outside_run(self):
        _, run, _ = self.run_job()
        matrix = dashboard.read_json(run / "raw/matrix_summary.json")
        directory = Path(matrix["records"][0]["evidence_dir"])
        attack = '</script><script>alert("injection")</script>'
        dashboard.write_json(directory / "route.json", {"note": attack})
        report = dashboard.generate(self.args.output_root)[0]
        html = (run /
                report["rows"][0]["json_view"]).read_text(encoding="utf-8")
        self.assertNotIn(attack, html)
        data = json.loads(
            html.split('<script id="case-json-data" type="application/json">')
            [1].split('</script>')[0])
        self.assertEqual(attack,
                         json.loads(data["documents"][0]["text"])["note"])
        outside = self.root / "outside"
        outside.mkdir()
        dashboard.write_json(outside / "route.json",
                             {"secret": "outside_only"})
        docs = dashboard.case_json_documents(
            run, "first", {"final_output_root": str(outside)}, [], {})
        self.assertEqual(docs, [])

    def test_case_json_keeps_large_files_complete_and_skips_profiler_traces(
            self):
        _, run, _ = self.run_job()
        matrix = dashboard.read_json(run / "raw/matrix_summary.json")
        directory = Path(matrix["records"][0]["evidence_dir"])
        text = "x" * 300000 + "end_of_full_document"
        dashboard.write_json(directory / "route.json", {"full_content": text})
        trace = directory / "profiler"
        trace.mkdir()
        dashboard.write_json(trace / "trace.json", {"large_trace": True})
        report = dashboard.generate(self.args.output_root)[0]
        row = report["rows"][0]
        html = (run / row["json_view"]).read_text(encoding="utf-8")
        self.assertIn("end_of_full_document", html)
        self.assertFalse(
            any(x["path"].endswith("trace.json") for x in row["json_files"]))

    def test_unexecuted_case_has_no_fake_json_documents(self):
        _, _, report = self.run_job(health={
            "ok": False,
            "status": "Critical",
            "output": "Critical"
        })
        self.assertIsNone(report["rows"][0]["json_view"])
        self.assertEqual(report["rows"][0]["json_files"], [])

    def test_unchanged_json_views_are_cached_and_changes_are_reflected(self):
        _, run, _ = self.run_job()
        with patch.object(dashboard,
                          "case_json_documents",
                          wraps=dashboard.case_json_documents) as collect:
            dashboard.generate(self.args.output_root)
            collect.assert_not_called()
        matrix = dashboard.read_json(run / "raw/matrix_summary.json")
        directory = Path(matrix["records"][0]["evidence_dir"])
        dashboard.write_json(directory / "route.json",
                             {"new_field_after_refresh": 123})
        report = dashboard.generate(self.args.output_root)[0]
        html = (run /
                report["rows"][0]["json_view"]).read_text(encoding="utf-8")
        self.assertIn("new_field_after_refresh", html)

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
