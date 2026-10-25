# CPU-only regressions for default manifests and mixed entry scheduling.
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from costmodel_tests.manifests import load_manifest
from costmodel_tests.update_ci_manifests import build_manifests
from costmodel_tests.test_batch import FAKE_WORKER

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / 'cases/ci'
GENERAL = CI / 'third_party/ascend/unittest/costmodel_ut/costmodel_generalization'


class ManifestTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_event_timing_synchronizes_once_after_all_repeats(self):
        benchmark_paths = [
            *sorted((ROOT / 'costmodel_tests/benchmarks').glob(
                'bench_*costmodel.py')),
            *sorted(GENERAL.glob('bench_*costmodel.py')),
            GENERAL / 'fze/bench_fze_costmodel.py',
        ]
        self.assertEqual(len(benchmark_paths), 38)

        def is_call(node, name):
            return (isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == name)

        def inspect_block(path, statements):
            event_loop_count = 0
            for index, statement in enumerate(statements):
                calls = [node for node in ast.walk(statement)
                         if isinstance(node, ast.Call)]
                if (isinstance(statement, ast.For)
                        and any(isinstance(call.func, ast.Attribute)
                                and call.func.attr == 'record'
                                for call in calls)):
                    event_loop_count += 1
                    self.assertFalse(
                        any(isinstance(call.func, ast.Attribute)
                            and call.func.attr == 'synchronize'
                            for call in calls),
                        f'{path}: event loop synchronizes each repeat')
                    self.assertLess(index + 1, len(statements), str(path))
                    self.assertTrue(
                        is_call(statements[index + 1], 'synchronize'),
                        f'{path}: event loop needs one trailing synchronize')
                for field in ('body', 'orelse', 'finalbody'):
                    nested = getattr(statement, field, None)
                    if nested:
                        event_loop_count += inspect_block(path, nested)
            return event_loop_count

        for path in benchmark_paths:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            self.assertEqual(inspect_block(path, tree.body), 1, str(path))

    def test_solve_tril_benchmarks_zero_initialize_unwritten_triangle(self):
        paths = [
            ROOT / 'costmodel_tests/benchmarks/bench_solve_tril_costmodel.py',
            GENERAL / 'bench_solve_tril_costmodel.py',
        ]
        for path in paths:
            source = path.read_text(encoding='utf-8')
            self.assertIn('compile_on_910_95=True', source, str(path))
            self.assertIn('enable_dynamic_cv_pipeline=True', source, str(path))
            tree = ast.parse(source, filename=str(path))
            initializers = [
                node.value.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == 'output' for target in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
            ]
            self.assertEqual(initializers, ['zeros_like'], str(path))

    def test_ci_manifests_match_original_100_and_partition_all_test_files(
            self):
        default, complete = build_manifests()
        self.assertEqual(default, json.loads(
            (CI / 'manifest.json').read_text()))
        self.assertEqual(complete,
                         json.loads((CI / 'manifest_all.json').read_text()))
        self.assertEqual(len(default['cases']), 100)
        self.assertNotIn('validation_status', default)
        self.assertEqual(complete['validation_status'], 'unverified')
        pytest_files = {
            c['driver']
            for c in complete['cases'] if c.get('runner') == 'pytest'
        }
        excluded = json.loads(
            (CI / 'manifest_excluded.json').read_text())['cases']
        excluded_files = {c['driver'] for c in excluded}
        self.assertTrue(all(c['runner'] == 'pytest' for c in excluded))
        self.assertTrue(pytest_files.isdisjoint(excluded_files))
        self.assertEqual(
            pytest_files | excluded_files,
            {p.relative_to(CI).as_posix()
             for p in CI.rglob('test_*.py')})
        self.assertTrue({c['name']
                         for c in complete['cases']
                         }.isdisjoint(c['name'] for c in excluded))
        self.assertEqual(len(excluded), len({c['name'] for c in excluded}))
        self.assertEqual(complete['excluded_manifest'],
                         'manifest_excluded.json')
        self.assertIn(
            'third_party/ascend/unittest/costmodel_ut/costmodel_generalization/fze/test_fze.py',
            pytest_files)
        self.assertEqual(len(complete['cases']), 100 + len(pytest_files))
        self.assertEqual(len(complete['cases']),
                         len({c['name']
                              for c in complete['cases']}))

    def test_default_run_never_selects_unverified_manifest_and_legacy_commands_removed(
            self):
        result = subprocess.run(
            [sys.executable, 'costmodel_tests/run.py', '--dry-run'],
            cwd=ROOT,
            capture_output=True,
            text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Cases: 100', result.stdout)
        self.assertNotIn('UNVERIFIED', result.stdout)
        command = json.loads(result.stdout.splitlines()[-1].split(': ', 1)[1])
        self.assertEqual(Path(command[command.index('--manifest') + 1]),
                         CI / 'manifest.json')
        for legacy in ('matrix', 'pytest', 'all', 'case'):
            result = subprocess.run([
                sys.executable, 'costmodel_tests/run.py', legacy, '--dry-run'
            ],
                                    cwd=ROOT,
                                    capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_explicit_manifest_overrides_category_and_can_mix_ci_and_personal(
            self):
        manifest = ROOT / 'costmodel_tests/mixed_manifest.example.json'
        cases = load_manifest(manifest)['cases']
        self.assertTrue(
            Path(cases[0]['driver']).is_relative_to(ROOT / 'cases/personal'))
        self.assertTrue(
            Path(cases[1]['driver']).is_relative_to(ROOT / 'cases/ci'))
        for category in (None, 'ci', 'personal', 'operator_repos',
                         'unused-category'):
            command = [
                sys.executable,
                str(ROOT / 'costmodel_tests/run.py'), 'list', '--manifest',
                str(manifest)
            ]
            if category is not None:
                command += ['--category', category]
            result = subprocess.run(command,
                                    cwd=ROOT,
                                    capture_output=True,
                                    text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Manifest source: explicit', result.stdout)
            self.assertIn('Cases: 2', result.stdout)
            self.assertNotIn('Category:', result.stdout)
            self.assertIn('custom_add_f32_n1024', result.stdout)
            self.assertIn('pytest_ci_vector_add', result.stdout)

    def test_driver_paths_are_relative_to_json_not_cwd_or_category(self):
        for category in ('ci', 'personal', 'operator_repos'):
            directory = self.root / category
            directory.mkdir()
            (directory / 'driver.py').write_text('# driver\n')
        folder = self.root / 'manifests'
        folder.mkdir()
        manifest = folder / 'mixed.json'
        manifest.write_text(
            json.dumps({
                'cases': [
                    {
                        'name': 'ci',
                        'driver': '../ci/driver.py'
                    },
                    {
                        'name': 'personal',
                        'driver': '../personal/driver.py'
                    },
                    {
                        'name': 'operators',
                        'driver': str(self.root / 'operator_repos/driver.py')
                    },
                ]
            }))
        result = subprocess.run([
            sys.executable,
            str(ROOT / 'costmodel_tests/run.py'), 'list', '--category', 'ci',
            '--manifest',
            str(manifest)
        ],
                                cwd=self.root,
                                capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Cases: 3', result.stdout)
        for category in ('ci', 'personal', 'operator_repos'):
            self.assertIn(str((self.root / category / 'driver.py').resolve()),
                          result.stdout)

    def test_explicit_run_summary_does_not_assign_cases_to_fallback_category(
            self):
        spec = importlib.util.spec_from_file_location(
            'manifest_priority_launcher', ROOT / 'costmodel_tests/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = ROOT / 'costmodel_tests/mixed_manifest.example.json'
        output = self.root / 'output'

        def fake_run(command, **kwargs):
            self.assertEqual(command[command.index('--manifest') + 1],
                             str(manifest))
            self.assertNotIn('--category', command)
            output.mkdir()
            (output / 'batch_summary.json').write_text('{"status": "PASS"}')
            return subprocess.CompletedProcess(command, 0)

        with patch.object(sys, 'argv', ['run.py', '--manifest', str(manifest), '--category', 'personal',
                                       '--output-root', str(output)]), \
                patch.object(module.subprocess, 'run', side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                module.main()
        self.assertEqual(stopped.exception.code, 0)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['manifest_source'], 'explicit')
        self.assertIsNone(summary['category'])
        self.assertEqual(summary['case_count'], 2)

    def test_bad_explicit_manifest_never_falls_back_and_category_is_checked_only_for_defaults(
            self):
        invalid = self.root / 'invalid.json'
        invalid.write_text('{"cases": "invalid"}')
        for manifest in (self.root / 'missing.json', invalid):
            result = subprocess.run([
                sys.executable,
                str(ROOT / 'costmodel_tests/run.py'), '--manifest',
                str(manifest), '--category', 'ci', '--dry-run'
            ],
                                    cwd=ROOT,
                                    capture_output=True,
                                    text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Cases: 100', result.stdout)
        for category in ('missing-category', ''):
            result = subprocess.run([
                sys.executable,
                str(ROOT / 'costmodel_tests/run.py'), '--category', category,
                '--dry-run'
            ],
                                    cwd=ROOT,
                                    capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_manifest_validation_rejects_duplicate_unknown_runner_missing_driver_and_bad_arguments(
            self):
        driver = self.root / 'driver.py'
        driver.write_text('# driver\n')
        valid = {'name': 'example', 'driver': 'driver.py'}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'cases': [valid]}))
        self.assertEqual(
            load_manifest(path)['cases'][0]['runner'], 'benchmark')
        for cases in ([valid, valid], [{
                **valid, 'runner': 'unknown'
        }], [{
                **valid, 'driver': 'missing.py'
        }], [{
                **valid, 'arguments': [1]
        }]):
            path.write_text(json.dumps({'cases': cases}))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_empty_category_can_be_listed_but_cannot_run(self):
        manifest = self.root / 'empty.json'
        manifest.write_text('{"cases": []}')
        command = [
            sys.executable, 'costmodel_tests/run.py', '--category', 'personal',
            '--manifest',
            str(manifest), '--output-root',
            str(self.root / 'output')
        ]
        self.assertEqual(
            subprocess.run(command, cwd=ROOT, capture_output=True).returncode,
            2)
        self.assertFalse((self.root / 'output').exists())
        command.insert(2, 'list')
        result = subprocess.run(command,
                                cwd=ROOT,
                                capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Cases: 0', result.stdout)

    def test_functional_pytest_manifest_reports_timing_not_checked(self):
        driver = self.root / 'test_host.py'
        driver.write_text('def test_add():\n    assert 1 + 1 == 2\n')
        manifest = self.root / 'manifest.json'
        manifest.write_text(
            json.dumps({
                'cases': [{
                    'name': 'functional',
                    'runner': 'pytest',
                    'driver': 'test_host.py',
                    'arguments': ['--costmodel-functional-only']
                }]
            }))
        profile = self.root / 'profile.json'
        profile.write_text('{}')
        output = self.root / 'output'
        result = subprocess.run(
            [
                sys.executable, 'costmodel_tests/run.py', '--manifest',
                str(manifest), '--profile',
                str(profile), '--retries', '0', '--output-root',
                str(output)
            ],
            cwd=ROOT,
            env={
                **os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
                'TRITON_ASCEND_COMPILE_MODE': ''
            },
            capture_output=True,
            text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        case = json.loads((output / 'report.json').read_text())['cases'][0]
        self.assertEqual(
            json.loads(
                (output / 'report.json').read_text())['validation_scope'],
            'functional')
        self.assertEqual(case['validation_scope'], 'functional')
        self.assertTrue(case['precision_pass'])
        self.assertIsNone(case['timing_pass'])
        self.assertIn('Not checked', (output / 'report.md').read_text())

    def test_real_pytest_manifest_retries_only_failed_parameterized_node(self):
        tests = self.root / 'tests'
        tests.mkdir()
        (tests / 'conftest.py').write_text('''
from costmodel_tests.test_pytest_costmodel import FakeBackend

def pytest_sessionstart(session):
    session.config._costmodel_adapter.backend_factory = FakeBackend
''')
        (tests / 'test_example.py').write_text('''
from pathlib import Path
import pytest
from costmodel_tests.test_pytest_costmodel import FakeJit

@pytest.mark.parametrize('n', [1, 2])
def test_add(n):
    here = Path(__file__).parent
    calls = here / f'calls_{n}.txt'
    with calls.open('a') as stream:
        stream.write('called\\n')
    first = n == 2 and not (here / 'retried').exists()
    if first:
        (here / 'retried').write_text('yes')
    value = 3 if first else 1
    kernel = FakeJit()
    kernel.run(value, grid=(1,), warmup=False)
    assert kernel.actual[-1] == value
''')
        manifest = self.root / 'manifest.json'
        manifest.write_text(
            json.dumps({
                'cases': [{
                    'name': 'add',
                    'runner': 'pytest',
                    'driver': 'tests/test_example.py'
                }]
            }))
        profile = self.root / 'profile.json'
        profile.write_text('{}')
        output = self.root / 'output'
        env = {
            **os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
            'TRITON_ASCEND_COMPILE_MODE': ''
        }
        command = [
            sys.executable, 'costmodel_tests/run.py', '--manifest',
            str(manifest), '--category', 'operator_repos', '--profile',
            str(profile), '--output-root',
            str(output), '--retries', '2'
        ]
        result = subprocess.run(command,
                                cwd=ROOT,
                                env=env,
                                capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'PASS')
        self.assertEqual(summary['manifest_source'], 'explicit')
        self.assertIsNone(summary['category'])
        batch = json.loads((output / 'batch_summary.json').read_text())
        case = batch['cases'][0]
        self.assertEqual(case['attempt_count'], 2)
        self.assertEqual([a['status'] for a in case['attempts']],
                         ['FAIL', 'PASS'])
        self.assertEqual(len(case['pytest_results']), 2)
        self.assertEqual((tests / 'calls_1.txt').read_text().splitlines(),
                         ['called'])
        self.assertEqual((tests / 'calls_2.txt').read_text().splitlines(),
                         ['called', 'called'])
        self.assertEqual(len(case['attempts'][1]['targets']), 1)
        self.assertTrue(
            case['attempts'][1]['targets'][0].endswith('::test_add[2]'))
        self.assertTrue((output / 'report.md').is_file())
        self.assertTrue(
            json.loads((output /
                        'report.json').read_text())['cases'][0]['timing_pass'])
        self.assertFalse((output / 'generalization_audit.json').exists())

    def test_mixed_manifest_audits_benchmarks_without_expectations_for_pytest_routes(
            self):
        driver = self.root / 'driver.py'
        driver.write_text('# benchmark placeholder\n')
        worker = self.root / 'worker.py'
        worker.write_text(FAKE_WORKER)
        profile = self.root / 'profile.json'
        profile.write_text(
            json.dumps({
                'kernel_timing': {
                    'system_counter_frequency_mhz': 1,
                    'boundary_residual_system_cycles_by_physical_program_count':
                    {
                        '1': 0,
                        '2': 0
                    }
                }
            }))
        manifest = self.root / 'manifest.json'
        manifest.write_text(
            json.dumps({
                'cases': [{
                    'name': 'stable',
                    'driver': 'driver.py'
                }, {
                    'name': 'pytest_example',
                    'runner': 'pytest',
                    'driver': 'driver.py'
                }]
            }))
        spec = importlib.util.spec_from_file_location(
            'mixed_batch', GENERAL / 'run_generalization_batch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = self.root / 'output'
        fake_pytest = {
            'case': 'pytest_example',
            'runner': 'pytest',
            'status': 'PASS',
            'attempt_count': 1,
            'final_output_root':
            str(output / 'cases/pytest_example/attempt_01'),
            'attempts': [],
            'precision_pass': True,
            'timing_pass': True,
            'pytest_results': []
        }
        with patch.object(sys, 'argv', ['batch.py', '--manifest', str(manifest), '--profile', str(profile),
                                       '--expected-bishengir', sys.executable, '--output-root', str(output)]), \
                patch.object(module, 'SINGLE_CASE_SCRIPT', worker), \
                patch.object(module, 'run_pytest_case', return_value=fake_pytest), \
                patch.dict(os.environ, {'BATCH_TEST_AUDIT': str(GENERAL / 'audit_generalization_matrix.py'),
                                        'TRITON_ASCEND_COMPILE_MODE': ''}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                module.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(
            json.loads((output / 'batch_summary.json').read_text())['status'],
            'PASS')
        audit = json.loads((output / 'generalization_audit.json').read_text())
        self.assertEqual(audit['expected_record_count'], 1)
        self.assertTrue(audit['all_pass'])
        self.assertEqual(
            len(json.loads((output / 'report.json').read_text())['cases']), 2)


if __name__ == '__main__':
    unittest.main()
