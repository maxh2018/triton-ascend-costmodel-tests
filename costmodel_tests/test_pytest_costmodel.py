# CPU-only regressions for the pytest CostModel adapter; no NPU required.
import contextlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

import pytest

from costmodel_tests.pytest_costmodel import CostModelPytest, pytest_configure
from costmodel_tests.pytest_npu import LaunchCapture, NpuBackend, resolve_grid
from costmodel_tests.test_profile_resolution import fake_ta, write_test_profile

ROOT = Path(__file__).resolve().parents[1]


def vector_kernel(n, BLOCK=4):
    pass


class FakeJit:
    def __init__(self):
        self.fn = vector_kernel
        self.signature = inspect.signature(vector_kernel)
        self.actual = []
        self.compiles = []

    def run(self, n, *, grid, warmup, **kwargs):
        if warmup:
            self.compiles.append((grid, kwargs))
            if n != 4:
                Path(kwargs['auto_simt_scope_dump']).write_text(json.dumps({
                    'selected_route_elapsed_prediction': {
                        'applied': True, 'predicted_microseconds': 12 if n == 3 else 10}}))
        else:
            self.actual.append(n)
        return types.SimpleNamespace(name='vector_kernel')


class FakeBackend:
    jit_class = FakeJit

    def __init__(self):
        self.measurements = 0

    def vector_cores(self):
        return 24

    def measure(self, launch, output, kernel_name):
        self.measurements += 1
        return launch(), {'average_us': 10, 'profile_dir': str(output)}


class PytestCostModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / 'profile.json'
        self.profile.write_text('{}')

    def test_pytest_configuration_snapshots_shared_profile_dependency(self):
        dependency = self.root / 'microbench/hardware.json'
        dependency.parent.mkdir()
        dependency.write_text('{"measurements": {}}')
        write_test_profile(self.profile, {'microbenchmark_profile': 'microbench/hardware.json'})
        environment, _ = fake_ta(self.root)
        output = self.root / 'output'
        options = {'costmodel_output': output, 'costmodel_expected_bishengir': None,
                   'costmodel_profile': self.profile, 'costmodel_functional_only': False}
        config = MagicMock()
        config.getoption.side_effect = options.__getitem__
        config.option.numprocesses = 0
        config.option.collectonly = False
        with patch.dict(os.environ, {**environment, 'TRITON_ASCEND_COMPILE_MODE': ''}):
            pytest_configure(config)
        snapshot = json.loads((output / 'costmodel_profile.json').read_text())
        self.assertEqual((output / snapshot['microbenchmark_profile']).read_bytes(), dependency.read_bytes())
        self.assertEqual(config._costmodel_adapter.profile, (output / 'costmodel_profile.json').resolve())

    def test_callable_grid_kwargs_and_no_replayed_in_place_launch(self):
        backend, kernel = FakeBackend(), FakeJit()
        capture = LaunchCapture(self.root / 'evidence', self.profile, backend)
        for _ in range(2):
            capture.run(FakeJit.run, kernel, (1,), lambda meta: (meta['BLOCK'], 2), False, {'BLOCK': 4})
        self.assertEqual(kernel.actual, [1, 1])
        self.assertEqual(len(kernel.compiles), 1)
        self.assertEqual(backend.measurements, 1)
        grid, options = kernel.compiles[0]
        self.assertEqual(grid, (4, 2))
        self.assertEqual(options['logical_program_count_hint'], 8)
        self.assertEqual(options['physical_vector_core_count_hint'], 24)
        self.assertEqual(options['compile_mode'], 'simd_simt')
        self.assertEqual(options['auto_simt_scope_mode'], 'auto')
        self.assertEqual(capture.public_records()[0]['launch_count'], 2)
        self.assertNotIn('_options', capture.public_records()[0])

    def test_specializations_do_not_reuse_measurements_and_hints_are_preserved(self):
        backend, kernel = FakeBackend(), FakeJit()
        capture = LaunchCapture(self.root, self.profile, backend)
        for n, block in [(1, 4), (1, 8), (2, 8)]:
            capture.run(FakeJit.run, kernel, (n,), (2,), False, {
                'BLOCK': block, 'logical_program_count_hint': 100,
                'physical_vector_core_count_hint': 20})
        self.assertEqual(backend.measurements, 3)
        self.assertTrue(all(o['logical_program_count_hint'] == 100 for _, o in kernel.compiles))
        self.assertTrue(all(o['physical_vector_core_count_hint'] == 20 for _, o in kernel.compiles))

    def test_missing_predictions_and_profiler_failures_never_pass(self):
        capture = LaunchCapture(self.root, self.profile, FakeBackend())
        kernel = FakeJit()
        capture.run(FakeJit.run, kernel, (4,), (1,), False, {})
        self.assertFalse(capture.public_records()[0]['pass'])
        self.assertIn('missing_or_invalid_prediction', capture.public_records()[0]['failure_reasons'][0])
        with patch.object(capture.backend, 'measure', side_effect=RuntimeError('profiler missing')):
            with self.assertRaisesRegex(RuntimeError, 'profiler missing'):
                capture.run(FakeJit.run, kernel, (2,), (1,), False, {})
        self.assertFalse(capture.public_records()[1]['pass'])

    def test_compile_only_calls_not_counted_as_runtime_evidence(self):
        kernel = FakeJit()
        with patch.object(FakeJit, 'run', return_value='compiled') as original:
            capture = LaunchCapture(self.root, self.profile, FakeBackend())
            self.assertEqual(capture.run(original, kernel, (1,), (1,), True, {}), 'compiled')
            self.assertFalse(capture.records)
        with self.assertRaisesRegex(ValueError, 'invalid runtime grid'):
            resolve_grid(kernel, (1,), {}, (0,))

    def test_profiler_reads_only_matching_device_time_and_rejects_invalid_data(self):
        backend = NpuBackend.__new__(NpuBackend)
        backend.torch = MagicMock()
        backend.profiler = MagicMock()
        backend.profiler.profile.return_value = contextlib.nullcontext()
        csv = self.root / 'op_statistic.csv'
        csv.write_text('OP Type,Avg Time(us),Count\nother_kernel,999,1\nvector_kernel,10,1\n')
        launch = MagicMock(return_value='output')
        result, timing = backend.measure(launch, self.root, 'vector_kernel')
        self.assertEqual(result, 'output')
        self.assertEqual(timing['average_us'], 10)
        launch.assert_called_once()
        for value in ('nan', '0', '-1'):
            csv.write_text(f'OP Type,Avg Time(us),Count\nvector_kernel,{value},1\n')
            backend.profiler.profile.return_value = contextlib.nullcontext()
            with self.assertRaisesRegex(ValueError, 'invalid profiler time'):
                backend.measure(lambda: None, self.root, 'vector_kernel')

    def run_pytest(self, source, functional_only=False):
        tests = self.root / 'tests'
        tests.mkdir()
        (tests / 'test_adapter.py').write_text(source)
        output = self.root / 'output'
        output.mkdir()
        kernel = FakeJit()
        fake = types.ModuleType('fake_costmodel_runtime')
        fake.kernel = kernel
        adapter = CostModelPytest(output, self.profile, FakeBackend, functional_only=functional_only)
        before = FakeJit.run
        with patch.dict(sys.modules, {'fake_costmodel_runtime': fake}), \
                patch.dict(os.environ, {'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = pytest.main([str(tests), '--confcutdir', str(tests), '-q',
                                  '-o', 'addopts=', '-p', 'no:cacheprovider',
                                  f'--junitxml={self.root / "pytest.xml"}'], plugins=[adapter])
        self.assertIs(FakeJit.run, before)
        return int(status), json.loads((output / 'summary.json').read_text()), kernel

    def test_functional_only_preserves_calls_and_assertions_without_timing_pass(self):
        status, report, kernel = self.run_pytest('''
from fake_costmodel_runtime import kernel

def test_kernel():
    kernel.run(3, grid=(1,), warmup=False)
    assert kernel.actual[-1] == 3

def test_host():
    assert 1 + 1 == 2

def test_wrong_result():
    assert False, 'incorrect result'
''', functional_only=True)
        self.assertEqual(status, 1)
        self.assertEqual(report['validation_scope'], 'functional')
        self.assertIsNone(report['maximum_absolute_relative_error'])
        self.assertEqual(report['counts']['PASS'], 2)
        self.assertEqual(report['counts']['FAIL'], 1)
        self.assertTrue(all(row['timing_pass'] is None for row in report['results']))
        self.assertEqual(kernel.actual, [3])
        self.assertEqual(kernel.compiles, [])

    def test_real_pytest_and_junit_fail_on_timing_missing_data_or_unsupported_tests(self):
        status, report, kernel = self.run_pytest('''
import pytest
from fake_costmodel_runtime import kernel

@pytest.mark.parametrize('n', [1, 2])
def test_correct(n):
    kernel.run(n, grid=(1,), warmup=False)
    assert kernel.actual[-1] == n

def test_timing_error():
    kernel.run(3, grid=(1,), warmup=False)
    assert kernel.actual[-1] == 3

def test_missing_route():
    kernel.run(4, grid=(1,), warmup=False)
    assert kernel.actual[-1] == 4

def test_precision_error():
    kernel.run(5, grid=(1,), warmup=False)
    assert False, 'wrong result'

def test_host_only():
    assert 1 + 1 == 2

def test_expected_exception():
    with pytest.raises(ValueError):
        raise ValueError('expected')

@pytest.mark.skip(reason='no device support')
def test_skipped():
    assert False
''')
        self.assertEqual(status, 1)
        self.assertEqual(report['counts'], {'PASS': 2, 'FAIL': 3, 'UNSUPPORTED': 2, 'SKIP': 1, 'PENDING': 0})
        self.assertEqual(report['status'], 'FAIL')
        rows = {r['nodeid'].split('::')[-1]: r for r in report['results']}
        self.assertTrue(rows['test_timing_error']['precision_pass'])
        self.assertFalse(rows['test_timing_error']['timing_pass'])
        self.assertFalse(rows['test_precision_error']['precision_pass'])
        self.assertFalse(rows['test_host_only']['timing_pass'])
        self.assertEqual(len(ET.parse(self.root / 'pytest.xml').findall('.//failure')), 5)
        self.assertEqual(kernel.actual, [1, 2, 3, 4, 5])

    def test_valid_test_is_audited_pass_and_environment_restored(self):
        with patch.dict(os.environ, {'TRITON_CACHE_DIR': 'existing-cache',
                                    'TRITON_ASCEND_AUTO_SIMT_PROFILE': 'existing-profile'}):
            status, report, kernel = self.run_pytest('''
from fake_costmodel_runtime import kernel

def test_correct():
    kernel.run(1, grid=(1,), warmup=False)
    assert kernel.actual[-1] == 1
''')
            self.assertEqual(os.environ['TRITON_CACHE_DIR'], 'existing-cache')
            self.assertEqual(os.environ['TRITON_ASCEND_AUTO_SIMT_PROFILE'], 'existing-profile')
        self.assertEqual(status, 0)
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['results'][0]['timing_pass'])

    def test_teardown_kernel_failure_cannot_leave_a_pass_result(self):
        status, report, _ = self.run_pytest('''
import pytest
from fake_costmodel_runtime import kernel

@pytest.fixture
def work():
    yield
    kernel.run(3, grid=(1,), warmup=False)

def test_correct(work):
    kernel.run(1, grid=(1,), warmup=False)
    assert kernel.actual[-1] == 1
''')
        self.assertEqual(status, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['counts']['PASS'], 0)
        self.assertIn('teardown_kernel_timing_audit_failed', report['results'][0]['failure_reasons'])

    def test_real_cli_collection_is_not_a_timing_pass_and_output_cannot_be_reused(self):
        tests = self.root / 'tests'
        tests.mkdir()
        (tests / 'test_collect.py').write_text('def test_collect():\n    assert True\n')
        output = self.root / 'collected'
        command = [sys.executable, '-m', 'pytest', str(tests), '--confcutdir', str(tests),
                   '-p', 'costmodel_tests.pytest_costmodel', '--costmodel-output', str(output),
                   '--costmodel-profile', str(self.profile), '--collect-only', '-q', '-o', 'addopts=']
        env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
        env.pop('TRITON_ASCEND_COMPILE_MODE', None)
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'NOT_RUN')
        self.assertEqual(report['counts']['PASS'], 0)
        self.assertEqual(report['collected_case_count'], 1)
        self.assertTrue((output / 'collected_cases.json').is_file())
        again = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(again.returncode, 4)
        self.assertIn('new or empty', again.stderr)

    def test_launcher_selects_pytest_entry_from_explicit_all_manifest(self):
        command = [sys.executable, str(ROOT / 'costmodel_tests/run.py'),
                   '--manifest', 'cases/ci/manifest_all.json',
                   '--case', 'pytest_test_01_vector_add', '--dry-run']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('UNVERIFIED', result.stdout)
        arguments = json.loads(result.stdout.splitlines()[-1].split(': ', 1)[1])
        self.assertEqual(arguments[arguments.index('--case') + 1], 'pytest_test_01_vector_add')
        self.assertTrue(arguments[arguments.index('--manifest') + 1].endswith('manifest_all.json'))

    def test_launcher_requires_batch_audit_even_when_process_exit_is_zero(self):
        spec = importlib.util.spec_from_file_location('adapted_launcher', ROOT / 'costmodel_tests/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for audit_status, expected in [('PASS', 0), ('FAIL', 1), ('MISSING', 1), ('NOT_RUN', 1)]:
            output = self.root / audit_status

            def fake_run(command, **kwargs):
                self.assertIn('run_generalization_batch.py', command[1])
                self.assertIn('--profile', command)
                if audit_status != 'MISSING':
                    output.mkdir()
                    (output / 'batch_summary.json').write_text(json.dumps({'status': audit_status}))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, 'argv', ['run.py', '--profile', str(self.profile), '--output-root', str(output)]), \
                    patch.object(module.subprocess, 'run', side_effect=fake_run), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    module.main()
            self.assertEqual(stopped.exception.code, expected)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['status'],
                             'PASS' if audit_status == 'PASS' else 'FAIL')


if __name__ == '__main__':
    unittest.main()
