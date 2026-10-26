"""CPU regressions using a fake installed TA, never a validation bypass."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from costmodel_tests.profiles import CONTEXT, CONTEXT_ID, digest, prepare_profile, read_json


def write_test_profile(path, extra=None):
    """A minimal profile accepted by the fake TA's own versioned schemas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 19,
        "profile_version": "test-profile-v1",
        "target": "Ascend950PR/dav-c310",
        "compatible_targets": ["Ascend950PR_*"],
        "microbenchmark_profile": "hardware.json",
        **(extra or {})
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    dependency = (path.parent / data["microbenchmark_profile"]).resolve()
    dependency.parent.mkdir(parents=True, exist_ok=True)
    measurements = read_json(dependency) if dependency.is_file() else {}
    dependency.write_text(json.dumps({
        "schema_version": 1,
        "profile_version": "hardware-v1",
        "target": data["target"],
        "measurements": {},
        **measurements
    }),
                          encoding="utf-8")
    return path


def fake_ta(root):
    package = Path(root) / "fake_install/triton"
    for name in ("backends/ascend", "runtime", "_C"):
        (package / name).mkdir(parents=True, exist_ok=True)
    files = {
        "__init__.py":
        "__version__ = 'test-triton'\n",
        "backends/__init__.py":
        "",
        "backends/ascend/__init__.py":
        "__version__ = 'test-ta'\n",
        "backends/ascend/compiler.py":
        "# fake installed compiler\n",
        "_C/__init__.py":
        "",
        "_C/libtriton.py":
        "# fake native identity\n",
        "runtime/__init__.py":
        "from . import driver\n",
        "runtime/driver.py":
        """import json, os
from types import SimpleNamespace
targets=json.loads(os.environ.get('PROFILE_TEST_TARGETS', '{}'))
device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','').split(',')[0]
active=SimpleNamespace(get_current_target=lambda: SimpleNamespace(
    backend='npu', arch=targets.get(device, 'Ascend950PR_9579')))
"""
    }
    for name, content in files.items():
        (package / name).write_text(content, encoding="utf-8")
    assets = package / "backends/ascend/costmodel_profiles"
    profile = write_test_profile(
        assets / "simd_simt/current.json",
        {"microbenchmark_profile": "../microbench/hardware.json"})
    selection_schema = {
        "type":
        "object",
        "required": [
            "schema_version", "profile_version", "target",
            "compatible_targets", "microbenchmark_profile"
        ],
        "properties": {
            "schema_version": {
                "const": 19
            }
        }
    }
    hardware_schema = {
        "type": "object",
        "required":
        ["schema_version", "profile_version", "target", "measurements"],
        "properties": {
            "schema_version": {
                "const": 1
            }
        }
    }
    for path, schema in ((assets / "simd_simt/simd_simt_profile_schema.json",
                          selection_schema),
                         (assets /
                          "microbench/microbenchmark_profile_schema.json",
                          hardware_schema)):
        path.write_text(json.dumps(schema), encoding="utf-8")
    return {
        "PYTHONPATH":
        str(package.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "ASCEND_RT_VISIBLE_DEVICES":
        "1",
        "TRITON_ASCEND_AUTO_SIMT_PROFILE":
        "",
        CONTEXT:
        "",
        CONTEXT_ID:
        ""
    }, profile


class ResolutionTests(unittest.TestCase):

    def test_profiles_bundled_beside_native_extension_are_discovered(self):
        assets = self.installed.parent.parent
        native_assets = self.installed.parents[
            4] / "_C/ascend/costmodel_profiles"
        native_assets.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(assets), str(native_assets))
        self.installed = native_assets / "simd_simt/current.json"
        prepare_profile(None, self.output, self.env)
        metadata = read_json(self.output.parent / "profile_metadata.json")
        self.assertEqual(metadata["source_path"], str(self.installed))
        self.assertEqual(metadata["source_kind"], "installed_ta")
        self.assertIsNone(metadata["runtimes"][0]["ta_distribution_version"])

    def test_snapshot_keeps_validated_bytes_if_source_changes(self):
        from costmodel_tests import profiles
        resolve = profiles.resolve_profile
        original_hash = digest(self.installed)

        def resolve_then_update(*args, **kwargs):
            plan = resolve(*args, **kwargs)
            self.installed.write_text('{"updated_during_run": true}')
            return plan

        with patch.object(profiles,
                          "resolve_profile",
                          side_effect=resolve_then_update):
            prepare_profile(None, self.output, self.env)
        metadata = read_json(self.output.parent / "profile_metadata.json")
        self.assertEqual(metadata["source_sha256"], original_hash)
        self.assertEqual(
            read_json(self.output)["profile_version"], "test-profile-v1")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        environment, self.installed = fake_ta(self.root)
        self.env = {**os.environ, **environment}
        self.output = self.root / "results/costmodel_profile.json"

    def test_installed_default_and_complete_hash_manifest(self):
        result = prepare_profile(None, self.output, self.env)
        self.assertEqual(result, self.output)
        metadata = read_json(result.parent / "profile_metadata.json")
        self.assertEqual(metadata["source_kind"], "installed_ta")
        self.assertEqual(metadata["runtimes"][0]["ta_version"], "test-ta")
        self.assertEqual(metadata["source_sha256"], digest(self.installed))
        self.assertEqual(
            (result.parent / "profile_sources/selection.json").read_bytes(),
            self.installed.read_bytes())
        for name, expected in metadata["artifacts"].items():
            self.assertEqual(digest(result.parent / name), expected)
        self.assertEqual(len(metadata["artifacts"]), 5)

    def test_explicit_overrides_environment_and_preserves_baseline(self):
        baseline = write_test_profile(self.root / "baseline.json",
                                      {"profile_version": "old-baseline"})
        self.env["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(self.root /
                                                          "missing.json")
        prepare_profile(baseline, self.output, self.env)
        metadata = read_json(self.output.parent / "profile_metadata.json")
        self.assertEqual(metadata["source_kind"], "explicit")
        self.assertEqual(metadata["profile_version"], "old-baseline")

    def test_environment_is_an_explicit_override(self):
        self.env["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(self.installed)
        prepare_profile(None, self.output, self.env)
        self.assertEqual(
            read_json(self.output.parent /
                      "profile_metadata.json")["source_kind"], "environment")

    def test_missing_installed_assets_never_use_repository_copy(self):
        shutil.rmtree(self.installed.parent.parent)
        with self.assertRaisesRegex(ValueError, "does not ship"):
            prepare_profile(None, self.output, self.env)
        self.assertFalse(self.output.parent.exists())

    def test_incompatible_schema_is_rejected_before_execution(self):
        data = read_json(self.installed)
        data["schema_version"] = 999
        self.installed.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "schema"):
            prepare_profile(self.installed, self.output, self.env)
        self.assertFalse(self.output.parent.exists())

    def test_every_requested_device_must_match(self):
        self.env["PROFILE_TEST_TARGETS"] = json.dumps(
            {"2": "Ascend950DT_9999"})
        with self.assertRaisesRegex(ValueError, "Ascend950DT"):
            prepare_profile(self.installed,
                            self.output,
                            self.env,
                            devices=["1", "2"])
        self.assertFalse(self.output.parent.exists())

    def test_dependency_target_and_measurement_references_are_checked(self):
        dependency = self.installed.parent.parent / "microbench/hardware.json"
        data = read_json(dependency)
        data["target"] = "different"
        dependency.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "different targets"):
            prepare_profile(self.installed, self.output, self.env)
        data["target"] = "Ascend950PR/dav-c310"
        dependency.write_text(json.dumps(data))
        selection = read_json(self.installed)
        selection["ops"] = {"throughput_measurement": "unknown.id"}
        self.installed.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, "unknown.id"):
            prepare_profile(self.installed, self.output, self.env)

    def test_ambiguous_defaults_require_explicit_selection(self):
        shutil.copy2(self.installed, self.installed.with_name("other.json"))
        with self.assertRaisesRegex(ValueError, "found 2"):
            prepare_profile(None, self.output, self.env)

    def test_worker_inherits_frozen_inputs_and_rejects_tampering(self):
        prepare_profile(None, self.output, self.env)
        child = self.root / "child/costmodel_profile.json"
        # Import resolution inside the worker is the same fake installed TA.
        fake_root = str(self.installed.parents[4])
        with patch.dict(
                sys.modules,
            {
                "triton":
                type("TA", (), {"__file__": fake_root + "/__init__.py"}),
                "triton.backends.ascend":
                type("Ascend", (), {
                    "__file__": str(self.installed.parents[2] / "__init__.py")
                })
            }):
            prepare_profile(self.output, child, self.env)
            self.assertEqual(child.read_bytes(), self.output.read_bytes())
            hardware = child.parent / "profile_dependencies/hardware.json"
            hardware.write_text('{}')
            with self.assertRaisesRegex(ValueError, "changed after preflight"):
                prepare_profile(child,
                                self.root / "next/costmodel_profile.json",
                                self.env)

    def test_functional_only_requires_neither_ta_nor_profile(self):
        with patch("costmodel_tests.profiles.probe_runtime",
                   side_effect=AssertionError("must not load TA")):
            self.assertIsNone(
                prepare_profile(None, self.output, {}, mode="functional"))
        self.assertFalse(self.output.exists())
        self.assertFalse(
            read_json(self.output.parent /
                      "profile_metadata.json")["profile_used_by_costmodel"])

    def test_repository_baseline_does_not_bypass_installed_schema(self):
        assets = Path(__file__).resolve(
        ).parents[1] / "cases/ci/third_party/ascend/costmodel/profiles"
        for folder, name in (("simd_simt", "simd_simt_profile_schema.json"),
                             ("microbench",
                              "microbenchmark_profile_schema.json")):
            shutil.copy2(assets / folder / name,
                         self.installed.parent.parent / folder / name)
        baseline = assets / "simd_simt/ascend_950pr_simd_simt_v1.json"
        # The archived measurement data contains a source_kind absent from this
        # archived schema. Preserve and report that incompatibility honestly.
        with self.assertRaisesRegex(ValueError, "derived_calibration"):
            prepare_profile(baseline, self.output, self.env)
        self.assertFalse(self.output.parent.exists())

    def test_nonfinite_parameters_are_rejected(self):
        data = read_json(self.installed)
        data["cost"] = float("nan")
        self.installed.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            prepare_profile(self.installed, self.output, self.env)

    def test_default_profile_flows_through_real_launcher_and_pytest_worker(
            self):
        tests = self.root / "tests"
        tests.mkdir()
        (tests / "conftest.py").write_text('''
from costmodel_tests.test_pytest_costmodel import FakeBackend
def pytest_sessionstart(session):
    session.config._costmodel_adapter.backend_factory = FakeBackend
''')
        (tests / "test_kernel.py").write_text('''
from costmodel_tests.test_pytest_costmodel import FakeJit
def test_kernel():
    kernel = FakeJit()
    kernel.run(1, grid=(1,), warmup=False)
    assert kernel.actual == [1]
''')
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps({
                "cases": [{
                    "name": "kernel",
                    "runner": "pytest",
                    "driver": "tests/test_kernel.py"
                }]
            }))
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable, "costmodel_tests/run.py", "--manifest",
                str(manifest), "--retries", "0", "--output-root",
                str(self.output.parent)
            ],
            cwd=root,
            env={
                **self.env, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "TRITON_ASCEND_COMPILE_MODE": ""
            },
            capture_output=True,
            text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = read_json(self.output.parent / "profile_metadata.json")
        child = read_json(
            self.output.parent /
            "cases/kernel/attempt_01/pytest_costmodel/profile_metadata.json")
        self.assertEqual(metadata["source_kind"], "installed_ta")
        self.assertEqual(metadata["source_sha256"], child["source_sha256"])
        self.assertIn("test-ta",
                      (self.output.parent / "report.md").read_text())
