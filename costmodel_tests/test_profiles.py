"""Regression checks for moving profiles through batch/case/merged outputs."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from costmodel_tests.profiles import snapshot_profile


class ProfileTests(unittest.TestCase):
    def test_nested_snapshots_keep_dependency_and_can_be_moved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "profiles/simd_simt/selection.json"
            source.parent.mkdir(parents=True)
            dependency = root / "profiles/microbench/hardware.json"
            dependency.parent.mkdir()
            dependency.write_text('{"measurements": {"clock": 1000}}')
            original = {"target": "test", "microbenchmark_profile": "../microbench/hardware.json"}
            source.write_text(json.dumps(original))
            outputs = [root / directory / "costmodel_profile.json"
                       for directory in ("batch", "case", "merged")]
            previous = source
            for output in outputs:
                output.parent.mkdir()
                snapshot_profile(previous, output)
                reference = json.loads(output.read_text())["microbenchmark_profile"]
                self.assertEqual((output.parent / reference).read_bytes(), dependency.read_bytes())
                previous = output
            self.assertEqual(outputs[0].read_bytes(), outputs[-1].read_bytes())
            self.assertEqual(json.loads(source.read_text()), original)
            moved = root / "moved"
            shutil.copytree(outputs[-1].parent, moved)
            shutil.rmtree(outputs[-1].parent)
            reference = json.loads((moved / "costmodel_profile.json").read_text())["microbenchmark_profile"]
            self.assertEqual((moved / reference).read_bytes(), dependency.read_bytes())

    def test_absolute_reference_and_missing_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "hardware.json"
            dependency.write_text('{}')
            source, target = root / "selection.json", root / "copy.json"
            source.write_text(json.dumps({"microbenchmark_profile": str(dependency)}))
            snapshot_profile(source, target)
            reference = json.loads(target.read_text())["microbenchmark_profile"]
            self.assertFalse(Path(reference).is_absolute())
            self.assertEqual((target.parent / reference).read_text(), '{}')
            dependency.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "microbenchmark_profile"):
                snapshot_profile(source, target)

    def test_profile_without_dependency_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "selection.json", root / "copy.json"
            source.write_bytes(b'{"kernel_timing": {}}\n')
            snapshot_profile(source, target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
