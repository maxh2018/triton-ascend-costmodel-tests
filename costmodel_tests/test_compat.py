import types
import unittest
from unittest.mock import patch

from costmodel_tests.compat import is_compile_on_910_95


class CompatTests(unittest.TestCase):
    def call(self, predicate, arch="current"):
        module = types.ModuleType("triton.backends.ascend.utils")
        module.is_compile_on_910_95 = predicate
        with patch.dict("sys.modules", {module.__name__: module}):
            return is_compile_on_910_95(arch)

    def test_current_and_legacy_predicate_signatures_preserve_results(self):
        self.assertTrue(self.call(lambda arch: arch == "current"))
        self.assertFalse(self.call(lambda arch: arch == "current", "other"))
        self.assertTrue(self.call(lambda: True))
        self.assertFalse(self.call(lambda: False))

    def test_errors_inside_predicate_are_not_hidden(self):
        def broken(arch):
            raise TypeError("backend error")
        with self.assertRaisesRegex(TypeError, "backend error"):
            self.call(broken)
