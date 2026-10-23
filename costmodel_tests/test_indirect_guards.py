"""CPU-only checks for portable validation-derived indirect guards."""
import ast
import json
from pathlib import Path
import unittest

from costmodel_tests.manifests import load_manifest

ROOT = Path(__file__).resolve().parents[1]
PERSONAL = ROOT / 'cases/personal'
DATA = PERSONAL / 'maxh2018/indirect_memory_guards.json'
DRIVER = PERSONAL / 'maxh2018/bench_indirect_memory_guards.py'


class IndirectGuardTests(unittest.TestCase):

    def test_all_29_geometries_registered_without_withdrawn_case(self):
        cases = json.loads(DATA.read_text())['cases']
        names = {c['guard_id'] for c in cases}
        self.assertEqual(len(cases), 29)
        self.assertEqual(len(names), 29)
        self.assertNotIn('simd_store_loaded_bool', names)
        manifest = load_manifest(PERSONAL / 'manifest.json')['cases']
        added = [c for c in manifest if Path(c['driver']) == DRIVER]
        self.assertEqual({c['name']
                          for c in added}, {'maxh2018_' + n
                                            for n in names})
        self.assertTrue(
            any(c['name'] == 'maxh2018_indirect_load_i16_n16_w2'
                for c in manifest))
        for c in added:
            self.assertEqual(c['runner'], 'benchmark')
            self.assertEqual(
                c['arguments'],
                ['--guard-id', c['name'].removeprefix('maxh2018_')])

    def test_geometry_and_source_provenance(self):
        for c in json.loads(DATA.read_text())['cases']:
            product = 1
            for axis in c['shape']:
                self.assertGreater(axis, 0)
                product *= axis
            self.assertEqual(product, c['E'])
            self.assertIn(c['F'], (1, 2, 4))
            self.assertIn(c['bits'], (8, 16, 32, 64))
            self.assertIn(c['index_bits'], (32, 64))
            self.assertIn(c['kind'], ('none', 'prefix', 'loaded'))
            self.assertTrue(0 <= c['K'] <= c['E'])
            self.assertTrue(c['id'] and c['validation_source'])

    def test_driver_uses_public_benchmark_interface(self):
        tree = ast.parse(DRIVER.read_text())
        self.assertEqual(
            [n.name for n in tree.body if isinstance(n, ast.FunctionDef)],
            ['indirect_guard', 'main'])
        source = DRIVER.read_text()
        for option in ('--route', '--factor', '--report', '--profile-dir',
                       '--warmup', '--repeats'):
            self.assertIn(option, source)
        for forbidden in ('subprocess', 'custom-aiv-number',
                          'custom-aic-number', 'sys.path'):
            self.assertNotIn(forbidden, source)
        self.assertNotIn('triton.reinterpret', source)


if __name__ == '__main__':
    unittest.main()
