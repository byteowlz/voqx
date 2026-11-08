"""Bank selection tests use analytic vectors, never private speaker fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bank_probe import choose_bank, digest, evaluate, score_rows, vector, verify_inputs


def row(identifier, target, role, value):
    return {'id': identifier, 'target': target, 'role': role, 'vector': np.asarray(value, dtype=float) if value is not None else None}


class BankProbeTests(unittest.TestCase):
    def test_test_queries_never_select_bank_or_threshold(self):
        references = [{'id': 'a', 'vector': np.array([1., 0.])}, {'id': 'b', 'vector': np.array([0., 1.])}]
        calibration = [row('positive', True, 'calibration', [1, 0]), row('negative', False, 'calibration', [0, 1])]
        first = choose_bank(references, calibration + [row('test', True, 'test', [1, 0])], 1, 'nearest')
        second = choose_bank(references, calibration + [row('test', True, 'test', [0, 1])], 1, 'nearest')
        self.assertEqual(first['selected_references'], ('a',))
        self.assertEqual(first['selected_references'], second['selected_references'])
        self.assertEqual(first['threshold_calibration_zero_fa'], second['threshold_calibration_zero_fa'])
        self.assertEqual(first['test_positive_accept'], 1)
        self.assertEqual(second['test_positive_accept'], 0)

    def test_more_references_can_raise_impostor_scores(self):
        queries = [row('target', True, 'test', [0, 1]), row('impostor', False, 'test', [0, 1])]
        single = score_rows(queries, [np.array([1., 0.])], 'nearest')
        bank = score_rows(queries, [np.array([1., 0.]), np.array([0., 1.])], 'nearest')
        self.assertEqual([r['score'] for r in single], [0, 0])
        self.assertEqual([r['score'] for r in bank], [1, 1])

    def test_abstentions_stay_in_denominators(self):
        references = [{'id': 'a', 'vector': np.array([1., 0.])}]
        queries = [row('positive', True, 'calibration', [1, 0]), row('negative', False, 'calibration', [0, 1]),
                   row('test-positive', True, 'test', None), row('test-negative', False, 'test', None)]
        result = choose_bank(references, queries, 1, 'nearest')
        self.assertEqual((result['positive_n'], result['negative_n']), (1, 1))
        self.assertEqual((result['positive_abstain'], result['negative_abstain']), (1, 1))
        self.assertEqual((result['test_positive_accept'], result['test_negative_accept']), (0, 0))

    def test_no_calibration_negatives_cannot_create_a_threshold(self):
        with self.assertRaises(ValueError):
            choose_bank([{'id': 'a', 'vector': np.array([1., 0.])}], [row('cal', True, 'calibration', [1, 0])], 1, 'nearest')

    def test_vector_validation_and_degenerate_centroid(self):
        value = np.zeros(256); value[0] = 1
        np.testing.assert_array_equal(vector(value), value)
        self.assertIsNone(vector(None))
        for bad in (np.zeros(256), np.ones(256), np.ones(192), np.full(256, np.nan)):
            with self.assertRaises(ValueError):
                vector(bad)
        with self.assertRaises(ValueError):
            score_rows([], [np.array([1., 0.]), np.array([-1., 0.])], 'centroid')

    def test_reference_roles_cannot_be_queries(self):
        config = {'scenarios': [{'name': 'owner-mixed', 'enrollment': {'owner': ['a']}, 'queries': []}],
                  'records': [{'id': 'a', 'speaker': 'owner', 'role': 'test', 'sha256': 'different'}]}
        with self.assertRaisesRegex(ValueError, 'entered the reference bank'):
            evaluate(config, {}, {})

    def test_registered_input_integrity_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for name in ('config.json', 'run-2/vectors.jsonl', 'full-enrollment/enrollment-vectors.json'):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('[]')
                entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
            manifest = root / 'manifest.jsonl'; manifest.write_text('\n'.join(json.dumps(r) for r in entries))
            expected = digest(manifest)
            self.assertEqual(len(verify_inputs(root, expected)), 3)
            (root / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_inputs(root, expected)
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                verify_inputs(root, 'incorrect')


if __name__ == '__main__':
    unittest.main()
