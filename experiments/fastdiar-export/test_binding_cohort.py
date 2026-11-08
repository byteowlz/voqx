import unittest
from itertools import product
import numpy as np
from binding_cohort import SCORE_MARGIN, pool_frames, encode_frames, decode_frames, count_at, compare_bindings


class BindingControls(unittest.TestCase):
    def test_pooling_and_degenerate_abstention(self):
        frames = np.random.default_rng(17).normal(size=(7, 192)).astype(np.float32)
        value = pool_frames(frames)['mean']
        self.assertEqual(value.dtype, np.float64)
        expected = frames.astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(value, expected / np.linalg.norm(expected), rtol=0, atol=0)
        self.assertEqual(pool_frames(np.zeros((2, 192))), {'mean': None, 'last': None})

    def test_ipc_roundtrip_shape_finiteness(self):
        values = np.arange(384, dtype=np.float32).reshape(2, 192)
        np.testing.assert_array_equal(decode_frames(encode_frames(values), values.shape), values)
        with self.assertRaises(ValueError):
            decode_frames(encode_frames(values), (1, 192))
        with self.assertRaises(ValueError):
            decode_frames(encode_frames(np.full((1, 192), np.nan)), (1, 192))

    def test_abstentions_remain_in_denominators(self):
        rows = [{'id': 'a', 'target': True, 'role': 'test', 'score': None},
                {'id': 'b', 'target': False, 'role': 'test', 'score': 0.1}]
        result = count_at(rows, 0.2)['test']
        self.assertEqual(result, {'positive_n': 1, 'negative_n': 1, 'positive_accept': 0, 'negative_accept': 0, 'abstain': 1})

    def test_float_step_threshold_is_not_binding_portable_but_fixed_margin_is(self):
        def vector(score):
            value = np.zeros(192); value[0] = score; value[1] = np.sqrt(1 - score * score)
            return value
        ids = ['ref' + str(i) for i in range(7)]
        lookup = {i: {'speaker': speaker, 'role': role} for i, speaker, role in [
            ('pc', 'owner', 'calibration'), ('nc', 'other', 'calibration'),
            ('pt', 'owner', 'test'), ('nt', 'other', 'test')]}
        scenario = {'enrollment': {'owner': ids}, 'queries': list(lookup)}
        banks = {}
        for name in ('torch80', 'native_ort', 'browser_wasm'):
            refs = {(i, align, pool): vector(1) for i, align, pool in product(ids, ('prefix', 'onset'), ('mean', 'last'))}
            queries = {(i, seconds, align, pool): vector({'pc': .9, 'nc': .5 + (1e-5 if name != 'torch80' else 0), 'pt': .8, 'nt': .49}[i])
                       for i, seconds, align, pool in product(lookup, (1., 1.5, 2.6), ('prefix', 'onset'), ('mean', 'last'))}
            banks[name] = (refs, queries)
        result = compare_bindings(scenario, lookup, banks)
        control = result['fixed_torch_bank_and_threshold'][0]['bindings']['native_ort']
        self.assertEqual(SCORE_MARGIN, 0.001)
        self.assertEqual(control['raw']['calibration']['negative_accept'], 1)
        self.assertEqual(control['guarded']['calibration']['negative_accept'], 0)
        self.assertEqual(control['guarded_disagreements'], {'calibration': 0, 'test': 0})


if __name__ == '__main__':
    unittest.main()
