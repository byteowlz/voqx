"""Analytic pooling controls; importing the probe does not require Torch or a model."""
import unittest

import numpy as np

from fastdiar_probe import deny_network, pool_frames


class FastdiarPoolTests(unittest.TestCase):
    def test_pooling_modes_remain_distinct_and_unit_length(self):
        frames = np.zeros((2, 192), dtype=np.float32)
        frames[0, 0] = 1; frames[1, 1] = 1
        pooled = pool_frames(frames)
        self.assertAlmostEqual(float(np.linalg.norm(pooled['mean'])), 1, places=6)
        np.testing.assert_array_equal(pooled['last'], frames[-1])
        self.assertNotEqual(float(pooled['mean'][0]), float(pooled['last'][0]))

    def test_cancelling_or_nonfinite_pooling_abstains(self):
        frames = np.zeros((2, 192), dtype=np.float32)
        frames[0, 0] = 1; frames[1, 0] = -1
        self.assertIsNone(pool_frames(frames)['mean'])
        self.assertIsNotNone(pool_frames(frames)['last'])
        with np.errstate(invalid='ignore'):
            pooled = pool_frames(np.full((2, 192), np.nan))
        self.assertIsNone(pooled['mean']); self.assertIsNone(pooled['last'])

    def test_inference_network_attempts_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'Network disabled'):
            deny_network('example.invalid', 443)


if __name__ == '__main__':
    unittest.main()
