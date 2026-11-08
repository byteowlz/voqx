import unittest
import numpy as np
from export_step import ATTRS, COUNTERS, fixture


class ExportControlTests(unittest.TestCase):
    def test_fixture_is_deterministic_finite_and_frame_aligned(self):
        audio = fixture(240, 1280)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (240 * 1280,))
        self.assertTrue(np.isfinite(audio).all())
        np.testing.assert_array_equal(audio, fixture(240, 1280))

    def test_fixture_has_silence_and_different_amplitude_regimes(self):
        audio = fixture(240, 1280).reshape(4, -1)
        self.assertTrue(np.any(audio[0]))
        self.assertTrue(np.all(audio[1] == 0))
        self.assertGreater(float(audio[2].std()), 0.05)
        self.assertGreater(float(audio[3].mean()), 0.07)

    def test_frontend_state_includes_both_dynamic_counters(self):
        self.assertEqual(len(set(ATTRS)), len(ATTRS))
        self.assertEqual(COUNTERS, {'audio_count', 'cms_count'})
        self.assertTrue(COUNTERS.issubset(ATTRS))


if __name__ == '__main__':
    unittest.main()
