import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import wave

import numpy as np

from audio import MASK_FRAMES, RATE, WINDOW, activity, crop, read_wav, tensors, unit
from metrics import auc, check_split, summarize
from probe import embed
from phrase_probe import compare


class AudioTests(unittest.TestCase):
    def test_uniform_reproduces_full_window_contract(self):
        pcm = np.full(RATE, 0.1, np.float32)
        result = tensors(pcm, "uniform")
        self.assertEqual(result["waveform"].shape, (1, 1, WINDOW))
        np.testing.assert_array_equal(result["weights"], np.ones((1, MASK_FRAMES)))
        np.testing.assert_array_equal(result["waveform"][0, 0, :RATE], pcm)
        self.assertEqual(np.count_nonzero(result["waveform"][0, 0, RATE:]), 0)

    def test_valid_mask_ignores_padding_not_real_samples(self):
        pcm = np.ones(RATE, np.float32)
        weights = tensors(pcm, "valid")["weights"][0]
        self.assertEqual(int(weights.sum()), 59)
        np.testing.assert_array_equal(weights[59:], np.zeros(MASK_FRAMES - 59))
        np.testing.assert_array_equal(tensors(np.ones(WINDOW), "valid")["weights"],
                                      tensors(np.ones(WINDOW), "uniform")["weights"])

    def test_onset_crop_is_not_file_start(self):
        pcm = np.concatenate([np.zeros(RATE), np.full(RATE * 2, 0.1)]).astype(np.float32)
        prefix, a = crop(pcm, 1, "prefix")
        onset, b = crop(pcm, 1, "onset")
        self.assertEqual(a, 0)
        self.assertFalse(np.any(prefix))
        self.assertGreater(b, RATE * 0.8)
        self.assertTrue(np.any(onset))
        self.assertEqual(len(onset), RATE)

    def test_activity_excludes_silence_without_assigning_identity(self):
        pcm = np.zeros(RATE * 2, np.float32)
        pcm[RATE // 2:RATE] = 0.1
        mask = tensors(pcm, "activity")["weights"][0]
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), tensors(pcm, "valid")["weights"].sum())
        self.assertFalse(np.any(activity(np.zeros(RATE))))
        self.assertFalse(np.any(tensors(np.zeros(RATE), "activity")["weights"]))

    def test_input_errors_fail_not_silently_coerce(self):
        for audio in [np.zeros(0), np.zeros((2, 4)), np.asarray([float("nan")])]:
            with self.assertRaises(ValueError):
                tensors(audio, "valid")
        with self.assertRaises(ValueError):
            unit(np.zeros(256))
        with self.assertRaises(ValueError):
            unit(np.ones(96))
        with self.assertRaises(ValueError):
            crop(np.ones(RATE), -1, "prefix")
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.wav"
            with wave.open(str(p), "wb") as out:
                out.setparams((1, 2, 24000, 0, "NONE", "NONE"))
                out.writeframes(np.zeros(24000, dtype="<i2").tobytes())
            with self.assertRaises(ValueError):
                read_wav(p)


class ModelFailureTests(unittest.TestCase):
    def test_sparse_mask_nan_abstains_instead_of_becoming_negative(self):
        class InvalidSession:
            def run(self, names, inputs):
                return [np.full((1, 256), np.nan)]
        vector, elapsed, reason = embed(InvalidSession(), np.full(RATE, 0.1), "activity")
        self.assertIsNone(vector)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(reason, "invalid_embedding")


class PhraseTests(unittest.TestCase):
    def test_phrase_comparison_holds_language_and_reference_count(self):
        records = [{"id": i, "role": role, "speaker": "owner", "sha256": i}
                   for i, role in [("same", "enroll"), ("other", "enroll"), ("foreign", "enroll"), ("q", "test")]]
        phrases = {"same": {"language": "de", "phrase": "hey"}, "q": {"language": "de", "phrase": "hey"},
                   "other": {"language": "de", "phrase": "bare"}, "foreign": {"language": "en", "phrase": "hey"}}
        condition = (1.0, "onset", "valid")
        vectors = {(i, *condition): np.asarray(v) for i, v in
                   [("same", [1, 0]), ("other", [0, 1]), ("foreign", [-1, 0]), ("q", [1, 0])]}
        result = compare({"records": records}, vectors, phrases)[0]
        self.assertEqual(result["paired_queries"], 1)
        self.assertEqual(result["median_paired_delta"], 1)
        self.assertEqual(result["same_wins"], 1)


class MetricTests(unittest.TestCase):
    def test_auc_ties_and_empty(self):
        self.assertEqual(auc([1, 1], [1, 1]), 0.5)
        self.assertEqual(auc([1], [0]), 1)
        self.assertEqual(auc([0], [1]), 0)
        self.assertIsNone(auc([], [1]))

    def test_threshold_uses_calibration_and_abstentions_stay(self):
        cal = [{"target": False, "score": 0.5}]
        test = [{"target": True, "score": 0.8}, {"target": True, "score": None},
                {"target": False, "score": 0.9}, {"target": False, "score": None}]
        result = summarize(test, cal)
        self.assertEqual(result["positive_n"], 2)
        self.assertEqual(result["positive_abstain"], 1)
        self.assertEqual(result["negative_abstain"], 1)
        self.assertEqual(result["test_negative_accept"], 1)
        self.assertGreater(result["threshold_calibration_zero_fa"], 0.5)
        self.assertLess(result["threshold_calibration_zero_fa"], 0.9)
        self.assertIsNone(summarize(test, [])["test_negative_accept"])

    def test_source_cannot_cross_enrollment_query(self):
        records = [{"id": "a", "sha256": "a", "source_sha256": "same"},
                   {"id": "b", "sha256": "b", "source_sha256": "same"}]
        with self.assertRaises(ValueError):
            check_split(records, ["a"], ["b"])
        records[1]["source_sha256"] = "different"
        check_split(records, ["a"], ["b"])


if __name__ == "__main__":
    unittest.main()
