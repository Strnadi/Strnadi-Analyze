import asyncio
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import UploadFile

import runner


class AudioWindowTests(unittest.TestCase):
    def check_windows(self, count, bounds):
        # Eight samples per second makes the half-second hop exactly four samples.
        audio = np.arange(count, dtype=np.float32) + 1
        windows = list(runner.chunk_audio(audio, target_sr=8))
        self.assertEqual([(start * 8, end * 8) for _, start, end in windows], bounds)
        for (window, _, _), (start, end) in zip(windows, bounds):
            self.assertEqual(len(window), 32)
            np.testing.assert_array_equal(window[:end - start], audio[start:end])
            self.assertTrue(np.all(window[end - start:] == 0))

    def test_exact_four_seconds(self):
        self.check_windows(32, [(0, 32)])

    def test_short_clip_reports_only_recorded_samples(self):
        self.check_windows(28, [(0, 28)])
        self.check_windows(1, [(0, 1)])

    def test_empty_audio_has_no_windows(self):
        self.check_windows(0, [])

    def test_long_clip_ends_on_a_hop(self):
        self.check_windows(40, [(0, 32), (4, 36), (8, 40)])

    def test_long_clip_ends_between_hops(self):
        self.check_windows(42, [(0, 32), (4, 36), (8, 40), (10, 42)])

    def test_no_detections_returns_empty_result(self):
        with patch.object(runner, "load_audio", return_value=(np.zeros(32), 8)), \
             patch.object(runner, "process_audio", return_value=[]):
            response = asyncio.run(runner.process(UploadFile(file=io.BytesIO())))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {"representantId": -1, "segments": []})


if __name__ == "__main__":
    unittest.main()
