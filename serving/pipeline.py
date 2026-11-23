"""Song-level inference on a whole recording, with no torch dependency (numpy + scipy + onnxruntime).

    detect songs -> align a window on each song -> classify -> one label per song

1. Song events from the tonal-SNR segmenter (`processing/segment.py`); pieces of one song are merged.
2. Each song is classified from 3 windows starting 0.6 / 0.4 / 0.2 s before its onset (probabilities averaged),
   so the whole song is inside every window. For long songs the windows start later, so that at least 1.1 s
   follows the detected end: the detector often stops at the B element, before the quiet terminal C / E.
   Digital silence in the upload is filled with background noise, as in training.
3. Songs classified "None" are dropped. A dialect below `min_confidence` is reported as "Unknown".

Why not the old sliding-window logic (`runner.py`, 4 s windows every 0.5 s, keep the most confident of
overlapping windows): most of those windows cut a song somewhere, which is where "complete songs reported as
Unfinished" came from; and its max-over-windows merging and the bout aggregation of `runner_v2` were never
evaluated. On full recordings of held-out Vizina males this pipeline finds 0.875 of songs and labels 0.88 of the
complete BhBl songs correctly (production logic: 0.774 and 0.68).

The model is an ONNX graph exported by `serving/export_onnx.py`: waveform (B, 128000) @ 32 kHz -> probs (B, 7);
peak normalisation is inside the graph.
"""

import numpy as np
from processing.audio_repair import background_noise, fill_zero_runs
from processing.segment import segment_songs

SR, WIN = 32000, 128000
OFFSETS = (-0.6, -0.4, -0.2)
TAIL_AFTER_END = 1.1  # seconds a window should keep after the detected song end


class SongClassifier:
    def __init__(self, model_path, labels, threads=4, min_confidence=0.6, min_representant=0.65):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        self.labels = list(labels)
        self.min_conf, self.min_rep = min_confidence, min_representant
        self.dialects = [l for l in self.labels if l not in ('None', 'Unfinished')]

    def _windows(self, y, starts):
        if (
            len(y) < WIN
        ):  # same rule as training: pad with the recording's own background, never zeros
            y = np.concatenate([y, background_noise(y, WIN - len(y))])
        out = []
        for t in starts:
            i0 = max(0, min(int(round(t * SR)), len(y) - WIN))
            out.append(y[i0 : i0 + WIN])
        return np.stack(out).astype(np.float32)

    def _predict(self, X, bs=16):
        return np.concatenate(
            [self.sess.run(['probs'], {'waveform': X[i : i + bs]})[0] for i in range(0, len(X), bs)]
        )

    def songs(self, y):
        segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.4, max_dur=6.0)
        events = []
        for s0, s1, _ in segs:  # merge pieces of one song (gap < 0.6 s, total <= 4 s)
            if events and s0 - events[-1][1] < 0.6 and s1 - events[-1][0] <= 4.0:
                events[-1][1] = s1
            else:
                events.append([s0, s1])
        return [(s0, s1) for s0, s1 in events if s1 - s0 >= 0.6]

    @staticmethod
    def window_start(onset, end, offset):
        """Start of a 4 s window for a song detected at [onset, end].

        Normally `onset + offset` (0.6 / 0.4 / 0.2 s of lead-in). The detected end is often the B
        element; the quiet terminal C / E element follows up to 1.1 s later (research log 17.4).
        For a long song the window therefore starts later, so that it still reaches
        `end + TAIL_AFTER_END`, but never later than 0.1 s before the onset.
        """
        latest_useful = min(end + TAIL_AFTER_END - WIN / SR, onset - 0.1)
        return max(onset + offset, latest_useful)

    def classify(self, y):
        """y: mono float32 waveform at 32 kHz. Returns the `/classify` response dict."""
        # Digital silence (decoder delay, zero padding) is filled like in training: a black block
        # after a song reads as "no ending".
        y = fill_zero_runs(np.asarray(y, np.float32))
        events = self.songs(y)
        if not events:
            return {'representantId': -1, 'segments': []}
        starts = [self.window_start(s0, s1, o) for s0, s1 in events for o in OFFSETS]
        P = self._predict(self._windows(y, starts)).reshape(len(events), len(OFFSETS), -1).mean(1)
        segments = []
        for (s0, s1), p in zip(events, P):
            best = int(p.argmax())
            label = self.labels[best]
            if label == 'None':
                continue
            if label in self.dialects and p[best] < self.min_conf:
                label = 'Unknown'
            segments.append(
                {
                    'interval': [float(s0), float(s1)],
                    'label': label,
                    'confidence': float(p[best]),
                    'fullPredictions': {l: float(v) for l, v in zip(self.labels, p)},
                }
            )
        dialect = [i for i, s in enumerate(segments) if s['label'] in self.dialects]
        rep = max(dialect, key=lambda i: segments[i]['confidence']) if dialect else -1
        for i, s in enumerate(segments):
            s['isRepresentant'] = i == rep or (
                rep != -1
                and s['label'] == segments[rep]['label']
                and s['confidence'] >= self.min_rep
            )
        return {
            'representantId': rep,
            'segments': [{k: v for k, v in s.items() if k != 'confidence'} for s in segments],
        }
