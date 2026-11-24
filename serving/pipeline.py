"""Song-level inference on a whole recording, with no torch dependency (numpy + scipy + onnxruntime).

    find the songs -> one label per song -> the `/classify` response

With a model exported since 2026-11-24 (its graph also returns the temporal attention):

1. The recording is scanned with a 4 s window every 0.4 s.
2. Songs are the peaks of the "attention vote" curve (serving/scan.py); each song's class
   probabilities are the weighted mean of the windows that voted for it.
3. A dialect below `min_confidence` is reported as "Unknown".

On whole recordings of held-out males this finds 99.5 % of the annotated songs and recognises
97.5 % of the complete BhBl songs. The logic it replaces (4 s windows every 0.5 s, keep the most
confident of overlapping windows) recognises 57 % with the same model, because a window that cuts
a song's ending is confidently "Unfinished" (docs/RESULTS.md, "From windows to songs").

With an older graph (no attention output) the previous method is used: song events from the
tonal-SNR segmenter (`processing/segment.py`), each classified from 3 windows starting
0.6 / 0.4 / 0.2 s before its onset. It finds 87.5 % of the songs.

Digital silence in the upload is filled with background noise, as in training. The model is an ONNX
graph exported by `serving/export_onnx.py`; peak normalisation is inside the graph.
"""

import numpy as np
from processing.audio_repair import background_noise, fill_zero_runs
from processing.segment import segment_songs
from serving.scan import detect_songs

SR, WIN = 32000, 128000
OFFSETS = (-0.6, -0.4, -0.2)
TAIL_AFTER_END = 1.1  # seconds a window should keep after the detected song end
SCAN_HOP = 0.4  # seconds between windows of the scan (0.2 and 0.4 find the same songs)
SONG_BEFORE_ENDING, SONG_AFTER_ENDING = 3.0, 0.4  # the reported interval around a detected ending


class SongClassifier:
    def __init__(self, model_path, labels, threads=4, min_confidence=0.6, min_representant=0.65):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        self.labels = list(labels)
        self.min_conf, self.min_rep = min_confidence, min_representant
        self.dialects = [l for l in self.labels if l not in ('None', 'Unfinished')]
        # graphs exported since 2026-11-24 also return the temporal attention: scan with them
        self.has_attention = 'att' in [output.name for output in self.sess.get_outputs()]

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
        """y: mono float32 waveform at 32 kHz. Returns the `/classify` response dict.

        With a model that returns its attention, songs are found by scanning the recording
        (serving/scan.py); older graphs fall back to the energy segmenter with three aligned
        windows per song.
        """
        # Digital silence (decoder delay, zero padding) is filled like in training: a black block
        # after a song reads as "no ending".
        y = fill_zero_runs(np.asarray(y, np.float32))
        if self.has_attention:
            songs = self.scan(y)
        else:
            songs = self.segment_and_classify(y)
        return self.response(songs)

    def scan(self, y):
        """Songs found by attention voting: list of ((start, end), class probabilities)."""
        duration = max(len(y), WIN) / SR
        starts = np.arange(0, duration - WIN / SR + 1e-6, SCAN_HOP)
        windows = self._windows(y, starts)
        probabilities, attention = [], []
        for first in range(0, len(windows), 16):
            batch = self.sess.run(['probs', 'att'], {'waveform': windows[first : first + 16]})
            probabilities.append(batch[0])
            attention.append(batch[1])
        probabilities, attention = np.concatenate(probabilities), np.concatenate(attention)
        attention_peak = attention.argmax(1) * (WIN / SR / attention.shape[1])

        none_index = self.labels.index('None')
        songs = []
        for ending, best, mean_probabilities in detect_songs(
            starts, probabilities, attention_peak, SCAN_HOP, none_index
        ):
            # a detection asserts a song: report its most likely class other than "None"
            song_probabilities = mean_probabilities.copy()
            song_probabilities[none_index] = 0
            song_probabilities /= song_probabilities.sum()
            interval = (
                max(0.0, ending - SONG_BEFORE_ENDING),
                min(len(y) / SR, ending + SONG_AFTER_ENDING),
            )
            songs.append((interval, song_probabilities))
        return songs

    def segment_and_classify(self, y):
        """Songs from the energy segmenter, each classified from three aligned windows."""
        events = self.songs(y)
        if not events:
            return []
        starts = [self.window_start(s0, s1, o) for s0, s1 in events for o in OFFSETS]
        P = self._predict(self._windows(y, starts)).reshape(len(events), len(OFFSETS), -1).mean(1)
        return list(zip(events, P))

    def response(self, songs):
        """The `/classify` response for a list of ((start, end), class probabilities)."""
        if not songs:
            return {'representantId': -1, 'segments': []}
        segments = []
        for (s0, s1), p in songs:
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
