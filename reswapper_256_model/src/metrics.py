import time
import collections


class FPSCounter:
    def __init__(self, window=60):
        # Rolling window of timestamps; old entries drop out automatically
        self._timestamps = collections.deque(maxlen=window)

    def tick(self):
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self):
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        # Frames processed = entries - 1 (we measure intervals, not events)
        return 0.0 if elapsed <= 0 else (len(self._timestamps) - 1) / elapsed


class LatencyTracker:
    def __init__(self, window=60):
        # Rolling window; old samples drop out so stats reflect recent performance
        self._samples = collections.deque(maxlen=window)

    def record(self, start_time: float):
        # start_time is the camera capture timestamp, so this measures true
        # end-to-end pipeline delay (camera-out → virtual-camera-in).
        self._samples.append((time.perf_counter() - start_time) * 1000.0)

    @property
    def mean_ms(self):
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def p95_ms(self):
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        # Floor index gives the value at or below the 95th percentile
        return s[int(len(s) * 0.95)]

    @property
    def min_ms(self):
        return min(self._samples) if self._samples else 0.0

    @property
    def max_ms(self):
        return max(self._samples) if self._samples else 0.0
