"""Hysteresis, duration, and alarm merging.

Reduce alarm jitter via high/low dual thresholds, minimum duration, and adjacent-interval
merging, organizing point-level scores into event-level alarms. Parameters should be set
based on anomaly duration and alarm timeliness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_1d, to_frame


class HysteresisMergeTool(BaseTool):
    name = "hysteresis_merge"
    category = "Data Augmentation, Sampling, and Post-processing"
    version = "1.0.0"
    description = "High/low dual thresholds + minimum duration + adjacent merging, organizing point-level scores into events."
    libraries = ("pandas", "numpy")

    def _events_1d(self, s: np.ndarray, high: float, low: float, min_duration: int, merge_gap: int):
        events = []
        start = None
        for t, v in enumerate(s):
            if start is None and v >= high:
                start = t
            elif start is not None and v <= low:
                if t - start >= min_duration:
                    events.append((start, t - 1))
                start = None
        if start is not None and len(s) - start >= min_duration:
            events.append((start, len(s) - 1))
        if merge_gap and events:
            merged = [events[0]]
            for e in events[1:]:
                if e[0] - merged[-1][1] - 1 <= merge_gap:
                    merged[-1] = (merged[-1][0], e[1])
                else:
                    merged.append(e)
            events = merged
        return events

    def run(self, data, high=0.8, low=0.5, min_duration=3, merge_gap=0) -> ToolResult:
        params = {"high": high, "low": low, "min_duration": min_duration, "merge_gap": merge_gap}
        try:
            if isinstance(data, pd.DataFrame):
                result = {
                    col: self._events_1d(data[col].to_numpy(dtype=float), high, low, min_duration, merge_gap)
                    for col in data.columns
                }
            else:
                result = self._events_1d(to_1d(data), high, low, min_duration, merge_gap)
            return self.ok(result, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = HysteresisMergeTool()
