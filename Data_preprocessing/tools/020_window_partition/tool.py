"""Sliding window and window partitioning.

Split the sequence by fixed length, overlap ratio, and stride. Besides equal-length windows, it can
also segment by event, device state, or changepoint.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class WindowPartitionTool(BaseTool):
    name = "window_partition"
    category = "Sampling, Windowing, and Data Volume Control"
    version = "1.0.0"
    description = "Slice sliding windows by fixed length/stride/overlap, returning a list of windows and their intervals."
    libraries = ("pandas",)

    def run(self, data, window=64, stride=32) -> ToolResult:
        params = {"window": window, "stride": stride}
        try:
            df = to_frame(data)
            n = len(df)
            if window <= 0 or stride <= 0 or window > n:
                return self.fail(ValueError(f"illegal window parameters window={window} stride={stride} n={n}"), params)
            windows, ranges = [], []
            for start in range(0, n - window + 1, stride):
                windows.append(df.iloc[start:start + window])
                ranges.append((start, start + window))
            return self.ok({"windows": windows, "ranges": ranges}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = WindowPartitionTool()
