"""Savitzky–Golay filtering.

Smooth the signal by local polynomial fitting while preserving local peak shape and derivative
information as much as possible. The window length and polynomial order affect the result.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class SavgolFilterTool(BaseTool):
    name = "savgol_filter"
    category = "Filtering, Decomposition, and Time-Frequency Analysis"
    version = "1.0.0"
    description = "Savitzky–Golay local polynomial smoothing, preserving local peak shape and derivative information."
    libraries = ("scipy",)

    def run(self, data, window_length=11, polyorder=2) -> ToolResult:
        params = {"window_length": window_length, "polyorder": polyorder}
        try:
            require(["scipy"])
            from scipy.signal import savgol_filter

            df = to_frame(data)
            out = df.apply(lambda s: savgol_filter(s.to_numpy(), window_length, polyorder))
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SavgolFilterTool()
