"""Sensor stuck and constant-segment detection.

Identify intervals that hold the same value for a long time, change unusually few times,
or have variance near zero. Constant segments can be marked as a data quality issue or
an extra feature; indiscriminate deletion is not recommended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class DetectStuckTool(BaseTool):
    name = "detect_stuck"
    category = "Data Cleaning and Data Quality"
    version = "1.0.0"
    description = "Mark constant segments that hold the same value for a long time (stuck) or have variance near zero."
    libraries = ("pandas", "numpy")

    def _constant_runs(self, values: np.ndarray, min_len: int) -> np.ndarray:
        mask = np.zeros(len(values), dtype=bool)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[j] == values[i]:
                j += 1
            if j - i >= min_len:
                mask[i:j] = True
            i = j
        return mask

    def run(self, data, min_len=10, var_threshold=None) -> ToolResult:
        params = {"min_len": min_len, "var_threshold": var_threshold}
        try:
            df = to_frame(data)
            mask = pd.DataFrame(False, index=df.index, columns=df.columns)
            for col in df.columns:
                values = df[col].to_numpy()
                mask[col] = self._constant_runs(values, min_len)
                if var_threshold is not None:
                    mask[col] = mask[col] | (df[col].rolling(min_len, min_periods=1).var() <= var_threshold)
            stuck_cols = [c for c in df.columns if mask[c].any()]
            warnings = [f"Columns marked as stuck: {stuck_cols}"] if stuck_cols else []
            return self.ok(mask, params, warnings)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DetectStuckTool()
