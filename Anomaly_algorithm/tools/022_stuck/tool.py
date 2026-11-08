"""Stuck detection: constant run length -- a value that does not change for a long time is stuck; acts directly on the raw data.

Does not depend on the direction of detector scores, but instead finds "near-constant segments" in the raw
sequence: the longer the constant segment a point belongs to, the more likely it is a stuck sensor. Provides a
more direct criterion for flatline-type anomalies that does not require a score direction.
"""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.utils import to_frame


def _runlen_constant(x: np.ndarray, eps: float) -> np.ndarray:
    """Length of the near-constant segment each point belongs to."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.ones(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(x[j + 1] - x[j]) <= eps:
            j += 1
        out[i : j + 1] = j - i + 1
        i = j + 1
    return out


class StuckTool(BaseTool):
    name = "stuck"
    category = "Algorithm Libraries & Frameworks"
    version = "1.0.0"
    description = "Stuck detection: constant run length (a value unchanged for a long time is stuck), acts directly on the raw data."
    libraries = ("numpy", "pandas")
    origin = "Native implementation"

    def run(self, data, eps=1e-3, **kwargs) -> ToolResult:
        params = {"eps": eps, **kwargs}
        try:
            require(["numpy", "pandas"])
            df = to_frame(data)
            per_col = df.apply(lambda c: _runlen_constant(c.to_numpy(dtype=float), eps))
            score = per_col.max(axis=1).to_numpy(dtype=float)
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = StuckTool()
