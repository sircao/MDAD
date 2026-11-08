"""DTW shape distance.

Compute the sequence shape distance after time warping, usable as a feature against
templates or other time segments.
(Native implementation; tslearn provides an optimized version.)
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_1d


def _dtw(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.full((len(x) + 1, len(y) + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = abs(x[i - 1] - y[j - 1])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[len(x), len(y)])


class DtwDistanceTool(BaseTool):
    name = "dtw_distance"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Compute the DTW shape distance between a series and several templates."
    libraries = ("numpy",)

    def run(self, data, templates=None) -> ToolResult:
        params = {"templates": "list" if templates is not None else None}
        try:
            x = to_1d(data)
            if templates is None:
                return self.fail(ValueError("Please provide templates (a list of reference templates)"), params)
            dists = [_dtw(x, np.asarray(t, dtype=float)) for t in templates]
            return self.ok(np.array(dists), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DtwDistanceTool()
