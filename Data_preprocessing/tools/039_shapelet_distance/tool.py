"""Shapelet subsequence distance.

Transform a time series into distances to a set of representative subsequences,
suitable for constructing local shape-pattern features.
(Native implementation: use representative subsequences as shapelets and compute the
minimum DTW distance of each sliding sub-segment; tslearn provides learned shapelet
extraction.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


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


class ShapeletDistanceTool(BaseTool):
    name = "shapelet_distance"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Convert a series into distance features to a set of representative subsequences (shapelets)."
    libraries = ("numpy",)

    def _distance_to_shapelet(self, s: np.ndarray, shapelet: np.ndarray) -> float:
        m = len(shapelet)
        if m == 0 or len(s) < m:
            return float(np.inf)
        best = float("inf")
        for i in range(len(s) - m + 1):
            best = min(best, _dtw(s[i:i + m], shapelet))
        return best

    def run(self, data, shapelets=None, window=None, n_shapelets=3, seed=0) -> ToolResult:
        params = {"window": window, "n_shapelets": n_shapelets, "seed": seed}
        try:
            df = to_frame(data)
            rng = np.random.default_rng(seed)
            w = window or max(2, len(df) // 10)
            out = pd.DataFrame(index=df.columns, columns=[f"shapelet_{i}" for i in range(n_shapelets)])
            for col in df.columns:
                s = df[col].to_numpy(dtype=float)
                if shapelets is None:
                    starts = rng.integers(0, max(1, len(s) - w), size=n_shapelets)
                    shp = [s[st:st + w] for st in starts]
                else:
                    shp = [np.asarray(x, dtype=float) for x in shapelets]
                out.loc[col] = [self._distance_to_shapelet(s, x) for x in shp]
            return self.ok(out.astype(float), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ShapeletDistanceTool()
