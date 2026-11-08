"""Rolling volatility anomaly detection: uses the volatility within a local window as the pointwise anomaly score.

Especially effective for **segment-level** anomalies such as "level drift / variance increase" -- the anomaly is
not a single extreme point but a sustained stretch of high volatility. This is a common practice in industrial
process monitoring (e.g. a rolling standard deviation control chart).

A larger window is smoother and better at capturing slow drift, but also lags more; a window that is too small
is sensitive to single-point noise. Choosing the window for given data is an unsupervised hyperparameter choice
that can be performed by an upper-layer agent via Pareto-front / orthogonal search.

Univariate input (1-D) directly outputs pointwise scores; multivariate input (2-D, rows=time, columns=variables)
computes rolling volatility per column and then **takes the maximum across columns** -- any variable whose
volatility rises is flagged as anomalous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.utils import to_frame

_STATS = ("std", "range", "mad")


def _rolling_stat(s: np.ndarray, stat: str, window: int) -> np.ndarray:
    ser = pd.Series(np.asarray(s, dtype=float))
    min_periods = max(2, int(window // 2))
    if stat == "std":
        score = ser.rolling(window, min_periods=min_periods).std()
    elif stat == "range":

        def _rng(w):
            w = np.asarray(w, dtype=float)
            return float(w.max() - w.min())

        score = ser.rolling(window, min_periods=min_periods).apply(_rng, raw=True)
    else:  # mad -- median absolute deviation, more robust to outliers

        def _mad(w):
            w = np.asarray(w, dtype=float)
            return float(np.median(np.abs(w - np.median(w))))

        score = ser.rolling(window, min_periods=min_periods).apply(_mad, raw=True)
    return np.nan_to_num(score.to_numpy(dtype=float), nan=0.0)


class RollingVolatilityTool(BaseTool):
    name = "rolling_volatility"
    category = "Algorithm Libraries & Frameworks"
    version = "1.0.0"
    description = "Rolling-window volatility (std/range/mad) as pointwise anomaly scores, good at segment-level anomalies like level drift/variance increase; for multivariate data takes the maximum across columns."
    libraries = ("numpy", "pandas")
    origin = "https://en.wikipedia.org/wiki/Shewhart_individuals_control_chart (common rolling standard deviation control chart practice)"

    def run(self, data, stat="std", window=100, **kwargs) -> ToolResult:
        params = {"stat": stat, "window": window, **kwargs}
        try:
            require(["numpy", "pandas"])
            if stat not in _STATS:
                return self.fail(ValueError(f"Supported statistics: {sorted(_STATS)}"), params)
            df = to_frame(data)  # unify into a wide table (rows=time, columns=variables)
            per_col = df.apply(lambda c: _rolling_stat(c.to_numpy(), stat, window), raw=False)
            score = per_col.max(axis=1).to_numpy(dtype=float)
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = RollingVolatilityTool()
