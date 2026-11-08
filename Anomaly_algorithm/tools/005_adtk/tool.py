"""ADTK-style rule/statistical detectors (native implementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.utils import to_frame

_DETECTORS = ("quantile", "threshold", "persist", "level_shift")


def _level_shift(s: np.ndarray, window: int, c: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    out = np.zeros(len(s), dtype=bool)
    for t in range(window, len(s) - window):
        left, right = s[t - window:t], s[t:t + window]
        if abs(np.median(right) - np.median(left)) > c * (np.std(left) + 1e-8):
            out[t] = True
    return out


class AdtkTool(BaseTool):
    name = "adtk"
    category = "Algorithm Libraries & Frameworks"
    version = "2.0.0"
    description = "Unsupervised/rule-based time-series detection: quantile/fixed-threshold/persistence/level-shift, outputs boolean anomaly flags."
    libraries = ("numpy", "pandas")
    origin = "https://github.com/arundo/adtk (native equivalent implementation)"

    def run(self, data, detector="quantile", **kwargs) -> ToolResult:
        params = {"detector": detector, **kwargs}
        try:
            require(["numpy", "pandas"])
            if detector not in _DETECTORS:
                return self.fail(ValueError(f"Supported detectors: {sorted(_DETECTORS)}"), params)
            df = to_frame(data)

            if detector == "quantile":
                high, low = kwargs.get("high", 0.99), kwargs.get("low", 0.01)
                q_hi, q_lo = df.quantile(high), df.quantile(low)
                anomalies = (df > q_hi) | (df < q_lo)
            elif detector == "threshold":
                high, low = kwargs.get("high"), kwargs.get("low")
                anomalies = pd.DataFrame(False, index=df.index, columns=df.columns)
                if high is not None:
                    anomalies |= df > high
                if low is not None:
                    anomalies |= df < low
            elif detector == "persist":
                c = kwargs.get("c", 3.0)
                anomalies = df.diff().abs() > c * df.std()
            else:  # level_shift
                window, c = kwargs.get("window", 10), kwargs.get("c", 6.0)
                anomalies = df.apply(lambda s: _level_shift(s.to_numpy(), window, c))

            return self.ok(anomalies, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = AdtkTool()
