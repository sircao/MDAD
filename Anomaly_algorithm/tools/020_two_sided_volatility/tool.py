"""Two-sided volatility: deviation from the median volatility is anomalous, regardless of direction.

Rolling volatility has an inherent directional ambiguity -- whether "high volatility is anomalous" or "low
volatility / stuck is anomalous" depends on the data. Taking ``|rolling std - median rolling std|`` treats both
tails as anomalous, fixing the inversion for stuck-type data, at the cost of only a small loss of discrimination
for high-volatility data.
"""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult, require
from Data_preprocessing.core.utils import to_frame


class TwoSidedVolatilityTool(BaseTool):
    name = "two_sided_volatility"
    category = "Algorithm Libraries & Frameworks"
    version = "1.0.0"
    description = "Two-sided volatility: |rolling std - median rolling std|, direction-agnostic, covering both high-volatility and low-volatility/stuck anomalies."
    libraries = ("numpy", "pandas")
    origin = "Two-sided transform (direction-agnostic), native implementation"

    def run(self, data, window=100, **kwargs) -> ToolResult:
        params = {"window": window, **kwargs}
        try:
            require(["numpy", "pandas"])
            df = to_frame(data)
            vol = df.rolling(window, min_periods=max(2, window // 2)).std()
            vol = np.nan_to_num(vol.to_numpy(dtype=float), nan=0.0)
            vol = vol.max(axis=1)  # multivariate: take maximum across columns
            med = np.median(vol)
            score = np.abs(vol - med)
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TwoSidedVolatilityTool()
