"""Autocorrelation and autoregressive features.

Autocorrelation, partial autocorrelation, and AR coefficients at different lags,
used to express local temporal dependencies and dynamic patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult, require
from ...core.utils import to_frame


class AutocorrFeaturesTool(BaseTool):
    name = "autocorr_features"
    category = "Feature Extraction and Feature Representation"
    version = "1.0.0"
    description = "Extract multi-lag autocorrelation, partial autocorrelation, and AR(1) coefficients."
    libraries = ("statsmodels", "numpy")

    def _features(self, s: np.ndarray, max_lag: int) -> dict:
        from statsmodels.tsa.stattools import acf, pacf

        s = np.asarray(s, dtype=float)
        n = len(s)
        nlags = min(max_lag, max(1, n - 2))
        ac = acf(s, nlags=nlags, fft=False)
        pc = pacf(s, nlags=nlags, method="ywm")
        out = {f"acf_lag{i}": float(ac[i]) for i in range(1, nlags + 1)}
        out.update({f"pacf_lag{i}": float(pc[i]) for i in range(1, nlags + 1)})
        if n >= 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0:
            out["ar1_coef"] = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        else:
            out["ar1_coef"] = 0.0
        return out

    def run(self, data, max_lag=5) -> ToolResult:
        params = {"max_lag": max_lag}
        try:
            require(["statsmodels"])
            df = to_frame(data)
            rows = {
                col: self._features(df[col].to_numpy(dtype=float), max_lag)
                for col in df.columns
            }
            out = pd.DataFrame(rows).T
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = AutocorrFeaturesTool()
