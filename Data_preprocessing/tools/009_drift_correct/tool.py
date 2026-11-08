"""Data drift and sensor bias correction.

Handle sensor offset using baseline correction, reference-channel correction, segmented
centering, or distribution alignment. Correction parameters should come from calibration
data or the training interval, avoiding information from the test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class DriftCorrectTool(FitApplyTool):
    name = "drift_correct"
    category = "Data Cleaning and Data Quality"
    version = "1.0.0"
    description = "Fit a per-channel baseline on the training interval and subtract it (supports median/mean baselines and rolling baselines)."
    libraries = ("pandas", "numpy")

    def fit(self, data, baseline="median", window=None) -> ToolResult:
        params = {"baseline": baseline, "window": window}
        try:
            df = to_frame(data)
            self._fitted = {
                "baseline": baseline,
                "window": window,
                "centers": df.median() if baseline == "median" else df.mean(),
            }
            self._fit_provenance = self._provenance(params)
            return self.ok(self._fitted["centers"], params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data) -> ToolResult:
        params = {}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("Not fitted yet; call fit() on the training set first"), params)
            df = to_frame(data)
            centers = self._fitted["centers"]
            window = self._fitted["window"]
            if window is None:
                out = df - centers.reindex(df.columns).to_numpy()
            else:
                roll = df.rolling(window, min_periods=1)
                out = df - (roll.median() if self._fitted["baseline"] == "median" else roll.mean())
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DriftCorrectTool()
