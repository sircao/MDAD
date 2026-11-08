"""Detrending and baseline correction.

Includes linear, polynomial, rolling-median, and baseline subtraction. Suitable for extracting
deviations under slow drift, but the window or fitting interval must not contain future information
that is unavailable at detection time (rolling uses a lagged window).
"""

from __future__ import annotations

import numpy as np

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class DetrendTool(FitApplyTool):
    name = "detrend"
    category = "Transformation and Distribution Correction"
    version = "1.0.0"
    description = "Linear/polynomial detrending or lagged rolling-median baseline correction."
    libraries = ("numpy", "pandas")

    def fit(self, data, method="linear", order=1, window=None) -> ToolResult:
        params = {"method": method, "order": order, "window": window}
        try:
            df = to_frame(data)
            if method == "polynomial":
                self._fitted = {
                    "method": method,
                    "coeffs": {
                        col: np.polyfit(np.arange(len(df)), df[col].to_numpy(dtype=float), order)
                        for col in df.columns
                    },
                }
            elif method == "linear":
                self._fitted = {
                    "method": "polynomial",
                    "coeffs": {
                        col: np.polyfit(np.arange(len(df)), df[col].to_numpy(dtype=float), 1)
                        for col in df.columns
                    },
                }
            elif method == "rolling":
                if not window:
                    return self.fail(ValueError("rolling detrending requires the window parameter"), params)
                self._fitted = {"method": "rolling", "window": window}
            else:
                return self.fail(ValueError(f"method only supports linear / polynomial / rolling"), params)
            self._fit_provenance = self._provenance(params)
            return self.ok("Fitted", params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data) -> ToolResult:
        params = {}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("Not yet fit; please call fit() on the training set first"), params)
            df = to_frame(data)
            if self._fitted["method"] == "rolling":
                roll = df.rolling(self._fitted["window"], min_periods=1)
                out = df - roll.median()
            else:
                x = np.arange(len(df))
                out = df.copy().astype(float)
                for col in df.columns:
                    trend = np.polyval(self._fitted["coeffs"][col], x)
                    out[col] = df[col].to_numpy(dtype=float) - trend
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DetrendTool()
