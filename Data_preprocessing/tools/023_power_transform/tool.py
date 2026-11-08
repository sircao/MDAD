"""Logarithmic, power, and variance-stabilizing transforms.

Includes logarithmic, square-root, Box–Cox, and Yeo–Johnson transforms, which can compress
long-tailed distributions or stabilize variance. Box–Cox generally requires positive values;
Yeo–Johnson can handle zeros and negative values.
"""

from __future__ import annotations

import numpy as np

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class PowerTransformTool(FitApplyTool):
    name = "power_transform"
    category = "Transformation and Distribution Correction"
    version = "1.0.0"
    description = "log/sqrt/Box-Cox/Yeo-Johnson transforms; Box-Cox and Yeo-Johnson require fitting first."
    libraries = ("sklearn", "numpy")

    def fit(self, data, method="yeo-johnson") -> ToolResult:
        params = {"method": method}
        try:
            if method not in ("log", "sqrt", "box-cox", "yeo-johnson"):
                return self.fail(ValueError(f"unsupported transform: {method}"), params)
            self._fitted = {"method": method, "transformer": None}
            if method in ("box-cox", "yeo-johnson"):
                from sklearn.preprocessing import PowerTransformer

                self._fitted["transformer"] = PowerTransformer(method=method, standardize=False)
                self._fitted["transformer"].fit(to_frame(data))
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
            method = self._fitted["method"]
            if method == "log":
                if (df <= 0).any().any():
                    return self.fail(ValueError("log transform requires positive values; consider using yeo-johnson instead"), params)
                out = np.log(df)
            elif method == "sqrt":
                out = np.sqrt(df)
            else:
                out = self._fitted["transformer"].transform(df)
                out = type(df)(out, index=df.index, columns=df.columns)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = PowerTransformTool()
