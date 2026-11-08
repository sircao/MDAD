"""Conditional prediction residuals.

Explain a target channel with one or more channels, then use the residual between
observed and conditionally predicted values as a relationship feature. Anomalies in the
training data may contaminate the conditional relationship model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class ConditionalResidualTool(FitApplyTool):
    name = "conditional_residual"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Regress the target channel with several channels and output the residual between observations and conditional predictions."
    libraries = ("statsmodels", "numpy")

    def fit(self, data, target=None, predictors=None) -> ToolResult:
        params = {"target": target, "predictors": predictors}
        try:
            import statsmodels.api as sm

            df = to_frame(data)
            cols = list(df.columns)
            target = target or cols[0]
            predictors = list(predictors) if predictors else [c for c in cols if c != target]
            X = df[predictors].to_numpy(dtype=float)
            y = df[target].to_numpy(dtype=float)
            model = sm.OLS(y, sm.add_constant(X)).fit()
            self._fitted = {
                "target": target,
                "predictors": predictors,
                "params": model.params,
            }
            self._fit_provenance = self._provenance(params)
            return self.ok("Fitted", params)
        except Exception as exc:
            return self.fail(exc, params)

    def apply(self, data) -> ToolResult:
        params = {}
        try:
            if self._fitted is None:
                return self.fail(RuntimeError("Not fitted yet; call fit() on the training set first"), params)
            df = to_frame(data)
            X = df[self._fitted["predictors"]].to_numpy(dtype=float)
            y = df[self._fitted["target"]].to_numpy(dtype=float)
            pred = X @ self._fitted["params"][1:] + self._fitted["params"][0]
            residual = y - pred
            return self.ok(pd.Series(residual, index=df.index, name=f"{self._fitted['target']}_residual"), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ConditionalResidualTool()
