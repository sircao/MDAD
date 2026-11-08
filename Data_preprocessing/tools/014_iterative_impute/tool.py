"""Iterative multivariate imputation (IterativeImputer).

Estimate each variable's missing values progressively from the other variables, suited to
scenarios with strong multivariate correlation. When used on time series, be mindful of
temporal ordering and future information leakage.
"""

from __future__ import annotations

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class IterativeImputeTool(FitApplyTool):
    name = "iterative_impute"
    category = "Missing Value Handling"
    version = "1.0.0"
    description = "Iteratively estimate multivariate missing values using IterativeImputer; must fit on the training set first."
    libraries = ("sklearn",)

    def fit(self, data, max_iter=10, random_state=0) -> ToolResult:
        params = {"max_iter": max_iter, "random_state": random_state}
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            self._fitted = IterativeImputer(max_iter=max_iter, random_state=random_state)
            self._fitted.fit(to_frame(data))
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
            out = type(df)(
                self._fitted.transform(df), index=df.index, columns=df.columns
            )
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = IterativeImputeTool()
