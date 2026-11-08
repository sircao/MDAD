"""Neighbor-based imputation (KNNImputer).

Estimate missing values from similar samples or similar variables, suited to data with
correlational structure among variables. In high-dimensional or strongly non-stationary
data, neighbors may not be representative.
"""

from __future__ import annotations

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class KnnImputeTool(FitApplyTool):
    name = "knn_impute"
    category = "Missing Value Handling"
    version = "1.0.0"
    description = "Estimate missing values from similar samples using KNNImputer; must fit on the training set first."
    libraries = ("sklearn",)

    def fit(self, data, n_neighbors=5, weights="uniform") -> ToolResult:
        params = {"n_neighbors": n_neighbors, "weights": weights}
        try:
            from sklearn.impute import KNNImputer

            self._fitted = KNNImputer(n_neighbors=n_neighbors, weights=weights)
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


TOOL = KnnImputeTool()
