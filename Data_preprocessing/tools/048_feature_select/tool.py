"""Feature selection and screening.

Reduce redundant features via statistical tests, correlation, low variance, or
recursive selection. Feature screening should be performed within the training data.
"""

from __future__ import annotations

import numpy as np

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class FeatureSelectTool(FitApplyTool):
    name = "feature_select"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Variance/correlation/univariate/recursive feature selection; requires fitting on the training set first."
    libraries = ("sklearn", "numpy")

    def fit(self, data, method="variance", threshold=0.0, k=10, labels=None) -> ToolResult:
        params = {"method": method, "threshold": threshold, "k": k}
        try:
            df = to_frame(data)
            if method == "variance":
                from sklearn.feature_selection import VarianceThreshold

                model = VarianceThreshold(threshold=threshold).fit(df)
                keep = model.get_support(indices=True)
            elif method == "correlation":
                corr = df.corr().abs()
                drop = set()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                for col in upper.columns:
                    for idx in upper.index[upper[col] > threshold]:
                        if col != idx:
                            drop.add(col)
                keep = [i for i, c in enumerate(df.columns) if c not in drop]
            elif method == "kbest":
                if labels is None:
                    return self.fail(ValueError("kbest requires the labels parameter"), params)
                from sklearn.feature_selection import SelectKBest, f_regression

                model = SelectKBest(f_regression, k=k).fit(df, np.asarray(labels).ravel())
                keep = model.get_support(indices=True)
            else:
                return self.fail(ValueError(f"method only supports variance / correlation / kbest"), params)
            self._fitted = {"keep": list(keep), "columns": list(df.columns)}
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
            cols = [self._fitted["columns"][i] for i in self._fitted["keep"]]
            return self.ok(df[cols], params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = FeatureSelectTool()
