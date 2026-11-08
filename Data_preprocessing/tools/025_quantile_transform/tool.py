"""Rank transform and quantile transform.

Map values to a uniform or normal distribution according to the empirical distribution of the
training data, reducing skewness and scale differences across variables. The relative spacing of
extreme values may change as a result.
"""

from __future__ import annotations

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class QuantileTransformTool(FitApplyTool):
    name = "quantile_transform"
    category = "Transformation and Distribution Correction"
    version = "1.0.0"
    description = "Quantile transform (uniform/normal) by the empirical distribution of the training set; requires fitting first."
    libraries = ("sklearn",)

    def fit(self, data, output_distribution="normal", n_quantiles=1000) -> ToolResult:
        params = {"output_distribution": output_distribution, "n_quantiles": n_quantiles}
        try:
            from sklearn.preprocessing import QuantileTransformer

            self._fitted = QuantileTransformer(
                output_distribution=output_distribution, n_quantiles=n_quantiles
            )
            self._fitted.fit(to_frame(data))
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
            out = type(df)(
                self._fitted.transform(df), index=df.index, columns=df.columns
            )
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = QuantileTransformTool()
