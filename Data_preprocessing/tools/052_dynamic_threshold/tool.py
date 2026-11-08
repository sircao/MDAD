"""Dynamic thresholding and score standardization.

Apply rolling quantile thresholding, standardization, or multi-score scale unification
to anomaly scores output by a detector. This is post-processing of detection scores,
not raw-data cleaning.
"""

from __future__ import annotations

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class DynamicThresholdTool(FitApplyTool):
    name = "dynamic_threshold"
    category = "Data Augmentation, Sampling, and Post-processing"
    version = "1.0.0"
    description = "Rolling/global quantile thresholding and score standardization, outputting an alarm mask."
    libraries = ("pandas", "numpy")

    def fit(self, data, method="global", window=None, quantile=0.99) -> ToolResult:
        params = {"method": method, "window": window, "quantile": quantile}
        try:
            df = to_frame(data)
            if method not in ("global", "rolling"):
                return self.fail(ValueError(f"method only supports global / rolling"), params)
            if method == "rolling" and not window:
                return self.fail(ValueError("rolling requires the window parameter"), params)
            self._fitted = {
                "method": method,
                "window": window,
                "quantile": quantile,
                "mean": df.mean(),
                "std": df.std().replace(0, 1e-6),
                "global_threshold": df.quantile(quantile),
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
            normalized = (df - self._fitted["mean"]) / self._fitted["std"]
            if self._fitted["method"] == "rolling":
                threshold = df.rolling(self._fitted["window"], min_periods=1).quantile(
                    self._fitted["quantile"]
                )
            else:
                threshold = self._fitted["global_threshold"]
            alarm = df > threshold
            return self.ok({"normalized": normalized, "alarm": alarm, "threshold": threshold}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DynamicThresholdTool()
