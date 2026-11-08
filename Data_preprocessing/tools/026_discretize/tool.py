"""Discretization and symbolization.

Map continuous values to intervals, levels, or symbol sequences to construct rules, patterns, and
sequence features. Bin boundaries should be determined from the training data to avoid boundaries
changing after outliers participate.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import FitApplyTool, ToolResult
from ...core.utils import to_frame


class DiscretizeTool(FitApplyTool):
    name = "discretize"
    category = "Transformation and Distribution Correction"
    version = "1.0.0"
    description = "Equal-width/quantile binning maps continuous values to interval symbols; boundaries are determined at fit time."
    libraries = ("pandas",)

    def fit(self, data, n_bins=10, strategy="quantile") -> ToolResult:
        params = {"n_bins": n_bins, "strategy": strategy}
        try:
            if strategy not in ("quantile", "uniform"):
                return self.fail(ValueError(f"strategy only supports quantile / uniform"), params)
            df = to_frame(data)
            if strategy == "uniform":
                self._fitted = {
                    col: pd.cut(df[col], bins=n_bins, retbins=True)[1] for col in df.columns
                }
            else:
                self._fitted = {
                    col: pd.qcut(df[col].rank(method="first"), q=n_bins, retbins=True)[1]
                    for col in df.columns
                }
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
            out = pd.DataFrame(index=df.index, columns=df.columns)
            for col in df.columns:
                out[col] = pd.cut(df[col], bins=self._fitted[col], include_lowest=True).astype(str)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DiscretizeTool()
