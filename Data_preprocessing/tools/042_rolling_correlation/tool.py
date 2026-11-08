"""Rolling correlation and covariance.

Compute the correlation and covariance changes of multiple variables within sliding
windows, to express how relationships between sensors change over time.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class RollingCorrelationTool(BaseTool):
    name = "rolling_correlation"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Compute correlation/covariance changes of two columns (or column pairs) within sliding windows."
    libraries = ("pandas",)

    def run(self, data, col_a=None, col_b=None, window=30, kind="corr") -> ToolResult:
        params = {"col_a": col_a, "col_b": col_b, "window": window, "kind": kind}
        try:
            df = to_frame(data)
            cols = list(df.columns)
            if col_a is None or col_b is None:
                if len(cols) < 2:
                    return self.fail(ValueError("At least two columns are required"), params)
                col_a, col_b = cols[0], cols[1]
            if kind == "corr":
                out = df[col_a].rolling(window).corr(df[col_b])
            elif kind == "cov":
                out = df[col_a].rolling(window).cov(df[col_b])
            else:
                return self.fail(ValueError(f"kind only supports corr / cov"), params)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = RollingCorrelationTool()
