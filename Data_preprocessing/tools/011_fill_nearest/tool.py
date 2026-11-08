"""Nearest-value imputation.

Forward fill, backward fill, or nearest-neighbor fill, suited to short gaps or
state-holding data. Using nearest-value imputation for long gaps may create unrealistic
flat intervals.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class FillNearestTool(BaseTool):
    name = "fill_nearest"
    category = "Missing Value Handling"
    version = "1.0.0"
    description = "Forward fill, backward fill, or nearest-neighbor interpolation to fill missing values, suited to short gaps."
    libraries = ("pandas",)

    def run(self, data, method="ffill", limit=None) -> ToolResult:
        params = {"method": method, "limit": limit}
        try:
            df = to_frame(data)
            if method == "nearest":
                out = df.interpolate(method="nearest", limit=limit, limit_direction="both")
            elif method == "ffill":
                out = df.ffill(limit=limit)
            elif method == "bfill":
                out = df.bfill(limit=limit)
            else:
                return self.fail(
                    ValueError(f"method only supports ffill / bfill / nearest, got {method}"), params
                )
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = FillNearestTool()
