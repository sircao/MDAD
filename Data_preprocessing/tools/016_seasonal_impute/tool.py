"""Seasonal template imputation.

Fill missing values using historical mean, median, or adjacent-period values at the same
period, weekday, or seasonal position. Suited to data with stable seasonal regularity.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class SeasonalImputeTool(BaseTool):
    name = "seasonal_impute"
    category = "Missing Value Handling"
    version = "1.0.0"
    description = "Fill missing values at seasonal positions (hour / weekday / month, etc.) using historical mean or median."
    libraries = ("pandas",)

    def run(self, data, period="hour", method="mean") -> ToolResult:
        params = {"period": period, "method": method}
        try:
            df = to_frame(data)
            if not isinstance(df.index, pd.DatetimeIndex):
                return self.fail(
                    ValueError("Seasonal template imputation requires a DatetimeIndex as the time coordinate"), params
                )
            key = getattr(df.index, period)
            grouped = df.groupby(key).transform(method)
            out = df.fillna(grouped)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SeasonalImputeTool()
