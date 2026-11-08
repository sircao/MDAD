"""Downsampling and resample aggregation.

Aggregate high-frequency data to a lower frequency, using mean, median, maximum, minimum,
standard deviation, or count as the aggregation result. Max and min sometimes preserve
spike anomalies better than the mean.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class ResampleAggregateTool(BaseTool):
    name = "resample_aggregate"
    category = "Sampling, Windowing, and Data Volume Control"
    version = "1.0.0"
    description = "Aggregate high-frequency data to a lower frequency by rule (mean / median / max / min / std / count)."
    libraries = ("pandas",)

    def run(self, data, rule="1min", agg="mean") -> ToolResult:
        params = {"rule": rule, "agg": agg}
        try:
            df = to_frame(data)
            if not isinstance(df.index, pd.DatetimeIndex):
                return self.fail(ValueError("Resample aggregation requires a DatetimeIndex"), params)
            agg_func = agg if isinstance(agg, (list, dict)) else [agg]
            out = df.resample(rule).agg(agg_func)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ResampleAggregateTool()
