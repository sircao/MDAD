"""Multi-source sensor fusion.

Align by time and merge multi-sensor, device-state, environment, and load data to form
joint features. Before fusion, check clock bias, sampling frequency, and measurement
units.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import BaseTool, ToolResult


class MultisensorFuseTool(BaseTool):
    name = "multisensor_fuse"
    category = "Multivariate Fusion, Encoding, and Dimensionality Reduction"
    version = "1.0.0"
    description = "Align and merge multi-source data by time, optionally resampling/interpolating to a unified time axis."
    libraries = ("pandas",)

    def run(self, frames, how="outer", resample_rule=None, agg="mean") -> ToolResult:
        params = {"how": how, "resample_rule": resample_rule, "agg": agg}
        try:
            frames = [pd.DataFrame(f) if not isinstance(f, pd.DataFrame) else f for f in frames]
            if not frames:
                return self.fail(ValueError("frames cannot be empty"), params)
            if resample_rule is not None:
                frames = [
                    f.resample(resample_rule).agg(agg) if isinstance(f.index, pd.DatetimeIndex) else f
                    for f in frames
                ]
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.join(f, how=how, lsuffix="_l", rsuffix="_r")
            return self.ok(merged, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = MultisensorFuseTool()
