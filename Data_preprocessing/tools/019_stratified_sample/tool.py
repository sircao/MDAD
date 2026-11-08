"""Stratified sampling and representative sampling.

Perform stratified sampling by device, operating condition, time period, or category, preserving
different operating states as much as possible while reducing the size of the training data. Time
series should generally keep their temporal order to avoid future information leakage from random
splitting.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class StratifiedSampleTool(BaseTool):
    name = "stratified_sample"
    category = "Sampling, Windowing, and Data Volume Control"
    version = "1.0.0"
    description = "Stratified sampling by category/device/operating condition, or draw a representative subset by group."
    libraries = ("pandas",)

    def run(self, data, stratify_by=None, fraction=0.1, random_state=0) -> ToolResult:
        params = {"stratify_by": stratify_by, "fraction": fraction, "random_state": random_state}
        try:
            df = to_frame(data)
            if stratify_by is None:
                out = df.sample(frac=fraction, random_state=random_state)
            else:
                key = df[stratify_by] if isinstance(stratify_by, str) else pd.Series(stratify_by, index=df.index)
                out = df.groupby(key, group_keys=False).apply(
                    lambda g: g.sample(frac=fraction, random_state=random_state)
                )
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = StratifiedSampleTool()
