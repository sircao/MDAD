"""Duplicate record detection and conflict handling.

Detect fully duplicated records, duplicated timestamps, and conflicting observations at
the same moment. Duplicate data can be handled by keeping the first, keeping the last,
averaging, taking the median, or applying business priority.
"""

from __future__ import annotations

import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class DeduplicateTool(BaseTool):
    name = "deduplicate"
    category = "Data Cleaning and Data Quality"
    version = "1.0.0"
    description = "Detect and handle duplicate records or duplicate timestamps (first / last / mean / median)."
    libraries = ("pandas",)

    def run(self, data, mode="records", subset=None, keep="first") -> ToolResult:
        params = {"mode": mode, "subset": subset, "keep": keep}
        try:
            df = to_frame(data)
            before = len(df)
            if mode == "records":
                out = df.drop_duplicates(subset=subset, keep=keep if keep in ("first", "last") else "first")
            elif mode == "index":
                if keep == "mean":
                    out = df.groupby(level=0).mean()
                elif keep == "median":
                    out = df.groupby(level=0).median()
                else:
                    out = df[~df.index.duplicated(keep=keep if keep in ("first", "last") else "first")]
            else:
                return self.fail(ValueError(f"mode only supports records / index, got {mode}"), params)

            removed = before - len(out)
            warnings = [f"Removed {removed} duplicate rows"] if removed else []
            return self.ok(out, params, warnings)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DeduplicateTool()
