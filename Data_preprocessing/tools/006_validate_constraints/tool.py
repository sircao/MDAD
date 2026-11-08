"""Data validation and constraint checking.

Check data validity by data type, value range, unit, time interval, and business rules,
for example whether temperature exceeds the physical range or whether a status value
belongs to a known category. Suitable for catching acquisition/format errors before
imputation and feature generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class ValidateConstraintsTool(BaseTool):
    name = "validate_constraints"
    category = "Data Cleaning and Data Quality"
    version = "1.0.0"
    description = "Check data validity by range / allowed values / type / time monotonicity, returning a violation mask and summary."
    libraries = ("pandas", "numpy")

    def run(
        self,
        data,
        ranges=None,
        allowed_values=None,
        dtypes=None,
        check_time_monotonic=False,
    ) -> ToolResult:
        params = {
            "ranges": ranges, "allowed_values": allowed_values,
            "dtypes": dtypes, "check_time_monotonic": check_time_monotonic,
        }
        try:
            df = to_frame(data)
            mask = pd.DataFrame(False, index=df.index, columns=df.columns)
            summary = {}

            ranges = ranges or {}
            for col, (lo, hi) in ranges.items():
                if col in df.columns:
                    bad = (df[col] < lo) | (df[col] > hi) | df[col].isna()
                    mask[col] = mask[col] | bad
                    summary.setdefault(col, {"out_of_range": int(bad.sum())})

            allowed_values = allowed_values or {}
            for col, allowed in allowed_values.items():
                if col in df.columns:
                    bad = ~df[col].isin(allowed) & df[col].notna()
                    mask[col] = mask[col] | bad
                    summary.setdefault(col, {})["not_in_allowed"] = int(bad.sum())

            dtypes = dtypes or {}
            for col, dtype in dtypes.items():
                if col in df.columns:
                    bad = ~pd.api.types.is_dtype_equal(
                        pd.Series(df[col]).dtype, pd.api.types.pandas_dtype(dtype)
                    ) if not isinstance(dtype, (list, tuple, set)) else ~df[col].isin(dtype)
                    summary.setdefault(col, {})["dtype_mismatch"] = int(bool(bad))

            if check_time_monotonic and isinstance(df.index, pd.DatetimeIndex):
                not_increasing = (df.index.to_series().diff().dt.total_seconds() < 0).sum()
                summary["__index__"] = {"time_not_increasing": int(not_increasing)}

            return self.ok(
                {"violation_mask": mask, "summary": pd.DataFrame(summary).T}, params
            )
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ValidateConstraintsTool()
