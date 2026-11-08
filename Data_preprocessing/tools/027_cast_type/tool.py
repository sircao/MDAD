"""Type casting and precision control.

Conversion of numeric, categorical, and time types as well as floating-point precision adjustment.
Precision compression can reduce storage, but may make small anomalies disappear.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class CastTypeTool(BaseTool):
    name = "cast_type"
    category = "Transformation and Distribution Correction"
    version = "1.0.0"
    description = "Numeric/categorical/time type conversion and floating-point precision compression."
    libraries = ("pandas", "numpy")

    def run(self, data, dtype=None, decimals=None) -> ToolResult:
        params = {"dtype": dtype, "decimals": decimals}
        try:
            df = to_frame(data)
            out = df
            if dtype is not None:
                out = out.astype(dtype)
            if decimals is not None:
                numeric = out.select_dtypes(include=[np.number])
                out[numeric.columns] = numeric.round(decimals)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = CastTypeTool()
