"""Multi-AXIS input decoding — extract numeric values."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class ExtractValuesTool(BaseTool):
    name = "multi_axis.extract_values"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Extract series.values from the record, rejecting empty/malformed sequences or sequences containing non-numeric/infinite values."
    libraries = ("numpy", "torch")
    origin = "Multi-AXIS-B1DC/src/multi_axis/data.py:64-76 extract_values()"

    def run(self, row) -> ToolResult:
        params = {}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.data import extract_values

            return self.ok(extract_values(row), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ExtractValuesTool()
