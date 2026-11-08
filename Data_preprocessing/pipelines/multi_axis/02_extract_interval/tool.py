"""Multi-AXIS interval validation — extract the interval."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class ExtractIntervalTool(BaseTool):
    name = "multi_axis.extract_interval"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Extract and validate the target window interval, ensuring the target window is valid and preventing out-of-bounds slicing."
    libraries = ("numpy", "torch")
    origin = "Multi-AXIS-B1DC/src/multi_axis/data.py:79-86 extract_interval()"

    def run(self, row, steps) -> ToolResult:
        params = {"steps": steps}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.data import extract_interval

            return self.ok(extract_interval(row, steps), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ExtractIntervalTool()
