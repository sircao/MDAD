"""Multi-AXIS batch alignment — collate multi-axis sequences."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class CollateBatchTool(BaseTool):
    name = "multi_axis.collate_batch"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Collate variable-length sequences into tensors, using time_mask/channel_mask to prevent padding from being treated as real evidence."
    libraries = ("numpy", "torch")
    origin = "Multi-AXIS-B1DC/src/multi_axis/data.py:274-317 collate_multiaxis()"

    def run(self, samples) -> ToolResult:
        params = {"n_samples": len(samples)}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.data import collate_multiaxis

            return self.ok(collate_multiaxis(samples), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = CollateBatchTool()
