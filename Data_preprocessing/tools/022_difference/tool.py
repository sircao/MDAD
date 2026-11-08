"""Differencing and seasonal differencing.

Weaken trend and periodicity through first-order, higher-order, or seasonal differencing and
emphasize local changes. Differencing amplifies noise and may also make persistent level shifts
less apparent.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class DifferenceTool(BaseTool):
    name = "difference"
    category = "Transformation and Distribution Correction"
    version = "1.0.0"
    description = "First-order/higher-order differencing and seasonal differencing, weakening trend and periodicity and emphasizing local changes."
    libraries = ("pandas",)

    def run(self, data, order=1, seasonal=0) -> ToolResult:
        params = {"order": order, "seasonal": seasonal}
        try:
            df = to_frame(data)
            out = df
            for _ in range(order):
                out = out.diff()
            if seasonal:
                out = out.diff(seasonal)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DifferenceTool()
