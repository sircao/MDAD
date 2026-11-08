"""Time interpolation and piecewise interpolation.

Linear, spline, or piecewise-polynomial interpolation over time intervals. If the sampling
interval is uneven, use the actual time coordinates rather than treating sample indices as
equally spaced time.
"""

from __future__ import annotations

from ...core.base import BaseTool, ToolResult
from ...core.utils import to_frame


class InterpolateTimeTool(BaseTool):
    name = "interpolate_time"
    category = "Missing Value Handling"
    version = "1.0.0"
    description = "Linear / spline / polynomial interpolation on the actual time coordinates."
    libraries = ("pandas",)

    def run(self, data, method="time", order=3, limit=None) -> ToolResult:
        params = {"method": method, "order": order, "limit": limit}
        try:
            df = to_frame(data)
            if method == "time":
                out = df.interpolate(method="time", limit=limit, limit_direction="both")
            elif method in ("linear", "slinear", "quadratic", "cubic"):
                out = df.interpolate(method=method, order=order, limit=limit, limit_direction="both")
            elif method in ("spline", "polynomial"):
                out = df.interpolate(method=method, order=order, limit=limit, limit_direction="both")
            else:
                return self.fail(ValueError(f"Unsupported interpolation method: {method}"), params)
            return self.ok(out, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = InterpolateTimeTool()
