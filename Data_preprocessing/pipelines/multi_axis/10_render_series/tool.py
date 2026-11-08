"""Multi-AXIS visual rendering — render the normalized series."""

from __future__ import annotations

from pathlib import Path

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class RenderSeriesTool(BaseTool):
    name = "multi_axis.render_series"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Render an image view of the same observations as the numeric window; the rendering carries no labels/scores/causal information."
    libraries = ("numpy", "matplotlib", "Pillow")
    origin = "Multi-AXIS-B1DC/src/multi_axis/rendering.py:94-198 render_normalized_series()"

    def run(self, normalized_series, interval, channel_ids, output_path, dpi=600) -> ToolResult:
        params = {"interval": interval, "dpi": dpi}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.rendering import render_normalized_series

            audit = render_normalized_series(
                normalized_series,
                tuple(map(int, interval)),
                list(channel_ids),
                Path(output_path),
                dpi=dpi,
            )
            return self.ok(audit, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = RenderSeriesTool()
