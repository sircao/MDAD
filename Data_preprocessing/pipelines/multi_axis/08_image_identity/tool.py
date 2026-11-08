"""Multi-AXIS image identity — determine the image's unique id and path."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class ImageIdentityTool(BaseTool):
    name = "multi_axis.image_identity"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Give each sample a deterministic, verifiable image id and relative path, avoiding duplicate rendering."
    libraries = ("numpy", "torch")
    origin = "Multi-AXIS-B1DC/src/multi_axis/data.py:114-127 visual_image_id() / visual_image_relpath()"

    def run(self, base_sample_id, interval, renderer_version="multi-axis-vl-render-v1") -> ToolResult:
        params = {"base_sample_id": base_sample_id, "interval": interval, "renderer_version": renderer_version}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.data import visual_image_id, visual_image_relpath

            image_id = visual_image_id(str(base_sample_id), tuple(map(int, interval)), renderer_version)
            return self.ok({"image_id": image_id, "image_relpath": visual_image_relpath(image_id)}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = ImageIdentityTool()
