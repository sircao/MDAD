"""Multi-AXIS numeric serialization — round to the nearest integer."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MULTI_AXIS_SRC, ensure


class SerializeWindowTool(BaseTool):
    name = "multi_axis.serialize_window"
    category = "Multi-AXIS"
    version = "1.0.0"
    description = "Round floating-point observations to nearest-integer tokens readable by the LLM, reducing text length and keeping channels aligned."
    libraries = ("numpy", "torch")
    origin = "Multi-AXIS-B1DC/src/multi_axis/data.py:140 normalize_and_serialize() — np.rint(normalized[start:end].T * scale)"

    def run(self, values, interval, epsilon=1e-5, scale=100) -> ToolResult:
        params = {"interval": interval, "epsilon": epsilon, "scale": scale}
        try:
            ensure(MULTI_AXIS_SRC)
            from multi_axis.data import normalize_and_serialize

            _, serialized, _, _ = normalize_and_serialize(
                values, tuple(map(int, interval)), epsilon=epsilon, scale=scale
            )
            return self.ok(serialized, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = SerializeWindowTool()
