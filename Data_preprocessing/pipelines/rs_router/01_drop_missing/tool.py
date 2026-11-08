"""RS-TOUTER input reading — drop missing values."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.utils import to_frame


class DropMissingTool(BaseTool):
    name = "rs_router.drop_missing"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Drop sample rows that contain missing values, ensuring downstream standardization and matrix operations do not produce non-numeric values."
    libraries = ("pandas",)
    origin = "TS-Router-D8FF/infer.py:17 read_input() — frame = pd.read_csv(path).dropna()"

    def run(self, data) -> ToolResult:
        params = {}
        try:
            df = to_frame(data)
            out = df.dropna()
            warnings = [f"Removed {len(df) - len(out)} missing-value row(s)"] if len(out) < len(df) else []
            return self.ok(out, params, warnings)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = DropMissingTool()
