"""RS-TOUTER input reading — label stripping."""

from __future__ import annotations

import numpy as np

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.utils import to_frame


class StripLabelsTool(BaseTool):
    name = "rs_router.strip_labels"
    category = "RS-TOUTER"
    version = "1.0.0"
    description = "Separate the label column from the numeric features, keeping only numeric values in the feature matrix and reserving labels for evaluation."
    libraries = ("pandas", "numpy")
    origin = "TS-Router-D8FF/infer.py:18-19 read_input() — labels=frame['Label']; data=frame.drop(columns=['Label'])"

    def run(self, data, label_column="Label") -> ToolResult:
        params = {"label_column": label_column}
        try:
            df = to_frame(data)
            labels = df[label_column].to_numpy(dtype=int) if label_column in df.columns else None
            features = df.drop(columns=[label_column], errors="ignore").to_numpy(dtype=float)
            return self.ok({"features": features, "labels": labels}, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = StripLabelsTool()
