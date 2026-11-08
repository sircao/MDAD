"""MTS view organization and expansion — window sample reconstruction."""

from __future__ import annotations

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import MTS_SRC, ensure


class RebuildWindowSampleTool(BaseTool):
    name = "mts_generation.rebuild_window_sample"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Generates anomaly/normal views from the same window, expanding question diversity without rewriting ground-truth labels."
    libraries = ("numpy",)
    origin = "MTS_Data/src/mvaxis/frame_builder.py:91 rebuild_window_sample() (batch entry point :159 build_question_frames())"

    def run(self, raw_row, window_row) -> ToolResult:
        params = {}
        try:
            ensure(MTS_SRC)
            from mvaxis.frame_builder import rebuild_window_sample

            return self.ok(rebuild_window_sample(raw_row, window_row), params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = RebuildWindowSampleTool()
