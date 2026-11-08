"""Reuse the unified tool contract from Data_preprocessing and add a base class for knowledge retrieval tools."""

from __future__ import annotations

from Data_preprocessing.core.base import (
    BaseTool,
    FitApplyTool,
    ToolResult,
    dist_version,
    require,
)


class ReferenceTool(BaseTool):
    """Knowledge/resource index tools: run() returns structured metadata for Agent knowledge retrieval.

    Suitable for model introductions, benchmark notes, paper indexes, resource collections, surveys, and other entries that do not directly run algorithms.
    """

    metadata: dict = {}

    def run(self, *args, **kwargs) -> ToolResult:
        return self.ok(dict(self.metadata), {})


__all__ = ["BaseTool", "FitApplyTool", "ToolResult", "dist_version", "require", "ReferenceTool"]
