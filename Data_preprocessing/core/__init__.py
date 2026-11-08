"""Core framework: common base classes, result and provenance structures, and the tool registry."""

from .base import BaseTool, FitApplyTool, ToolResult, dist_version, require
from .registry import availability, by_category, discover, get, list_tools
from .utils import to_1d, to_frame

__all__ = [
    "BaseTool",
    "FitApplyTool",
    "ToolResult",
    "dist_version",
    "require",
    "to_frame",
    "to_1d",
    "discover",
    "get",
    "list_tools",
    "by_category",
    "availability",
]
