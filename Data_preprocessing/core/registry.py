"""Tool registry: scan tools/*/tool.py and collect each tool's ``TOOL`` singleton.

Tool modules are loaded by file path, so each tool folder is discoverable without an
``__init__.py``; tools import their dependencies lazily, so a missing library does not
make other tools unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

from .base import BaseTool

# Add the project root (E:\\DATA_PREPROCESSING) to sys.path so that
# ``from Data_preprocessing.core.base import ...`` inside tools works regardless of
# where the code is run from.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
_PIPELINES_DIR = Path(__file__).resolve().parent.parent / "pipelines"

_CACHE: Dict[str, BaseTool] | None = None


def _load_tool_module(module_name: str, tool_py: Path, found: Dict[str, BaseTool]) -> None:
    spec = importlib.util.spec_from_file_location(module_name, tool_py)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a single tool import failure must not break the whole registry
        found[f"<LOAD_FAILED>{tool_py.parent.name}"] = _BrokenTool(tool_py.parent.name, exc)
        return
    tool = getattr(module, "TOOL", None)
    if isinstance(tool, BaseTool):
        found[tool.name] = tool


def discover(force: bool = False) -> Dict[str, BaseTool]:
    """Scan and load all tools (including pipelines/), returning {tool name: tool instance}."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    found: Dict[str, BaseTool] = {}
    if _TOOLS_DIR.is_dir():
        for tool_py in sorted(_TOOLS_DIR.glob("*/tool.py")):
            module_name = f"Data_preprocessing.tools.{tool_py.parent.name}.tool"
            _load_tool_module(module_name, tool_py, found)

    if _PIPELINES_DIR.is_dir():
        for tool_py in sorted(_PIPELINES_DIR.glob("*/*/tool.py")):
            pipeline = tool_py.parent.parent.name
            stage = tool_py.parent.name
            module_name = f"Data_preprocessing.pipelines.{pipeline}.{stage}.tool"
            _load_tool_module(module_name, tool_py, found)

    _CACHE = found
    return _CACHE


class _BrokenTool(BaseTool):
    """Placeholder used when a tool module fails to import, so the registry can still report it and explain why."""

    def __init__(self, folder: str, exc: Exception) -> None:
        self.name = f"<LOAD_FAILED>{folder}"
        self.category = "load failure"
        self.description = f"Tool {folder} failed to import: {type(exc).__name__}: {exc}"
        self.libraries = ()

    def run(self, data, **params):
        from .base import ToolResult

        return ToolResult(ok=False, error=self.description)


def get(name: str) -> BaseTool:
    """Get a tool by name; raise an error listing available tools when not found."""
    tools = discover()
    if name not in tools:
        raise KeyError(f"Tool not found: {name}. Available: {sorted(tools)}")
    return tools[name]


def list_tools() -> List[dict]:
    """List metadata for all tools (for the Agent to read and dispatch)."""
    rows = []
    for tool in sorted(discover().values(), key=lambda t: t.name):
        rows.append(
            {
                "name": tool.name,
                "category": tool.category,
                "version": tool.version,
                "description": tool.description,
                "libraries": list(tool.libraries),
                "stateful": tool.stateful,
                "origin": tool.origin,
                "installed": all(
                    importlib.util.find_spec(lib) is not None for lib in tool.libraries
                ),
            }
        )
    return rows


def by_category() -> Dict[str, List[str]]:
    """Return tool names grouped by category."""
    grouped: Dict[str, List[str]] = {}
    for tool in sorted(discover().values(), key=lambda t: t.name):
        grouped.setdefault(tool.category, []).append(tool.name)
    return grouped


def availability() -> Dict[str, List[str]]:
    """Summarize dependency installation across tools: {installed: [...], missing: [...]}."""
    libs = {}
    for tool in discover().values():
        for lib in tool.libraries:
            libs.setdefault(lib, set()).add(tool.name)
    installed, missing = [], []
    for lib in sorted(libs):
        if importlib.util.find_spec(lib) is not None:
            installed.append(lib)
        else:
            missing.append(lib)
    return {"installed": installed, "missing": missing}
