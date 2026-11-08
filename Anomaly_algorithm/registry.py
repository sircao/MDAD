"""Tool registry: scan tools/*/tool.py and collect each tool's TOOL singleton."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

from .core import BaseTool, ReferenceTool

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
_CACHE: Dict[str, BaseTool] | None = None
_LOAD_ERRORS: dict[str, str] = {}


def discover(force: bool = False) -> Dict[str, BaseTool]:
    global _CACHE, _LOAD_ERRORS
    if _CACHE is not None and not force:
        return _CACHE
    found: Dict[str, BaseTool] = {}
    _LOAD_ERRORS = {}
    if _TOOLS_DIR.is_dir():
        for tool_py in sorted(_TOOLS_DIR.glob("*/tool.py")):
            module_name = f"Anomaly_algorithm.tools.{tool_py.parent.name}.tool"
            spec = importlib.util.spec_from_file_location(module_name, tool_py)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                _LOAD_ERRORS[str(tool_py)] = f"{type(exc).__name__}: {exc}"
                continue
            tool = getattr(module, "TOOL", None)
            if isinstance(tool, BaseTool):
                found[tool.name] = tool
    _CACHE = found
    return _CACHE


def load_errors() -> dict[str, str]:
    """Return import failures from the latest discovery pass."""
    discover()
    return dict(_LOAD_ERRORS)


def get(name: str) -> BaseTool:
    tools = discover()
    if name not in tools:
        failures = load_errors()
        raise KeyError(f"Tool not found: {name}. Available: {sorted(tools)}; load failures: {failures}")
    return tools[name]


def list_tools() -> List[dict]:
    rows = []
    for tool in sorted(discover().values(), key=lambda t: t.name):
        rows.append(
            {
                "name": tool.name,
                "category": tool.category,
                "version": tool.version,
                "description": tool.description,
                "libraries": list(tool.libraries),
                "runnable": not isinstance(tool, ReferenceTool),
                "origin": tool.origin,
                "installed": all(
                    importlib.util.find_spec(lib) is not None for lib in tool.libraries
                ),
            }
        )
    return rows


def by_category() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for tool in sorted(discover().values(), key=lambda t: t.name):
        grouped.setdefault(tool.category, []).append(tool.name)
    return grouped
