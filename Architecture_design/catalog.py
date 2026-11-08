"""Unified tool catalog: aggregates the registries of the four algorithm modules
for the agent to search and schedule.

Each of the four modules maintains ``registry.list_tools()``; here they are
combined into a single unified catalog:

* each row carries a ``module`` field for namespace isolation (the four modules
  may have colliding names);
* :func:`catalog_text` produces a compact text for injection into the system
  prompt so the LLM knows "what is available to schedule";
* :func:`get_tool` fetches a real tool instance by ``(module, name)`` for the
  executor to call;
* :func:`metadata` serves the ``query_tool`` knowledge lookup (without stuffing
  all 149 schemas into the model).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# (namespace, registry module, name of the get function that returns a single tool)
_MODULES = [
    ("data_preprocessing", "Data_preprocessing.registry"),
    ("anomaly", "Anomaly_algorithm.registry"),
    ("explanation", "Explanation_Algorithm.registry"),
    ("root_causal", "Root_Causal_Analysis_algorithm.registry"),
]

_MODULE_LABELS = {
    "data_preprocessing": "Data Preprocessing",
    "anomaly": "Anomaly Detection",
    "explanation": "Explainability",
    "root_causal": "Root Cause Localization",
}

_CACHE: Optional[List[dict]] = None


def _load() -> List[dict]:
    """Load and cache the tool catalog of the four modules (a single module failure does not sink the whole)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    rows: List[dict] = []
    for module, import_path in _MODULES:
        try:
            mod = __import__(import_path, fromlist=["list_tools"])
            for row in mod.list_tools():
                rows.append({"module": module, **row})
        except Exception as exc:  # noqa: BLE001 -- a single unavailable module only adds a placeholder, not fatal
            rows.append(
                {
                    "module": module,
                    "name": "<module load failed>",
                    "category": "load failed",
                    "description": f"{type(exc).__name__}: {exc}",
                    "runnable": False,
                    "installed": False,
                }
            )
    _CACHE = rows
    return _CACHE


def list_all() -> List[dict]:
    """Return the unified catalog (with the module field), for framework/test reads."""
    return _load()


def get_tool(module: str, name: str) -> Any:
    """Fetch a real tool instance by (module, name)."""
    for m, import_path in _MODULES:
        if m == module:
            mod = __import__(import_path, fromlist=["get"])
            return mod.get(name)
    raise KeyError(f"unknown module: {module}. available: {[m for m, _ in _MODULES]}")


def metadata(name: str, module: Optional[str] = None) -> Optional[dict]:
    """Fetch metadata by tool name (optionally restricted to a module), for query_tool knowledge lookup."""
    for row in _load():
        if row["name"] == name and (module is None or row["module"] == module):
            return {k: v for k, v in row.items()}
    return None


def catalog_text() -> str:
    """Generate a compact catalog text for injection into the system prompt."""
    grouped: Dict[str, List[dict]] = {}
    for row in _load():
        grouped.setdefault(row["module"], []).append(row)
    lines: List[str] = []
    for module, _ in _MODULES:
        items = grouped.get(module, [])
        lines.append(f"## {_MODULE_LABELS.get(module, module)} ({len(items)} items)")
        for r in sorted(items, key=lambda x: x["name"]):
            kind = "runnable" if r.get("runnable") else "knowledge"
            lines.append(f"- {r['name']} [{kind}] {r['description']}")
    return "\n".join(lines)


def catalog_brief() -> str:
    """Generate a more compact catalog (tool names only, one line per module), suitable for the system prompt without diluting the instructions."""
    grouped: Dict[str, List[dict]] = {}
    for row in _load():
        grouped.setdefault(row["module"], []).append(row)
    lines: List[str] = []
    for module, _ in _MODULES:
        items = grouped.get(module, [])
        names = ", ".join(sorted(r["name"] for r in items))
        lines.append(f"- {_MODULE_LABELS.get(module, module)} ({len(items)} items): {names}")
    return "\n".join(lines)
