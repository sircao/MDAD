"""Common base classes, result structure, and provenance structure for preprocessing tools.

Conventions
-----------
* Each tool is ``tools/<NNN>_<name>/tool.py``; the module defines a ``TOOL`` singleton.
* Tool inputs and outputs go through :class:`ToolResult` with provenance (tool version,
  dependency library versions, call parameters, and runtime), for audit and reproduction.
* Tools that need to fit parameters are stateful (``stateful=True``) and must ``fit(training
  set)`` first, then ``apply(new data)``, to avoid leaking test-time information into
  scaling, thresholds, or feature selection.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np


# Common import-name -> distribution-name mismatches, used to look up installed versions.
_ALIASES = {
    "sklearn": "scikit-learn",
    "pywt": "PyWavelets",
    "PyEMD": "EMD-signal",
    "imblearn": "imbalanced-learn",
}


def dist_version(import_name: str) -> str:
    """Return the installed version of a library; return ``"not installed"`` when absent."""
    for candidate in {import_name, _ALIASES.get(import_name, "")}:
        if not candidate:
            continue
        try:
            return _pkg_version(candidate)
        except PackageNotFoundError:
            continue
    return "not installed"


def require(libraries: List[str]) -> None:
    """Check that dependency libraries are importable; raise an error with install hints if missing."""
    missing = [name for name in libraries if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            "Required libraries for the tool are not installed: " + ", ".join(missing)
            + ". Please run: pip install " + " ".join(missing)
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    """Convert parameters to a JSON-serializable form for audit records."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


@dataclass
class ToolResult:
    """Unified return structure: success, data, warnings, and provenance."""

    ok: bool
    data: Any = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable summary without the ``data`` body (for audit/logging)."""
        return {
            "ok": self.ok,
            "error": self.error,
            "warnings": self.warnings,
            "provenance": self.provenance,
        }


class BaseTool:
    """Base class for all preprocessing tools.

    Subclasses must provide ``name``, ``category``, ``description``, ``libraries``,
    and implement :meth:`run`.
    """

    name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = ""
    libraries: ClassVar[tuple] = ()
    stateful: ClassVar[bool] = False
    origin: ClassVar[str] = ""  # provenance: original project path + function/class + line number

    def _provenance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "tool_version": self.version,
            "libraries": {lib: dist_version(lib) for lib in self.libraries},
            "params": _jsonable(params),
            "ran_at": _now_iso(),
        }

    def ok(self, data: Any, params: Dict[str, Any], warnings=()) -> ToolResult:
        return ToolResult(
            ok=True,
            data=data,
            warnings=list(warnings),
            provenance=self._provenance(params),
        )

    def fail(self, exc: Exception, params: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            provenance=self._provenance(params),
        )

    def run(self, data: Any, **params) -> ToolResult:
        raise NotImplementedError


class FitApplyTool(BaseTool):
    """Base class for stateful tools that need to fit parameters.

    Provides separate ``fit`` / ``apply`` interfaces; ``run`` is a one-shot
    fit+apply (for quick trials only) and attaches a note: production flows should
    ``fit`` on the training set first, then ``apply`` to the test set, to avoid
    information leakage.
    """

    stateful = True

    def __init__(self) -> None:
        self._fitted: Any = None
        self._fit_provenance: Dict[str, Any] = {}

    def fit(self, data: Any, **params) -> ToolResult:
        raise NotImplementedError

    def apply(self, data: Any) -> ToolResult:
        raise NotImplementedError

    def run(self, data: Any, **params) -> ToolResult:
        fit_result = self.fit(data, **params)
        if not fit_result.ok:
            return fit_result
        out = self.apply(data)
        if out.ok:
            out.warnings.append(
                "run() is a one-shot fit+apply; in production, fit on the training set first, then apply to the test set, to avoid information leakage"
            )
            out.provenance["fit_provenance"] = self._fit_provenance
        return out
