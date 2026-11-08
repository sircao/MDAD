"""Executor: maps the tool calls sent by the LLM to the real tools' run() and
compresses the results to send back.

Design points:

* exposes only the curated functions in schemas.py, each of which "lands on" a
  real tool and returns a digested JSON (without full arrays), preserving
  provenance audit while keeping context length under control;
* maintains workspace state (continuous scores / alarm masks / events) so that
  post-processing tools can chain on;
* ``score_history`` records every detector call that produced continuous scores,
  so the framework can independently evaluate threshold-free metrics such as AUC
  -- the quality of the model's "which detector to pick" decision can be
  quantified objectively.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import catalog, schemas

# legal parameter names declared in each callable function's schema (used to clean up keys the model crams in).
_VALID_PARAMS = {
    fn["function"]["name"]: set(fn["function"]["parameters"]["properties"].keys())
    for fn in schemas.FUNCTIONS
}

# function name -> (module, real tool name)
_DETECTORS = {
    "pyod": ("anomaly", "pyod"),
    "adtk": ("anomaly", "adtk"),
    "merlion": ("anomaly", "merlion"),
    "tods": ("anomaly", "tods"),
    "elki": ("anomaly", "elki"),
    "pysad": ("anomaly", "pysad"),
    "time_rcd": ("anomaly", "time_rcd"),
    "tsbad_model": ("anomaly", "tsbad_model"),
    "rolling_volatility": ("anomaly", "rolling_volatility"),
    "two_sided_volatility": ("anomaly", "two_sided_volatility"),
    "two_sided_knn": ("anomaly", "two_sided_knn"),
    "stuck": ("anomaly", "stuck"),
}
_POST = {
    "dynamic_threshold": ("data_preprocessing", "dynamic_threshold"),
    "hysteresis_merge": ("data_preprocessing", "hysteresis_merge"),
}
_EXPLAIN = {
    "causelens": ("explanation", "causelens"),
    "kfasl": ("explanation", "kfasl"),
}

# search_toolchain candidate grid: detector x hyperparams (label-free search space).
_CANDIDATES = [
    ("rolling_volatility", {"stat": "std", "window": 20}),
    ("rolling_volatility", {"stat": "std", "window": 50}),
    ("rolling_volatility", {"stat": "std", "window": 100}),
    ("rolling_volatility", {"stat": "range", "window": 50}),
    ("rolling_volatility", {"stat": "range", "window": 100}),
    ("tods", {"detector": "moving_average", "return_scores": True}),
    ("pyod", {"algorithm": "iforest"}),
    ("merlion", {"detector": "ewma"}),
    ("two_sided_volatility", {"window": 100}),
    ("two_sided_knn", {"n_neighbors": 5}),
    ("stuck", {"eps": 1e-3}),
]

# ensemble_fuse default fusion panel (diverse and complementary).
_ENSEMBLE_PANEL = [
    ("rolling_volatility", {"stat": "std", "window": 50}),
    ("rolling_volatility", {"stat": "std", "window": 100}),
    ("rolling_volatility", {"stat": "range", "window": 50}),
    ("rolling_volatility", {"stat": "range", "window": 100}),
    ("tods", {"detector": "moving_average", "return_scores": True}),
    ("pyod", {"algorithm": "iforest"}),
    ("two_sided_volatility", {"window": 100}),
    ("two_sided_knn", {"n_neighbors": 5}),
    ("stuck", {"eps": 1e-3}),
]


def _acf1(s: np.ndarray) -> float:
    """First-order autocorrelation of a score series (a label-free intrinsic metric).

    Good scores for segment-level anomalies should be smooth and persistent (a
    volatility rise is a segment, not scattered points), so a high acf1 means the
    detector produces coherent anomaly regions. Its Spearman correlation with AUC is
    about +0.9 in experiments.
    """
    s = np.asarray(s, dtype=float)
    s = np.nan_to_num(s, nan=0.0)
    if np.std(s) < 1e-12:
        return 0.0
    r = rankdata(s) / len(s)
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def _runlen_top(s: np.ndarray, q: float = 0.99) -> int:
    """Longest consecutive run length at the top-q quantile (a label-free intrinsic metric measuring anomaly persistence)."""
    mask = s >= np.quantile(s, q)
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def _pareto_frontier(points: List[dict]) -> List[str]:
    """Non-dominated frontier on the (acf1, runlen) bi-objective (both larger-is-better)."""
    frontier = []
    for p in points:
        dominated = any(
            q["acf1"] >= p["acf1"]
            and q["runlen"] >= p["runlen"]
            and (q["acf1"] > p["acf1"] or q["runlen"] > p["runlen"])
            for q in points
        )
        if not dominated:
            frontier.append(p["name"])
    return frontier


def _is_mask(arr: np.ndarray) -> bool:
    """Decide whether an array is a boolean alarm mask or continuous scores."""
    if arr.dtype == bool:
        return True
    u = np.unique(arr)
    return u.size <= 2 and set(u.tolist()).issubset({0, 1, 0.0, 1.0, False, True})


def _digest_array(arr: np.ndarray) -> dict:
    """Compress an array into a summary (masks give alarm indices; scores give statistics and top indices)."""
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return {"n": 0}
    if _is_mask(arr):
        idx = np.where(arr.astype(bool))[0]
        return {
            "kind": "mask",
            "n": int(arr.size),
            "n_alarm": int(idx.size),
            "alarm_indices": idx[:50].tolist(),
        }
    s = arr.astype(float)
    top = int(min(10, s.size))
    top_idx = np.argsort(-s)[:top]
    return {
        "kind": "scores",
        "n": int(s.size),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "top_indices": top_idx.tolist(),
        "top_scores": [round(float(s[i]), 4) for i in top_idx],
    }


def _digest(data: Any) -> Any:
    """Generic recursive digestion: array/DataFrame/Series/dict/list/scalar."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return _digest_array(data)
    if isinstance(data, pd.DataFrame):
        col = data.iloc[:, 0].to_numpy() if data.shape[1] else np.array([])
        return _digest_array(col)
    if isinstance(data, pd.Series):
        return _digest_array(data.to_numpy())
    if isinstance(data, dict):
        return {str(k): _digest(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        if (
            data
            and isinstance(data[0], (list, tuple))
            and len(data[0]) == 2
            and all(isinstance(x, (int, np.integer)) for x in data[0])
        ):
            return [list(map(int, e)) for e in data]  # event intervals [(start, end), ...]
        return {"length": len(data), "head": list(data)[:20]}
    if isinstance(data, (str, int, float, bool, np.integer, np.floating)):
        return data
    return str(data)


class ToolExecutor:
    """Execute tool calls and maintain workspace state."""

    def __init__(
        self,
        test_data: np.ndarray,
        train_data: np.ndarray,
        y_test: np.ndarray,
        meta: Optional[dict] = None,
        raw_full: Optional[np.ndarray] = None,
        train_len: Optional[int] = None,
        y_full: Optional[np.ndarray] = None,
    ) -> None:
        self.test_data = np.asarray(test_data, dtype=float)
        self.train_data = np.asarray(train_data, dtype=float)
        self.y_test = np.asarray(y_test)
        self.meta = meta or {}
        # tsbad_model needs the full raw series (train+test) and the split point
        self.raw_full = np.asarray(raw_full, dtype=float) if raw_full is not None else None
        self.train_len = train_len
        self.y_full = np.asarray(y_full) if y_full is not None else None
        self.last_scores: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_events: List[list] = []
        self.score_history: List[dict] = []

    # ---- entry ----
    def call(self, name: str, args: dict) -> dict:
        """Execute one tool call and always return a JSON-serializable result dict."""
        try:
            # keep only legal parameters declared in the schema, to avoid the model stuffing in irrelevant keys (such as final JSON fields)
            allowed = _VALID_PARAMS.get(name, set())
            clean_args = {k: v for k, v in (args or {}).items() if k in allowed}
            return self._dispatch(name, clean_args)
        except Exception as exc:  # noqa: BLE001 -- any exception must not interrupt the main loop
            return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}

    def _dispatch(self, name: str, args: dict) -> dict:
        if name in _DETECTORS:
            return self._run_detector(name, args)
        if name == "dynamic_threshold":
            return self._run_dynamic_threshold(args)
        if name == "hysteresis_merge":
            return self._run_hysteresis(args)
        if name == "ensemble_fuse":
            return self._run_ensemble(args)
        if name == "search_toolchain":
            return self._run_search(args)
        if name in _EXPLAIN:
            return self._run_explain(name, args)
        if name == "query_tool":
            return self._run_query(args)
        known = sorted(set(_DETECTORS) | set(_POST) | set(_EXPLAIN) | {"query_tool", "ensemble_fuse", "search_toolchain"})
        return {"ok": False, "tool": name, "error": f"unknown function: {name}. callable: {known}"}

    # ---- detectors ----
    def _detector_scores(self, name: str, args: dict):
        """Run a detector and return (scores or None, ToolResult), for reuse by detection/ensemble/search."""
        module, tool_name = _DETECTORS[name]
        tool = catalog.get_tool(module, tool_name)
        if name == "tsbad_model":
            # TSB-AD models need the full raw series (train+test); without raw_full fall back to test_data
            if self.raw_full is None:
                return None, type("_R", (), {"ok": False, "error": "tsbad_model needs the full raw series raw_full (already passed by main.py)"})()
            args = {**args, "train_len": self.train_len}
            result = tool.run(self.raw_full, **args)
        else:
            result = tool.run(self.test_data, **args)
        if not result.ok:
            return None, result
        arr = np.asarray(result.data)
        if isinstance(result.data, pd.DataFrame):
            arr = result.data.to_numpy()
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr.max(axis=1)  # multivariate per-column scores: anomalous in any path counts as anomalous
        arr = arr.ravel()
        return arr, result

    def _run_detector(self, name: str, args: dict) -> dict:
        arr, result = self._detector_scores(name, args)
        if arr is None:
            return {"ok": False, "tool": name, "args": args, "error": result.error,
                    "provenance": result.provenance}
        if _is_mask(arr):
            self.last_mask = arr.astype(bool)
            self.last_scores = None
            digest = _digest_array(arr.astype(bool))
        else:
            scores = arr.astype(float)
            self.last_scores = scores
            self.last_mask = None
            entry = {"tool": name, "args": dict(args), "scores": scores}
            if name == "tsbad_model":
                entry["official"] = True  # evaluate on the full series using the official get_metrics
            self.score_history.append(entry)
            digest = _digest_array(scores)
        return {"ok": True, "tool": name, "args": args, **digest,
                "provenance": result.provenance}

    # ---- multi-detector ensemble (rank fusion) ----
    def _run_ensemble(self, args: dict) -> dict:
        method = args.get("method", "rank_max")
        if method not in ("rank_max", "rank_mean"):
            return {"ok": False, "tool": "ensemble_fuse", "args": args,
                    "error": "method only supports rank_max / rank_mean"}
        ranks, members = [], []
        for tool_name, params in _ENSEMBLE_PANEL:
            arr, result = self._detector_scores(tool_name, params)
            if arr is None:
                continue
            s = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
            ranks.append(rankdata(s) / len(s))
            members.append(f"{tool_name}({params})")
        if not ranks:
            return {"ok": False, "tool": "ensemble_fuse", "args": args, "error": "no usable detector in the fusion panel"}
        fused = np.max(ranks, axis=0) if method == "rank_max" else np.mean(ranks, axis=0)
        self.last_scores = fused
        self.last_mask = None
        self.score_history.append({"tool": "ensemble_fuse", "args": dict(args), "scores": fused})
        return {
            "ok": True, "tool": "ensemble_fuse", "args": args,
            "members": members, "method": method, **_digest_array(fused),
        }

    # ---- label-free toolchain search: grid + Pareto frontier ----
    def _run_search(self, args: dict) -> dict:
        points = []
        for tool_name, params in _CANDIDATES:
            arr, result = self._detector_scores(tool_name, params)
            if arr is None:
                points.append({"name": f"{tool_name}({params})", "acf1": None, "runlen": None, "error": result.error})
                continue
            s = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
            points.append({
                "name": f"{tool_name}({json.dumps(params, ensure_ascii=False)})",
                "acf1": round(_acf1(s), 4),
                "runlen": _runlen_top(s),
            })
        valid = [p for p in points if p["acf1"] is not None]
        ranked = sorted(valid, key=lambda p: (-p["acf1"], -p["runlen"]))
        frontier = _pareto_frontier(valid)
        rec = ranked[0] if ranked else None
        return {
            "ok": True, "tool": "search_toolchain", "args": args,
            "criterion": "acf1(score persistence)+runlen(top1% segment length), label-free; acf1 correlates with AUC ≈ +0.9",
            "pareto_frontier": frontier,
            "recommended": rec,
            "ranked": ranked,
        }

    # ---- post-processing ----
    def _run_dynamic_threshold(self, args: dict) -> dict:
        module, tool_name = _POST["dynamic_threshold"]
        tool = catalog.get_tool(module, tool_name)
        if self.last_scores is None:
            return {"ok": False, "tool": "dynamic_threshold", "args": args,
                    "error": "no continuous scores yet; please first run a score-based detector (e.g. pyod/tods return_scores)"}
        result = tool.run(self.last_scores, **args)  # FitApplyTool.run = fit+apply
        if not result.ok:
            return {"ok": False, "tool": "dynamic_threshold", "args": args,
                    "error": result.error, "provenance": result.provenance}
        data = result.data
        alarm = data.get("alarm") if isinstance(data, dict) else data
        if isinstance(alarm, pd.DataFrame):
            alarm = alarm.iloc[:, 0].to_numpy()
        mask = np.asarray(alarm, dtype=bool).ravel()
        self.last_mask = mask
        threshold = data.get("threshold") if isinstance(data, dict) else None
        return {
            "ok": True, "tool": "dynamic_threshold", "args": args,
            "n_alarm": int(mask.sum()),
            "alarm_indices": np.where(mask)[0][:50].tolist(),
            "threshold": _digest(threshold),
            "provenance": result.provenance,
        }

    def _run_hysteresis(self, args: dict) -> dict:
        module, tool_name = _POST["hysteresis_merge"]
        tool = catalog.get_tool(module, tool_name)
        if self.last_mask is not None:
            signal = self.last_mask.astype(float)
            merged = {"high": 1.0, "low": 0.0, "min_duration": 1, "merge_gap": 0, **args}
        elif self.last_scores is not None:
            # without a mask, auto-binarize with the 95th-percentile score (degraded) and attach the alarm
            s = self.last_scores
            mask = s >= np.quantile(s, 0.95)
            self.last_mask = mask
            signal = mask.astype(float)
            merged = {"high": 1.0, "low": 0.0, "min_duration": 1, "merge_gap": 0, **args}
        else:
            return {"ok": False, "tool": "hysteresis_merge", "args": args,
                    "error": "no detection result yet; please first run a detector"}
        result = tool.run(signal, **merged)
        if not result.ok:
            return {"ok": False, "tool": "hysteresis_merge", "args": merged,
                    "error": result.error, "provenance": result.provenance}
        events = result.data
        if isinstance(events, dict):
            events = list(events.values())[0] if events else []
        self.last_events = [list(map(int, e)) for e in events]
        return {
            "ok": True, "tool": "hysteresis_merge", "args": merged,
            "n_events": len(self.last_events), "events": self.last_events,
            "provenance": result.provenance,
        }

    # ---- explanation (multivariate attribution; univariate degrades to single-channel) ----
    def _run_explain(self, name: str, args: dict) -> dict:
        module, tool_name = _EXPLAIN[name]
        tool = catalog.get_tool(module, tool_name)
        result = tool.run(self.test_data.reshape(-1, 1), **args)
        if not result.ok:
            return {"ok": False, "tool": name, "args": args, "error": result.error,
                    "provenance": result.provenance}
        return {
            "ok": True, "tool": name, "args": args, "result": _digest(result.data),
            "note": "univariate series has only one channel, attribution degrades",
            "provenance": result.provenance,
        }

    # ---- knowledge lookup ----
    def _run_query(self, args: dict) -> dict:
        name = str(args.get("name", "")).strip()
        if not name:
            return {"ok": False, "tool": "query_tool", "error": "missing name parameter"}
        row = catalog.metadata(name)
        if row is None:
            hint = [r["name"] for r in catalog.list_all() if name.lower() in r["name"].lower()][:10]
            return {"ok": False, "tool": "query_tool", "args": args,
                    "error": f"tool not found: {name}", "suggestions": hint}
        return {"ok": True, "tool": "query_tool", "args": args, "result": row}
