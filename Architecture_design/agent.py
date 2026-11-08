"""ReAct agent main loop: Ollama LLM decides -> execute tool -> return result -> until a conclusion is reached.

The flow is split into two stages so a local 8B model does not confuse "calling
tools" with "emitting JSON":

* **acting stage**: the system prompt only says "call tools to gather evidence"
  and never mentions the final JSON format; the model calls exactly one tool per
  round and replies "done" when the evidence is sufficient.
* **finalize stage**: after receiving the done signal, a new message **without
  tools** hands the tool-result summary and the JSON contract to the model so it
  emits only the final JSON.

The LLM plays the role of a "scheduling decision-maker": it decides which detector
to call, with what parameters, how to set thresholds, and how to organize events.
The entire decision trajectory is recorded in ``trace`` and is auditable and
reproducible.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from . import catalog, config, schemas
from .client import OllamaClient
from .executor import ToolExecutor

_ACTING_SYSTEM = """You are an industrial time-series anomaly detection agent. Task: **schedule tools** to detect anomalies on the test segment of the given univariate time series. Always respond in Chinese.

Rules:
1. Judge only from the results returned by tools; never fabricate values.
2. Call exactly one tool per round, and decide the next step from the previous step's result.
3. Standard flow:
   ① First call search_toolchain to do a label-free toolchain search (Pareto front to pick the best detector);
   ② Call the recommended detector as suggested (e.g. rolling_volatility with std/range) to obtain continuous scores;
   ③ Then call dynamic_threshold (method=global, quantile=0.99) to turn scores into an alarm mask;
   ④ If debouncing is needed, call hysteresis_merge; for more stability, switch to ensemble_fuse for multi-detector fusion.
4. When you need methodology/reference, call query_tool to look up tool metadata.
5. When evidence is sufficient, reply with only the word: done (do not output JSON, do not summarize, do not explain).

Note: adtk directly gives boolean alarms; dynamic_threshold / hysteresis_merge need continuous scores first.

Overview of registered tools (use query_tool for details):
{catalog}"""

_FINALIZE_SYSTEM = """You are an industrial time-series anomaly detection agent. Based on the tool results below, output the final JSON conclusion.
Output only this single JSON object, without any other text, explanation, or code fence:
{{"toolchain": ["tool:param", ...], "summary": "one-sentence conclusion", "detected_events": [[start,end], ...], "anomaly_ratio": 0.0, "confidence": "high/medium/low", "reasoning": "basis"}}
where start/end are 0-based closed-interval indices relative to the test segment start."""


def _user_prompt(task: str, meta: Dict[str, Any], stats: Dict[str, Any]) -> str:
    return (
        f"Task: {task}\n\n"
        "Series information:\n"
        f"- Data source: {meta.get('src', 'unknown')}, id={meta.get('id', '?')}\n"
        f"- Train segment length: {meta.get('train_len')}, first anomaly relative to test segment start: {meta.get('first_anomaly')}\n"
        f"- Test segment (standardized against the train segment): {json.dumps(stats, ensure_ascii=False)}\n\n"
        "Please schedule tools following the flow to complete detection. When evidence is sufficient, reply: done."
    )


def _to_ollama_tool_calls(tool_calls: List[dict]) -> List[dict]:
    """Restore normalized tool_calls into the format Ollama assistant messages expect."""
    return [
        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
        for tc in tool_calls
    ]


def _parse_final(text: str) -> Dict[str, Any]:
    """Loosely parse the JSON block in the final conclusion (from the first { to the last })."""
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"raw": text}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"raw": text}


def _evidence_summary(trace: List[dict], executor: ToolExecutor) -> str:
    """Compress the decision trajectory + workspace state into a text block for the finalize stage to cite."""
    lines: List[str] = []
    for i, step in enumerate(trace, 1):
        d, r = step["decision"], step["result"]
        a = json.dumps(d["args"], ensure_ascii=False, default=str)
        if not r.get("ok"):
            lines.append(f"{i}. {d['tool']}({a}) failed: {r.get('error')}")
        elif r.get("kind") == "scores":
            lines.append(f"{i}. {d['tool']}({a}) → continuous scores n={r['n']} mean={r['mean']} max={r['max']}")
        elif r.get("kind") == "mask":
            lines.append(f"{i}. {d['tool']}({a}) → alarm {r['n_alarm']} points")
        elif "n_events" in r:
            lines.append(f"{i}. {d['tool']}({a}) → {r['n_events']} events: {r['events']}")
        else:
            lines.append(f"{i}. {d['tool']}({a}) → {json.dumps(r, ensure_ascii=False, default=str)[:200]}")
    if executor.last_mask is not None:
        idx = np.where(executor.last_mask)[0]
        lines.append(f"alarm mask: {int(idx.size)} points, indices {idx[:30].tolist()}")
    if executor.last_events:
        lines.append(f"event intervals: {executor.last_events}")
    return "\n".join(lines) or "(no tool results)"


class IndustrialAgent:
    """Diagnostic-layer agent whose tool scheduling decisions are made by a local LLM."""

    def __init__(
        self,
        executor: ToolExecutor,
        client: Optional[OllamaClient] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        self.executor = executor
        self.client = client or OllamaClient()
        self.max_iterations = max_iterations or config.AGENT_MAX_ITERATIONS
        self.trace: List[dict] = []

    def run(
        self,
        task: str,
        meta: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agent and return final / trace / iterations / parsed."""
        self.trace = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _ACTING_SYSTEM.format(catalog=catalog.catalog_brief())},
            {"role": "user", "content": _user_prompt(task, meta or {}, stats or {})},
        ]

        # ---- acting stage: model decides -> execute tool -> return result ----
        for _ in range(self.max_iterations):
            reply = self.client.chat(messages, tools=schemas.FUNCTIONS)
            if reply["tool_calls"]:
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply["content"],
                        "tool_calls": _to_ollama_tool_calls(reply["tool_calls"]),
                    }
                )
                for tc in reply["tool_calls"]:
                    result = self.executor.call(tc["name"], tc["arguments"])
                    self.trace.append(
                        {"decision": {"tool": tc["name"], "args": tc["arguments"]}, "result": result}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                            "tool_name": tc["name"],
                        }
                    )
                continue
            # The model wants to finish: guardrail -- refuse to accept it if no detector has been run yet, forcing it to call a tool first
            if not self.executor.score_history and self.executor.last_mask is None:
                messages.append({"role": "assistant", "content": reply["content"]})
                messages.append(
                    {
                        "role": "user",
                        "content": "You have not called any detection tool yet. Please first call a score-based detector (e.g. pyod), "
                        "then continue after getting real results; do not fabricate a conclusion.",
                    }
                )
                continue
            break  # evidence is enough, enter finalize

        # ---- finalize stage: no tools, emit only the final JSON ----
        finalize_messages = [
            {"role": "system", "content": _FINALIZE_SYSTEM},
            {
                "role": "user",
                "content": "Tool result summary:\n" + _evidence_summary(self.trace, self.executor)
                + "\n\nPlease output the final JSON conclusion.",
            },
        ]
        final = self.client.chat(finalize_messages)["content"]

        return {
            "final": final,
            "trace": self.trace,
            "iterations": len(self.trace),
            "parsed": _parse_final(final or ""),
        }
