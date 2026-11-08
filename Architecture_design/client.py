"""Ollama client: uses requests to talk directly to the native ``/api/chat``,
with native tool calling support.

No dependency on the third-party ollama / openai packages, only requests. The
returned message is normalized to
``{"content": str, "tool_calls": [{"name": str, "arguments": dict}]}``, where
arguments is always parsed into a dict (handling both cases where Ollama returns
a dict or a JSON string -- different versions behave inconsistently).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests

from . import config


class OllamaError(RuntimeError):
    """Ollama call failure."""


def _extract_tool_calls_from_content(content: str) -> List[dict]:
    """Fallback: parse tool calls out of JSON text when the model does not emit
    native tool_calls but instead writes the tool call as JSON text.

    Supports ``{"name": ..., "arguments": {...}}``, ``{"function": {...}}`` and
    the array form, tolerating markdown code fences and surrounding prefix/suffix
    text.
    """
    if not content:
        return []
    text = re.sub(r"```(?:json)?", "", content).strip()
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]|\{.*\}", text, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        fn = it.get("function") if isinstance(it.get("function"), dict) else it
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            out.append(
                {"name": str(name), "arguments": _parse_arguments(fn.get("arguments"))}
            )
    return out


def _parse_arguments(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {"_raw": str(raw)}


class OllamaClient:
    """Lightweight client that talks directly to the native Ollama API."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.model = model or config.OLLAMA_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout if timeout is not None else config.OLLAMA_TIMEOUT
        self.temperature = (
            temperature if temperature is not None else config.OLLAMA_TEMPERATURE
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Send a conversation and normalize the reply.

        Parameters
        ----------
        messages : OpenAI-compatible message list (role=system/user/assistant/tool).
        tools : list of native function schemas; empty means tool calling is disabled.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama call failed ({self.host}): {exc}") from exc

        message = resp.json().get("message") or {}
        content = message.get("content") or ""
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                {"name": fn.get("name", ""), "arguments": _parse_arguments(fn.get("arguments"))}
            )
        if not tool_calls:
            tool_calls = _extract_tool_calls_from_content(content)
            if tool_calls:
                content = ""  # tool calls have already been parsed out; avoid treating them as body text again
        return {"content": content, "tool_calls": tool_calls}
