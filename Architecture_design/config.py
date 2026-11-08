"""Agent runtime configuration: model, server address, budget, etc., all
overridable via environment variables.

Example::

    OLLAMA_MODEL=llama3.1:8b python Architecture_design/main.py --file 001_...
"""

from __future__ import annotations

import os

# Ollama server address (native API, localhost by default).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Default model: llama3.1:8b has stable native tool calling; qwen3-coder-30b-a3b is more accurate but slower.
# Note: qwen2.5-coder:7b's Ollama template is incomplete and writes tool calls into content (the client already parses it as a fallback).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Timeout (seconds) for a single /api/chat request. Heavy tools such as time_rcd may be slow on first load.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))

# Sampling temperature: 0 makes decisions more deterministic and reproducible.
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.0"))

# Maximum rounds of the agent's ReAct loop (each "model -> tool -> result" counts as one round).
AGENT_MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "8"))
