"""Industrial time-series anomaly detection agent (diagnostic layer).

A local Ollama LLM makes the tool-scheduling decisions, reusing tools already
registered in the project's four algorithm modules (data preprocessing /
anomaly detection / explainability / root-cause localization) to run end-to-end
anomaly detection on TSB-AD-U univariate time series. See README.md.
"""

from __future__ import annotations
