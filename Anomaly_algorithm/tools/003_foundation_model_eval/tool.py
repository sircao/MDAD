"""Systematic evaluation of foundation models for time-series anomaly detection (ICLR 2026 paper)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class FoundationModelEvalTool(ReferenceTool):
    name = "foundation_model_eval"
    category = "Zero-shot & General Models"
    version = "1.0.0"
    description = "Systematically evaluates TSAD performance of MOMENT/Chronos/TimesFM/Time-MoE/TSPulse and compares against simple baselines."
    libraries = ()
    origin = "https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html"

    metadata = {
        "name": "Systematic Evaluation of Foundation Models for Time-Series Anomaly Detection (ICLR 2026)",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html",
        "type": "paper / evaluation study",
        "description": (
            "Systematically evaluates the time-series anomaly detection performance of models such as "
            "MOMENT, Chronos, TimesFM, Time-MoE and TSPulse, and compares them against simple baselines "
            "such as moving-window variance and squared deviation."
        ),
        "usage_notes": (
            "Anomaly detection ability cannot be inferred from a foundation model's prediction or reconstruction "
            "ability alone; this conclusion applies to the models and experimental settings tested in the paper "
            "and should not be generalized to mean that all foundation models are ineffective."
        ),
    }


TOOL = FoundationModelEvalTool()
