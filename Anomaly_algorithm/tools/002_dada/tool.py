"""DADA general time-series anomaly detection model."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class DadaTool(ReferenceTool):
    name = "dada"
    category = "Zero-shot & General Models"
    version = "1.0.0"
    description = "DADA multi-domain pretrained general anomaly detection model."
    libraries = ()
    origin = "https://github.com/iambowen/DADA"

    metadata = {
        "name": "DADA",
        "url": "https://github.com/iambowen/DADA",
        "type": "general time-series anomaly detection model (multi-domain pretrained)",
        "description": (
            "A general anomaly detection model based on multi-domain time-series pretraining, "
            "aimed at applying to multiple scenarios without target-domain-specific training."
        ),
        "usage_notes": (
            "Performance is affected by the distribution gap between pretraining data and the target domain; "
            "the official repository states that the pretraining code is not released due to IP restrictions, "
            "and the public repository does not include the full pretraining pipeline."
        ),
    }


TOOL = DadaTool()
