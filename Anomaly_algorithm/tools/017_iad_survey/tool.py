"""Industrial anomaly detection survey resource (IAD-Survey)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class IadSurveyTool(ReferenceTool):
    name = "iad_survey"
    category = "Algorithm & Paper Index"
    version = "1.0.0"
    description = "Industrial anomaly detection survey resource: aggregates related papers and research information."
    libraries = ()
    origin = "https://github.com/IHPCRits/IAD-Survey"

    metadata = {
        "name": "IAD-Survey",
        "url": "https://github.com/IHPCRits/IAD-Survey",
        "type": "survey / resource index",
        "description": "Aggregates papers and research information related to industrial anomaly detection.",
        "usage_notes": (
            "When citing specific numbers such as \"the survey covers 196 papers\" or \"the number of distinct "
            "algorithms\", refer to the repository's current README, the paper version and the counting criteria, "
            "and avoid equating the paper count directly with the number of distinct algorithms."
        ),
    }


TOOL = IadSurveyTool()
