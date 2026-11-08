"""Anomaly detection research resource collection (yzhao062)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class AnomalyDetectionResourcesTool(ReferenceTool):
    name = "anomaly_detection_resources"
    category = "Algorithm & Paper Index"
    version = "1.0.0"
    description = "An index aggregating anomaly detection books/papers/courses/datasets/tools/conferences."
    libraries = ()
    origin = "https://github.com/yzhao062/anomaly-detection-resources"

    metadata = {
        "name": "anomaly-detection-resources",
        "url": "https://github.com/yzhao062/anomaly-detection-resources",
        "type": "resource collection / index",
        "description": (
            "Aggregates anomaly detection books, papers, course videos, datasets, open-source and "
            "commercial tool libraries, conferences and journal resources, covering topics such as "
            "time series, graph anomaly detection, LLM and VLM."
        ),
        "usage_notes": "LLM/VLM-related content is updated through late 2025; it is a literature and resource index, not a directly runnable unified algorithm library.",
    }


TOOL = AnomalyDetectionResourcesTool()
