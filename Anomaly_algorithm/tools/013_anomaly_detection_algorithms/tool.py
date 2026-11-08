"""Anomaly detection algorithm and implementation index (eug/anomaly-detection-algorithms)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class AnomalyDetectionAlgorithmsTool(ReferenceTool):
    name = "anomaly_detection_algorithms"
    category = "Algorithm & Paper Index"
    version = "1.0.0"
    description = "An index of anomaly detection algorithms and papers categorized by statistical/distance/density/ensemble/spatial/subspace."
    libraries = ()
    origin = "https://github.com/eug/anomaly-detection-algorithms"

    metadata = {
        "name": "anomaly-detection-algorithms",
        "url": "https://github.com/eug/anomaly-detection-algorithms",
        "type": "algorithm & paper index",
        "description": (
            "Organizes anomaly detection algorithms by categories such as statistical, distance, density, "
            "ensemble, spatial and subspace, and collects related papers or code implementations."
        ),
        "usage_notes": "Suitable as an algorithm lookup entry point; it is not a unified software package or benchmarking framework.",
    }


TOOL = AnomalyDetectionAlgorithmsTool()
