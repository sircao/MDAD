"""General anomaly detection resource collection (awesome-anomaly-detection)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class AwesomeAnomalyDetectionTool(ReferenceTool):
    name = "awesome_anomaly_detection"
    category = "Algorithm & Paper Index"
    version = "1.0.0"
    description = "An index of algorithms, papers, datasets, benchmarks, tool libraries and surveys for general anomaly detection."
    libraries = ()
    origin = "https://github.com/hoya012/awesome-anomaly-detection"

    metadata = {
        "name": "awesome-anomaly-detection",
        "url": "https://github.com/hoya012/awesome-anomaly-detection",
        "type": "resource collection / index",
        "description": "Organizes algorithms, papers, datasets, benchmarks, tool libraries and surveys for general anomaly detection.",
        "usage_notes": "Suitable for retrieving methods and experimental resources in anomaly detection; the repository content is updated as it is maintained.",
    }


TOOL = AwesomeAnomalyDetectionTool()
