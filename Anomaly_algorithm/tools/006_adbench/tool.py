"""ADBench algorithm and dataset benchmark."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class AdbenchTool(ReferenceTool):
    name = "adbench"
    category = "Benchmarks & Datasets"
    version = "1.0.0"
    description = "ADBench: anomaly detection benchmark with 30 algorithms and 57 datasets, suitable for horizontal comparison."
    libraries = ()
    origin = "https://github.com/Minqi824/ADBench"

    metadata = {
        "name": "ADBench",
        "url": "https://github.com/Minqi824/ADBench",
        "type": "anomaly detection benchmark (mainly tabular)",
        "description": "Contains 30 algorithms and 57 datasets, suitable for horizontal comparison.",
        "usage_notes": (
            "Primarily a tabular anomaly detection benchmark; it does not directly represent zero-shot "
            "time-series anomaly detection performance."
        ),
    }


TOOL = AdbenchTool()
