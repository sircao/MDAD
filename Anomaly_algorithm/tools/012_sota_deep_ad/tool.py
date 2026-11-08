"""Collection of deep anomaly detection method implementations (SOTA-Deep-Anomaly-Detection)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class SotaDeepAdTool(ReferenceTool):
    name = "sota_deep_ad"
    category = "Algorithm & Paper Index"
    version = "1.0.0"
    description = "Indexes deep anomaly detection methods and code links from papers, tagging venue/year/framework/supervision/data type."
    libraries = ()
    origin = "https://github.com/GuansongPang/SOTA-Deep-Anomaly-Detection"

    metadata = {
        "name": "SOTA-Deep-Anomaly-Detection",
        "url": "https://github.com/GuansongPang/SOTA-Deep-Anomaly-Detection",
        "type": "algorithm implementation index",
        "description": (
            "Indexes deep anomaly detection methods from papers and their code links, tagging publication venue, "
            "year, implementation framework, supervision type and data type, covering images, video, tabular, "
            "graph, sequence and time series (e.g. OmniAnomaly)."
        ),
        "usage_notes": "Primarily an algorithm implementation index; each method's code is maintained by different projects, so dependencies, licenses and code status should be checked individually before use.",
    }


TOOL = SotaDeepAdTool()
