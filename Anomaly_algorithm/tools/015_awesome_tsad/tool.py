"""Time-series anomaly detection resource collection (awesome-TSAD)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class AwesomeTsadTool(ReferenceTool):
    name = "awesome_tsad"
    category = "Benchmarks & Datasets"
    version = "1.0.0"
    description = "An index of TSAD algorithms, papers, datasets, benchmarks, tool libraries and surveys (including TSB-AD/TSB-UAD)."
    libraries = ()
    origin = "https://github.com/yanexr/awesome-TSAD"

    metadata = {
        "name": "awesome-TSAD",
        "url": "https://github.com/yanexr/awesome-TSAD",
        "type": "resource collection / index",
        "description": (
            "Dedicated to organizing time-series anomaly detection algorithms, papers, datasets, benchmarks, "
            "tool libraries and surveys; further time-series evaluation resources such as TSB-AD and TSB-UAD "
            "can be found here."
        ),
        "usage_notes": (
            "When comparing zero-shot models, verify that the benchmark controls for factors such as pretraining "
            "data overlap, target-domain training and threshold calibration."
        ),
    }


TOOL = AwesomeTsadTool()
