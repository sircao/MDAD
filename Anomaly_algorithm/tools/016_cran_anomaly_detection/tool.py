"""R language anomaly detection package index (CRAN)."""

from __future__ import annotations

from Anomaly_algorithm.core import ReferenceTool


class CranAnomalyDetectionTool(ReferenceTool):
    name = "cran_anomaly_detection"
    category = "Algorithm & Paper Index"
    version = "1.0.0"
    description = "CRAN anomaly detection task view: an index of R language packages (statistical/density/distance/clustering/time-series/spatial)."
    libraries = ()
    origin = "https://cran.r-project.org/web/views/AnomalyDetection.html"

    metadata = {
        "name": "CRAN AnomalyDetection Task View",
        "url": "https://cran.r-project.org/web/views/AnomalyDetection.html",
        "type": "package index (R language)",
        "description": (
            "CRAN's anomaly detection task view, organizing R language packages covering statistical, density, "
            "distance, clustering, time-series and spatial data directions."
        ),
        "usage_notes": "It is a package index, not a single algorithm implementation.",
    }


TOOL = CranAnomalyDetectionTool()
