"""MTS dataset synthesis — series synthesis and anomaly injection."""

from __future__ import annotations

from pathlib import Path

from Data_preprocessing.core.base import BaseTool, ToolResult
from Data_preprocessing.core.origins import DATA_ROOT, MTS_SRC, ensure

_DEFAULT_LEGACY = DATA_ROOT / "Multi-AXIS-B1DC" / "MTS_Data" / "third_party" / "datasets_rcd"


class GenerateSeriesTool(BaseTool):
    name = "mts_generation.generate_series"
    category = "MTS dataset synthesis"
    version = "1.0.0"
    description = "Calls the underlying generator to synthesize multivariate series and inject anomalies with controllable ground-truth labels."
    libraries = ("numpy", "networkx", "scipy", "statsmodels")
    origin = "MTS_Data/src/mvaxis/legacy_tsad_generator.py:96 load_legacy_tsad_generate_dataset() + third_party/datasets_rcd/src/generate_dataset.py:274 generate_dataset()"

    def run(
        self,
        num_samples=100,
        seq_len=180,
        num_features=10,
        anomaly_sample_ratio=1.0,
        activate_function=False,
        use_attribute_set=True,
        num_workers=1,
        legacy_src=None,
    ) -> ToolResult:
        params = {
            "num_samples": num_samples, "seq_len": seq_len, "num_features": num_features,
            "anomaly_sample_ratio": anomaly_sample_ratio, "activate_function": activate_function,
            "use_attribute_set": use_attribute_set, "num_workers": num_workers,
        }
        try:
            ensure(MTS_SRC)
            from mvaxis.legacy_tsad_generator import load_legacy_tsad_generate_dataset

            src = Path(legacy_src) if legacy_src else _DEFAULT_LEGACY
            generate = load_legacy_tsad_generate_dataset(src)
            rows = generate(
                num_samples=int(num_samples),
                seq_len=int(seq_len),
                anomaly_sample_ratio=float(anomaly_sample_ratio),
                is_multivariate=True,
                num_features=int(num_features),
                activate_function=bool(activate_function),
                use_attribute_set=bool(use_attribute_set),
                num_workers=int(num_workers),
            )
            return self.ok(rows, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = GenerateSeriesTool()
