"""TSB-AD strong-model tool: wraps models in the TSB-AD models directory as schedulable tools.

Keeps model performance unchanged: uses the optimal hyperparameters from HP_list + the official
run_<Model> wrappers + seed=2024, and outputs pointwise anomaly scores. ``data`` receives the **raw full
sequence** (train segment + test segment), and ``train_len`` is the split point (needed by semi-supervised
models, parsed from the tr_N filename). Time_RCD uses a local checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# make the TSB_AD package + hurst stub importable
_TSBAD_ROOT = Path(__file__).resolve().parents[3] / "TSB-AD"
for _p in (str(_TSBAD_ROOT), str(_TSBAD_ROOT / "stubs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Data_preprocessing.core.base import BaseTool, ToolResult, require  # noqa: E402


# TSB-AD models (verified runnable strong models + common classical/deep models)
TSBAD_MODELS = [
    "Syntropic", "Time_RCD", "OmniAnomaly", "StreamVAE", "Sub_PCA", "IForest",
    "LOF", "KNN", "HBOS", "COPOD", "TranAD", "USAD", "AE", "CNN", "LSTMAD",
    "TimesNet", "FITS", "PatchTST", "AnomalyTransformer", "M2N2", "OFA", "Donut",
]


class TSBADModelTool(BaseTool):
    name = "tsbad_model"
    category = "Anomaly Detection"
    version = "1.0.0"
    description = (
        "TSB-AD model-zoo detector: runs models from TSB-AD with optimal hyperparameters "
        "(Syntropic/Time_RCD/OmniAnomaly/StreamVAE/Sub_PCA/IForest/LOF/...), outputting pointwise "
        "anomaly scores. data receives the raw full sequence (train+test); train_len receives the split point."
    )
    libraries = ("numpy", "pandas", "torch", "scikit-learn")
    origin = "TSB-AD benchmark model zoo (official run_<Model> wrappers + optimal hyperparameters)"

    def run(self, data, model="Syntropic", train_len=None, **kwargs) -> ToolResult:
        params = {"model": model, "train_len": train_len, **kwargs}
        try:
            require(["numpy", "pandas", "torch"])
            import torch
            import random

            from TSB_AD.model_wrapper import (run_Semisupervise_AD, run_Unsupervise_AD,
                                              Unsupervise_AD_Pool, Semisupervise_AD_Pool)
            from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

            # align the seed with the official protocol
            seed = 2024
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            np.random.seed(seed); random.seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            x = np.asarray(data, dtype=float)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            n = len(x)
            if train_len is None or train_len >= n:
                train_len = max(1, int(n * 0.5))
            train = x[: int(train_len), :]

            multi = x.shape[1] > 1
            HP = Optimal_Multi_algo_HP_dict[model] if multi else Optimal_Uni_algo_HP_dict[model]
            HP = dict(HP)
            if model == "Time_RCD":
                HP["checkpoint"] = str(_TSBAD_ROOT.parent / "Time-RCD/best_model")  # directory; auto-selects uni/multi

            if model in Semisupervise_AD_Pool:
                out = run_Semisupervise_AD(model, train, x, **HP)
            elif model in Unsupervise_AD_Pool:
                out = run_Unsupervise_AD(model, x, **HP)
            else:
                return self.fail(ValueError(f"Unknown model: {model}"), params)

            score = np.nan_to_num(np.asarray(out, dtype=float).ravel(), nan=0.0)
            return self.ok(score, params)
        except Exception as exc:
            return self.fail(exc, params)


TOOL = TSBADModelTool()
