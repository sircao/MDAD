# Data_preprocessing — a time series preprocessing toolset for data-centric architectures

Breaks industrial time-series preprocessing algorithms into **53 general-purpose tools + 33 pipeline tools (86 in total)**, each in its own folder. Every tool returns through `ToolResult` with provenance (tool version, dependency library versions, call parameters, runtime, and the `origin` source), serving the "data layer" of a unified platform: ingestion, cleaning, alignment, transformation, and feature construction, and can be dispatched directly by an Agent against a whitelist. The pipeline tools directly reuse the original project modules of three existing pipelines (RS-TOUTER / Multi-AXIS / MTS dataset synthesis).

## Directory structure

```
Data_preprocessing/
├── core/            # common base classes, result/provenance structures, registry
│   ├── base.py      #   BaseTool / FitApplyTool / ToolResult / require / dist_version
│   ├── utils.py     #   to_frame / to_1d input normalization
│   └── registry.py  #   discover / get / list_tools / by_category / availability
├── registry.py      # convenience entry point (re-exports core.registry)
├── tools/           # 53 general-purpose tools, one folder each (NNN_name/tool.py)
├── pipelines/       # 33 pipeline tools (reuse original project modules; each has tool.py + README.md)
│   ├── rs_router/       # 16 preprocessing stages of RS-TOUTER (TS-Router-D8FF)
│   ├── multi_axis/      # 10 preprocessing stages of Multi-AXIS (Multi-AXIS-B1DC)
│   └── mts_generation/  # 7 stages for dataset synthesis and QA sample generation (MTS_Data)
├── requirements-optional.txt   # optional dependencies (install on demand)
├── smoke_test.py    # smoke test for general-purpose tools
└── pipelines_smoke_test.py   # smoke test for pipeline tools
```

## Usage

```python
from Data_preprocessing.registry import get, list_tools

# List all tools (for the Agent to read: name/category/dependencies/installed)
for row in list_tools():
    print(row["name"], row["category"], row["installed"])

# Call a tool
tool = get("interpolate_time")
result = tool.run(df, method="linear")
assert result.ok
out_df = result.data          # result data
audit = result.to_dict()      # serializable audit info without data
```

## Unified contract

- **Unified return**: every tool's `run()` returns `ToolResult` with `ok / data / error /
  warnings / provenance`. The Agent only needs to branch on `result.ok`.
- **Provenance**: `provenance` records the tool name, tool version, dependency library
  versions, call parameters, and runtime, satisfying the requirement that diagnostics be
  replayable and auditable.
- **Leak prevention (stateful tools)**: transforms that need fitted parameters (scaling,
  binning, dimensionality reduction, imputation models, thresholds, etc.) inherit
  `FitApplyTool` and must first `fit()` on the training set, then `apply()` to the test
  set. `run()` is a one-shot fit+apply for quick trials only and attaches a note. This
  corresponds to the platform constraint "scale/threshold/features are fitted only on the
  training or validation portion, to avoid information leakage".
- **Lazy dependency loading**: tools import their dependencies lazily. When a library is
  missing, calls return `ok=False` with install hints without affecting other tools.
  `availability()` summarizes missing libraries.

## Tool list (by category)

| Category | Tools |
|---|---|
| Time Series Processing and Feature Extraction | `tsflex_feature` `sktime_transform` `tsai_augment` `tsfresh_extract` `tsfel_extract` |
| Data Cleaning and Data Quality | `validate_constraints` `deduplicate` `detect_stuck` `drift_correct` `denoise` |
| Missing Value Handling | `fill_nearest` `interpolate_time` `knn_impute` `iterative_impute` `kalman_impute` `seasonal_impute` |
| Sampling, Windowing, and Data Volume Control | `resample_aggregate` `resample_balance` `stratified_sample` `window_partition` `changepoint_segment` |
| Transformation and Distribution Correction | `difference` `power_transform` `detrend` `quantile_transform` `discretize` `cast_type` |
| Filtering, Decomposition, and Time-Frequency Analysis | `savgol_filter` `kalman_smooth` `wavelet_denoise` `filterbank` `emd_decompose` `vmd_decompose` |
| Feature Extraction and Feature Representation | `peak_features` `spectral_features` `entropy_features` `autocorr_features` `dtw_distance` `shapelet_distance` `matrix_profile` `auto_feature_extract` |
| Multivariate Fusion, Encoding, and Dimensionality Reduction | `rolling_correlation` `interaction_features` `conditional_residual` `multisensor_fuse` `calendar_encode` `pca_ica` `feature_select` `random_projection` `series_compress` |
| Data Augmentation, Sampling, and Post-processing | `series_augment` `dynamic_threshold` `hysteresis_merge` |

## Dependencies

Core dependencies (`pandas / numpy / scipy / scikit-learn / statsmodels`) ship with the
project. Install the rest on demand:

```bash
pip install -r Data_preprocessing/requirements-optional.txt
```

For tools whose library is not installed (e.g. `tsfresh_extract`, `wavelet_denoise`,
`emd_decompose`), calls return a clear install hint; several of those tools
(peak/spectral/entropy/DTW/Matrix Profile/PAA-SAX/augmentation) are implemented natively
with numpy/scipy and run without installing any library.

## Pipeline tools (pipelines/)

Each preprocessing stage of the three existing pipelines is a tool, with `name` prefixed by
the pipeline (`rs_router.*` / `multi_axis.*` / `mts_generation.*`). Each tool folder
contains `tool.py` (the TOOL singleton) and `README.md` (explaining the **reuse source**:
original project path + function + line numbers, and the migration advantages retained).

- **Direct import of original modules**: for modules that import cleanly
  (`find_length_rank`, `Window`, `standardizer`, `multi_axis.*`, `mvaxis.*`, etc.), tools
  call `Data_preprocessing.core.origins.ensure(...)` inside `run()` to add the original
  project's source root to `sys.path`, then import the original function directly,
  achieving zero porting deviation.
- **Logic-equivalent port**: for heavy dependencies (e.g. `ts_router.py` requires
  torch+checkpoint) or small logic inlined in large functions (`zscore`/`minmax`/
  `mean_fuse`/removing zero columns/adding noise, etc.), the logic is ported line-by-line
  into standalone functions, and the README records the original file:function:line.

`list_tools()` additionally returns an `origin` field, so the Agent can trace each tool's
source.

## Notes

- Input is a wide `DataFrame` (rows=time, columns=variables) or a numpy 2-D array;
  univariate tools accept a Series or a 1-D array. Tools normalize input internally with
  `to_frame / to_1d`.
- Each tool is `tools/<NNN>_<name>/tool.py`, defining a module-level `TOOL` singleton; the
  registry scans by file path, so adding a tool only requires creating a new folder with a
  `tool.py`, with no registry changes.
