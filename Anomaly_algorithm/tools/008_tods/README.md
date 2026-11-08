# tods

**Category**: Algorithm libraries & frameworks　**Source**: https://github.com/datamllab/tods

**Implementation**: Native equivalent implementation (does not depend on the tods package). z-score / IQR / moving-average residual / Isolation Forest, supporting pointwise scores or boolean labels.

**Usage**:
```python
tool = get("tods")
r = tool.run(data, detector="moving_average", window=10, return_scores=True)
```
Supported detectors: `zscore` `iqr` `moving_average` `isolation_forest`.
