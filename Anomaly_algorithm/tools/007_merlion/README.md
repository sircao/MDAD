# merlion

**Category**: Algorithm libraries & frameworks　**Source**: https://github.com/salesforce/Merlion

**Implementation**: Native equivalent implementation (does not depend on the merlion package). Four detection types — default (z-score) / IQR / EWMA residual / Isolation Forest — output boolean anomaly labels.

**Usage**:
```python
tool = get("merlion")
r = tool.run(data, detector="default", threshold=3.0)
```
Supported detectors: `default`(=zscore) `iqr` `ewma` `isolation_forest`.
