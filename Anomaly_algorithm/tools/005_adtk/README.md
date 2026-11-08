# adtk

**Category**: Algorithm libraries & frameworks　**Source**: https://github.com/arundo/adtk

**Implementation**: Native equivalent implementation (does not depend on the adtk package). Four rule/statistical detectors — quantile, fixed threshold, persist, and level shift — output boolean anomaly labels.

**Usage**:
```python
tool = get("adtk")
r = tool.run(data, detector="quantile", high=0.99, low=0.01)
```
Supported detectors: `quantile` `threshold` `persist` `level_shift`.
