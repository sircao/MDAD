# rolling_volatility

**Category**: Algorithm libraries & frameworks　**Source**: A common industrial practice based on the rolling standard deviation control chart (Shewhart control chart), implemented natively.

**Implementation**: Native numpy/pandas implementation. A rolling-window volatility statistic is used as the pointwise anomaly score:

- `std` —— rolling standard deviation (default, most general)
- `range` —— rolling range (max−min)
- `mad` —— rolling median absolute deviation (more robust to outliers)

**Applicability**: Segment-level anomalies such as level drift and increased variance (many sequences in TSB-AD-U fall into this category). It is weaker than z-score / Isolation Forest for isolated single-point spikes, but it can stably capture "sustained high volatility".

**Usage**:
```python
tool = get("rolling_volatility")
r = tool.run(data, stat="std", window=100)   # r.data is a pointwise anomaly score
```

Window selection is an unsupervised hyperparameter; it is recommended that the upper-level agent automatically select it over candidate windows (e.g., 20/50/100) via Pareto frontier / orthogonal search.
