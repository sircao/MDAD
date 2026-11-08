# pysad

**Category**: Algorithm libraries & frameworks　**Source**: https://github.com/selimfirat/pysad

**Implementation**: Native streaming implementation (does not depend on the pysad package). KNN distance scoring within a sliding buffer (ExactSTORM style), KNND, and sliding z-score receive data incrementally and update scores.

**Usage**:
```python
tool = get("pysad")
r = tool.run(data, model="exact_storm", window=100, n_neighbors=5)   # r.data is a streaming anomaly score
```
Supported models: `exact_storm` `knnd` `sliding_zscore`.
