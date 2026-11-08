# elki

**Category**: Algorithm libraries & frameworks　**Source**: https://elki-project.github.io/algorithms/

**Implementation**: Native equivalent implementation (does not depend on ELKI/Java). LOF (scikit-learn), k-nearest-neighbor average distance, and k-th nearest-neighbor distance (KNNOutlier style), outputting pointwise anomaly scores.

**Usage**:
```python
tool = get("elki")
r = tool.run(data, algorithm="lof", n_neighbors=20)   # r.data is a pointwise anomaly score
```
Supported algorithms: `lof` `knn` `kth_neighbor`.
