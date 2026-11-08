# pyod

**Category**: Algorithm libraries & frameworks　**Source**: https://github.com/yzhao062/pyod

**Implementation**: Native equivalent implementation (does not depend on the pyod package). Isolation Forest / LOF / OCSVM use scikit-learn; HBOS, KNN distance, and PCA reconstruction error are implemented natively with numpy/sklearn, outputting pointwise anomaly scores (larger = more anomalous).

**Usage**:
```python
tool = get("pyod")
r = tool.run(data, algorithm="iforest", n_estimators=100)   # r.data is a pointwise anomaly score
```
Supported algorithms: `iforest` `lof` `ocsvm` `hbos` `knn` `pca`.
