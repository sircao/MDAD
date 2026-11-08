# rs_router.internal_standardizer

**Stage**: Model internals **Purpose**: Built-in standardizer, standardizing again inside the expert to satisfy the zero-mean unit-variance assumption of principal component analysis.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `utils/utility.py:253-291` → `standardizer(X, X_t=None, keep_scalar=False)`; called by `models/PCA.py:247`
- Reuse method: **import the original module directly**, wrapped as `fit/apply` (using `keep_scalar=True` to retrieve the `StandardScaler`)

## Migration Benefits
- Zero porting drift: directly calls the original `standardizer`, preserving all of its `check_array` validation, `StandardScaler().fit(X)`, and `scaler.transform` behavior.
- Enforces "standardization fits only on the training set" through the `fit/apply` separation, which is better at preventing leakage than the original function's one-shot usage.

## Usage
```python
tool = get("rs_router.internal_standardizer")
tool.fit(train); r = tool.apply(test)
```
