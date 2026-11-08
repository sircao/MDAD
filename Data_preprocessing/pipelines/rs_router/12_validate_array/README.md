# rs_router.validate_array

**Stage**: Model internals **Purpose**: Input array validation, checking shape and numeric type to expose invalid inputs as early as possible.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `models/PCA.py:230`, `utils/utility.py:273` → both use `sklearn.utils.check_array(X)`
- Reuse method: logic-equivalent port (directly calls `sklearn.utils.check_array`)

## Migration Benefits
- Identical source to the original implementation: both use `check_array`, whose default `force_all_finite=True` intercepts NaN/infinite values and converts to float64, ensuring downstream matrix operations work.

## Usage
```python
tool = get("rs_router.validate_array")
r = tool.run(data)                 # r.data is the validated array cast to float64
```
