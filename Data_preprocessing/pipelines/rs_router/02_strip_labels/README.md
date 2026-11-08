# rs_router.strip_labels

**Stage**: Input reading **Purpose**: Label stripping, keeping only numeric columns in the feature matrix and reserving labels for evaluation, avoiding contamination of the model input.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `infer.py:18-19` → inside `read_input()`, `labels = frame["Label"]` and `frame.drop(columns=["Label"])`
- Reuse method: logic-equivalent port

## Migration Benefits
- Preserves the "label column is optional" tolerance: when there is no Label column, `labels=None`, matching `infer.py`.
- Features are coerced to `float` and labels to `int`, matching the original read convention.

## Usage
```python
tool = get("rs_router.strip_labels")
r = tool.run(df, label_column="Label")   # r.data == {"features": 2-D array, "labels": 1-D array or None}
```
