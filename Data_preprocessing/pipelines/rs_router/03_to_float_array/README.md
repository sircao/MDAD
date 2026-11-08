# rs_router.to_float_array

**Stage**: Input reading **Purpose**: Type conversion, unifying the input into a float array for use by matrix computation libraries and deep learning frameworks.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `infer.py:19` (`.to_numpy(dtype=float)`), `ts_router.py:109` (`np.asarray(data, dtype=float)`)
- Reuse method: logic-equivalent port

## Migration Benefits
- Keeps the original "lenient reading, strict float coercion" convention without changing numeric precision semantics.

## Usage
```python
tool = get("rs_router.to_float_array")
r = tool.run(df)         # r.data is a float64 2-D array
```
