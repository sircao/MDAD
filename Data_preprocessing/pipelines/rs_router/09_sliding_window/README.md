# rs_router.sliding_window

**Stage**: Sliding-window feature construction **Purpose**: Convert a univariate series into windowed samples with context, serving as input features for principal component analysis / one-class support vector machines.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `models/feature.py:11-14` → `Window.convert()` (based on `numpy.lib.stride_tricks.sliding_window_view`)
- Reuse method: **import the original module directly**

## Migration Benefits
- Zero porting drift: preserves the exact shape semantics of the original `sliding_window_view` copy-free view plus `reshape((T-(w-1), -1))[::stride]`.
- The window result is identical to `Window(...).convert(X)` inside PCA/OCSVM.

## Usage
```python
tool = get("rs_router.sliding_window")
r = tool.run(data, window=100, stride=1)   # r.data is the window feature matrix
```
