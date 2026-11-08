# rs_router.acf_period

**Stage**: Period estimation **Purpose**: Autocorrelation analysis, automatically inferring the series period and providing an appropriate window length for sliding-window features.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `utils/slidingWindows.py:8-69` → `find_length_rank(data, rank=1)`
- Reuse method: **import the original module directly** (after `ensure(TS-Router-D8FF)`, `from utils.slidingWindows import find_length_rank`)

## Migration Benefits
- Zero porting drift: it calls the original function itself, including all its details such as reducing multivariate data to one dimension, `rank`-based local-maximum selection, and window bounds (falling back to 125 when <3 or >300).
- The dependencies (statsmodels/scipy) are already in the project and can run directly.

## Usage
```python
tool = get("rs_router.acf_period")
r = tool.run(data, rank=1)        # r.data is the inferred window length
```
