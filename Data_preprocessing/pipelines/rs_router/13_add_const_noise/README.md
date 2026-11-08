# rs_router.add_const_noise

**Stage**: Anti-degeneracy handling **Purpose**: Add noise to constant columns, preventing division by zero during standardization and keeping every multivariate column computable.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:142-145` (the multivariate branch adds noise when `std <= 1e-12`), `models/PCA.py:252-258` (adds noise when all zero)
- Reuse method: logic-equivalent port (generalizing "trigger only when all columns are constant" to "trigger per constant column", keeping the `1e-12` threshold and `1e-6` noise scale)

## Migration Benefits
- Preserves the original threshold `std <= 1e-12` and noise scale `scale=1e-6`.
- Per-column handling is closer to the "anti-degeneracy" goal than the original "add noise only when all columns are constant", and leaves non-degenerate columns unchanged.

## Usage
```python
tool = get("rs_router.add_const_noise")
r = tool.run(data, scale=1e-6, seed=0)   # r.data is the processed matrix
```
