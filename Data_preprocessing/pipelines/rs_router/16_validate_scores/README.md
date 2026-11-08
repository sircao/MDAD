# rs_router.validate_scores

**Stage**: Output validation **Purpose**: Finiteness and length validation, intercepting non-numeric / infinite / length-inconsistent outputs to ensure downstream evaluation remains usable.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `ts_router.py:64` → inside `run_specialist()`, raises when `len(score) != len(test) or not np.isfinite(score).all()`
- Reuse method: logic-equivalent port

## Migration Benefits
- Fully preserves the two validation dimensions: the score length must match the test suffix, and all scores must be finite.
- Validation failure returns a structured `ToolResult` (ok=False + error), making it easy for the upper layer to downgrade as "insufficient evidence".

## Usage
```python
tool = get("rs_router.validate_scores")
r = tool.run(scores, expected_length=len(test))
```
