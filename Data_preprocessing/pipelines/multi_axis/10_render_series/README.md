# multi_axis.render_series

**Stage**: visual rendering **Purpose**: render the normalized series, generating an image view of the same observations as the numeric window for use by the vision-language model; the rendering carries no labels/scores/causal information.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/rendering.py:94-198` → `render_normalized_series()`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: preserve "what is rendered is `normalized_series` (not normalized a second time)", the fixed 600 dpi, Times New Roman font auditing, dark-blue highlighting of the target interval, deterministic rendering without labels/scores/causal information, and the rendering audit (sha256/size/font).

## Usage
```python
tool = get("multi_axis.render_series")
r = tool.run(normalized, interval=(s, e), channel_ids=ids, output_path="out.png", dpi=600)
```
