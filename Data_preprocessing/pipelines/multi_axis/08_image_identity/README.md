# multi_axis.image_identity

**Stage**: image identity **Purpose**: determine the image's unique id and path, giving each sample a deterministic, verifiable image path and avoiding duplicate rendering.

## Reuse Source
- Original project: `Multi-AXIS-B1DC`
- Source location: `src/multi_axis/data.py:114-121` → `visual_image_id()`; `data.py:124-127` → `visual_image_relpath()`
- Reuse method: **direct import of the original module**

## Migration Advantages
- Zero porting deviation: preserve the deterministic id `SHA-256(renderer_version\0base_sample_id\0start\0end)` and the relative path rule `images/<xx>/<id>.png`.

## Usage
```python
tool = get("multi_axis.image_identity")
r = tool.run(base_sample_id, interval=(start, end))   # r.data == {"image_id", "image_relpath"}
```
