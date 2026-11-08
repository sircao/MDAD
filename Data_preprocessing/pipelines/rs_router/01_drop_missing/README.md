# rs_router.drop_missing

**Stage**: Input reading **Purpose**: Drop missing values, ensuring downstream standardization and matrix operations do not error or produce non-numeric values due to missing data.

## Reuse Source
- Original project: `TS-Router-D8FF`
- Source location: `infer.py:17` → inside `read_input()`, `frame = pd.read_csv(path).dropna()`
- Reuse method: logic-equivalent port (the original logic is inlined in the read function; there is no standalone function)

## Migration Benefits
- Fully preserves the "drop the entire row on missing values" semantics: any missing value drops that sample, avoiding errors in downstream `check_array`/matrix operations.
- Emits a warning stating exactly how many rows were removed, rather than silently discarding data.

## Usage
```python
from Data_Reprocessing.registry import get
tool = get("rs_router.drop_missing")
r = tool.run(df)          # r.data is the DataFrame after dropna
```
