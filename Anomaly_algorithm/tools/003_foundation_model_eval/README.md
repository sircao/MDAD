# foundation_model_eval

**Category**: Zero-shot & general models　**Source**: ICLR 2026 paper (see tool metadata)

**Overview**: Systematically evaluates the TSAD performance of foundation models such as MOMENT/Chronos/TimesFM/Time-MoE/TSPulse, comparing against simple baselines such as moving-window variance and squared difference.

**Reuse**: Knowledge-retrieval tool (paper conclusions, including the boundary note that "anomaly detection capability cannot be inferred from prediction/reconstruction capability alone").

**Usage**:
```python
r = get("foundation_model_eval").run()
```
