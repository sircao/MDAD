"""Data_preprocessing — a time series preprocessing toolset for data-centric architectures.

Each algorithm lives in its own folder (tools/<NNN>_<name>/tool.py) and returns results
through ``ToolResult`` with provenance: tool version, dependency library versions, call
parameters, and runtime. Tools that need to fit parameters are stateful (separate
fit/apply), honoring the constraint that scaling/thresholds/features are fitted on the
training set only, to prevent test-information leakage.

Usage::

    from Data_preprocessing.registry import get, list_tools

    tool = get("interpolate_time")
    result = tool.run(df, method="linear")
    assert result.ok
    out_df = result.data
"""

__version__ = "0.1.0"
