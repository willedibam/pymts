"""
Adapters for bridging PyMTS datasets with external ecosystems.

Step 1 sets up stubs so downstream modules can rely on the public interface while
implementations are completed in later steps.
"""

from __future__ import annotations

from typing import Any

import xarray as xr


def to_pandas_panel(dataset: xr.Dataset, **kwargs: Any) -> Any:
    """
    Convert a dataset into a pandas-style panel representation.

    Notes
    -----
    Implemented in Step 2. Currently raises ``NotImplementedError``.
    """

    if not isinstance(dataset, xr.Dataset):
        msg = "dataset must be an xarray.Dataset."
        raise TypeError(msg)

    raise NotImplementedError("TODO: Implement pandas adapter in Step 2.")


def to_polars_frame(dataset: xr.Dataset, **kwargs: Any) -> Any:
    """
    Convert a dataset into a Polars DataFrame.

    Notes
    -----
    Implemented in Step 2. Currently raises ``NotImplementedError``.
    """

    if not isinstance(dataset, xr.Dataset):
        msg = "dataset must be an xarray.Dataset."
        raise TypeError(msg)

    raise NotImplementedError("TODO: Implement Polars adapter in Step 2.")
