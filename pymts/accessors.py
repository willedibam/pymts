"""
Custom xarray accessors exposed via ``Dataset.pymts``.

The accessor provides ergonomic conversions to NumPy arrays and pandas
DataFrames while enforcing PyMTS dimensional conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["PyMTSAccessor"]


@xr.register_dataset_accessor("pymts")
class PyMTSAccessor:
    """Xarray dataset accessor providing PyMTS-specific transformations."""

    def __init__(self, dataset: xr.Dataset) -> None:
        self._dataset = dataset

    def to_numpy(self, var: str = "data") -> np.ndarray:
        """
        Convert the dataset into a NumPy ndarray following (time, channel, realization).

        Parameters
        ----------
        var:
            Name of the data variable to extract. Defaults to ``"data"``.
        """

        data_array = self._select_data_array(var)
        return data_array.transpose("time", "channel", "realization").values

    def to_dataframe(self, var: str = "data") -> pd.DataFrame:
        """
        Convert the dataset into a tidy pandas DataFrame.

        The output columns are ``time``, ``channel``, ``realization``, ``value``,
        and ``config_id`` (the latter populated from dataset attributes, falling
        back to ``None`` when unavailable).
        """

        data_array = self._select_data_array(var)
        data_array = data_array.transpose("time", "channel", "realization")

        coords_update: dict[str, Any] = {}
        for dim in ("time", "channel", "realization"):
            if dim not in data_array.coords:
                coords_update[dim] = np.arange(data_array.sizes[dim])
        if coords_update:
            data_array = data_array.assign_coords(coords_update)

        df = data_array.to_dataframe(name="value").reset_index()
        df["config_id"] = self._dataset.attrs.get("config_id")
        return df[["time", "channel", "realization", "value", "config_id"]]

    def _select_data_array(self, var: str) -> xr.DataArray:
        """Internal helper ensuring the requested variable exists and has valid dims."""

        if var not in self._dataset.data_vars:
            msg = f"Dataset does not contain variable '{var}'."
            raise KeyError(msg)
        data_array = self._dataset[var]

        expected_dims = {"time", "channel", "realization"}
        if not expected_dims.issubset(set(data_array.dims)):
            msg = (
                f"Variable '{var}' must include dimensions {expected_dims}, "
                f"found {tuple(data_array.dims)}."
            )
            raise ValueError(msg)
        return data_array
