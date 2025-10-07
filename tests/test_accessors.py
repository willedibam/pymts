from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import pymts.accessors  # noqa: F401  # ensures accessor registration


def test_accessor_registered() -> None:
    ds = xr.Dataset()
    assert hasattr(ds, "pymts")


def test_accessor_conversions() -> None:
    data = np.arange(12, dtype=float).reshape(3, 2, 2)
    coords = {
        "time": [0, 1, 2],
        "channel": ["a", "b"],
        "realization": [0, 1],
    }
    ds = xr.Dataset(
        {"data": (("time", "channel", "realization"), data)},
        coords=coords,
        attrs={"config_id": "demo__deadbeef"},
    )

    array = ds.pymts.to_numpy()
    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 2, 2)

    df = ds.pymts.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["time", "channel", "realization", "value", "config_id"]
    assert len(df) == 12
    assert df["config_id"].unique().tolist() == ["demo__deadbeef"]
