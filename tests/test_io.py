from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from pymts.io import save_csv, save_parquet, write_sidecar_metadata


def _make_dataset() -> xr.Dataset:
    data = [[[0.0], [1.0]], [[2.0], [3.0]]]
    coords = {
        "time": [0, 1],
        "channel": ["c0", "c1"],
        "realization": [0],
    }
    return xr.Dataset(
        {"data": (("time", "channel", "realization"), data)},
        coords=coords,
        attrs={"config_id": "demo__deadbeef", "model": "demo"},
    )


def test_save_parquet_and_sidecar(tmp_path: Path) -> None:
    ds = _make_dataset()
    metadata = {"extra": {"note": "demo"}}
    parquet_path = save_parquet(ds, tmp_path / "demo.parquet", metadata=metadata)
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    table_meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in table.schema.metadata.items()}
    assert table_meta["config_id"] == "demo__deadbeef"
    assert json.loads(table_meta["extra"]) == metadata["extra"]

    sidecar_path = write_sidecar_metadata(tmp_path / "demo.metadata.json", metadata | {"config_id": "demo__deadbeef"})
    assert sidecar_path.exists()
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["config_id"] == "demo__deadbeef"


def test_save_csv(tmp_path: Path) -> None:
    ds = _make_dataset()
    df = ds.pymts.to_dataframe()
    csv_path = save_csv(df, tmp_path / "demo.csv")
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == len(df)
