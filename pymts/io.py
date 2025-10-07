"""
Persistence utilities for PyMTS datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from .utils import canonical_json

__all__ = ["save_parquet", "save_csv", "write_sidecar_metadata"]


def save_parquet(ds: xr.Dataset, path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
    """
    Persist an :class:`xarray.Dataset` as a Parquet file with embedded metadata.
    """

    if not isinstance(ds, xr.Dataset):
        msg = "ds must be an xarray.Dataset instance."
        raise TypeError(msg)
    if metadata is not None and not isinstance(metadata, Mapping):
        msg = "metadata must be a mapping when provided."
        raise TypeError(msg)

    output_path = Path(path)
    if not output_path.suffix:
        msg = "path must include a file extension."
        raise ValueError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = ds.pymts.to_dataframe()
    table = pa.Table.from_pandas(df, preserve_index=False)

    existing_meta = table.schema.metadata or {}
    merged = _stringify_metadata({**dict(ds.attrs), **(metadata or {})})
    table = table.replace_schema_metadata({**existing_meta, **merged})

    pq.write_table(table, output_path)
    return output_path


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Persist a pandas DataFrame to CSV (UTF-8, no index).
    """

    if not isinstance(df, pd.DataFrame):
        msg = "df must be a pandas.DataFrame instance."
        raise TypeError(msg)

    output_path = Path(path)
    if not output_path.suffix:
        msg = "path must include a file extension."
        raise ValueError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def write_sidecar_metadata(meta_path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """
    Write metadata to a JSON sidecar file with deterministic formatting.
    """

    if not isinstance(metadata, Mapping):
        msg = "metadata must be a mapping."
        raise TypeError(msg)

    output_path = Path(meta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(canonical_json(metadata))
    output_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return output_path


def _stringify_metadata(metadata: Mapping[str, Any]) -> dict[bytes, bytes]:
    """Convert metadata values into bytes suitable for Parquet key-value storage."""

    result: dict[bytes, bytes] = {}
    for key, value in metadata.items():
        key_str = str(key)
        if isinstance(value, str):
            value_str = value
        elif isinstance(value, (int, float, bool)) or value is None:
            value_str = json.dumps(value, sort_keys=True)
        else:
            value_str = canonical_json(value)
        result[key_str.encode("utf-8")] = value_str.encode("utf-8")
    return result
