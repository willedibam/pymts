"""
Plotting utilities for visualising PyMTS outputs.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def plot_heatmap(
    data: xr.Dataset | pd.DataFrame,
    *,
    var: str = "data",
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Render a heatmap for a selected realization using seaborn's ``icefire`` cmap.

    Parameters
    ----------
    data:
        Dataset or long-form DataFrame with columns ``time``, ``channel``,
        ``realization``, ``value``.
    var:
        Variable name to render when passing a dataset.
    ax:
        Optional axes to draw on; otherwise a new figure/axes is created.
    kwargs:
        Additional keyword arguments passed to ``pcolormesh``. ``realization`` and
        ``config_id`` may also be supplied to select specific subsets.
    """

    realization = kwargs.pop("realization", 0)
    config_id = kwargs.pop("config_id", None)
    df = _ensure_dataframe(data, var)

    subset = df[df["realization"] == realization]
    if config_id is not None:
        subset = subset[subset["config_id"] == config_id]
    if subset.empty:
        raise ValueError("No data available for the requested realization/config_id.")

    time_vals = np.sort(subset["time"].unique())
    channel_vals = np.sort(subset["channel"].unique())

    matrix = (
        subset.pivot_table(index="channel", columns="time", values="value")
        .reindex(index=channel_vals, columns=time_vals)
        .to_numpy()
    )

    time_edges = _compute_edges(time_vals)
    channel_edges = _compute_edges(channel_vals)

    if ax is None:
        _, ax = plt.subplots()

    cmap = kwargs.pop("cmap", None)
    if cmap is None:
        import seaborn as sns  # lazy import

        cmap = sns.color_palette("icefire", as_cmap=True)

    mesh = ax.pcolormesh(time_edges, channel_edges, matrix, cmap=cmap, shading="auto", **kwargs)
    ax.figure.colorbar(mesh, ax=ax)
    ax.set_xlabel("time")
    ax.set_ylabel("channel")
    ax.grid(False)
    return ax


def plot_timeseries(
    data: xr.Dataset | pd.DataFrame,
    *,
    var: str = "data",
    ax: Optional[plt.Axes] = None,
    stems: bool = False,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plot time-series for each channel in a selected realization.

    Parameters
    ----------
    stems:
        When ``True`` renders coloured stems instead of continuous lines.
    """

    realization = kwargs.pop("realization", 0)
    config_id = kwargs.pop("config_id", None)
    df = _ensure_dataframe(data, var)

    subset = df[df["realization"] == realization]
    if config_id is not None:
        subset = subset[subset["config_id"] == config_id]
    if subset.empty:
        raise ValueError("No data available for the requested realization/config_id.")

    if ax is None:
        _, ax = plt.subplots()

    import seaborn as sns  # lazy import

    channels = np.sort(subset["channel"].unique())
    palette = sns.color_palette(n_colors=len(channels))
    base_kwargs = {k: v for k, v in kwargs.items() if k not in {"realization", "config_id"}}

    for color, channel in zip(palette, channels, strict=True):
        series = subset[subset["channel"] == channel]
        times = series["time"].to_numpy()
        values = series["value"].to_numpy()

        if stems:
            ax.vlines(times, 0.0, values, color=color, alpha=0.7, linewidth=1.0)
            ax.scatter(times, values, color=color, s=15)
        else:
            ax.plot(times, values, color=color, label=f"channel {channel}", **base_kwargs)

    if not stems:
        ax.legend(loc="best", fontsize="small")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(False)
    return ax


def plot_phase(
    data: xr.Dataset | pd.DataFrame,
    *,
    var: str = "data",
    ax: Optional[plt.Axes] = None,
    max_channels: int = 4,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plot unwrapped phase trajectories for Kuramoto-style datasets.
    """

    realization = kwargs.pop("realization", 0)
    config_id = kwargs.pop("config_id", None)
    df = _ensure_dataframe(data, var)

    if ax is None:
        _, ax = plt.subplots()

    import seaborn as sns  # lazy import

    subset = df[df["realization"] == realization]
    if config_id is not None:
        subset = subset[subset["config_id"] == config_id]
    if subset.empty:
        raise ValueError("No data available for the requested realization/config_id.")

    channels = np.sort(subset["channel"].unique())[:max_channels]
    palette = sns.color_palette(n_colors=len(channels))

    for color, channel in zip(palette, channels, strict=True):
        series = subset[subset["channel"] == channel]
        times = series["time"].to_numpy()
        values = series["value"].to_numpy()
        unwrapped = np.unwrap(values)
        ax.plot(times, unwrapped, color=color, label=f"channel {channel}")

    ax.set_xlabel("time")
    ax.set_ylabel("phase (rad)")
    ax.grid(False)
    if channels.size > 1:
        ax.legend(loc="best", fontsize="small")
    return ax


def _ensure_dataframe(data: xr.Dataset | pd.DataFrame, var: str) -> pd.DataFrame:
    if isinstance(data, xr.Dataset):
        return data.pymts.to_dataframe(var=var)
    if isinstance(data, pd.DataFrame):
        required = {"time", "channel", "realization", "value"}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            msg = f"DataFrame must contain columns {required}, missing {missing}."
            raise ValueError(msg)
        return data
    msg = "Data must be an xarray.Dataset or pandas.DataFrame."
    raise TypeError(msg)


def _compute_edges(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.array([arr[0] - 0.5, arr[0] + 0.5], dtype=float)
    diffs = np.diff(arr)
    edges = np.empty(arr.size + 1, dtype=float)
    edges[0] = arr[0] - diffs[0] / 2
    edges[1:-1] = arr[:-1] + diffs / 2
    edges[-1] = arr[-1] + diffs[-1] / 2
    return edges
