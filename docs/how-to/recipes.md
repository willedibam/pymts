---
title: Task-oriented recipes
---

# Recipes

Bite-sized workflows for everyday pymts tasks.

## Stack multiple configurations

`python
from pymts import Generator

configs = [
    {"model": "kuramoto", "M": 4, "T": 128, "K": 0.5, "seed": 1},
    {"model": "ou", "M": 3, "T": 96, "theta": 0.8, "seed": 2},
]

generator = Generator()
datasets = generator.generate(configs)
`

Each dataset preserves its own config_id, so you can concatenate only after adding that identifier to your index (e.g., xr.concat(datasets, dim="config_id")).

## Access metadata when saving

`python
from pymts.io import save_parquet, write_sidecar_metadata
from pathlib import Path

ds = datasets[0]
store = Path("data/batch") / ds.attrs["config_id"]
store.mkdir(parents=True, exist_ok=True)

save_parquet(ds, store / f"{ds.attrs['config_id']}.parquet")
write_sidecar_metadata(store / f"{ds.attrs['config_id']}.metadata.json", ds.attrs)
`

## Slice by realization

Use xarray indexing or DataFrame filters:

`python
first_realization = ds.sel(realization=0)
second_realization_df = ds.pymts.to_dataframe()
second_realization_df = second_realization_df[second_realization_df["realization"] == 1]
`

## Align datasets with different horizons

If two configs have different T, pad manually using xarray:

`python
import xarray as xr

aligned = xr.align(datasets[0], datasets[1], join="outer", fill_value=float("nan"))
`

Alternatively, convert to DataFrame and merge on 	ime, channel, realization, filling missing values as needed.

## Export publication-ready figures

`python
import matplotlib.pyplot as plt
from pymts.plotting import plot_heatmap

fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
plot_heatmap(ds, realization=0, ax=ax)
ax.set_title("Kuramoto heatmap")
fig.savefig("figure01.png", bbox_inches="tight")
`

Set Matplotlib RC parameters (fonts, linewidths) before plotting to match journal styles.
