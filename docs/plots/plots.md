---
title: Plotting playbook
---

# Plotting playbook

pymts bundles light helpers built on Matplotlib + Seaborn. Use them to create consistent visuals.

## Heatmaps (icefire)

`python
import matplotlib.pyplot as plt
from pymts.plotting import plot_heatmap

dataset = ...  # xarray.Dataset
fig, ax = plt.subplots(figsize=(4, 3))
plot_heatmap(dataset, realization=0, ax=ax)
ax.set_title("Realization 0")
fig.savefig("heatmap.png", dpi=300, bbox_inches="tight")
`

The helper uses Seaborn's "icefire" palette via pcolormesh and disables grid lines for clarity.

## Coloured stems

`python
from pymts.plotting import plot_timeseries

fig, ax = plt.subplots(figsize=(5, 3))
plot_timeseries(dataset, realization=0, stems=True, ax=ax)
ax.set_ylabel("value")
`

Each channel receives a different colour; stems emphasise temporal spikes.

## Kuramoto phase plots

`python
from pymts.plotting import plot_phase

fig, ax = plt.subplots(figsize=(5, 3))
plot_phase(dataset, realization=0, max_channels=3, ax=ax)
`

plot_phase unwraps phases before plotting. Useful for observing synchrony.

## Publication tips

- Set matplotlib.rcParams to match journal fonts (e.g., cParams['font.size'] = 9).
- Use vector formats (.svg) for line drawings; use high-resolution PNGs for raster output.
- Include config_id in figure captions to document reproducibility.
- When comparing multiple configs, align colour palettes and axes ranges.
