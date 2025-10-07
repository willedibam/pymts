---
title: Welcome to pymts
---

# pymts

pymts synthesises richly annotated multivariate time series (MTS) with deterministic pipelines, metadata-first design, and integrations across analytics ecosystems.

!!! info "Core invariants"
    - Outputs are always xarray.Dataset objects with dims (time, channel, realization) and matching coordinates.
    - Each run is fingerprinted as config_id = "<slug>__<hash8>", where hash8 is derived from the SHA-256 canonical JSON of the model and parameters.
    - Randomness flows through **top-level seed ? 
umpy.random.SeedSequence.spawn() ? per-config default_rng(PCG64)**.
    - Kuramoto simulations default to RK4 with dt = 0.05.
    - Heatmaps adopt Matplotlib + Seaborn "icefire" via pcolormesh; optional coloured stems show per-channel trajectories.

---

## Why pymts?

| Challenge | pymts answer |
| --- | --- |
| Reproducible synthetic cohorts | Deterministic seeding, stable hash IDs, metadata sidecars |
| Model breadth | Kuramoto, coupled map lattices, AR/VAR, OU, Brownian/GBM, IID noise |
| Downstream compatibility | .pymts.to_dataframe(), .pymts.to_numpy(), CLI grids, Parquet + JSON outputs |
| Developer ergonomics | Pydantic schemas, stability guards, mkdocstrings API reference |
| Research velocity | Notebook tutorials, integration guides (PySPI, tsfresh, sklearn, statsmodels) |

---

## Micro demo

`python
from pymts import Generator

generator = Generator()

config = {
    "model": "kuramoto",
    "M": 4,
    "T": 128,
    "K": 0.7,
    "topology": "ring",
    "n_realizations": 2,
    "seed": 424242,
}

dataset = generator.generate([config])[0]
print(dataset)
`

You receive an xarray.Dataset with attrs such as {"model": "kuramoto", "config_id": "kuramoto_M4_T128_K0.7__1165c0d6", ...}. Re-running reproduces the trajectories exactly.

Convert and visualise:

`python
import matplotlib.pyplot as plt
from pymts.plotting import plot_heatmap, plot_timeseries

df = dataset.pymts.to_dataframe()
array = dataset.pymts.to_numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
plot_heatmap(dataset, realization=0, ax=ax1)
plot_timeseries(dataset, realization=0, stems=True, ax=ax2)
plt.show()
`

!!! warning "Assets"
    Example plots reference local assets; generate your own figures or replace paths under docs/assets/images.

---

## Documentation map

| Track | Highlights |
| --- | --- |
| [Quickstart](quickstart.md) | Install, generate Kuramoto + GBM datasets, convert to pandas/NumPy, save Parquet, plot heatmap + stems. |
| [Tutorials](tutorials/01_kuramoto_and_friends.ipynb) | Executable notebooks for Kuramoto/OU/noise and CLI YAML grids. |
| [How-to recipes](how-to/recipes.md) | Stack configs, wrangle metadata, slice by realization, export figures. |
| [Integrations](integrations/pyspi_hctsa.md) | Bridges to PySPI/hctsa, scikit-learn, statsmodels, tsfresh. |
| [Developer docs](developer/architecture.md) | Architecture, extending pymts, contribution workflow, shipping guide. |
| [Reference](reference/api.md) | mkdocstrings API reference and CLI command table. |
| [FAQ & troubleshooting](faq.md) | Determinism checklist, shape pitfalls, OS tips. |

---

## Scaling up safely

!!! tip "Production checklist"
    1. Pin pymts and documentation dependencies in your lock file.
    2. Version-control YAML configs and base seeds; pymts handles sub-seeds deterministically.
    3. Use --limit when exploring large grids and monitor storage layout data/<model>/<config_id>/.
    4. Snapshot Parquet + metadata sidecars with an artifact store or DVC.
    5. For HPC runs, assign distinct top-level seeds per node; sub-seeds remain reproducible.

Jump to the [Quickstart](quickstart.md) to build your first workflows.
