---
title: Quickstart
---

# Quickstart: pymts in 5 minutes

Welcome! This guide takes you from installation to plots and persisted datasets in a single sitting.

## 1. Install pymts and documentation extras

Pick your preferred toolchain. Each option installs pymts in editable mode and the documentation dependencies (MkDocs Material, mkdocstrings, mkdocs-jupyter, mike, plotting packages).

=== "uv (recommended)"
    `ash
    uv venv -p 3.11
    source .venv/bin/activate           # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    uv pip install -e .
    uv pip install -r requirements-docs.txt
    mkdocs serve
    `

=== "pip + venv"
    `ash
    python3 -m venv .venv
    source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
    pip install -e .
    pip install -r requirements-docs.txt
    mkdocs serve
    `

=== "Poetry"
    `ash
    poetry env use 3.11
    poetry install
    poetry run mkdocs serve
    `

!!! tip "Docs stack"
    equirements-docs.txt keeps MkDocs Material, mkdocstrings, mkdocs-jupyter, mike, and plotting dependencies in sync with the tutorials.

## 2. Generate deterministic datasets

`python
from pymts import Generator

generator = Generator()

kuramoto_cfg = {
    "model": "kuramoto",
    "M": 4,
    "T": 128,
    "K": 0.6,
    "topology": "ring",
    "n_realizations": 2,
    "seed": 2025,
}

gbm_cfg = {
    "model": "gbm",
    "M": 3,
    "T": 96,
    "dt": 0.05,
    "mu": 0.08,
    "sigma": 0.25,
    "S0": 1.0,
    "n_realizations": 2,
    "seed": 99,
}

kuramoto_ds, gbm_ds = generator.generate([kuramoto_cfg, gbm_cfg])
`

Each dataset carries attrs including model, config_id, slug, hash8, params, and seed_entropy.

## 3. Convert to pandas and NumPy

`python
kuramoto_df = kuramoto_ds.pymts.to_dataframe()
kuramoto_array = kuramoto_ds.pymts.to_numpy()

kuramoto_df.head()
`

!!! important "Shape gospel"
    - .pymts.to_numpy() returns an array ordered (time, channel, realization).
    - .pymts.to_dataframe() yields long-form rows with columns 	ime, channel, ealization, alue, config_id.

## 4. Persist to Parquet and metadata JSON

`python
from pathlib import Path
from pymts.io import save_parquet, write_sidecar_metadata

output_dir = Path("data") / "kuramoto" / kuramoto_ds.attrs["config_id"]
output_dir.mkdir(parents=True, exist_ok=True)

parquet_path = save_parquet(kuramoto_ds, output_dir / f"{kuramoto_ds.attrs['config_id']}.parquet")
metadata_path = write_sidecar_metadata(
    output_dir / f"{kuramoto_ds.attrs['config_id']}.metadata.json",
    kuramoto_ds.attrs,
)

parquet_path, metadata_path
`

!!! warning "Keep metadata in sync"
    Always regenerate the JSON sidecar after you alter attrs or slice datasets you intend to persist. Downstream jobs depend on it.

## 5. Plot heatmaps and stems

`python
import matplotlib.pyplot as plt
from pymts.plotting import plot_heatmap, plot_timeseries

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
plot_heatmap(kuramoto_ds, realization=0, ax=ax1)
plot_timeseries(kuramoto_ds, realization=0, stems=True, ax=ax2)
plt.show()
`

The heatmap uses Seaborn's "icefire" palette with pcolormesh; stems emphasise per-channel oscillations.

## 6. Command-line grids

`ash
cat > configs/demo.yaml <<'YAML'
configs:
  - model: kuramoto
    M: [3]
    T: [64]
    K: [0.5, 0.9]
    n_realizations: 1
    seed: 123
  - model: gbm
    M: 2
    T: 64
    mu: [0.0, 0.05]
    sigma: 0.2
    dt: 0.02
    n_realizations: 1
    seed: 321
YAML

pymts generate --config configs/demo.yaml --outdir data/demo --save --csv
`

Outputs land in data/demo/<model>/<config_id>/ with Parquet + metadata JSON (and CSV if requested).

---

## Scaling up safely

- Increase T and M gradually; check dataset.nbytes for memory impact.
- Use --limit during exploratory CLI runs to tame huge YAML grids.
- Commit YAML configs and base seeds; pymts handles sub-seeds deterministically.
- Archive Parquet + metadata with an artifact store or DVC for reproducibility.
- On HPC, assign unique top-level seeds per job; sub-seeds remain stable across reruns.

Continue with the [Kuramoto, OU & Noise tutorial](tutorials/01_kuramoto_and_friends.ipynb).
