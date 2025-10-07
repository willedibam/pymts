# PyMTS

Synthetic multivariate time series (MTS) generation toolkit. The project now
ships with a catalogue of stochastic and deterministic models, grid-aware CLI,
and plotting helpers for quick inspection.

## Invariants
- Internal numerics rely on NumPy.
- Datasets are `xarray.Dataset` objects with dims `(time, channel, realization)` plus coordinates.
- Accessor helpers: `ds.pymts.to_dataframe()` → long form `{time, channel, realization, value, config_id}` and `ds.pymts.to_numpy()`.
- `config_id = "<slug>__<hash8>"`, where `hash8` is the SHA-256 digest of the canonical JSON describing the model parameters.
- Default storage: `data/<model>/<config_id>/` with `<slug>__<hash8>.parquet` + `<slug>__<hash8>.metadata.json`.
- Kuramoto default integrator: RK4 with `dt = 0.05`.
- RNG pipeline: top-level seed → `SeedSequence(seed)` → per-config sub-seeds via `numpy.random.PCG64` in `default_rng`.
- Parameter validation via Pydantic v2; every model exposes `help()` and machine schemas via `params()`.
- AR and VAR generators stabilise coefficient matrices automatically (companion matrix spectral radius < 1) for numerical safety.
- CLI built on Typer with YAML grid expansion.
- Plotting utilities rely on Matplotlib + Seaborn’s *icefire* palette.

## Installation & Environment

Python 3.11+ is required.

### Recommended: uv
```bash
uv venv
uv pip install -e .
uv pip install -r requirements.txt
```

### Pip + venv

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Poetry (optional)
```bash
poetry env use 3.11
poetry install
```

## Development Workflow
- Editable install: `pip install -e .`
- Run tests: `pytest -q`
- CLI help: `pymts --help`
- Dataset conversions: `ds.pymts.to_numpy()` / `ds.pymts.to_dataframe()`
- Persistence utilities: `pymts.io.save_parquet(ds, path)` + sidecar JSON writer

## CLI Grid Example
Create a YAML describing grids across models:

```yaml
# configs/example_grid.yaml
configs:
  - model: kuramoto
    M: [5, 8]
    T: 256
    K: [0.5, 1.0]
    n_realizations: 2
    seed: 123
  - model: gbm
    M: 4
    T: 128
    mu: [0.0, 0.05]
    sigma: 0.2
    dt: 0.01
    n_realizations: 3
```

Dry-run to view the Cartesian expansion:

```bash
pymts generate --config configs/example_grid.yaml --dry-run
```

Run the full generation and persist outputs (Parquet + metadata; add `--csv` to export CSV as well):

```bash
pymts generate --config configs/example_grid.yaml --save --csv
```

## Plotting
```python
import matplotlib.pyplot as plt
from pymts.plotting import plot_heatmap, plot_timeseries

dataset = Generator().generate([{"model": "noise_gaussian", "M": 4, "T": 64, "sigma": 0.5, "seed": 7}])[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_heatmap(dataset, realization=0, ax=ax1)
plot_timeseries(dataset, realization=0, stems=True, ax=ax2)
plt.tight_layout()
plt.show()
```

## Determinism
- Re-running with the same top-level seed and configuration grid yields identical outputs.
- Seeds are split via `SeedSequence.spawn`, so each configuration receives an independent sub-seed.
- Model-specific initialisations (e.g., AR/VAR coefficient sampling) honour the provided RNG.

## Roadmap
The core library is feature-complete for the initial release. Future work may
include richer visualisations, additional models, and storage backends.
