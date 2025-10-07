---
title: Architecture overview
---

# Architecture

Understanding pymts internals makes it easier to extend the model catalogue and maintain determinism.

## Core flow

1. **Generator.generate**
   - Normalises configs (lowercases model names, applies defaults).
   - Validates parameters with the registered Pydantic schema.
   - Builds slug, hash8, and config_id via canonical JSON.
   - Derives sub-seeds with SeedSequence.spawn(len(configs)) and initialises default_rng(PCG64) for each config.
   - Dispatches to the registered model implementation.
2. **Model implementations**
   - Live in pymts/models/. Each inherits BaseGenerator, exposes param_model(), help(), and implements generate_one.
   - Must return an xarray.Dataset with dims (time, channel, realization).
   - Populate attrs: model, config_id, slug, hash8, 
_realizations, seed_entropy, and JSON-friendly params.
3. **Post-processing**
   - Ensures dimension order, optional z-scoring, and attribute merging.
4. **Persistence**
   - pymts.io.save_parquet writes datasets to Parquet and merges attrs/metadata.
   - write_sidecar_metadata stores JSON sidecars for provenance.

## Registries

pymts.models.REGISTRY maps model names to singleton generator objects. Generator.__init__ copies this mapping, so you can register new models dynamically in notebooks or tests without affecting the global registry.

## Hashing and slugs

- slugify_config(model, params) builds a human-readable slug using select keys plus an 8-character SHA-256 digest.
- hash8_from_obj operates on the canonical JSON representation to keep IDs stable across Python versions.

## RNG discipline

- All randomness flows through NumPy's PCG64 generator.
- Top-level seed (Generator.generate(..., seed=123)) creates a SeedSequence which spawns sub-sequences per config; each model receives an independent RNG.
- Model code should not call 
p.random module-level functions—always use the provided ng argument.

## Storage layout

`
data/<model>/<config_id>/
  +-- <config_id>.parquet
  +-- <config_id>.metadata.json
  +-- <config_id>.csv   # optional
`

The metadata sidecar mirrors dataset attrs and makes it easy to load JSON without materialising the full Parquet file.

## CLI execution

cli/app.py handles YAML grids, dry runs, limit flags, saving, and CSV toggles. It reuses the same Generator API, so observability (logging, profiling) can be implemented centrally.

## Documentation build

- MkDocs + Material with mkdocstrings renders Python API docs.
- mkdocs-jupyter executes tutorials during the build; keep notebooks light enough for CI.
- mike manages versioned docs (latest, stable).

## Tests and CI

- Unit tests live in 	ests/. Each new model requires coverage for shape, determinism, and invariants.
- .github/workflows/tests.yml runs pytest -q.
- .github/workflows/docs.yml installs docs dependencies and deploys the MkDocs site to gh-pages using mike.
