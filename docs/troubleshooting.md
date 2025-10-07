---
title: Troubleshooting
---

# Troubleshooting checklist

## Installation

- **Missing 
bformat/docs deps:** Install with uv pip install -r requirements-docs.txt.
- **Compilation errors on Windows:** pymts is pure Python; ensure you are using Python 3.11 and an activated virtual environment.

## Runtime errors

- **ValueError: Dataset missing required dimensions** — ensure your model returns (time, channel, realization) and does not squeeze dimensions.
- **AttributeError: Dataset has no attribute 'pymts'** — import pymts before loading datasets; the accessor registers on import.
- **FileNotFoundError when saving** — create parent directories (Path(...).mkdir(parents=True, exist_ok=True)) before calling save_parquet.

## CLI issues

- **No configs found:** Confirm YAML structure starts with configs:.
- **Dry run prints nothing:** The expansion may be empty—check for empty lists or limit=0.

## Plotting

- **Matplotlib backend errors on headless servers:** Use matplotlib.use('Agg', force=True) before plotting or rely on the provided plotting utilities.

## Performance

- Start with small T and M; profile memory with dataset['data'].nbytes.
- Use CLI --limit to sample grids during prototyping.

## Determinism checklist

1. Set top-level seed.
2. Avoid global RNG calls in custom pipelines; use the provided ng or seeded instantiations.
3. Record config_id and seed_entropy when logging results.
