---
title: Frequently asked questions
---

# FAQ

## Why are config IDs so long?

config_id = "<slug>__<hash8>" combines a human-readable slug with an 8-character SHA-256 digest of the canonical JSON parameters. This guarantees stability across Python versions and platforms.

## How do I ensure deterministic runs?

- Set a top-level seed (Generator.generate(..., seed=123)).
- Use YAML configs stored in version control.
- Avoid using 
p.random at module scope; rely on the provided ng in model code.

## What dims should I expect?

Every dataset is (time, channel, realization). Converters .pymts.to_numpy() and .pymts.to_dataframe() honour this order.

## I see NaNs after z-scoring—why?

Channel/realization pairs with zero variance produce zero std. pymts protects against division-by-zero by substituting 1.0, but if you z-score manually use where(std > 0, 1).

## Can I add extra attrs?

Yes. Attach them to the dataset before saving. Custom attrs will be merged into the Parquet metadata and JSON sidecar.

## How do CLI grids stay reproducible?

The CLI uses the same Generator API and SeedSequence.spawn() pathway. Each expanded config receives a deterministic sub-seed based on the base seed and its index.

## Will mkdocs-jupyter execute notebooks on every build?

Yes. Keep tutorials light (<30s). If an example requires longer runtimes, gate it behind a flag or mark cells with 	ags: ["skip-execution"].
