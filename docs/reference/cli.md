---
title: CLI reference
---

# Command-line interface

The pymts CLI wraps the Generator class with YAML grid expansion, deterministic seeding, and persistence helpers.

`
pymts --help
`

## generate

`
pymts generate --config CONFIG [--outdir OUTDIR] [--save/--no-save] [--csv/--no-csv] [--zscore/--no-zscore] [--dry-run] [--limit N]
`

### Options

| Option | Description |
| --- | --- |
| --config PATH | YAML or JSON configuration file. Required. |
| --outdir PATH | Override base output directory; defaults to data/<model>/<config_id>/. |
| --save/--no-save | Persist Parquet + metadata. Defaults to --save. |
| --csv/--no-csv | Write CSV alongside Parquet when saving. Off by default. |
| --zscore/--no-zscore | Apply channel-wise z-score after generation. Off by default. |
| --dry-run | Print expanded configurations without generating. |
| --limit N | Process only the first N expanded configs. |

### YAML grid format

`yaml
configs:
  - model: kuramoto
    M: [3]
    T: [64]
    K: [0.5, 0.9]
    seed: 123
  - model: gbm
    M: 2
    T: 64
    mu: [0.0, 0.05]
    sigma: 0.2
    dt: 0.02
    n_realizations: 1
    seed: 321
`

Lists trigger Cartesian products; scalars pass through unchanged. Reserved keys (save, csv, outdir, seed, zscore, zscore_axis, 
_realizations) are treated as runtime options, not model parameters.
