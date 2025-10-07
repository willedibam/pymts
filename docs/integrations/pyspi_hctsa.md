---
title: Integrating with PySPI and hctsa
---

# PySPI and hctsa integration guide

Both [PySPI](https://github.com/username/pyspi) and [hctsa](https://github.com/benfulcher/hctsa) expect long-form time-series tables. pymts provides .pymts.to_dataframe() which already yields the required columns.

## 1. Generate data and convert

`python
from pymts import Generator

gen = Generator()
config = {
    "model": "kuramoto",
    "M": 4,
    "T": 256,
    "K": 0.55,
    "seed": 123,
}

ds = gen.generate([config])[0]
df = ds.pymts.to_dataframe()
`

The DataFrame contains 	ime, channel, ealization, alue, config_id.

## 2. Prepare PySPI schema

PySPI typically expects unit, sensor, 	ime, alue columns. Map pymts columns accordingly:

`python
df_pyspi = df.rename(columns={
    "realization": "unit",
    "channel": "sensor",
    "value": "measurement",
})
`

Feed this DataFrame into PySPI's ingestion routines.

## 3. Prepare hctsa input

hctsa uses Matlab tables or CSV with series stored per file. A convenient approach is to export each channel/realization pair:

`python
for (unit, sensor), subset in df_pyspi.groupby(["unit", "sensor"]):
    subset[["time", "measurement"]].to_csv(
        f"hctsa_input/unit{unit}_sensor{sensor}.csv", index=False
    )
`

Point hctsa's import scripts to the resulting CSV folder.

## 4. Feature extraction example

`python
from pyspi import spi

spi_table = spi.calculate_features(df_pyspi, time_col="time", group_cols=["unit", "sensor"])
spi_table.head()
`

For hctsa, follow their MATLAB workflow to import the CSVs, compute features, and export summary statistics back into Python.

## 5. Determinism and scaling considerations

- Keep config_id alongside extracted features to tie results back to generator parameters.
- When expanding YAML grids, record the base seed; pymts handles sub-seeds automatically.
- For large cohorts, stream data to disk (CLI --save) and process features lazily to avoid loading entire arrays into memory.
