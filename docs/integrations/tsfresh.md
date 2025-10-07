---
title: tsfresh feature extraction
---

# Extracting features with tsfresh

ts fresh expects long-form data with identifier columns. pymts provides the necessary structure out of the box.

## 1. Generate sample data

`python
from pymts import Generator

gen = Generator()
noise_cfg = {
    "model": "noise_gaussian",
    "M": 4,
    "T": 256,
    "sigma": 0.5,
    "n_realizations": 3,
    "seed": 2024,
}
noise_ds = gen.generate([noise_cfg])[0]
noise_df = noise_ds.pymts.to_dataframe()
`

## 2. Prepare tsfresh input

Rename columns to the expected identifiers:

`python
from tsfresh import extract_features

long_df = noise_df.rename(columns={
    "config_id": "id",
    "time": "time",
    "value": "value",
    "channel": "kind",
    "realization": "unit",
})
`

tsfresh needs id, 	ime, kind, and alue. We combine unit and kind if we want channel-level features per realization:

`python
long_df["id"] = long_df["unit"].astype(str) + "_" + long_df["kind"].astype(str)
features = extract_features(
    long_df,
    column_id="id",
    column_sort="time",
    column_kind="kind",
    column_value="value",
    disable_progressbar=True,
)
`

## 3. Select relevant features

`python
from tsfresh.feature_selection import select_features

labels = long_df.drop_duplicates("id")["id"].apply(lambda x: int("ou" in x)).values
selected = select_features(features, labels)
selected.head()
`

## 4. Scaling up

- Use the CLI to generate large cohorts, persist to Parquet, then load slices for feature extraction.
- Keep config_id as a column to link extracted feature rows back to generator parameters and seeds.
- tsfresh computations can be heavy; start with modest T and M, then scale once you are confident in the pipeline.
