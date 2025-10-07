---
title: scikit-learn pipelines
---

# Using pymts with scikit-learn

This guide shows how to transform pymts outputs into feature matrices for scikit-learn classification and regression pipelines.

## 1. Generate labelled datasets

We create two configs with different dynamics and store labels manually.

`python
from pymts import Generator
import numpy as np

gen = Generator()
configs = [
    {"model": "kuramoto", "M": 4, "T": 128, "K": 0.5, "seed": 101},
    {"model": "ou", "M": 4, "T": 128, "theta": 0.9, "seed": 202},
]
labels = [0, 1]

datasets = gen.generate(configs)
`

## 2. Flatten each dataset into a feature vector

A simple approach is to compute summary statistics per channel/realization.

`python
features = []
targets = []

for ds, label in zip(datasets, labels):
    df = ds.pymts.to_dataframe()
    summary = df.groupby(["channel", "realization"]) ["value"].agg(["mean", "std", "max", "min"])
    features.append(summary.values.flatten())
    targets.append(label)

X = np.vstack(features)
y = np.array(targets)
`

## 3. Build a pipeline

`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline.fit(X, y)
`

Add cross-validation, feature unions, or time-series embeddings as needed.

## 4. Working with large grids

- Use the CLI to generate batches and persist them with metadata.
- Stream DataFrame chunks to avoid loading entire datasets into RAM.
- Keep config_id in your feature tables so you can trace back to generator parameters.

## 5. Reproducibility

Because pymts assigns sub-seeds via SeedSequence.spawn(), re-running the pipeline with the same configurations reproduces identical features and model performance (subject to scikit-learn RNG settings).
