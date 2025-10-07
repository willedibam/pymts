from __future__ import annotations

import numpy as np

from pymts import Generator


def test_zscore_normalises_per_channel_realization() -> None:
    config = {
        "model": "noise_gaussian",
        "M": 3,
        "T": 24,
        "sigma": 0.5,
        "n_realizations": 2,
        "seed": 99,
    }

    gen = Generator()
    ds = gen.generate([config], zscore=True)[0]
    data = ds["data"].values  # shape (time, channel, realization)

    means = data.mean(axis=0)
    stds = data.std(axis=0)

    assert np.allclose(means, 0.0, atol=1e-7)
    assert np.allclose(stds, 1.0, atol=1e-7)
