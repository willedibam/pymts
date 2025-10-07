from __future__ import annotations

import numpy as np

from pymts import Generator


def test_gaussian_noise_shape_and_determinism() -> None:
    config = {
        "model": "noise_gaussian",
        "M": 2,
        "T": 16,
        "sigma": 0.3,
        "n_realizations": 3,
        "seed": 7,
    }

    gen = Generator()
    ds = gen.generate([config])[0]

    assert ds["data"].shape == (16, 2, 3)
    assert ds["data"].dims == ("time", "channel", "realization")

    ds_repeat = gen.generate([config])[0]
    np.testing.assert_allclose(ds["data"].values, ds_repeat["data"].values)


def test_cauchy_noise_shape_and_determinism() -> None:
    config = {
        "model": "cauchy_noise",
        "M": 2,
        "T": 16,
        "gamma": 0.8,
        "n_realizations": 3,
        "seed": 7,
    }

    gen = Generator()
    ds = gen.generate([config])[0]

    assert ds["data"].shape == (16, 2, 3)
    assert ds["data"].dims == ("time", "channel", "realization")

    ds_repeat = gen.generate([config])[0]
    np.testing.assert_allclose(ds["data"].values, ds_repeat["data"].values)
