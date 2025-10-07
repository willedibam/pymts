from __future__ import annotations

import numpy as np

from pymts import Generator


def test_ar_shape_and_determinism() -> None:
    config = {
        "model": "ar",
        "M": 3,
        "T": 64,
        "p": 2,
        "phi": [0.6, -0.2],
        "noise_std": 0.1,
        "n_realizations": 2,
        "seed": 5,
    }

    gen = Generator()
    ds1 = gen.generate([config])[0]
    ds2 = gen.generate([config])[0]

    assert ds1["data"].shape == (64, 3, 2)
    np.testing.assert_allclose(ds1["data"].values[:10], ds2["data"].values[:10])


def test_var_shape_covariance_and_determinism() -> None:
    config = {
        "model": "var",
        "M": 2,
        "T": 64,
        "p": 1,
        "noise_scale": 0.1,
        "n_realizations": 2,
        "seed": 7,
    }

    gen = Generator()
    ds1 = gen.generate([config])[0]
    ds2 = gen.generate([config])[0]

    assert ds1["data"].shape == (64, 2, 2)
    np.testing.assert_allclose(ds1["data"].values[:10], ds2["data"].values[:10])

    final_slice = ds1["data"].isel(time=-1).values  # shape (2, 2)
    cov = np.cov(final_slice, rowvar=True)
    assert np.all(np.isfinite(cov))
