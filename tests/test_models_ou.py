from __future__ import annotations

import numpy as np

from pymts import Generator


def test_ou_shape_mean_and_determinism() -> None:
    config = {
        "model": "ou",
        "M": 3,
        "T": 32,
        "mu": 1.0,
        "theta": 0.7,
        "sigma": 0.5,
        "dt": 0.1,
        "n_realizations": 4,
        "seed": 42,
    }

    gen = Generator()
    ds = gen.generate([config])[0]

    assert ds["data"].shape == (32, 3, 4)
    assert ds["data"].dims == ("time", "channel", "realization")

    final_mean = float(ds["data"].isel(time=-1).mean().item())
    assert abs(final_mean - 1.0) < 0.4

    ds_repeat = gen.generate([config])[0]
    np.testing.assert_allclose(ds["data"].values, ds_repeat["data"].values)
