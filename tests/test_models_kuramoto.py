from __future__ import annotations

import numpy as np

from pymts import Generator


def test_kuramoto_shape_and_determinism() -> None:
    config = {
        "model": "kuramoto",
        "M": 4,
        "T": 64,
        "K": 1.0,
        "dt": 0.05,
        "method": "rk4",
        "n_realizations": 2,
        "seed": 123,
    }

    gen = Generator()
    ds = gen.generate([config])[0]

    assert ds["data"].shape == (64, 4, 2)
    assert ds["data"].dims == ("time", "channel", "realization")

    ds_repeat = gen.generate([config])[0]
    np.testing.assert_allclose(ds["data"].values[:5], ds_repeat["data"].values[:5])


def test_kuramoto_topologies() -> None:
    gen = Generator()

    config_ring = {
        "model": "kuramoto",
        "M": 3,
        "T": 16,
        "K": 0.8,
        "topology": "ring",
        "n_realizations": 1,
        "seed": 321,
    }
    ds_ring = gen.generate([config_ring])[0]
    assert ds_ring["data"].shape == (16, 3, 1)

    config_er = {
        "model": "kuramoto",
        "M": 3,
        "T": 16,
        "K": 0.5,
        "topology": "erdos_renyi",
        "p": 0.2,
        "n_realizations": 1,
        "seed": 654,
    }
    ds_er = gen.generate([config_er])[0]
    assert ds_er["data"].shape == (16, 3, 1)
