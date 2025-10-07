from __future__ import annotations

import numpy as np

from pymts import Generator


def test_brownian_family_models() -> None:
    gen = Generator()

    brownian_cfg = {
        "model": "brownian",
        "M": 2,
        "T": 64,
        "dt": 0.1,
        "sigma": 0.5,
        "n_realizations": 3,
        "seed": 3,
    }
    abm_cfg = {
        "model": "abm",
        "M": 2,
        "T": 64,
        "dt": 0.1,
        "mu": 0.2,
        "sigma": 0.3,
        "n_realizations": 2,
        "seed": 4,
    }
    gbm_cfg = {
        "model": "gbm",
        "M": 2,
        "T": 64,
        "dt": 0.1,
        "mu": 0.1,
        "sigma": 0.2,
        "S0": 1.0,
        "n_realizations": 2,
        "seed": 4,
    }

    ds_brownian = gen.generate([brownian_cfg])[0]
    ds_abm = gen.generate([abm_cfg])[0]
    ds_gbm = gen.generate([gbm_cfg])[0]

    assert ds_brownian["data"].shape == (64, 2, 3)
    assert ds_abm["data"].shape == (64, 2, 2)
    assert ds_gbm["data"].shape == (64, 2, 2)

    ds_brownian_repeat = gen.generate([brownian_cfg])[0]
    np.testing.assert_allclose(ds_brownian["data"].values[:10], ds_brownian_repeat["data"].values[:10])

    assert np.all(ds_gbm["data"].values > 0.0)
