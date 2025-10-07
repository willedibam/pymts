from __future__ import annotations

import numpy as np

from pymts import Generator


def test_cml_shape_range_and_determinism() -> None:
    config = {
        "model": "cml",
        "M": 6,
        "T": 64,
        "r": 3.8,
        "epsilon": 0.15,
        "topology": "ring",
        "n_realizations": 2,
        "seed": 11,
    }

    gen = Generator()
    ds1 = gen.generate([config])[0]
    ds2 = gen.generate([config])[0]

    assert ds1["data"].shape == (64, 6, 2)
    np.testing.assert_allclose(ds1["data"].values[:8], ds2["data"].values[:8])

    steady = ds1["data"].values[5:]
    assert np.all(steady >= -1e-8)
    assert np.all(steady <= 1.0 + 1e-8)
