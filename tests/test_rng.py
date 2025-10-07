from __future__ import annotations

import numpy as np

from pymts.core import spawn_subrngs


def test_spawn_subrngs_deterministic() -> None:
    rngs = spawn_subrngs(seed=123, n=2)
    assert len(rngs) == 2

    draws_first = rngs[0].integers(0, 1000, size=5)
    draws_second = rngs[1].integers(0, 1000, size=5)

    expected_first = np.array([959, 193, 29, 754, 436], dtype=np.int64)
    expected_second = np.array([739, 552, 43, 228, 591], dtype=np.int64)

    np.testing.assert_array_equal(draws_first, expected_first)
    np.testing.assert_array_equal(draws_second, expected_second)
    assert not np.array_equal(draws_first, draws_second)
