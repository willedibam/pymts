from __future__ import annotations

import pymts
from pymts import Generator, version


def test_imports() -> None:
    assert isinstance(version, str)
    assert version == pymts.__version__
    assert Generator is pymts.Generator
