"""
Top-level package for the PyMTS synthetic multivariate time series library.

This module exposes the public package version and the high-level ``Generator``
interface that orchestrates model execution and dataset assembly.
"""

from __future__ import annotations

from .core import Generator
from . import accessors as _accessors  # noqa: F401  # ensure accessor registration

__all__ = ["Generator", "__version__", "version"]

__version__ = "0.1.0"
version = __version__
