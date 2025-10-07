"""
Base interfaces for PyMTS model implementations.

Concrete models implement ``generate_one`` along with human- and machine-readable
metadata helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel


class BaseGenerator(ABC):
    """
    Base protocol for model generators.

    Implementations are responsible for turning validated parameters into xarray
    datasets following the ``(time, channel, realization)`` convention.
    """

    @classmethod
    @abstractmethod
    def help(cls) -> str:
        """Return human-readable documentation for the model."""

    @classmethod
    @abstractmethod
    def param_model(cls) -> type[BaseModel]:
        """Return the Pydantic parameter model associated with this generator."""

    @classmethod
    def params(cls) -> Mapping[str, Any]:
        """Return a machine-readable parameter schema."""

        return cls.param_model().model_json_schema()

    @abstractmethod
    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        """
        Generate a dataset for the given parameterisation.

        Parameters
        ----------
        params:
            Validated parameter mapping or Pydantic model instance.
        rng:
            NumPy random number generator seeded per configuration.
        n_realizations:
            Number of independent realizations to synthesise.
        zscore:
            Whether to z-score the output (handled by the orchestrator).
        zscore_axis:
            Axis across which to apply z-scoring (handled by the orchestrator).
        """
