"""
Independent noise model generators.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import CauchyNoiseParams, GaussianNoiseParams
from .base import BaseGenerator


class GaussianNoiseModel(BaseGenerator):
    """Independent Gaussian noise with configurable variance."""

    @classmethod
    def help(cls) -> str:
        return "Independent Gaussian noise with configurable standard deviation."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return GaussianNoiseParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis  # handled in the orchestrator

        params_obj = self._coerce_params(params)
        data = rng.normal(
            loc=0.0,
            scale=float(params_obj.sigma),
            size=(params_obj.T, params_obj.M, n_realizations),
        )

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={
                "time": np.arange(params_obj.T, dtype=int),
                "channel": np.arange(params_obj.M, dtype=int),
                "realization": np.arange(n_realizations, dtype=int),
            },
            attrs={"distribution": "gaussian", "sigma": float(params_obj.sigma)},
        )
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> GaussianNoiseParams:
        if isinstance(params, GaussianNoiseParams):
            return params
        if isinstance(params, BaseModel):
            return GaussianNoiseParams.model_validate(params.model_dump(mode="python"))
        return GaussianNoiseParams.model_validate(dict(params))


class CauchyNoiseModel(BaseGenerator):
    """Independent Cauchy noise with configurable scale."""

    @classmethod
    def help(cls) -> str:
        return "Independent Cauchy noise with configurable scale parameter."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return CauchyNoiseParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis  # handled in the orchestrator

        params_obj = self._coerce_params(params)
        data = rng.standard_cauchy(size=(params_obj.T, params_obj.M, n_realizations)) * float(params_obj.gamma)

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={
                "time": np.arange(params_obj.T, dtype=int),
                "channel": np.arange(params_obj.M, dtype=int),
                "realization": np.arange(n_realizations, dtype=int),
            },
            attrs={"distribution": "cauchy", "gamma": float(params_obj.gamma)},
        )
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> CauchyNoiseParams:
        if isinstance(params, CauchyNoiseParams):
            return params
        if isinstance(params, BaseModel):
            return CauchyNoiseParams.model_validate(params.model_dump(mode="python"))
        return CauchyNoiseParams.model_validate(dict(params))
