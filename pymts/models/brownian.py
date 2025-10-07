"""
Brownian-family process implementations (Brownian, ABM, GBM).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import ABMParams, BrownianParams, GBMParams
from .base import BaseGenerator


class BrownianModel(BaseGenerator):
    """Standard Wiener process with configurable volatility."""

    @classmethod
    def help(cls) -> str:
        return "Brownian (Wiener) process with configurable volatility and initial value."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return BrownianParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis

        params_obj = self._coerce_params(params)
        return _simulate_linear_process(
            rng=rng,
            params=params_obj,
            n_realizations=n_realizations,
            drift=0.0,
            diffusion=params_obj.sigma,
            initial=params_obj.X0,
            is_multiplicative=False,
        )

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> BrownianParams:
        if isinstance(params, BrownianParams):
            return params
        if isinstance(params, BaseModel):
            return BrownianParams.model_validate(params.model_dump(mode="python"))
        return BrownianParams.model_validate(dict(params))


class ABMModel(BaseGenerator):
    """Arithmetic Brownian motion with drift."""

    @classmethod
    def help(cls) -> str:
        return "Arithmetic Brownian motion with drift and volatility parameters."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return ABMParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis

        params_obj = self._coerce_params(params)
        return _simulate_linear_process(
            rng=rng,
            params=params_obj,
            n_realizations=n_realizations,
            drift=params_obj.mu,
            diffusion=params_obj.sigma,
            initial=params_obj.X0,
            is_multiplicative=False,
        )

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> ABMParams:
        if isinstance(params, ABMParams):
            return params
        if isinstance(params, BaseModel):
            return ABMParams.model_validate(params.model_dump(mode="python"))
        return ABMParams.model_validate(dict(params))


class GBMModel(BaseGenerator):
    """Geometric Brownian motion with drift and volatility."""

    @classmethod
    def help(cls) -> str:
        return "Geometric Brownian motion ensuring strictly positive trajectories."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return GBMParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis

        params_obj = self._coerce_params(params)
        return _simulate_geometric_process(
            rng=rng,
            params=params_obj,
            n_realizations=n_realizations,
        )

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> GBMParams:
        if isinstance(params, GBMParams):
            return params
        if isinstance(params, BaseModel):
            return GBMParams.model_validate(params.model_dump(mode="python"))
        return GBMParams.model_validate(dict(params))


def _simulate_linear_process(
    rng: NPGenerator,
    params: BrownianParams | ABMParams,
    n_realizations: int,
    drift: float,
    diffusion: float,
    initial: float | Sequence[float] | np.ndarray,
    is_multiplicative: bool,
) -> xr.Dataset:
    M = params.M
    T = params.T
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    init = _broadcast_initial(initial, M, n_realizations)
    data = np.zeros((T, M, n_realizations), dtype=float)
    data[0] = init

    for t in range(1, T):
        noise = rng.standard_normal(size=(M, n_realizations))
        increment = drift * dt + diffusion * sqrt_dt * noise
        if is_multiplicative:
            data[t] = data[t - 1] * np.exp(increment)
        else:
            data[t] = data[t - 1] + increment

    time = np.arange(T, dtype=float) * dt
    channels = np.arange(M, dtype=int)
    realizations = np.arange(n_realizations, dtype=int)

    dataset = xr.Dataset(
        {"data": (("time", "channel", "realization"), data)},
        coords={"time": time, "channel": channels, "realization": realizations},
        attrs={
            "dt": dt,
            "mu": drift,
            "sigma": diffusion,
        },
    )
    return dataset


def _simulate_geometric_process(
    rng: NPGenerator,
    params: GBMParams,
    n_realizations: int,
) -> xr.Dataset:
    M = params.M
    T = params.T
    dt = params.dt
    sqrt_dt = np.sqrt(dt)
    mu = params.mu
    sigma = params.sigma

    init = _broadcast_initial(params.S0, M, n_realizations)
    if np.any(init <= 0):
        msg = "GBM initial values must be strictly positive."
        raise ValueError(msg)

    data = np.zeros((T, M, n_realizations), dtype=float)
    data[0] = init

    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_scale = sigma * sqrt_dt

    for t in range(1, T):
        noise = rng.standard_normal(size=(M, n_realizations))
        data[t] = data[t - 1] * np.exp(drift_term + diffusion_scale * noise)

    time = np.arange(T, dtype=float) * dt
    channels = np.arange(M, dtype=int)
    realizations = np.arange(n_realizations, dtype=int)

    dataset = xr.Dataset(
        {"data": (("time", "channel", "realization"), data)},
        coords={"time": time, "channel": channels, "realization": realizations},
        attrs={
            "dt": dt,
            "mu": mu,
            "sigma": sigma,
        },
    )
    return dataset


def _broadcast_initial(initial: float | Sequence[float] | np.ndarray, M: int, R: int) -> np.ndarray:
    array = np.asarray(initial, dtype=float)
    if array.ndim == 0:
        return np.full((M, R), float(array))
    if array.ndim == 1:
        if array.shape[0] == M:
            return np.broadcast_to(array[:, None], (M, R))
        if array.shape[0] == R:
            return np.broadcast_to(array[None, :], (M, R))
        msg = "Initial array must match number of channels or realizations."
        raise ValueError(msg)
    if array.ndim == 2:
        if array.shape == (M, R):
            return array
        if array.shape == (R, M):
            return array.T
        msg = "Initial array must be broadcastable to shape (M, R)."
        raise ValueError(msg)
    msg = "Initial array must have at most two dimensions."
    raise ValueError(msg)
