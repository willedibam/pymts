"""
Ornstein-Uhlenbeck (OU) model generator.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import OUParams
from .base import BaseGenerator


class OUModel(BaseGenerator):
    """Mean-reverting Ornstein-Uhlenbeck processes with exact discretisation."""

    @classmethod
    def help(cls) -> str:
        return (
            "Ornstein-Uhlenbeck process with exact discretisation and optional "
            "vectorised initial conditions."
        )

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return OUParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis  # handled by the orchestrator

        params_obj = self._coerce_params(params)

        M = params_obj.M
        T = params_obj.T
        dt = params_obj.dt

        theta = self._expand_parameter(params_obj.theta, M)
        mu = self._expand_parameter(params_obj.mu, M)
        sigma = self._expand_parameter(params_obj.sigma, M)

        phi = np.exp(-theta * dt)
        sigma_eff = np.empty_like(theta, dtype=float)
        positive_mask = theta > 0
        sigma_eff[positive_mask] = sigma[positive_mask] * np.sqrt(
            (1.0 - phi[positive_mask] ** 2) / (2.0 * theta[positive_mask])
        )
        sigma_eff[~positive_mask] = sigma[~positive_mask] * np.sqrt(dt)

        state = self._initial_state(params_obj, M, n_realizations)

        data = np.empty((T, M, n_realizations), dtype=float)
        data[0] = state

        mu_expanded = mu[:, None]
        phi_expanded = phi[:, None]
        sigma_expanded = sigma_eff[:, None]

        for t in range(1, T):
            noise = rng.standard_normal(size=state.shape)
            state = mu_expanded + (state - mu_expanded) * phi_expanded + sigma_expanded * noise
            data[t] = state

        time = np.arange(T, dtype=float) * dt
        channels = np.arange(M, dtype=int)
        realizations = np.arange(n_realizations, dtype=int)

        attrs = {
            "dt": dt,
            "theta": theta.tolist(),
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
        }

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={"time": time, "channel": channels, "realization": realizations},
            attrs=attrs,
        )
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> OUParams:
        if isinstance(params, OUParams):
            return params
        if isinstance(params, BaseModel):
            return OUParams.model_validate(params.model_dump(mode="python"))
        return OUParams.model_validate(dict(params))

    @staticmethod
    def _expand_parameter(value: Any, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return np.full(size, float(array))
        if array.shape != (size,):
            msg = f"Parameter must be scalar or length {size}."
            raise ValueError(msg)
        return array

    @staticmethod
    def _initial_state(params: OUParams, M: int, n_realizations: int) -> np.ndarray:
        if params.x0 is None:
            return np.zeros((M, n_realizations), dtype=float)

        x0 = np.array(params.x0, dtype=float, copy=True)
        if x0.ndim == 1:
            if x0.shape[0] != M:
                msg = "x0 must have length M."
                raise ValueError(msg)
            return np.tile(x0[:, None], (1, n_realizations))

        if x0.shape == (n_realizations, M):
            x0 = x0.T
        if x0.shape != (M, n_realizations):
            msg = "x0 must have shape (M,) or (M, n_realizations)."
            raise ValueError(msg)
        return x0
