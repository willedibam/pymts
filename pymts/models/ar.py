"""
Autoregressive (AR) model implementation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import ARParams
from .base import BaseGenerator


class ARModel(BaseGenerator):
    """Independent autoregressive processes per channel."""

    @classmethod
    def help(cls) -> str:
        return "Autoregressive process with optional user-supplied coefficients and history."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return ARParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis  # handled upstream

        params_obj = self._coerce_params(params)

        M = params_obj.M
        T = params_obj.T
        p = params_obj.p

        phi = self._prepare_coefficients(params_obj, rng, M)
        history = self._initial_history(params_obj, M, p, n_realizations)

        data = np.zeros((T, M, n_realizations), dtype=float)
        initial_len = min(p, T)
        data[:initial_len] = history[:initial_len]

        if T > p:
            for t in range(p, T):
                past = np.stack([data[t - k - 1] for k in range(p)], axis=0)  # (p, M, R)
                past = np.transpose(past, (1, 0, 2))  # (M, p, R)
                deterministic = np.sum(phi[:, :, None] * past, axis=1)
                noise = rng.normal(0.0, params_obj.noise_std, size=(M, n_realizations))
                data[t] = deterministic + noise

        time = np.arange(T, dtype=int)
        channels = np.arange(M, dtype=int)
        realizations = np.arange(n_realizations, dtype=int)

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={"time": time, "channel": channels, "realization": realizations},
            attrs={
                "p": p,
                "noise_std": params_obj.noise_std,
            },
        )
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> ARParams:
        if isinstance(params, ARParams):
            return params
        if isinstance(params, BaseModel):
            return ARParams.model_validate(params.model_dump(mode="python"))
        return ARParams.model_validate(dict(params))

    def _prepare_coefficients(self, params: ARParams, rng: NPGenerator, M: int) -> np.ndarray:
        p = params.p
        if params.phi is None:
            phi = rng.normal(scale=0.5, size=(M, p))
            phi = self._stabilize_matrix(phi)
            return phi

        phi = np.array(params.phi, dtype=float, copy=True)
        if phi.ndim == 1:
            if phi.shape[0] != p:
                msg = "phi must have length equal to p."
                raise ValueError(msg)
            phi = np.tile(phi[None, :], (M, 1))
        elif phi.ndim == 2:
            if phi.shape == (M, p):
                pass
            elif phi.shape == (p, M):
                phi = phi.T
            else:
                msg = "phi must have shape (p,), (M, p), or (p, M)."
                raise ValueError(msg)
        else:
            msg = "phi must be 1D or 2D."
            raise ValueError(msg)

        phi = self._stabilize_matrix(phi)
        return phi

    @staticmethod
    def _stabilize_matrix(phi: np.ndarray) -> np.ndarray:
        stabilized = phi.copy()
        for idx in range(stabilized.shape[0]):
            stabilized[idx] = _stabilize_ar_vector(stabilized[idx])
        return stabilized

    def _initial_history(self, params: ARParams, M: int, p: int, R: int) -> np.ndarray:
        if params.x0 is None:
            return np.zeros((p, M, R), dtype=float)

        x0 = np.array(params.x0, dtype=float, copy=True)
        if x0.ndim == 1:
            if x0.shape[0] != p:
                msg = "x0 must have length equal to p."
                raise ValueError(msg)
            return np.broadcast_to(x0[:, None, None], (p, M, R))

        if x0.ndim == 2:
            if x0.shape == (p, M):
                return np.broadcast_to(x0[:, :, None], (p, M, R))
            if x0.shape == (M, p):
                return np.broadcast_to(x0.T[:, :, None], (p, M, R))
            if x0.shape == (p, R):
                return np.broadcast_to(x0[:, None, :], (p, M, R))
            msg = "x0 must have shape (p,), (p, M), (M, p), or (p, R)."
            raise ValueError(msg)

        if x0.ndim == 3:
            if x0.shape == (p, M, R):
                return x0
            if x0.shape == (M, p, R):
                return np.transpose(x0, (1, 0, 2))
            if x0.shape == (p, R, M):
                return np.transpose(x0, (0, 2, 1))
            msg = "x0 must be broadcastable to shape (p, M, R)."
            raise ValueError(msg)

        msg = "x0 must have between 1 and 3 dimensions."
        raise ValueError(msg)


def _stabilize_ar_vector(coeffs: np.ndarray) -> np.ndarray:
    """Scale AR coefficients until all roots lie outside the unit circle."""

    stabilized = coeffs.copy()
    if np.allclose(stabilized, 0):
        return stabilized

    for _ in range(20):
        poly = np.concatenate(([1.0], -stabilized))
        roots = np.roots(poly)
        if roots.size == 0:
            return stabilized
        max_abs = np.max(np.abs(roots))
        if max_abs < 0.98:
            return stabilized
        stabilized /= (max_abs + 0.05)
    return np.zeros_like(stabilized)
