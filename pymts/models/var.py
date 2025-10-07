"""
Vector autoregressive (VAR) model implementation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import VARParams
from .base import BaseGenerator


class VARModel(BaseGenerator):
    """Vector autoregressive process with optional user-supplied coefficients."""

    @classmethod
    def help(cls) -> str:
        return "Vector autoregressive process with automatically stabilised coefficient matrices."

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return VARParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis  # handled by orchestrator

        params_obj = self._coerce_params(params)

        M = params_obj.M
        T = params_obj.T
        p = params_obj.p

        Phi = self._prepare_coefficients(params_obj, rng, M)
        Sigma, chol = self._prepare_covariance(params_obj, M)
        history = self._initial_history(params_obj, M, p, n_realizations)

        data = np.zeros((T, M, n_realizations), dtype=float)
        initial_len = min(p, T)
        data[:initial_len] = history[:initial_len]

        sqrt_dt_noise = chol  # alias for readability

        if T > p:
            for t in range(p, T):
                mean = np.zeros((M, n_realizations), dtype=float)
                for lag in range(1, p + 1):
                    past = data[t - lag]
                    mean += Phi[lag - 1] @ past
                innovations = sqrt_dt_noise @ rng.standard_normal(size=(M, n_realizations))
                data[t] = mean + innovations

        time = np.arange(T, dtype=int)
        channels = np.arange(M, dtype=int)
        realizations = np.arange(n_realizations, dtype=int)

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={"time": time, "channel": channels, "realization": realizations},
            attrs={
                "p": p,
                "noise_scale": params_obj.noise_scale,
            },
        )
        dataset.attrs["Sigma"] = Sigma.tolist()
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> VARParams:
        if isinstance(params, VARParams):
            return params
        if isinstance(params, BaseModel):
            return VARParams.model_validate(params.model_dump(mode="python"))
        return VARParams.model_validate(dict(params))

    def _prepare_coefficients(self, params: VARParams, rng: NPGenerator, M: int) -> np.ndarray:
        p = params.p
        if params.Phi is None:
            Phi = rng.normal(scale=0.25, size=(p, M, M))
        else:
            Phi = np.array(params.Phi, dtype=float, copy=True)
            if Phi.ndim == 2:
                Phi = Phi[None, :, :]
            if Phi.ndim != 3 or Phi.shape != (p, M, M):
                msg = "Phi must have shape (p, M, M) or (M, M) when p=1."
                raise ValueError(msg)

        Phi = _stabilize_var_matrices(Phi)
        return Phi

    def _prepare_covariance(self, params: VARParams, M: int) -> tuple[np.ndarray, np.ndarray]:
        if params.Sigma is None:
            Sigma = (params.noise_scale**2) * np.eye(M, dtype=float)
        else:
            Sigma = np.array(params.Sigma, dtype=float, copy=True)
            if Sigma.shape != (M, M):
                msg = "Sigma must have shape (M, M)."
                raise ValueError(msg)
        try:
            chol = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            msg = "Sigma must be positive definite."
            raise ValueError(msg) from exc
        return Sigma, chol

    def _initial_history(self, params: VARParams, M: int, p: int, R: int) -> np.ndarray:
        if params.x0 is None:
            return np.zeros((p, M, R), dtype=float)

        x0 = np.array(params.x0, dtype=float, copy=True)
        if x0.ndim == 1:
            if x0.shape[0] != M:
                msg = "x0 must have length M when provided as 1D."
                raise ValueError(msg)
            return np.broadcast_to(x0[None, :, None], (p, M, R))

        if x0.ndim == 2:
            if x0.shape == (p, M):
                return np.broadcast_to(x0[:, :, None], (p, M, R))
            if x0.shape == (M, p):
                return np.broadcast_to(x0.T[:, :, None], (p, M, R))
            if x0.shape == (M, R):
                return np.broadcast_to(x0[None, :, :], (p, M, R))
            msg = "x0 must be broadcastable to shape (p, M, R)."
            raise ValueError(msg)

        if x0.ndim == 3:
            if x0.shape == (p, M, R):
                return x0
            if x0.shape == (M, p, R):
                return np.transpose(x0, (1, 0, 2))
            msg = "x0 with three dimensions must match (p, M, R) or (M, p, R)."
            raise ValueError(msg)

        msg = "x0 must have between 1 and 3 dimensions."
        raise ValueError(msg)


def _stabilize_var_matrices(Phi: np.ndarray) -> np.ndarray:
    """Scale coefficient matrices so the companion matrix has spectral radius < 1."""

    stabilized = Phi.copy()
    p, M, _ = stabilized.shape

    for _ in range(30):
        companion = _companion_matrix(stabilized)
        eigvals = np.linalg.eigvals(companion)
        radius = np.max(np.abs(eigvals)) if eigvals.size else 0.0
        if radius < 0.95:
            return stabilized
        factor = 0.95 / max(radius, 1e-6)
        stabilized *= factor

    return stabilized * 0.0  # fall back to zeros if we fail to stabilise


def _companion_matrix(Phi: np.ndarray) -> np.ndarray:
    """Construct the VAR companion matrix used for stability checks."""

    p, M, _ = Phi.shape
    companion = np.zeros((M * p, M * p), dtype=float)
    companion[:M, : M * p] = np.hstack(Phi)
    if p > 1:
        companion[M:, :-M] = np.eye(M * (p - 1))
    return companion
