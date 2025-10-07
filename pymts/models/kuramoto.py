"""
Kuramoto oscillator network model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import KuramotoParams
from .base import BaseGenerator


class KuramotoModel(BaseGenerator):
    """Coupled phase oscillators with configurable topology and integrator."""

    @classmethod
    def help(cls) -> str:
        return (
            "Kuramoto oscillator network with configurable topology, natural "
            "frequency distributions, and integration schemes (RK4 or Euler)."
        )

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return KuramotoParams

    def generate_one(
        self,
        params: Mapping[str, Any] | BaseModel,
        rng: NPGenerator,
        *,
        n_realizations: int,
        zscore: bool,
        zscore_axis: str | None,
    ) -> xr.Dataset:
        del zscore, zscore_axis  # handled in the orchestration layer

        params_obj = self._coerce_params(params)

        M = params_obj.M
        T = params_obj.T
        dt = params_obj.dt
        method = params_obj.method

        adjacency = self._build_adjacency(params_obj, rng)
        omega = self._draw_frequencies(params_obj, rng, M)
        theta = self._initial_phases(params_obj, rng, M, n_realizations)
        noise_std = float(params_obj.noise_std)
        sqrt_dt = np.sqrt(dt)

        data = np.empty((T, M, n_realizations), dtype=float)
        data[0] = theta

        def coupling(phases: np.ndarray) -> np.ndarray:
            """Compute deterministic phase derivatives for all oscillators."""

            # phases shape: (M, R)
            diff = phases[None, :, :] - phases[:, None, :]
            interaction = np.sin(diff)
            weighted = adjacency[:, :, None] * interaction
            return omega[:, None] + (params_obj.K / M) * weighted.sum(axis=1)

        for t in range(1, T):
            if method == "euler":
                drift = coupling(theta)
                if noise_std > 0:
                    noise = noise_std * sqrt_dt * rng.standard_normal(size=theta.shape)
                else:
                    noise = 0.0
                theta = theta + dt * drift + noise
            else:  # rk4
                k1 = coupling(theta)
                k2 = coupling(theta + 0.5 * dt * k1)
                k3 = coupling(theta + 0.5 * dt * k2)
                k4 = coupling(theta + dt * k3)
                increment = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                if noise_std > 0:
                    noise = noise_std * sqrt_dt * rng.standard_normal(size=theta.shape)
                else:
                    noise = 0.0
                theta = theta + increment + noise

            data[t] = theta

        time = np.arange(T, dtype=float) * dt
        channels = np.arange(M, dtype=int)
        realizations = np.arange(n_realizations, dtype=int)

        attrs = {
            "dt": dt,
            "method": method,
            "topology": params_obj.topology,
        }

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={"time": time, "channel": channels, "realization": realizations},
            attrs=attrs,
        )
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> KuramotoParams:
        if isinstance(params, KuramotoParams):
            return params
        if isinstance(params, BaseModel):
            return KuramotoParams.model_validate(params.model_dump(mode="python"))
        return KuramotoParams.model_validate(dict(params))

    def _build_adjacency(self, params: KuramotoParams, rng: NPGenerator) -> np.ndarray:
        if params.A is not None:
            matrix = np.array(params.A, dtype=float, copy=True)
            matrix = 0.5 * (matrix + matrix.T)
        else:
            topology = params.topology
            if topology == "complete":
                matrix = np.ones((params.M, params.M), dtype=float) - np.eye(params.M)
            elif topology == "ring":
                matrix = np.zeros((params.M, params.M), dtype=float)
                indices = np.arange(params.M)
                matrix[indices, (indices - 1) % params.M] = 1.0
                matrix[indices, (indices + 1) % params.M] = 1.0
            elif topology == "erdos_renyi":
                p = float(params.p if params.p is not None else 0.0)
                upper = rng.random((params.M, params.M))
                matrix = (upper < p).astype(float)
                matrix = np.triu(matrix, k=1)
                matrix = matrix + matrix.T
            elif topology == "sbm":
                block_sizes = params.block_sizes or []
                labels = np.repeat(np.arange(len(block_sizes)), block_sizes)
                p_intra = float(params.p_intra)
                p_inter = float(params.p_inter)
                prob = np.where(
                    labels[:, None] == labels[None, :],
                    p_intra,
                    p_inter,
                )
                random = rng.random((params.M, params.M))
                matrix = (random < prob).astype(float)
                matrix = np.triu(matrix, k=1)
                matrix = matrix + matrix.T
            else:
                msg = f"Unsupported topology '{topology}'."
                raise ValueError(msg)

        np.fill_diagonal(matrix, 0.0)
        return matrix

    @staticmethod
    def _draw_frequencies(params: KuramotoParams, rng: NPGenerator, size: int) -> np.ndarray:
        if params.omega_dist == "normal":
            return rng.normal(loc=params.omega_mu, scale=params.omega_sigma, size=size)
        return rng.uniform(low=params.omega_low, high=params.omega_high, size=size)

    @staticmethod
    def _initial_phases(
        params: KuramotoParams,
        rng: NPGenerator,
        M: int,
        n_realizations: int,
    ) -> np.ndarray:
        if params.theta0 is None:
            return rng.uniform(0.0, 2.0 * np.pi, size=(M, n_realizations))

        theta0 = np.array(params.theta0, dtype=float, copy=True)
        if theta0.ndim == 1:
            theta = np.tile(theta0[:, None], (1, n_realizations))
        else:
            if theta0.shape == (n_realizations, M):
                theta0 = theta0.T
            if theta0.shape != (M, n_realizations):
                msg = "theta0 must have shape (M,) or (M, n_realizations)."
                raise ValueError(msg)
            theta = theta0
        return theta
