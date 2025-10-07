"""
Coupled map lattice (CML) model implementation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from pydantic import BaseModel

from ..schemas import CMLParams
from .base import BaseGenerator


class CMLModel(BaseGenerator):
    """Logistic coupled map lattice with configurable topology."""

    @classmethod
    def help(cls) -> str:
        return (
            "Diffusively coupled logistic maps supporting complete, ring, "
            "Erdos-Renyi, and stochastic block topologies."
        )

    @classmethod
    def param_model(cls) -> type[BaseModel]:
        return CMLParams

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
        epsilon = params_obj.epsilon
        r = params_obj.r

        adjacency = self._build_adjacency(params_obj, rng)
        degrees = adjacency.sum(axis=1)
        safe_degrees = np.where(degrees > 0, degrees, 1.0)

        state = self._initial_state(params_obj, rng, M, n_realizations)
        data = np.empty((T, M, n_realizations), dtype=float)
        data[0] = state

        for t in range(1, T):
            f_state = r * state * (1.0 - state)
            neighbor_term = adjacency @ f_state
            neighbor_term = neighbor_term / safe_degrees[:, None]
            state = (1.0 - epsilon) * f_state + epsilon * neighbor_term
            state = np.clip(state, 0.0, 1.0)
            data[t] = state

        time = np.arange(T, dtype=int)
        channels = np.arange(M, dtype=int)
        realizations = np.arange(n_realizations, dtype=int)

        dataset = xr.Dataset(
            {"data": (("time", "channel", "realization"), data)},
            coords={"time": time, "channel": channels, "realization": realizations},
            attrs={
                "topology": params_obj.topology,
                "epsilon": epsilon,
                "r": r,
                "boundary": params_obj.boundary,
            },
        )
        return dataset

    @staticmethod
    def _coerce_params(params: Mapping[str, Any] | BaseModel) -> CMLParams:
        if isinstance(params, CMLParams):
            return params
        if isinstance(params, BaseModel):
            return CMLParams.model_validate(params.model_dump(mode="python"))
        return CMLParams.model_validate(dict(params))

    def _build_adjacency(self, params: CMLParams, rng: NPGenerator) -> np.ndarray:
        M = params.M
        topology = params.topology

        if topology == "complete":
            matrix = np.ones((M, M), dtype=float) - np.eye(M)
        elif topology == "ring":
            matrix = np.zeros((M, M), dtype=float)
            indices = np.arange(M)
            matrix[indices, (indices - 1) % M] = 1.0
            matrix[indices, (indices + 1) % M] = 1.0
        elif topology == "erdos_renyi":
            p = float(params.p if params.p is not None else 0.0)
            tri = rng.random((M, M))
            tri = (tri < p).astype(float)
            matrix = np.triu(tri, k=1)
            matrix = matrix + matrix.T
        elif topology == "sbm":
            block_sizes = params.block_sizes or []
            labels = np.repeat(np.arange(len(block_sizes)), block_sizes)
            p_intra = float(params.p_intra)
            p_inter = float(params.p_inter)
            prob = np.where(labels[:, None] == labels[None, :], p_intra, p_inter)
            tri = rng.random((M, M))
            matrix = np.triu((tri < prob).astype(float), k=1)
            matrix = matrix + matrix.T
        else:
            msg = f"Unsupported topology '{topology}'."
            raise ValueError(msg)

        np.fill_diagonal(matrix, 0.0)
        return matrix

    @staticmethod
    def _initial_state(params: CMLParams, rng: NPGenerator, M: int, R: int) -> np.ndarray:
        if params.x0 is None:
            return rng.uniform(1e-3, 1 - 1e-3, size=(M, R))

        x0 = np.array(params.x0, dtype=float, copy=True)
        if x0.ndim == 1:
            if x0.shape[0] != M:
                msg = "x0 must match number of channels (M)."
                raise ValueError(msg)
            return np.broadcast_to(x0[:, None], (M, R))

        if x0.ndim == 2:
            if x0.shape == (M, R):
                return x0
            if x0.shape == (R, M):
                return x0.T
            msg = "x0 must have shape (M,), (M, R), or (R, M)."
            raise ValueError(msg)

        msg = "x0 must have at most two dimensions."
        raise ValueError(msg)
