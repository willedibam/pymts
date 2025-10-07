"""
Parameter schemas and validation utilities for PyMTS models.

The classes defined here provide strongly typed contracts via Pydantic v2,
ensuring callers receive detailed validation errors before numerical routines
are executed.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Type, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BaseParams",
    "BaseModelParams",
    "KuramotoParams",
    "CMLParams",
    "ARParams",
    "VARParams",
    "OUParams",
    "GaussianNoiseParams",
    "CauchyNoiseParams",
    "BrownianParams",
    "ABMParams",
    "GBMParams",
    "schema_dict",
]

ParamsT = TypeVar("ParamsT", bound=BaseModel)


class BaseParams(BaseModel):
    """Common foundation for per-model parameter schemas."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, arbitrary_types_allowed=True)

    M: int = Field(default=5, ge=1, description="Number of channels (>=1).")
    T: int = Field(default=100, ge=1, description="Number of timesteps (>=1).")
    n_realizations: int = Field(default=1, ge=1, description="Number of independent realizations.")

    @field_validator("M", "T", "n_realizations")
    @classmethod
    def _ensure_positive(cls, value: int) -> int:
        if value < 1:
            msg = "M, T, and n_realizations must be positive integers."
            raise ValueError(msg)
        return value


# Backwards compatibility alias used in early scaffolding.
BaseModelParams = BaseParams


class KuramotoParams(BaseParams):
    """Parameterisation for the Kuramoto oscillator network."""

    model: Literal["kuramoto"] = "kuramoto"
    K: float = Field(default=1.0, description="Coupling strength.")
    dt: float = Field(default=0.05, gt=0, description="Integration step size.")
    method: Literal["rk4", "euler"] = Field(default="rk4", description="Numerical integrator.")

    omega_dist: Literal["normal", "uniform"] = Field(default="normal", description="Distribution for natural frequencies.")
    omega_mu: float = Field(default=0.0, description="Mean frequency (normal).")
    omega_sigma: float = Field(default=1.0, gt=0, description="Standard deviation for normal distribution.")
    omega_low: float = Field(default=-1.0, description="Lower bound for uniform distribution.")
    omega_high: float = Field(default=1.0, description="Upper bound for uniform distribution.")

    noise_std: float = Field(default=0.0, ge=0, description="Gaussian noise std applied to phase increments.")

    topology: Literal["complete", "ring", "erdos_renyi", "sbm"] = Field(default="complete", description="Network topology.")
    p: float | None = Field(default=0.1, gt=0, lt=1, description="Edge probability for Erdos-Renyi graphs.")
    block_sizes: list[int] | None = Field(default=None, description="Block sizes for stochastic block models.")
    p_intra: float | None = Field(default=None, gt=0, lt=1, description="Intra-block connection probability (SBM).")
    p_inter: float | None = Field(default=None, gt=0, lt=1, description="Inter-block connection probability (SBM).")

    A: np.ndarray | None = Field(default=None, description="Optional adjacency matrix overriding topology.")
    theta0: np.ndarray | None = Field(default=None, description="Optional initial phases (radians).")

    @field_validator("omega_high")
    @classmethod
    def _check_uniform_bounds(cls, value: float, values: dict[str, Any]) -> float:
        low = values.get("omega_low", -1.0)
        if value <= low:
            msg = "omega_high must be greater than omega_low."
            raise ValueError(msg)
        return value

    @field_validator("A", "theta0", mode="before")
    @classmethod
    def _coerce_ndarray(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return array.reshape(1)
        return array

    @field_validator("block_sizes", mode="before")
    @classmethod
    def _coerce_block_sizes(cls, value: Any) -> list[int] | None:
        if value is None:
            return None
        return [int(v) for v in value]

    @model_validator(mode="after")
    def _validate_topology(self) -> "KuramotoParams":
        if self.A is not None:
            if self.A.ndim != 2 or self.A.shape[0] != self.M or self.A.shape[1] != self.M:
                msg = "A must be a square (M x M) adjacency matrix."
                raise ValueError(msg)
            return self

        if self.topology == "erdos_renyi":
            if self.p is None:
                msg = "p must be provided for Erdos-Renyi topology."
                raise ValueError(msg)
        if self.topology == "sbm":
            if not self.block_sizes or sum(self.block_sizes) != self.M:
                msg = "block_sizes must be provided for SBM and sum to M."
                raise ValueError(msg)
            if self.p_intra is None or self.p_inter is None:
                msg = "p_intra and p_inter must be provided for SBM topology."
                raise ValueError(msg)

        if self.theta0 is not None:
            if self.theta0.ndim == 1 and self.theta0.shape[0] != self.M:
                msg = "theta0 must have length M."
                raise ValueError(msg)
            if self.theta0.ndim == 2 and self.theta0.shape not in {(self.M, self.n_realizations), (self.n_realizations, self.M)}:
                msg = "theta0 must have shape (M, n_realizations) or (n_realizations, M)."
                raise ValueError(msg)
            if self.theta0.ndim > 2:
                msg = "theta0 must be 1D or 2D."
                raise ValueError(msg)

        return self


class CMLParams(BaseParams):
    """Parameterisation for coupled map lattice models with logistic dynamics."""

    model: Literal["cml"] = "cml"
    r: float = Field(default=3.8, gt=0, description="Logistic map growth rate.")
    epsilon: float = Field(default=0.1, ge=0, le=1, description="Diffusive coupling strength.")
    topology: Literal["complete", "ring", "erdos_renyi", "sbm"] = Field(default="ring", description="Network topology.")
    p: float | None = Field(default=0.1, gt=0, lt=1, description="Edge probability for Erdos-Renyi graphs.")
    block_sizes: list[int] | None = Field(default=None, description="Block sizes for SBM topology.")
    p_intra: float | None = Field(default=None, gt=0, lt=1, description="Intra-block connection probability (SBM).")
    p_inter: float | None = Field(default=None, gt=0, lt=1, description="Inter-block connection probability (SBM).")
    boundary: Literal["periodic"] = Field(default="periodic", description="Boundary condition (periodic supported).")
    x0: np.ndarray | None = Field(default=None, description="Optional initial states in (0, 1).")

    @field_validator("x0", mode="before")
    @classmethod
    def _coerce_state(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return array.reshape(1)
        return array

    @field_validator("block_sizes", mode="before")
    @classmethod
    def _coerce_block_sizes(cls, value: Any) -> list[int] | None:
        if value is None:
            return None
        return [int(v) for v in value]

    @model_validator(mode="after")
    def _validate_topology(self) -> "CMLParams":
        if self.topology == "erdos_renyi" and self.p is None:
            msg = "p must be provided for Erdos-Renyi topology."
            raise ValueError(msg)
        if self.topology == "sbm":
            if not self.block_sizes or sum(self.block_sizes) != self.M:
                msg = "block_sizes must be provided for SBM and sum to M."
                raise ValueError(msg)
            if self.p_intra is None or self.p_inter is None:
                msg = "p_intra and p_inter must be provided for SBM topology."
                raise ValueError(msg)
        if self.x0 is not None:
            if np.any(self.x0 <= 0) or np.any(self.x0 >= 1):
                msg = "CML initial states must lie strictly within (0, 1)."
                raise ValueError(msg)
        return self


class ARParams(BaseParams):
    """Parameterisation for autoregressive (AR) processes."""

    model: Literal["ar"] = "ar"
    p: int = Field(default=1, ge=1, description="AR order.")
    phi: np.ndarray | None = Field(default=None, description="AR coefficients (shared or per-channel).")
    noise_std: float = Field(default=1.0, ge=0, description="Standard deviation of innovations.")
    x0: np.ndarray | None = Field(default=None, description="Initial history (length p).")

    @field_validator("phi", "x0", mode="before")
    @classmethod
    def _coerce_array(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return array.reshape(1)
        return array


class VARParams(BaseParams):
    """Parameterisation for vector autoregressive (VAR) processes."""

    model: Literal["var"] = "var"
    p: int = Field(default=1, ge=1, description="VAR order.")
    Phi: np.ndarray | None = Field(default=None, description="Coefficient matrices with shape (p, M, M).")
    Sigma: np.ndarray | None = Field(default=None, description="Innovation covariance matrix (M x M).")
    noise_scale: float = Field(default=1.0, ge=0, description="Scale factor for default covariance.")
    x0: np.ndarray | None = Field(default=None, description="Initial history (length p).")

    @field_validator("Phi", "Sigma", "x0", mode="before")
    @classmethod
    def _coerce_array(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return array.reshape(1)
        return array


class OUParams(BaseParams):
    """Parameterisation for Ornstein-Uhlenbeck processes with exact discretisation."""

    model: Literal["ou"] = "ou"
    mu: float | Sequence[float] = Field(default=0.0, description="Long-run mean.")
    theta: float | Sequence[float] = Field(default=1.0, description="Mean reversion rate.")
    sigma: float | Sequence[float] = Field(default=1.0, ge=0, description="Volatility coefficient.")
    dt: float = Field(default=1.0, gt=0, description="Time step.")
    x0: np.ndarray | None = Field(default=None, description="Optional initial state.")

    @field_validator("x0", mode="before")
    @classmethod
    def _coerce_x0(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return array.reshape(1)
        return array


class GaussianNoiseParams(BaseParams):
    """Parameterisation for independent Gaussian noise generators."""

    model: Literal["noise_gaussian"] = "noise_gaussian"
    sigma: float = Field(default=1.0, ge=0, description="Standard deviation.")


class CauchyNoiseParams(BaseParams):
    """Parameterisation for independent Cauchy noise generators."""

    model: Literal["cauchy_noise"] = "cauchy_noise"
    gamma: float = Field(default=1.0, gt=0, description="Scale parameter (gamma).")


class BrownianParams(BaseParams):
    """Brownian (Wiener) process parameters."""

    model: Literal["brownian"] = "brownian"
    dt: float = Field(default=1.0, gt=0, description="Time step.")
    sigma: float = Field(default=1.0, ge=0, description="Volatility coefficient.")
    X0: float | Sequence[float] | np.ndarray = Field(default=0.0, description="Initial value.")


class ABMParams(BaseParams):
    """Arithmetic Brownian motion parameters."""

    model: Literal["abm"] = "abm"
    dt: float = Field(default=1.0, gt=0, description="Time step.")
    mu: float = Field(default=0.0, description="Drift term.")
    sigma: float = Field(default=1.0, ge=0, description="Volatility coefficient.")
    X0: float | Sequence[float] | np.ndarray = Field(default=0.0, description="Initial value.")


class GBMParams(BaseParams):
    """Geometric Brownian motion parameters."""

    model: Literal["gbm"] = "gbm"
    dt: float = Field(default=1.0, gt=0, description="Time step.")
    mu: float = Field(default=0.0, description="Drift term.")
    sigma: float = Field(default=1.0, ge=0, description="Volatility coefficient.")
    S0: float | Sequence[float] | np.ndarray = Field(default=1.0, description="Initial positive value.")


def schema_dict(model: Type[ParamsT]) -> dict[str, Any]:
    """
    Convenience helper returning the JSON schema for a Pydantic model.
    """
    if not issubclass(model, BaseModel):
        msg = "schema_dict expects a Pydantic BaseModel subclass."
        raise TypeError(msg)
    return model.model_json_schema()
