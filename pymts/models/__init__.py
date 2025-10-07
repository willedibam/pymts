"""
Model registry exports for PyMTS.

Concrete implementations are introduced in later steps; for now we expose the
available generator classes to keep the public API stable.
"""

from __future__ import annotations

from .ar import ARModel
from .base import BaseGenerator
from .brownian import ABMModel, BrownianModel, GBMModel
from .cml import CMLModel
from .kuramoto import KuramotoModel
from .noise import CauchyNoiseModel, GaussianNoiseModel
from .ou import OUModel
from .var import VARModel

REGISTRY: dict[str, BaseGenerator] = {
    "kuramoto": KuramotoModel(),
    "cml": CMLModel(),
    "ar": ARModel(),
    "var": VARModel(),
    "ou": OUModel(),
    "brownian": BrownianModel(),
    "abm": ABMModel(),
    "gbm": GBMModel(),
    "noise_gaussian": GaussianNoiseModel(),
    "cauchy_noise": CauchyNoiseModel(),
}

__all__ = [
    "BaseGenerator",
    "KuramotoModel",
    "CMLModel",
    "ARModel",
    "VARModel",
    "OUModel",
    "BrownianModel",
    "ABMModel",
    "GBMModel",
    "GaussianNoiseModel",
    "CauchyNoiseModel",
    "REGISTRY",
]
