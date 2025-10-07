"""
Core orchestration logic for PyMTS generation workflows.

Step 2 wires together configuration normalisation, RNG spawning, optional
post-processing hooks, and persistence. Model implementations remain stubs for
now; the pipeline raises a friendly ``NotImplementedError`` when invoked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import xarray as xr
from numpy.random import Generator as NPGenerator
from numpy.random import SeedSequence
from pydantic import BaseModel

from .io import save_csv, save_parquet, write_sidecar_metadata
from .models import REGISTRY
from .models.base import BaseGenerator
from .utils import canonical_json, hash8_from_obj, slugify_config

__all__ = ["Generator", "spawn_subrngs", "build_config_id"]


def spawn_subrngs(seed: int | None, n: int) -> list[NPGenerator]:
    """
    Spawn ``n`` independent NumPy generators from a base seed.

    Parameters
    ----------
    seed:
        Base seed used to initialise the :class:`numpy.random.SeedSequence`.
        ``None`` delegates to entropy from the OS.
    n:
        Number of sub-generators to spawn. Must be non-negative.
    """

    if n < 0:
        msg = "n must be non-negative."
        raise ValueError(msg)

    base_sequence = SeedSequence(seed)
    return [np.random.default_rng(child) for child in base_sequence.spawn(n)]


def build_config_id(model: str, params: Mapping[str, Any]) -> tuple[str, str, str]:
    """
    Construct the ``(slug, hash8, config_id)`` tuple for a configuration.

    ``config_id`` follows the invariant ``<slug>__<hash8>``.
    """

    slug = slugify_config(model, dict(params))
    payload = {"model": model, "params": dict(params)}
    digest = hash8_from_obj(payload)
    return slug, digest, f"{slug}__{digest}"


@dataclass(slots=True)
class _GenerationPlan:
    """Canonical representation of a single generation request."""

    model: str
    params_model: BaseModel
    params_dict: Dict[str, Any]
    n_realizations: int
    zscore: bool
    zscore_axis: str | None
    save: bool
    csv: bool
    storage_dir: Path
    slug: str
    hash8: str
    config_id: str
    metadata: Dict[str, Any]
    seed_override: int | None


@dataclass(slots=True)
class _ExecutionContext:
    """Runtime data required during generation."""

    plan: _GenerationPlan
    rng: NPGenerator
    seed_sequence: SeedSequence


class Generator:
    """
    High-level generation façade for synthetic multivariate time series.

    Parameters
    ----------
    model_registry:
        Optional mapping from model names to :class:`BaseGenerator` subclasses.
        Models can also be registered post-instantiation via :meth:`register_model`.
    """

    def __init__(self, model_registry: Mapping[str, BaseGenerator] | None = None) -> None:
        registry = model_registry or REGISTRY
        self._models: Dict[str, BaseGenerator] = {name.lower(): model for name, model in registry.items()}

    def register_model(self, name: str, model: BaseGenerator) -> None:
        """Register or replace a model implementation."""

        if not name or not isinstance(name, str):
            msg = "Model name must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(model, BaseGenerator):
            msg = "model must be an instance of BaseGenerator."
            raise TypeError(msg)
        self._models[name.lower()] = model

    def available_models(self) -> Sequence[str]:
        """Return the sorted list of registered model identifiers."""

        return sorted(self._models)

    def generate(
        self,
        configs: list[MutableMapping[str, Any]],
        *,
        save: bool | None = None,
        outdir: str | None = None,
        zscore: bool = False,
        zscore_axis: str | None = "time",
        n_realizations: int | None = None,
        seed: int | None = None,
        csv: bool = False,
    ) -> list[xr.Dataset]:
        """
        Generate synthetic multivariate time series datasets for the given configs.

        Notes
        -----
        Model kernels are introduced in later steps. Invoking a registered model
        currently raises ``NotImplementedError`` to highlight the upcoming work.
        """

        self._validate_args(configs, save, outdir, zscore_axis, n_realizations, seed, csv)
        if not configs:
            return []

        plans = [self._normalise_config(cfg, save, csv, outdir, zscore, zscore_axis, n_realizations) for cfg in configs]

        rngs = spawn_subrngs(seed, len(plans))
        contexts = self._assemble_contexts(plans, rngs)

        datasets: list[xr.Dataset] = []
        for ctx in contexts:
            ds = self._execute_plan(ctx)
            datasets.append(ds)

        return datasets

    @staticmethod
    def _validate_args(
        configs: Iterable[Any],
        save: bool | None,
        outdir: str | None,
        zscore_axis: str | None,
        n_realizations: int | None,
        seed: int | None,
        csv: bool,
    ) -> None:
        """Validate top-level arguments prior to config preparation."""

        if not isinstance(configs, Iterable):
            msg = "configs must be an iterable of dictionaries."
            raise TypeError(msg)
        if save is not None and not isinstance(save, bool):
            msg = "save must be a boolean or None."
            raise TypeError(msg)
        if not isinstance(csv, bool):
            msg = "csv must be a boolean."
            raise TypeError(msg)
        if outdir is not None and not isinstance(outdir, str):
            msg = "outdir must be a string path when provided."
            raise TypeError(msg)
        if zscore_axis not in {None, "time", "channel", "realization"}:
            msg = "zscore_axis must be one of {'time', 'channel', 'realization', None}."
            raise ValueError(msg)
        if n_realizations is not None and (not isinstance(n_realizations, int) or n_realizations <= 0):
            msg = "n_realizations must be a positive integer when provided."
            raise ValueError(msg)
        if seed is not None and not isinstance(seed, int):
            msg = "seed must be an integer when provided."
            raise TypeError(msg)
        if outdir is not None:
            Path(outdir)

    def _normalise_config(
        self,
        raw_cfg: MutableMapping[str, Any],
        save: bool | None,
        csv: bool,
        outdir: str | None,
        zscore: bool,
        zscore_axis: str | None,
        n_realizations: int | None,
    ) -> _GenerationPlan:
        """Normalise a raw configuration dictionary."""

        if not isinstance(raw_cfg, MutableMapping):
            msg = "Each configuration must be a mutable mapping."
            raise TypeError(msg)

        cfg = dict(raw_cfg)
        if "model" not in cfg:
            msg = "Each configuration dictionary must include a 'model' key."
            raise KeyError(msg)

        model_name = str(cfg.pop("model")).lower()
        cfg_seed = cfg.pop("seed", None)
        cfg_save = cfg.pop("save", None)
        cfg_csv = cfg.pop("csv", None)
        cfg_outdir = cfg.pop("outdir", None)
        cfg_zscore = cfg.pop("zscore", None)
        cfg_zaxis = cfg.pop("zscore_axis", None)
        cfg_realizations = cfg.pop("n_realizations", None)

        params_raw = dict(cfg)
        if cfg_realizations is not None:
            params_raw["n_realizations"] = cfg_realizations

        model = self._models.get(model_name)
        if model is None:
            msg = f"No model registered under name '{model_name}'."
            raise KeyError(msg)

        schema_model = model.param_model()
        params_model = schema_model.model_validate(params_raw)

        resolved_realizations = self._resolve_realizations(params_model.n_realizations, n_realizations)
        if resolved_realizations != params_model.n_realizations:
            params_model = params_model.model_copy(update={"n_realizations": resolved_realizations})

        params_dict = params_model.model_dump(mode="python")

        resolved_zscore = cfg_zscore if cfg_zscore is not None else zscore
        axis_preference = cfg_zaxis if cfg_zaxis is not None else zscore_axis
        resolved_axis = axis_preference or "time"
        resolved_save = cfg_save if cfg_save is not None else (save if save is not None else False)
        resolved_csv = cfg_csv if cfg_csv is not None else csv

        slug, hash8, config_id = build_config_id(model_name, params_dict)

        if outdir is not None:
            target_dir = Path(outdir) / model_name / config_id
        elif cfg_outdir is not None:
            target_dir = Path(cfg_outdir)
        else:
            target_dir = Path("data") / model_name / config_id
        params_serialisable = json.loads(canonical_json(params_dict))

        metadata = {
            "model": model_name,
            "params": params_serialisable,
            "slug": slug,
            "hash8": hash8,
            "config_id": config_id,
            "n_realizations": resolved_realizations,
            "zscore": bool(resolved_zscore),
            "zscore_axis": resolved_axis,
        }

        return _GenerationPlan(
            model=model_name,
            params_model=params_model,
            params_dict=params_dict,
            n_realizations=resolved_realizations,
            zscore=bool(resolved_zscore),
            zscore_axis=resolved_axis,
            save=bool(resolved_save),
            csv=bool(resolved_csv),
            storage_dir=target_dir,
            slug=slug,
            hash8=hash8,
            config_id=config_id,
            metadata=metadata,
            seed_override=int(cfg_seed) if cfg_seed is not None else None,
        )

    @staticmethod
    def _resolve_realizations(config_value: Any, override: int | None) -> int:
        """Determine the number of realizations to produce for a config."""

        if override is not None:
            return override
        if config_value is None:
            return 1
        if not isinstance(config_value, int) or config_value <= 0:
            msg = "n_realizations values must be positive integers."
            raise ValueError(msg)
        return config_value

    def _assemble_contexts(self, plans: list[_GenerationPlan], rngs: list[NPGenerator]) -> list[_ExecutionContext]:
        """Combine generation plans with RNG state."""

        contexts: list[_ExecutionContext] = []
        for plan, base_rng in zip(plans, rngs, strict=True):
            if plan.seed_override is not None:
                seed_sequence = SeedSequence(plan.seed_override)
                rng = np.random.default_rng(seed_sequence)
            else:
                bit_gen = base_rng.bit_generator
                seed_sequence = bit_gen.seed_seq if hasattr(bit_gen, "seed_seq") else SeedSequence()
                rng = base_rng
            contexts.append(_ExecutionContext(plan=plan, rng=rng, seed_sequence=seed_sequence))
        return contexts

    def _execute_plan(self, ctx: _ExecutionContext) -> xr.Dataset:
        """Execute a single generation plan."""

        plan = ctx.plan
        model = self._models.get(plan.model)
        if model is None:
            msg = f"No model registered under name '{plan.model}'."
            raise KeyError(msg)

        try:
            dataset = model.generate_one(
                plan.params_model,
                ctx.rng,
                n_realizations=plan.n_realizations,
                zscore=plan.zscore,
                zscore_axis=plan.zscore_axis,
            )
        except NotImplementedError as exc:  # pragma: no cover - triggered when models remain stubs.
            msg = (
                f"Model '{plan.model}' generation is not implemented yet. "
                "Step 3 will provide numerical kernels."
            )
            raise NotImplementedError(msg) from exc

        dataset = self._post_process_dataset(dataset, ctx)
        if plan.save:
            self._persist_dataset(dataset, ctx)
        return dataset

    def _post_process_dataset(self, dataset: xr.Dataset, ctx: _ExecutionContext) -> xr.Dataset:
        """Ensure dataset conforms to invariants and apply optional transforms."""

        if not isinstance(dataset, xr.Dataset):
            msg = "Model generate_one must return an xarray.Dataset."
            raise TypeError(msg)

        expected_dims = ("time", "channel", "realization")
        missing = [dim for dim in expected_dims if dim not in dataset.dims]
        if missing and missing != ["realization"]:
            msg = f"Dataset missing required dimensions: {missing}"
            raise ValueError(msg)
        if "realization" not in dataset.dims:
            dataset = dataset.expand_dims({"realization": ctx.plan.n_realizations})

        dataset = dataset.transpose(*expected_dims)

        realization_size = dataset.sizes.get("realization", ctx.plan.n_realizations)
        if realization_size != ctx.plan.n_realizations:
            msg = (
                "Dataset realization dimension does not match requested n_realizations "
                f"({realization_size} != {ctx.plan.n_realizations})."
            )
            raise ValueError(msg)

        if ctx.plan.zscore and "data" in dataset.data_vars:
            dataset = dataset.copy()
            data_var = dataset["data"]
            axis = ctx.plan.zscore_axis or "time"
            if axis not in data_var.dims:
                msg = f"zscore axis '{axis}' is not present in the data variable dimensions."
                raise ValueError(msg)
            mean = data_var.mean(dim=axis)
            std = data_var.std(dim=axis)
            safe_std = std.where(std > 0, 1.0)
            dataset["data"] = (data_var - mean) / safe_std

        attrs = dict(dataset.attrs)
        attrs.update(
            {
                "model": ctx.plan.model,
                "config_id": ctx.plan.config_id,
                "slug": ctx.plan.slug,
                "hash8": ctx.plan.hash8,
                "n_realizations": ctx.plan.n_realizations,
                "seed_entropy": int(ctx.seed_sequence.entropy),
                "params": ctx.plan.metadata.get("params"),
            }
        )
        dataset = dataset.assign_attrs(attrs)
        return dataset

    def _persist_dataset(self, dataset: xr.Dataset, ctx: _ExecutionContext) -> None:
        """Persist dataset outputs to disk."""

        plan = ctx.plan
        storage_dir = plan.storage_dir
        storage_dir.mkdir(parents=True, exist_ok=True)

        metadata = dict(plan.metadata)
        metadata.update(
            {
                "seed_entropy": int(ctx.seed_sequence.entropy),
                "storage_dir": str(storage_dir),
            }
        )

        parquet_path = storage_dir / f"{plan.config_id}.parquet"
        save_parquet(dataset, parquet_path, metadata=metadata)

        sidecar_path = storage_dir / f"{plan.config_id}.metadata.json"
        write_sidecar_metadata(sidecar_path, metadata)

        if plan.csv:
            csv_path = storage_dir / f"{plan.config_id}.csv"
            save_csv(dataset.pymts.to_dataframe(), csv_path)
