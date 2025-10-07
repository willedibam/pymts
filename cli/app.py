"""
Command-line interface for the PyMTS package.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import typer
import yaml

from pymts import __version__ as VERSION
from pymts.core import Generator, build_config_id

app = typer.Typer(help="Synthetic multivariate time series generation toolkit.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show the PyMTS version and exit."),
) -> None:
    """Display help or version information."""

    if version:
        typer.echo(f"pymts {VERSION}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def generate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML/JSON configuration file."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Base output directory (default: data/<model>/<config>)."),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist generated datasets to disk.", show_default=True),
    csv: bool = typer.Option(False, "--csv/--no-csv", help="Write CSV alongside Parquet when saving.", show_default=True),
    zscore: bool = typer.Option(False, "--zscore/--no-zscore", help="Apply channel-wise z-score per realization.", show_default=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print expanded configurations without generating."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Limit the number of expanded configurations."),
) -> None:
    """
    Generate synthetic multivariate time series from configuration grids.
    """

    configs = _load_configs(config)
    expanded = list(_expand_configs(configs))
    if limit is not None:
        expanded = expanded[:limit]

    if not expanded:
        typer.echo("No configurations to process.")
        raise typer.Exit()

    base_outdir = Path(outdir) if outdir is not None else None

    typer.echo("Planned configurations:")
    summaries: list[tuple[dict[str, Any], str, str, Path]] = []
    for idx, cfg in enumerate(expanded, start=1):
        params = _extract_params(cfg)
        slug, hash8, config_id = build_config_id(str(cfg.get("model", "")).lower(), params)
        storage_dir = _determine_storage_dir(cfg, base_outdir, config_id)
        summaries.append((cfg, slug, config_id, storage_dir))
        typer.echo(f"[{idx}] model={cfg.get('model')} config_id={config_id} -> {storage_dir}")

    if dry_run:
        typer.echo(f"Dry run complete ({len(summaries)} configuration(s)).")
        raise typer.Exit()

    generator = Generator()
    datasets = generator.generate(
        [dict(cfg) for cfg, _, _, _ in summaries],
        save=save,
        outdir=str(base_outdir) if base_outdir is not None else None,
        csv=csv,
        zscore=zscore,
    )

    typer.echo("Generation complete:")
    for (cfg, _slug, config_id, storage_dir), dataset in zip(summaries, datasets, strict=True):
        model = str(cfg.get("model"))
        message = f"[{model}] {config_id}"
        if save:
            message += f" saved -> {storage_dir}"
        else:
            message += " (not saved)"
        typer.echo(message)


def _load_configs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise typer.BadParameter(f"Configuration file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            data = json.loads(raw)

    if data is None:
        return []

    if isinstance(data, dict) and "configs" in data:
        records = data["configs"]
    else:
        records = data

    if isinstance(records, dict):
        return [dict(records)]

    if isinstance(records, list):
        return [dict(item) for item in records if isinstance(item, dict)]

    raise typer.BadParameter("Unsupported configuration structure.")


GRID_EXCLUDE_KEYS = {
    "model",
    "block_sizes",
    "phi",
    "Phi",
    "Sigma",
    "A",
    "theta0",
    "x0",
}

RUNTIME_KEYS = {
    "model",
    "save",
    "csv",
    "outdir",
    "seed",
    "zscore",
    "zscore_axis",
    "n_realizations",
    "dry_run",
    "limit",
}


def _expand_configs(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for record in records:
        yield from _expand_single(record)


def _expand_single(record: dict[str, Any]) -> list[dict[str, Any]]:
    base_items: dict[str, Any] = {}
    grid_keys: list[str] = []
    grid_values: list[list[Any]] = []

    for key, value in record.items():
        if isinstance(value, list) and key not in GRID_EXCLUDE_KEYS and _is_grid_candidate(value):
            grid_keys.append(key)
            grid_values.append(list(value))
        else:
            base_items[key] = value

    if not grid_keys:
        return [dict(base_items)]

    expanded: list[dict[str, Any]] = []
    for combo in product(*grid_values):
        cfg = dict(base_items)
        cfg.update({key: val for key, val in zip(grid_keys, combo, strict=True)})
        expanded.append(cfg)
    return expanded


def _is_grid_candidate(value: list[Any]) -> bool:
    if not value:
        return False
    return all(not isinstance(item, (dict, list)) for item in value)


def _extract_params(record: dict[str, Any]) -> dict[str, Any]:
    if "model" not in record:
        raise ValueError("configuration missing 'model'")
    params = {key: value for key, value in record.items() if key not in RUNTIME_KEYS}
    if "M" not in params or "T" not in params:
        raise ValueError("configuration requires 'M' and 'T'")
    return params


def _determine_storage_dir(cfg: dict[str, Any], base_outdir: Optional[Path], config_id: str) -> Path:
    cfg_outdir = cfg.get("outdir")
    model_name = str(cfg.get("model", "model")).lower()

    if base_outdir is not None:
        return Path(base_outdir) / model_name / config_id
    if cfg_outdir is not None:
        return Path(cfg_outdir)
    return Path("data") / model_name / config_id
