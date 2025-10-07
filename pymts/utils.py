"""
Utility helpers for canonical JSON serialisation, hashing, and slug construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["canonical_json", "hash8_from_obj", "slugify_config"]

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def canonical_json(obj: Any) -> str:
    """
    Serialise an arbitrary Python object into deterministic JSON.

    Parameters
    ----------
    obj:
        Object to serialise. Handles mappings, dataclasses, numpy scalars/arrays,
        and objects exposing ``model_dump`` (e.g., Pydantic models).

    Returns
    -------
    str
        JSON string with sorted keys, compact separators, and canonicalised data
        representations suitable for hashing.
    """

    normalised = _normalize(obj)
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"))


def hash8_from_obj(obj: Any) -> str:
    """
    Compute the first eight hexadecimal characters of the SHA-256 hash of ``obj``.
    """

    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:8]


def slugify_config(model: str, params: Mapping[str, Any], keys: tuple[str, ...] = ("M", "T")) -> str:
    """
    Construct a human-readable configuration slug.

    Parameters
    ----------
    model:
        Name of the model (e.g., ``"kuramoto"``). Leading/trailing whitespace is stripped.
    params:
        Mapping of parameter names to values.
    keys:
        Ordered parameter names that should always appear (if present) before other
        parameters in the slug, typically ``("M", "T")``.

    Returns
    -------
    str
        Slug of the form ``model_M{M}_T{T}_...`` sanitised to ``[A-Za-z0-9._-]``.
    """

    if not isinstance(model, str) or not model.strip():
        msg = "model must be a non-empty string."
        raise ValueError(msg)
    if not isinstance(params, Mapping):
        msg = "params must be a mapping."
        raise TypeError(msg)

    base = _sanitize(model.strip().lower())
    parts: list[str] = [base]
    max_extra = 4
    extra_count = 0

    def append_component(key: str) -> None:
        if key not in params:
            return
        raw_value = params[key]
        if raw_value is None:
            return
        value = _format_slug_value(raw_value)
        if value == "":
            return
        part = f"{key}{value}"
        sanitized = _sanitize(part)
        if len(sanitized) > 48:
            sanitized = sanitized[:48].rstrip("-_.")
        parts.append(sanitized)

    for key in keys:
        append_component(key)

    for key in sorted(k for k in params if k not in keys):
        if extra_count >= max_extra:
            break
        before = len(parts)
        append_component(key)
        if len(parts) > before:
            extra_count += 1

    slug = "_".join(filter(None, parts))
    return slug or "config"


def _sanitize(value: str) -> str:
    """Restrict value to permitted slug characters."""

    return _SANITIZE_PATTERN.sub("-", value).strip("-_.")


def _format_slug_value(value: Any) -> str:
    """Format parameter values for inclusion in slugs."""

    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        if np.isnan(float_value):  # pragma: no cover - rare edge.
            return "nan"
        if np.isinf(float_value):
            return "inf" if float_value > 0 else "-inf"
        if float_value.is_integer():
            return f"{float_value:.1f}"
        return format(float_value, ".6g")
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        inner = "-".join(_format_slug_value(item) for item in value)
        return f"[{inner}]"
    return str(value)


def _normalize(obj: Any) -> Any:
    """Recursively normalise objects into JSON-serialisable forms."""

    if isinstance(obj, Mapping):
        return {str(key): _normalize(value) for key, value in sorted(obj.items(), key=lambda item: str(item[0]))}

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_normalize(item) for item in obj]

    if isinstance(obj, set):
        return sorted((_normalize(item) for item in obj), key=_sequence_key)

    if is_dataclass(obj):
        return _normalize(asdict(obj))

    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _normalize(obj.model_dump(mode="python"))

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    if isinstance(obj, Path):
        return str(obj)

    return obj


def _sequence_key(value: Any) -> Any:
    """Normalise sequence sorting keys."""

    if isinstance(value, Mapping):
        return tuple((str(k), _sequence_key(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_sequence_key(item) for item in value)
    return value
