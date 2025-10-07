from __future__ import annotations

from pymts.utils import canonical_json, hash8_from_obj, slugify_config


def test_hash8_stability() -> None:
    obj_a = {"model": "kuramoto", "params": {"K": 0.5, "T": 100, "M": 5}}
    obj_b = {"params": {"T": 100, "M": 5, "K": 0.5}, "model": "kuramoto"}
    assert canonical_json(obj_a) == canonical_json(obj_b)
    assert hash8_from_obj(obj_a) == hash8_from_obj(obj_b)


def test_slugify_config_prefix() -> None:
    params = {"M": 5, "T": 500, "K": 1.0}
    slug = slugify_config("Kuramoto", params)
    assert slug.startswith("kuramoto_M5_T500_K1.0")
    slug_alt = slugify_config("Kuramoto", {"T": 500, "M": 5, "K": 1.0})
    assert slug == slug_alt
