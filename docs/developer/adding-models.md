---
title: Adding new models
---

# Adding a new model to pymts

Follow these steps to introduce a new generator while preserving determinism and API guarantees.

## 1. Define a Pydantic schema

- Add a class to pymts/schemas.py inheriting from BaseParams.
- Provide defaults, validation (e.g., stability guards, shape checks), and optional arrays (
p.ndarray | None).
- Ensure all fields serialize cleanly via model_dump(mode="python").

Example:

`python
class MyProcessParams(BaseParams):
    model: Literal["my_process"] = "my_process"
    alpha: float = Field(default=0.5, ge=0)
    beta: float = Field(default=1.0)
    x0: np.ndarray | None = None
`

## 2. Implement the generator

Create pymts/models/my_process.py:

`python
from pymts.schemas import MyProcessParams
from .base import BaseGenerator

class MyProcessModel(BaseGenerator):
    @classmethod
    def help(cls) -> str:
        return "My process description."

    @classmethod
    def param_model(cls):
        return MyProcessParams

    def generate_one(self, params, rng, *, n_realizations, zscore, zscore_axis):
        params_obj = MyProcessParams.model_validate(params)
        # produce data (time, channel, realization)
        ...
        return xr.Dataset({...}, coords=..., attrs={...})
`

Key points:

- Use the provided ng (a 
umpy.random.Generator).
- Return dims (time, channel, realization) in that order.
- Attach attrs: model, config_id, slug, hash8, 
_realizations, seed_entropy, and JSON-friendly params.

## 3. Register the model

Update pymts/models/__init__.py:

`python
from .my_process import MyProcessModel

REGISTRY["my_process"] = MyProcessModel()
`

## 4. Write tests

Add 	ests/test_models_my_process.py covering:

- Shape/dim order
- Determinism with repeated seeds
- Edge-case validation (invalid parameters raise errors)
- Optional z-scoring and metadata

Run pytest -q before committing.

## 5. Update docs

- Describe the model in the reference or tutorials.
- Include configuration examples and guidance on parameter stability.

## 6. Submit a PR

- Format code with uff or lack if configured.
- Ensure docs build (mkdocs build) succeeds.
- Open a pull request, tagging maintainers and linking any relevant issues.
