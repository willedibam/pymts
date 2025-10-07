---
title: Contributing
---

# Contributing guide

We welcome contributions of new models, bug fixes, documentation improvements, and integration recipes.

## Before you start

1. **Discuss** significant ideas in an issue before you begin coding.
2. Ensure you have Python 3.11; install dev dependencies via uv pip install -r requirements-docs.txt or your preferred manager.
3. Fork https://github.com/willedibam/pymts and clone your fork.

## Development workflow

`ash
git checkout -b feature/my-awesome-change
uv pip install -e .[dev]  # or pip install -e .[dev]
pytest -q
`

We expect:

- Deterministic tests (no flaky RNG usage).
- New models to include schema validation, stability checks, and docs.
- Updates to changelog (GitHub Releases) when user-facing behaviour changes.

## Documentation

`ash
uv pip install -r requirements-docs.txt
mkdocs serve
`

Notebooks in docs/tutorials/ are executed during mkdocs build. Keep runtimes short (<30s).

## Commit style

- Use present-tense, descriptive messages (docs: add tsfresh guide).
- Group logically related changes into single commits; avoid large monolithic diffs.

## Pull requests

1. Rebase onto latest main.
2. Run pytest -q locally.
3. Optional: mkdocs build to verify docs.
4. Push (git push origin feature/my-awesome-change).
5. Open a PR with:
   - Summary of changes
   - Testing evidence
   - Screenshot for docs updates (optional but helpful)

CI runs 	ests.yml (pytest) and docs.yml (MkDocs + mike). Address any failures promptly.

## Code of Conduct

Interactions are governed by the project-wide [Code of Conduct](../../CODE_OF_CONDUCT.md). We strive for respectful, inclusive collaboration.
