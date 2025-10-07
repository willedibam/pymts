# Contributing to pymts

Thank you for your interest in improving pymts! We welcome bug reports, documentation fixes, new models, tutorials, and integration guides.

## Ways to contribute

- Report bugs and feature requests via GitHub issues
- Polish the documentation or tutorials
- Extend the model catalogue with well-tested generators
- Improve integration guides for downstream libraries

## Getting started

1. Fork https://github.com/willedibam/pymts and clone your fork.
2. Create a virtual environment (`uv venv -p 3.11` or `python -m venv .venv`).
3. Install development dependencies (optional but recommended):
   ```bash
   uv pip install -r requirements-docs.txt
   ```
4. Install pymts in editable mode: `uv pip install -e .` (or use pip/poetry).

## Development workflow

```bash
git checkout -b feature/awesome-improvement
pytest -q
```

- Update or add tests for your change.
- Ensure `pytest -q` passes before pushing.
- If you touch docs, run `mkdocs build` to confirm notebooks render.

## Submitting changes

1. Commit with clear messages (`model: add stochastic volatility generator`).
2. Push to your fork (`git push origin feature/awesome-improvement`).
3. Open a pull request against `main`, describing the change and testing performed.

## Code of Conduct

Please review and follow the [Code of Conduct](CODE_OF_CONDUCT.md). We expect respectful, inclusive communication in issues and PRs.

## Questions?

Open a discussion or ping the maintainers in GitHub issues—we are happy to help!
