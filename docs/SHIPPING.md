---
title: Shipping checklist
---

# Shipping pymts releases

## 1. Repository setup

1. Fork/clone https://github.com/willedibam/pymts.
2. Set upstream remote: git remote add upstream https://github.com/willedibam/pymts.git.
3. Enable branch protections on main (require PR reviews, passing CI).

## 2. Build and test locally

`ash
uv pip install -e .
uv pip install -r requirements-docs.txt
pytest -q
mkdocs build
`

## 3. Deploy docs with mike

`ash
mike deploy --push --update-aliases v0.1.0 latest
mike set-default --push latest
`

The first command publishes version 0.1.0 and aliases it as latest; set-default wires the site root to the latest docs. Repeat for new releases (e.g., 0.2.0).

## 4. GitHub Pages

- Settings ? Pages ? Source = gh-pages branch.
- Wait for GitHub Pages build; docs available at https://<org>.github.io/pymts/.

## 5. Publishing workflow

1. Update tutorials/recipes and ensure notebooks remain lightweight.
2. Run pytest -q and mkdocs build locally.
3. git tag vX.Y.Z and git push origin vX.Y.Z.
4. Create a GitHub Release summarising changes.
5. Use mike to publish docs for the tag and update stable if appropriate: mike deploy --push --update-aliases vX.Y.Z latest stable.

## 6. CI/CD

- .github/workflows/tests.yml runs pytest on pushes and PRs.
- .github/workflows/docs.yml builds + deploys docs when changes land on main or when triggered manually with workflow_dispatch.

## 7. Future updates

- Add new models via feature branches and pull requests.
- Keep dependencies pinned in equirements-docs.txt to ensure reproducible documentation builds.
- Rotate maintainers for release reviews to distribute knowledge.
