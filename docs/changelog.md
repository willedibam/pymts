---
title: Changelog policy
---

# Changelog policy

pymts tracks releases via GitHub Releases rather than a manual CHANGELOG.md. Each release includes:

- Summary of new models or algorithms
- Breaking changes (if any)
- Notable bug fixes
- Documentation highlights

When preparing a release:

1. Update tutorials/recipes if behaviour changed.
2. Tag the commit (git tag vX.Y.Z), push tags, and draft a GitHub Release.
3. Use mike deploy vX.Y.Z latest and mike set-default latest to publish docs.

For work-in-progress updates, communicate via pull requests and issues; keep release notes concise but informative.
