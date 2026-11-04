<!--

This source file is part of the Multilingual QA Workbench open-source project

SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Development checks

After [setting up the environment](../README.md#getting-started), run these commands from the repository root:

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
ruff check src tests scripts
ruff format --check src tests scripts
python -m compileall -q src tests scripts
python scripts/check_notebooks.py
coverage run -m unittest discover -s tests
coverage report
reuse lint
```

The [Build and Test workflow](../.github/workflows/build-and-test.yml) runs linting, compilation, notebook validation, and offline regression tests on Python 3.14 and 3.15. It also uploads coverage to Codecov and calls the lab's shared repository standards workflow. A local test pass does not verify those external services.

[Test fixtures](helpers.py) use synthetic configurations, and tests mock API calls. They cover configuration loading, translation approvals, returned-workbook validation, checkpoints, exports, and rating summaries. They do not establish live model availability or current provider retention behavior.

The [notebook check](../scripts/check_notebooks.py) rejects saved outputs and execution counts. Clear the [workbench notebook](../notebooks/workbench.ipynb) before committing. Keep experiments, credentials, generated data, and legacy study files out of Git; contributions must use synthetic data under the [lab contribution guidelines](https://github.com/SchmiedmayerLab/.github/blob/main/CONTRIBUTING.md).

Dependency and packaging settings are in [pyproject.toml](../pyproject.toml), with check tools in [requirements-dev.txt](../requirements-dev.txt). [Dependabot](../.github/dependabot.yml) groups Python and GitHub Actions updates weekly. Follow the [repository standards](https://github.com/SchmiedmayerLab/.github/blob/main/REPOSITORY_STANDARDS.md) for maintenance conventions.
