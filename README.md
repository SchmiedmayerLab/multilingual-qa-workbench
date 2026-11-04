<!--

This source file is part of the Multilingual QA Workbench open-source project

SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Multilingual QA Workbench

[![Build and Test](https://github.com/SchmiedmayerLab/multilingual-qa-workbench/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/SchmiedmayerLab/multilingual-qa-workbench/actions/workflows/build-and-test.yml)
[![Codecov](https://codecov.io/gh/SchmiedmayerLab/multilingual-qa-workbench/branch/main/graph/badge.svg)](https://codecov.io/gh/SchmiedmayerLab/multilingual-qa-workbench)
[![REUSE status](https://api.reuse.software/badge/github.com/SchmiedmayerLab/multilingual-qa-workbench)](https://api.reuse.software/info/github.com/SchmiedmayerLab/multilingual-qa-workbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

Evaluate model answers across languages using human-approved translations and configurable review criteria. Multilingual QA Workbench sends requests through OpenRouter, exports Excel workbooks for reviewers, and summarizes their ratings. Questions, languages, models, and rubrics live in an experiment configuration.

## Getting started

Use Python 3.14 or 3.15, the versions covered by [CI](.github/workflows/build-and-test.yml). Run these commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
[ -f .env ] || cp .env.example .env
mkdir -p experiments/demo
[ -f experiments/demo/experiment.json ] || cp examples/experiment.json experiments/demo/experiment.json
qa-workbench validate --config experiments/demo/experiment.json
```

For a fresh copy of the [sample configuration](examples/experiment.json), validation prints a JSON summary with one placeholder question, two languages, two answer models, and four possible answer requests. It makes no API calls and creates no run outputs. Validation checks configuration structure, not model availability.

Edit `experiments/demo/experiment.json` to supply your questions, languages, models, and review criteria. Before generating translations or answers, set `OPENROUTER_API_KEY` in `.env` and fund the OpenRouter account. The [environment template](.env.example) also sets the notebook's experiment and workflow. Generation requires `--execute` in the CLI or `EXECUTE = True` in the notebook.

## Workflows

Commands below use the demo configuration and run `initial`, except the rerun command, which uses `refresh`. Follow the [experiment guide](examples/README.md) for reviewer handoffs between stages.

| Workflow | Result and instructions | Entry command |
| --- | --- | --- |
| Configure an experiment | [Set questions, languages, models, rubrics, and pending revisions](examples/README.md#configuration). | `qa-workbench validate --config experiments/demo/experiment.json` |
| Estimate costs | [Estimate the authored matrix using current prices](examples/README.md#estimate-costs); estimates are not spending limits. | `qa-workbench budget --config experiments/demo/experiment.json` |
| Translate and answer | [Review translations before generating answers](examples/README.md#translate-answer-and-review), then collect ratings in Excel. | `qa-workbench translate --config experiments/demo/experiment.json --execute` |
| Refresh existing answers | [Prepare a rerun from returned workbooks](examples/README.md#rerun-approved-response-workbooks), preserving their approved translations and supplied row combinations. | `qa-workbench rerun --config experiments/demo/experiment.json --run-name refresh` |
| Summarize ratings | [Count review choices and their shares](examples/README.md#summarize-ratings) by model, language, and criterion. | `qa-workbench summarize --config experiments/demo/experiment.json` |
| Use the notebook | [Select an experiment and workflow](examples/README.md#notebook-workbench) to inspect plans, results, and configured plots interactively. | `jupyter lab notebooks/workbench.ipynb` |

Local experiment files and outputs belong under `experiments/`, which Git ignores. Keep a separate backup. [Run storage and resuming](examples/README.md#files-and-resuming) explains checkpoints and when to start a new run; [provider routing](examples/README.md#provider-routing) explains retention settings and explicit exceptions.

For implementation changes, see the [development checks](tests/README.md), [source package](src/qa_workbench/), and [build configuration](pyproject.toml). Tests use synthetic inputs and mocked API calls.

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/SchmiedmayerLab/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/SchmiedmayerLab/.github/blob/main/CODE_OF_CONDUCT.md) first. You can find a list of contributors in the [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for more information.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff), which GitHub surfaces through the [*Cite this repository*](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) button.

## Our Research

For more information, visit the [Schmiedmayer Lab GitHub organization](https://github.com/SchmiedmayerLab).

![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-light.png#gh-light-mode-only)
![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-dark.png#gh-dark-mode-only)
