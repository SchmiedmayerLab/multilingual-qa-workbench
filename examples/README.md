<!--

This source file is part of the Multilingual QA Workbench open-source project

SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Experiment guide

## Configuration

Copy the [sample configuration](experiment.json) into a local experiment as shown in [Getting started](../README.md#getting-started). Paths are relative to the directory containing `experiment.json` and must stay inside that directory. Use stable question IDs and language codes so approvals and responses can be matched across workbooks.

| Setting | Contents |
| --- | --- |
| `schema_version`, `name` | Version `1` and the experiment's display name. |
| `questions` | Objects with `question_id` and `question_text`. Optional `source_language_code`, `source_language_name`, `reference_answer`, `category`, `topic`, and `risk_level` metadata are preserved. Source language defaults to English. |
| `languages` | Target `language_code` and `language_name` pairs. |
| `models` | Named model configurations: display `name`, OpenRouter `model_id`, and optional `temperature`, `max_tokens`, `timeout_seconds`, `max_retries`, `concurrency`, and `extra_body`. Use `null` to omit temperature. |
| `translation` | Translator `model` configuration and optional `system_prompt`. |
| `answers` | Optional `system_prompt` and `repeats_per_prompt` (default `1`). |
| `reviews` | Named reviewer groups, each containing criteria with `field_key`, `field_label`, and pipe-separated `options`. |
| `pending_revisions` | Optional exclusions with `question_id`, `language_code`, and `reason`. Applied to both workflows until removed. |
| `rerun` | `source_dir`, `review_group`, `candidate_models`, `source_provider_aliases`, and `reference_answers`; see [reruns](#rerun-approved-response-workbooks). |
| `budget` | Optional `activity_file`, `activity_model_aliases`, `safety_multiplier`, and `round_up_usd`. |
| `plots` | Optional summary charts selecting a `review_group`, `field_key`, `option`, and `title`. |

## Provider routing

Models default to `provider.zdr: true` and always send `data_collection: deny`. A model's optional `zero_retention_exception` must contain an explicit reason to disable its per-request ZDR requirement. The exception is saved in run manifests. OpenRouter's account and guardrail settings still apply, including whether [Safety Retention](https://openrouter.ai/docs/guides/features/zdr) is allowed. An exception cannot bypass account or guardrail enforcement, and `zdr: true` alone does not exclude endpoints using Safety Retention when the account allows them. Keep credentials in environment variables, never in configuration files.

## Estimate costs

```sh
qa-workbench budget --config experiments/demo/experiment.json --run-name initial
```

This fetches current OpenRouter prices without inference calls and prints estimates and recommended budgets. Estimates cover the authored question/language matrix, including pending pairs; approved inputs and returned-workbook reruns may have fewer rows. Available run samples and an optional activity export calibrate the estimates. These are estimates, not enforced spending limits.

## Translate, answer, and review

Use this workflow for new questions. Run commands from the repository root with the same configuration and run name for each step. Optional [cost estimates](#estimate-costs) can be checked before generation:

```sh
qa-workbench prepare --config experiments/demo/experiment.json --run-name initial
qa-workbench translate --config experiments/demo/experiment.json --run-name initial --execute
```

Generation commands with `--execute` incur OpenRouter charges. This run writes its files under `experiments/demo/runs/initial/`; paths below are relative to that directory.

Send the run's `review/translations/generated/` workbooks to reviewers. They fill `translation_final`, mark `translation_review_status` as `approved`, and return the files to `review/translations/completed/`. Approval must match the configured question text. Missing approvals are skipped; machine suggestions are never used as a fallback.

```sh
qa-workbench answers --config experiments/demo/experiment.json --run-name initial --execute
```

Distribute the workbooks in each `review/<group>/generated/` directory. Put rated copies in that group's `completed/` directory.

## Summarize ratings

After collecting ratings, run:

```sh
qa-workbench summarize --config experiments/demo/experiment.json --run-name initial
```

Summaries are printed and saved to `summaries/<group>.parquet` within the run. They count nonempty ratings and calculate each option's share for each model, language, and criterion. Avoid storing duplicate copies of the same assessment in `completed/`: all workbook rows are counted.

## Rerun approved response workbooks

Put returned answer workbooks in the configured `rerun.source_dir`. Each `.xlsx` file needs `responses` and `criteria` sheets in the exported format; source rubrics must agree across files. `translated_question` is treated as approved and is sent unchanged. Map each old `candidate_name` to a key in `models` through `rerun.candidate_models`; `source_provider_aliases` handles renamed provider prefixes. The configured `rerun.review_group` selects the output rubric.

```sh
qa-workbench rerun --config experiments/demo/experiment.json --run-name refresh
```

This prepares `plan.json`, `answer_requests.parquet`, `pending_revisions.csv`, and `review_schema.csv` under `experiments/demo/runs/refresh/` without API calls. It preserves exactly the supplied question/language/model/repeat combinations, excludes pending revisions, and rejects conflicting or duplicate rows. `rerun.reference_answers` defaults to `empty`; use `preserve` to keep supplied references. Previous answers and ratings are excluded from the new outputs. Reruns use the workbook rows rather than expanding the configured question/language matrix.

Inspect the plan, then execute it:

```sh
qa-workbench rerun --config experiments/demo/experiment.json --run-name refresh --execute
```

Source files remain unchanged. New workbooks have empty ratings. Save new assessments under the run's `review/<group>/completed/`, separate from its source inputs. [Summarize those ratings](#summarize-ratings) with `--run-name refresh`.

## Notebook workbench

The [workbench notebook](../notebooks/workbench.ipynb) uses the same workflows as the CLI. Select the experiment in `.env` using the keys from the [environment template](../.env.example):

```dotenv
HEALTHQA_CONFIG=experiments/demo/experiment.json
HEALTHQA_RUN=initial
HEALTHQA_WORKFLOW=authoring
```

These are also the notebook defaults. Use `HEALTHQA_WORKFLOW=rerun` and a separate run name such as `refresh` for returned answer workbooks. Start Jupyter from the repository root:

```sh
jupyter lab notebooks/workbench.ipynb
```

`EXECUTE = False` disables paid generation; set it to `True` when ready. `FETCH_PRICING = False` separately controls the optional pricing lookup. Rerun mode prepares a local plan even with generation disabled. Run the translation and answer stages separately when waiting for reviewer approvals. The final cells summarize ratings and draw the charts configured in `plots`.

## Files and resuming

```text
experiments/<experiment>/     Local only, ignored by Git
  experiment.json             Questions and experiment settings
  inputs/                     Returned workbooks and optional activity exports
  runs/<run>/                 Config snapshot or rerun plan, checkpoints, summaries
    review/<group>/           Generated and completed review workbooks
```

Use separate run names for authoring and reruns, and a new name after changing configuration or rerun sources. Authoring saves the configuration in `experiment.json`; reruns embed it in `plan.json` with source hashes. Checkpoints resume only when the model, request settings, prompt, and input text match. On a provider failure, queued calls stop and successful calls already in flight are saved. Long workbook text continues in adjacent `__part_2`, `__part_3`, etc. columns.

Existing studies can keep earlier files in an experiment's `legacy/` directory. Historical manifests remain provenance; prepare new plans against the current paths. Keep a separate backup of local experiments because Git does not store them. Removing data from tracked files does not remove it from earlier Git commits.
