#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

from dataclasses import asdict

from qa_workbench.evaluation import ModelConfig


def synthetic_models(concurrency=1):
    return {
        key: ModelConfig(
            name=key.title(),
            model_id=f"{key}/new",
            concurrency=concurrency,
            zero_retention_exception="Synthetic approved exception" if key == "delta" else None,
        )
        for key in ["alpha", "beta", "gamma", "delta"]
    }


def experiment_document():
    return {
        "schema_version": 1,
        "name": "Synthetic experiment",
        "questions": [{"question_id": "q1", "question_text": "Synthetic question?"}],
        "languages": [{"language_code": "fr", "language_name": "French"}],
        "models": {key: asdict(model) for key, model in synthetic_models().items()},
        "translation": {
            "model": asdict(ModelConfig(name="Translator", model_id="alpha/translator")),
            "system_prompt": "Synthetic translation instruction",
        },
        "answers": {"system_prompt": None, "repeats_per_prompt": 1},
        "reviews": {
            "reviewers": [{"field_key": "helpful", "field_label": "Helpful", "options": "Yes | No"}]
        },
        "pending_revisions": [
            {"question_id": "held", "language_code": code, "reason": "Awaiting revision"}
            for code in ["xx", "yy"]
        ],
        "rerun": {
            "source_dir": "source",
            "review_group": "reviewers",
            "candidate_models": {f"Old {key.title()}": key for key in synthetic_models()},
            "reference_answers": "empty",
        },
    }
