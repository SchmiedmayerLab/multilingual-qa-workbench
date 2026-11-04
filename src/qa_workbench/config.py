#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""Load experiment settings without importing a notebook or accessing the network."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation import (
    ModelConfig,
    build_question_language_matrix,
    languages_to_dataframe,
    questions_to_dataframe,
    review_schema_to_dataframe,
)


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def identifier(value: Any, label: str) -> str:
    _text(value, label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            f"{label} must contain only letters, numbers, dots, hyphens, or underscores"
        )
    return value


def _keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if unknown := set(value) - allowed:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")


def _positive(value: Any, label: str, *, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive {'integer' if integer else 'number'}")


def model_config(value: dict) -> ModelConfig:
    _keys(value, {field.name for field in fields(ModelConfig)}, "model")
    for key in ["name", "model_id"]:
        _text(value.get(key), f"model.{key}")
    if "/" not in value["model_id"]:
        raise ValueError("model_id must include the OpenRouter provider prefix")
    model = ModelConfig(**value)
    for key in ["max_retries", "concurrency", "timeout_seconds"]:
        _positive(getattr(model, key), key, integer=True)
    if model.max_tokens is not None:
        _positive(model.max_tokens, "max_tokens", integer=True)
    if model.temperature is not None and (
        isinstance(model.temperature, bool)
        or not isinstance(model.temperature, (int, float))
        or not math.isfinite(model.temperature)
        or model.temperature < 0
    ):
        raise ValueError("temperature must be null or a nonnegative number")
    if not isinstance(model.extra_body, dict):
        raise ValueError("extra_body must be an object")
    if {"model", "messages"} & model.extra_body.keys():
        raise ValueError("extra_body cannot override model or messages")
    if "provider" in model.extra_body and not isinstance(model.extra_body["provider"], dict):
        raise ValueError("extra_body.provider must be an object")
    if model.zero_retention_exception is not None:
        _text(model.zero_retention_exception, "zero_retention_exception")
    return model


def _rows(value: Any, required: list[str], key: str, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    seen = set()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"Each {label} entry must be an object")
        for column in required:
            _text(row.get(column), f"{label}.{column}")
        name = identifier(row[key], f"{label}.{key}")
        if name in seen:
            raise ValueError(f"Duplicate {label}.{key}: {name}")
        seen.add(name)


@dataclass(frozen=True)
class Experiment:
    path: Path
    document: dict
    models: dict[str, ModelConfig]
    translation_model: ModelConfig

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def questions(self) -> pd.DataFrame:
        return questions_to_dataframe(self.document["questions"])

    @property
    def languages(self) -> pd.DataFrame:
        return languages_to_dataframe(self.document["languages"])

    @property
    def reviews(self) -> dict[str, pd.DataFrame]:
        return {
            name: review_schema_to_dataframe(rows)
            for name, rows in self.document["reviews"].items()
        }

    @property
    def pending_revisions(self) -> dict[tuple[str, str], str]:
        return {
            (row["question_id"], row["language_code"]): row["reason"]
            for row in self.document.get("pending_revisions", [])
        }

    def filter_pending(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        return frame.loc[
            [
                (row.question_id, row.language_code) not in self.pending_revisions
                for row in frame.itertuples()
            ]
        ].copy()

    @property
    def matrix(self) -> pd.DataFrame:
        return self.filter_pending(build_question_language_matrix(self.questions, self.languages))

    @property
    def translation_prompt(self) -> str | None:
        return self.document["translation"].get("system_prompt")

    @property
    def answer_prompt(self) -> str | None:
        return self.document.get("answers", {}).get("system_prompt")

    @property
    def repeats(self) -> int:
        return self.document.get("answers", {}).get("repeats_per_prompt", 1)

    def local_path(self, value: str) -> Path:
        _text(value, "path")
        path = Path(value)
        resolved = (self.root / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(self.root):
            raise ValueError("Experiment paths must stay inside the experiment directory")
        return resolved

    def run_path(self, name: str) -> Path:
        return self.local_path(f"runs/{identifier(name, 'run name')}")

    def snapshot(self) -> dict:
        return {
            **self.document,
            "models": {name: asdict(model) for name, model in self.models.items()},
            "translation": {
                **self.document["translation"],
                "model": asdict(self.translation_model),
            },
        }


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    _keys(
        document,
        {
            "schema_version",
            "name",
            "questions",
            "languages",
            "models",
            "translation",
            "answers",
            "reviews",
            "pending_revisions",
            "rerun",
            "budget",
            "plots",
        },
        "experiment",
    )
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported schema_version; expected 1")
    _text(document.get("name"), "name")
    _rows(document.get("questions"), ["question_id", "question_text"], "question_id", "questions")
    _rows(
        document.get("languages"), ["language_code", "language_name"], "language_code", "languages"
    )
    models_data = document.get("models")
    if not isinstance(models_data, dict) or not models_data:
        raise ValueError("models must be a nonempty object")
    models = {
        identifier(key, "model key"): model_config(value) for key, value in models_data.items()
    }
    if len({m.model_id for m in models.values()}) != len(models):
        raise ValueError("Each answer model must have a distinct model_id")
    translation = document.get("translation")
    _keys(translation, {"model", "system_prompt"}, "translation")
    translator = model_config(translation.get("model"))
    answers = document.get("answers", {})
    _keys(answers, {"system_prompt", "repeats_per_prompt"}, "answers")
    _positive(answers.get("repeats_per_prompt", 1), "repeats_per_prompt", integer=True)
    for section in [translation, answers]:
        if section.get("system_prompt") is not None:
            _text(section["system_prompt"], "system_prompt")
    reviews = document.get("reviews")
    if not isinstance(reviews, dict) or not reviews:
        raise ValueError("reviews must contain at least one review group")
    for name, rows in reviews.items():
        identifier(name, "review group")
        if name == "translations":
            raise ValueError(
                "The review group name 'translations' is reserved for translation review"
            )
        _rows(rows, ["field_key", "field_label", "options"], "field_key", "review criteria")
        for row in rows:
            if any(not option.strip() for option in row["options"].split("|")):
                raise ValueError("Review options cannot contain empty choices")
    pending = document.get("pending_revisions", [])
    if not isinstance(pending, list):
        raise ValueError("pending_revisions must be a list")
    pairs = set()
    for row in pending:
        _keys(row, {"question_id", "language_code", "reason"}, "pending revision")
        for column in ["question_id", "language_code", "reason"]:
            _text(row.get(column), column)
        pair = (row["question_id"], row["language_code"])
        if pair in pairs:
            raise ValueError("Duplicate pending revision")
        pairs.add(pair)
    rerun = document.get("rerun", {})
    _keys(
        rerun,
        {
            "source_dir",
            "review_group",
            "candidate_models",
            "source_provider_aliases",
            "reference_answers",
        },
        "rerun",
    )
    if rerun:
        if rerun.get("review_group") not in reviews:
            raise ValueError("rerun.review_group must name a configured review group")
        mapping = rerun.get("candidate_models", {})
        if not isinstance(mapping, dict) or any(value not in models for value in mapping.values()):
            raise ValueError(
                "rerun.candidate_models must map candidate names to configured model keys"
            )
        aliases = rerun.get("source_provider_aliases", {})
        if not isinstance(aliases, dict) or any(not isinstance(v, str) for v in aliases.values()):
            raise ValueError("source_provider_aliases must map provider prefixes to strings")
        if rerun.get("reference_answers", "empty") not in {"empty", "preserve"}:
            raise ValueError("rerun.reference_answers must be empty or preserve")
    budget = document.get("budget", {})
    _keys(
        budget,
        {"activity_file", "activity_model_aliases", "safety_multiplier", "round_up_usd"},
        "budget",
    )
    for key in ["safety_multiplier", "round_up_usd"]:
        _positive(budget.get(key, 1), f"budget.{key}")
    if not isinstance(budget.get("activity_model_aliases", {}), dict):
        raise ValueError("budget.activity_model_aliases must be an object")
    plots = document.get("plots", [])
    if not isinstance(plots, list):
        raise ValueError("plots must be a list")
    for plot in plots:
        _keys(plot, {"review_group", "field_key", "option", "title"}, "plot")
        for key in ["review_group", "field_key", "option", "title"]:
            _text(plot.get(key), f"plot.{key}")
        criteria = {row["field_key"]: row for row in reviews.get(plot["review_group"], [])}
        criterion = criteria.get(plot["field_key"])
        if criterion is None or plot["option"] not in [
            v.strip() for v in criterion["options"].split("|")
        ]:
            raise ValueError("Plot must select a configured review group, criterion, and option")
    experiment = Experiment(path, document, models, translator)
    if rerun:
        experiment.local_path(rerun.get("source_dir"))
    if budget.get("activity_file") is not None:
        experiment.local_path(budget["activity_file"])
    return experiment
