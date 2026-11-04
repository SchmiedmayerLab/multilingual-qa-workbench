#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""Rerun approved questions from collaborator response workbooks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import Experiment
from .evaluation import (
    ModelConfig,
    _read_review_workbook_sheet,
    export_review_workbooks,
    generate_answer_requests,
    save_table,
)

QUESTION_COLUMNS = [
    "question_id",
    "language_code",
    "language_name",
    "question_text",
    "translated_question",
]


@dataclass
class RerunPlan:
    source_dir: Path
    requests: pd.DataFrame
    pending: pd.DataFrame
    review_schema: pd.DataFrame
    models: list[ModelConfig]
    source_hashes: dict[str, str]
    experiment: Experiment

    def manifest(self) -> dict:
        return {
            "source_dir": str(self.source_dir),
            "source_sha256": self.source_hashes,
            "models": [asdict(model) for model in self.models],
            "answer_system_prompt": self.experiment.answer_prompt,
            "reference_answers": self.experiment.document["rerun"].get(
                "reference_answers", "empty"
            ),
            "experiment": self.experiment.snapshot(),
            "request_sha256": hashlib.sha256(
                self.requests.to_json(orient="records").encode()
            ).hexdigest(),
            "answer_count": len(self.requests),
            "pending_count": len(self.pending),
            "pending_revisions": [
                {"question_id": question, "language_code": language, "reason": reason}
                for (question, language), reason in self.experiment.pending_revisions.items()
            ],
        }


def load_rerun_plan(experiment: Experiment) -> RerunPlan:
    """Treat supplied translated_question cells as approved; never retranslate them."""
    settings = experiment.document.get("rerun")
    if not settings:
        raise ValueError("Configure rerun settings before importing response workbooks")
    source_dir = experiment.local_path(settings["source_dir"])
    models = experiment.models
    candidate_families = {
        **{model.name: family for family, model in models.items()},
        **settings.get("candidate_models", {}),
    }
    frames = []
    source_hashes = {}
    schema = None
    for workbook in sorted(source_dir.glob("*.xlsx")):
        if workbook.name.startswith("~$"):
            continue
        source_hashes[workbook.name] = hashlib.sha256(workbook.read_bytes()).hexdigest()
        frame = _read_review_workbook_sheet(workbook, sheet_name="responses")
        required = {*QUESTION_COLUMNS, "candidate_name", "model_id", "repeat_index", "response_key"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{workbook.name}: missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{workbook.name}: no response rows")
        for column in required - {"repeat_index"}:
            if (
                not frame[column]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
            ):
                raise ValueError(f"{workbook.name}: blank or non-text {column}")
        repeats = pd.to_numeric(frame["repeat_index"], errors="coerce")
        if (repeats.isna() | repeats.lt(1) | repeats.mod(1).ne(0)).any():
            raise ValueError(f"{workbook.name}: repeat_index must be a positive integer")
        frame["repeat_index"] = repeats.astype(int)

        workbook_schema = _read_review_workbook_sheet(workbook, sheet_name="criteria")
        schema_columns = ["field_key", "field_label", "options"]
        if not set(schema_columns).issubset(workbook_schema.columns):
            raise ValueError(f"{workbook.name}: missing review rubric columns")
        workbook_schema = workbook_schema[schema_columns]
        if (
            workbook_schema.empty
            or workbook_schema["field_key"].duplicated().any()
            or workbook_schema.map(lambda value: not isinstance(value, str) or not value.strip())
            .any()
            .any()
        ):
            raise ValueError(f"{workbook.name}: invalid review rubric")
        if schema is not None and not schema.equals(workbook_schema):
            raise ValueError(f"{workbook.name}: review rubric differs from other workbooks")
        schema = workbook_schema

        selected = frame[
            QUESTION_COLUMNS
            + [column for column in ["category", "topic", "risk_level"] if column in frame]
        ].copy()
        selected["reference_answer"] = (
            frame.get("reference_answer", "")
            if settings.get("reference_answers", "empty") == "preserve"
            else ""
        )
        selected["repeat_index"] = frame["repeat_index"]
        selected["source_file"] = workbook.name
        selected["source_row"] = range(2, len(frame) + 2)
        selected["source_sha256"] = source_hashes[workbook.name]
        for column in ["response_key", "candidate_name", "model_id"]:
            selected[f"source_{column}"] = frame[column]
        target_models = []
        for row in frame.itertuples(index=False):
            family = candidate_families.get(row.candidate_name)
            if family not in models:
                raise ValueError(f"{workbook.name}: unmapped candidate_name {row.candidate_name!r}")
            source_family = row.model_id.split("/", 1)[0]
            source_family = settings.get("source_provider_aliases", {}).get(
                source_family, source_family
            )
            if source_family != models[family].model_id.split("/", 1)[0]:
                raise ValueError(f"{workbook.name}: candidate_name and model_id families disagree")
            target_models.append(models[family].model_id)
        selected["model_id"] = target_models
        frames.append(selected)

    if not frames:
        raise ValueError(f"No response workbooks found in {source_dir}")
    requests = pd.concat(frames, ignore_index=True)
    keys = ["question_id", "language_code", "model_id", "repeat_index"]
    if requests.duplicated(keys).any() or requests["source_response_key"].duplicated().any():
        raise ValueError("Duplicate response rows across source workbooks")
    for group, columns in [
        (["question_id", "language_code"], ["translated_question", "question_text"]),
        (["language_code"], ["language_name"]),
    ]:
        if requests.groupby(group)[columns].nunique().gt(1).any().any():
            raise ValueError(f"Conflicting source values for {', '.join(columns)}")

    requests["pending_reason"] = [
        experiment.pending_revisions.get((row.question_id, row.language_code), "")
        for row in requests.itertuples(index=False)
    ]
    pending_mask = requests["pending_reason"].ne("")
    pending = requests.loc[pending_mask].reset_index(drop=True)
    requests = requests.loc[~pending_mask].drop(columns="pending_reason").reset_index(drop=True)
    return RerunPlan(
        source_dir,
        requests,
        pending,
        experiment.reviews[settings["review_group"]],
        list(models.values()),
        source_hashes,
        experiment,
    )


def prepare_rerun(plan: RerunPlan, output_dir: str | Path) -> Path:
    """Save a local plan and bind its output directory to these sources and settings."""
    output_dir = Path(output_dir).resolve()
    if (
        output_dir == plan.source_dir
        or output_dir in plan.source_dir.parents
        or plan.source_dir in output_dir.parents
    ):
        raise ValueError("Use a separate output directory outside the source directory")
    manifest_path = output_dir / "plan.json"
    manifest = plan.manifest()
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Sources or settings changed; choose a new output directory")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty and has no matching rerun plan")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    save_table(plan.requests, output_dir / "answer_requests.parquet")
    save_table(plan.pending, output_dir / "pending_revisions.csv")
    save_table(plan.review_schema, output_dir / "review_schema.csv")
    return output_dir


async def run_rerun(plan: RerunPlan, output_dir: str | Path) -> pd.DataFrame:
    output_dir = prepare_rerun(plan, output_dir)
    answers = await generate_answer_requests(
        plan.requests,
        models=plan.models,
        system_prompt=plan.experiment.answer_prompt,
        checkpoint_path=output_dir / "answers.parquet",
    )
    if not answers.empty:
        save_table(answers, output_dir / "answers.parquet")
        export_review_workbooks(
            answers,
            review_schema=plan.review_schema,
            output_dir=output_dir / "review" / plan.experiment.document["rerun"]["review_group"],
            review_type=plan.experiment.document["rerun"]["review_group"],
        )
    return answers
