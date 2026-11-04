#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""Experiment workflows shared by the command line and notebook."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import evaluation as ev
from .config import Experiment
from .rerun import load_rerun_plan, prepare_rerun, run_rerun


class Workbench:
    def __init__(self, experiment: Experiment, run: str):
        self.experiment = experiment
        self.directory = experiment.run_path(run)

    def prepare(self) -> Path:
        """Bind a run to a configuration before writing generated files."""
        path = self.directory / "experiment.json"
        snapshot = self.experiment.snapshot()
        if path.exists():
            if json.loads(path.read_text()) != snapshot:
                raise ValueError("Experiment settings changed; choose a new run name")
        elif self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError("Run directory is not empty and has no matching experiment snapshot")
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        return self.directory

    def _table(self, name: str) -> pd.DataFrame:
        path = self.directory / name
        return ev.load_table(path) if path.exists() else pd.DataFrame()

    def summary(self) -> dict:
        experiment = self.experiment
        return {
            "name": experiment.document["name"],
            "questions": len(experiment.questions),
            "languages": len(experiment.languages),
            "eligible_question_language_pairs": len(experiment.matrix),
            "answer_requests_if_all_approved": len(experiment.matrix)
            * len(experiment.models)
            * experiment.repeats,
            "pending_revisions": len(experiment.pending_revisions),
            "models": {name: asdict(model) for name, model in experiment.models.items()},
            "run_directory": str(self.directory),
        }

    async def translate(self) -> pd.DataFrame:
        self.prepare()
        experiment = self.experiment
        if experiment.matrix.empty:
            return experiment.matrix
        translations = await ev.propose_translations(
            experiment.matrix,
            translation_model=experiment.translation_model,
            translation_system_prompt=experiment.translation_prompt,
            checkpoint_path=self.directory / "translation_suggestions.parquet",
        )
        ev.save_table(translations, self.directory / "translation_suggestions.parquet")
        ev.export_translation_review_workbooks(translations, self.directory / "review/translations")
        return translations

    def approvals(self) -> pd.DataFrame:
        return self.experiment.filter_pending(
            ev.load_reviewed_translations(
                self.directory / "review/translations/completed",
                current_matrix=self.experiment.matrix,
            )
        )

    async def answer(self) -> pd.DataFrame:
        self.prepare()
        approved = self.approvals()
        if approved.empty:
            return pd.DataFrame()
        ev.save_table(approved, self.directory / "approved_translations.parquet")
        answers = await ev.generate_answers(
            approved,
            models=self.experiment.models.values(),
            system_prompt=self.experiment.answer_prompt,
            repeats_per_prompt=self.experiment.repeats,
            checkpoint_path=self.directory / "answers.parquet",
        )
        ev.save_table(answers, self.directory / "answers.parquet")
        for group, schema in self.experiment.reviews.items():
            ev.export_review_workbooks(
                answers,
                review_schema=schema,
                output_dir=self.directory / "review" / group,
                review_type=group,
            )
        return answers

    def prepare_rerun(self):
        plan = load_rerun_plan(self.experiment)
        prepare_rerun(plan, self.directory)
        return plan

    async def rerun(self) -> pd.DataFrame:
        return await run_rerun(load_rerun_plan(self.experiment), self.directory)

    def summarize(self) -> dict[str, pd.DataFrame]:
        summaries = {}
        for group, schema in self.experiment.reviews.items():
            reviews = ev.load_completed_reviews(self.directory / "review" / group / "completed")
            _, summary = ev.summarize_review_answers(reviews, review_schema=schema)
            summaries[group] = summary
            # Write empty summaries too, so removing reviews cannot leave stale results.
            if self.directory.exists():
                ev.save_table(summary, self.directory / "summaries" / f"{group}.parquet")
        return summaries

    def budget(self) -> ev.BudgetReport:
        """Fetch current pricing; no inference requests. Estimates the authored experiment."""
        exp = self.experiment
        translations = self._table("translation_suggestions.parquet")
        answers = self._table("answers.parquet")
        if "translation_request_signature" in translations:
            translations = translations.loc[
                translations.translation_request_signature.eq(
                    exp.translation_model.request_signature(exp.translation_prompt)
                )
            ]
        else:
            translations = pd.DataFrame()
        if "answer_request_signature" in answers:
            signatures = {m.request_signature(exp.answer_prompt) for m in exp.models.values()}
            answers = answers.loc[answers.answer_request_signature.isin(signatures)]
        else:
            answers = pd.DataFrame()
        settings = exp.document.get("budget", {})
        activity = (
            ev.load_table(exp.local_path(settings["activity_file"]))
            if settings.get("activity_file")
            else pd.DataFrame()
        )
        return ev.build_openrouter_budget_report(
            questions=exp.questions,
            languages=exp.languages,
            models=exp.models.values(),
            translation_model=exp.translation_model,
            repeats_per_prompt=exp.repeats,
            translation_samples=translations,
            answer_samples=answers,
            approved_translations=self.approvals(),
            openrouter_activity=activity,
            openrouter_activity_model_aliases=settings.get("activity_model_aliases", {}),
            translation_system_prompt=exp.translation_prompt,
            answer_system_prompt=exp.answer_prompt,
            calibrated_safety_multiplier=settings.get("safety_multiplier", 1.5),
            calibrated_round_up_to_usd=settings.get("round_up_usd", 10.0),
        )
