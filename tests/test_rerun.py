#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

from qa_workbench import evaluation, rerun
from qa_workbench.config import load_experiment
from tests.helpers import experiment_document, synthetic_models


class RerunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "run"
        self.models = synthetic_models(concurrency=1)
        self.document = experiment_document()
        self.schema = pd.DataFrame(
            [{"field_key": "helpful", "field_label": "Helpful", "options": "Yes | No"}]
        )

    def load_plan(self, models=None):
        document = {
            **self.document,
            "models": {
                k: asdict(m) for k, m in (self.models if models is None else models).items()
            },
        }
        path = self.root / "experiment.json"
        path.write_text(json.dumps(document))
        return rerun.load_rerun_plan(load_experiment(path))

    def row(self, **overrides):
        return {
            "question_id": "q1",
            "language_code": "fr",
            "language_name": "French",
            "question_text": "Synthetic question?",
            "translated_question": "  Texte approuvé\n文  ",
            "candidate_name": "Old Alpha",
            "model_id": "alpha/old",
            "repeat_index": 1,
            "response_key": "synthetic-source-key",
            "reference_answer": "Discarded reference",
            "answer_text": "Discarded old answer",
            "review__helpful": "Yes",
            "reviewer_name": "Synthetic reviewer",
            **overrides,
        }

    def write_workbook(self, rows, name="source.xlsx", schema=None):
        path = self.source / name
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="responses", index=False)
            (self.schema if schema is None else schema).to_excel(
                writer, sheet_name="criteria", index=False
            )
        return path

    def test_import_preserves_approved_text_and_discards_answers_references_and_ratings(self):
        source = self.write_workbook([self.row()])
        original = source.read_bytes()
        (self.source / "~$source.xlsx").write_bytes(b"Excel lock file")
        plan = self.load_plan()
        row = plan.requests.iloc[0]
        self.assertEqual(row.translated_question, "  Texte approuvé\n文  ")
        self.assertEqual(row.reference_answer, "")
        self.assertEqual(row.model_id, self.models["alpha"].model_id)
        self.assertEqual(row.source_model_id, "alpha/old")
        self.assertEqual(row.source_row, 2)
        self.assertEqual(row.source_response_key, "synthetic-source-key")
        self.assertNotIn("answer_text", plan.requests)
        self.assertNotIn("review__helpful", plan.requests)
        self.assertNotIn("reviewer_name", plan.requests)
        rerun.prepare_rerun(plan, self.output)
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue((self.output / "plan.json").exists())
        manifest_models = {model["model_id"]: model for model in plan.manifest()["models"]}
        self.assertTrue(manifest_models["delta/new"]["zero_retention_exception"])
        self.assertIsNone(
            manifest_models[self.models["alpha"].model_id]["zero_retention_exception"]
        )

    def test_pending_pairs_are_excluded_for_every_model_without_adding_missing_pairs(self):
        rows = []
        for item in self.document["pending_revisions"]:
            question, language = item["question_id"], item["language_code"]
            for candidate, family in self.document["rerun"]["candidate_models"].items():
                rows.append(
                    self.row(
                        question_id=question,
                        language_code=language,
                        language_name=language,
                        candidate_name=candidate,
                        model_id=f"{family}/old",
                        response_key=f"{language}-{family}",
                    )
                )
        rows.append(self.row(repeat_index=3))
        self.write_workbook(rows)
        plan = self.load_plan()
        self.assertEqual(len(plan.pending), 8)
        self.assertEqual(len(plan.requests), 1)
        self.assertEqual(plan.requests.iloc[0].repeat_index, 3)

    def test_invalid_sources_stop_before_generation(self):
        for updates, message in [
            ({"translated_question": " "}, "blank or non-text translated_question"),
            ({"candidate_name": "Unknown"}, "unmapped candidate_name"),
            ({"model_id": "gamma/old"}, "families disagree"),
            ({"repeat_index": 1.5}, "positive integer"),
            ({"repeat_index": 0}, "positive integer"),
        ]:
            with self.subTest(updates=updates):
                self.write_workbook([self.row(**updates)])
                with self.assertRaisesRegex(ValueError, message):
                    self.load_plan()

    def test_duplicate_requests_and_conflicting_translations_are_rejected(self):
        self.write_workbook([self.row()])
        self.write_workbook([self.row(response_key="another-key")], "copy.xlsx")
        with self.assertRaisesRegex(ValueError, "Duplicate response"):
            self.load_plan()
        self.write_workbook(
            [
                self.row(
                    response_key="another-key",
                    candidate_name="Old Beta",
                    model_id="beta/old",
                    translated_question="Conflicting approved text",
                )
            ],
            "copy.xlsx",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting source"):
            self.load_plan()

    def test_rubric_mismatch_is_rejected(self):
        self.write_workbook([self.row()])
        self.write_workbook(
            [self.row(question_id="q2", response_key="q2")],
            "other.xlsx",
            self.schema.assign(options="Different options"),
        )
        with self.assertRaisesRegex(ValueError, "rubric differs"):
            self.load_plan()

    def test_rerun_configuration_controls_prompt_rubric_aliases_and_references(self):
        self.document["answers"]["system_prompt"] = "Configured rerun instruction"
        self.document["reviews"]["reviewers"][0]["field_label"] = "Configured criterion"
        self.document["rerun"]["reference_answers"] = "preserve"
        self.document["rerun"]["source_provider_aliases"] = {"previous-provider": "alpha"}
        self.write_workbook([self.row(model_id="previous-provider/old")])
        plan = self.load_plan()
        call = AsyncMock(return_value=evaluation.OpenRouterResult(text="New answer"))
        with patch.object(evaluation, "_call_openrouter_result_async", call):
            answers = asyncio.run(rerun.run_rerun(plan, self.output))
        self.assertEqual(call.call_args.kwargs["system_prompt"], "Configured rerun instruction")
        self.assertEqual(answers.reference_answer.tolist(), ["Discarded reference"])
        criteria = evaluation._read_review_workbook_sheet(
            self.output / "review/reviewers/generated/reviewers_fr.xlsx", sheet_name="criteria"
        )
        self.assertEqual(criteria.field_label.tolist(), ["Configured criterion"])

    def test_changed_source_or_model_requires_a_separate_run_directory(self):
        self.write_workbook([self.row()])
        plan = self.load_plan()
        rerun.prepare_rerun(plan, self.output)
        rerun.prepare_rerun(plan, self.output)
        changed = {**self.models, "alpha": replace(self.models["alpha"], max_tokens=10)}
        with self.assertRaisesRegex(ValueError, "Sources or settings changed"):
            rerun.prepare_rerun(self.load_plan(models=changed), self.output)
        self.write_workbook([self.row(translated_question="Revised approved text")])
        with self.assertRaisesRegex(ValueError, "Sources or settings changed"):
            rerun.prepare_rerun(self.load_plan(), self.output)
        with self.assertRaisesRegex(ValueError, "separate output directory"):
            rerun.prepare_rerun(plan, self.source)

    def test_sparse_requests_export_only_new_answers_and_resume_without_api_calls(self):
        self.write_workbook(
            [
                self.row(),
                self.row(
                    question_id="q2",
                    response_key="q2",
                    candidate_name="Old Gamma",
                    model_id="gamma/old",
                    repeat_index=3,
                ),
            ]
        )
        plan = self.load_plan(models=self.models)
        call = AsyncMock(return_value=evaluation.OpenRouterResult(text="Fresh synthetic answer"))
        with patch.object(evaluation, "_call_openrouter_result_async", call):
            answers = asyncio.run(rerun.run_rerun(plan, self.output))
            resumed = asyncio.run(rerun.run_rerun(plan, self.output))
        self.assertEqual(call.await_count, 2)
        self.assertEqual(answers.repeat_index.tolist(), [1, 3])
        self.assertEqual(answers.model_id.tolist(), plan.requests.model_id.tolist())
        self.assertEqual(
            answers.translated_question.tolist(), plan.requests.translated_question.tolist()
        )
        pd.testing.assert_frame_equal(answers, resumed, check_dtype=False)
        exported = evaluation._read_review_workbook_sheet(
            self.output / "review/reviewers/generated/reviewers_fr.xlsx", sheet_name="responses"
        )
        self.assertEqual(len(exported), 2)
        self.assertEqual(exported.answer_text.tolist(), ["Fresh synthetic answer"] * 2)
        for column in ["reference_answer", "review__helpful", "reviewer_name", "review_status"]:
            self.assertEqual(exported[column].tolist(), [""] * 2)
        self.assertTrue(exported.response_key.str.contains("::").all())

    def test_failed_run_saves_completed_answers_and_resumes_remaining_requests(self):
        self.write_workbook([self.row(), self.row(question_id="q2", response_key="q2")])
        plan = self.load_plan(models=self.models)

        async def fail_second(**kwargs):
            if call.await_count == 2:
                await asyncio.sleep(0.01)
                raise RuntimeError("Synthetic provider failure")
            return evaluation.OpenRouterResult(text="First answer")

        call = AsyncMock(side_effect=fail_second)
        with patch.object(evaluation, "_call_openrouter_result_async", call):
            with self.assertRaisesRegex(RuntimeError, "Synthetic provider failure"):
                asyncio.run(rerun.run_rerun(plan, self.output))
        self.assertEqual(len(evaluation.load_table(self.output / "answers.parquet")), 1)
        self.assertFalse((self.output / "review/reviewers/generated").exists())
        call = AsyncMock(return_value=evaluation.OpenRouterResult(text="Second answer"))
        with patch.object(evaluation, "_call_openrouter_result_async", call):
            answers = asyncio.run(rerun.run_rerun(plan, self.output))
        self.assertEqual(call.await_count, 1)
        self.assertEqual(answers.answer_text.tolist(), ["First answer", "Second answer"])

    def test_failure_preserves_inflight_answers_and_stops_queued_requests(self):
        self.write_workbook(
            [self.row(question_id=f"q{index}", response_key=f"q{index}") for index in range(3)]
        )
        models = {family: replace(model, concurrency=2) for family, model in self.models.items()}
        plan = self.load_plan(models=models)

        async def fail_first(**kwargs):
            if call.await_count == 1:
                await asyncio.sleep(0.01)
                raise RuntimeError("Synthetic routing failure")
            await asyncio.sleep(0.03)
            return evaluation.OpenRouterResult(text="In-flight answer")

        call = AsyncMock(side_effect=fail_first)
        with patch.object(evaluation, "_call_openrouter_result_async", call):
            with self.assertRaisesRegex(RuntimeError, "Synthetic routing failure"):
                asyncio.run(rerun.run_rerun(plan, self.output))
        self.assertEqual(call.await_count, 2)
        saved = evaluation.load_table(self.output / "answers.parquet")
        self.assertEqual(saved.question_id.tolist(), ["q1"])
        self.assertEqual(saved.answer_text.tolist(), ["In-flight answer"])


if __name__ == "__main__":
    unittest.main()
