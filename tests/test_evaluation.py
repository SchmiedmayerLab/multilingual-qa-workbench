#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
from openpyxl import load_workbook

from qa_workbench import evaluation
from tests.helpers import synthetic_models


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = evaluation.ModelConfig(name="Test", model_id="test/model")
        self.matrix = evaluation.build_question_language_matrix(
            evaluation.questions_to_dataframe([{"question_id": "q1", "question_text": "Test?"}]),
            evaluation.languages_to_dataframe([{"language_code": "fr", "language_name": "French"}]),
        )

    def write_review(self, name, question="Test?", translation="Texte", status="approved"):
        frame = self.matrix.copy()
        frame["question_text"] = question
        frame["translation_final"] = translation
        frame["translation_review_status"] = status
        frame.to_excel(self.root / name, sheet_name="translations", index=False)

    def test_privacy_settings_are_in_provider_and_cannot_be_overridden(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "Test"}}]}
        config = evaluation.ModelConfig(
            name="Test",
            model_id="test/model",
            extra_body={
                "provider": {"order": ["test"], "zdr": False},
                "reasoning": {"max_tokens": 8},
            },
        )
        with (
            patch.object(evaluation.requests, "post", return_value=response) as post,
            patch.object(evaluation, "_openrouter_headers", return_value={}),
        ):
            evaluation._call_openrouter_result(model=config, prompt="Test")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload["provider"],
            {
                "order": ["test"],
                "zdr": True,
                "data_collection": "deny",
            },
        )
        self.assertNotIn("extra_body", payload)
        self.assertEqual(payload["reasoning"], {"max_tokens": 8})
        self.assertFalse(config.extra_body["provider"]["zdr"])

    def test_empty_api_response_is_not_reported_as_timeout(self):
        response = Mock()
        response.json.return_value = {"choices": [None]}
        with (
            patch.object(evaluation.requests, "post", return_value=response),
            patch.object(evaluation, "_openrouter_headers", return_value={}),
            patch.object(evaluation.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "no assistant text"),
        ):
            evaluation._call_openrouter_result(model=self.model, prompt="Test")

    def test_explicit_retention_exception_keeps_no_training(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "Test"}}]}
        models = {
            key: replace(model, extra_body={"provider": {"data_collection": "allow"}})
            for key, model in synthetic_models().items()
        }
        for family, model in models.items():
            with (
                self.subTest(family=family),
                patch.object(evaluation.requests, "post", return_value=response) as post,
                patch.object(evaluation, "_openrouter_headers", return_value={}),
            ):
                evaluation._call_openrouter_result(model=model, prompt="Test")
                provider = post.call_args.kwargs["json"]["provider"]
                self.assertEqual(provider["zdr"], family != "delta")
                self.assertEqual(provider["data_collection"], "deny")
                self.assertEqual(bool(model.zero_retention_exception), family == "delta")
        exception_model = models["delta"]
        self.assertNotEqual(
            exception_model.request_signature(None),
            replace(exception_model, zero_retention_exception=None).request_signature(None),
        )

    def test_temperature_is_omitted_when_unset_and_preserves_explicit_zero(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "Test"}}]}
        for temperature in [None, 0.0]:
            with (
                self.subTest(temperature=temperature),
                patch.object(evaluation.requests, "post", return_value=response) as post,
                patch.object(evaluation, "_openrouter_headers", return_value={}),
            ):
                evaluation._call_openrouter_result(
                    model=replace(self.model, temperature=temperature), prompt="Test"
                )
                payload = post.call_args.kwargs["json"]
                if temperature is None:
                    self.assertNotIn("temperature", payload)
                else:
                    self.assertEqual(payload["temperature"], 0.0)

    def test_changed_question_rejects_stale_approval_before_conflict_detection(self):
        self.write_review("old.xlsx", question="Old wording", translation="Old translation")
        self.write_review("new.xlsx")
        (self.root / "~$new.xlsx").write_bytes(b"Excel lock file")
        approved = evaluation.load_reviewed_translations(self.root, current_matrix=self.matrix)
        self.assertEqual(approved["translated_question"].tolist(), ["Texte"])

    def test_only_nonempty_explicit_approvals_are_loaded(self):
        self.write_review("blank.xlsx", translation=" ")
        self.write_review("pending.xlsx", status="needs_changes")
        self.assertTrue(evaluation.load_reviewed_translations(self.root).empty)

    def test_conflicting_current_approvals_are_rejected(self):
        self.write_review("first.xlsx")
        self.write_review("second.xlsx", translation="Different")
        with self.assertRaisesRegex(ValueError, "Conflicting approved translations"):
            evaluation.load_reviewed_translations(self.root, current_matrix=self.matrix)

    def test_long_answers_and_formula_like_text_survive_excel_round_trip(self):
        answer = "=literal text " + "é文" * 35000
        answers = self.matrix.copy()
        answers["answer_text"] = answer
        answers["translated_question"] = "=not a formula"
        answers["response_key"] = "q1::fr::test/model::1"
        schema = evaluation.review_schema_to_dataframe(
            [
                {"field_key": "helpful", "field_label": "Helpful", "options": "Yes|No"},
            ]
        )
        paths = evaluation.export_review_workbooks(
            answers,
            review_schema=schema,
            output_dir=self.root,
            review_type="clinician",
        )
        workbook = load_workbook(paths[0])
        self.addCleanup(workbook.close)
        sheet = workbook["responses"]
        headers = {cell.value: cell.column for cell in sheet[1]}
        self.assertIn("answer_text__part_3", headers)
        self.assertEqual(sheet.cell(2, headers["answer_text"]).data_type, "s")
        self.assertLessEqual(sheet.row_dimensions[2].height, 409)
        imported = evaluation._read_review_workbook_sheet(paths[0], sheet_name="responses")
        self.assertEqual(imported.loc[0, "answer_text"], answer)
        self.assertEqual(imported.loc[0, "translated_question"], "=not a formula")

    def test_failed_checkpoint_write_preserves_previous_file(self):
        path = self.root / "checkpoint.parquet"
        original = pd.DataFrame({"value": ["saved"]})
        evaluation.save_table(original, path)
        with (
            patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            evaluation.save_table(pd.DataFrame({"value": ["new"]}), path)
        pd.testing.assert_frame_equal(evaluation.load_table(path), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_changed_question_regenerates_translation_checkpoint(self):
        path = self.root / "translations.parquet"
        old = self.matrix.assign(question_text="Old wording", translation_suggestion="Old")
        evaluation.save_table(old, path)
        with patch.object(
            evaluation,
            "_call_openrouter_result_async",
            new=AsyncMock(return_value=evaluation.OpenRouterResult(text="New")),
        ) as call:
            result = asyncio.run(
                evaluation.propose_translations(
                    self.matrix,
                    translation_model=self.model,
                    checkpoint_path=path,
                    show_progress=False,
                )
            )
        self.assertEqual(result.loc[0, "translation_suggestion"], "New")
        call.assert_awaited_once()

    def test_answer_checkpoint_resumes_without_another_api_call(self):
        path = self.root / "answers.parquet"
        translations = self.matrix.assign(translated_question="Texte")
        with patch.object(
            evaluation,
            "_call_openrouter_result_async",
            new=AsyncMock(return_value=evaluation.OpenRouterResult(text="Answer", cost_usd=0.01)),
        ) as call:
            first = asyncio.run(
                evaluation.generate_answers(
                    translations,
                    models=[self.model],
                    checkpoint_path=path,
                    show_progress=False,
                )
            )
            resumed = asyncio.run(
                evaluation.generate_answers(
                    translations,
                    models=[self.model],
                    checkpoint_path=path,
                    show_progress=False,
                )
            )
        call.assert_awaited_once()
        pd.testing.assert_frame_equal(first, resumed, check_dtype=False)

    def test_generation_checkpoints_reject_old_models_settings_and_missing_provenance(self):
        variants = {
            "model": (replace(self.model, model_id="test/new-model"), None),
            "temperature": (replace(self.model, temperature=None), None),
            "tokens": (replace(self.model, max_tokens=100), None),
            "reasoning": (replace(self.model, extra_body={"reasoning": {"effort": "high"}}), None),
            "prompt": (self.model, "New system prompt"),
            "legacy": (self.model, None),
        }
        translations = self.matrix.assign(translated_question="Synthetic translation")
        for stage in ["translation", "answer"]:

            def generate(model, prompt, path):
                if stage == "translation":
                    return evaluation.propose_translations(
                        self.matrix,
                        translation_model=model,
                        translation_system_prompt=prompt,
                        checkpoint_path=path,
                        show_progress=False,
                    )
                return evaluation.generate_answers(
                    translations,
                    models=[model],
                    system_prompt=prompt,
                    checkpoint_path=path,
                    show_progress=False,
                )

            for label, (model, prompt) in variants.items():
                with (
                    self.subTest(stage=stage, changed=label),
                    patch.object(
                        evaluation,
                        "_call_openrouter_result_async",
                        new=AsyncMock(
                            side_effect=[
                                evaluation.OpenRouterResult(text="Old"),
                                evaluation.OpenRouterResult(text="New"),
                            ],
                        ),
                    ) as call,
                ):
                    path = self.root / f"{stage}-{label}.parquet"
                    original = asyncio.run(generate(self.model, None, path))
                    if label == "legacy":
                        evaluation.save_table(
                            original.drop(columns=[f"{stage}_request_signature"]), path
                        )
                    refreshed = asyncio.run(generate(model, prompt, path))
                    text_column = (
                        "translation_suggestion" if stage == "translation" else "answer_text"
                    )
                    self.assertEqual(refreshed.loc[0, text_column], "New")
                    self.assertEqual(call.await_count, 2)

    def test_translation_checkpoint_resumes_after_operational_settings_change(self):
        path = self.root / "translations.parquet"
        with patch.object(
            evaluation,
            "_call_openrouter_result_async",
            new=AsyncMock(return_value=evaluation.OpenRouterResult(text="Synthetic translation")),
        ) as call:
            for model in [self.model, replace(self.model, concurrency=1, timeout_seconds=900)]:
                asyncio.run(
                    evaluation.propose_translations(
                        self.matrix,
                        translation_model=model,
                        checkpoint_path=path,
                        show_progress=False,
                    )
                )
        call.assert_awaited_once()

    def test_activity_spend_uses_activity_count_instead_of_sample_count(self):
        prices = pd.DataFrame(
            [
                {
                    "model_id": self.model.model_id,
                    "model_name": "Test",
                    "pricing_source": "test",
                    "input_cost_per_million": 1.0,
                    "output_cost_per_million": 1.0,
                }
            ]
        )
        samples = pd.DataFrame(
            [
                {
                    "model_id": self.model.model_id,
                    "answer_text": "Answer",
                    "answer_openrouter_cost_usd": 1.0,
                }
            ]
        )
        with patch.object(evaluation, "resolve_openrouter_model_pricing", return_value=prices):
            _, _, per_model = evaluation.estimate_openrouter_cost_scenarios(
                questions=self.matrix,
                languages=self.matrix,
                scenarios={"test": [self.model.model_id]},
                answer_samples=samples,
                observed_answer_spend_by_model={self.model.model_id: 20.0},
                observed_answer_request_count_by_model={self.model.model_id: 10},
            )
        self.assertEqual(per_model.loc[0, "observed_average_cost_per_request_usd"], 2.0)


if __name__ == "__main__":
    unittest.main()
