#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
from openpyxl import load_workbook

from qa_workbench import evaluation as ev
from qa_workbench.cli import main
from qa_workbench.config import load_experiment
from qa_workbench.workbench import Workbench
from tests.helpers import experiment_document


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.document = experiment_document()
        self.path = self.root / "experiment.json"

    def load(self):
        self.path.write_text(json.dumps(self.document))
        return load_experiment(self.path)

    def test_authoring_workflow_uses_config_and_round_trips_reviews(self):
        self.document["answers"]["system_prompt"] = "Configured answer instruction"
        bench = Workbench(self.load(), "test")
        call = AsyncMock(return_value=ev.OpenRouterResult(text="Synthetic translation"))
        with patch.object(ev, "_call_openrouter_result_async", call):
            translations = asyncio.run(bench.translate())
        self.assertEqual(len(translations), 1)
        self.assertEqual(
            call.call_args.kwargs["system_prompt"], "Synthetic translation instruction"
        )
        self.assertTrue(bench.approvals().empty)
        self.assertTrue(asyncio.run(bench.answer()).empty)
        folder = bench.directory / "review/translations"
        book = load_workbook(folder / "generated/translation_review_fr.xlsx")
        sheet = book["translations"]
        columns = {c.value: c.column for c in sheet[1]}
        sheet.cell(2, columns["translation_final"], "Reviewed wording")
        sheet.cell(2, columns["translation_review_status"], "approved")
        book.save(folder / "completed/reviewed.xlsx")
        book.close()
        call = AsyncMock(return_value=ev.OpenRouterResult(text="Synthetic answer"))
        with patch.object(ev, "_call_openrouter_result_async", call):
            answers = asyncio.run(bench.answer())
            resumed = asyncio.run(bench.answer())
        self.assertEqual(call.await_count, 4)
        self.assertEqual(call.call_args.kwargs["system_prompt"], "Configured answer instruction")
        self.assertEqual(answers.translated_question.tolist(), ["Reviewed wording"] * 4)
        pd.testing.assert_frame_equal(answers, resumed, check_dtype=False)
        folder = bench.directory / "review/reviewers"
        book = load_workbook(folder / "generated/reviewers_fr.xlsx")
        sheet = book["responses"]
        columns = {c.value: c.column for c in sheet[1]}
        sheet.cell(2, columns["review__helpful"], "Yes")
        rated = folder / "completed/synthetic-rating.xlsx"
        book.save(rated)
        book.close()
        summary = bench.summarize()["reviewers"]
        self.assertEqual(summary["count"].sum(), 1)
        rated.unlink()
        self.assertTrue(bench.summarize()["reviewers"].empty)
        self.assertTrue(ev.load_table(bench.directory / "summaries/reviewers.parquet").empty)

    def test_pending_inputs_make_no_translation_or_answer_calls(self):
        self.document["pending_revisions"] = [
            {"question_id": "q1", "language_code": "fr", "reason": "Awaiting review"}
        ]
        bench = Workbench(self.load(), "held")
        with patch.object(
            ev, "_call_openrouter_result_async", side_effect=AssertionError("No calls allowed")
        ):
            self.assertTrue(asyncio.run(bench.translate()).empty)
            self.assertTrue(asyncio.run(bench.answer()).empty)

    def test_changed_configuration_cannot_reuse_run_directory(self):
        Workbench(self.load(), "run").prepare()
        self.document["questions"][0]["question_text"] = "Revised question"
        with self.assertRaisesRegex(ValueError, "settings changed"):
            Workbench(self.load(), "run").prepare()
        Workbench(self.load(), "new-run").prepare()

    def test_experiment_paths_and_run_names_cannot_escape(self):
        for value in ["../outside", str(self.root / "absolute")]:
            with self.subTest(value=value):
                self.document["rerun"]["source_dir"] = value
                with self.assertRaisesRegex(ValueError, "paths must stay"):
                    self.load()
        self.document["rerun"]["source_dir"] = "source"
        with self.assertRaises(ValueError):
            Workbench(self.load(), "../escape")

    def test_configuration_validation_rejects_ambiguous_or_invalid_requests(self):
        for change in [
            lambda d: d.update(unrecognized=True),
            lambda d: d["questions"].append(dict(d["questions"][0])),
            lambda d: d["questions"][0].update(question_text=" "),
            lambda d: d["models"]["alpha"].update(concurrency=0),
            lambda d: d["models"]["alpha"].update(extra_body={"model": "another/model"}),
            lambda d: d["models"]["alpha"].update(zero_retention_exception=""),
            lambda d: d["answers"].update(repeats_per_prompt=0),
            lambda d: d["rerun"]["candidate_models"].update(unknown="absent"),
            lambda d: d["reviews"]["reviewers"][0].update(options="Yes||No"),
        ]:
            self.document = experiment_document()
            change(self.document)
            with self.assertRaises(ValueError):
                self.load()

    def test_cli_validation_is_offline_and_execution_requires_explicit_flag(self):
        self.load()
        with patch.object(ev.requests, "post", side_effect=AssertionError("Paid request")):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["validate", "--config", str(self.path)])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["translate", "--config", str(self.path)])
        self.assertFalse((self.root / "runs").exists())

    def test_tracked_example_loads_without_private_configuration(self):
        example = Path(__file__).resolve().parents[1] / "examples/experiment.json"
        experiment = load_experiment(example)
        self.assertTrue(experiment.questions.question_text.str.contains("placeholder").all())
        self.assertFalse(experiment.pending_revisions)
        self.assertTrue(
            all(model.zero_retention_exception is None for model in experiment.models.values())
        )


if __name__ == "__main__":
    unittest.main()
