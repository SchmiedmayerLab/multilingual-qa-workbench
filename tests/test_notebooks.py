#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

import tempfile
import unittest
from pathlib import Path

import nbformat

from scripts.check_notebooks import validate_notebook


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_saved_outputs_and_attachments_are_rejected(self):
        for kind in ["outputs", "execution_count", "attachments", "widgets"]:
            with self.subTest(kind=kind):
                notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
                if kind == "outputs":
                    notebook.cells[0].outputs = [
                        nbformat.v4.new_output("stream", name="stdout", text="Synthetic output")
                    ]
                elif kind == "execution_count":
                    notebook.cells[0].execution_count = 1
                elif kind == "attachments":
                    notebook.cells.append(
                        nbformat.v4.new_markdown_cell(
                            "![image](attachment:test.png)",
                            attachments={"test.png": {"image/png": "dGVzdA=="}},
                        )
                    )
                else:
                    notebook.metadata["widgets"] = {"saved": "state"}
                path = self.root / f"{kind}.ipynb"
                nbformat.write(notebook, path)
                with self.assertRaises(ValueError):
                    validate_notebook(path)

    def test_clean_notebook_allows_top_level_await_without_executing_it(self):
        path = self.root / "clean.ipynb"
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell("await api_call_that_must_not_run()"),
            ]
        )
        nbformat.write(notebook, path)
        validate_notebook(path)

    def test_workbench_notebook_compiles_without_saved_state(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/workbench.ipynb"
        validate_notebook(path)


if __name__ == "__main__":
    unittest.main()
