#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""Validate tracked notebooks without executing API calls or reading study data."""

import ast
import subprocess
from pathlib import Path

import nbformat


def validate_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if notebook.metadata.get("widgets"):
        raise ValueError(f"{path}: saved widget state must be removed")
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.get("attachments"):
            raise ValueError(f"{path}: cell {index} contains embedded attachments")
        if cell.cell_type != "code":
            continue
        if cell.outputs or cell.execution_count is not None:
            raise ValueError(f"{path}: clear outputs and execution counts in cell {index}")
        if cell.metadata.get("execution") or cell.metadata.get("ExecuteTime"):
            raise ValueError(f"{path}: clear execution metadata in cell {index}")
        compile(cell.source, f"{path}:cell {index}", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "--", "*.ipynb"], cwd=root)
        .decode()
        .split("\0")
    )
    for name in filter(None, paths):
        validate_notebook(root / name)
        print(f"Validated {name}: no saved outputs or attachments")
