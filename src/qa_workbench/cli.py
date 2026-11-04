#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
# SPDX-License-Identifier: MIT

"""Command-line entry point for an experiment workbench."""

import argparse
import asyncio
import json

from dotenv import load_dotenv

from .config import load_experiment
from .workbench import Workbench


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["validate", "prepare", "budget", "translate", "answers", "rerun", "summarize"],
    )
    parser.add_argument("--config", required=True, help="Path to experiment.json")
    parser.add_argument(
        "--run-name", default="initial", help="Directory under this experiment's runs/"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Allow paid translation/answer requests"
    )
    args = parser.parse_args(argv)
    if args.command in {"translate", "answers"} and not args.execute:
        parser.error(f"{args.command} requires --execute because it can make paid API calls")
    experiment = load_experiment(args.config)
    workbench = Workbench(experiment, args.run_name)
    if args.command == "validate":
        print(json.dumps(workbench.summary(), indent=2))
    elif args.command == "prepare":
        workbench.prepare()
        print(json.dumps(workbench.summary(), indent=2))
    elif args.command == "budget":
        report = workbench.budget()
        print(report.overview.to_string(index=False))
        print(report.recommended_budgets.to_string(index=False))
    elif args.command == "summarize":
        for group, summary in workbench.summarize().items():
            print(group)
            print(summary.to_string(index=False))
    elif args.command == "rerun":
        plan = workbench.prepare_rerun()
        print(
            f"Workbooks: {len(plan.source_hashes)}; answers: {len(plan.requests)}; "
            f"held back: {len(plan.pending)}"
        )
        print(plan.requests.groupby("model_id").size().to_string())
        print(f"Plan: {workbench.directory / 'plan.json'}")
        if args.execute:
            load_dotenv()
            print(f"Saved {len(asyncio.run(workbench.rerun()))} answers")
        else:
            print("Preparation only. Add --execute to make paid answer calls.")
    else:
        load_dotenv()
        method = workbench.translate if args.command == "translate" else workbench.answer
        print(f"Saved {len(asyncio.run(method()))} rows in {workbench.directory}")


if __name__ == "__main__":
    main()
