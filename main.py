"""Command-line entrypoint for running project tasks."""

from __future__ import annotations

import argparse

from src.sequence_experiments import run_all_sequence_experiments
from src.smoke_tests import run_all_smoke_experiments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run experiment workflows for the algorithmic solvability project.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["smoke", "sequence"],
        default="smoke",
        help="Workflow to run. Defaults to the TASK-11 smoke suite.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Directory where experiment artifacts should be written.",
    )
    args = parser.parse_args()

    if args.command == "smoke":
        artifacts = run_all_smoke_experiments(output_root=args.output_root)
    elif args.command == "sequence":
        artifacts = run_all_sequence_experiments(output_root=args.output_root)

    for experiment_id, artifact in artifacts.items():
        print(f"{experiment_id}: {artifact.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
