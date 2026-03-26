"""Command-line entrypoint for running project tasks."""

from __future__ import annotations

import argparse

from src.bonus_experiments import run_all_bonus_experiments
from src.classification_experiments import run_all_classification_experiments
from src.diagnostic_experiments import run_all_diagnostic_experiments
from src.sequence_experiments import run_all_sequence_experiments
from src.smoke_tests import run_all_smoke_experiments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run experiment workflows for the algorithmic solvability project.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["smoke", "sequence", "classification", "diagnostic", "bonus"],
        default="smoke",
        help="Workflow to run. Defaults to the TASK-11 smoke suite. Use 'bonus' for TASK-15.",
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
    elif args.command == "classification":
        artifacts = run_all_classification_experiments(output_root=args.output_root)
    elif args.command == "diagnostic":
        artifacts = run_all_diagnostic_experiments(
            output_root=args.output_root,
            results_root=args.output_root,
        )
    elif args.command == "bonus":
        artifacts = run_all_bonus_experiments(output_root=args.output_root)

    for experiment_id, artifact in artifacts.items():
        print(f"{experiment_id}: {artifact.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
