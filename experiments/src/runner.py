"""CLI runner for Experiment Automation Framework.

This module provides the command-line interface for validating, running,
listing, and resuming experiments.

Usage:
    experiment validate config.yaml      # Validate configuration
    experiment run config.yaml           # Run experiment
    experiment list                      # List past experiments
    experiment resume <experiment_id>    # Resume paused experiment
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, FullExperimentConfig, expand_path, load_config
from .orchestrator import (
    ExperimentContext,
    ExperimentStatus,
    Orchestrator,
    OrchestratorError,
    ScenarioResult,
)
from .reproducibility import ReproducibilityManager, create_snapshot
from .schema import (
    ExperimentParseError,
    ExperimentValidationError,
    load_experiment,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the CLI.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="experiment",
        description="Experiment Automation Framework - Run and manage multi-agent experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  experiment validate experiment.yaml      Validate a configuration file
  experiment run experiment.yaml           Run an experiment
  experiment list                          List past experiments
  experiment list --status completed       List completed experiments only
  experiment resume exp_20231215_143022    Resume a paused experiment
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Override experiments directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an experiment configuration file",
        description="Validate YAML configuration against the schema",
    )
    validate_parser.add_argument(
        "config_file",
        type=str,
        help="Path to the experiment configuration YAML file",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enable strict validation (reject unknown fields)",
    )
    validate_parser.add_argument(
        "--schema",
        action="store_true",
        help="Output JSON schema instead of validating",
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from configuration",
        description="Execute an experiment defined in a YAML configuration file",
    )
    run_parser.add_argument(
        "config_file",
        type=str,
        help="Path to the experiment configuration YAML file",
    )
    run_parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Custom experiment ID (auto-generated if not specified)",
    )
    run_parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Run only specified scenarios (by name)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and prepare but don't execute",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results",
    )
    run_parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saving",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List past experiments",
        description="Display a list of past experiment runs",
    )
    list_parser.add_argument(
        "--status",
        type=str,
        choices=["pending", "running", "paused", "completed", "failed", "cancelled"],
        default=None,
        help="Filter by experiment status",
    )
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Maximum number of experiments to show (default: 20)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all experiments (ignore limit)",
    )

    # Resume command
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a paused experiment",
        description="Continue execution of a paused experiment from checkpoint",
    )
    resume_parser.add_argument(
        "experiment_id",
        type=str,
        help="ID of the experiment to resume",
    )
    resume_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (auto-detected if not specified)",
    )

    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    config_path = Path(args.config_file)

    # Output schema if requested
    if args.schema:
        from .schema import get_schema_info
        schema = get_schema_info()
        print(json.dumps(schema, indent=2))
        return 0

    # Validate the configuration file
    try:
        config = load_experiment(config_path, strict=args.strict)
        print(f"Valid: {config_path}")
        print(f"  Experiment: {config.experiment.name}")
        print(f"  Version: {config.experiment.version}")
        print(f"  Agents: {len(config.agents)}")
        print(f"  Scenarios: {len(config.scenarios)}")

        if config.experiment.seed is not None:
            print(f"  Seed: {config.experiment.seed}")

        if config.experiment.tags:
            print(f"  Tags: {', '.join(config.experiment.tags)}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1

    except ExperimentParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ExperimentValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace, app_config: Config) -> int:
    """Execute the run command.

    Args:
        args: Parsed command-line arguments.
        app_config: Application configuration.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    config_path = Path(args.config_file)

    # Load and validate configuration
    try:
        experiment_config = load_experiment(config_path)
    except (FileNotFoundError, ExperimentParseError, ExperimentValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Dry run - just validate
    if args.dry_run:
        print(f"Dry run: {experiment_config.experiment.name}")
        print("  Configuration validated successfully")
        print("  Experiment would be executed with:")
        print(f"    Scenarios: {len(experiment_config.scenarios)}")
        print(f"    Agents: {len(experiment_config.agents)}")
        return 0

    # Set up reproducibility
    seed = args.seed if args.seed is not None else experiment_config.experiment.seed
    repro_manager = ReproducibilityManager(
        expand_path(app_config.experiments_dir) / "snapshots"
    )
    snapshot = repro_manager.setup(experiment_config, seed=seed)

    # Determine checkpoint directory
    checkpoint_dir = None
    if not args.no_checkpoint and experiment_config.experiment.save_checkpoints:
        checkpoint_dir = expand_path(app_config.checkpoints_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment ID
    experiment_id = args.experiment_id
    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_config.experiment.name}_{timestamp}"

    # Create orchestrator
    orchestrator = Orchestrator(experiment_config, checkpoint_dir=checkpoint_dir)

    # Note: In a real implementation, agents would be registered here based on
    # the agent configurations. For now, we demonstrate the flow.
    print(f"Starting experiment: {experiment_config.experiment.name}")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Config hash: {snapshot.config_hash[:16]}...")

    if seed is not None:
        print(f"  Seed: {seed}")

    # Check if agents are configured
    if not experiment_config.agents:
        print("  Warning: No agents configured")
    else:
        print(f"  Agents: {', '.join(a.id for a in experiment_config.agents)}")

    if not experiment_config.scenarios:
        print("  Warning: No scenarios configured")
        return 0

    scenarios_to_run = args.scenarios or [s.name for s in experiment_config.scenarios]
    print(f"  Scenarios: {', '.join(scenarios_to_run)}")

    # Run the experiment
    try:
        # Note: This is a placeholder for actual agent registration and execution
        # In a production system, agents would be instantiated based on their types
        print("\n[Experiment execution would start here]")
        print("Note: Agent implementations must be registered before execution")
        print("      See documentation for registering custom agents\n")

        # Save snapshot
        output_dir = Path(args.output_dir) if args.output_dir else expand_path(
            app_config.experiments_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = output_dir / f"{experiment_id}_snapshot.json"
        snapshot.save(snapshot_path)
        print(f"Snapshot saved: {snapshot_path}")

        # Save experiment metadata
        metadata = {
            "experiment_id": experiment_id,
            "config_file": str(config_path),
            "config_hash": snapshot.config_hash,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "seed": seed,
            "scenarios": scenarios_to_run,
        }
        metadata_path = output_dir / f"{experiment_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {metadata_path}")

        return 0

    except OrchestratorError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nExperiment interrupted")
        return 130


def cmd_list(args: argparse.Namespace, app_config: Config) -> int:
    """Execute the list command.

    Args:
        args: Parsed command-line arguments.
        app_config: Application configuration.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    experiments_dir = expand_path(app_config.experiments_dir)

    if not experiments_dir.exists():
        if args.output_json:
            print(json.dumps([]))
        else:
            print("No experiments found.")
        return 0

    # Find all experiment metadata files
    metadata_files = list(experiments_dir.glob("*_metadata.json"))
    experiments: list[dict[str, Any]] = []

    for metadata_path in metadata_files:
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Filter by status if specified
            if args.status and metadata.get("status") != args.status:
                continue

            experiments.append(metadata)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read metadata: {metadata_path} - {e}")

    # Sort by start time (newest first)
    experiments.sort(
        key=lambda x: x.get("started_at", ""),
        reverse=True,
    )

    # Apply limit
    if not args.all and len(experiments) > args.limit:
        experiments = experiments[: args.limit]

    # Output results
    if args.output_json:
        print(json.dumps(experiments, indent=2))
    else:
        if not experiments:
            print("No experiments found.")
            return 0

        print(f"Found {len(experiments)} experiment(s):\n")
        print(f"{'ID':<40} {'Status':<12} {'Started':<20} {'Seed':<8}")
        print("-" * 84)

        for exp in experiments:
            exp_id = exp.get("experiment_id", "unknown")[:38]
            status = exp.get("status", "unknown")
            started = exp.get("started_at", "")[:19].replace("T", " ")
            seed = str(exp.get("seed", "-"))

            print(f"{exp_id:<40} {status:<12} {started:<20} {seed:<8}")

    return 0


def cmd_resume(args: argparse.Namespace, app_config: Config) -> int:
    """Execute the resume command.

    Args:
        args: Parsed command-line arguments.
        app_config: Application configuration.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    experiment_id = args.experiment_id

    # Find checkpoint file
    checkpoint_path: Path | None = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = expand_path(app_config.checkpoints_dir)
        potential_checkpoint = checkpoint_dir / f"{experiment_id}.json"
        if potential_checkpoint.exists():
            checkpoint_path = potential_checkpoint

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"Error: Checkpoint not found for experiment: {experiment_id}", file=sys.stderr)
        print("  Use --checkpoint to specify a custom checkpoint file", file=sys.stderr)
        return 1

    # Load checkpoint
    try:
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Failed to read checkpoint - {e}", file=sys.stderr)
        return 1

    # Find and load the original config
    experiments_dir = expand_path(app_config.experiments_dir)
    metadata_path = experiments_dir / f"{experiment_id}_metadata.json"

    if not metadata_path.exists():
        print(f"Error: Metadata not found for experiment: {experiment_id}", file=sys.stderr)
        return 1

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Failed to read metadata - {e}", file=sys.stderr)
        return 1

    config_file = metadata.get("config_file")
    if not config_file or not Path(config_file).exists():
        print(f"Error: Original config file not found: {config_file}", file=sys.stderr)
        return 1

    # Load the experiment config
    try:
        experiment_config = load_experiment(config_file)
    except (ExperimentParseError, ExperimentValidationError) as e:
        print(f"Error: Failed to load config - {e}", file=sys.stderr)
        return 1

    # Restore context
    checkpoint_dir = expand_path(app_config.checkpoints_dir)
    context = ExperimentContext.from_checkpoint_data(
        checkpoint_data,
        experiment_config,
        checkpoint_path=checkpoint_dir / f"{experiment_id}.json",
    )

    if context.status != ExperimentStatus.PAUSED:
        print(
            f"Error: Experiment is not paused (status: {context.status.value})",
            file=sys.stderr,
        )
        return 1

    print(f"Resuming experiment: {experiment_id}")
    print(f"  From scenario: {context.current_scenario}")
    print(f"  From step: {context.current_step}")

    # Note: Actual resumption would require agent registration
    print("\n[Experiment resumption would start here]")
    print("Note: Agent implementations must be registered before execution")

    return 0


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging
    if parsed_args.quiet:
        log_level = "ERROR"
    elif parsed_args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    setup_logging(level=log_level, log_file=parsed_args.log_file)

    # Show help if no command specified
    if parsed_args.command is None:
        parser.print_help()
        return 0

    # Load application config
    app_config = load_config()

    # Override experiments directory if specified
    if parsed_args.config_dir:
        app_config = Config(
            experiments_dir=parsed_args.config_dir,
            checkpoints_dir=str(Path(parsed_args.config_dir) / "checkpoints"),
        )

    # Dispatch to command handler
    if parsed_args.command == "validate":
        return cmd_validate(parsed_args)
    elif parsed_args.command == "run":
        return cmd_run(parsed_args, app_config)
    elif parsed_args.command == "list":
        return cmd_list(parsed_args, app_config)
    elif parsed_args.command == "resume":
        return cmd_resume(parsed_args, app_config)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
