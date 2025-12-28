"""Storage module for experiment result persistence, querying, and export.

This module provides:
- ExperimentResult dataclass for structured result representation
- ResultStorage class for saving and loading experiment results
- Query functions for filtering and searching experiments
- Export capabilities for JSON and YAML formats

Follows patterns from TheLoom's configuration management with
Pydantic models and YAML safe operations.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .config import FullExperimentConfig, OutputFormat

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScenarioResult:
    """Result from executing a single scenario.

    Attributes:
        name: Scenario name.
        status: Execution status.
        steps_completed: Number of steps completed.
        steps_total: Total number of steps.
        duration_seconds: Execution duration.
        outputs: Step outputs keyed by step index.
        errors: Any errors encountered.
        started_at: When execution started.
        completed_at: When execution completed.
    """

    name: str
    status: ExperimentStatus
    steps_completed: int = 0
    steps_total: int = 0
    duration_seconds: float = 0.0
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "duration_seconds": self.duration_seconds,
            "outputs": self.outputs,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioResult:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            status=ExperimentStatus(data["status"]),
            steps_completed=data.get("steps_completed", 0),
            steps_total=data.get("steps_total", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            outputs=data.get("outputs", {}),
            errors=data.get("errors", []),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
        )


@dataclass
class ExperimentResult:
    """Complete result of an experiment run.

    Attributes:
        experiment_id: Unique identifier for this run.
        experiment_name: Name from configuration.
        status: Overall experiment status.
        config_hash: Hash of the configuration used.
        seed: Random seed used (if any).
        scenarios: Results from each scenario.
        metrics: Collected metrics.
        parameters: Parameters used for the run.
        tags: Tags from configuration.
        duration_seconds: Total execution time.
        started_at: When experiment started.
        completed_at: When experiment completed.
        error_message: Error message if failed.
        checkpoint_path: Path to checkpoint file if paused.
        metadata: Additional metadata.
    """

    experiment_id: str
    experiment_name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    config_hash: str = ""
    seed: int | None = None
    scenarios: list[ScenarioResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error_message: str | None = None
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "metrics": self.metrics,
            "parameters": self.parameters,
            "tags": self.tags,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "checkpoint_path": self.checkpoint_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            experiment_name=data["experiment_name"],
            status=ExperimentStatus(data["status"]),
            config_hash=data.get("config_hash", ""),
            seed=data.get("seed"),
            scenarios=[ScenarioResult.from_dict(s) for s in data.get("scenarios", [])],
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            tags=data.get("tags", []),
            duration_seconds=data.get("duration_seconds", 0.0),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            error_message=data.get("error_message"),
            checkpoint_path=data.get("checkpoint_path"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export result as JSON string.

        Args:
            indent: Indentation level for pretty printing.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        """Export result as YAML string.

        Returns:
            YAML string representation.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentResult:
        """Create from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            ExperimentResult instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ExperimentResult:
        """Create from YAML string.

        Args:
            yaml_str: YAML string representation.

        Returns:
            ExperimentResult instance.
        """
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_config(
        cls,
        config: FullExperimentConfig,
        experiment_id: str,
        config_hash: str = "",
    ) -> ExperimentResult:
        """Create an initial result from experiment configuration.

        Args:
            config: Experiment configuration.
            experiment_id: Unique run identifier.
            config_hash: Hash of the configuration.

        Returns:
            ExperimentResult initialized from config.
        """
        return cls(
            experiment_id=experiment_id,
            experiment_name=config.experiment.name,
            status=ExperimentStatus.PENDING,
            config_hash=config_hash,
            seed=config.experiment.seed,
            parameters=config.experiment.parameters,
            tags=config.experiment.tags,
        )

    @property
    def is_finished(self) -> bool:
        """Check if experiment has finished (completed, failed, or cancelled)."""
        return self.status in (
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        )

    @property
    def is_successful(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == ExperimentStatus.COMPLETED

    def mark_running(self) -> None:
        """Mark experiment as running."""
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def mark_failed(self, error_message: str) -> None:
        """Mark experiment as failed with error message."""
        self.status = ExperimentStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def mark_paused(self, checkpoint_path: str | None = None) -> None:
        """Mark experiment as paused."""
        self.status = ExperimentStatus.PAUSED
        self.checkpoint_path = checkpoint_path

    def mark_cancelled(self) -> None:
        """Mark experiment as cancelled."""
        self.status = ExperimentStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def add_scenario_result(self, scenario_result: ScenarioResult) -> None:
        """Add a scenario result to the experiment."""
        self.scenarios.append(scenario_result)

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Update metrics dictionary."""
        self.metrics.update(metrics)


class ResultStorage:
    """Storage backend for experiment results.

    Provides methods to save, load, query, and export experiment results.
    Results are stored as JSON files in a structured directory hierarchy.

    Attributes:
        base_dir: Root directory for result storage.
        index_file: Path to the results index file.

    Example:
        >>> storage = ResultStorage("./results")
        >>> storage.save(result)
        >>> loaded = storage.load(result.experiment_id)
        >>> all_results = storage.list_results()
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize result storage.

        Args:
            base_dir: Root directory for storing results.
        """
        self._base_dir = Path(base_dir).expanduser()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._base_dir / "index.json"
        self._lock = threading.Lock()

        # Initialize index if needed
        if not self._index_file.exists():
            self._save_index({})

        logger.debug(f"ResultStorage initialized at: {self._base_dir}")

    @property
    def base_dir(self) -> Path:
        """Root directory for result storage."""
        return self._base_dir

    @property
    def index_file(self) -> Path:
        """Path to the results index file."""
        return self._index_file

    def _get_result_dir(self, experiment_name: str, experiment_id: str) -> Path:
        """Get directory path for a specific experiment result.

        Args:
            experiment_name: Name of the experiment.
            experiment_id: Unique run identifier.

        Returns:
            Path to the result directory.
        """
        # Sanitize experiment name for filesystem
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in experiment_name
        )
        return self._base_dir / safe_name / experiment_id

    def _get_result_file(self, experiment_name: str, experiment_id: str) -> Path:
        """Get file path for a specific experiment result.

        Args:
            experiment_name: Name of the experiment.
            experiment_id: Unique run identifier.

        Returns:
            Path to the result JSON file.
        """
        result_dir = self._get_result_dir(experiment_name, experiment_id)
        return result_dir / "result.json"

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the results index.

        Returns:
            Dictionary mapping experiment IDs to index entries.
        """
        with self._lock:
            if self._index_file.exists():
                with open(self._index_file) as f:
                    return json.load(f)
            return {}

    def _save_index(self, index: dict[str, dict[str, Any]]) -> None:
        """Save the results index.

        Args:
            index: Index dictionary to save.
        """
        with self._lock:
            with open(self._index_file, "w") as f:
                json.dump(index, f, indent=2, default=str)

    def _update_index(
        self,
        experiment_id: str,
        experiment_name: str,
        status: ExperimentStatus,
        started_at: datetime,
        tags: list[str],
        config_hash: str,
    ) -> None:
        """Update index with experiment info.

        Args:
            experiment_id: Unique run identifier.
            experiment_name: Name of the experiment.
            status: Current experiment status.
            started_at: When experiment started.
            tags: Experiment tags.
            config_hash: Configuration hash.
        """
        index = self._load_index()
        index[experiment_id] = {
            "experiment_name": experiment_name,
            "status": status.value,
            "started_at": started_at.isoformat(),
            "tags": tags,
            "config_hash": config_hash,
        }
        self._save_index(index)

    def _check_disk_space(self, required_bytes: int = 1024 * 1024) -> bool:
        """Check if sufficient disk space is available.

        Args:
            required_bytes: Minimum required bytes (default 1MB).

        Returns:
            True if sufficient space available.
        """
        try:
            stat = os.statvfs(self._base_dir)
            available = stat.f_frsize * stat.f_bavail
            return available >= required_bytes
        except (OSError, AttributeError):
            # statvfs not available on all platforms
            return True

    def save(self, result: ExperimentResult) -> Path:
        """Save an experiment result.

        Args:
            result: ExperimentResult to save.

        Returns:
            Path to the saved result file.

        Raises:
            OSError: If insufficient disk space or write fails.
        """
        if not self._check_disk_space():
            raise OSError("Insufficient disk space to save result")

        result_dir = self._get_result_dir(result.experiment_name, result.experiment_id)
        result_dir.mkdir(parents=True, exist_ok=True)

        result_file = result_dir / "result.json"

        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Update index
        self._update_index(
            experiment_id=result.experiment_id,
            experiment_name=result.experiment_name,
            status=result.status,
            started_at=result.started_at,
            tags=result.tags,
            config_hash=result.config_hash,
        )

        logger.info(f"Saved experiment result: {result.experiment_id}")
        return result_file

    def load(self, experiment_id: str) -> ExperimentResult | None:
        """Load an experiment result by ID.

        Args:
            experiment_id: Unique run identifier.

        Returns:
            ExperimentResult if found, None otherwise.
        """
        index = self._load_index()
        entry = index.get(experiment_id)

        if not entry:
            logger.warning(f"Experiment not found in index: {experiment_id}")
            return None

        experiment_name = entry["experiment_name"]
        result_file = self._get_result_file(experiment_name, experiment_id)

        if not result_file.exists():
            logger.warning(f"Result file not found: {result_file}")
            return None

        with open(result_file) as f:
            data = json.load(f)

        return ExperimentResult.from_dict(data)

    def exists(self, experiment_id: str) -> bool:
        """Check if an experiment result exists.

        Args:
            experiment_id: Unique run identifier.

        Returns:
            True if result exists.
        """
        index = self._load_index()
        return experiment_id in index

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment result.

        Args:
            experiment_id: Unique run identifier.

        Returns:
            True if deleted, False if not found.
        """
        index = self._load_index()
        entry = index.get(experiment_id)

        if not entry:
            return False

        experiment_name = entry["experiment_name"]
        result_dir = self._get_result_dir(experiment_name, experiment_id)

        if result_dir.exists():
            shutil.rmtree(result_dir)

        del index[experiment_id]
        self._save_index(index)

        logger.info(f"Deleted experiment result: {experiment_id}")
        return True

    def list_results(
        self,
        experiment_name: str | None = None,
        status: ExperimentStatus | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List experiment results with optional filtering.

        Args:
            experiment_name: Filter by experiment name.
            status: Filter by status.
            tags: Filter by tags (any match).
            since: Filter by start date (inclusive).
            until: Filter by start date (inclusive).
            limit: Maximum number of results to return.

        Returns:
            List of index entries matching filters.
        """
        index = self._load_index()
        results = []

        for exp_id, entry in index.items():
            # Apply filters
            if experiment_name and entry["experiment_name"] != experiment_name:
                continue

            if status and entry["status"] != status.value:
                continue

            if tags:
                entry_tags = entry.get("tags", [])
                if not any(tag in entry_tags for tag in tags):
                    continue

            if since:
                started = datetime.fromisoformat(entry["started_at"])
                if started < since:
                    continue

            if until:
                started = datetime.fromisoformat(entry["started_at"])
                if started > until:
                    continue

            results.append({"experiment_id": exp_id, **entry})

        # Sort by started_at descending (most recent first)
        results.sort(
            key=lambda x: x.get("started_at", ""),
            reverse=True,
        )

        if limit:
            results = results[:limit]

        return results

    def get_result_count(self) -> int:
        """Get total number of stored results.

        Returns:
            Number of experiment results.
        """
        index = self._load_index()
        return len(index)

    def export(
        self,
        experiment_id: str,
        output_format: OutputFormat = OutputFormat.JSON,
        output_path: Path | str | None = None,
        include_raw_responses: bool = False,
    ) -> str:
        """Export an experiment result to file.

        Args:
            experiment_id: Unique run identifier.
            output_format: Export format (JSON or YAML).
            output_path: Optional output file path.
            include_raw_responses: Whether to include raw responses.

        Returns:
            Exported content as string.

        Raises:
            ValueError: If experiment not found.
        """
        result = self.load(experiment_id)
        if result is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Optionally filter raw responses
        result_dict = result.to_dict()
        if not include_raw_responses:
            for scenario in result_dict.get("scenarios", []):
                scenario.pop("outputs", None)

        if output_format == OutputFormat.JSON:
            content = json.dumps(result_dict, indent=2, default=str)
        else:
            content = yaml.dump(result_dict, default_flow_style=False, sort_keys=False)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(content)
            logger.info(f"Exported result to: {output_path}")

        return content

    def export_all(
        self,
        output_dir: Path | str,
        output_format: OutputFormat = OutputFormat.JSON,
        filter_status: ExperimentStatus | None = None,
    ) -> list[Path]:
        """Export all experiment results to a directory.

        Args:
            output_dir: Directory to export to.
            output_format: Export format.
            filter_status: Optional status filter.

        Returns:
            List of exported file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        results = self.list_results(status=filter_status)

        for entry in results:
            exp_id = entry["experiment_id"]
            ext = "json" if output_format == OutputFormat.JSON else "yaml"
            output_path = output_dir / f"{exp_id}.{ext}"

            try:
                self.export(exp_id, output_format, output_path)
                exported.append(output_path)
            except Exception as e:
                logger.error(f"Failed to export {exp_id}: {e}")

        return exported

    def cleanup_old_results(
        self,
        days: int = 30,
        keep_successful: bool = True,
        dry_run: bool = False,
    ) -> list[str]:
        """Clean up old experiment results.

        Args:
            days: Delete results older than this many days.
            keep_successful: Keep successful experiments regardless of age.
            dry_run: If True, only return IDs without deleting.

        Returns:
            List of deleted (or would-be-deleted) experiment IDs.
        """
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff = cutoff.replace(day=cutoff.day - days)

        to_delete = []
        results = self.list_results()

        for entry in results:
            started = datetime.fromisoformat(entry["started_at"])

            # Skip if not old enough
            if started >= cutoff:
                continue

            # Skip successful if configured
            if keep_successful and entry["status"] == ExperimentStatus.COMPLETED.value:
                continue

            to_delete.append(entry["experiment_id"])

        if not dry_run:
            for exp_id in to_delete:
                self.delete(exp_id)

        return to_delete


# ============================================================================
# Query Functions
# ============================================================================


def query_experiments(
    storage: ResultStorage,
    name: str | None = None,
    status: ExperimentStatus | None = None,
    tags: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Query experiments with flexible filtering.

    Provides a high-level interface for querying experiment results
    with support for custom predicates.

    Args:
        storage: ResultStorage instance to query.
        name: Filter by experiment name.
        status: Filter by status.
        tags: Filter by tags (any match).
        since: Filter by start date (inclusive).
        until: Filter by start date (inclusive).
        limit: Maximum number of results.
        predicate: Custom filter function.

    Returns:
        List of matching index entries.

    Example:
        >>> storage = ResultStorage("./results")
        >>> results = query_experiments(
        ...     storage,
        ...     status=ExperimentStatus.COMPLETED,
        ...     tags=["benchmark"],
        ...     limit=10,
        ... )
    """
    results = storage.list_results(
        experiment_name=name,
        status=status,
        tags=tags,
        since=since,
        until=until,
        limit=None,  # Apply limit after predicate
    )

    if predicate:
        results = [r for r in results if predicate(r)]

    if limit:
        results = results[:limit]

    return results


def get_latest_result(
    storage: ResultStorage,
    experiment_name: str,
    status: ExperimentStatus | None = None,
) -> ExperimentResult | None:
    """Get the most recent result for an experiment.

    Args:
        storage: ResultStorage instance.
        experiment_name: Name of the experiment.
        status: Optional status filter.

    Returns:
        Most recent ExperimentResult or None.
    """
    results = storage.list_results(
        experiment_name=experiment_name,
        status=status,
        limit=1,
    )

    if not results:
        return None

    return storage.load(results[0]["experiment_id"])


def find_by_config_hash(
    storage: ResultStorage,
    config_hash: str,
) -> list[ExperimentResult]:
    """Find all experiments with a specific configuration hash.

    Useful for finding all runs of the same configuration.

    Args:
        storage: ResultStorage instance.
        config_hash: Configuration hash to search for.

    Returns:
        List of matching ExperimentResults.
    """
    results = []
    index = storage._load_index()

    for exp_id, entry in index.items():
        if entry.get("config_hash") == config_hash:
            result = storage.load(exp_id)
            if result:
                results.append(result)

    return results


def find_by_tag(
    storage: ResultStorage,
    tag: str,
) -> list[ExperimentResult]:
    """Find all experiments with a specific tag.

    Args:
        storage: ResultStorage instance.
        tag: Tag to search for.

    Returns:
        List of matching ExperimentResults.
    """
    results = []
    entries = storage.list_results(tags=[tag])

    for entry in entries:
        result = storage.load(entry["experiment_id"])
        if result:
            results.append(result)

    return results


def aggregate_metrics(
    results: list[ExperimentResult],
    metric_name: str,
) -> dict[str, float]:
    """Aggregate a metric across multiple experiment results.

    Args:
        results: List of experiment results.
        metric_name: Name of the metric to aggregate.

    Returns:
        Dictionary with aggregation statistics (mean, std, min, max, count).
    """
    values = []
    for result in results:
        if metric_name in result.metrics:
            value = result.metrics[metric_name]
            if isinstance(value, (int, float)):
                values.append(float(value))

    if not values:
        return {"count": 0}

    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n if n > 1 else 0
    std_val = variance ** 0.5

    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


# ============================================================================
# Factory Functions
# ============================================================================


def create_storage(
    base_dir: str | Path | None = None,
) -> ResultStorage:
    """Create a ResultStorage instance.

    Args:
        base_dir: Base directory for storage.
            Defaults to ~/.experiments/results.

    Returns:
        Configured ResultStorage instance.
    """
    if base_dir is None:
        base_dir = Path.home() / ".experiments" / "results"
    return ResultStorage(base_dir)


def create_storage_from_config(config: dict[str, Any]) -> ResultStorage:
    """Create a ResultStorage from configuration dictionary.

    Expected config structure:
        {
            "storage": {
                "path": "./results",
            }
        }

    Args:
        config: Configuration dictionary.

    Returns:
        Configured ResultStorage instance.
    """
    storage_config = config.get("storage", {})
    base_dir = storage_config.get("path", "~/.experiments/results")
    return ResultStorage(base_dir)
