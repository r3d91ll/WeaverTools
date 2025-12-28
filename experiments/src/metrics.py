"""Metrics collection for Experiment Automation Framework.

Provides metric collection with configurable backends including:
- File-based storage (always available)
- MLflow tracking (optional, lazy loaded)

Follows patterns from TheLoom's metrics implementation with lazy
initialization and context managers for timing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency on MLflow
_mlflow_available = False
_mlflow_initialized = False
_mlflow_init_lock = threading.Lock()
_mlflow: Any = None

try:
    import mlflow as _mlflow_module

    _mlflow = _mlflow_module
    _mlflow_available = True
except ImportError:
    logger.debug("mlflow not installed, MLflow backend will be unavailable")


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class MetricValue:
    """A single metric value with metadata."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricValue:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            step=data.get("step"),
            tags=data.get("tags", {}),
        )


@dataclass
class TimingResult:
    """Result from a timing context manager."""

    start_time: float
    end_time: float | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds, or 0 if not finished."""
        if self.duration is not None:
            return self.duration
        if self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0


# ============================================================================
# Backend Protocol and Implementations
# ============================================================================


class MetricsBackend(ABC):
    """Abstract base class for metrics storage backends."""

    @abstractmethod
    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a single metric value.

        Args:
            name: Metric name.
            value: Metric value (must be numeric).
            step: Optional step number for ordered metrics.
            timestamp: Optional timestamp (defaults to now).
            tags: Optional key-value tags for the metric.
        """
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary mapping metric names to values.
            step: Optional step number for all metrics.
        """
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (configuration values).

        Args:
            params: Dictionary of parameter names to values.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics to storage."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""
        ...


class FileMetricsBackend(MetricsBackend):
    """File-based metrics storage backend.

    Stores metrics as JSON files organized by experiment and run.
    Always available as it has no external dependencies.
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        run_id: str | None = None,
    ) -> None:
        """Initialize file backend.

        Args:
            output_dir: Directory to store metrics files.
            experiment_name: Name of the experiment.
            run_id: Optional run identifier (auto-generated if not provided).
        """
        self.output_dir = Path(output_dir).expanduser()
        self.experiment_name = experiment_name
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Create directory structure
        self.run_dir = self.output_dir / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Metrics and params storage
        self._metrics: list[MetricValue] = []
        self._params: dict[str, Any] = {}
        self._lock = threading.Lock()

        # Metrics file path
        self._metrics_file = self.run_dir / "metrics.json"
        self._params_file = self.run_dir / "params.json"

        logger.debug(f"FileMetricsBackend initialized: {self.run_dir}")

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a single metric value."""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=timestamp or datetime.now(timezone.utc),
            step=step,
            tags=tags or {},
        )
        with self._lock:
            self._metrics.append(metric)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        timestamp = datetime.now(timezone.utc)
        for name, value in metrics.items():
            self.log_metric(name, value, step=step, timestamp=timestamp)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        with self._lock:
            self._params.update(params)

    def flush(self) -> None:
        """Flush metrics to file."""
        with self._lock:
            # Write metrics
            metrics_data = [m.to_dict() for m in self._metrics]
            with open(self._metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)

            # Write params
            with open(self._params_file, "w") as f:
                json.dump(self._params, f, indent=2)

        logger.debug(f"Flushed {len(self._metrics)} metrics to {self._metrics_file}")

    def close(self) -> None:
        """Close backend, ensuring all data is flushed."""
        self.flush()

    def get_metrics(self) -> list[MetricValue]:
        """Get all logged metrics.

        Returns:
            List of MetricValue objects.
        """
        with self._lock:
            return list(self._metrics)

    def get_params(self) -> dict[str, Any]:
        """Get all logged parameters.

        Returns:
            Dictionary of parameters.
        """
        with self._lock:
            return dict(self._params)


class MLflowMetricsBackend(MetricsBackend):
    """MLflow-based metrics storage backend.

    Requires mlflow package to be installed. Uses lazy initialization
    to avoid import overhead when not in use.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
    ) -> None:
        """Initialize MLflow backend.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: MLflow experiment name.
            run_id: Existing run ID to resume (optional).
            run_name: Name for new run (optional).

        Raises:
            ImportError: If mlflow is not installed.
        """
        if not _mlflow_available:
            raise ImportError(
                "mlflow is not installed. Install with: pip install mlflow"
            )

        self._tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "mlruns"
        )
        self._experiment_name = experiment_name
        self._run_id = run_id
        self._run_name = run_name
        self._active_run: Any = None
        self._initialized = False
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazily initialize MLflow connection."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            _mlflow.set_tracking_uri(self._tracking_uri)

            if self._experiment_name:
                _mlflow.set_experiment(self._experiment_name)

            if self._run_id:
                # Resume existing run
                self._active_run = _mlflow.start_run(run_id=self._run_id)
            else:
                # Start new run
                self._active_run = _mlflow.start_run(run_name=self._run_name)

            self._initialized = True
            logger.info(
                f"MLflow backend initialized: {self._tracking_uri}, "
                f"run_id={_mlflow.active_run().info.run_id}"
            )

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a single metric to MLflow."""
        self._ensure_initialized()
        # MLflow log_metric accepts step but not timestamp directly
        _mlflow.log_metric(name, value, step=step)

        # Log tags separately if provided
        if tags:
            _mlflow.set_tags({f"{name}_{k}": v for k, v in tags.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics to MLflow."""
        self._ensure_initialized()
        _mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        self._ensure_initialized()
        # Convert non-string values to strings for MLflow
        string_params = {k: str(v) for k, v in params.items()}
        _mlflow.log_params(string_params)

    def flush(self) -> None:
        """Flush is a no-op for MLflow (auto-flushes)."""
        pass

    def close(self) -> None:
        """End the MLflow run."""
        if self._active_run is not None:
            _mlflow.end_run()
            self._active_run = None
            self._initialized = False

    @property
    def run_id(self) -> str | None:
        """Get the current MLflow run ID."""
        if not self._initialized:
            return self._run_id
        return _mlflow.active_run().info.run_id if _mlflow.active_run() else None


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """Main metrics collection interface.

    Provides a unified API for collecting metrics across different backends.
    Supports multiple backends simultaneously and automatic flushing.
    """

    def __init__(
        self,
        backends: list[MetricsBackend] | None = None,
        auto_flush_interval: float | None = None,
    ) -> None:
        """Initialize metrics collector.

        Args:
            backends: List of backends to use. If None, uses in-memory only.
            auto_flush_interval: Seconds between auto-flushes (None to disable).
        """
        self._backends: list[MetricsBackend] = backends or []
        self._auto_flush_interval = auto_flush_interval
        self._last_flush_time = time.time()
        self._lock = threading.Lock()
        self._step_counter = 0
        self._closed = False

    def add_backend(self, backend: MetricsBackend) -> None:
        """Add a backend to the collector.

        Args:
            backend: Backend instance to add.
        """
        with self._lock:
            self._backends.append(backend)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a metric to all backends.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.
            timestamp: Optional timestamp.
            tags: Optional tags.
        """
        if self._closed:
            logger.warning("Attempted to log metric after collector was closed")
            return

        for backend in self._backends:
            try:
                backend.log_metric(name, value, step=step, timestamp=timestamp, tags=tags)
            except Exception as e:
                logger.error(f"Failed to log metric to {type(backend).__name__}: {e}")

        self._maybe_auto_flush()

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log multiple metrics to all backends.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number for all metrics.
        """
        if self._closed:
            logger.warning("Attempted to log metrics after collector was closed")
            return

        for backend in self._backends:
            try:
                backend.log_metrics(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to {type(backend).__name__}: {e}")

        self._maybe_auto_flush()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to all backends.

        Args:
            params: Dictionary of parameter names to values.
        """
        if self._closed:
            logger.warning("Attempted to log params after collector was closed")
            return

        for backend in self._backends:
            try:
                backend.log_params(params)
            except Exception as e:
                logger.error(f"Failed to log params to {type(backend).__name__}: {e}")

    def increment_step(self) -> int:
        """Increment and return the step counter.

        Returns:
            New step value.
        """
        with self._lock:
            self._step_counter += 1
            return self._step_counter

    @property
    def current_step(self) -> int:
        """Get current step counter value."""
        return self._step_counter

    def flush(self) -> None:
        """Flush all backends."""
        for backend in self._backends:
            try:
                backend.flush()
            except Exception as e:
                logger.error(f"Failed to flush {type(backend).__name__}: {e}")

        self._last_flush_time = time.time()

    def close(self) -> None:
        """Close all backends."""
        self._closed = True
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                logger.error(f"Failed to close {type(backend).__name__}: {e}")

    def _maybe_auto_flush(self) -> None:
        """Auto-flush if interval has passed."""
        if self._auto_flush_interval is None:
            return

        if time.time() - self._last_flush_time >= self._auto_flush_interval:
            self.flush()

    def __enter__(self) -> MetricsCollector:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close collector."""
        self.close()


# ============================================================================
# Context Managers and Decorators
# ============================================================================


@contextmanager
def track_duration(
    name: str,
    collector: MetricsCollector | None = None,
    step: int | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[TimingResult, None, None]:
    """Track execution duration and optionally log to collector.

    Example:
        with track_duration("my_operation", collector) as timing:
            do_something()
        print(f"Took {timing.duration_seconds:.2f}s")

    Args:
        name: Name for the duration metric.
        collector: Optional metrics collector to log to.
        step: Optional step number.
        tags: Optional tags for the metric.

    Yields:
        TimingResult object with timing information.
    """
    timing = TimingResult(start_time=time.perf_counter())
    try:
        yield timing
    finally:
        timing.end_time = time.perf_counter()
        timing.duration = timing.end_time - timing.start_time

        if collector is not None:
            collector.log_metric(
                f"{name}_duration_seconds",
                timing.duration,
                step=step,
                tags=tags,
            )


@contextmanager
def track_experiment(
    experiment_name: str,
    collector: MetricsCollector,
    params: dict[str, Any] | None = None,
) -> Generator[TimingResult, None, None]:
    """Track an entire experiment execution.

    Logs experiment start, duration, and success/failure status.

    Args:
        experiment_name: Name of the experiment.
        collector: Metrics collector to use.
        params: Optional parameters to log at start.

    Yields:
        TimingResult object.
    """
    if params:
        collector.log_params(params)

    collector.log_metric("experiment_start", 1.0)
    success = True

    timing = TimingResult(start_time=time.perf_counter())
    try:
        yield timing
    except Exception:
        success = False
        raise
    finally:
        timing.end_time = time.perf_counter()
        timing.duration = timing.end_time - timing.start_time

        collector.log_metrics({
            f"{experiment_name}_duration_seconds": timing.duration,
            "experiment_success": 1.0 if success else 0.0,
        })


# ============================================================================
# Factory Functions
# ============================================================================


def create_file_backend(
    output_dir: str | Path,
    experiment_name: str,
    run_id: str | None = None,
) -> FileMetricsBackend:
    """Create a file-based metrics backend.

    Args:
        output_dir: Directory to store metrics.
        experiment_name: Name of the experiment.
        run_id: Optional run identifier.

    Returns:
        Configured FileMetricsBackend instance.
    """
    return FileMetricsBackend(output_dir, experiment_name, run_id)


def create_mlflow_backend(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
) -> MLflowMetricsBackend:
    """Create an MLflow metrics backend.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        run_id: Existing run ID to resume.
        run_name: Name for new run.

    Returns:
        Configured MLflowMetricsBackend instance.

    Raises:
        ImportError: If mlflow is not installed.
    """
    return MLflowMetricsBackend(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_id=run_id,
        run_name=run_name,
    )


def is_mlflow_available() -> bool:
    """Check if MLflow is available.

    Returns:
        True if mlflow package is installed.
    """
    return _mlflow_available


def create_collector_from_config(
    config: dict[str, Any],
    experiment_name: str,
    run_id: str | None = None,
) -> MetricsCollector:
    """Create a MetricsCollector from configuration dict.

    Expected config structure:
        {
            "storage": {
                "type": "file" | "mlflow",
                "path": "./results",  # for file backend
                "tracking_uri": "...",  # for mlflow backend
            },
            "flush_interval_seconds": 30,
        }

    Args:
        config: Metrics configuration dictionary.
        experiment_name: Name of the experiment.
        run_id: Optional run identifier.

    Returns:
        Configured MetricsCollector instance.
    """
    backends: list[MetricsBackend] = []

    storage = config.get("storage", {})
    storage_type = storage.get("type", "file")

    if storage_type == "file":
        output_dir = storage.get("path", "./results")
        backends.append(create_file_backend(output_dir, experiment_name, run_id))

    elif storage_type == "mlflow":
        if not is_mlflow_available():
            logger.warning(
                "MLflow backend requested but not available. "
                "Falling back to file backend."
            )
            output_dir = storage.get("path", "./results")
            backends.append(create_file_backend(output_dir, experiment_name, run_id))
        else:
            tracking_uri = storage.get("tracking_uri")
            backends.append(
                create_mlflow_backend(
                    tracking_uri=tracking_uri,
                    experiment_name=experiment_name,
                    run_name=run_id,
                )
            )

    flush_interval = config.get("flush_interval_seconds")

    return MetricsCollector(backends=backends, auto_flush_interval=flush_interval)
