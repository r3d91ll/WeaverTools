"""Reproducibility module for seed management and configuration snapshots.

This module provides tools to ensure experiment reproducibility through:
- Random seed management across Python, NumPy, and PyTorch
- Configuration snapshotting and versioning
- Environment capture for exact reproduction
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import yaml

from .config import FullExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class SeedState:
    """Captured state of random number generators.

    Attributes:
        seed: The seed value used.
        python_state: Python random module state (if captured).
        numpy_state: NumPy random state (if captured and available).
        torch_state: PyTorch random state (if captured and available).
        cuda_state: CUDA random state (if captured and available).
    """

    seed: int
    python_state: tuple[Any, ...] | None = None
    numpy_state: dict[str, Any] | None = None
    torch_state: bytes | None = None
    cuda_state: bytes | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the seed state.
        """
        return {
            "seed": self.seed,
            "timestamp": self.timestamp.isoformat(),
            "has_python_state": self.python_state is not None,
            "has_numpy_state": self.numpy_state is not None,
            "has_torch_state": self.torch_state is not None,
            "has_cuda_state": self.cuda_state is not None,
        }


@dataclass
class EnvironmentSnapshot:
    """Captured environment information for reproducibility.

    Attributes:
        python_version: Python version string.
        platform_info: Platform information.
        packages: Installed package versions.
        env_vars: Relevant environment variables.
        timestamp: When the snapshot was taken.
    """

    python_version: str
    platform_info: str
    packages: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the environment snapshot.
        """
        return {
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "packages": self.packages,
            "env_vars": self.env_vars,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def capture(cls, include_env_vars: list[str] | None = None) -> "EnvironmentSnapshot":
        """Capture current environment information.

        Args:
            include_env_vars: List of environment variable names to capture.
                Defaults to common experiment-related variables.

        Returns:
            EnvironmentSnapshot with captured information.
        """
        # Default environment variables to capture
        default_env_vars = [
            "EXPERIMENT_STORAGE_DIR",
            "EXPERIMENT_MLFLOW_TRACKING_URI",
            "EXPERIMENT_LOG_LEVEL",
            "PYTHONPATH",
            "PYTHONHASHSEED",
        ]
        env_var_names = include_env_vars or default_env_vars

        env_vars = {}
        for var in env_var_names:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        # Try to get installed packages
        packages = {}
        try:
            import importlib.metadata

            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"]] = dist.version
        except Exception:
            logger.debug("Could not capture package versions")

        return cls(
            python_version=sys.version,
            platform_info=platform.platform(),
            packages=packages,
            env_vars=env_vars,
        )


@dataclass
class ConfigSnapshot:
    """Snapshot of experiment configuration for reproducibility.

    Attributes:
        config: The experiment configuration.
        config_hash: SHA-256 hash of the configuration.
        config_yaml: YAML representation of the configuration.
        environment: Environment snapshot.
        seed_state: Random seed state.
        timestamp: When the snapshot was created.
        notes: Optional notes about the snapshot.
    """

    config: FullExperimentConfig
    config_hash: str
    config_yaml: str
    environment: EnvironmentSnapshot
    seed_state: SeedState | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the snapshot.
        """
        return {
            "config_hash": self.config_hash,
            "config_yaml": self.config_yaml,
            "environment": self.environment.to_dict(),
            "seed_state": self.seed_state.to_dict() if self.seed_state else None,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        """Save snapshot to a file.

        Args:
            path: Path to save the snapshot.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON for structured data
        snapshot_data = self.to_dict()
        with open(path, "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

        logger.info(f"Saved config snapshot to {path}")

    @classmethod
    def load(cls, path: Path, config: FullExperimentConfig) -> "ConfigSnapshot":
        """Load snapshot from a file.

        Args:
            path: Path to load the snapshot from.
            config: The experiment configuration (needed for full reconstruction).

        Returns:
            ConfigSnapshot loaded from the file.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        environment = EnvironmentSnapshot(
            python_version=data["environment"]["python_version"],
            platform_info=data["environment"]["platform_info"],
            packages=data["environment"].get("packages", {}),
            env_vars=data["environment"].get("env_vars", {}),
            timestamp=datetime.fromisoformat(data["environment"]["timestamp"]),
        )

        seed_state = None
        if data.get("seed_state"):
            seed_data = data["seed_state"]
            seed_state = SeedState(
                seed=seed_data["seed"],
                timestamp=datetime.fromisoformat(seed_data["timestamp"]),
            )

        return cls(
            config=config,
            config_hash=data["config_hash"],
            config_yaml=data["config_yaml"],
            environment=environment,
            seed_state=seed_state,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            notes=data.get("notes", ""),
        )


def set_seed(seed: int, include_numpy: bool = True, include_torch: bool = True) -> SeedState:
    """Set random seeds for reproducibility across libraries.

    Sets the random seed for Python's random module and optionally for
    NumPy and PyTorch if they are available and requested.

    Args:
        seed: The seed value to use.
        include_numpy: Whether to set NumPy seed if available.
        include_torch: Whether to set PyTorch seed if available.

    Returns:
        SeedState capturing the seed configuration.

    Example:
        >>> state = set_seed(42)
        >>> print(state.seed)
        42
    """
    logger.info(f"Setting random seed: {seed}")

    # Set Python random seed
    random.seed(seed)
    python_state = random.getstate()

    # Set PYTHONHASHSEED for hash reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy_state = None
    torch_state = None
    cuda_state = None

    # Set NumPy seed if available and requested
    if include_numpy:
        try:
            import numpy as np

            np.random.seed(seed)
            numpy_state = {"seed": seed}
            logger.debug("NumPy seed set")
        except ImportError:
            logger.debug("NumPy not available, skipping")

    # Set PyTorch seed if available and requested
    if include_torch:
        try:
            import torch

            torch.manual_seed(seed)
            torch_state = b"set"  # Indicate torch seed was set

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                cuda_state = b"set"
                logger.debug("CUDA seeds set")

            # For fully deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("PyTorch seed set with deterministic mode")
        except ImportError:
            logger.debug("PyTorch not available, skipping")

    return SeedState(
        seed=seed,
        python_state=python_state,
        numpy_state=numpy_state,
        torch_state=torch_state,
        cuda_state=cuda_state,
    )


def get_seed_from_config(config: FullExperimentConfig) -> int | None:
    """Extract seed from experiment configuration.

    Args:
        config: The experiment configuration.

    Returns:
        The seed value if specified, None otherwise.
    """
    return config.experiment.seed


def hash_config(config: FullExperimentConfig) -> str:
    """Generate a deterministic hash of the configuration.

    Creates a SHA-256 hash of the configuration for version tracking
    and change detection.

    Args:
        config: The experiment configuration.

    Returns:
        Hexadecimal hash string.
    """
    # Convert config to JSON for consistent hashing
    config_dict = config.model_dump(mode="json")
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()


def config_to_yaml(config: FullExperimentConfig) -> str:
    """Convert configuration to YAML string.

    Args:
        config: The experiment configuration.

    Returns:
        YAML representation of the configuration.
    """
    config_dict = config.model_dump(mode="json")
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


def create_snapshot(
    config: FullExperimentConfig,
    seed_state: SeedState | None = None,
    notes: str = "",
) -> ConfigSnapshot:
    """Create a complete snapshot of experiment configuration and environment.

    This captures everything needed to reproduce an experiment:
    - Full configuration
    - Configuration hash for versioning
    - Environment information (Python version, packages)
    - Random seed state

    Args:
        config: The experiment configuration.
        seed_state: Optional seed state to include.
        notes: Optional notes about the snapshot.

    Returns:
        ConfigSnapshot with all reproducibility information.

    Example:
        >>> config = load_experiment("experiment.yaml")
        >>> seed_state = set_seed(42)
        >>> snapshot = create_snapshot(config, seed_state)
        >>> snapshot.save(Path("./snapshots/experiment_v1.json"))
    """
    return ConfigSnapshot(
        config=config,
        config_hash=hash_config(config),
        config_yaml=config_to_yaml(config),
        environment=EnvironmentSnapshot.capture(),
        seed_state=seed_state,
        notes=notes,
    )


def compare_snapshots(
    snapshot1: ConfigSnapshot,
    snapshot2: ConfigSnapshot,
) -> dict[str, Any]:
    """Compare two configuration snapshots for differences.

    Args:
        snapshot1: First snapshot.
        snapshot2: Second snapshot.

    Returns:
        Dictionary describing differences between snapshots.
    """
    differences: dict[str, Any] = {
        "config_changed": snapshot1.config_hash != snapshot2.config_hash,
        "seed_changed": False,
        "environment_changes": {},
    }

    # Check seed changes
    if snapshot1.seed_state and snapshot2.seed_state:
        differences["seed_changed"] = snapshot1.seed_state.seed != snapshot2.seed_state.seed
    elif snapshot1.seed_state or snapshot2.seed_state:
        differences["seed_changed"] = True

    # Check environment changes
    env1 = snapshot1.environment
    env2 = snapshot2.environment

    if env1.python_version != env2.python_version:
        differences["environment_changes"]["python_version"] = {
            "before": env1.python_version,
            "after": env2.python_version,
        }

    # Check for package version changes
    all_packages = set(env1.packages.keys()) | set(env2.packages.keys())
    package_changes = {}
    for pkg in all_packages:
        v1 = env1.packages.get(pkg)
        v2 = env2.packages.get(pkg)
        if v1 != v2:
            package_changes[pkg] = {"before": v1, "after": v2}

    if package_changes:
        differences["environment_changes"]["packages"] = package_changes

    return differences


class ReproducibilityManager:
    """Manages reproducibility for experiment execution.

    This class provides a high-level interface for managing experiment
    reproducibility, including seed management, snapshot creation and
    storage, and experiment comparison.

    Attributes:
        snapshot_dir: Directory for storing snapshots.
        current_seed_state: Current random seed state.
        current_snapshot: Current configuration snapshot.

    Example:
        >>> manager = ReproducibilityManager(Path("./snapshots"))
        >>> config = load_experiment("experiment.yaml")
        >>> manager.setup(config)
        >>> # Run experiment...
        >>> manager.save_snapshot("run_001")
    """

    def __init__(self, snapshot_dir: Path | str | None = None) -> None:
        """Initialize the reproducibility manager.

        Args:
            snapshot_dir: Directory for storing snapshots.
                Defaults to ~/.experiments/snapshots.
        """
        if snapshot_dir is None:
            snapshot_dir = Path.home() / ".experiments" / "snapshots"
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._current_seed_state: SeedState | None = None
        self._current_snapshot: ConfigSnapshot | None = None
        self._config: FullExperimentConfig | None = None

        logger.debug(f"ReproducibilityManager initialized with dir: {self._snapshot_dir}")

    @property
    def snapshot_dir(self) -> Path:
        """Directory for storing snapshots."""
        return self._snapshot_dir

    @property
    def current_seed_state(self) -> SeedState | None:
        """Current random seed state."""
        return self._current_seed_state

    @property
    def current_snapshot(self) -> ConfigSnapshot | None:
        """Current configuration snapshot."""
        return self._current_snapshot

    def setup(
        self,
        config: FullExperimentConfig,
        seed: int | None = None,
        notes: str = "",
    ) -> ConfigSnapshot:
        """Set up reproducibility for an experiment.

        This method:
        1. Sets the random seed (from config or parameter)
        2. Creates a configuration snapshot
        3. Stores the snapshot for later reference

        Args:
            config: Experiment configuration.
            seed: Optional seed override. Uses config seed if not provided.
            notes: Optional notes about this setup.

        Returns:
            ConfigSnapshot created during setup.
        """
        self._config = config

        # Determine seed to use
        effective_seed = seed if seed is not None else get_seed_from_config(config)

        # Set seed if specified
        if effective_seed is not None:
            self._current_seed_state = set_seed(effective_seed)
            logger.info(f"Set up reproducibility with seed: {effective_seed}")
        else:
            self._current_seed_state = None
            logger.info("No seed specified, random behavior will not be reproducible")

        # Create snapshot
        self._current_snapshot = create_snapshot(
            config=config,
            seed_state=self._current_seed_state,
            notes=notes,
        )

        return self._current_snapshot

    def save_snapshot(self, name: str) -> Path:
        """Save the current snapshot to disk.

        Args:
            name: Name for the snapshot file (without extension).

        Returns:
            Path to the saved snapshot.

        Raises:
            RuntimeError: If no snapshot has been created.
        """
        if self._current_snapshot is None:
            raise RuntimeError("No snapshot available. Call setup() first.")

        path = self._snapshot_dir / f"{name}.json"
        self._current_snapshot.save(path)
        return path

    def load_snapshot(self, name: str) -> ConfigSnapshot:
        """Load a snapshot from disk.

        Args:
            name: Name of the snapshot file (without extension).

        Returns:
            Loaded ConfigSnapshot.

        Raises:
            FileNotFoundError: If snapshot file doesn't exist.
            RuntimeError: If no config is set for reconstruction.
        """
        if self._config is None:
            raise RuntimeError("No config available. Call setup() first.")

        path = self._snapshot_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")

        return ConfigSnapshot.load(path, self._config)

    def list_snapshots(self) -> list[str]:
        """List all available snapshots.

        Returns:
            List of snapshot names (without extensions).
        """
        return [
            p.stem for p in self._snapshot_dir.glob("*.json")
            if p.is_file()
        ]

    def compare_with_snapshot(self, name: str) -> dict[str, Any]:
        """Compare current configuration with a saved snapshot.

        Args:
            name: Name of the snapshot to compare with.

        Returns:
            Dictionary describing differences.

        Raises:
            RuntimeError: If no current snapshot exists.
        """
        if self._current_snapshot is None:
            raise RuntimeError("No current snapshot. Call setup() first.")

        saved_snapshot = self.load_snapshot(name)
        return compare_snapshots(saved_snapshot, self._current_snapshot)

    def reset_seed(self) -> None:
        """Reset the random seed to the initial value.

        Useful for running multiple trials with the same seed.

        Raises:
            RuntimeError: If no seed state was captured.
        """
        if self._current_seed_state is None:
            raise RuntimeError("No seed state captured. Setup with a seed first.")

        set_seed(
            self._current_seed_state.seed,
            include_numpy=self._current_seed_state.numpy_state is not None,
            include_torch=self._current_seed_state.torch_state is not None,
        )
        logger.info(f"Reset seed to: {self._current_seed_state.seed}")

    @contextmanager
    def reproducible_context(
        self,
        config: FullExperimentConfig,
        seed: int | None = None,
    ) -> Generator[ConfigSnapshot, None, None]:
        """Context manager for reproducible experiment execution.

        Sets up reproducibility on entry and can reset seed on exit
        for subsequent runs.

        Args:
            config: Experiment configuration.
            seed: Optional seed override.

        Yields:
            ConfigSnapshot for the experiment.

        Example:
            >>> manager = ReproducibilityManager()
            >>> with manager.reproducible_context(config) as snapshot:
            ...     # Run experiment with reproducibility guaranteed
            ...     result = run_experiment(config)
        """
        snapshot = self.setup(config, seed)
        try:
            yield snapshot
        finally:
            # Log completion
            logger.debug("Exiting reproducible context")

    def get_reproducibility_report(self) -> dict[str, Any]:
        """Generate a report on current reproducibility state.

        Returns:
            Dictionary with reproducibility information.
        """
        report: dict[str, Any] = {
            "snapshot_dir": str(self._snapshot_dir),
            "has_current_snapshot": self._current_snapshot is not None,
            "has_seed_state": self._current_seed_state is not None,
            "saved_snapshots": self.list_snapshots(),
        }

        if self._current_snapshot:
            report["current_config_hash"] = self._current_snapshot.config_hash
            report["snapshot_timestamp"] = self._current_snapshot.timestamp.isoformat()

        if self._current_seed_state:
            report["seed"] = self._current_seed_state.seed

        return report


def verify_reproducibility(
    config: FullExperimentConfig,
    snapshot_path: Path | str,
) -> bool:
    """Verify that current configuration matches a saved snapshot.

    Args:
        config: Current experiment configuration.
        snapshot_path: Path to the snapshot file.

    Returns:
        True if configurations match, False otherwise.
    """
    snapshot_path = Path(snapshot_path)
    current_hash = hash_config(config)

    with open(snapshot_path) as f:
        data = json.load(f)

    saved_hash = data.get("config_hash", "")
    matches = current_hash == saved_hash

    if not matches:
        logger.warning(
            f"Configuration mismatch: current={current_hash[:16]}... "
            f"saved={saved_hash[:16]}..."
        )
    else:
        logger.info("Configuration verified - matches saved snapshot")

    return matches
