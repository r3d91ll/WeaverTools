"""Configuration models for Experiment Automation Framework.

This module defines Pydantic models for experiment configuration,
following patterns from TheLoom's configuration management.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class AgentType(str, Enum):
    """Supported agent types."""

    CLAUDE = "claude"
    LOCAL = "local"
    AUTOGEN = "autogen"
    CUSTOM = "custom"


class MetricCollectorType(str, Enum):
    """Supported metric collector types."""

    LATENCY = "latency"
    TOKENS = "tokens"
    CUSTOM = "custom"


class MetricStorageType(str, Enum):
    """Supported metric storage backends."""

    FILE = "file"
    MLFLOW = "mlflow"


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    YAML = "yaml"


class AggregationType(str, Enum):
    """Supported aggregation types for metrics."""

    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"
    COUNT = "count"
    SUM = "sum"


class StepConfig(BaseModel):
    """Configuration for a single step within a scenario."""

    agent: str = Field(..., description="Agent ID to execute this step")
    action: str = Field(..., description="Action to perform (e.g., 'generate', 'review')")
    prompt_template: str | None = Field(
        default=None, description="Path to prompt template file"
    )
    input_from: str | None = Field(
        default=None,
        description="Reference to previous step output (e.g., 'agent_id.output')",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Step-specific parameter overrides"
    )
    timeout_seconds: int | None = Field(
        default=None, ge=1, description="Timeout for this step in seconds"
    )


class ScenarioConfig(BaseModel):
    """Configuration for a multi-agent scenario."""

    name: str = Field(..., description="Unique scenario name")
    description: str = Field(default="", description="Human-readable description")
    steps: list[StepConfig] = Field(
        default_factory=list, description="Ordered list of steps to execute"
    )
    max_retries: int = Field(
        default=3, ge=0, description="Maximum retries for failed steps"
    )
    continue_on_failure: bool = Field(
        default=False,
        description="Continue with remaining steps if one fails",
    )


class AgentConfig(BaseModel):
    """Configuration for a single agent in the experiment."""

    id: str = Field(..., description="Unique agent identifier")
    type: AgentType = Field(..., description="Agent type (claude, local, autogen, custom)")
    role: str = Field(default="", description="Agent's role in the experiment")
    model: str | None = Field(default=None, description="Model identifier")
    endpoint: str | None = Field(
        default=None, description="API endpoint for local/custom agents"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific parameters"
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable name containing API key (never store secrets directly)",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate agent ID format."""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Agent ID must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v


class MetricCollectorConfig(BaseModel):
    """Configuration for a metric collector."""

    type: MetricCollectorType = Field(..., description="Collector type")
    granularity: str = Field(
        default="step", description="Collection granularity (step, scenario, experiment)"
    )
    per_agent: bool = Field(
        default=False, description="Collect metrics per agent"
    )
    name: str | None = Field(
        default=None, description="Custom metric name (for custom type)"
    )
    script: str | None = Field(
        default=None, description="Path to custom metric script (for custom type)"
    )


class MetricStorageConfig(BaseModel):
    """Configuration for metric storage backend."""

    type: MetricStorageType = Field(
        default=MetricStorageType.FILE, description="Storage backend type"
    )
    path: str = Field(
        default="./results", description="Path to store results (for file backend)"
    )
    tracking_uri: str | None = Field(
        default=None, description="MLflow tracking URI (for mlflow backend)"
    )
    experiment_name: str | None = Field(
        default=None, description="MLflow experiment name"
    )


class MetricsConfig(BaseModel):
    """Configuration for experiment metrics collection."""

    collectors: list[MetricCollectorConfig] = Field(
        default_factory=list, description="List of metric collectors"
    )
    storage: MetricStorageConfig = Field(
        default_factory=MetricStorageConfig, description="Metric storage configuration"
    )
    flush_interval_seconds: int = Field(
        default=30, ge=1, description="Interval between metric flushes"
    )


class OutputConfig(BaseModel):
    """Configuration for experiment output."""

    format: OutputFormat = Field(
        default=OutputFormat.JSON, description="Output format"
    )
    include_raw_responses: bool = Field(
        default=False, description="Include raw agent responses in output"
    )
    aggregations: list[AggregationType] = Field(
        default_factory=lambda: [AggregationType.MEAN, AggregationType.STD],
        description="Aggregations to compute",
    )
    output_dir: str | None = Field(
        default=None, description="Override output directory"
    )


class ExperimentConfig(BaseModel):
    """Configuration for experiment definition."""

    name: str = Field(..., description="Unique experiment name")
    description: str = Field(default="", description="Human-readable description")
    version: str = Field(default="1.0", description="Experiment version")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    save_checkpoints: bool = Field(
        default=True, description="Save checkpoints during execution"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Global experiment parameters"
    )
    max_iterations: int = Field(
        default=100, ge=1, description="Maximum iterations for the experiment"
    )
    timeout_seconds: int = Field(
        default=3600, ge=1, description="Overall experiment timeout in seconds"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for experiment organization"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate experiment name format."""
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty")
        return v.strip()


class FullExperimentConfig(BaseModel):
    """Complete experiment configuration including all sections.

    This model represents the full YAML configuration structure with
    experiment, agents, scenarios, metrics, and outputs sections.
    """

    experiment: ExperimentConfig = Field(..., description="Experiment definition")
    agents: list[AgentConfig] = Field(
        default_factory=list, description="Agent configurations"
    )
    scenarios: list[ScenarioConfig] = Field(
        default_factory=list, description="Scenario definitions"
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig, description="Metrics configuration"
    )
    outputs: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )

    @field_validator("agents")
    @classmethod
    def validate_unique_agent_ids(cls, v: list[AgentConfig]) -> list[AgentConfig]:
        """Ensure all agent IDs are unique."""
        ids = [agent.id for agent in v]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
            raise ValueError(f"Duplicate agent IDs found: {set(duplicates)}")
        return v

    @field_validator("scenarios")
    @classmethod
    def validate_unique_scenario_names(
        cls, v: list[ScenarioConfig]
    ) -> list[ScenarioConfig]:
        """Ensure all scenario names are unique."""
        names = [scenario.name for scenario in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate scenario names found: {set(duplicates)}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    file: str | None = Field(
        default=None, description="Log file path (None for stdout only)"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )


class Config(BaseSettings):
    """Main configuration for Experiment Automation Framework.

    Supports environment variable overrides with EXPERIMENT_ prefix.
    """

    experiments_dir: str = Field(
        default="~/.experiments",
        description="Directory to store experiment results",
    )
    checkpoints_dir: str = Field(
        default="~/.experiments/checkpoints",
        description="Directory to store experiment checkpoints",
    )
    default_timeout_seconds: int = Field(
        default=3600, ge=1, description="Default experiment timeout"
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # MLflow integration
    mlflow_tracking_uri: str | None = Field(
        default=None, description="MLflow tracking URI (optional)"
    )
    mlflow_experiment_prefix: str = Field(
        default="experiment_", description="Prefix for MLflow experiment names"
    )

    model_config = {
        "env_prefix": "EXPERIMENT_",
        "env_nested_delimiter": "__",
    }


def find_config_file() -> Path | None:
    """Locate an experiment configuration file by searching common locations.

    Searches these locations (in order):
    - ./experiment.yaml
    - ./experiment.yml
    - ./config/experiment.yaml
    - ~/.experiments/config.yaml

    Returns:
        Path to the first existing config file, or None if not found.
    """
    search_paths = [
        Path.cwd() / "experiment.yaml",
        Path.cwd() / "experiment.yml",
        Path.cwd() / "config" / "experiment.yaml",
        Path.home() / ".experiments" / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_config(config_path: str | Path | None = None) -> Config:
    """Load main configuration from file and environment.

    Priority (highest to lowest):
    1. Environment variables (EXPERIMENT_*)
    2. Specified config file
    3. Auto-discovered config file
    4. Default values

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Config instance with loaded settings.
    """
    config_data: dict[str, Any] = {}

    # Find config file
    if config_path is None:
        config_path = find_config_file()
    elif isinstance(config_path, str):
        config_path = Path(config_path)

    # Load from YAML if found
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                config_data = yaml_data

    # Create config with loaded data (env vars override via pydantic-settings)
    return Config(**config_data)


def expand_path(path: str) -> Path:
    """Expand a path string with user home directory expansion.

    Args:
        path: Path string, potentially starting with ~

    Returns:
        Expanded Path object.
    """
    return Path(path).expanduser()
