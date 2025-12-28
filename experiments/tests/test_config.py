"""Tests for configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AgentConfig,
    AgentType,
    AggregationType,
    Config,
    ExperimentConfig,
    FullExperimentConfig,
    LoggingConfig,
    MetricCollectorConfig,
    MetricCollectorType,
    MetricsConfig,
    MetricStorageConfig,
    MetricStorageType,
    OutputConfig,
    OutputFormat,
    ScenarioConfig,
    StepConfig,
    expand_path,
    load_config,
)


class TestEnums:
    """Test enum definitions."""

    def test_agent_type_values(self):
        assert AgentType.CLAUDE == "claude"
        assert AgentType.LOCAL == "local"
        assert AgentType.AUTOGEN == "autogen"
        assert AgentType.CUSTOM == "custom"

    def test_metric_collector_type_values(self):
        assert MetricCollectorType.LATENCY == "latency"
        assert MetricCollectorType.TOKENS == "tokens"
        assert MetricCollectorType.CUSTOM == "custom"

    def test_metric_storage_type_values(self):
        assert MetricStorageType.FILE == "file"
        assert MetricStorageType.MLFLOW == "mlflow"

    def test_output_format_values(self):
        assert OutputFormat.JSON == "json"
        assert OutputFormat.YAML == "yaml"

    def test_aggregation_type_values(self):
        assert AggregationType.MEAN == "mean"
        assert AggregationType.STD == "std"
        assert AggregationType.MIN == "min"
        assert AggregationType.MAX == "max"
        assert AggregationType.PERCENTILE_95 == "percentile_95"
        assert AggregationType.PERCENTILE_99 == "percentile_99"
        assert AggregationType.COUNT == "count"
        assert AggregationType.SUM == "sum"


class TestConfigDefaults:
    """Test default configuration values."""

    def test_step_config_defaults(self):
        config = StepConfig(agent="test-agent", action="generate")
        assert config.agent == "test-agent"
        assert config.action == "generate"
        assert config.prompt_template is None
        assert config.input_from is None
        assert config.parameters == {}
        assert config.timeout_seconds is None

    def test_scenario_config_defaults(self):
        config = ScenarioConfig(name="test-scenario")
        assert config.name == "test-scenario"
        assert config.description == ""
        assert config.steps == []
        assert config.max_retries == 3
        assert config.continue_on_failure is False

    def test_agent_config_defaults(self):
        config = AgentConfig(id="test-agent", type=AgentType.CLAUDE)
        assert config.id == "test-agent"
        assert config.type == AgentType.CLAUDE
        assert config.role == ""
        assert config.model is None
        assert config.endpoint is None
        assert config.parameters == {}
        assert config.api_key_env is None

    def test_metric_collector_config_defaults(self):
        config = MetricCollectorConfig(type=MetricCollectorType.LATENCY)
        assert config.type == MetricCollectorType.LATENCY
        assert config.granularity == "step"
        assert config.per_agent is False
        assert config.name is None
        assert config.script is None

    def test_metric_storage_config_defaults(self):
        config = MetricStorageConfig()
        assert config.type == MetricStorageType.FILE
        assert config.path == "./results"
        assert config.tracking_uri is None
        assert config.experiment_name is None

    def test_metrics_config_defaults(self):
        config = MetricsConfig()
        assert config.collectors == []
        assert config.storage.type == MetricStorageType.FILE
        assert config.flush_interval_seconds == 30

    def test_output_config_defaults(self):
        config = OutputConfig()
        assert config.format == OutputFormat.JSON
        assert config.include_raw_responses is False
        assert config.aggregations == [AggregationType.MEAN, AggregationType.STD]
        assert config.output_dir is None

    def test_experiment_config_defaults(self):
        config = ExperimentConfig(name="test-experiment")
        assert config.name == "test-experiment"
        assert config.description == ""
        assert config.version == "1.0"
        assert config.seed is None
        assert config.save_checkpoints is True
        assert config.parameters == {}
        assert config.max_iterations == 100
        assert config.timeout_seconds == 3600
        assert config.tags == []

    def test_logging_config_defaults(self):
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.file is None
        assert "%(asctime)s" in config.format

    def test_main_config_defaults(self):
        config = Config()
        assert config.experiments_dir == "~/.experiments"
        assert config.checkpoints_dir == "~/.experiments/checkpoints"
        assert config.default_timeout_seconds == 3600
        assert config.logging.level == "INFO"
        assert config.mlflow_tracking_uri is None
        assert config.mlflow_experiment_prefix == "experiment_"

    def test_full_experiment_config_defaults(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="test")
        )
        assert config.experiment.name == "test"
        assert config.agents == []
        assert config.scenarios == []
        assert config.metrics.collectors == []
        assert config.outputs.format == OutputFormat.JSON


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_from_yaml(self):
        config_data = {
            "experiments_dir": "/custom/experiments",
            "default_timeout_seconds": 7200,
            "logging": {
                "level": "DEBUG",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.experiments_dir == "/custom/experiments"
            assert config.default_timeout_seconds == 7200
            assert config.logging.level == "DEBUG"
            # Defaults should still apply
            assert config.checkpoints_dir == "~/.experiments/checkpoints"
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file_uses_defaults(self):
        config = load_config("/nonexistent/path/config.yaml")
        assert config.experiments_dir == "~/.experiments"  # Default
        assert config.default_timeout_seconds == 3600  # Default

    def test_load_empty_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_config(temp_path)
            # Should use defaults for empty file
            assert config.experiments_dir == "~/.experiments"
        finally:
            os.unlink(temp_path)

    def test_load_with_path_object(self):
        config_data = {"experiments_dir": "/path/from/pathlib"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.experiments_dir == "/path/from/pathlib"
        finally:
            os.unlink(temp_path)

    def test_load_with_none_uses_defaults(self):
        # When no config path is given and no config file exists in search paths
        config = load_config(None)
        assert config.experiments_dir == "~/.experiments"


class TestConfigValidation:
    """Test configuration validation."""

    def test_experiment_name_cannot_be_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ExperimentConfig(name="")

        with pytest.raises(ValueError, match="cannot be empty"):
            ExperimentConfig(name="   ")

    def test_experiment_name_strips_whitespace(self):
        config = ExperimentConfig(name="  test-experiment  ")
        assert config.name == "test-experiment"

    def test_agent_id_cannot_be_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            AgentConfig(id="", type=AgentType.CLAUDE)

        with pytest.raises(ValueError, match="cannot be empty"):
            AgentConfig(id="   ", type=AgentType.CLAUDE)

    def test_agent_id_must_be_alphanumeric(self):
        # Valid IDs
        AgentConfig(id="test-agent", type=AgentType.CLAUDE)
        AgentConfig(id="test_agent", type=AgentType.CLAUDE)
        AgentConfig(id="test123", type=AgentType.CLAUDE)
        AgentConfig(id="test-agent_123", type=AgentType.CLAUDE)

        # Invalid IDs
        with pytest.raises(ValueError, match="alphanumeric"):
            AgentConfig(id="test agent", type=AgentType.CLAUDE)  # space

        with pytest.raises(ValueError, match="alphanumeric"):
            AgentConfig(id="test.agent", type=AgentType.CLAUDE)  # dot

        with pytest.raises(ValueError, match="alphanumeric"):
            AgentConfig(id="test@agent", type=AgentType.CLAUDE)  # special char

    def test_max_iterations_minimum(self):
        config = ExperimentConfig(name="test", max_iterations=1)
        assert config.max_iterations == 1

        with pytest.raises(ValueError):
            ExperimentConfig(name="test", max_iterations=0)

        with pytest.raises(ValueError):
            ExperimentConfig(name="test", max_iterations=-1)

    def test_timeout_minimum(self):
        config = ExperimentConfig(name="test", timeout_seconds=1)
        assert config.timeout_seconds == 1

        with pytest.raises(ValueError):
            ExperimentConfig(name="test", timeout_seconds=0)

    def test_step_timeout_minimum(self):
        config = StepConfig(agent="test", action="run", timeout_seconds=1)
        assert config.timeout_seconds == 1

        with pytest.raises(ValueError):
            StepConfig(agent="test", action="run", timeout_seconds=0)

    def test_scenario_max_retries_minimum(self):
        config = ScenarioConfig(name="test", max_retries=0)
        assert config.max_retries == 0

        with pytest.raises(ValueError):
            ScenarioConfig(name="test", max_retries=-1)

    def test_metrics_flush_interval_minimum(self):
        config = MetricsConfig(flush_interval_seconds=1)
        assert config.flush_interval_seconds == 1

        with pytest.raises(ValueError):
            MetricsConfig(flush_interval_seconds=0)

    def test_default_timeout_minimum(self):
        config = Config(default_timeout_seconds=1)
        assert config.default_timeout_seconds == 1

        with pytest.raises(ValueError):
            Config(default_timeout_seconds=0)

    def test_unique_agent_ids(self):
        agent1 = AgentConfig(id="agent1", type=AgentType.CLAUDE)
        agent2 = AgentConfig(id="agent2", type=AgentType.LOCAL)

        # Valid - unique IDs
        FullExperimentConfig(
            experiment=ExperimentConfig(name="test"),
            agents=[agent1, agent2],
        )

        # Invalid - duplicate IDs
        agent3 = AgentConfig(id="agent1", type=AgentType.CUSTOM)
        with pytest.raises(ValueError, match="Duplicate agent IDs"):
            FullExperimentConfig(
                experiment=ExperimentConfig(name="test"),
                agents=[agent1, agent3],
            )

    def test_unique_scenario_names(self):
        scenario1 = ScenarioConfig(name="scenario1")
        scenario2 = ScenarioConfig(name="scenario2")

        # Valid - unique names
        FullExperimentConfig(
            experiment=ExperimentConfig(name="test"),
            scenarios=[scenario1, scenario2],
        )

        # Invalid - duplicate names
        scenario3 = ScenarioConfig(name="scenario1")
        with pytest.raises(ValueError, match="Duplicate scenario names"):
            FullExperimentConfig(
                experiment=ExperimentConfig(name="test"),
                scenarios=[scenario1, scenario3],
            )


class TestAgentTypeFromString:
    """Test agent type parsing from strings."""

    def test_agent_type_from_string(self):
        config = AgentConfig(id="test", type="claude")
        assert config.type == AgentType.CLAUDE

        config = AgentConfig(id="test", type="local")
        assert config.type == AgentType.LOCAL

    def test_invalid_agent_type(self):
        with pytest.raises(ValueError):
            AgentConfig(id="test", type="invalid")


class TestExpandPath:
    """Test path expansion utility."""

    def test_expand_home_directory(self):
        path = expand_path("~/.experiments")
        assert str(path).startswith(str(Path.home()))
        assert str(path).endswith(".experiments")

    def test_expand_regular_path(self):
        path = expand_path("/absolute/path")
        assert str(path) == "/absolute/path"

    def test_expand_relative_path(self):
        path = expand_path("./relative/path")
        assert str(path) == "relative/path"


class TestFullExperimentConfig:
    """Test complete experiment configuration."""

    def test_full_config_with_all_sections(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(
                name="comprehensive-test",
                description="A comprehensive test experiment",
                seed=42,
                max_iterations=50,
            ),
            agents=[
                AgentConfig(
                    id="reviewer",
                    type=AgentType.CLAUDE,
                    role="senior-reviewer",
                    model="claude-3-opus",
                    parameters={"temperature": 0.3},
                ),
                AgentConfig(
                    id="coder",
                    type=AgentType.LOCAL,
                    role="implementer",
                    endpoint="http://localhost:11434",
                ),
            ],
            scenarios=[
                ScenarioConfig(
                    name="review-flow",
                    steps=[
                        StepConfig(agent="coder", action="generate"),
                        StepConfig(
                            agent="reviewer",
                            action="review",
                            input_from="coder.output",
                        ),
                    ],
                ),
            ],
            metrics=MetricsConfig(
                collectors=[
                    MetricCollectorConfig(type=MetricCollectorType.LATENCY),
                    MetricCollectorConfig(
                        type=MetricCollectorType.TOKENS,
                        per_agent=True,
                    ),
                ],
                storage=MetricStorageConfig(
                    type=MetricStorageType.FILE,
                    path="./custom-results",
                ),
            ),
            outputs=OutputConfig(
                format=OutputFormat.YAML,
                include_raw_responses=True,
                aggregations=[
                    AggregationType.MEAN,
                    AggregationType.PERCENTILE_95,
                ],
            ),
        )

        # Verify experiment section
        assert config.experiment.name == "comprehensive-test"
        assert config.experiment.seed == 42

        # Verify agents section
        assert len(config.agents) == 2
        assert config.agents[0].id == "reviewer"
        assert config.agents[1].endpoint == "http://localhost:11434"

        # Verify scenarios section
        assert len(config.scenarios) == 1
        assert len(config.scenarios[0].steps) == 2
        assert config.scenarios[0].steps[1].input_from == "coder.output"

        # Verify metrics section
        assert len(config.metrics.collectors) == 2
        assert config.metrics.storage.path == "./custom-results"

        # Verify outputs section
        assert config.outputs.format == OutputFormat.YAML
        assert config.outputs.include_raw_responses is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_agents_list(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="test"),
            agents=[],
        )
        assert config.agents == []

    def test_empty_scenarios_list(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="test"),
            scenarios=[],
        )
        assert config.scenarios == []

    def test_empty_parameters_dict(self):
        config = ExperimentConfig(name="test", parameters={})
        assert config.parameters == {}

    def test_very_long_experiment_name(self):
        long_name = "a" * 1000
        config = ExperimentConfig(name=long_name)
        assert config.name == long_name

    def test_unicode_experiment_name(self):
        config = ExperimentConfig(name="测试实验-émoji-🚀")
        assert config.name == "测试实验-émoji-🚀"

    def test_nested_parameters(self):
        config = ExperimentConfig(
            name="test",
            parameters={
                "nested": {
                    "deep": {
                        "value": 123,
                    },
                },
                "list_param": [1, 2, 3],
            },
        )
        assert config.parameters["nested"]["deep"]["value"] == 123
        assert config.parameters["list_param"] == [1, 2, 3]

    def test_optional_seed_none(self):
        config = ExperimentConfig(name="test", seed=None)
        assert config.seed is None

    def test_seed_with_value(self):
        config = ExperimentConfig(name="test", seed=42)
        assert config.seed == 42

    def test_negative_seed(self):
        # Negative seeds should be allowed (some RNGs accept them)
        config = ExperimentConfig(name="test", seed=-1)
        assert config.seed == -1

    def test_tags_as_list(self):
        config = ExperimentConfig(
            name="test",
            tags=["ml", "experiment", "test"],
        )
        assert len(config.tags) == 3
        assert "ml" in config.tags

    def test_custom_metric_with_script(self):
        config = MetricCollectorConfig(
            type=MetricCollectorType.CUSTOM,
            name="code_quality",
            script="metrics/quality.py",
        )
        assert config.type == MetricCollectorType.CUSTOM
        assert config.name == "code_quality"
        assert config.script == "metrics/quality.py"

    def test_mlflow_storage_config(self):
        config = MetricStorageConfig(
            type=MetricStorageType.MLFLOW,
            tracking_uri="http://localhost:5000",
            experiment_name="my-experiment",
        )
        assert config.type == MetricStorageType.MLFLOW
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "my-experiment"
