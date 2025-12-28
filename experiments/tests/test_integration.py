"""Integration tests for full experiment lifecycle.

These tests validate the complete experiment workflow:
validate -> run -> results

Run with: pytest tests/test_integration.py -v

Run all tests including slow ones:
    pytest tests/test_integration.py -v -m ''

Run only fast integration tests:
    pytest tests/test_integration.py -v -m 'not slow'
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import (
    AgentConfig,
    AgentType,
    AggregationType,
    ExperimentConfig,
    FullExperimentConfig,
    MetricCollectorConfig,
    MetricCollectorType,
    MetricsConfig,
    MetricStorageConfig,
    MetricStorageType,
    OutputConfig,
    OutputFormat,
    ScenarioConfig,
    StepConfig,
)
from src.metrics import (
    FileMetricsBackend,
    MetricsCollector,
    create_collector_from_config,
    create_file_backend,
    track_duration,
    track_experiment,
)
from src.orchestrator import (
    AgentBase,
    AgentExecutionError,
    ExperimentContext,
    ExperimentStatus,
    Orchestrator,
    OrchestratorError,
    ScenarioResult,
    StepResult,
    StepStatus,
)
from src.reproducibility import (
    ReproducibilityManager,
    create_snapshot,
    hash_config,
    set_seed,
)
from src.schema import (
    ExperimentParseError,
    ExperimentValidationError,
    get_schema_info,
    load_experiment,
    validate_experiment,
    validate_yaml_string,
)
from src.storage import (
    ExperimentResult,
    ExperimentStatus as StorageExperimentStatus,
    ResultStorage,
    ScenarioResult as StorageScenarioResult,
    aggregate_metrics,
    create_storage,
    find_by_config_hash,
    find_by_tag,
    get_latest_result,
    query_experiments,
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_config():
    """Create a basic experiment configuration."""
    return FullExperimentConfig(
        experiment=ExperimentConfig(
            name="integration-test",
            description="Integration test experiment",
            seed=42,
            save_checkpoints=True,
            parameters={"temperature": 0.7, "max_tokens": 100},
            tags=["test", "integration"],
        ),
        agents=[
            AgentConfig(
                id="agent-1",
                type=AgentType.CLAUDE,
                role="generator",
                parameters={"temperature": 0.3},
            ),
            AgentConfig(
                id="agent-2",
                type=AgentType.LOCAL,
                role="processor",
            ),
        ],
        scenarios=[
            ScenarioConfig(
                name="test-scenario",
                steps=[
                    StepConfig(agent="agent-1", action="generate"),
                    StepConfig(
                        agent="agent-2",
                        action="process",
                        input_from="agent-1.output",
                    ),
                ],
                max_retries=1,
            ),
        ],
        metrics=MetricsConfig(
            collectors=[
                MetricCollectorConfig(type=MetricCollectorType.LATENCY),
                MetricCollectorConfig(type=MetricCollectorType.TOKENS),
            ],
            storage=MetricStorageConfig(type=MetricStorageType.FILE, path="./results"),
        ),
        outputs=OutputConfig(
            format=OutputFormat.JSON,
            aggregations=[AggregationType.MEAN, AggregationType.STD],
        ),
    )


@pytest.fixture
def example_yaml_config(temp_dir):
    """Create an example YAML configuration file."""
    yaml_content = """
experiment:
  name: "yaml-test-experiment"
  description: "Test experiment from YAML"
  version: "1.0"
  seed: 12345
  save_checkpoints: true
  parameters:
    temperature: 0.7
    max_iterations: 50
  tags:
    - yaml-test
    - integration

agents:
  - id: "generator"
    type: "claude"
    role: "generator"
    parameters:
      temperature: 0.5

  - id: "evaluator"
    type: "local"
    role: "evaluator"
    endpoint: "http://localhost:11434"

scenarios:
  - name: "generate-and-evaluate"
    description: "Generate content and evaluate it"
    max_retries: 2
    steps:
      - agent: "generator"
        action: "generate"
        parameters:
          prompt: "Hello world"
      - agent: "evaluator"
        action: "evaluate"
        input_from: "generator.output"

metrics:
  collectors:
    - type: "latency"
      granularity: "step"
    - type: "tokens"
      per_agent: true
  storage:
    type: "file"
    path: "./results"
  flush_interval_seconds: 30

outputs:
  format: "json"
  include_raw_responses: false
  aggregations:
    - "mean"
    - "std"
"""
    yaml_file = temp_dir / "test_config.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


class MockAgent:
    """Mock agent for integration testing."""

    def __init__(
        self,
        agent_id: str,
        return_value: Any = None,
        delay: float = 0.0,
        should_fail: bool = False,
        fail_count: int = 0,
    ):
        self._id = agent_id
        self._return_value = return_value or {"result": f"output from {agent_id}"}
        self._delay = delay
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._call_count = 0
        self._calls: list[dict[str, Any]] = []

    @property
    def id(self) -> str:
        return self._id

    async def execute(
        self,
        action: str,
        context: ExperimentContext,
        **kwargs: Any,
    ) -> Any:
        self._call_count += 1
        self._calls.append({
            "action": action,
            "kwargs": kwargs,
            "call_number": self._call_count,
        })

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._should_fail:
            if self._fail_count == 0 or self._call_count <= self._fail_count:
                raise RuntimeError(f"Mock agent {self._id} failed")

        return self._return_value


# ============================================================================
# Configuration Loading and Validation Tests
# ============================================================================


class TestConfigurationLoading:
    """Test loading and validating experiment configurations."""

    def test_load_valid_yaml_file(self, example_yaml_config):
        """Test loading a valid YAML configuration file."""
        config = load_experiment(example_yaml_config)

        assert config.experiment.name == "yaml-test-experiment"
        assert config.experiment.seed == 12345
        assert len(config.agents) == 2
        assert len(config.scenarios) == 1
        assert config.scenarios[0].name == "generate-and-evaluate"

    def test_validate_experiment_config(self, basic_config):
        """Test validating a configuration object."""
        # Convert to dict and back to simulate YAML roundtrip
        config_dict = {
            "experiment": {
                "name": basic_config.experiment.name,
                "seed": basic_config.experiment.seed,
            },
            "agents": [
                {"id": a.id, "type": a.type.value}
                for a in basic_config.agents
            ],
            "scenarios": [
                {
                    "name": s.name,
                    "steps": [
                        {"agent": step.agent, "action": step.action}
                        for step in s.steps
                    ],
                }
                for s in basic_config.scenarios
            ],
        }

        validated = validate_experiment(config_dict)
        assert validated.experiment.name == basic_config.experiment.name

    def test_validate_yaml_string(self):
        """Test validating configuration from a YAML string."""
        yaml_str = """
experiment:
  name: "string-test"
  seed: 99
agents:
  - id: "test-agent"
    type: "claude"
scenarios:
  - name: "test-scenario"
    steps:
      - agent: "test-agent"
        action: "test"
"""
        config = validate_yaml_string(yaml_str)
        assert config.experiment.name == "string-test"
        assert config.experiment.seed == 99

    def test_invalid_yaml_syntax_error(self, temp_dir):
        """Test handling of invalid YAML syntax."""
        invalid_yaml = """
experiment:
  name: "test"
  invalid_indent
    nested: value
"""
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text(invalid_yaml)

        with pytest.raises(ExperimentParseError) as exc_info:
            load_experiment(yaml_file)

        assert "YAML syntax" in str(exc_info.value) or "Line" in str(exc_info.value)

    def test_missing_required_field_error(self, temp_dir):
        """Test handling of missing required fields."""
        missing_name = """
experiment:
  description: "No name provided"
agents: []
scenarios: []
"""
        yaml_file = temp_dir / "missing.yaml"
        yaml_file.write_text(missing_name)

        with pytest.raises(ExperimentValidationError) as exc_info:
            load_experiment(yaml_file)

        assert "experiment" in str(exc_info.value).lower()

    def test_get_json_schema(self):
        """Test getting the JSON schema for documentation."""
        schema = get_schema_info()

        assert "properties" in schema
        assert "experiment" in schema["properties"]
        assert "agents" in schema["properties"]
        assert "scenarios" in schema["properties"]


# ============================================================================
# Orchestrator and Execution Tests
# ============================================================================


class TestOrchestratorExecution:
    """Test the orchestrator's execution of experiments."""

    def test_orchestrator_agent_registration(self, basic_config):
        """Test registering agents with the orchestrator."""
        orchestrator = Orchestrator(basic_config)

        agent1 = MockAgent("agent-1")
        agent2 = MockAgent("agent-2")

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        assert "agent-1" in orchestrator.agents
        assert "agent-2" in orchestrator.agents
        assert orchestrator.get_agent("agent-1") is agent1

    def test_orchestrator_duplicate_agent_error(self, basic_config):
        """Test that registering duplicate agents fails."""
        orchestrator = Orchestrator(basic_config)

        agent1 = MockAgent("agent-1")
        orchestrator.register_agent(agent1)

        with pytest.raises(OrchestratorError) as exc_info:
            orchestrator.register_agent(MockAgent("agent-1"))

        assert "already registered" in str(exc_info.value).lower()

    def test_orchestrator_missing_agent_validation(self, basic_config):
        """Test that running with missing agents fails validation."""
        orchestrator = Orchestrator(basic_config)

        # Only register one of the two required agents
        agent1 = MockAgent("agent-1")
        orchestrator.register_agent(agent1)

        with pytest.raises(OrchestratorError) as exc_info:
            asyncio.get_event_loop().run_until_complete(orchestrator.run())

        assert "missing" in str(exc_info.value).lower() or "agent-2" in str(exc_info.value)

    def test_orchestrator_run_scenario(self, basic_config, temp_dir):
        """Test running a complete scenario."""
        orchestrator = Orchestrator(basic_config, checkpoint_dir=temp_dir)

        agent1 = MockAgent("agent-1", return_value={"generated": "content"})
        agent2 = MockAgent("agent-2", return_value={"processed": "result"})

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Run the experiment
        results = asyncio.get_event_loop().run_until_complete(
            orchestrator.run(experiment_id="test-run-001")
        )

        assert len(results) == 1  # One scenario
        assert results[0].name == "test-scenario"
        assert results[0].status == StepStatus.COMPLETED
        assert len(results[0].steps) == 2

        # Verify both agents were called
        assert agent1._call_count == 1
        assert agent2._call_count == 1

    def test_orchestrator_step_output_passing(self, basic_config, temp_dir):
        """Test that outputs are passed between steps correctly."""
        orchestrator = Orchestrator(basic_config, checkpoint_dir=temp_dir)

        expected_output = {"key": "value", "nested": {"data": 123}}
        received_input = {}

        class CaptureAgent:
            def __init__(self, agent_id: str):
                self._id = agent_id

            @property
            def id(self) -> str:
                return self._id

            async def execute(self, action: str, context: ExperimentContext, **kwargs):
                nonlocal received_input
                if "input_data" in kwargs:
                    received_input = kwargs["input_data"]
                return expected_output

        orchestrator.register_agent(CaptureAgent("agent-1"))
        orchestrator.register_agent(CaptureAgent("agent-2"))

        asyncio.get_event_loop().run_until_complete(orchestrator.run())

        # Second agent should have received output from first agent
        assert received_input == expected_output

    def test_orchestrator_step_failure_handling(self, basic_config):
        """Test handling of step failures."""
        orchestrator = Orchestrator(basic_config)

        # First agent fails
        agent1 = MockAgent("agent-1", should_fail=True)
        agent2 = MockAgent("agent-2")

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())

        assert len(results) == 1
        assert results[0].status == StepStatus.FAILED
        assert results[0].error is not None

    def test_orchestrator_retry_on_failure(self):
        """Test that failed steps are retried."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="retry-test"),
            agents=[AgentConfig(id="agent-1", type=AgentType.CLAUDE)],
            scenarios=[
                ScenarioConfig(
                    name="retry-scenario",
                    steps=[StepConfig(agent="agent-1", action="test")],
                    max_retries=2,
                )
            ],
        )

        orchestrator = Orchestrator(config)

        # Agent fails first 2 times, succeeds on third
        agent = MockAgent("agent-1", should_fail=True, fail_count=2)
        orchestrator.register_agent(agent)

        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())

        assert agent._call_count == 3  # Initial + 2 retries
        assert results[0].status == StepStatus.COMPLETED

    def test_orchestrator_context_checkpoint(self, basic_config, temp_dir):
        """Test that context can be serialized for checkpointing."""
        orchestrator = Orchestrator(basic_config, checkpoint_dir=temp_dir)

        agent1 = MockAgent("agent-1")
        agent2 = MockAgent("agent-2")
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        asyncio.get_event_loop().run_until_complete(orchestrator.run())

        # Verify context can be serialized
        context = orchestrator.context
        checkpoint_data = context.to_checkpoint_data()

        assert "experiment_id" in checkpoint_data
        assert "experiment_name" in checkpoint_data
        assert "step_outputs" in checkpoint_data

        # Verify it can be restored
        restored = ExperimentContext.from_checkpoint_data(
            checkpoint_data, basic_config
        )
        assert restored.experiment_name == context.experiment_name


# ============================================================================
# Metrics Collection Tests
# ============================================================================


class TestMetricsIntegration:
    """Test metrics collection during experiment execution."""

    def test_file_backend_metrics_logging(self, temp_dir):
        """Test logging metrics to file backend."""
        backend = FileMetricsBackend(
            output_dir=temp_dir,
            experiment_name="metrics-test",
            run_id="run-001",
        )

        backend.log_metric("latency", 1.5)
        backend.log_metric("tokens", 100)
        backend.log_params({"temperature": 0.7})
        backend.flush()

        # Verify files were created
        metrics_file = temp_dir / "metrics-test" / "run-001" / "metrics.json"
        params_file = temp_dir / "metrics-test" / "run-001" / "params.json"

        assert metrics_file.exists()
        assert params_file.exists()

        metrics_data = json.loads(metrics_file.read_text())
        assert len(metrics_data) == 2
        assert any(m["name"] == "latency" for m in metrics_data)

        params_data = json.loads(params_file.read_text())
        assert params_data["temperature"] == 0.7

    def test_metrics_collector_with_backends(self, temp_dir):
        """Test MetricsCollector with multiple backends."""
        backend = create_file_backend(temp_dir, "collector-test", "run-001")
        collector = MetricsCollector(backends=[backend])

        collector.log_metric("test_metric", 42.0)
        collector.log_metrics({"metric_a": 1.0, "metric_b": 2.0})
        collector.log_params({"param1": "value1"})

        collector.close()

        # Verify metrics were logged
        assert backend.get_metrics()[-1].name in ["test_metric", "metric_a", "metric_b"]

    def test_track_duration_context_manager(self, temp_dir):
        """Test the track_duration context manager."""
        backend = create_file_backend(temp_dir, "timing-test", "run-001")
        collector = MetricsCollector(backends=[backend])

        with track_duration("operation", collector) as timing:
            time.sleep(0.1)

        assert timing.duration_seconds >= 0.1
        assert timing.duration_seconds < 0.5  # Shouldn't take too long

        collector.close()

        # Verify duration was logged
        metrics = backend.get_metrics()
        duration_metrics = [m for m in metrics if "duration" in m.name]
        assert len(duration_metrics) >= 1

    def test_track_experiment_context_manager(self, temp_dir):
        """Test the track_experiment context manager."""
        backend = create_file_backend(temp_dir, "exp-track-test", "run-001")
        collector = MetricsCollector(backends=[backend])

        with track_experiment("test-exp", collector, {"seed": 42}) as timing:
            time.sleep(0.05)

        assert timing.duration_seconds >= 0.05

        collector.close()

        # Verify experiment metrics were logged
        metrics = backend.get_metrics()
        assert any("experiment_start" in m.name for m in metrics)
        assert any("experiment_success" in m.name for m in metrics)

        params = backend.get_params()
        assert params["seed"] == 42

    def test_create_collector_from_config(self, temp_dir):
        """Test creating collector from configuration dict."""
        config = {
            "storage": {
                "type": "file",
                "path": str(temp_dir / "config-collector"),
            },
            "flush_interval_seconds": 10,
        }

        collector = create_collector_from_config(config, "config-exp", "run-001")

        collector.log_metric("test", 1.0)
        collector.close()

        assert (temp_dir / "config-collector" / "config-exp" / "run-001").exists()


# ============================================================================
# Storage and Results Tests
# ============================================================================


class TestStorageIntegration:
    """Test result storage and retrieval."""

    def test_save_and_load_result(self, temp_dir):
        """Test saving and loading experiment results."""
        storage = ResultStorage(temp_dir / "results")

        result = ExperimentResult(
            experiment_id="test-001",
            experiment_name="storage-test",
            status=StorageExperimentStatus.COMPLETED,
            seed=42,
            tags=["test"],
            metrics={"accuracy": 0.95},
        )

        saved_path = storage.save(result)
        assert saved_path.exists()

        loaded = storage.load("test-001")
        assert loaded is not None
        assert loaded.experiment_id == "test-001"
        assert loaded.experiment_name == "storage-test"
        assert loaded.metrics["accuracy"] == 0.95

    def test_list_and_filter_results(self, temp_dir):
        """Test listing and filtering results."""
        storage = ResultStorage(temp_dir / "results")

        # Save multiple results
        for i in range(5):
            status = (
                StorageExperimentStatus.COMPLETED
                if i % 2 == 0
                else StorageExperimentStatus.FAILED
            )
            result = ExperimentResult(
                experiment_id=f"exp-{i:03d}",
                experiment_name="filter-test",
                status=status,
                tags=["batch-1"] if i < 3 else ["batch-2"],
            )
            storage.save(result)

        # List all results
        all_results = storage.list_results()
        assert len(all_results) == 5

        # Filter by status
        completed = storage.list_results(status=StorageExperimentStatus.COMPLETED)
        assert len(completed) == 3

        # Filter by tags
        batch1 = storage.list_results(tags=["batch-1"])
        assert len(batch1) == 3

        # Limit results
        limited = storage.list_results(limit=2)
        assert len(limited) == 2

    def test_query_experiments(self, temp_dir):
        """Test querying experiments with predicates."""
        storage = ResultStorage(temp_dir / "results")

        for i in range(10):
            result = ExperimentResult(
                experiment_id=f"query-{i:03d}",
                experiment_name="query-test",
                status=StorageExperimentStatus.COMPLETED,
                metrics={"score": i * 10},
            )
            storage.save(result)

        # Query with custom predicate (high scores)
        high_scores = query_experiments(
            storage,
            name="query-test",
            predicate=lambda r: r.get("experiment_id", "").endswith(("7", "8", "9")),
        )
        assert len(high_scores) == 3

    def test_get_latest_result(self, temp_dir):
        """Test getting the most recent result."""
        storage = ResultStorage(temp_dir / "results")

        # Save results with slight delay to ensure ordering
        for i in range(3):
            time.sleep(0.01)
            result = ExperimentResult(
                experiment_id=f"latest-{i:03d}",
                experiment_name="latest-test",
                status=StorageExperimentStatus.COMPLETED,
            )
            storage.save(result)

        latest = get_latest_result(storage, "latest-test")
        assert latest is not None
        assert latest.experiment_id == "latest-002"

    def test_find_by_config_hash(self, temp_dir, basic_config):
        """Test finding results by configuration hash."""
        storage = ResultStorage(temp_dir / "results")
        config_hash = hash_config(basic_config.model_dump())

        # Save results with same config hash
        for i in range(3):
            result = ExperimentResult(
                experiment_id=f"hash-{i:03d}",
                experiment_name="hash-test",
                status=StorageExperimentStatus.COMPLETED,
                config_hash=config_hash,
            )
            storage.save(result)

        # Save one with different hash
        result = ExperimentResult(
            experiment_id="hash-different",
            experiment_name="hash-test",
            status=StorageExperimentStatus.COMPLETED,
            config_hash="different-hash",
        )
        storage.save(result)

        found = find_by_config_hash(storage, config_hash)
        assert len(found) == 3

    def test_aggregate_metrics_across_results(self, temp_dir):
        """Test aggregating metrics across multiple results."""
        storage = ResultStorage(temp_dir / "results")

        # Create results with varying metrics
        for i in range(5):
            result = ExperimentResult(
                experiment_id=f"agg-{i:03d}",
                experiment_name="agg-test",
                status=StorageExperimentStatus.COMPLETED,
                metrics={"accuracy": 0.9 + i * 0.01, "loss": 0.1 - i * 0.01},
            )
            storage.save(result)

        # Load results and aggregate
        entries = storage.list_results(experiment_name="agg-test")
        results = [storage.load(e["experiment_id"]) for e in entries]
        results = [r for r in results if r is not None]

        agg = aggregate_metrics(results, "accuracy")

        assert agg["count"] == 5
        assert 0.9 <= agg["mean"] <= 0.94
        assert agg["min"] == 0.9
        assert agg["max"] == 0.94

    def test_result_serialization_roundtrip(self):
        """Test that results survive JSON/YAML serialization."""
        original = ExperimentResult(
            experiment_id="serial-001",
            experiment_name="serialization-test",
            status=StorageExperimentStatus.COMPLETED,
            seed=42,
            metrics={"value": 1.5},
            parameters={"param": "value"},
            tags=["tag1", "tag2"],
        )

        # JSON roundtrip
        json_str = original.to_json()
        from_json = ExperimentResult.from_json(json_str)
        assert from_json.experiment_id == original.experiment_id
        assert from_json.metrics == original.metrics

        # YAML roundtrip
        yaml_str = original.to_yaml()
        from_yaml = ExperimentResult.from_yaml(yaml_str)
        assert from_yaml.experiment_id == original.experiment_id
        assert from_yaml.tags == original.tags


# ============================================================================
# Reproducibility Tests
# ============================================================================


class TestReproducibility:
    """Test experiment reproducibility features."""

    def test_seed_setting(self):
        """Test that setting seeds produces reproducible results."""
        import random

        # First run
        set_seed(42)
        values1 = [random.random() for _ in range(10)]

        # Second run with same seed
        set_seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    def test_config_hashing(self, basic_config):
        """Test that config hashing is deterministic."""
        config_dict = basic_config.model_dump()

        hash1 = hash_config(config_dict)
        hash2 = hash_config(config_dict)

        assert hash1 == hash2

        # Different config produces different hash
        config_dict["experiment"]["seed"] = 999
        hash3 = hash_config(config_dict)
        assert hash3 != hash1

    def test_create_snapshot(self, basic_config):
        """Test creating reproducibility snapshots."""
        snapshot = create_snapshot(
            config=basic_config.model_dump(),
            notes="Test snapshot",
        )

        assert snapshot.config is not None
        assert snapshot.config_hash is not None
        assert snapshot.environment is not None
        assert snapshot.notes == "Test snapshot"

        # Verify environment info
        assert snapshot.environment.python_version is not None
        assert snapshot.environment.platform is not None

    def test_reproducibility_manager(self, temp_dir, basic_config):
        """Test the ReproducibilityManager class."""
        manager = ReproducibilityManager(snapshots_dir=temp_dir / "snapshots")

        # Setup reproducibility
        manager.setup(seed=42, config=basic_config.model_dump())

        # Save snapshot
        snapshot_path = manager.save_snapshot(
            experiment_id="repro-001",
            notes="Initial run",
        )

        assert snapshot_path.exists()

        # Load and compare snapshot
        loaded = manager.load_snapshot("repro-001")
        assert loaded is not None
        assert loaded.config == basic_config.model_dump()

    def test_same_seed_same_results(self, basic_config, temp_dir):
        """Test that same seed produces same orchestrator results."""
        async def run_experiment(seed: int) -> list[str]:
            set_seed(seed)

            orchestrator = Orchestrator(basic_config, checkpoint_dir=temp_dir)

            # Agents with deterministic output based on seed
            import random
            random.seed(seed)

            class DeterministicAgent:
                def __init__(self, agent_id: str):
                    self._id = agent_id
                    self._random_value = random.random()

                @property
                def id(self) -> str:
                    return self._id

                async def execute(self, action: str, context, **kwargs):
                    return {"random": self._random_value}

            orchestrator.register_agent(DeterministicAgent("agent-1"))
            orchestrator.register_agent(DeterministicAgent("agent-2"))

            results = await orchestrator.run(experiment_id=f"seed-{seed}")

            return [str(s.steps[0].output) for s in results]

        # Run twice with same seed
        results1 = asyncio.get_event_loop().run_until_complete(run_experiment(42))
        results2 = asyncio.get_event_loop().run_until_complete(run_experiment(42))

        assert results1 == results2


# ============================================================================
# Full Lifecycle Integration Tests
# ============================================================================


class TestFullLifecycle:
    """Test complete experiment lifecycle: validate -> run -> results."""

    def test_complete_lifecycle_from_yaml(self, example_yaml_config, temp_dir):
        """Test the complete experiment lifecycle starting from YAML."""
        # Step 1: Load and validate configuration
        config = load_experiment(example_yaml_config)
        assert config.experiment.name == "yaml-test-experiment"

        # Step 2: Setup storage and metrics
        storage = ResultStorage(temp_dir / "results")
        metrics_backend = create_file_backend(
            temp_dir / "metrics",
            config.experiment.name,
            "run-001",
        )
        collector = MetricsCollector(backends=[metrics_backend])

        # Step 3: Create and run orchestrator
        orchestrator = Orchestrator(config, checkpoint_dir=temp_dir / "checkpoints")

        # Register mock agents
        orchestrator.register_agent(MockAgent("generator", {"content": "generated"}))
        orchestrator.register_agent(MockAgent("evaluator", {"score": 0.95}))

        # Step 4: Execute experiment with metrics tracking
        with track_experiment(config.experiment.name, collector) as timing:
            scenario_results = asyncio.get_event_loop().run_until_complete(
                orchestrator.run(experiment_id="lifecycle-001")
            )

        # Step 5: Create and save result
        result = ExperimentResult.from_config(
            config,
            experiment_id="lifecycle-001",
            config_hash=hash_config(config.model_dump()),
        )
        result.mark_running()

        for scenario in scenario_results:
            scenario_result = StorageScenarioResult(
                name=scenario.name,
                status=StorageExperimentStatus.COMPLETED,
                steps_completed=len(scenario.steps),
                steps_total=len(scenario.steps),
                duration_seconds=scenario.duration_seconds,
            )
            result.add_scenario_result(scenario_result)

        result.update_metrics({"duration": timing.duration_seconds})
        result.mark_completed()

        saved_path = storage.save(result)

        # Step 6: Verify results
        assert saved_path.exists()

        loaded = storage.load("lifecycle-001")
        assert loaded is not None
        assert loaded.is_successful
        assert len(loaded.scenarios) == 1
        assert loaded.scenarios[0].status == StorageExperimentStatus.COMPLETED

        # Verify metrics were collected
        collector.close()
        metrics = metrics_backend.get_metrics()
        assert len(metrics) > 0

    def test_lifecycle_with_failures_and_retries(self, temp_dir):
        """Test lifecycle with step failures and retries."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(
                name="failure-test",
                seed=123,
            ),
            agents=[
                AgentConfig(id="flaky-agent", type=AgentType.CLAUDE),
            ],
            scenarios=[
                ScenarioConfig(
                    name="flaky-scenario",
                    steps=[
                        StepConfig(agent="flaky-agent", action="unstable"),
                    ],
                    max_retries=3,
                ),
            ],
        )

        storage = ResultStorage(temp_dir / "results")
        orchestrator = Orchestrator(config)

        # Agent fails twice, succeeds on third attempt
        flaky = MockAgent("flaky-agent", should_fail=True, fail_count=2)
        orchestrator.register_agent(flaky)

        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())

        # Verify retries happened
        assert flaky._call_count == 3
        assert results[0].status == StepStatus.COMPLETED

        # Save and verify result
        result = ExperimentResult(
            experiment_id="failure-001",
            experiment_name="failure-test",
            status=StorageExperimentStatus.COMPLETED,
            metrics={"retries": flaky._call_count - 1},
        )
        storage.save(result)

        loaded = storage.load("failure-001")
        assert loaded.metrics["retries"] == 2

    def test_lifecycle_pause_and_resume(self, basic_config, temp_dir):
        """Test pausing and resuming an experiment."""
        orchestrator = Orchestrator(basic_config, checkpoint_dir=temp_dir)

        agent1 = MockAgent("agent-1")
        agent2 = MockAgent("agent-2")
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Run and capture context
        asyncio.get_event_loop().run_until_complete(orchestrator.run())
        context = orchestrator.context

        # Simulate pause by getting checkpoint data
        checkpoint_data = context.to_checkpoint_data()

        # Verify checkpoint can be serialized
        checkpoint_json = json.dumps(checkpoint_data)
        restored_data = json.loads(checkpoint_json)

        # Verify restoration
        restored = ExperimentContext.from_checkpoint_data(
            restored_data, basic_config
        )
        assert restored.experiment_name == context.experiment_name
        assert restored.step_outputs == context.step_outputs

    @pytest.mark.slow
    def test_lifecycle_with_timing(self, basic_config, temp_dir):
        """Test lifecycle with realistic timing delays."""
        orchestrator = Orchestrator(basic_config)

        # Agents with delays to simulate real work
        agent1 = MockAgent("agent-1", delay=0.2)
        agent2 = MockAgent("agent-2", delay=0.1)
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        start = time.perf_counter()
        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())
        duration = time.perf_counter() - start

        # Verify timing is captured
        assert duration >= 0.3  # At least sum of delays
        assert results[0].duration_seconds >= 0.3
        assert results[0].steps[0].duration_seconds >= 0.2
        assert results[0].steps[1].duration_seconds >= 0.1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_scenarios_list(self):
        """Test handling of configuration with no scenarios."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="no-scenarios"),
            agents=[],
            scenarios=[],
        )

        orchestrator = Orchestrator(config)
        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())

        assert results == []

    def test_single_step_scenario(self):
        """Test scenario with a single step."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="single-step"),
            agents=[AgentConfig(id="solo", type=AgentType.CLAUDE)],
            scenarios=[
                ScenarioConfig(
                    name="solo-scenario",
                    steps=[StepConfig(agent="solo", action="work")],
                )
            ],
        )

        orchestrator = Orchestrator(config)
        orchestrator.register_agent(MockAgent("solo"))

        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())

        assert len(results) == 1
        assert len(results[0].steps) == 1
        assert results[0].status == StepStatus.COMPLETED

    def test_multiple_scenarios(self):
        """Test running multiple scenarios in sequence."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="multi-scenario"),
            agents=[AgentConfig(id="worker", type=AgentType.CLAUDE)],
            scenarios=[
                ScenarioConfig(
                    name=f"scenario-{i}",
                    steps=[StepConfig(agent="worker", action=f"action-{i}")],
                )
                for i in range(3)
            ],
        )

        orchestrator = Orchestrator(config)
        worker = MockAgent("worker")
        orchestrator.register_agent(worker)

        results = asyncio.get_event_loop().run_until_complete(orchestrator.run())

        assert len(results) == 3
        assert worker._call_count == 3
        assert all(r.status == StepStatus.COMPLETED for r in results)

    def test_unicode_experiment_names(self, temp_dir):
        """Test handling of unicode in experiment names."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(
                name="test-\u00e9xp\u00e9rience",  # test-expérience
                description="Test with unicode: \u2764",
            ),
            agents=[],
            scenarios=[],
        )

        storage = ResultStorage(temp_dir / "results")
        result = ExperimentResult(
            experiment_id="unicode-001",
            experiment_name=config.experiment.name,
            status=StorageExperimentStatus.COMPLETED,
        )

        storage.save(result)
        loaded = storage.load("unicode-001")

        assert loaded is not None
        assert loaded.experiment_name == config.experiment.name

    def test_large_metrics_collection(self, temp_dir):
        """Test handling of many metrics."""
        backend = create_file_backend(temp_dir, "large-metrics", "run-001")
        collector = MetricsCollector(backends=[backend])

        # Log many metrics
        for i in range(1000):
            collector.log_metric(f"metric_{i}", float(i), step=i)

        collector.close()

        # Verify all were captured
        metrics = backend.get_metrics()
        assert len(metrics) == 1000

    def test_concurrent_result_storage(self, temp_dir):
        """Test concurrent access to result storage."""
        import threading

        storage = ResultStorage(temp_dir / "concurrent")
        errors = []

        def save_result(index: int):
            try:
                result = ExperimentResult(
                    experiment_id=f"concurrent-{index:04d}",
                    experiment_name="concurrent-test",
                    status=StorageExperimentStatus.COMPLETED,
                )
                storage.save(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_result, args=(i,)) for i in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert storage.get_result_count() == 20
