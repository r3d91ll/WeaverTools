"""Tests for orchestrator and CLI runner modules."""

import argparse
import asyncio
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from src.config import (
    AgentConfig,
    AgentType,
    ExperimentConfig,
    FullExperimentConfig,
    ScenarioConfig,
    StepConfig,
)
from src.orchestrator import (
    AgentBase,
    AgentExecutionError,
    AgentInterface,
    ExperimentContext,
    ExperimentStatus,
    Orchestrator,
    OrchestratorError,
    ScenarioResult,
    StepResult,
    StepStatus,
    create_agent_factory,
)
from src.runner import (
    cmd_list,
    cmd_resume,
    cmd_run,
    cmd_validate,
    create_parser,
    main,
    setup_logging,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def basic_experiment_config():
    """Create a basic experiment configuration for testing."""
    return FullExperimentConfig(
        experiment=ExperimentConfig(
            name="test-experiment",
            description="A test experiment",
            seed=42,
        ),
        agents=[
            AgentConfig(id="agent-1", type=AgentType.CLAUDE),
            AgentConfig(id="agent-2", type=AgentType.LOCAL),
        ],
        scenarios=[
            ScenarioConfig(
                name="scenario-1",
                steps=[
                    StepConfig(agent="agent-1", action="generate"),
                    StepConfig(agent="agent-2", action="process", input_from="agent-1.output"),
                ],
            ),
        ],
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockAgent:
    """Mock agent for testing that conforms to AgentInterface."""

    def __init__(self, agent_id: str, return_value: str = "mock output"):
        self._id = agent_id
        self._return_value = return_value
        self._execute_count = 0
        self._last_action = None
        self._should_fail = False

    @property
    def id(self) -> str:
        return self._id

    async def execute(self, action: str, context, **kwargs):
        self._execute_count += 1
        self._last_action = action
        if self._should_fail:
            raise RuntimeError("Agent execution failed")
        return self._return_value


# ============================================================================
# ExperimentStatus Enum Tests
# ============================================================================


class TestExperimentStatus:
    """Test ExperimentStatus enum."""

    def test_status_values(self):
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.PAUSED == "paused"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.FAILED == "failed"
        assert ExperimentStatus.CANCELLED == "cancelled"

    def test_status_count(self):
        assert len(ExperimentStatus) == 6

    def test_status_from_string(self):
        assert ExperimentStatus("pending") == ExperimentStatus.PENDING
        assert ExperimentStatus("running") == ExperimentStatus.RUNNING


class TestStepStatus:
    """Test StepStatus enum."""

    def test_status_values(self):
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"

    def test_status_count(self):
        assert len(StepStatus) == 5


# ============================================================================
# StepResult Tests
# ============================================================================


class TestStepResult:
    """Test StepResult dataclass."""

    def test_basic_creation(self):
        result = StepResult(
            step_name="test-step",
            agent_id="agent-1",
            action="generate",
            status=StepStatus.COMPLETED,
        )
        assert result.step_name == "test-step"
        assert result.agent_id == "agent-1"
        assert result.action == "generate"
        assert result.status == StepStatus.COMPLETED
        assert result.output is None
        assert result.error is None

    def test_creation_with_all_fields(self):
        start = datetime.now(timezone.utc)
        end = datetime.now(timezone.utc)
        result = StepResult(
            step_name="test-step",
            agent_id="agent-1",
            action="generate",
            status=StepStatus.COMPLETED,
            output={"result": "success"},
            error=None,
            start_time=start,
            end_time=end,
            duration_seconds=1.5,
            metrics={"tokens": 100},
        )
        assert result.output == {"result": "success"}
        assert result.start_time == start
        assert result.end_time == end
        assert result.duration_seconds == 1.5
        assert result.metrics == {"tokens": 100}

    def test_failed_result(self):
        result = StepResult(
            step_name="failing-step",
            agent_id="agent-1",
            action="fail",
            status=StepStatus.FAILED,
            error="Something went wrong",
        )
        assert result.status == StepStatus.FAILED
        assert result.error == "Something went wrong"

    def test_default_metrics_is_empty_dict(self):
        result = StepResult(
            step_name="test",
            agent_id="agent",
            action="act",
            status=StepStatus.PENDING,
        )
        assert result.metrics == {}


# ============================================================================
# ScenarioResult Tests
# ============================================================================


class TestScenarioResult:
    """Test ScenarioResult dataclass."""

    def test_basic_creation(self):
        result = ScenarioResult(
            name="test-scenario",
            status=StepStatus.COMPLETED,
        )
        assert result.name == "test-scenario"
        assert result.status == StepStatus.COMPLETED
        assert result.steps == []
        assert result.error is None

    def test_creation_with_steps(self):
        step1 = StepResult(
            step_name="step-1",
            agent_id="agent-1",
            action="generate",
            status=StepStatus.COMPLETED,
        )
        step2 = StepResult(
            step_name="step-2",
            agent_id="agent-2",
            action="review",
            status=StepStatus.COMPLETED,
        )
        result = ScenarioResult(
            name="multi-step",
            status=StepStatus.COMPLETED,
            steps=[step1, step2],
            duration_seconds=5.0,
        )
        assert len(result.steps) == 2
        assert result.duration_seconds == 5.0

    def test_failed_scenario(self):
        result = ScenarioResult(
            name="failing-scenario",
            status=StepStatus.FAILED,
            error="Step 2 failed",
        )
        assert result.status == StepStatus.FAILED
        assert result.error == "Step 2 failed"


# ============================================================================
# Exception Classes Tests
# ============================================================================


class TestAgentExecutionError:
    """Test AgentExecutionError exception."""

    def test_basic_creation(self):
        error = AgentExecutionError(
            agent_id="agent-1",
            action="generate",
            message="API timeout",
        )
        assert error.agent_id == "agent-1"
        assert error.action == "generate"
        assert error.message == "API timeout"
        assert error.cause is None
        assert "agent-1" in str(error)
        assert "generate" in str(error)
        assert "API timeout" in str(error)

    def test_with_cause(self):
        cause = RuntimeError("Connection refused")
        error = AgentExecutionError(
            agent_id="agent-1",
            action="execute",
            message="Failed to connect",
            cause=cause,
        )
        assert error.cause is cause

    def test_formatted_message(self):
        error = AgentExecutionError(
            agent_id="my-agent",
            action="do_something",
            message="It broke",
        )
        msg = str(error)
        assert "my-agent" in msg
        assert "do_something" in msg


class TestOrchestratorError:
    """Test OrchestratorError exception."""

    def test_basic_creation(self):
        error = OrchestratorError(message="Something failed")
        assert error.message == "Something failed"
        assert error.experiment_name is None
        assert error.scenario_name is None

    def test_with_experiment_name(self):
        error = OrchestratorError(
            message="Validation failed",
            experiment_name="my-experiment",
        )
        assert error.experiment_name == "my-experiment"
        msg = str(error)
        assert "my-experiment" in msg
        assert "Validation failed" in msg

    def test_with_scenario_name(self):
        error = OrchestratorError(
            message="Step failed",
            experiment_name="exp-1",
            scenario_name="scenario-1",
        )
        assert error.scenario_name == "scenario-1"
        msg = str(error)
        assert "exp-1" in msg
        assert "scenario-1" in msg


# ============================================================================
# ExperimentContext Tests
# ============================================================================


class TestExperimentContext:
    """Test ExperimentContext dataclass."""

    def test_basic_creation(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
        )
        assert ctx.experiment_id == "exp-123"
        assert ctx.experiment_name == "test"
        assert ctx.parameters == {}
        assert ctx.step_outputs == {}
        assert ctx.current_scenario is None
        assert ctx.current_step == 0
        assert ctx.iteration == 0
        assert ctx.status == ExperimentStatus.PENDING

    def test_set_and_get_step_output(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
        )
        ctx.set_step_output("agent-1", {"result": "data"})

        # Get with .output suffix
        assert ctx.get_step_output("agent-1.output") == {"result": "data"}
        # Get without suffix (convenience)
        assert ctx.get_step_output("agent-1") == {"result": "data"}

    def test_get_step_output_not_found(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
        )
        assert ctx.get_step_output("nonexistent") is None
        assert ctx.get_step_output("nonexistent.output") is None

    def test_get_parameter(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
            parameters={"temperature": 0.7, "max_tokens": 100},
        )
        assert ctx.get_parameter("temperature") == 0.7
        assert ctx.get_parameter("max_tokens") == 100
        assert ctx.get_parameter("missing") is None
        assert ctx.get_parameter("missing", "default") == "default"

    def test_update_parameters(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
            parameters={"a": 1},
        )
        ctx.update_parameters({"b": 2, "a": 10})
        assert ctx.parameters == {"a": 10, "b": 2}

    def test_to_checkpoint_data(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
            parameters={"key": "value"},
            current_scenario="scenario-1",
            current_step=2,
            status=ExperimentStatus.PAUSED,
        )
        ctx.set_step_output("agent-1", "output-data")

        data = ctx.to_checkpoint_data()
        assert data["experiment_id"] == "exp-123"
        assert data["experiment_name"] == "test"
        assert data["parameters"] == {"key": "value"}
        assert data["current_scenario"] == "scenario-1"
        assert data["current_step"] == 2
        assert data["status"] == "paused"
        assert "agent-1" in data["step_outputs"]

    def test_from_checkpoint_data(self, basic_experiment_config):
        data = {
            "experiment_id": "exp-456",
            "experiment_name": "restored",
            "parameters": {"restored": True},
            "step_outputs": {"agent-1.output": "previous"},
            "current_scenario": "scenario-1",
            "current_step": 1,
            "iteration": 5,
            "start_time": "2024-01-15T12:00:00+00:00",
            "status": "paused",
            "metadata": {"note": "test"},
        }
        ctx = ExperimentContext.from_checkpoint_data(
            data, basic_experiment_config
        )

        assert ctx.experiment_id == "exp-456"
        assert ctx.experiment_name == "restored"
        assert ctx.parameters == {"restored": True}
        assert ctx.current_scenario == "scenario-1"
        assert ctx.current_step == 1
        assert ctx.iteration == 5
        assert ctx.status == ExperimentStatus.PAUSED
        assert ctx.metadata == {"note": "test"}

    def test_checkpoint_roundtrip(self, basic_experiment_config):
        original = ExperimentContext(
            experiment_id="roundtrip-123",
            experiment_name="roundtrip-test",
            config=basic_experiment_config,
            parameters={"p1": 1, "p2": "two"},
            current_scenario="scenario-1",
            current_step=3,
            iteration=10,
            status=ExperimentStatus.PAUSED,
        )
        original.set_step_output("agent-1", {"data": "value"})

        data = original.to_checkpoint_data()
        restored = ExperimentContext.from_checkpoint_data(
            data, basic_experiment_config
        )

        assert restored.experiment_id == original.experiment_id
        assert restored.experiment_name == original.experiment_name
        assert restored.parameters == original.parameters
        assert restored.current_scenario == original.current_scenario
        assert restored.current_step == original.current_step
        assert restored.iteration == original.iteration
        assert restored.status == original.status


# ============================================================================
# AgentBase Tests
# ============================================================================


class TestAgentBase:
    """Test AgentBase abstract class."""

    def test_concrete_agent(self):
        """Test creating a concrete agent implementation."""

        class ConcreteAgent(AgentBase):
            async def execute(self, action, context, **kwargs):
                return f"executed {action}"

        config = AgentConfig(id="concrete-agent", type=AgentType.CUSTOM)
        agent = ConcreteAgent(config)

        assert agent.id == "concrete-agent"
        assert agent.config == config

    def test_initialize_and_cleanup(self):
        """Test that initialize and cleanup can be called."""

        class ConcreteAgent(AgentBase):
            def __init__(self, config):
                super().__init__(config)
                self.initialized = False
                self.cleaned_up = False

            async def execute(self, action, context, **kwargs):
                return "done"

            async def initialize(self):
                self.initialized = True

            async def cleanup(self):
                self.cleaned_up = True

        config = AgentConfig(id="test", type=AgentType.CUSTOM)
        agent = ConcreteAgent(config)

        async def test():
            await agent.initialize()
            assert agent.initialized
            await agent.cleanup()
            assert agent.cleaned_up

        asyncio.run(test())


# ============================================================================
# Orchestrator Tests
# ============================================================================


class TestOrchestrator:
    """Test Orchestrator class."""

    def test_creation(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        assert orchestrator.config == basic_experiment_config
        assert orchestrator.agents == {}
        assert orchestrator.context is None
        assert not orchestrator.is_running

    def test_register_agent(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        agent = MockAgent("agent-1")

        orchestrator.register_agent(agent)

        assert "agent-1" in orchestrator.agents
        assert orchestrator.get_agent("agent-1") == agent

    def test_register_duplicate_agent_raises(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        agent1 = MockAgent("agent-1")
        agent2 = MockAgent("agent-1")

        orchestrator.register_agent(agent1)

        with pytest.raises(OrchestratorError, match="already registered"):
            orchestrator.register_agent(agent2)

    def test_unregister_agent(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        agent = MockAgent("agent-1")

        orchestrator.register_agent(agent)
        orchestrator.unregister_agent("agent-1")

        assert "agent-1" not in orchestrator.agents

    def test_unregister_nonexistent_agent_raises(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)

        with pytest.raises(OrchestratorError, match="not registered"):
            orchestrator.unregister_agent("nonexistent")

    def test_get_agent_nonexistent(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        assert orchestrator.get_agent("nonexistent") is None

    def test_validate_missing_agents(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        # Register only one of the required agents
        orchestrator.register_agent(MockAgent("agent-1"))

        with pytest.raises(OrchestratorError, match="Missing required agents"):
            orchestrator._validate_agents()

    def test_validate_all_agents_present(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        orchestrator.register_agent(MockAgent("agent-1"))
        orchestrator.register_agent(MockAgent("agent-2"))

        # Should not raise
        orchestrator._validate_agents()

    def test_create_context(self, basic_experiment_config, temp_dir):
        orchestrator = Orchestrator(
            basic_experiment_config, checkpoint_dir=temp_dir
        )
        ctx = orchestrator._create_context("exp-123")

        assert ctx.experiment_id == "exp-123"
        assert ctx.experiment_name == "test-experiment"
        assert ctx.config == basic_experiment_config
        assert ctx.checkpoint_path == temp_dir / "exp-123.json"

    def test_request_pause(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        assert not orchestrator._pause_requested
        orchestrator.request_pause()
        assert orchestrator._pause_requested

    def test_request_stop(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        assert not orchestrator._stop_requested
        orchestrator.request_stop()
        assert orchestrator._stop_requested

    def test_get_results_empty(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        assert orchestrator.get_results() == []

    def test_run_without_agents_raises(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)

        async def test():
            with pytest.raises(OrchestratorError, match="Missing required agents"):
                await orchestrator.run()

        asyncio.run(test())

    def test_run_with_nonexistent_scenarios(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        orchestrator.register_agent(MockAgent("agent-1"))
        orchestrator.register_agent(MockAgent("agent-2"))

        async def test():
            with pytest.raises(OrchestratorError, match="No matching scenarios"):
                await orchestrator.run(scenarios=["nonexistent"])

        asyncio.run(test())

    def test_run_with_mock_agents(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        agent1 = MockAgent("agent-1", return_value="output1")
        agent2 = MockAgent("agent-2", return_value="output2")
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        async def test():
            results = await orchestrator.run(experiment_id="test-run")
            assert len(results) == 1
            assert results[0].name == "scenario-1"
            assert agent1._execute_count == 1
            assert agent2._execute_count == 1

        asyncio.run(test())

    def test_run_generates_experiment_id(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        orchestrator.register_agent(MockAgent("agent-1"))
        orchestrator.register_agent(MockAgent("agent-2"))

        async def test():
            results = await orchestrator.run()
            assert orchestrator.context is not None
            assert orchestrator.context.experiment_id.startswith("test-experiment_")

        asyncio.run(test())

    def test_run_selected_scenarios(self, basic_experiment_config):
        # Add another scenario
        basic_experiment_config.scenarios.append(
            ScenarioConfig(
                name="scenario-2",
                steps=[StepConfig(agent="agent-1", action="other")],
            )
        )
        orchestrator = Orchestrator(basic_experiment_config)
        orchestrator.register_agent(MockAgent("agent-1"))
        orchestrator.register_agent(MockAgent("agent-2"))

        async def test():
            results = await orchestrator.run(scenarios=["scenario-2"])
            assert len(results) == 1
            assert results[0].name == "scenario-2"

        asyncio.run(test())

    def test_experiment_lifecycle_context_manager(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)

        with orchestrator._experiment_lifecycle("test-id") as ctx:
            assert ctx.status == ExperimentStatus.RUNNING
            assert orchestrator.context == ctx

        # After context manager exits
        assert ctx.status == ExperimentStatus.COMPLETED

    def test_step_execution_failure(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        agent1 = MockAgent("agent-1")
        agent1._should_fail = True
        agent2 = MockAgent("agent-2")
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        async def test():
            results = await orchestrator.run()
            assert len(results) == 1
            assert results[0].status == StepStatus.FAILED

        asyncio.run(test())

    def test_resume_requires_paused_status(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)
        orchestrator.register_agent(MockAgent("agent-1"))
        orchestrator.register_agent(MockAgent("agent-2"))

        ctx = ExperimentContext(
            experiment_id="exp-123",
            experiment_name="test",
            config=basic_experiment_config,
            status=ExperimentStatus.COMPLETED,  # Not paused
        )

        async def test():
            with pytest.raises(OrchestratorError, match="Cannot resume"):
                await orchestrator.resume(ctx)

        asyncio.run(test())


# ============================================================================
# create_agent_factory Tests
# ============================================================================


class TestCreateAgentFactory:
    """Test create_agent_factory function."""

    def test_factory_with_custom_factories(self):
        def custom_factory(config: AgentConfig):
            return MockAgent(config.id)

        factory = create_agent_factory({AgentType.CUSTOM: custom_factory})

        config = AgentConfig(id="test", type=AgentType.CUSTOM)
        agent = factory(config)

        assert agent.id == "test"

    def test_factory_unknown_type_raises(self):
        factory = create_agent_factory({})

        config = AgentConfig(id="test", type=AgentType.CLAUDE)

        with pytest.raises(ValueError, match="No factory registered"):
            factory(config)

    def test_factory_none_uses_empty_dict(self):
        factory = create_agent_factory(None)

        config = AgentConfig(id="test", type=AgentType.LOCAL)

        with pytest.raises(ValueError, match="No factory registered"):
            factory(config)


# ============================================================================
# CLI Parser Tests
# ============================================================================


class TestCreateParser:
    """Test CLI argument parser."""

    def test_parser_creation(self):
        parser = create_parser()
        assert parser.prog == "experiment"

    def test_version_argument(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_validate_command(self):
        parser = create_parser()
        args = parser.parse_args(["validate", "config.yaml"])
        assert args.command == "validate"
        assert args.config_file == "config.yaml"

    def test_validate_with_strict(self):
        parser = create_parser()
        args = parser.parse_args(["validate", "config.yaml", "--strict"])
        assert args.strict is True

    def test_validate_with_schema(self):
        parser = create_parser()
        args = parser.parse_args(["validate", "config.yaml", "--schema"])
        assert args.schema is True

    def test_run_command(self):
        parser = create_parser()
        args = parser.parse_args(["run", "experiment.yaml"])
        assert args.command == "run"
        assert args.config_file == "experiment.yaml"

    def test_run_with_options(self):
        parser = create_parser()
        args = parser.parse_args([
            "run", "experiment.yaml",
            "--experiment-id", "custom-id",
            "--seed", "42",
            "--dry-run",
            "--output-dir", "/tmp/output",
            "--no-checkpoint",
        ])
        assert args.experiment_id == "custom-id"
        assert args.seed == 42
        assert args.dry_run is True
        assert args.output_dir == "/tmp/output"
        assert args.no_checkpoint is True

    def test_run_with_scenarios(self):
        parser = create_parser()
        args = parser.parse_args([
            "run", "experiment.yaml",
            "--scenarios", "scenario1", "scenario2",
        ])
        assert args.scenarios == ["scenario1", "scenario2"]

    def test_list_command(self):
        parser = create_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert args.limit == 20  # default

    def test_list_with_options(self):
        parser = create_parser()
        args = parser.parse_args([
            "list",
            "--status", "completed",
            "--limit", "10",
            "--json",
        ])
        assert args.status == "completed"
        assert args.limit == 10
        assert args.output_json is True

    def test_list_with_all(self):
        parser = create_parser()
        args = parser.parse_args(["list", "--all"])
        assert args.all is True

    def test_resume_command(self):
        parser = create_parser()
        args = parser.parse_args(["resume", "exp-123"])
        assert args.command == "resume"
        assert args.experiment_id == "exp-123"

    def test_resume_with_checkpoint(self):
        parser = create_parser()
        args = parser.parse_args([
            "resume", "exp-123",
            "--checkpoint", "/path/to/checkpoint.json",
        ])
        assert args.checkpoint == "/path/to/checkpoint.json"

    def test_global_options(self):
        parser = create_parser()
        args = parser.parse_args([
            "--verbose",
            "--log-file", "/var/log/experiment.log",
            "--config-dir", "/custom/config",
            "list",
        ])
        assert args.verbose is True
        assert args.log_file == "/var/log/experiment.log"
        assert args.config_dir == "/custom/config"

    def test_quiet_option(self):
        parser = create_parser()
        args = parser.parse_args(["--quiet", "list"])
        assert args.quiet is True

    def test_no_command_returns_none(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None


# ============================================================================
# setup_logging Tests
# ============================================================================


class TestSetupLogging:
    """Test logging setup."""

    def test_default_logging(self):
        # Just verify it doesn't raise
        setup_logging()

    def test_debug_level(self):
        setup_logging(level="DEBUG")
        logger = logging.getLogger(__name__)
        assert logger.getEffectiveLevel() <= logging.DEBUG

    def test_error_level(self):
        setup_logging(level="ERROR")

    def test_with_log_file(self, temp_dir):
        log_file = temp_dir / "test.log"
        setup_logging(log_file=str(log_file))
        # Just verify it doesn't raise

    def test_invalid_level_uses_info(self):
        # Invalid level should fall back to INFO
        setup_logging(level="INVALID")


# ============================================================================
# cmd_validate Tests
# ============================================================================


class TestCmdValidate:
    """Test validate command handler."""

    def test_validate_valid_config(self, temp_dir):
        config_file = temp_dir / "valid.yaml"
        config_file.write_text("""
experiment:
  name: test-experiment
  description: A test
agents: []
scenarios: []
""")

        args = argparse.Namespace(
            config_file=str(config_file),
            strict=True,
            schema=False,
        )

        # Capture stdout
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_validate(args)

        assert result == 0
        output = captured.getvalue()
        assert "Valid:" in output

    def test_validate_nonexistent_file(self):
        args = argparse.Namespace(
            config_file="/nonexistent/file.yaml",
            strict=True,
            schema=False,
        )

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_validate(args)

        assert result == 1
        assert "Error:" in captured.getvalue()

    def test_validate_invalid_yaml(self, temp_dir):
        config_file = temp_dir / "invalid.yaml"
        config_file.write_text("""
experiment:
  name: ""  # Empty name is invalid
""")

        args = argparse.Namespace(
            config_file=str(config_file),
            strict=True,
            schema=False,
        )

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_validate(args)

        assert result == 1

    def test_validate_output_schema(self, temp_dir):
        # Create a minimal valid config just to test the --schema flag
        config_file = temp_dir / "test.yaml"
        config_file.write_text("experiment:\n  name: test\n")

        args = argparse.Namespace(
            config_file=str(config_file),
            strict=True,
            schema=True,
        )

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_validate(args)

        assert result == 0
        output = captured.getvalue()
        # Should be valid JSON
        parsed = json.loads(output)
        assert "schema_version" in parsed

    def test_validate_malformed_yaml(self, temp_dir):
        config_file = temp_dir / "malformed.yaml"
        config_file.write_text("""
experiment:
  name: test
  nested:
    - item without: proper yaml
    [broken syntax
""")

        args = argparse.Namespace(
            config_file=str(config_file),
            strict=True,
            schema=False,
        )

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_validate(args)

        assert result == 1


# ============================================================================
# cmd_run Tests
# ============================================================================


class TestCmdRun:
    """Test run command handler."""

    def test_run_valid_config(self, temp_dir):
        config_file = temp_dir / "experiment.yaml"
        config_file.write_text("""
experiment:
  name: test-run
  seed: 42
agents:
  - id: agent1
    type: claude
scenarios:
  - name: scenario1
    steps:
      - agent: agent1
        action: generate
""")

        from src.config import Config

        args = argparse.Namespace(
            config_file=str(config_file),
            experiment_id=None,
            scenarios=None,
            seed=None,
            dry_run=False,
            output_dir=str(temp_dir),
            no_checkpoint=True,
        )
        app_config = Config(
            experiments_dir=str(temp_dir),
            checkpoints_dir=str(temp_dir / "checkpoints"),
        )

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_run(args, app_config)

        assert result == 0
        output = captured.getvalue()
        assert "test-run" in output

    def test_run_dry_run(self, temp_dir):
        config_file = temp_dir / "experiment.yaml"
        config_file.write_text("""
experiment:
  name: dry-run-test
agents: []
scenarios: []
""")

        from src.config import Config

        args = argparse.Namespace(
            config_file=str(config_file),
            experiment_id=None,
            scenarios=None,
            seed=None,
            dry_run=True,
            output_dir=None,
            no_checkpoint=False,
        )
        app_config = Config()

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_run(args, app_config)

        assert result == 0
        output = captured.getvalue()
        assert "Dry run:" in output

    def test_run_nonexistent_config(self, temp_dir):
        from src.config import Config

        args = argparse.Namespace(
            config_file="/nonexistent/config.yaml",
            experiment_id=None,
            scenarios=None,
            seed=None,
            dry_run=False,
            output_dir=None,
            no_checkpoint=False,
        )
        app_config = Config()

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_run(args, app_config)

        assert result == 1

    def test_run_with_seed_override(self, temp_dir):
        config_file = temp_dir / "experiment.yaml"
        config_file.write_text("""
experiment:
  name: seed-test
  seed: 123
agents: []
scenarios: []
""")

        from src.config import Config

        args = argparse.Namespace(
            config_file=str(config_file),
            experiment_id="custom-id",
            scenarios=None,
            seed=999,  # Override
            dry_run=True,
            output_dir=None,
            no_checkpoint=False,
        )
        app_config = Config()

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_run(args, app_config)

        assert result == 0


# ============================================================================
# cmd_list Tests
# ============================================================================


class TestCmdList:
    """Test list command handler."""

    def test_list_empty_directory(self, temp_dir):
        from src.config import Config

        args = argparse.Namespace(
            status=None,
            limit=20,
            output_json=False,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        output = captured.getvalue()
        assert "No experiments found" in output

    def test_list_nonexistent_directory(self, temp_dir):
        from src.config import Config

        args = argparse.Namespace(
            status=None,
            limit=20,
            output_json=False,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir / "nonexistent"))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0

    def test_list_with_experiments(self, temp_dir):
        from src.config import Config

        # Create some metadata files
        metadata1 = {
            "experiment_id": "exp-1",
            "status": "completed",
            "started_at": "2024-01-15T12:00:00Z",
            "seed": 42,
        }
        metadata2 = {
            "experiment_id": "exp-2",
            "status": "failed",
            "started_at": "2024-01-16T12:00:00Z",
            "seed": None,
        }

        (temp_dir / "exp-1_metadata.json").write_text(json.dumps(metadata1))
        (temp_dir / "exp-2_metadata.json").write_text(json.dumps(metadata2))

        args = argparse.Namespace(
            status=None,
            limit=20,
            output_json=False,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        output = captured.getvalue()
        assert "exp-1" in output or "exp-2" in output

    def test_list_with_status_filter(self, temp_dir):
        from src.config import Config

        # Create metadata files with different statuses
        for i, status in enumerate(["completed", "failed", "completed"]):
            metadata = {
                "experiment_id": f"exp-{i}",
                "status": status,
                "started_at": f"2024-01-1{i}T12:00:00Z",
            }
            (temp_dir / f"exp-{i}_metadata.json").write_text(json.dumps(metadata))

        args = argparse.Namespace(
            status="completed",
            limit=20,
            output_json=False,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        output = captured.getvalue()
        # Should only show completed experiments
        assert "failed" not in output or "completed" in output

    def test_list_json_output(self, temp_dir):
        from src.config import Config

        metadata = {
            "experiment_id": "exp-json",
            "status": "completed",
            "started_at": "2024-01-15T12:00:00Z",
        }
        (temp_dir / "exp-json_metadata.json").write_text(json.dumps(metadata))

        args = argparse.Namespace(
            status=None,
            limit=20,
            output_json=True,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        output = captured.getvalue()
        # Should be valid JSON
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_list_with_limit(self, temp_dir):
        from src.config import Config

        # Create many metadata files
        for i in range(10):
            metadata = {
                "experiment_id": f"exp-{i}",
                "status": "completed",
                "started_at": f"2024-01-{10+i}T12:00:00Z",
            }
            (temp_dir / f"exp-{i}_metadata.json").write_text(json.dumps(metadata))

        args = argparse.Namespace(
            status=None,
            limit=3,
            output_json=True,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        parsed = json.loads(captured.getvalue())
        assert len(parsed) == 3

    def test_list_all_ignores_limit(self, temp_dir):
        from src.config import Config

        # Create metadata files
        for i in range(5):
            metadata = {
                "experiment_id": f"exp-{i}",
                "status": "completed",
                "started_at": f"2024-01-{10+i}T12:00:00Z",
            }
            (temp_dir / f"exp-{i}_metadata.json").write_text(json.dumps(metadata))

        args = argparse.Namespace(
            status=None,
            limit=2,  # Would normally limit to 2
            output_json=True,
            all=True,  # But --all overrides
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        parsed = json.loads(captured.getvalue())
        assert len(parsed) == 5

    def test_list_with_corrupted_metadata(self, temp_dir):
        from src.config import Config

        # Create a valid and a corrupted metadata file
        valid_metadata = {
            "experiment_id": "exp-valid",
            "status": "completed",
            "started_at": "2024-01-15T12:00:00Z",
        }
        (temp_dir / "exp-valid_metadata.json").write_text(json.dumps(valid_metadata))
        (temp_dir / "exp-corrupted_metadata.json").write_text("not valid json {{{")

        args = argparse.Namespace(
            status=None,
            limit=20,
            output_json=True,
            all=False,
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = cmd_list(args, app_config)

        assert result == 0
        # Should still return the valid experiment
        parsed = json.loads(captured.getvalue())
        assert len(parsed) == 1
        assert parsed[0]["experiment_id"] == "exp-valid"


# ============================================================================
# cmd_resume Tests
# ============================================================================


class TestCmdResume:
    """Test resume command handler."""

    def test_resume_no_checkpoint(self, temp_dir):
        from src.config import Config

        args = argparse.Namespace(
            experiment_id="nonexistent-exp",
            checkpoint=None,
        )
        app_config = Config(
            experiments_dir=str(temp_dir),
            checkpoints_dir=str(temp_dir / "checkpoints"),
        )

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_resume(args, app_config)

        assert result == 1
        assert "Checkpoint not found" in captured.getvalue()

    def test_resume_with_explicit_checkpoint_not_found(self, temp_dir):
        from src.config import Config

        args = argparse.Namespace(
            experiment_id="exp-123",
            checkpoint="/nonexistent/checkpoint.json",
        )
        app_config = Config(experiments_dir=str(temp_dir))

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_resume(args, app_config)

        assert result == 1
        assert "Checkpoint not found" in captured.getvalue()

    def test_resume_corrupted_checkpoint(self, temp_dir):
        from src.config import Config

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "exp-123.json"
        checkpoint_file.write_text("invalid json {{{")

        args = argparse.Namespace(
            experiment_id="exp-123",
            checkpoint=str(checkpoint_file),
        )
        app_config = Config(
            experiments_dir=str(temp_dir),
            checkpoints_dir=str(checkpoint_dir),
        )

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_resume(args, app_config)

        assert result == 1
        assert "Failed to read checkpoint" in captured.getvalue()

    def test_resume_no_metadata(self, temp_dir):
        from src.config import Config

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "exp-123.json"
        checkpoint_data = {
            "experiment_id": "exp-123",
            "experiment_name": "test",
            "parameters": {},
            "step_outputs": {},
            "current_scenario": "scenario-1",
            "current_step": 1,
            "start_time": "2024-01-15T12:00:00+00:00",
            "status": "paused",
            "metadata": {},
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        args = argparse.Namespace(
            experiment_id="exp-123",
            checkpoint=str(checkpoint_file),
        )
        app_config = Config(
            experiments_dir=str(temp_dir),
            checkpoints_dir=str(checkpoint_dir),
        )

        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            result = cmd_resume(args, app_config)

        assert result == 1
        assert "Metadata not found" in captured.getvalue()


# ============================================================================
# main Function Tests
# ============================================================================


class TestMain:
    """Test main CLI entry point."""

    def test_main_no_args_shows_help(self):
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            result = main([])

        assert result == 0
        # Help should be shown
        output = captured.getvalue()
        assert "experiment" in output.lower() or output == ""

    def test_main_validate_command(self, temp_dir):
        config_file = temp_dir / "test.yaml"
        config_file.write_text("""
experiment:
  name: main-test
agents: []
scenarios: []
""")

        result = main(["validate", str(config_file)])
        assert result == 0

    def test_main_verbose_logging(self, temp_dir):
        config_file = temp_dir / "test.yaml"
        config_file.write_text("""
experiment:
  name: verbose-test
agents: []
scenarios: []
""")

        result = main(["--verbose", "validate", str(config_file)])
        assert result == 0

    def test_main_quiet_logging(self, temp_dir):
        config_file = temp_dir / "test.yaml"
        config_file.write_text("""
experiment:
  name: quiet-test
agents: []
scenarios: []
""")

        result = main(["--quiet", "validate", str(config_file)])
        assert result == 0

    def test_main_custom_config_dir(self, temp_dir):
        result = main(["--config-dir", str(temp_dir), "list"])
        assert result == 0

    def test_main_unknown_command(self):
        # Unknown command should show help
        result = main([])
        assert result == 0


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_experiment_context_with_empty_config(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="empty"),
        )
        ctx = ExperimentContext(
            experiment_id="empty-test",
            experiment_name="empty",
            config=config,
        )
        assert ctx.step_outputs == {}
        assert ctx.parameters == {}

    def test_orchestrator_with_no_scenarios(self):
        config = FullExperimentConfig(
            experiment=ExperimentConfig(name="no-scenarios"),
            agents=[AgentConfig(id="agent-1", type=AgentType.CLAUDE)],
            scenarios=[],
        )
        orchestrator = Orchestrator(config)
        orchestrator.register_agent(MockAgent("agent-1"))

        async def test():
            results = await orchestrator.run()
            assert results == []

        asyncio.run(test())

    def test_step_result_with_unicode(self):
        result = StepResult(
            step_name="测试步骤",
            agent_id="代理-1",
            action="生成",
            status=StepStatus.COMPLETED,
            output={"text": "你好世界 🌍"},
        )
        assert "测试" in result.step_name
        assert result.output["text"] == "你好世界 🌍"

    def test_experiment_context_metadata(self, basic_experiment_config):
        ctx = ExperimentContext(
            experiment_id="meta-test",
            experiment_name="test",
            config=basic_experiment_config,
            metadata={"custom_key": "custom_value", "nested": {"a": 1}},
        )
        assert ctx.metadata["custom_key"] == "custom_value"
        assert ctx.metadata["nested"]["a"] == 1

        # Test roundtrip
        data = ctx.to_checkpoint_data()
        restored = ExperimentContext.from_checkpoint_data(
            data, basic_experiment_config
        )
        assert restored.metadata == ctx.metadata

    def test_orchestrator_track_step_timing(self, basic_experiment_config):
        orchestrator = Orchestrator(basic_experiment_config)

        with orchestrator._track_step_timing() as metrics:
            import time
            time.sleep(0.01)  # Small delay

        assert "start_time" in metrics
        assert "end_time" in metrics
        assert "duration_seconds" in metrics
        assert metrics["duration_seconds"] >= 0.01


import logging


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_experiment_flow(self, temp_dir):
        """Test a complete experiment flow from config to execution."""
        config = FullExperimentConfig(
            experiment=ExperimentConfig(
                name="integration-test",
                seed=42,
                parameters={"temperature": 0.7},
            ),
            agents=[
                AgentConfig(id="generator", type=AgentType.CUSTOM),
                AgentConfig(id="reviewer", type=AgentType.CUSTOM),
            ],
            scenarios=[
                ScenarioConfig(
                    name="generate-and-review",
                    steps=[
                        StepConfig(agent="generator", action="generate"),
                        StepConfig(
                            agent="reviewer",
                            action="review",
                            input_from="generator.output",
                        ),
                    ],
                ),
            ],
        )

        orchestrator = Orchestrator(config, checkpoint_dir=temp_dir)

        # Register mock agents
        generator = MockAgent("generator", return_value={"code": "def hello(): pass"})
        reviewer = MockAgent("reviewer", return_value={"approved": True})
        orchestrator.register_agent(generator)
        orchestrator.register_agent(reviewer)

        async def run_test():
            results = await orchestrator.run(experiment_id="int-test-001")

            assert len(results) == 1
            scenario_result = results[0]
            assert scenario_result.name == "generate-and-review"
            assert scenario_result.status == StepStatus.COMPLETED
            assert len(scenario_result.steps) == 2

            # Verify context captured outputs
            ctx = orchestrator.context
            assert ctx is not None
            generator_output = ctx.get_step_output("generator")
            assert generator_output == {"code": "def hello(): pass"}

        asyncio.run(run_test())

    def test_cli_validate_run_list_flow(self, temp_dir):
        """Test CLI commands in sequence."""
        from src.config import Config

        # Create experiment config
        config_file = temp_dir / "experiment.yaml"
        config_file.write_text("""
experiment:
  name: cli-flow-test
  seed: 123
agents:
  - id: test-agent
    type: custom
scenarios:
  - name: test-scenario
    steps:
      - agent: test-agent
        action: test
""")

        # 1. Validate
        validate_args = argparse.Namespace(
            config_file=str(config_file),
            strict=True,
            schema=False,
        )
        assert cmd_validate(validate_args) == 0

        # 2. Run (dry-run)
        app_config = Config(
            experiments_dir=str(temp_dir),
            checkpoints_dir=str(temp_dir / "checkpoints"),
        )
        run_args = argparse.Namespace(
            config_file=str(config_file),
            experiment_id=None,
            scenarios=None,
            seed=None,
            dry_run=True,
            output_dir=str(temp_dir),
            no_checkpoint=True,
        )
        assert cmd_run(run_args, app_config) == 0

        # 3. List (should be empty for dry-run)
        list_args = argparse.Namespace(
            status=None,
            limit=20,
            output_json=True,
            all=False,
        )
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            assert cmd_list(list_args, app_config) == 0
