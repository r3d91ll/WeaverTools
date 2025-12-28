"""Multi-agent scenario orchestration for experiment execution.

This module provides the core orchestration logic for executing multi-agent
experiments, including agent management, step execution, and context handling.
"""

from __future__ import annotations

import logging
import signal
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Protocol

from .config import (
    AgentConfig,
    AgentType,
    FullExperimentConfig,
    ScenarioConfig,
    StepConfig,
)

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of an experiment execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of a step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result from executing a single step.

    Attributes:
        step_name: Name/identifier of the step.
        agent_id: ID of the agent that executed the step.
        action: Action that was performed.
        status: Execution status.
        output: Output from the step execution.
        error: Error message if step failed.
        start_time: When step execution started.
        end_time: When step execution ended.
        duration_seconds: Execution duration in seconds.
        metrics: Additional metrics collected during execution.
    """

    step_name: str
    agent_id: str
    action: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Result from executing a complete scenario.

    Attributes:
        name: Scenario name.
        status: Overall execution status.
        steps: List of step results.
        start_time: When scenario execution started.
        end_time: When scenario execution ended.
        duration_seconds: Total execution duration.
        error: Error message if scenario failed.
    """

    name: str
    status: StepStatus
    steps: list[StepResult] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    error: str | None = None


class AgentInterface(Protocol):
    """Protocol defining the interface for agent implementations.

    All agent implementations must conform to this interface to be used
    with the orchestrator. This uses Python's Protocol for structural
    subtyping, allowing duck-typed agent implementations.

    Example:
        >>> class MyAgent:
        ...     @property
        ...     def id(self) -> str:
        ...         return "my-agent"
        ...
        ...     async def execute(
        ...         self,
        ...         action: str,
        ...         context: ExperimentContext,
        ...         **kwargs: Any
        ...     ) -> Any:
        ...         return {"result": "success"}
    """

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        ...

    async def execute(
        self,
        action: str,
        context: "ExperimentContext",
        **kwargs: Any,
    ) -> Any:
        """Execute an action with the given context.

        Args:
            action: The action to perform (e.g., 'generate', 'review').
            context: Current experiment execution context.
            **kwargs: Additional action-specific parameters.

        Returns:
            Action result (type depends on the action).

        Raises:
            AgentExecutionError: If the action fails.
        """
        ...


class AgentBase(ABC):
    """Abstract base class for agent implementations.

    This provides a concrete base class alternative to the AgentInterface
    protocol for agents that prefer inheritance over duck typing.

    Attributes:
        config: Agent configuration from experiment definition.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent with configuration.

        Args:
            config: Agent configuration from experiment definition.
        """
        self._config = config

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        return self._config.id

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    @abstractmethod
    async def execute(
        self,
        action: str,
        context: "ExperimentContext",
        **kwargs: Any,
    ) -> Any:
        """Execute an action with the given context.

        Args:
            action: The action to perform.
            context: Current experiment execution context.
            **kwargs: Additional action-specific parameters.

        Returns:
            Action result.
        """
        pass

    async def initialize(self) -> None:
        """Initialize the agent before execution.

        Override this method to perform any setup required before
        the agent can execute actions (e.g., establishing connections).
        """
        pass

    async def cleanup(self) -> None:
        """Clean up agent resources after execution.

        Override this method to release any resources held by the agent.
        """
        pass


class AgentExecutionError(Exception):
    """Exception raised when an agent fails to execute an action.

    Attributes:
        agent_id: ID of the agent that failed.
        action: The action that failed.
        message: Human-readable error description.
        cause: Original exception that caused the failure.
    """

    def __init__(
        self,
        agent_id: str,
        action: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.action = action
        self.message = message
        self.cause = cause
        super().__init__(f"Agent '{agent_id}' failed to execute '{action}': {message}")


class OrchestratorError(Exception):
    """Exception raised when orchestration fails.

    Attributes:
        message: Human-readable error description.
        experiment_name: Name of the experiment that failed.
        scenario_name: Name of the scenario that failed (if applicable).
    """

    def __init__(
        self,
        message: str,
        experiment_name: str | None = None,
        scenario_name: str | None = None,
    ) -> None:
        self.message = message
        self.experiment_name = experiment_name
        self.scenario_name = scenario_name
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = []
        if self.experiment_name:
            parts.append(f"Experiment: {self.experiment_name}")
        if self.scenario_name:
            parts.append(f"Scenario: {self.scenario_name}")
        parts.append(self.message)
        return " | ".join(parts)


@dataclass
class ExperimentContext:
    """Context for experiment execution, holding state across steps.

    The context maintains the current state of an experiment execution,
    including outputs from previous steps, global parameters, and
    experiment metadata. It is passed to each agent during execution.

    Attributes:
        experiment_id: Unique identifier for this experiment run.
        experiment_name: Name from the experiment configuration.
        config: Full experiment configuration.
        parameters: Merged global and step-specific parameters.
        step_outputs: Outputs from completed steps, keyed by 'agent_id.output'.
        current_scenario: Currently executing scenario name.
        current_step: Currently executing step index.
        iteration: Current iteration number (for iterative experiments).
        start_time: When the experiment started.
        status: Current experiment status.
        metadata: Additional metadata for the experiment.
        checkpoint_path: Path to save checkpoints.
    """

    experiment_id: str
    experiment_name: str
    config: FullExperimentConfig
    parameters: dict[str, Any] = field(default_factory=dict)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    current_scenario: str | None = None
    current_step: int = 0
    iteration: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ExperimentStatus = ExperimentStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None

    def get_step_output(self, reference: str) -> Any:
        """Get output from a previous step by reference.

        Args:
            reference: Reference string in format 'agent_id.output' or 'agent_id'.

        Returns:
            The step output, or None if not found.

        Example:
            >>> ctx.get_step_output("junior.output")
            {"code": "def foo(): pass"}
        """
        if reference in self.step_outputs:
            return self.step_outputs[reference]

        # Try adding .output suffix if not present
        if not reference.endswith(".output"):
            output_key = f"{reference}.output"
            if output_key in self.step_outputs:
                return self.step_outputs[output_key]

        return None

    def set_step_output(self, agent_id: str, output: Any) -> None:
        """Store output from a step execution.

        Args:
            agent_id: ID of the agent that produced the output.
            output: The output to store.
        """
        self.step_outputs[f"{agent_id}.output"] = output
        # Also store with just the agent ID for convenience
        self.step_outputs[agent_id] = output

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value with fallback to default.

        Args:
            name: Parameter name.
            default: Default value if parameter not found.

        Returns:
            Parameter value or default.
        """
        return self.parameters.get(name, default)

    def update_parameters(self, params: dict[str, Any]) -> None:
        """Update context parameters with new values.

        Args:
            params: New parameter values to merge.
        """
        self.parameters.update(params)

    def to_checkpoint_data(self) -> dict[str, Any]:
        """Convert context to checkpoint-serializable format.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "step_outputs": self.step_outputs,
            "current_scenario": self.current_scenario,
            "current_step": self.current_step,
            "iteration": self.iteration,
            "start_time": self.start_time.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_checkpoint_data(
        cls,
        data: dict[str, Any],
        config: FullExperimentConfig,
        checkpoint_path: Path | None = None,
    ) -> "ExperimentContext":
        """Restore context from checkpoint data.

        Args:
            data: Checkpoint data dictionary.
            config: Experiment configuration.
            checkpoint_path: Path for future checkpoints.

        Returns:
            Restored ExperimentContext instance.
        """
        return cls(
            experiment_id=data["experiment_id"],
            experiment_name=data["experiment_name"],
            config=config,
            parameters=data.get("parameters", {}),
            step_outputs=data.get("step_outputs", {}),
            current_scenario=data.get("current_scenario"),
            current_step=data.get("current_step", 0),
            iteration=data.get("iteration", 0),
            start_time=datetime.fromisoformat(data["start_time"]),
            status=ExperimentStatus(data.get("status", "pending")),
            metadata=data.get("metadata", {}),
            checkpoint_path=checkpoint_path,
        )


class Orchestrator:
    """Orchestrates multi-agent scenario execution.

    The Orchestrator manages the execution of experiments, coordinating
    agents, executing steps in sequence, and handling experiment lifecycle
    events (pause, resume, stop).

    Attributes:
        config: Full experiment configuration.
        agents: Dictionary of initialized agents keyed by ID.

    Example:
        >>> config = load_experiment("experiment.yaml")
        >>> orchestrator = Orchestrator(config)
        >>> await orchestrator.register_agent(MyAgent(config.agents[0]))
        >>> result = await orchestrator.run()
    """

    def __init__(
        self,
        config: FullExperimentConfig,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Full experiment configuration.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self._config = config
        self._agents: dict[str, AgentInterface] = {}
        self._context: ExperimentContext | None = None
        self._checkpoint_dir = checkpoint_dir
        self._pause_requested = False
        self._stop_requested = False
        self._scenario_results: list[ScenarioResult] = []

        # Set up signal handlers for graceful shutdown
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

    @property
    def config(self) -> FullExperimentConfig:
        """Experiment configuration."""
        return self._config

    @property
    def agents(self) -> dict[str, AgentInterface]:
        """Registered agents by ID."""
        return self._agents

    @property
    def context(self) -> ExperimentContext | None:
        """Current experiment context, if running."""
        return self._context

    @property
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return (
            self._context is not None
            and self._context.status == ExperimentStatus.RUNNING
        )

    def register_agent(self, agent: AgentInterface) -> None:
        """Register an agent for use in the experiment.

        Args:
            agent: Agent instance conforming to AgentInterface.

        Raises:
            OrchestratorError: If an agent with the same ID is already registered.
        """
        if agent.id in self._agents:
            raise OrchestratorError(
                f"Agent with ID '{agent.id}' is already registered",
                experiment_name=self._config.experiment.name,
            )
        self._agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.

        Args:
            agent_id: ID of the agent to unregister.

        Raises:
            OrchestratorError: If the agent is not registered.
        """
        if agent_id not in self._agents:
            raise OrchestratorError(
                f"Agent with ID '{agent_id}' is not registered",
                experiment_name=self._config.experiment.name,
            )
        del self._agents[agent_id]
        logger.info(f"Unregistered agent: {agent_id}")

    def get_agent(self, agent_id: str) -> AgentInterface | None:
        """Get a registered agent by ID.

        Args:
            agent_id: Agent identifier.

        Returns:
            Agent instance or None if not found.
        """
        return self._agents.get(agent_id)

    def _validate_agents(self) -> None:
        """Validate that all required agents are registered.

        Raises:
            OrchestratorError: If required agents are missing.
        """
        required_agents = set()
        for scenario in self._config.scenarios:
            for step in scenario.steps:
                required_agents.add(step.agent)

        missing = required_agents - set(self._agents.keys())
        if missing:
            raise OrchestratorError(
                f"Missing required agents: {', '.join(sorted(missing))}",
                experiment_name=self._config.experiment.name,
            )

    def _create_context(self, experiment_id: str) -> ExperimentContext:
        """Create a new experiment context.

        Args:
            experiment_id: Unique identifier for this run.

        Returns:
            Initialized ExperimentContext.
        """
        checkpoint_path = None
        if self._checkpoint_dir:
            checkpoint_path = self._checkpoint_dir / f"{experiment_id}.json"

        return ExperimentContext(
            experiment_id=experiment_id,
            experiment_name=self._config.experiment.name,
            config=self._config,
            parameters=dict(self._config.experiment.parameters),
            checkpoint_path=checkpoint_path,
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def handle_interrupt(signum: int, frame: Any) -> None:
            logger.info("Received interrupt signal, requesting graceful pause...")
            self._pause_requested = True

        def handle_terminate(signum: int, frame: Any) -> None:
            logger.info("Received terminate signal, requesting stop...")
            self._stop_requested = True

        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_terminate)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

    @contextmanager
    def _experiment_lifecycle(
        self, experiment_id: str
    ) -> Generator[ExperimentContext, None, None]:
        """Context manager for experiment lifecycle.

        Handles setup and teardown of experiment execution, including
        signal handler management and status updates.

        Args:
            experiment_id: Unique identifier for this run.

        Yields:
            ExperimentContext for the execution.
        """
        self._setup_signal_handlers()
        self._context = self._create_context(experiment_id)
        self._context.status = ExperimentStatus.RUNNING
        self._pause_requested = False
        self._stop_requested = False
        self._scenario_results = []

        logger.info(
            f"Starting experiment: {self._config.experiment.name} "
            f"(ID: {experiment_id})"
        )

        try:
            yield self._context
        except Exception as e:
            self._context.status = ExperimentStatus.FAILED
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            self._restore_signal_handlers()
            if self._context.status == ExperimentStatus.RUNNING:
                if self._pause_requested:
                    self._context.status = ExperimentStatus.PAUSED
                elif self._stop_requested:
                    self._context.status = ExperimentStatus.CANCELLED
                else:
                    self._context.status = ExperimentStatus.COMPLETED
            logger.info(
                f"Experiment finished with status: {self._context.status.value}"
            )

    @contextmanager
    def _track_step_timing(self) -> Generator[dict[str, Any], None, None]:
        """Context manager to track step execution timing.

        Yields:
            Dictionary that will be populated with timing info.
        """
        metrics: dict[str, Any] = {
            "start_time": datetime.now(timezone.utc),
        }
        start = time.perf_counter()
        try:
            yield metrics
        finally:
            metrics["end_time"] = datetime.now(timezone.utc)
            metrics["duration_seconds"] = time.perf_counter() - start

    async def _execute_step(
        self,
        step: StepConfig,
        scenario: ScenarioConfig,
        step_index: int,
    ) -> StepResult:
        """Execute a single step within a scenario.

        Args:
            step: Step configuration.
            scenario: Parent scenario configuration.
            step_index: Index of the step within the scenario.

        Returns:
            StepResult with execution details.
        """
        step_name = f"{scenario.name}[{step_index}]:{step.agent}.{step.action}"
        logger.debug(f"Executing step: {step_name}")

        agent = self._agents.get(step.agent)
        if not agent:
            return StepResult(
                step_name=step_name,
                agent_id=step.agent,
                action=step.action,
                status=StepStatus.FAILED,
                error=f"Agent '{step.agent}' not found",
            )

        with self._track_step_timing() as timing:
            try:
                # Prepare input from previous step if specified
                input_data = None
                if step.input_from and self._context:
                    input_data = self._context.get_step_output(step.input_from)

                # Merge parameters (global < agent < step)
                merged_params = dict(self._config.experiment.parameters)
                agent_config = next(
                    (a for a in self._config.agents if a.id == step.agent), None
                )
                if agent_config:
                    merged_params.update(agent_config.parameters)
                merged_params.update(step.parameters)

                # Execute the step
                output = await agent.execute(
                    action=step.action,
                    context=self._context,
                    input_data=input_data,
                    prompt_template=step.prompt_template,
                    **merged_params,
                )

                # Store output in context
                if self._context:
                    self._context.set_step_output(step.agent, output)

                result = StepResult(
                    step_name=step_name,
                    agent_id=step.agent,
                    action=step.action,
                    status=StepStatus.COMPLETED,
                    output=output,
                    start_time=timing["start_time"],
                    end_time=timing.get("end_time"),
                    duration_seconds=timing.get("duration_seconds", 0.0),
                )
                logger.debug(f"Step completed: {step_name}")
                return result

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Step failed: {step_name} - {error_msg}")
                return StepResult(
                    step_name=step_name,
                    agent_id=step.agent,
                    action=step.action,
                    status=StepStatus.FAILED,
                    error=error_msg,
                    start_time=timing["start_time"],
                    end_time=timing.get("end_time"),
                    duration_seconds=timing.get("duration_seconds", 0.0),
                )

    async def _execute_scenario(
        self,
        scenario: ScenarioConfig,
    ) -> ScenarioResult:
        """Execute all steps in a scenario.

        Args:
            scenario: Scenario configuration.

        Returns:
            ScenarioResult with execution details.
        """
        logger.info(f"Executing scenario: {scenario.name}")
        if self._context:
            self._context.current_scenario = scenario.name
            self._context.current_step = 0

        start_time = datetime.now(timezone.utc)
        step_results: list[StepResult] = []
        overall_status = StepStatus.COMPLETED
        error_msg = None

        for step_index, step in enumerate(scenario.steps):
            # Check for pause/stop requests
            if self._stop_requested:
                logger.info("Stop requested, aborting scenario")
                overall_status = StepStatus.SKIPPED
                break
            if self._pause_requested:
                logger.info("Pause requested, stopping at current step")
                overall_status = StepStatus.PENDING
                break

            if self._context:
                self._context.current_step = step_index

            # Execute with retries
            result: StepResult | None = None
            for attempt in range(scenario.max_retries + 1):
                result = await self._execute_step(step, scenario, step_index)
                if result.status == StepStatus.COMPLETED:
                    break
                if attempt < scenario.max_retries:
                    logger.debug(
                        f"Retrying step {step_index} (attempt {attempt + 2}/"
                        f"{scenario.max_retries + 1})"
                    )

            step_results.append(result)

            if result.status == StepStatus.FAILED:
                if not scenario.continue_on_failure:
                    overall_status = StepStatus.FAILED
                    error_msg = f"Step {step_index} failed: {result.error}"
                    break
                else:
                    logger.warning(
                        f"Step {step_index} failed but continuing: {result.error}"
                    )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        scenario_result = ScenarioResult(
            name=scenario.name,
            status=overall_status,
            steps=step_results,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            error=error_msg,
        )

        logger.info(
            f"Scenario {scenario.name} finished with status: {overall_status.value}"
        )
        return scenario_result

    async def run(
        self,
        experiment_id: str | None = None,
        scenarios: list[str] | None = None,
    ) -> list[ScenarioResult]:
        """Run the experiment.

        Executes all scenarios (or specified subset) with registered agents.

        Args:
            experiment_id: Optional experiment ID. Generated if not provided.
            scenarios: Optional list of scenario names to run. Runs all if None.

        Returns:
            List of ScenarioResult objects.

        Raises:
            OrchestratorError: If validation fails or execution errors occur.
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{self._config.experiment.name}_{timestamp}"

        # Validate agents
        self._validate_agents()

        # Determine which scenarios to run
        scenarios_to_run = self._config.scenarios
        if scenarios:
            scenarios_to_run = [
                s for s in self._config.scenarios if s.name in scenarios
            ]
            if not scenarios_to_run:
                raise OrchestratorError(
                    f"No matching scenarios found: {scenarios}",
                    experiment_name=self._config.experiment.name,
                )

        with self._experiment_lifecycle(experiment_id) as context:
            for scenario in scenarios_to_run:
                if self._stop_requested or self._pause_requested:
                    break
                result = await self._execute_scenario(scenario)
                self._scenario_results.append(result)

        return self._scenario_results

    async def resume(
        self,
        context: ExperimentContext,
    ) -> list[ScenarioResult]:
        """Resume a paused experiment from checkpoint.

        Args:
            context: ExperimentContext restored from checkpoint.

        Returns:
            List of ScenarioResult objects for remaining scenarios.

        Raises:
            OrchestratorError: If resumption fails.
        """
        if context.status != ExperimentStatus.PAUSED:
            raise OrchestratorError(
                f"Cannot resume experiment with status: {context.status.value}",
                experiment_name=context.experiment_name,
            )

        # Validate agents
        self._validate_agents()

        # Restore context
        self._context = context
        self._context.status = ExperimentStatus.RUNNING
        self._pause_requested = False
        self._stop_requested = False

        # Find where to resume
        resume_scenario = context.current_scenario
        resume_step = context.current_step

        logger.info(
            f"Resuming experiment {context.experiment_id} from "
            f"scenario '{resume_scenario}', step {resume_step}"
        )

        self._setup_signal_handlers()
        try:
            # Find the scenario to resume
            remaining_scenarios = []
            found_resume_point = False
            for scenario in self._config.scenarios:
                if scenario.name == resume_scenario:
                    found_resume_point = True
                    # Execute remaining steps in current scenario
                    partial_scenario = ScenarioConfig(
                        name=scenario.name,
                        description=scenario.description,
                        steps=scenario.steps[resume_step:],
                        max_retries=scenario.max_retries,
                        continue_on_failure=scenario.continue_on_failure,
                    )
                    remaining_scenarios.append(partial_scenario)
                elif found_resume_point:
                    remaining_scenarios.append(scenario)

            for scenario in remaining_scenarios:
                if self._stop_requested or self._pause_requested:
                    break
                result = await self._execute_scenario(scenario)
                self._scenario_results.append(result)

            if self._pause_requested:
                self._context.status = ExperimentStatus.PAUSED
            elif self._stop_requested:
                self._context.status = ExperimentStatus.CANCELLED
            else:
                self._context.status = ExperimentStatus.COMPLETED

        finally:
            self._restore_signal_handlers()

        return self._scenario_results

    def request_pause(self) -> None:
        """Request the experiment to pause at the next safe point."""
        logger.info("Pause requested")
        self._pause_requested = True

    def request_stop(self) -> None:
        """Request the experiment to stop immediately."""
        logger.info("Stop requested")
        self._stop_requested = True

    def get_results(self) -> list[ScenarioResult]:
        """Get results from the most recent run.

        Returns:
            List of ScenarioResult objects.
        """
        return self._scenario_results


def create_agent_factory(
    agent_factories: dict[AgentType, Callable[[AgentConfig], AgentInterface]] | None = None,
) -> Callable[[AgentConfig], AgentInterface]:
    """Create a factory function for building agents from configuration.

    This function creates a factory that can instantiate agents based on
    their type. Custom agent factories can be provided to extend the
    supported agent types.

    Args:
        agent_factories: Optional dictionary mapping AgentType to factory functions.

    Returns:
        Factory function that creates agents from AgentConfig.

    Raises:
        ValueError: If agent type is not supported.

    Example:
        >>> factory = create_agent_factory({
        ...     AgentType.CUSTOM: lambda config: MyCustomAgent(config)
        ... })
        >>> agent = factory(agent_config)
    """
    factories = agent_factories or {}

    def factory(config: AgentConfig) -> AgentInterface:
        if config.type in factories:
            return factories[config.type](config)
        raise ValueError(
            f"No factory registered for agent type: {config.type.value}. "
            f"Available types: {list(factories.keys())}"
        )

    return factory
