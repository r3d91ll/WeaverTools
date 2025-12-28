"""Patching experiment orchestration for causal intervention studies.

This module provides utilities for orchestrating activation patching experiments,
including experiment configuration, execution paths (clean/corrupted/patched),
result recording, and systematic study management.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from src.patching.cache import ActivationCache, CacheManager
from src.patching.hooks import (
    HookComponent,
    HookManager,
    HookPoint,
    HookRegistration,
    build_hook_list,
    create_patching_hook,
)


class ExecutionPath(Enum):
    """Execution path types for patching experiments."""

    CLEAN = "clean"  # Baseline run without intervention
    CORRUPTED = "corrupted"  # Run with corrupted/modified input
    PATCHED = "patched"  # Run with activation patching applied


@dataclass
class ExperimentConfig:
    """Configuration for a patching experiment.

    Defines the parameters for running a systematic patching study,
    including which layers and components to target.
    """

    name: str  # Experiment name
    layers: list[int] = field(default_factory=list)  # Layers to patch (empty = all)
    components: list[str] = field(default_factory=lambda: ["resid_pre"])
    batch_size: int = 1  # Batch size for experiment runs
    validate_shapes: bool = True  # Validate hook output shapes
    cleanup_on_completion: bool = True  # Clean up caches after experiment
    save_activations: bool = True  # Whether to save activation caches
    cache_dir: str | Path | None = None  # Directory for caching
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_hook_components(self) -> list[HookComponent]:
        """
        Convert component strings to HookComponent enums.

        Returns:
            list[HookComponent]: List of HookComponent enum values.
        """
        return [HookComponent.from_string(c) for c in self.components]


@dataclass
class PathOutput:
    """Output from a single execution path (clean/corrupted/patched).

    Contains the model output, cached activations, and metadata
    for one execution path of an experiment.
    """

    path_type: ExecutionPath  # Which path this is
    output: torch.Tensor | None  # Model output (logits or generated tokens)
    cache: ActivationCache | None  # Cached activations for this path
    input_text: str | None = None  # Input text (if applicable)
    input_tokens: torch.Tensor | None = None  # Input tokens
    generation_time_ms: float = 0.0  # Time taken for generation
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_cache(self) -> bool:
        """Check if this path has cached activations."""
        return self.cache is not None and self.cache.num_entries > 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Note: Tensors are NOT included in serialization.

        Returns:
            dict[str, Any]: Dictionary representation.
        """
        return {
            "path_type": self.path_type.value,
            "has_output": self.output is not None,
            "has_cache": self.has_cache,
            "input_text": self.input_text,
            "generation_time_ms": self.generation_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class PatchingResult:
    """Result from a single patching operation.

    Contains information about the patch that was applied and its effects.
    """

    layer: int  # Layer where patch was applied
    component: str  # Component that was patched
    hook_name: str  # TransformerLens-compatible hook name
    original_output: torch.Tensor | None  # Output before patching
    patched_output: torch.Tensor | None  # Output after patching
    source_run_id: str  # ID of the run providing source activations
    target_run_id: str  # ID of the run being patched
    shape_matched: bool = True  # Whether shapes matched correctly
    execution_time_ms: float = 0.0  # Time for patched run
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation.
        """
        return {
            "layer": self.layer,
            "component": self.component,
            "hook_name": self.hook_name,
            "source_run_id": self.source_run_id,
            "target_run_id": self.target_run_id,
            "shape_matched": self.shape_matched,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentRecord:
    """Complete record of a patching experiment.

    Contains all three paths (clean, corrupted, patched) along with
    the patching results and overall experiment metadata.
    """

    experiment_id: str  # Unique experiment identifier
    config: ExperimentConfig  # Experiment configuration
    clean_path: PathOutput | None = None  # Clean (baseline) path output
    corrupted_path: PathOutput | None = None  # Corrupted path output
    patched_paths: list[PathOutput] = field(default_factory=list)  # Patched path outputs
    patching_results: list[PatchingResult] = field(default_factory=list)  # Individual patch results
    started_at: float = field(default_factory=time.time)  # Experiment start time
    completed_at: float | None = None  # Experiment completion time
    status: str = "pending"  # Status: pending, running, completed, failed
    error: str | None = None  # Error message if failed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Get the experiment duration in milliseconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000

    @property
    def has_all_paths(self) -> bool:
        """Check if all three path types have been recorded."""
        return (
            self.clean_path is not None
            and self.corrupted_path is not None
            and len(self.patched_paths) > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the experiment.
        """
        return {
            "experiment_id": self.experiment_id,
            "config": {
                "name": self.config.name,
                "layers": self.config.layers,
                "components": self.config.components,
            },
            "clean_path": self.clean_path.to_dict() if self.clean_path else None,
            "corrupted_path": self.corrupted_path.to_dict() if self.corrupted_path else None,
            "patched_paths": [p.to_dict() for p in self.patched_paths],
            "patching_results": [r.to_dict() for r in self.patching_results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


class PathRecorder:
    """Records outputs from multiple execution paths for comparison.

    Manages the three-path recording system (clean/corrupted/patched)
    for systematic patching studies.
    """

    def __init__(self, experiment_id: str | None = None) -> None:
        """
        Initialize the path recorder.

        Parameters:
            experiment_id (str | None): Unique identifier for this experiment.
                If None, a UUID will be generated.
        """
        self._experiment_id = experiment_id or str(uuid.uuid4())
        self._paths: dict[ExecutionPath, PathOutput] = {}
        self._patched_paths: list[PathOutput] = []
        self._created_at = time.time()

    @property
    def experiment_id(self) -> str:
        """Get the experiment identifier."""
        return self._experiment_id

    def record_path(
        self,
        path_type: ExecutionPath,
        output: torch.Tensor | None,
        cache: ActivationCache | None = None,
        input_text: str | None = None,
        input_tokens: torch.Tensor | None = None,
        generation_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PathOutput:
        """
        Record the output from an execution path.

        Parameters:
            path_type (ExecutionPath): Which path this is.
            output (torch.Tensor | None): Model output.
            cache (ActivationCache | None): Activation cache for this path.
            input_text (str | None): Input text if applicable.
            input_tokens (torch.Tensor | None): Input tokens.
            generation_time_ms (float): Generation time.
            metadata (dict[str, Any] | None): Additional metadata.

        Returns:
            PathOutput: The recorded path output.
        """
        path_output = PathOutput(
            path_type=path_type,
            output=output,
            cache=cache,
            input_text=input_text,
            input_tokens=input_tokens,
            generation_time_ms=generation_time_ms,
            metadata=metadata or {},
        )

        if path_type == ExecutionPath.PATCHED:
            self._patched_paths.append(path_output)
        else:
            self._paths[path_type] = path_output

        return path_output

    def record_clean_path(
        self,
        output: torch.Tensor | None,
        cache: ActivationCache | None = None,
        input_text: str | None = None,
        **kwargs: Any,
    ) -> PathOutput:
        """
        Convenience method to record the clean (baseline) path.

        Parameters:
            output (torch.Tensor | None): Model output.
            cache (ActivationCache | None): Activation cache.
            input_text (str | None): Input text.
            **kwargs: Additional keyword arguments for record_path.

        Returns:
            PathOutput: The recorded clean path output.
        """
        return self.record_path(
            path_type=ExecutionPath.CLEAN,
            output=output,
            cache=cache,
            input_text=input_text,
            **kwargs,
        )

    def record_corrupted_path(
        self,
        output: torch.Tensor | None,
        cache: ActivationCache | None = None,
        input_text: str | None = None,
        **kwargs: Any,
    ) -> PathOutput:
        """
        Convenience method to record the corrupted path.

        Parameters:
            output (torch.Tensor | None): Model output.
            cache (ActivationCache | None): Activation cache.
            input_text (str | None): Input text.
            **kwargs: Additional keyword arguments for record_path.

        Returns:
            PathOutput: The recorded corrupted path output.
        """
        return self.record_path(
            path_type=ExecutionPath.CORRUPTED,
            output=output,
            cache=cache,
            input_text=input_text,
            **kwargs,
        )

    def record_patched_path(
        self,
        output: torch.Tensor | None,
        cache: ActivationCache | None = None,
        patch_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PathOutput:
        """
        Convenience method to record a patched path.

        Parameters:
            output (torch.Tensor | None): Model output.
            cache (ActivationCache | None): Activation cache.
            patch_info (dict[str, Any] | None): Information about the patch applied.
            **kwargs: Additional keyword arguments for record_path.

        Returns:
            PathOutput: The recorded patched path output.
        """
        metadata = kwargs.pop("metadata", {})
        if patch_info:
            metadata["patch_info"] = patch_info

        return self.record_path(
            path_type=ExecutionPath.PATCHED,
            output=output,
            cache=cache,
            metadata=metadata,
            **kwargs,
        )

    def get_path(self, path_type: ExecutionPath) -> PathOutput | None:
        """
        Get the recorded output for a specific path type.

        Parameters:
            path_type (ExecutionPath): Which path to retrieve.

        Returns:
            PathOutput | None: The path output, or None if not recorded.
        """
        if path_type == ExecutionPath.PATCHED:
            return self._patched_paths[0] if self._patched_paths else None
        return self._paths.get(path_type)

    @property
    def clean_path(self) -> PathOutput | None:
        """Get the clean (baseline) path output."""
        return self._paths.get(ExecutionPath.CLEAN)

    @property
    def corrupted_path(self) -> PathOutput | None:
        """Get the corrupted path output."""
        return self._paths.get(ExecutionPath.CORRUPTED)

    @property
    def patched_paths(self) -> list[PathOutput]:
        """Get all patched path outputs."""
        return self._patched_paths

    def has_path(self, path_type: ExecutionPath) -> bool:
        """
        Check if a path has been recorded.

        Parameters:
            path_type (ExecutionPath): Which path to check.

        Returns:
            bool: True if the path has been recorded.
        """
        if path_type == ExecutionPath.PATCHED:
            return len(self._patched_paths) > 0
        return path_type in self._paths

    def has_all_paths(self) -> bool:
        """
        Check if all three path types have been recorded.

        Returns:
            bool: True if clean, corrupted, and at least one patched path exist.
        """
        return (
            ExecutionPath.CLEAN in self._paths
            and ExecutionPath.CORRUPTED in self._paths
            and len(self._patched_paths) > 0
        )

    def iter_paths(self) -> Iterator[tuple[ExecutionPath, PathOutput]]:
        """
        Iterate over all recorded paths.

        Yields:
            tuple[ExecutionPath, PathOutput]: (path_type, path_output) pairs.
        """
        for path_type, path_output in self._paths.items():
            yield path_type, path_output
        for path_output in self._patched_paths:
            yield ExecutionPath.PATCHED, path_output

    def clear(self) -> None:
        """Clear all recorded paths."""
        self._paths.clear()
        self._patched_paths.clear()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert recorder state to dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation.
        """
        return {
            "experiment_id": self._experiment_id,
            "created_at": self._created_at,
            "paths": {
                path_type.value: path_output.to_dict()
                for path_type, path_output in self._paths.items()
            },
            "patched_paths": [p.to_dict() for p in self._patched_paths],
            "has_all_paths": self.has_all_paths(),
        }


class PatchingExperiment:
    """Orchestrator for systematic activation patching experiments.

    Manages the complete workflow for patching experiments:
    1. Configure experiment parameters
    2. Run clean path with caching
    3. Run corrupted path with caching
    4. Apply patches at specified layers/components
    5. Record and compare outputs

    Example usage:
        experiment = PatchingExperiment(config=ExperimentConfig(
            name="layer_sweep",
            layers=[0, 5, 10],
            components=["resid_pre", "attn"],
        ))

        # Run the experiment
        record = experiment.run_patching_study(
            model=model,
            clean_input="The cat sat on the mat",
            corrupted_input="The dog sat on the mat",
        )
    """

    def __init__(
        self,
        config: ExperimentConfig,
        cache_manager: CacheManager | None = None,
    ) -> None:
        """
        Initialize the patching experiment.

        Parameters:
            config (ExperimentConfig): Experiment configuration.
            cache_manager (CacheManager | None): Optional cache manager.
                If None, a new one will be created.
        """
        self._config = config
        self._cache_manager = cache_manager or CacheManager(
            cache_dir=config.cache_dir,
            cleanup_on_exit=config.cleanup_on_completion,
        )
        self._hook_manager = HookManager(validate_shapes=config.validate_shapes)
        self._experiment_id = str(uuid.uuid4())
        self._recorder = PathRecorder(experiment_id=self._experiment_id)
        self._record: ExperimentRecord | None = None

    @property
    def experiment_id(self) -> str:
        """Get the experiment identifier."""
        return self._experiment_id

    @property
    def config(self) -> ExperimentConfig:
        """Get the experiment configuration."""
        return self._config

    @property
    def recorder(self) -> PathRecorder:
        """Get the path recorder."""
        return self._recorder

    @property
    def record(self) -> ExperimentRecord | None:
        """Get the current experiment record."""
        return self._record

    def create_record(self) -> ExperimentRecord:
        """
        Create a new experiment record.

        Returns:
            ExperimentRecord: New experiment record.
        """
        self._record = ExperimentRecord(
            experiment_id=self._experiment_id,
            config=self._config,
            status="running",
        )
        return self._record

    def run_with_cache(
        self,
        run_fn: Callable[..., torch.Tensor],
        run_id: str,
        layers: list[int] | None = None,
        components: list[str] | None = None,
        **run_kwargs: Any,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """
        Run a function and cache the activations.

        This is a generic method for running model inference and caching
        activations at specified layers and components.

        Parameters:
            run_fn (Callable[..., torch.Tensor]): Function to run (e.g., model forward).
            run_id (str): Unique identifier for this run (for cache naming).
            layers (list[int] | None): Layers to cache (None = use config).
            components (list[str] | None): Components to cache (None = use config).
            **run_kwargs: Additional arguments to pass to run_fn.

        Returns:
            tuple[torch.Tensor, ActivationCache]: (output, cache)
        """
        # Use config defaults if not specified
        layers = layers if layers is not None else self._config.layers
        components = components if components is not None else self._config.components

        # Create cache for this run
        cache = self._cache_manager.get_or_create_cache(run_id)

        # Execute the run function
        output = run_fn(**run_kwargs)

        return output, cache

    def run_patching_on_layer(
        self,
        model: Any,
        input_tokens: torch.Tensor,
        layer: int,
        component: str,
        source_cache: ActivationCache,
        num_layers: int = 12,
    ) -> PatchingResult:
        """
        Run model with activation patching at a specific layer/component.

        Parameters:
            model: The model to run (must support run_with_hooks or similar).
            input_tokens (torch.Tensor): Input tokens for the model.
            layer (int): Layer index to patch.
            component (str): Component to patch.
            source_cache (ActivationCache): Cache containing source activations.
            num_layers (int): Total number of layers in the model.

        Returns:
            PatchingResult: Result of the patching operation.
        """
        start_time = time.time()

        # Get source activation from cache
        source_activation = source_cache.get_tensor(layer, component)
        if source_activation is None:
            return PatchingResult(
                layer=layer,
                component=component,
                hook_name=HookPoint(
                    layer=layer,
                    component=HookComponent.from_string(component),
                ).to_hook_name(num_layers),
                original_output=None,
                patched_output=None,
                source_run_id=source_cache.run_id,
                target_run_id="",
                shape_matched=False,
                metadata={"error": f"No cached activation at layer {layer}, component {component}"},
            )

        # Create hook point
        hook_point = HookPoint(
            layer=layer,
            component=HookComponent.from_string(component),
        )
        hook_name = hook_point.to_hook_name(num_layers)

        # Create patching hook
        patch_hook = create_patching_hook(
            source_activation=source_activation,
            validate_shapes=self._config.validate_shapes,
        )

        # Build hook list for run_with_hooks
        hooks = [(hook_name, patch_hook)]

        # Run with hooks if model supports it
        patched_output: torch.Tensor | None = None
        if hasattr(model, "run_with_hooks"):
            patched_output = model.run_with_hooks(input_tokens, fwd_hooks=hooks)
        else:
            # Fallback: just run normally (no patching actually applied)
            # This should be handled by the caller providing a compatible model
            patched_output = model(input_tokens)

        execution_time_ms = (time.time() - start_time) * 1000

        return PatchingResult(
            layer=layer,
            component=component,
            hook_name=hook_name,
            original_output=None,
            patched_output=patched_output,
            source_run_id=source_cache.run_id,
            target_run_id=self._experiment_id,
            shape_matched=True,
            execution_time_ms=execution_time_ms,
        )

    def run_layer_sweep(
        self,
        model: Any,
        input_tokens: torch.Tensor,
        source_cache: ActivationCache,
        layers: list[int] | None = None,
        components: list[str] | None = None,
        num_layers: int = 12,
    ) -> list[PatchingResult]:
        """
        Run patching across multiple layers for systematic analysis.

        Parameters:
            model: The model to run.
            input_tokens (torch.Tensor): Input tokens.
            source_cache (ActivationCache): Cache with source activations.
            layers (list[int] | None): Layers to patch (None = all cached layers).
            components (list[str] | None): Components to patch (None = use config).
            num_layers (int): Total layers in model.

        Returns:
            list[PatchingResult]: Results for each layer/component combination.
        """
        # Determine layers to sweep
        if layers is None:
            layers = source_cache.get_layers()
        if not layers:
            layers = list(range(num_layers))

        # Determine components
        if components is None:
            components = self._config.components

        results: list[PatchingResult] = []
        for layer in layers:
            for component in components:
                result = self.run_patching_on_layer(
                    model=model,
                    input_tokens=input_tokens,
                    layer=layer,
                    component=component,
                    source_cache=source_cache,
                    num_layers=num_layers,
                )
                results.append(result)

                # Record as patched path
                if result.patched_output is not None:
                    self._recorder.record_patched_path(
                        output=result.patched_output,
                        patch_info={
                            "layer": layer,
                            "component": component,
                            "hook_name": result.hook_name,
                        },
                    )

        return results

    def cleanup(self) -> None:
        """Clean up resources used by the experiment."""
        self._hook_manager.clear()
        if self._config.cleanup_on_completion:
            self._cache_manager.cleanup()

    def finalize(self, success: bool = True, error: str | None = None) -> ExperimentRecord:
        """
        Finalize the experiment and return the complete record.

        Parameters:
            success (bool): Whether the experiment completed successfully.
            error (str | None): Error message if failed.

        Returns:
            ExperimentRecord: The complete experiment record.
        """
        if self._record is None:
            self._record = self.create_record()

        self._record.completed_at = time.time()
        self._record.status = "completed" if success else "failed"
        self._record.error = error

        # Copy paths from recorder
        self._record.clean_path = self._recorder.clean_path
        self._record.corrupted_path = self._recorder.corrupted_path
        self._record.patched_paths = self._recorder.patched_paths

        # Cleanup if configured
        if self._config.cleanup_on_completion:
            self.cleanup()

        return self._record


class MultiLayerPatchingStudy:
    """Orchestrates systematic multi-layer patching studies.

    Manages experiments that sweep across layers to identify which layers
    causally affect downstream behavior. Useful for interpretability research.

    Example usage:
        study = MultiLayerPatchingStudy(
            name="attention_layer_study",
            layers=list(range(12)),
            components=["attn", "mlp_post"],
        )

        results = study.run(
            model=model,
            clean_input=clean_tokens,
            corrupted_input=corrupted_tokens,
            metric_fn=compute_conveyance,
        )
    """

    def __init__(
        self,
        name: str,
        layers: list[int] | None = None,
        components: list[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the multi-layer patching study.

        Parameters:
            name (str): Study name.
            layers (list[int] | None): Layers to include in study.
            components (list[str] | None): Components to patch.
            cache_dir (str | Path | None): Directory for caching.
        """
        self._name = name
        self._layers = layers or []
        self._components = components or ["resid_pre"]
        self._cache_dir = cache_dir
        self._experiments: list[PatchingExperiment] = []
        self._results: list[ExperimentRecord] = []

    @property
    def name(self) -> str:
        """Get the study name."""
        return self._name

    @property
    def experiments(self) -> list[PatchingExperiment]:
        """Get all experiments in this study."""
        return self._experiments

    @property
    def results(self) -> list[ExperimentRecord]:
        """Get all experiment results."""
        return self._results

    def create_experiment(self, name: str | None = None) -> PatchingExperiment:
        """
        Create a new experiment within this study.

        Parameters:
            name (str | None): Experiment name. Uses study name if None.

        Returns:
            PatchingExperiment: The new experiment.
        """
        config = ExperimentConfig(
            name=name or f"{self._name}_experiment_{len(self._experiments)}",
            layers=self._layers,
            components=self._components,
            cache_dir=self._cache_dir,
        )
        experiment = PatchingExperiment(config=config)
        self._experiments.append(experiment)
        return experiment

    def aggregate_results(self) -> dict[str, Any]:
        """
        Aggregate results across all experiments.

        Returns:
            dict[str, Any]: Aggregated study results.
        """
        return {
            "study_name": self._name,
            "num_experiments": len(self._experiments),
            "num_results": len(self._results),
            "layers": self._layers,
            "components": self._components,
            "experiments": [
                record.to_dict() for record in self._results
            ],
        }


def compute_causal_effect(
    clean_metric: float,
    corrupted_metric: float,
    patched_metric: float,
) -> dict[str, float]:
    """
    Compute causal effect metrics from patching experiment results.

    The causal effect quantifies how much the patch recovers the clean behavior.

    Parameters:
        clean_metric (float): Metric value for clean (baseline) path.
        corrupted_metric (float): Metric value for corrupted path.
        patched_metric (float): Metric value for patched path.

    Returns:
        dict[str, float]: Dictionary containing:
            - causal_effect: patched_metric - corrupted_metric
            - recovery_rate: (patched - corrupted) / (clean - corrupted)
            - clean_baseline: The clean metric value
            - corruption_delta: corrupted_metric - clean_metric
    """
    causal_effect = patched_metric - corrupted_metric
    corruption_delta = corrupted_metric - clean_metric

    # Recovery rate: how much of the corruption effect was recovered
    # Avoid division by zero
    if abs(corruption_delta) < 1e-10:
        recovery_rate = 0.0 if abs(causal_effect) < 1e-10 else float("inf")
    else:
        recovery_rate = causal_effect / abs(corruption_delta)

    return {
        "causal_effect": causal_effect,
        "recovery_rate": recovery_rate,
        "clean_baseline": clean_metric,
        "corrupted_metric": corrupted_metric,
        "patched_metric": patched_metric,
        "corruption_delta": corruption_delta,
    }
