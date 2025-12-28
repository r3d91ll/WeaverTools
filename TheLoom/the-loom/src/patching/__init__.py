"""Activation patching utilities for causal intervention analysis."""

from src.patching.cache import (
    ActivationCache,
    CachedActivation,
    CacheManager,
    CacheMetadata,
    compute_cache_size_estimate,
    get_cache_device_recommendation,
)
from src.patching.experiments import (
    ExecutionPath,
    ExperimentConfig,
    ExperimentRecord,
    MultiLayerPatchingStudy,
    PatchingExperiment,
    PatchingResult,
    PathOutput,
    PathRecorder,
    compute_causal_effect,
)
from src.patching.hooks import (
    HookComponent,
    HookManager,
    HookPoint,
    HookRegistration,
    PatchingHookResult,
    build_hook_list,
    create_ablation_hook,
    create_mean_ablation_hook,
    create_noise_hook,
    create_patching_hook,
    get_hook_names_for_layer,
    validate_hook_shapes,
)

__all__: list[str] = [
    # Hook types
    "HookComponent",
    "HookPoint",
    "HookRegistration",
    "HookManager",
    "PatchingHookResult",
    # Hook factories
    "create_patching_hook",
    "create_ablation_hook",
    "create_mean_ablation_hook",
    "create_noise_hook",
    # Utilities
    "validate_hook_shapes",
    "build_hook_list",
    "get_hook_names_for_layer",
    # Cache types
    "CacheMetadata",
    "CachedActivation",
    "ActivationCache",
    "CacheManager",
    # Cache utilities
    "compute_cache_size_estimate",
    "get_cache_device_recommendation",
    # Experiment types
    "ExecutionPath",
    "ExperimentConfig",
    "PathOutput",
    "PatchingResult",
    "ExperimentRecord",
    "PathRecorder",
    "PatchingExperiment",
    "MultiLayerPatchingStudy",
    # Experiment utilities
    "compute_causal_effect",
]
