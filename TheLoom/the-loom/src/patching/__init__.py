"""Activation patching utilities for causal intervention analysis."""

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

# Submodule imports will be added as they are implemented:
# - cache: Activation cache management
# - experiments: Experiment orchestration for patching studies

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
]
