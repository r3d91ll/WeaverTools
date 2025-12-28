"""Patching hook injection system for activation interventions.

This module provides utilities for creating and managing activation patching hooks
that can be injected into transformer models for causal intervention analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch


class HookComponent(Enum):
    """Supported activation components for patching."""

    RESID_PRE = "resid_pre"  # Residual stream before layer processing
    RESID_POST = "resid_post"  # Residual stream after layer processing
    ATTN = "attn"  # Attention output
    ATTN_Q = "attn_q"  # Query vectors
    ATTN_K = "attn_k"  # Key vectors
    ATTN_V = "attn_v"  # Value vectors
    ATTN_PATTERN = "attn_pattern"  # Attention patterns
    MLP_PRE = "mlp_pre"  # Before MLP
    MLP_POST = "mlp_post"  # After MLP

    @classmethod
    def from_string(cls, value: str) -> HookComponent:
        """
        Convert a string to a HookComponent enum value.

        Parameters:
            value (str): The string representation of the component.

        Returns:
            HookComponent: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match any known component.
        """
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown hook component: {value}. Valid options: {valid}")


@dataclass
class HookPoint:
    """Specification for a hook point in the model.

    Identifies where in the model to inject a patching hook.
    """

    layer: int  # Layer index (0-indexed, -1 for last layer)
    component: HookComponent  # Which component to hook
    position: int | None = None  # Optional token position to patch (None = all)
    head: int | None = None  # Optional attention head to patch (None = all)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_hook_name(self, num_layers: int) -> str:
        """
        Convert this hook point to a TransformerLens-compatible hook name.

        Parameters:
            num_layers (int): Total number of layers in the model for resolving -1.

        Returns:
            str: The hook name in TransformerLens format (e.g., 'blocks.5.hook_resid_pre').
        """
        # Resolve negative layer indices
        resolved_layer = self.layer if self.layer >= 0 else num_layers + self.layer

        # Map component to TransformerLens hook name
        component_map = {
            HookComponent.RESID_PRE: "hook_resid_pre",
            HookComponent.RESID_POST: "hook_resid_post",
            HookComponent.ATTN: "attn.hook_result",
            HookComponent.ATTN_Q: "attn.hook_q",
            HookComponent.ATTN_K: "attn.hook_k",
            HookComponent.ATTN_V: "attn.hook_v",
            HookComponent.ATTN_PATTERN: "attn.hook_pattern",
            HookComponent.MLP_PRE: "mlp.hook_pre",
            HookComponent.MLP_POST: "mlp.hook_post",
        }

        hook_suffix = component_map[self.component]
        return f"blocks.{resolved_layer}.{hook_suffix}"


@dataclass
class PatchingHookResult:
    """Result from executing a patching hook."""

    hook_point: HookPoint
    original_shape: tuple[int, ...]
    patched_shape: tuple[int, ...]
    shape_matched: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookRegistration:
    """Container for a registered patching hook with its cleanup handle."""

    hook_point: HookPoint
    hook_fn: Callable[[torch.Tensor, Any], torch.Tensor]
    handle: Any | None = None  # PyTorch hook handle for removal
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def remove(self) -> None:
        """
        Remove this hook from the model.

        Calls the PyTorch handle.remove() if available and marks as inactive.
        """
        if self.handle is not None:
            self.handle.remove()
        self.is_active = False


def validate_hook_shapes(
    original: torch.Tensor,
    patched: torch.Tensor,
    strict: bool = True,
) -> bool:
    """
    Validate that patched activation matches original shape.

    Per TransformerLens contract, hooks must return tensors with identical shapes.

    Parameters:
        original (torch.Tensor): Original activation tensor.
        patched (torch.Tensor): Patched activation tensor to validate.
        strict (bool): If True, raise ValueError on mismatch; otherwise return False.

    Returns:
        bool: True if shapes match, False if they don't (when strict=False).

    Raises:
        ValueError: If strict=True and shapes don't match.
    """
    if original.shape != patched.shape:
        if strict:
            raise ValueError(
                f"Hook shape mismatch: expected {original.shape}, got {patched.shape}. "
                "Hooks must return tensors with identical shapes to the input."
            )
        return False
    return True


def create_patching_hook(
    source_activation: torch.Tensor,
    position: int | None = None,
    head: int | None = None,
    validate_shapes: bool = True,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """
    Create a patching hook function that replaces activations with source values.

    This is the core function for activation patching. The returned hook can be
    registered with PyTorch or TransformerLens to intervene on activations.

    Parameters:
        source_activation (torch.Tensor): The activation tensor to patch in (from clean run).
        position (int | None): If specified, only patch this token position.
        head (int | None): If specified, only patch this attention head (for attention hooks).
        validate_shapes (bool): If True, validate that shapes match before patching.

    Returns:
        Callable[[torch.Tensor, Any], torch.Tensor]: A hook function compatible with
            PyTorch's register_forward_hook and TransformerLens's run_with_hooks.
    """

    def patch_hook(activation: torch.Tensor, hook: Any = None) -> torch.Tensor:
        """
        Replace activation with source values.

        Parameters:
            activation (torch.Tensor): Current activation from corrupted/target run.
            hook: Hook metadata (from TransformerLens, optional).

        Returns:
            torch.Tensor: Patched activation tensor with same shape as input.
        """
        # Validate shapes if enabled
        if validate_shapes:
            validate_hook_shapes(activation, source_activation, strict=True)

        # Full replacement (most common case)
        if position is None and head is None:
            activation[:] = source_activation
            return activation

        # Position-specific patching
        if position is not None and head is None:
            # activation shape: [batch, seq_len, hidden_dim]
            activation[:, position, :] = source_activation[:, position, :]
            return activation

        # Head-specific patching (for attention components)
        if head is not None and position is None:
            # activation shape: [batch, num_heads, seq_len, head_dim] or similar
            if activation.ndim >= 3:
                activation[:, head, ...] = source_activation[:, head, ...]
            return activation

        # Both position and head specified
        if position is not None and head is not None:
            # activation shape: [batch, num_heads, seq_len, head_dim]
            if activation.ndim >= 4:
                activation[:, head, position, :] = source_activation[:, head, position, :]
            return activation

        return activation

    return patch_hook


def create_ablation_hook(
    value: float = 0.0,
    position: int | None = None,
    head: int | None = None,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """
    Create an ablation hook that sets activations to a constant value.

    This is useful for causal analysis where we want to "knock out" a component
    by setting its activations to zero or another constant.

    Parameters:
        value (float): The constant value to set activations to (default 0.0).
        position (int | None): If specified, only ablate this token position.
        head (int | None): If specified, only ablate this attention head.

    Returns:
        Callable[[torch.Tensor, Any], torch.Tensor]: An ablation hook function.
    """

    def ablation_hook(activation: torch.Tensor, hook: Any = None) -> torch.Tensor:
        """
        Set activation values to a constant.

        Parameters:
            activation (torch.Tensor): Current activation to ablate.
            hook: Hook metadata (from TransformerLens, optional).

        Returns:
            torch.Tensor: Ablated activation tensor with same shape as input.
        """
        # Full ablation
        if position is None and head is None:
            activation.fill_(value)
            return activation

        # Position-specific ablation
        if position is not None and head is None:
            activation[:, position, :].fill_(value)
            return activation

        # Head-specific ablation
        if head is not None and position is None:
            if activation.ndim >= 3:
                activation[:, head, ...].fill_(value)
            return activation

        # Both position and head specified
        if position is not None and head is not None:
            if activation.ndim >= 4:
                activation[:, head, position, :].fill_(value)
            return activation

        return activation

    return ablation_hook


def create_mean_ablation_hook(
    reference_activations: torch.Tensor,
    dim: int = 0,
    position: int | None = None,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """
    Create a hook that replaces activations with their mean value across a dimension.

    This is useful for analyzing the importance of variance in activations.
    Common use: replace with mean across batch or sequence dimension.

    Parameters:
        reference_activations (torch.Tensor): Tensor to compute mean from.
        dim (int): Dimension to compute mean over.
        position (int | None): If specified, only apply to this token position.

    Returns:
        Callable[[torch.Tensor, Any], torch.Tensor]: A mean ablation hook function.
    """
    mean_activation = reference_activations.mean(dim=dim, keepdim=True)

    def mean_ablation_hook(activation: torch.Tensor, hook: Any = None) -> torch.Tensor:
        """
        Replace activation with mean value.

        Parameters:
            activation (torch.Tensor): Current activation to replace.
            hook: Hook metadata (from TransformerLens, optional).

        Returns:
            torch.Tensor: Activation tensor with same shape, filled with mean values.
        """
        if position is None:
            # Broadcast mean to full shape
            return mean_activation.expand_as(activation)

        # Only replace specific position
        result = activation.clone()
        result[:, position, :] = mean_activation.squeeze(dim)[:, position, :]
        return result

    return mean_ablation_hook


def create_noise_hook(
    scale: float = 0.1,
    position: int | None = None,
    head: int | None = None,
    seed: int | None = None,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """
    Create a hook that adds Gaussian noise to activations.

    Useful for robustness analysis and understanding sensitivity to perturbations.

    Parameters:
        scale (float): Standard deviation of Gaussian noise to add.
        position (int | None): If specified, only add noise to this token position.
        head (int | None): If specified, only add noise to this attention head.
        seed (int | None): If specified, use this seed for reproducible noise.

    Returns:
        Callable[[torch.Tensor, Any], torch.Tensor]: A noise injection hook function.
    """

    def noise_hook(activation: torch.Tensor, hook: Any = None) -> torch.Tensor:
        """
        Add Gaussian noise to activation.

        Parameters:
            activation (torch.Tensor): Current activation to perturb.
            hook: Hook metadata (from TransformerLens, optional).

        Returns:
            torch.Tensor: Perturbed activation tensor with same shape as input.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=activation.device)
            generator.manual_seed(seed)

        # Generate noise with same shape as activation
        noise = torch.randn(
            activation.shape,
            device=activation.device,
            dtype=activation.dtype,
            generator=generator,
        ) * scale

        # Apply noise selectively if position/head specified
        if position is None and head is None:
            return activation + noise

        if position is not None and head is None:
            activation[:, position, :] += noise[:, position, :]
            return activation

        if head is not None and position is None:
            if activation.ndim >= 3:
                activation[:, head, ...] += noise[:, head, ...]
            return activation

        if position is not None and head is not None:
            if activation.ndim >= 4:
                activation[:, head, position, :] += noise[:, head, position, :]
            return activation

        return activation

    return noise_hook


class HookManager:
    """Manager for registering, tracking, and cleaning up patching hooks."""

    def __init__(self, validate_shapes: bool = True) -> None:
        """
        Initialize the hook manager.

        Parameters:
            validate_shapes (bool): If True, validate hook shapes by default.
        """
        self._registrations: list[HookRegistration] = []
        self._validate_shapes = validate_shapes

    @property
    def active_hooks(self) -> list[HookRegistration]:
        """
        Get list of currently active hook registrations.

        Returns:
            list[HookRegistration]: List of active (not removed) hooks.
        """
        return [r for r in self._registrations if r.is_active]

    def register_patch_hook(
        self,
        hook_point: HookPoint,
        source_activation: torch.Tensor,
        model: Any = None,
    ) -> HookRegistration:
        """
        Register a patching hook at the specified hook point.

        Parameters:
            hook_point (HookPoint): Where to inject the hook.
            source_activation (torch.Tensor): The activation to patch in.
            model: Optional model to register hook on (if None, hook is just stored).

        Returns:
            HookRegistration: Container with hook function and cleanup handle.
        """
        hook_fn = create_patching_hook(
            source_activation=source_activation,
            position=hook_point.position,
            head=hook_point.head,
            validate_shapes=self._validate_shapes,
        )

        registration = HookRegistration(
            hook_point=hook_point,
            hook_fn=hook_fn,
            handle=None,
            is_active=True,
        )

        self._registrations.append(registration)
        return registration

    def register_ablation_hook(
        self,
        hook_point: HookPoint,
        value: float = 0.0,
    ) -> HookRegistration:
        """
        Register an ablation hook at the specified hook point.

        Parameters:
            hook_point (HookPoint): Where to inject the hook.
            value (float): Constant value to set activations to.

        Returns:
            HookRegistration: Container with hook function and cleanup handle.
        """
        hook_fn = create_ablation_hook(
            value=value,
            position=hook_point.position,
            head=hook_point.head,
        )

        registration = HookRegistration(
            hook_point=hook_point,
            hook_fn=hook_fn,
            handle=None,
            is_active=True,
        )

        self._registrations.append(registration)
        return registration

    def get_hooks_for_layer(self, layer: int) -> list[HookRegistration]:
        """
        Get all active hooks registered for a specific layer.

        Parameters:
            layer (int): Layer index to filter by.

        Returns:
            list[HookRegistration]: Active hooks targeting the specified layer.
        """
        return [r for r in self.active_hooks if r.hook_point.layer == layer]

    def get_hooks_for_component(self, component: HookComponent) -> list[HookRegistration]:
        """
        Get all active hooks registered for a specific component type.

        Parameters:
            component (HookComponent): Component type to filter by.

        Returns:
            list[HookRegistration]: Active hooks targeting the specified component.
        """
        return [r for r in self.active_hooks if r.hook_point.component == component]

    def remove_all_hooks(self) -> int:
        """
        Remove all registered hooks.

        Returns:
            int: Number of hooks removed.
        """
        count = 0
        for registration in self._registrations:
            if registration.is_active:
                registration.remove()
                count += 1
        return count

    def clear(self) -> None:
        """
        Remove all hooks and clear the registration list.
        """
        self.remove_all_hooks()
        self._registrations.clear()


def build_hook_list(
    hook_registrations: list[HookRegistration],
    num_layers: int,
) -> list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]]:
    """
    Convert hook registrations to TransformerLens-compatible hook list.

    Parameters:
        hook_registrations (list[HookRegistration]): List of registered hooks.
        num_layers (int): Number of layers in the model for resolving layer indices.

    Returns:
        list[tuple[str, Callable]]: List of (hook_name, hook_fn) tuples for
            use with TransformerLens's run_with_hooks method.
    """
    hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = []

    for registration in hook_registrations:
        if not registration.is_active:
            continue

        hook_name = registration.hook_point.to_hook_name(num_layers)
        hooks.append((hook_name, registration.hook_fn))

    return hooks


def get_hook_names_for_layer(
    layer: int,
    components: list[HookComponent] | None = None,
    num_layers: int = 12,
) -> list[str]:
    """
    Get TransformerLens hook names for a layer and optional components.

    Parameters:
        layer (int): Layer index (0-indexed, -1 for last).
        components (list[HookComponent] | None): Components to include.
            If None, returns all standard components.
        num_layers (int): Total layers for resolving negative indices.

    Returns:
        list[str]: List of TransformerLens hook names.
    """
    if components is None:
        components = [
            HookComponent.RESID_PRE,
            HookComponent.RESID_POST,
            HookComponent.ATTN,
            HookComponent.MLP_POST,
        ]

    hook_names: list[str] = []
    for component in components:
        hook_point = HookPoint(layer=layer, component=component)
        hook_names.append(hook_point.to_hook_name(num_layers))

    return hook_names
