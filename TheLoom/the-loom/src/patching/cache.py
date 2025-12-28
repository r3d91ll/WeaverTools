"""Activation cache management for patching experiments.

This module provides utilities for caching and managing activations during
patching experiments. It handles memory management, cleanup, and provides
efficient retrieval of cached activations for causal intervention analysis.
"""

from __future__ import annotations

import gc
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


@dataclass
class CacheMetadata:
    """Metadata for a cached activation entry."""

    layer: int  # Layer index this activation is from
    component: str  # Component type (e.g., 'resid_pre', 'attn')
    shape: tuple[int, ...]  # Shape of the activation tensor
    dtype: str  # Data type as string
    created_at: float = field(default_factory=time.time)  # Timestamp
    size_bytes: int = 0  # Size in bytes
    run_id: str = ""  # Identifier for the run that produced this
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the metadata.
        """
        return {
            "layer": self.layer,
            "component": self.component,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "run_id": self.run_id,
            "extra": self.extra,
        }


@dataclass
class CachedActivation:
    """Container for a cached activation tensor with metadata."""

    activation: torch.Tensor  # The cached activation
    metadata: CacheMetadata  # Associated metadata

    @property
    def layer(self) -> int:
        """Get the layer index for this activation."""
        return self.metadata.layer

    @property
    def component(self) -> str:
        """Get the component type for this activation."""
        return self.metadata.component

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the activation tensor."""
        return tuple(self.activation.shape)

    @property
    def size_bytes(self) -> int:
        """Get the size of the activation in bytes."""
        return self.activation.element_size() * self.activation.numel()

    def to_numpy(self) -> np.ndarray:
        """
        Convert the activation to a numpy array.

        Handles bfloat16 by converting to float32 first.

        Returns:
            np.ndarray: The activation as a numpy array.
        """
        tensor = self.activation
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.cpu().numpy()

    def clone(self) -> CachedActivation:
        """
        Create a deep copy of this cached activation.

        Returns:
            CachedActivation: A new instance with cloned tensor and metadata.
        """
        return CachedActivation(
            activation=self.activation.clone(),
            metadata=CacheMetadata(
                layer=self.metadata.layer,
                component=self.metadata.component,
                shape=self.metadata.shape,
                dtype=self.metadata.dtype,
                created_at=self.metadata.created_at,
                size_bytes=self.metadata.size_bytes,
                run_id=self.metadata.run_id,
                extra={**self.metadata.extra},
            ),
        )

    def to_device(self, device: str | torch.device) -> CachedActivation:
        """
        Move the activation to a different device.

        Parameters:
            device (str | torch.device): Target device.

        Returns:
            CachedActivation: New instance with activation on target device.
        """
        return CachedActivation(
            activation=self.activation.to(device),
            metadata=CacheMetadata(
                layer=self.metadata.layer,
                component=self.metadata.component,
                shape=self.metadata.shape,
                dtype=self.metadata.dtype,
                created_at=self.metadata.created_at,
                size_bytes=self.metadata.size_bytes,
                run_id=self.metadata.run_id,
                extra={**self.metadata.extra, "device": str(device)},
            ),
        )


class ActivationCache:
    """In-memory cache for storing activations during patching experiments.

    This cache stores activations keyed by (layer, component) pairs and provides
    efficient retrieval and memory management.

    Example usage:
        cache = ActivationCache(run_id="clean_run")
        cache.store(layer=5, component="resid_pre", activation=tensor)
        activation = cache.get(layer=5, component="resid_pre")
    """

    def __init__(
        self,
        run_id: str = "",
        device: str | torch.device | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        """
        Initialize the activation cache.

        Parameters:
            run_id (str): Identifier for this cache's run.
            device (str | torch.device | None): Device to store activations on.
            max_size_bytes (int | None): Maximum cache size in bytes.
        """
        self._cache: dict[tuple[int, str], CachedActivation] = {}
        self._run_id = run_id
        self._device = device
        self._max_size_bytes = max_size_bytes
        self._created_at = time.time()

    @property
    def run_id(self) -> str:
        """Get the run identifier for this cache."""
        return self._run_id

    @property
    def size_bytes(self) -> int:
        """Get the total size of cached activations in bytes."""
        return sum(entry.size_bytes for entry in self._cache.values())

    @property
    def num_entries(self) -> int:
        """Get the number of cached activation entries."""
        return len(self._cache)

    def store(
        self,
        layer: int,
        component: str,
        activation: torch.Tensor,
        extra: dict[str, Any] | None = None,
    ) -> CachedActivation:
        """
        Store an activation in the cache.

        Parameters:
            layer (int): Layer index this activation is from.
            component (str): Component type (e.g., 'resid_pre').
            activation (torch.Tensor): The activation tensor to cache.
            extra (dict[str, Any] | None): Additional metadata.

        Returns:
            CachedActivation: The stored cached activation entry.
        """
        # Move to cache device if specified
        if self._device is not None:
            activation = activation.to(self._device)

        # Create metadata
        metadata = CacheMetadata(
            layer=layer,
            component=component,
            shape=tuple(activation.shape),
            dtype=str(activation.dtype).replace("torch.", ""),
            size_bytes=activation.element_size() * activation.numel(),
            run_id=self._run_id,
            extra=extra or {},
        )

        # Create and store cached activation
        cached = CachedActivation(activation=activation, metadata=metadata)
        key = (layer, component)

        # Check size limit before storing
        if self._max_size_bytes is not None:
            new_total = self.size_bytes + cached.size_bytes
            if key in self._cache:
                new_total -= self._cache[key].size_bytes
            if new_total > self._max_size_bytes:
                raise MemoryError(
                    f"Cache size limit exceeded: {new_total} bytes > {self._max_size_bytes} bytes. "
                    "Consider clearing old entries or increasing the limit."
                )

        self._cache[key] = cached
        return cached

    def get(
        self,
        layer: int,
        component: str,
    ) -> CachedActivation | None:
        """
        Retrieve a cached activation.

        Parameters:
            layer (int): Layer index to retrieve.
            component (str): Component type to retrieve.

        Returns:
            CachedActivation | None: The cached activation, or None if not found.
        """
        return self._cache.get((layer, component))

    def get_tensor(
        self,
        layer: int,
        component: str,
    ) -> torch.Tensor | None:
        """
        Retrieve just the activation tensor.

        Parameters:
            layer (int): Layer index to retrieve.
            component (str): Component type to retrieve.

        Returns:
            torch.Tensor | None: The activation tensor, or None if not found.
        """
        cached = self.get(layer, component)
        return cached.activation if cached is not None else None

    def has(self, layer: int, component: str) -> bool:
        """
        Check if an activation is cached.

        Parameters:
            layer (int): Layer index to check.
            component (str): Component type to check.

        Returns:
            bool: True if the activation is cached.
        """
        return (layer, component) in self._cache

    def remove(self, layer: int, component: str) -> bool:
        """
        Remove a cached activation.

        Parameters:
            layer (int): Layer index to remove.
            component (str): Component type to remove.

        Returns:
            bool: True if entry was removed, False if not found.
        """
        key = (layer, component)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def get_layers(self) -> list[int]:
        """
        Get all layer indices in the cache.

        Returns:
            list[int]: Sorted list of layer indices.
        """
        layers = set(layer for layer, _ in self._cache.keys())
        return sorted(layers)

    def get_components(self, layer: int | None = None) -> list[str]:
        """
        Get all component types in the cache, optionally filtered by layer.

        Parameters:
            layer (int | None): If specified, only return components for this layer.

        Returns:
            list[str]: List of component types.
        """
        if layer is None:
            return list(set(component for _, component in self._cache.keys()))
        return [component for (l, component) in self._cache.keys() if l == layer]

    def get_by_layer(self, layer: int) -> dict[str, CachedActivation]:
        """
        Get all cached activations for a specific layer.

        Parameters:
            layer (int): Layer index to retrieve.

        Returns:
            dict[str, CachedActivation]: Mapping of component to cached activation.
        """
        return {
            component: cached
            for (l, component), cached in self._cache.items()
            if l == layer
        }

    def get_by_component(self, component: str) -> dict[int, CachedActivation]:
        """
        Get all cached activations for a specific component type.

        Parameters:
            component (str): Component type to retrieve.

        Returns:
            dict[int, CachedActivation]: Mapping of layer to cached activation.
        """
        return {
            layer: cached
            for (layer, comp), cached in self._cache.items()
            if comp == component
        }

    def iter_entries(self) -> Iterator[tuple[int, str, CachedActivation]]:
        """
        Iterate over all cached entries.

        Yields:
            tuple[int, str, CachedActivation]: (layer, component, cached_activation)
        """
        for (layer, component), cached in self._cache.items():
            yield layer, component, cached

    def clear(self) -> int:
        """
        Clear all cached activations.

        Returns:
            int: Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        gc.collect()
        return count

    def to_dict(self) -> dict[str, Any]:
        """
        Convert cache contents to a dictionary for serialization.

        Note: Activation tensors are NOT included, only metadata.

        Returns:
            dict[str, Any]: Dictionary representation of the cache.
        """
        return {
            "run_id": self._run_id,
            "created_at": self._created_at,
            "size_bytes": self.size_bytes,
            "num_entries": self.num_entries,
            "entries": [
                {
                    "layer": layer,
                    "component": component,
                    "metadata": cached.metadata.to_dict(),
                }
                for (layer, component), cached in self._cache.items()
            ],
        }

    def clone(self) -> ActivationCache:
        """
        Create a deep copy of this cache.

        Returns:
            ActivationCache: A new cache with cloned activations.
        """
        new_cache = ActivationCache(
            run_id=self._run_id,
            device=self._device,
            max_size_bytes=self._max_size_bytes,
        )
        for (layer, component), cached in self._cache.items():
            new_cache._cache[(layer, component)] = cached.clone()
        return new_cache


class CacheManager:
    """Manager for multiple activation caches with disk persistence support.

    Handles memory management, disk caching, and cleanup for patching experiments
    that require storing activations from multiple runs.

    Example usage:
        manager = CacheManager(cache_dir="/tmp/caches", max_size_mb=4096)
        clean_cache = manager.create_cache("clean_run")
        # ... populate cache ...
        manager.persist_cache("clean_run")  # Save to disk
        manager.clear_memory()  # Free memory
        loaded = manager.load_cache("clean_run")  # Load back
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_size_mb: int = 4096,
        cleanup_on_exit: bool = True,
    ) -> None:
        """
        Initialize the cache manager.

        Parameters:
            cache_dir (str | Path | None): Directory for disk persistence.
            max_size_mb (int): Maximum total cache size in megabytes.
            cleanup_on_exit (bool): Clean up disk caches on manager destruction.
        """
        self._caches: dict[str, ActivationCache] = {}
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._cleanup_on_exit = cleanup_on_exit
        self._created_at = time.time()

        # Create cache directory if specified
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def total_size_bytes(self) -> int:
        """Get the total size of all in-memory caches in bytes."""
        return sum(cache.size_bytes for cache in self._caches.values())

    @property
    def cache_ids(self) -> list[str]:
        """Get list of all cache IDs."""
        return list(self._caches.keys())

    @property
    def num_caches(self) -> int:
        """Get the number of in-memory caches."""
        return len(self._caches)

    def create_cache(
        self,
        cache_id: str,
        device: str | torch.device | None = None,
    ) -> ActivationCache:
        """
        Create a new activation cache.

        Parameters:
            cache_id (str): Unique identifier for the cache.
            device (str | torch.device | None): Device for storing activations.

        Returns:
            ActivationCache: The newly created cache.

        Raises:
            ValueError: If a cache with this ID already exists.
        """
        if cache_id in self._caches:
            raise ValueError(f"Cache with ID '{cache_id}' already exists")

        cache = ActivationCache(
            run_id=cache_id,
            device=device,
        )
        self._caches[cache_id] = cache
        return cache

    def get_cache(self, cache_id: str) -> ActivationCache | None:
        """
        Get an existing cache by ID.

        Parameters:
            cache_id (str): The cache identifier.

        Returns:
            ActivationCache | None: The cache, or None if not found.
        """
        return self._caches.get(cache_id)

    def get_or_create_cache(
        self,
        cache_id: str,
        device: str | torch.device | None = None,
    ) -> ActivationCache:
        """
        Get an existing cache or create a new one.

        Parameters:
            cache_id (str): The cache identifier.
            device (str | torch.device | None): Device for new cache.

        Returns:
            ActivationCache: The existing or newly created cache.
        """
        if cache_id in self._caches:
            return self._caches[cache_id]
        return self.create_cache(cache_id, device)

    def remove_cache(self, cache_id: str) -> bool:
        """
        Remove a cache from memory.

        Parameters:
            cache_id (str): The cache identifier.

        Returns:
            bool: True if cache was removed, False if not found.
        """
        if cache_id in self._caches:
            self._caches[cache_id].clear()
            del self._caches[cache_id]
            gc.collect()
            return True
        return False

    def persist_cache(self, cache_id: str) -> Path | None:
        """
        Save a cache to disk.

        Parameters:
            cache_id (str): The cache identifier.

        Returns:
            Path | None: Path to saved cache, or None if disk persistence disabled.

        Raises:
            ValueError: If cache not found.
        """
        if self._cache_dir is None:
            return None

        cache = self._caches.get(cache_id)
        if cache is None:
            raise ValueError(f"Cache with ID '{cache_id}' not found")

        # Create cache directory
        cache_path = self._cache_dir / cache_id
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save each activation as a separate file
        for layer, component, cached in cache.iter_entries():
            filename = f"layer_{layer}_{component}.pt"
            torch.save(
                {
                    "activation": cached.activation,
                    "metadata": cached.metadata.to_dict(),
                },
                cache_path / filename,
            )

        return cache_path

    def load_cache(
        self,
        cache_id: str,
        device: str | torch.device | None = None,
    ) -> ActivationCache | None:
        """
        Load a cache from disk.

        Parameters:
            cache_id (str): The cache identifier.
            device (str | torch.device | None): Device to load activations to.

        Returns:
            ActivationCache | None: The loaded cache, or None if not found.
        """
        if self._cache_dir is None:
            return None

        cache_path = self._cache_dir / cache_id
        if not cache_path.exists():
            return None

        cache = ActivationCache(run_id=cache_id, device=device)

        # Load each activation file
        for filepath in cache_path.glob("layer_*.pt"):
            data = torch.load(filepath, weights_only=True)
            activation = data["activation"]
            metadata = data["metadata"]

            cache.store(
                layer=metadata["layer"],
                component=metadata["component"],
                activation=activation,
                extra=metadata.get("extra", {}),
            )

        self._caches[cache_id] = cache
        return cache

    def clear_memory(self) -> int:
        """
        Clear all in-memory caches.

        Returns:
            int: Total number of entries cleared.
        """
        total_cleared = 0
        for cache in self._caches.values():
            total_cleared += cache.clear()
        self._caches.clear()
        gc.collect()
        return total_cleared

    def clear_disk(self, cache_id: str | None = None) -> bool:
        """
        Clear disk caches.

        Parameters:
            cache_id (str | None): Specific cache to clear, or None for all.

        Returns:
            bool: True if any caches were cleared.
        """
        if self._cache_dir is None:
            return False

        if cache_id is not None:
            cache_path = self._cache_dir / cache_id
            if cache_path.exists():
                shutil.rmtree(cache_path)
                return True
            return False

        # Clear all disk caches
        if self._cache_dir.exists():
            for item in self._cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            return True
        return False

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            dict[str, Any]: Memory statistics including sizes and limits.
        """
        return {
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": self.total_size_bytes / (1024 * 1024),
            "max_size_mb": self._max_size_bytes / (1024 * 1024),
            "utilization": self.total_size_bytes / self._max_size_bytes if self._max_size_bytes > 0 else 0,
            "num_caches": self.num_caches,
            "cache_sizes": {
                cache_id: cache.size_bytes
                for cache_id, cache in self._caches.items()
            },
        }

    def check_memory_available(self, needed_bytes: int) -> bool:
        """
        Check if enough memory is available for a new allocation.

        Parameters:
            needed_bytes (int): Number of bytes needed.

        Returns:
            bool: True if allocation would fit within limits.
        """
        return (self.total_size_bytes + needed_bytes) <= self._max_size_bytes

    def cleanup(self) -> None:
        """
        Clean up all caches and optionally disk storage.
        """
        self.clear_memory()
        if self._cleanup_on_exit and self._cache_dir is not None:
            self.clear_disk()


def compute_cache_size_estimate(
    num_layers: int,
    hidden_dim: int,
    seq_length: int,
    batch_size: int = 1,
    components_per_layer: int = 2,
    dtype_bytes: int = 4,
) -> int:
    """
    Estimate cache size for an activation patching experiment.

    Parameters:
        num_layers (int): Number of transformer layers.
        hidden_dim (int): Hidden dimension size.
        seq_length (int): Sequence length.
        batch_size (int): Batch size.
        components_per_layer (int): Number of components cached per layer.
        dtype_bytes (int): Bytes per element (4 for float32).

    Returns:
        int: Estimated cache size in bytes.
    """
    elements_per_activation = batch_size * seq_length * hidden_dim
    bytes_per_activation = elements_per_activation * dtype_bytes
    total_activations = num_layers * components_per_layer
    return bytes_per_activation * total_activations


def get_cache_device_recommendation(
    model_device: str | torch.device,
    cache_size_estimate: int,
    gpu_memory_available: int | None = None,
) -> str:
    """
    Recommend the best device for caching activations.

    Parameters:
        model_device (str | torch.device): Device the model is running on.
        cache_size_estimate (int): Estimated cache size in bytes.
        gpu_memory_available (int | None): Available GPU memory in bytes.

    Returns:
        str: Recommended cache device ("cuda", "cpu", or specific device).
    """
    model_device_str = str(model_device)

    # If model is on CPU, cache on CPU
    if "cpu" in model_device_str:
        return "cpu"

    # Check if GPU has enough memory
    if gpu_memory_available is not None:
        # Leave 20% headroom for model operations
        usable_memory = int(gpu_memory_available * 0.8)
        if cache_size_estimate > usable_memory:
            return "cpu"

    # Default to model device
    return model_device_str
