"""Hidden state extraction and analysis utilities.

This module provides utilities for extracting and processing hidden states
from transformer models - the core capability for conveyance measurement.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer
    from transformer_lens.ActivationCache import ActivationCache


@dataclass
class HiddenStateResult:
    """Container for extracted hidden state with metadata."""

    vector: np.ndarray  # The hidden state as numpy array
    shape: tuple[int, ...]
    layer: int  # Which layer (-1 = last)
    dtype: str  # Original dtype as string
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_list(self) -> list[float]:
        """
        Return the hidden state vector as a flat list of floats suitable for JSON serialization.
        
        Returns:
            list[float]: Flattened hidden-state values.
        """
        result: list[float] = self.vector.flatten().tolist()
        return result

    def l2_normalize(self) -> HiddenStateResult:
        """
        Produce a new HiddenStateResult whose vector is scaled to have unit L2 norm.
        
        If the original vector has zero L2 norm, the vector is left unchanged. The returned object's metadata is updated with "normalized": True.
        
        Returns:
            HiddenStateResult: New instance with an L2-normalized vector (or the original vector if its norm is zero) and updated metadata.
        """
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            normalized = self.vector / norm
        else:
            normalized = self.vector
        return HiddenStateResult(
            vector=normalized,
            shape=self.shape,
            layer=self.layer,
            dtype=self.dtype,
            metadata={**self.metadata, "normalized": True},
        )


# Mapping of precision string names to torch dtypes
PRECISION_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def extract_with_precision(
    hidden_states_dict: dict[int, torch.Tensor],
    precision: str = "fp32",
) -> dict[int, torch.Tensor]:
    """
    Convert hidden state tensors to the specified precision.

    This function enables memory-efficient storage and computation by converting
    hidden states to lower precision formats when full precision is not required.

    Parameters:
        hidden_states_dict (dict[int, torch.Tensor]): Mapping from layer index to
            hidden state tensors.
        precision (str): Target precision format. Supported values:
            - "fp32" / "float32": 32-bit floating point (default)
            - "fp16" / "float16": 16-bit floating point
            - "bf16" / "bfloat16": Brain floating point 16

    Returns:
        dict[int, torch.Tensor]: Mapping from layer index to tensors converted
            to the specified precision.

    Raises:
        ValueError: If an unsupported precision format is specified.
    """
    if precision not in PRECISION_DTYPE_MAP:
        valid_precisions = ", ".join(sorted(PRECISION_DTYPE_MAP.keys()))
        raise ValueError(
            f"Unsupported precision '{precision}'. Valid options: {valid_precisions}"
        )

    target_dtype = PRECISION_DTYPE_MAP[precision]
    results: dict[int, torch.Tensor] = {}

    for layer_idx, tensor in hidden_states_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Convert to target precision
            results[layer_idx] = tensor.to(dtype=target_dtype)
        else:
            # Convert array-like to tensor first, then to target precision
            results[layer_idx] = torch.tensor(tensor, dtype=target_dtype)

    return results


@dataclass
class SelectiveCacheResult:
    """Container for TransformerLens selective cache extraction results.

    This provides a memory-efficient alternative to caching all activations
    by only storing the hooks specified in names_filter.
    """

    cache: dict[str, torch.Tensor]  # Hook name -> activation tensor
    logits: torch.Tensor | None  # Model output logits (if not stopped early)
    hooks_cached: list[str]  # List of hook names that were cached
    stopped_at_layer: int | None  # Layer where processing stopped (if using stop_at_layer)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_residual_stream(self, layer: int) -> torch.Tensor | None:
        """Get the residual stream hidden state at the specified layer.

        Parameters:
            layer (int): Layer index to retrieve.

        Returns:
            torch.Tensor | None: The residual stream tensor, or None if not cached.
        """
        hook_name = f"blocks.{layer}.hook_resid_post"
        return self.cache.get(hook_name)

    def get_attention_pattern(self, layer: int) -> torch.Tensor | None:
        """Get the attention pattern at the specified layer.

        Parameters:
            layer (int): Layer index to retrieve.

        Returns:
            torch.Tensor | None: The attention pattern tensor, or None if not cached.
        """
        hook_name = f"blocks.{layer}.attn.hook_pattern"
        return self.cache.get(hook_name)

    def to_hidden_states_dict(self) -> dict[int, torch.Tensor]:
        """Convert cached residual stream activations to layer-indexed dict.

        Returns:
            dict[int, torch.Tensor]: Mapping from layer index to hidden state tensor,
                only including residual stream hooks (hook_resid_post).
        """
        result: dict[int, torch.Tensor] = {}
        for hook_name, tensor in self.cache.items():
            if "hook_resid_post" in hook_name:
                # Extract layer number from hook name like 'blocks.5.hook_resid_post'
                parts = hook_name.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    result[layer_idx] = tensor
        return result


def build_hook_filter(
    layers: list[int] | None = None,
    hook_types: list[str] | None = None,
    n_layers: int | None = None,
) -> list[str]:
    """
    Build a list of TransformerLens hook name strings for selective caching.

    This helper constructs hook names in the correct format for use with
    extract_with_selective_cache(). Returns a list of strings, not a lambda.

    Parameters:
        layers (list[int] | None): Specific layer indices to cache. If None
            and n_layers is provided, caches all layers [0, n_layers).
        hook_types (list[str] | None): Types of hooks to cache. Defaults to
            ['hook_resid_post']. Common types:
            - 'hook_resid_post': Residual stream after layer (most common)
            - 'hook_resid_pre': Residual stream before layer
            - 'attn.hook_pattern': Attention patterns
            - 'attn.hook_result': Attention output
            - 'mlp.hook_post': MLP output
        n_layers (int | None): Total number of layers in model. Required if
            layers is None to cache all layers.

    Returns:
        list[str]: List of hook name strings suitable for names_filter parameter.

    Raises:
        ValueError: If neither layers nor n_layers is provided.

    Example:
        >>> # Cache residual stream at layers 0 and 5
        >>> hooks = build_hook_filter(layers=[0, 5])
        >>> # ['blocks.0.hook_resid_post', 'blocks.5.hook_resid_post']

        >>> # Cache attention patterns and residual at layer 3
        >>> hooks = build_hook_filter(
        ...     layers=[3],
        ...     hook_types=['hook_resid_post', 'attn.hook_pattern']
        ... )
        >>> # ['blocks.3.hook_resid_post', 'blocks.3.attn.hook_pattern']
    """
    if layers is None and n_layers is None:
        raise ValueError("Must provide either 'layers' or 'n_layers'")

    if layers is None:
        layers = list(range(n_layers))  # type: ignore[arg-type]

    if hook_types is None:
        hook_types = ["hook_resid_post"]

    hook_names: list[str] = []
    for layer in layers:
        for hook_type in hook_types:
            hook_names.append(f"blocks.{layer}.{hook_type}")

    return hook_names


def extract_with_selective_cache(
    model: HookedTransformer,
    tokens: torch.Tensor,
    names_filter: list[str] | None = None,
    stop_at_layer: int | None = None,
    precision: str | None = None,
) -> SelectiveCacheResult:
    """
    Extract hidden states using TransformerLens selective caching.

    This function provides memory-efficient hidden state extraction by only
    caching the specified hooks rather than all activations (default TransformerLens
    behavior which causes 2-3x memory overhead).

    IMPORTANT: names_filter accepts a list of hook name STRINGS, not lambda functions.
    This is by design to ensure predictable memory usage and serializable configs.

    Parameters:
        model (HookedTransformer): TransformerLens model to run.
        tokens (torch.Tensor): Input token IDs of shape [batch, seq_len].
        names_filter (list[str] | None): List of hook names to cache. If None,
            defaults to caching residual stream at all layers. Common patterns:
            - 'blocks.{layer}.hook_resid_post': Residual stream after layer
            - 'blocks.{layer}.attn.hook_pattern': Attention patterns
            - 'blocks.{layer}.hook_resid_pre': Residual stream before layer
            Example: names_filter=['blocks.0.hook_resid_post', 'blocks.5.hook_resid_post']
        stop_at_layer (int | None): If specified, stop computation at this layer
            to save memory when only early layers are needed. The model will not
            compute layers >= stop_at_layer.
        precision (str | None): Target precision for cached tensors. If specified,
            cached tensors will be converted. Supported: 'fp16', 'fp32', 'bf16'.

    Returns:
        SelectiveCacheResult: Container with cached activations, logits (if computed),
            and metadata about what was cached.

    Raises:
        ImportError: If TransformerLens is not installed.
        ValueError: If invalid hook names are provided.

    Example:
        >>> from transformer_lens import HookedTransformer
        >>> model = HookedTransformer.from_pretrained("gpt2")
        >>> tokens = model.to_tokens("Hello world")
        >>> # Cache only layer 0 and layer 5 residual streams
        >>> result = extract_with_selective_cache(
        ...     model, tokens,
        ...     names_filter=['blocks.0.hook_resid_post', 'blocks.5.hook_resid_post'],
        ...     stop_at_layer=6  # Don't compute layers 6+
        ... )
        >>> layer_0_hidden = result.get_residual_stream(0)
    """
    try:
        from transformer_lens import HookedTransformer as TL_HookedTransformer
    except ImportError as e:
        raise ImportError(
            "TransformerLens is required for selective caching. "
            "Install with: pip install transformer-lens"
        ) from e

    # Build default names_filter if not provided (all residual stream positions)
    if names_filter is None:
        # Cache all residual stream post-layer activations
        n_layers = model.cfg.n_layers
        names_filter = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    # Validate that names_filter is a list of strings (not a lambda)
    if not isinstance(names_filter, list):
        raise ValueError(
            f"names_filter must be a list of hook name strings, not {type(names_filter).__name__}. "
            "Example: names_filter=['blocks.0.hook_resid_post']"
        )

    for hook_name in names_filter:
        if not isinstance(hook_name, str):
            raise ValueError(
                f"All items in names_filter must be strings, got {type(hook_name).__name__}. "
                "Example: names_filter=['blocks.0.hook_resid_post']"
            )

    # Run with selective caching - use inference mode for memory efficiency
    with torch.set_grad_enabled(False):
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=names_filter,
            stop_at_layer=stop_at_layer,
        )

    # Extract cached tensors into a plain dict
    cached_tensors: dict[str, torch.Tensor] = {}
    hooks_actually_cached: list[str] = []

    for hook_name in names_filter:
        if hook_name in cache:
            tensor = cache[hook_name]

            # Apply precision conversion if requested
            if precision is not None and precision in PRECISION_DTYPE_MAP:
                tensor = tensor.to(dtype=PRECISION_DTYPE_MAP[precision])

            cached_tensors[hook_name] = tensor
            hooks_actually_cached.append(hook_name)

    return SelectiveCacheResult(
        cache=cached_tensors,
        logits=logits if stop_at_layer is None else None,
        hooks_cached=hooks_actually_cached,
        stopped_at_layer=stop_at_layer,
        metadata={
            "requested_hooks": names_filter,
            "precision": precision,
            "n_tokens": tokens.shape[-1] if tokens.dim() > 0 else 0,
        },
    )


def extract_hidden_states(
    hidden_states_dict: dict[int, torch.Tensor],
    normalize: bool = False,
) -> dict[int, HiddenStateResult]:
    """
    Convert a mapping of layer indices to tensor/array-like hidden states into HiddenStateResult objects.
    
    Parameters:
        hidden_states_dict (dict[int, torch.Tensor | array-like]): Mapping from layer index to a tensor or array-like hidden state. Each value will be converted to a NumPy array and any leading batch dimension will be removed via squeeze.
        normalize (bool): If True, return L2-normalized vectors in the resulting HiddenStateResult objects.
    
    Returns:
        dict[int, HiddenStateResult]: Mapping from layer index to the corresponding HiddenStateResult containing the (optionally normalized) vector, its shape, layer index, dtype, and metadata.
    """
    results: dict[int, HiddenStateResult] = {}

    for layer_idx, tensor in hidden_states_dict.items():
        # Convert to numpy (convert bfloat16 to float32 first since numpy doesn't support bf16)
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                vector = tensor.cpu().float().numpy()
                dtype_str = "float32"
            else:
                vector = tensor.cpu().numpy()
                dtype_str = str(tensor.dtype).replace("torch.", "")
        else:
            vector = np.array(tensor)
            dtype_str = str(vector.dtype)

        result = HiddenStateResult(
            vector=vector.squeeze(),  # Remove batch dimension if present
            shape=tuple(vector.shape),
            layer=layer_idx,
            dtype=dtype_str,
        )

        if normalize:
            result = result.l2_normalize()

        results[layer_idx] = result

    return results


def compute_d_eff(
    embeddings: np.ndarray,
    variance_threshold: float = 0.90,
) -> int:
    """
    Compute the effective dimensionality (D_eff) of a set of embeddings using PCA.
    
    D_eff is the smallest number of principal components whose cumulative variance meets or exceeds the given variance_threshold. Each embedding row is L2-normalized and mean-centered before variance is computed.
    
    Parameters:
        embeddings (np.ndarray): Array of shape [n_samples, hidden_dim] or a 1-D array [hidden_dim].
        variance_threshold (float): Cumulative variance fraction to reach (e.g., 0.90).
    
    Returns:
        int: Number of dimensions required to capture at least `variance_threshold` of the variance (clamped to the feature dimension).
    """
    # Handle single vector case
    if embeddings.ndim == 1:
        # Single vector - return dimension count (can't compute variance)
        return embeddings.shape[0]

    if embeddings.shape[0] < 2:
        # Need at least 2 samples for variance
        return embeddings.shape[1]

    # L2 normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    normalized = embeddings / norms

    # Center the data
    centered = normalized - normalized.mean(axis=0)

    # Compute covariance matrix
    n_samples = centered.shape[0]
    cov = centered.T @ centered / (n_samples - 1)

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]  # Sort descending

    # Handle numerical issues
    eigenvalues = np.maximum(eigenvalues, 0)

    # Cumulative variance ratio
    total_var = eigenvalues.sum()
    if total_var == 0:
        return 1

    cumvar = np.cumsum(eigenvalues) / total_var

    # Count dimensions below threshold
    d_eff = int(np.searchsorted(cumvar, variance_threshold) + 1)

    return min(d_eff, embeddings.shape[1])


def compute_beta(
    input_d_eff: int,
    output_d_eff: int,
) -> float:
    """
    Compute the collapse indicator beta measuring the relative change in effective dimensionality.
    
    Parameters:
        input_d_eff (int): Effective dimensionality before processing.
        output_d_eff (int): Effective dimensionality after processing.
    
    Returns:
        float: The ratio input_d_eff / output_d_eff; returns `float('inf')` if output_d_eff is 0 to indicate complete collapse.
    """
    if output_d_eff == 0:
        return float("inf")  # Complete collapse
    return input_d_eff / output_d_eff


def compute_geometric_alignment(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """
    Measure geometric alignment between two embeddings by computing their cosine similarity.
    
    Inputs are flattened to 1-D; if either vector has zero L2 norm, the function returns 0.0.
    
    Parameters:
        embedding_a (np.ndarray): First embedding; will be flattened before computation.
        embedding_b (np.ndarray): Second embedding; will be flattened before computation.
    
    Returns:
        float: Cosine similarity in [-1, 1]; `0.0` if either input has zero L2 norm.
    """
    # Flatten if needed
    a = embedding_a.flatten()
    b = embedding_b.flatten()

    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze_hidden_state(
    hidden_state: HiddenStateResult,
) -> dict[str, Any]:
    """
    Compute diagnostic statistics for a HiddenStateResult.
    
    Parameters:
        hidden_state (HiddenStateResult): Container holding the hidden-state vector and related metadata.
    
    Returns:
        dict[str, Any]: Mapping of computed metrics including:
            - `shape`: original vector shape.
            - `layer`: layer index.
            - `dtype`: original data type as a string.
            - `mean`, `std`, `min`, `max`: basic summary statistics.
            - `l2_norm`: Euclidean norm of the flattened vector.
            - `sparsity`: fraction of elements with absolute value less than 1e-6.
            - `percentile_25`, `percentile_50`, `percentile_75`: quartile values when the vector contains at least one element.
    """
    vector = hidden_state.vector.flatten()

    # Handle empty vector case (prevents ValueError from np.min/np.max on empty arrays)
    if len(vector) == 0:
        return {
            "shape": hidden_state.shape,
            "layer": hidden_state.layer,
            "dtype": hidden_state.dtype,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "l2_norm": 0.0,
            "sparsity": 0.0,
        }

    # Basic statistics
    analysis = {
        "shape": hidden_state.shape,
        "layer": hidden_state.layer,
        "dtype": hidden_state.dtype,
        "mean": float(np.mean(vector)),
        "std": float(np.std(vector)),
        "min": float(np.min(vector)),
        "max": float(np.max(vector)),
        "l2_norm": float(np.linalg.norm(vector)),
        "sparsity": float(np.mean(np.abs(vector) < 1e-6)),  # Fraction near zero
    }

    # Distribution metrics
    analysis["percentile_25"] = float(np.percentile(vector, 25))
    analysis["percentile_50"] = float(np.percentile(vector, 50))
    analysis["percentile_75"] = float(np.percentile(vector, 75))

    return analysis


@dataclass
class StreamingChunkResult:
    """Container for a single chunk result from streaming extraction.

    This dataclass holds the hidden states extracted from a single chunk
    of a long sequence, along with metadata about the chunk's position
    within the full sequence.
    """

    chunk_index: int  # Index of this chunk (0-based)
    start_position: int  # Start token position in original sequence
    end_position: int  # End token position (exclusive) in original sequence
    hidden_states: dict[int, torch.Tensor]  # Layer index -> hidden state tensor
    is_last_chunk: bool  # Whether this is the final chunk
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_length(self) -> int:
        """Return the number of tokens in this chunk."""
        return self.end_position - self.start_position

    def to_hidden_state_results(
        self,
        normalize: bool = False,
    ) -> dict[int, HiddenStateResult]:
        """Convert hidden state tensors to HiddenStateResult objects.

        Parameters:
            normalize (bool): If True, L2-normalize the resulting vectors.

        Returns:
            dict[int, HiddenStateResult]: Mapping from layer index to
                HiddenStateResult containing the hidden state vector.
        """
        return extract_hidden_states(self.hidden_states, normalize=normalize)


def extract_streaming(
    model: HookedTransformer,
    tokens: torch.Tensor,
    chunk_size: int = 512,
    overlap: int = 0,
    layers: list[int] | None = None,
    precision: str | None = None,
    clear_cache_between_chunks: bool = True,
) -> Generator[StreamingChunkResult, None, None]:
    """
    Stream hidden state extraction for long sequences in memory-efficient chunks.

    This generator function processes long token sequences incrementally without
    loading the entire context into GPU memory at once. It yields results for
    each chunk, enabling processing of sequences that exceed available GPU memory.

    Parameters:
        model (HookedTransformer): TransformerLens model to run.
        tokens (torch.Tensor): Input token IDs of shape [batch, seq_len] or [seq_len].
            For batch dimension > 1, each batch element is processed independently.
        chunk_size (int): Number of tokens to process per chunk. Default is 512.
            Smaller values reduce memory usage but increase processing time.
        overlap (int): Number of tokens to overlap between consecutive chunks.
            Useful for maintaining context continuity. Default is 0.
            Must be less than chunk_size.
        layers (list[int] | None): Specific layer indices to extract hidden states
            from. If None, extracts from all layers. Use to reduce memory usage.
        precision (str | None): Target precision for extracted tensors. Supported:
            'fp16', 'fp32', 'bf16'. If None, uses model's native precision.
        clear_cache_between_chunks (bool): If True, clears GPU cache between chunks
            to prevent memory fragmentation. Default is True.

    Yields:
        StreamingChunkResult: Container with hidden states for each chunk, including
            position metadata and whether it's the final chunk.

    Raises:
        ValueError: If overlap >= chunk_size or if chunk_size <= 0.
        ImportError: If TransformerLens is not installed.

    Example:
        >>> from transformer_lens import HookedTransformer
        >>> model = HookedTransformer.from_pretrained("gpt2")
        >>> tokens = model.to_tokens("Very long text..." * 1000)  # Long sequence
        >>>
        >>> # Process in chunks of 512 tokens
        >>> for chunk_result in extract_streaming(model, tokens, chunk_size=512):
        ...     print(f"Chunk {chunk_result.chunk_index}: "
        ...           f"positions {chunk_result.start_position}-{chunk_result.end_position}")
        ...     # Process chunk hidden states
        ...     layer_0_hidden = chunk_result.hidden_states.get(0)

    Note:
        - Each chunk is processed independently; attention patterns within a chunk
          do not see tokens from other chunks.
        - For causal models, this is appropriate as future tokens shouldn't influence
          past token representations.
        - Use overlap > 0 if you need context continuity between chunks.
        - Memory usage scales with chunk_size * model_hidden_dim * num_layers.
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    try:
        from transformer_lens import HookedTransformer as TL_HookedTransformer
    except ImportError as e:
        raise ImportError(
            "TransformerLens is required for streaming extraction. "
            "Install with: pip install transformer-lens"
        ) from e

    # Ensure tokens has batch dimension
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    batch_size, seq_len = tokens.shape

    # Build hook filter for selective caching
    n_layers = model.cfg.n_layers
    if layers is None:
        target_layers = list(range(n_layers))
    else:
        target_layers = layers

    names_filter = build_hook_filter(layers=target_layers)

    # Calculate step size (accounting for overlap)
    step_size = chunk_size - overlap

    # Process tokens in chunks
    chunk_index = 0
    position = 0

    while position < seq_len:
        # Determine chunk boundaries
        start_pos = position
        end_pos = min(position + chunk_size, seq_len)
        is_last = end_pos >= seq_len

        # Extract chunk tokens
        chunk_tokens = tokens[:, start_pos:end_pos]

        # Run model with selective caching for memory efficiency
        with torch.set_grad_enabled(False):
            _, cache = model.run_with_cache(
                chunk_tokens,
                names_filter=names_filter,
            )

        # Extract hidden states from cache
        hidden_states: dict[int, torch.Tensor] = {}
        for layer_idx in target_layers:
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            if hook_name in cache:
                tensor = cache[hook_name]

                # Apply precision conversion if requested
                if precision is not None and precision in PRECISION_DTYPE_MAP:
                    tensor = tensor.to(dtype=PRECISION_DTYPE_MAP[precision])

                hidden_states[layer_idx] = tensor

        # Create result for this chunk
        result = StreamingChunkResult(
            chunk_index=chunk_index,
            start_position=start_pos,
            end_position=end_pos,
            hidden_states=hidden_states,
            is_last_chunk=is_last,
            metadata={
                "batch_size": batch_size,
                "total_seq_len": seq_len,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "precision": precision,
                "layers_extracted": target_layers,
            },
        )

        yield result

        # Clear GPU cache between chunks to prevent fragmentation
        if clear_cache_between_chunks and not is_last:
            # Delete cache reference to allow garbage collection
            del cache
            torch.cuda.empty_cache()

        # Move to next chunk position
        if is_last:
            break
        position += step_size
        chunk_index += 1


def collect_streaming_results(
    streaming_generator: Generator[StreamingChunkResult, None, None],
    aggregate_method: str = "concat",
) -> dict[int, torch.Tensor]:
    """
    Collect all chunks from a streaming extraction into a single result.

    This convenience function consumes the streaming generator and combines
    the chunk results into a single dictionary of hidden states. Use this
    when you need the full sequence hidden states but still want memory-efficient
    chunk-by-chunk processing.

    Parameters:
        streaming_generator (Generator[StreamingChunkResult, None, None]):
            Generator from extract_streaming().
        aggregate_method (str): How to combine chunk hidden states. Currently
            supported: 'concat' (concatenate along sequence dimension).

    Returns:
        dict[int, torch.Tensor]: Mapping from layer index to hidden state tensor
            containing the full sequence. Shape is [batch, total_seq_len, hidden_dim].

    Example:
        >>> generator = extract_streaming(model, tokens, chunk_size=512)
        >>> full_hidden_states = collect_streaming_results(generator)
        >>> layer_0_full = full_hidden_states[0]  # [batch, seq_len, hidden_dim]
    """
    if aggregate_method != "concat":
        raise ValueError(
            f"Unsupported aggregate_method '{aggregate_method}'. "
            "Currently only 'concat' is supported."
        )

    layer_chunks: dict[int, list[torch.Tensor]] = {}
    overlap = 0  # Will be set from first chunk's metadata

    for chunk_result in streaming_generator:
        # Get overlap from metadata (set on first chunk, consistent across all)
        if chunk_result.chunk_index == 0:
            overlap = chunk_result.metadata.get("overlap", 0)

        for layer_idx, tensor in chunk_result.hidden_states.items():
            if layer_idx not in layer_chunks:
                layer_chunks[layer_idx] = []

            # For chunks after the first, slice off the overlap region
            # to avoid duplicating tokens in the concatenated result
            if chunk_result.chunk_index > 0 and overlap > 0:
                # Slice from overlap position onward (dim=1 is sequence dimension)
                tensor = tensor[:, overlap:, ...]

            layer_chunks[layer_idx].append(tensor)

    # Concatenate chunks along sequence dimension (dim=1)
    result: dict[int, torch.Tensor] = {}
    for layer_idx, chunks in layer_chunks.items():
        result[layer_idx] = torch.cat(chunks, dim=1)

    return result


@dataclass
class TrainingExtractionResult:
    """Container for training-mode hidden state extraction results.

    This dataclass holds the results of running a model with gradients enabled,
    suitable for use in training loops where backward passes will be performed.

    Note:
        This does NOT automatically enable gradient checkpointing. To benefit
        from gradient checkpointing memory savings, configure it at the model
        level (e.g., model.cfg.use_checkpointing = True) before extraction.
    """

    hidden_states: dict[int, torch.Tensor]  # Layer index -> hidden state tensor
    logits: torch.Tensor | None  # Model output logits
    gradients_enabled: bool  # Whether gradients were enabled during extraction
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_hidden_states_dict(self) -> dict[int, torch.Tensor]:
        """Return the hidden states dictionary.

        Returns:
            dict[int, torch.Tensor]: Mapping from layer index to hidden state tensor.
        """
        return self.hidden_states

    def to_hidden_state_results(
        self,
        normalize: bool = False,
    ) -> dict[int, HiddenStateResult]:
        """Convert hidden state tensors to HiddenStateResult objects.

        Parameters:
            normalize (bool): If True, L2-normalize the resulting vectors.

        Returns:
            dict[int, HiddenStateResult]: Mapping from layer index to
                HiddenStateResult containing the hidden state vector.
        """
        return extract_hidden_states(self.hidden_states, normalize=normalize)


def extract_for_training(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layers: list[int] | None = None,
    precision: str | None = None,
) -> TrainingExtractionResult:
    """
    Extract hidden states with gradients enabled for use in training loops.

    This function extracts hidden states while preserving gradient computation,
    making the results suitable for training workflows where backward passes
    will be performed. The extracted tensors retain their computation graph.

    IMPORTANT: This function does NOT automatically enable gradient checkpointing.
    To benefit from gradient checkpointing memory savings (~50% reduction with
    ~30% compute overhead), configure it at the model level before calling:

        model.cfg.use_checkpointing = True  # Enable before extraction

    For pure inference without backpropagation, use extract_inference_optimized()
    instead, which disables gradients for better memory efficiency.

    Parameters:
        model (HookedTransformer): TransformerLens model to run.
        tokens (torch.Tensor): Input token IDs of shape [batch, seq_len] or [seq_len].
        layers (list[int] | None): Specific layer indices to extract hidden states
            from. If None, extracts from all layers.
        precision (str | None): Target precision for extracted tensors. Supported:
            'fp16', 'fp32', 'bf16'. If None, uses model's native precision.

    Returns:
        TrainingExtractionResult: Container with hidden states, logits, and metadata
            indicating whether gradients were enabled.

    Raises:
        ImportError: If TransformerLens is not installed.

    Example:
        >>> from transformer_lens import HookedTransformer
        >>> import torch
        >>>
        >>> model = HookedTransformer.from_pretrained("gpt2")
        >>> tokens = model.to_tokens("Training example text")
        >>>
        >>> # Optional: Enable gradient checkpointing at model level
        >>> # model.cfg.use_checkpointing = True
        >>>
        >>> # Extract with gradients for training
        >>> result = extract_for_training(model, tokens, layers=[0, 5, 11])
        >>>
        >>> # Use in training loop with backward pass
        >>> loss = compute_loss(result.hidden_states)
        >>> loss.backward()

    Note:
        - Gradients are preserved based on torch.is_grad_enabled() context
        - For gradient checkpointing benefits, set model.cfg.use_checkpointing = True
        - For inference, use extract_inference_optimized() instead
    """
    try:
        from transformer_lens import HookedTransformer as TL_HookedTransformer
    except ImportError as e:
        raise ImportError(
            "TransformerLens is required for training extraction. "
            "Install with: pip install transformer-lens"
        ) from e

    # Ensure tokens has batch dimension
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    # Determine target layers
    n_layers = model.cfg.n_layers
    if layers is None:
        target_layers = list(range(n_layers))
    else:
        target_layers = layers

    # Build hook filter for selective caching
    names_filter = build_hook_filter(layers=target_layers)

    # Check if gradients are enabled (training mode)
    gradients_enabled = torch.is_grad_enabled()

    # Run forward pass with caching, preserving gradient state
    # For gradient checkpointing benefits, configure model.cfg.use_checkpointing
    # before calling this function
    with torch.set_grad_enabled(gradients_enabled):
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=names_filter,
            return_type="logits",
        )

    # Extract hidden states from cache
    hidden_states: dict[int, torch.Tensor] = {}
    for layer_idx in target_layers:
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        if hook_name in cache:
            tensor = cache[hook_name]

            # Apply precision conversion if requested
            if precision is not None and precision in PRECISION_DTYPE_MAP:
                tensor = tensor.to(dtype=PRECISION_DTYPE_MAP[precision])

            hidden_states[layer_idx] = tensor

    return TrainingExtractionResult(
        hidden_states=hidden_states,
        logits=logits,
        gradients_enabled=gradients_enabled,
        metadata={
            "layers_extracted": target_layers,
            "precision": precision,
            "n_tokens": tokens.shape[-1] if tokens.dim() > 0 else 0,
            "gradients_enabled": gradients_enabled,
        },
    )


@dataclass
class InferenceResult:
    """Container for inference-optimized hidden state extraction results.

    This dataclass holds the results of running a model in inference mode
    with explicit gradient disabling for maximum memory efficiency during
    pure inference workloads.
    """

    hidden_states: dict[int, torch.Tensor]  # Layer index -> hidden state tensor
    logits: torch.Tensor | None  # Model output logits
    inference_mode_used: bool  # Whether torch.inference_mode was used
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_hidden_states_dict(self) -> dict[int, torch.Tensor]:
        """Return the hidden states dictionary.

        Returns:
            dict[int, torch.Tensor]: Mapping from layer index to hidden state tensor.
        """
        return self.hidden_states

    def to_hidden_state_results(
        self,
        normalize: bool = False,
    ) -> dict[int, HiddenStateResult]:
        """Convert hidden state tensors to HiddenStateResult objects.

        Parameters:
            normalize (bool): If True, L2-normalize the resulting vectors.

        Returns:
            dict[int, HiddenStateResult]: Mapping from layer index to
                HiddenStateResult containing the hidden state vector.
        """
        return extract_hidden_states(self.hidden_states, normalize=normalize)


def extract_inference_optimized(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layers: list[int] | None = None,
    precision: str | None = None,
    use_inference_mode: bool = True,
    clear_cache: bool = False,
) -> InferenceResult:
    """
    Extract hidden states with explicit gradient disabling for memory-efficient inference.

    This function provides maximum memory efficiency for pure inference workloads
    by using torch.inference_mode() (or torch.no_grad() as fallback). Unlike
    gradient checkpointing which is designed for training with backward passes,
    this function is optimized for scenarios where no gradients are needed.

    Key optimizations:
    - Uses torch.inference_mode() for more efficient inference than no_grad()
    - Explicitly disables gradient computation and storage
    - Selective caching to only store requested layer activations
    - Optional CUDA cache clearing to reduce memory fragmentation

    Parameters:
        model (HookedTransformer): TransformerLens model to run.
        tokens (torch.Tensor): Input token IDs of shape [batch, seq_len] or [seq_len].
        layers (list[int] | None): Specific layer indices to extract hidden states
            from. If None, extracts from all layers.
        precision (str | None): Target precision for extracted tensors. Supported:
            'fp16', 'fp32', 'bf16'. If None, uses model's native precision.
        use_inference_mode (bool): If True (default), uses torch.inference_mode()
            which is more efficient than torch.no_grad(). Set to False for
            compatibility with older PyTorch versions or specific use cases.
        clear_cache (bool): If True, clears CUDA cache before and after extraction
            to reduce memory fragmentation. Default is False to avoid overhead.

    Returns:
        InferenceResult: Container with hidden states, logits, and metadata
            about the inference configuration used.

    Raises:
        ImportError: If TransformerLens is not installed.

    Example:
        >>> from transformer_lens import HookedTransformer
        >>> model = HookedTransformer.from_pretrained("gpt2")
        >>> tokens = model.to_tokens("Example text for inference")
        >>>
        >>> # Extract hidden states in inference mode
        >>> result = extract_inference_optimized(
        ...     model, tokens,
        ...     layers=[0, 5, 11],
        ...     precision='fp16',  # Use half precision for memory savings
        ... )
        >>>
        >>> # Access results
        >>> layer_5_hidden = result.hidden_states[5]
        >>> print(f"Inference mode used: {result.inference_mode_used}")

    Note:
        - torch.inference_mode() is more efficient than torch.no_grad() because
          it also disables version tracking for autograd.
        - This function should be used for all pure inference workloads.
        - For training with backward passes, use extract_for_training() instead.
        - Combines well with precision='fp16' for additional memory savings.
    """
    try:
        from transformer_lens import HookedTransformer as TL_HookedTransformer
    except ImportError as e:
        raise ImportError(
            "TransformerLens is required for inference-optimized extraction. "
            "Install with: pip install transformer-lens"
        ) from e

    # Clear GPU cache before extraction if requested
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Ensure tokens has batch dimension
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    # Determine target layers
    n_layers = model.cfg.n_layers
    if layers is None:
        target_layers = list(range(n_layers))
    else:
        target_layers = layers

    # Build hook filter for selective caching
    names_filter = build_hook_filter(layers=target_layers)

    # Choose the appropriate gradient disabling context
    # torch.inference_mode() is more efficient than torch.no_grad()
    # as it also disables autograd's version tracking
    if use_inference_mode:
        context_manager = torch.inference_mode()
        inference_mode_used = True
    else:
        context_manager = torch.no_grad()
        inference_mode_used = False

    # Run model with explicit gradient disabling for memory efficiency
    with context_manager:
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=names_filter,
        )

        # Extract hidden states from cache (still within context for efficiency)
        hidden_states: dict[int, torch.Tensor] = {}
        for layer_idx in target_layers:
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            if hook_name in cache:
                tensor = cache[hook_name]

                # Apply precision conversion if requested
                if precision is not None and precision in PRECISION_DTYPE_MAP:
                    tensor = tensor.to(dtype=PRECISION_DTYPE_MAP[precision])

                hidden_states[layer_idx] = tensor

    # Clear GPU cache after extraction if requested
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return InferenceResult(
        hidden_states=hidden_states,
        logits=logits,
        inference_mode_used=inference_mode_used,
        metadata={
            "layers_extracted": target_layers,
            "precision": precision,
            "use_inference_mode": use_inference_mode,
            "clear_cache": clear_cache,
            "n_tokens": tokens.shape[-1] if tokens.dim() > 0 else 0,
        },
    )


# Backward compatibility aliases (deprecated)
# These will be removed in a future version
CheckpointingResult = TrainingExtractionResult
"""Deprecated alias for TrainingExtractionResult. Use TrainingExtractionResult instead."""

extract_with_checkpointing = extract_for_training
"""Deprecated alias for extract_for_training. Use extract_for_training instead."""
