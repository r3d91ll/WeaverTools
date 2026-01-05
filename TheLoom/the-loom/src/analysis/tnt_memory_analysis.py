"""TNT Olympian Memory Analysis for Conveyance Hypothesis Validation.

This module provides analysis functions for TNT Olympian memory states,
computing metrics relevant to the Conveyance Hypothesis:

- D_eff (Effective Dimensionality): Semantic richness via PCA
- Beta (Collapse Indicator): Ratio of top singular values
- Weight Norms: Memory capacity utilization
- Buffer Utilization: Context window usage

TNT Olympian Memory Structure:
    Per LocalMemoryModule:
        - W1, W2: Memory MLP weights
        - k_buffer, v_buffer: Context window buffers
        - momentum_W1, momentum_W2: Newton-Schulz momentum

    Per TNTHierarchicalMemory:
        - global: GlobalMemoryModule state
        - local: [LocalMemoryModule state x 4]
        - position: Position in sequence

    Full checkpoint memory_state:
        - block_0, block_1, ...: TNTHierarchicalMemory states

References:
    - Conveyance Hypothesis v4.1: D_eff and beta metrics
    - TNT (arXiv:2511.07343): Hierarchical memory architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Analysis constants
DEFAULT_VARIANCE_THRESHOLD = 0.90  # For D_eff calculation
SPARSITY_THRESHOLD = 1e-6  # For weight sparsity
TOP_SINGULAR_VALUES = 10  # Number of SVs to track


@dataclass
class LocalMemoryMetrics:
    """Metrics for a single LocalMemoryModule.

    Attributes:
        chunk_size: Processing chunk size for this memory.
        weight_norm_w1: Frobenius norm of W1 weight matrix.
        weight_norm_w2: Frobenius norm of W2 weight matrix.
        momentum_norm_w1: Norm of W1 momentum (None if not present).
        momentum_norm_w2: Norm of W2 momentum (None if not present).
        buffer_utilization: Fraction of context buffer filled (0-1).
        d_eff: Effective dimensionality of W1 (PCA-based).
        beta: Collapse indicator (ratio of top 2 singular values).
        sparsity_w1: Fraction of near-zero entries in W1.
        sparsity_w2: Fraction of near-zero entries in W2.
    """

    chunk_size: int
    weight_norm_w1: float
    weight_norm_w2: float
    momentum_norm_w1: float | None
    momentum_norm_w2: float | None
    buffer_utilization: float
    d_eff: int
    beta: float
    sparsity_w1: float = 0.0
    sparsity_w2: float = 0.0
    top_singular_values: list[float] = field(default_factory=list)


@dataclass
class BlockMemoryMetrics:
    """Metrics for a single TNTOlympianBlock's hierarchical memory.

    Attributes:
        block_idx: Block index (0-indexed).
        global_memory: Metrics for the global memory module.
        local_memories: Metrics for each local memory (4 by default).
        position: Current position in sequence.
        combined_d_eff: Average D_eff across all memories.
        combined_beta: Average beta across all memories.
    """

    block_idx: int
    global_memory: LocalMemoryMetrics
    local_memories: list[LocalMemoryMetrics]
    position: int
    combined_d_eff: float = 0.0
    combined_beta: float = 0.0

    def __post_init__(self) -> None:
        """Compute combined metrics."""
        all_memories = [self.global_memory] + self.local_memories
        self.combined_d_eff = float(np.mean([m.d_eff for m in all_memories]))
        self.combined_beta = float(np.mean([m.beta for m in all_memories]))


@dataclass
class TNTMemoryAnalysis:
    """Complete memory analysis for a TNT Olympian model.

    Attributes:
        blocks: Metrics per block (block_0, block_1, ...).
        aggregate_d_eff: Weighted average D_eff across all blocks.
        aggregate_beta: Weighted average beta across all blocks.
        total_weight_norm: Sum of all weight norms.
        health_status: Overall memory health assessment.
        metadata: Additional analysis information.
    """

    blocks: dict[str, BlockMemoryMetrics]
    aggregate_d_eff: float
    aggregate_beta: float
    total_weight_norm: float
    health_status: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_d_eff(
    weight: torch.Tensor | NDArray[np.floating[Any]],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> int:
    """Compute effective dimensionality via PCA.

    D_eff is the number of principal components needed to explain
    variance_threshold (default 90%) of the variance.

    Target for healthy memory: D_eff >= 20 for 512-dim models.

    Args:
        weight: Weight matrix (2D tensor or array).
        variance_threshold: Fraction of variance to explain (0-1).

    Returns:
        Number of components explaining variance_threshold.
    """
    # Convert to numpy
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    if weight.ndim != 2:
        logger.warning(f"Expected 2D weight matrix, got shape {weight.shape}")
        return 0

    # Compute SVD
    try:
        singular_values = linalg.svdvals(weight.astype(np.float64))
    except Exception as e:
        logger.warning(f"SVD failed: {e}")
        return 0

    # Compute cumulative variance explained
    variance = singular_values**2
    total_variance = variance.sum()

    if total_variance == 0:
        return 0

    cumulative_variance = np.cumsum(variance) / total_variance

    # Find number of components for threshold
    d_eff = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)

    return min(d_eff, len(singular_values))


def compute_beta(
    weight: torch.Tensor | NDArray[np.floating[Any]],
) -> float:
    """Compute collapse indicator beta.

    Beta is the ratio of the largest to second-largest singular value.
    High beta indicates dimensional collapse (bad).

    Target for healthy memory: beta < 2.0.

    Args:
        weight: Weight matrix (2D tensor or array).

    Returns:
        Collapse indicator (larger = more collapsed).
    """
    # Convert to numpy
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    if weight.ndim != 2:
        logger.warning(f"Expected 2D weight matrix, got shape {weight.shape}")
        return float("inf")

    # Compute SVD
    try:
        singular_values = linalg.svdvals(weight.astype(np.float64))
    except Exception as e:
        logger.warning(f"SVD failed: {e}")
        return float("inf")

    if len(singular_values) < 2:
        return float("inf")

    # Beta = sv1 / sv2
    if singular_values[1] > 1e-10:
        return float(singular_values[0] / singular_values[1])

    return float("inf")


def compute_weight_norm(
    weight: torch.Tensor | NDArray[np.floating[Any]],
) -> float:
    """Compute Frobenius norm of weight matrix.

    Args:
        weight: Weight matrix.

    Returns:
        Frobenius norm.
    """
    if isinstance(weight, torch.Tensor):
        return float(torch.norm(weight).item())
    return float(np.linalg.norm(weight))


def compute_sparsity(
    weight: torch.Tensor | NDArray[np.floating[Any]],
    threshold: float = SPARSITY_THRESHOLD,
) -> float:
    """Compute fraction of near-zero entries.

    Args:
        weight: Weight matrix.
        threshold: Threshold for "near-zero".

    Returns:
        Fraction of entries below threshold.
    """
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    near_zero = np.abs(weight) < threshold
    return float(near_zero.sum() / weight.size)


def analyze_local_memory(
    state: dict[str, Any],
    chunk_size: int = 0,
    context_size: int = 64,
) -> LocalMemoryMetrics:
    """Analyze a single LocalMemoryModule state.

    Args:
        state: LocalMemoryModule state dict with W1, W2, buffers, etc.
        chunk_size: Processing chunk size for this memory.
        context_size: Context window size for buffer utilization.

    Returns:
        LocalMemoryMetrics with all computed metrics.
    """
    # Extract weight matrices
    w1 = state.get("W1")
    w2 = state.get("W2")

    if w1 is None or w2 is None:
        logger.warning("Missing W1 or W2 in memory state")
        return LocalMemoryMetrics(
            chunk_size=chunk_size,
            weight_norm_w1=0.0,
            weight_norm_w2=0.0,
            momentum_norm_w1=None,
            momentum_norm_w2=None,
            buffer_utilization=0.0,
            d_eff=0,
            beta=float("inf"),
        )

    # Compute weight metrics
    weight_norm_w1 = compute_weight_norm(w1)
    weight_norm_w2 = compute_weight_norm(w2)
    d_eff = compute_d_eff(w1)
    beta = compute_beta(w1)
    sparsity_w1 = compute_sparsity(w1)
    sparsity_w2 = compute_sparsity(w2)

    # Compute momentum norms if present
    momentum_w1 = state.get("momentum_W1")
    momentum_w2 = state.get("momentum_W2")
    momentum_norm_w1 = compute_weight_norm(momentum_w1) if momentum_w1 is not None else None
    momentum_norm_w2 = compute_weight_norm(momentum_w2) if momentum_w2 is not None else None

    # Compute buffer utilization
    buffer_len = state.get("buffer_len", 0)
    buffer_utilization = buffer_len / context_size if context_size > 0 else 0.0

    # Get top singular values
    try:
        if isinstance(w1, torch.Tensor):
            w1_np = w1.detach().cpu().numpy()
        else:
            w1_np = w1
        svs = linalg.svdvals(w1_np.astype(np.float64))[:TOP_SINGULAR_VALUES]
        top_svs = [float(sv) for sv in svs]
    except Exception:
        top_svs = []

    return LocalMemoryMetrics(
        chunk_size=chunk_size,
        weight_norm_w1=weight_norm_w1,
        weight_norm_w2=weight_norm_w2,
        momentum_norm_w1=momentum_norm_w1,
        momentum_norm_w2=momentum_norm_w2,
        buffer_utilization=buffer_utilization,
        d_eff=d_eff,
        beta=beta,
        sparsity_w1=sparsity_w1,
        sparsity_w2=sparsity_w2,
        top_singular_values=top_svs,
    )


def analyze_block_memory(
    block_state: dict[str, Any],
    block_idx: int,
    local_chunk_sizes: list[int] | None = None,
    context_size: int = 64,
) -> BlockMemoryMetrics:
    """Analyze a single block's hierarchical memory.

    Args:
        block_state: TNTHierarchicalMemory state dict.
        block_idx: Block index.
        local_chunk_sizes: Chunk sizes for local memories.
        context_size: Context window size.

    Returns:
        BlockMemoryMetrics for the block.
    """
    if local_chunk_sizes is None:
        local_chunk_sizes = [8, 16, 32, 64]

    # Analyze global memory
    global_state = block_state.get("global", {})
    global_metrics = analyze_local_memory(
        global_state,
        chunk_size=2048,  # Default global chunk size
        context_size=context_size,
    )

    # Analyze local memories
    local_states = block_state.get("local", [])
    local_metrics = []
    for i, local_state in enumerate(local_states):
        chunk_size = local_chunk_sizes[i] if i < len(local_chunk_sizes) else 0
        metrics = analyze_local_memory(local_state, chunk_size, context_size)
        local_metrics.append(metrics)

    # Get position
    position = block_state.get("position", 0)

    return BlockMemoryMetrics(
        block_idx=block_idx,
        global_memory=global_metrics,
        local_memories=local_metrics,
        position=position,
    )


def analyze_tnt_memory(
    model_or_state: Any,
    config: dict[str, Any] | None = None,
) -> TNTMemoryAnalysis:
    """Analyze full TNT Olympian memory state.

    Args:
        model_or_state: Either a TNTOlympian model instance or a memory_state dict.
        config: Optional config dict with local_chunk_sizes, context_size.

    Returns:
        TNTMemoryAnalysis with complete metrics.
    """
    # Extract memory state if model instance
    if hasattr(model_or_state, "get_memory_state"):
        memory_state = model_or_state.get_memory_state()
    else:
        memory_state = model_or_state

    if not isinstance(memory_state, dict):
        raise ValueError(f"Expected dict memory state, got {type(memory_state)}")

    # Get config values
    if config is None:
        config = {}
    local_chunk_sizes = config.get("local_chunk_sizes", [8, 16, 32, 64])
    context_size = config.get("context_size", 64)

    # Analyze each block
    blocks: dict[str, BlockMemoryMetrics] = {}
    total_weight_norm = 0.0
    all_d_effs = []
    all_betas = []

    for key in sorted(memory_state.keys()):
        if not key.startswith("block_"):
            continue

        block_idx = int(key.split("_")[1])
        block_state = memory_state[key]

        block_metrics = analyze_block_memory(
            block_state,
            block_idx,
            local_chunk_sizes,
            context_size,
        )
        blocks[key] = block_metrics

        # Aggregate metrics
        all_memories = [block_metrics.global_memory] + block_metrics.local_memories
        for mem in all_memories:
            total_weight_norm += mem.weight_norm_w1 + mem.weight_norm_w2
            all_d_effs.append(mem.d_eff)
            if mem.beta != float("inf"):
                all_betas.append(mem.beta)

    # Compute aggregate metrics
    aggregate_d_eff = float(np.mean(all_d_effs)) if all_d_effs else 0.0
    aggregate_beta = float(np.mean(all_betas)) if all_betas else float("inf")

    # Determine health status
    if aggregate_beta > 10.0:
        health_status = "critical:high_collapse"
    elif aggregate_d_eff < 10:
        health_status = "warning:low_dimensionality"
    elif aggregate_beta > 2.0:
        health_status = "warning:moderate_collapse"
    elif aggregate_d_eff < 20:
        health_status = "fair:suboptimal_dimensionality"
    else:
        health_status = "healthy"

    return TNTMemoryAnalysis(
        blocks=blocks,
        aggregate_d_eff=aggregate_d_eff,
        aggregate_beta=aggregate_beta,
        total_weight_norm=total_weight_norm,
        health_status=health_status,
        metadata={
            "n_blocks": len(blocks),
            "local_chunk_sizes": local_chunk_sizes,
            "context_size": context_size,
        },
    )


def compare_memory_states(
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two memory states to measure change.

    Useful for analyzing how memory adapts to different inputs
    (concept probing for Conveyance Hypothesis).

    Args:
        state_before: Memory state before processing.
        state_after: Memory state after processing.
        config: Optional config for analysis.

    Returns:
        Comparison results with delta metrics.
    """
    analysis_before = analyze_tnt_memory(state_before, config)
    analysis_after = analyze_tnt_memory(state_after, config)

    return {
        "d_eff_delta": analysis_after.aggregate_d_eff - analysis_before.aggregate_d_eff,
        "beta_delta": analysis_after.aggregate_beta - analysis_before.aggregate_beta,
        "weight_norm_delta": analysis_after.total_weight_norm - analysis_before.total_weight_norm,
        "before": analysis_before,
        "after": analysis_after,
    }


def probe_concept_memory(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
) -> dict[str, TNTMemoryAnalysis]:
    """Run inference on concept prompts and analyze memory states.

    This function is designed for Conveyance Hypothesis validation.
    It processes multiple prompts and returns memory analysis for each,
    allowing comparison of how memory adapts to different concepts.

    Args:
        model: TNTOlympian model instance.
        tokenizer: Tokenizer instance.
        prompts: List of concept prompts to test.

    Returns:
        Dict mapping prompt to TNTMemoryAnalysis.
    """
    results: dict[str, TNTMemoryAnalysis] = {}

    for prompt in prompts:
        # Reset memory between prompts
        model.reset_all_memories()

        # Encode and run inference
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            device=next(model.parameters()).device,
        )

        with torch.no_grad():
            _ = model(input_ids)

        # Analyze resulting memory state
        memory_state = model.get_memory_state()
        analysis = analyze_tnt_memory(memory_state)
        results[prompt] = analysis

    return results


# Convenience function for loading and analyzing
def analyze_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> TNTMemoryAnalysis:
    """Load checkpoint and analyze memory state.

    Convenience function that loads a checkpoint and returns
    memory analysis without keeping the model loaded.

    Args:
        checkpoint_path: Path to TNT Olympian checkpoint.
        device: Device for loading.

    Returns:
        TNTMemoryAnalysis for the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Get memory state and config
    memory_state = checkpoint.get("memory_state", {})
    config = checkpoint.get("config", {})

    if not memory_state:
        raise ValueError("Checkpoint has no memory_state")

    return analyze_tnt_memory(memory_state, config)
