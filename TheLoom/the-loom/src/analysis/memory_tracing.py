"""Memory Episode Tracing for Atlas Model Analysis.

This module implements memory episode tracing for the Atlas model's Titans-style
matrix memory architecture. It extracts and analyzes the M (memory) and S (state)
matrices to understand how episodic memories evolve during training.

ATLAS MEMORY ARCHITECTURE
=========================
The Atlas model uses a matrix memory system inspired by Titans:
- M matrix: Memory content matrix [batch, memory_slots, memory_dim]
- S matrix: Memory state/salience matrix [batch, memory_slots, memory_dim]

Each layer has its own memory state, enabling analysis of:
- How memory content evolves during training
- Which memory slots are most active/salient
- How memory sparsity changes as training progresses
- Whether certain memory patterns correlate with model capabilities

ANALYSIS CAPABILITIES
=====================
1. Memory Statistics:
   - Magnitude: L2 norm and Frobenius norm of memory matrices
   - Sparsity: Fraction of near-zero entries (below threshold)
   - Effective rank: Number of significant singular values
   - Entropy: Information-theoretic measure of memory utilization

2. Cross-Epoch Analysis:
   - Track memory evolution across training epochs
   - Identify epochs where memory reorganization occurs
   - Detect memory consolidation patterns

3. Layer-wise Analysis:
   - Compare memory utilization across layers
   - Identify layers with highest/lowest memory activity
   - Track layer-specific memory patterns

4. Visualization:
   - Memory magnitude heatmaps
   - Sparsity evolution plots
   - Singular value distribution plots
   - Cross-layer comparison charts

INTEGRATION
===========
Works with AtlasLoader for checkpoint loading and memory state extraction.
Memory states are stored in checkpoint['memory_states'] as list of dicts
with 'M' and 'S' keys, or as list of (M, S) tuples.

Usage:
    from src.analysis.memory_tracing import analyze_memory_states, MemoryTracer

    # Single checkpoint analysis
    stats = analyze_memory_checkpoint("/path/to/checkpoint.pt")

    # Cross-epoch analysis
    tracer = MemoryTracer()
    results = tracer.analyze_epochs([0, 50, 100, 186])
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default checkpoint directory (from environment or empty)
# Set ATLAS_CHECKPOINT_DIR environment variable or pass --checkpoint-dir explicitly
DEFAULT_CHECKPOINT_DIR = os.environ.get("ATLAS_CHECKPOINT_DIR", "")

# Threshold for considering an entry as "near-zero" in sparsity calculation
SPARSITY_THRESHOLD = 1e-6

# Threshold for considering a singular value as "significant" in rank calculation
RANK_THRESHOLD = 1e-4

# Number of top singular values to track
TOP_SINGULAR_VALUES = 10

# Expected memory state shape components (from Atlas architecture)
EXPECTED_BATCH_SIZE = 32
EXPECTED_MEMORY_SLOTS = 1152
EXPECTED_MEMORY_DIM = 128

# Minimum samples for statistical analysis
MIN_EPOCHS_FOR_TREND = 3


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class LayerMemoryStats:
    """Statistics for a single layer's memory state.

    Captures comprehensive metrics about memory utilization in one layer
    of the Atlas model.

    Attributes:
        layer_idx: Index of the layer (0-indexed).
        m_magnitude_mean: Mean of L2 norms of M matrix rows.
        m_magnitude_std: Std of L2 norms of M matrix rows.
        m_frobenius_norm: Frobenius norm of M matrix.
        m_sparsity: Fraction of M entries below SPARSITY_THRESHOLD.
        m_effective_rank: Number of significant singular values of M.
        m_top_singular_values: Top singular values of M.
        s_magnitude_mean: Mean of L2 norms of S matrix rows.
        s_magnitude_std: Std of L2 norms of S matrix rows.
        s_frobenius_norm: Frobenius norm of S matrix.
        s_sparsity: Fraction of S entries below SPARSITY_THRESHOLD.
        s_effective_rank: Number of significant singular values of S.
        s_top_singular_values: Top singular values of S.
        m_shape: Shape of M matrix.
        s_shape: Shape of S matrix.
        metadata: Additional diagnostic information.
    """

    layer_idx: int
    # M matrix statistics
    m_magnitude_mean: float
    m_magnitude_std: float
    m_frobenius_norm: float
    m_sparsity: float
    m_effective_rank: int
    m_top_singular_values: NDArray[np.floating[Any]]
    # S matrix statistics
    s_magnitude_mean: float
    s_magnitude_std: float
    s_frobenius_norm: float
    s_sparsity: float
    s_effective_rank: int
    s_top_singular_values: NDArray[np.floating[Any]]
    # Shape information
    m_shape: tuple[int, ...]
    s_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_sparsity(self) -> float:
        """Combined sparsity of M and S matrices."""
        return (self.m_sparsity + self.s_sparsity) / 2

    @property
    def memory_utilization(self) -> float:
        """Memory utilization score (1 - sparsity)."""
        return 1.0 - self.total_sparsity

    @property
    def health_status(self) -> str:
        """Classify memory health based on statistics.

        Returns:
            'healthy': Normal memory utilization
            'sparse': High sparsity (>90%)
            'saturated': Very low sparsity (<10%)
            'degenerate': Very low rank
        """
        if self.m_effective_rank < 3 or self.s_effective_rank < 3:
            return "degenerate"
        if self.total_sparsity > 0.9:
            return "sparse"
        if self.total_sparsity < 0.1:
            return "saturated"
        return "healthy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_idx": self.layer_idx,
            "m_magnitude_mean": self.m_magnitude_mean,
            "m_magnitude_std": self.m_magnitude_std,
            "m_frobenius_norm": self.m_frobenius_norm,
            "m_sparsity": self.m_sparsity,
            "m_effective_rank": self.m_effective_rank,
            "m_top_singular_values": self.m_top_singular_values.tolist(),
            "s_magnitude_mean": self.s_magnitude_mean,
            "s_magnitude_std": self.s_magnitude_std,
            "s_frobenius_norm": self.s_frobenius_norm,
            "s_sparsity": self.s_sparsity,
            "s_effective_rank": self.s_effective_rank,
            "s_top_singular_values": self.s_top_singular_values.tolist(),
            "m_shape": self.m_shape,
            "s_shape": self.s_shape,
            "total_sparsity": self.total_sparsity,
            "memory_utilization": self.memory_utilization,
            "health_status": self.health_status,
            "metadata": self.metadata,
        }


@dataclass
class MemoryEpisodeStats:
    """Complete memory statistics for a single checkpoint/epoch.

    Aggregates layer-level statistics and provides epoch-level summaries.

    Attributes:
        epoch: Epoch number of the checkpoint.
        step: Training step number.
        layer_stats: List of LayerMemoryStats for each layer.
        num_layers: Number of layers with memory.
        total_m_memory_bytes: Estimated memory usage of M matrices.
        total_s_memory_bytes: Estimated memory usage of S matrices.
        analysis_time_seconds: Time taken for analysis.
        metadata: Additional checkpoint information.
    """

    epoch: int
    step: int
    layer_stats: list[LayerMemoryStats]
    num_layers: int
    total_m_memory_bytes: int
    total_s_memory_bytes: int
    analysis_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_m_sparsity(self) -> float:
        """Mean M matrix sparsity across layers."""
        if not self.layer_stats:
            return 0.0
        return float(np.mean([s.m_sparsity for s in self.layer_stats]))

    @property
    def mean_s_sparsity(self) -> float:
        """Mean S matrix sparsity across layers."""
        if not self.layer_stats:
            return 0.0
        return float(np.mean([s.s_sparsity for s in self.layer_stats]))

    @property
    def mean_m_effective_rank(self) -> float:
        """Mean M matrix effective rank across layers."""
        if not self.layer_stats:
            return 0.0
        return float(np.mean([s.m_effective_rank for s in self.layer_stats]))

    @property
    def mean_s_effective_rank(self) -> float:
        """Mean S matrix effective rank across layers."""
        if not self.layer_stats:
            return 0.0
        return float(np.mean([s.s_effective_rank for s in self.layer_stats]))

    @property
    def mean_m_magnitude(self) -> float:
        """Mean M matrix magnitude across layers."""
        if not self.layer_stats:
            return 0.0
        return float(np.mean([s.m_magnitude_mean for s in self.layer_stats]))

    @property
    def mean_s_magnitude(self) -> float:
        """Mean S matrix magnitude across layers."""
        if not self.layer_stats:
            return 0.0
        return float(np.mean([s.s_magnitude_mean for s in self.layer_stats]))

    @property
    def overall_health(self) -> str:
        """Overall memory health assessment."""
        if not self.layer_stats:
            return "no_data"

        health_counts = {}
        for stat in self.layer_stats:
            status = stat.health_status
            health_counts[status] = health_counts.get(status, 0) + 1

        # If any layer is degenerate, overall is concerning
        if health_counts.get("degenerate", 0) > 0:
            return "concerning:degenerate_layers"

        # If most layers are unhealthy, flag it
        unhealthy = health_counts.get("sparse", 0) + health_counts.get("saturated", 0)
        if unhealthy > len(self.layer_stats) / 2:
            return "warning:majority_unhealthy"

        return "healthy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "num_layers": self.num_layers,
            "mean_m_sparsity": self.mean_m_sparsity,
            "mean_s_sparsity": self.mean_s_sparsity,
            "mean_m_effective_rank": self.mean_m_effective_rank,
            "mean_s_effective_rank": self.mean_s_effective_rank,
            "mean_m_magnitude": self.mean_m_magnitude,
            "mean_s_magnitude": self.mean_s_magnitude,
            "overall_health": self.overall_health,
            "total_m_memory_bytes": self.total_m_memory_bytes,
            "total_s_memory_bytes": self.total_s_memory_bytes,
            "analysis_time_seconds": self.analysis_time_seconds,
            "layer_stats": [s.to_dict() for s in self.layer_stats],
            "metadata": self.metadata,
        }


@dataclass
class MemoryEvolutionResult:
    """Results from cross-epoch memory evolution analysis.

    Tracks how memory statistics change over training epochs.

    Attributes:
        epochs: List of analyzed epochs.
        epoch_stats: Dict mapping epoch to MemoryEpisodeStats.
        sparsity_trend: Correlation of sparsity with epoch (positive = increasing).
        rank_trend: Correlation of effective rank with epoch.
        magnitude_trend: Correlation of magnitude with epoch.
        outlier_epochs: Epochs with unusual memory patterns.
        total_analysis_time_seconds: Total time for all epoch analysis.
        metadata: Additional analysis information.
    """

    epochs: list[int]
    epoch_stats: dict[int, MemoryEpisodeStats]
    sparsity_trend: float  # Pearson correlation
    rank_trend: float
    magnitude_trend: float
    outlier_epochs: list[int]
    total_analysis_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        """Number of epochs analyzed."""
        return len(self.epochs)

    @property
    def has_trend(self) -> bool:
        """Check if there's a statistically significant trend."""
        return abs(self.sparsity_trend) > 0.5 or abs(self.rank_trend) > 0.5

    def get_epoch_summary(self, epoch: int) -> dict[str, float]:
        """Get summary statistics for a specific epoch."""
        if epoch not in self.epoch_stats:
            return {}
        stats = self.epoch_stats[epoch]
        return {
            "mean_m_sparsity": stats.mean_m_sparsity,
            "mean_s_sparsity": stats.mean_s_sparsity,
            "mean_m_rank": stats.mean_m_effective_rank,
            "mean_s_rank": stats.mean_s_effective_rank,
            "mean_m_magnitude": stats.mean_m_magnitude,
            "mean_s_magnitude": stats.mean_s_magnitude,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epochs": self.epochs,
            "n_epochs": self.n_epochs,
            "sparsity_trend": self.sparsity_trend,
            "rank_trend": self.rank_trend,
            "magnitude_trend": self.magnitude_trend,
            "outlier_epochs": self.outlier_epochs,
            "has_trend": self.has_trend,
            "total_analysis_time_seconds": self.total_analysis_time_seconds,
            "epoch_summaries": {e: self.get_epoch_summary(e) for e in self.epochs},
            "metadata": self.metadata,
        }


# ============================================================================
# Core Analysis Functions
# ============================================================================


def compute_matrix_stats(
    matrix: NDArray[np.floating[Any]],
    name: str = "matrix",
) -> dict[str, Any]:
    """Compute comprehensive statistics for a memory matrix.

    Parameters:
        matrix: Memory matrix, typically shape [batch, slots, dim] or [slots, dim].
        name: Name for logging purposes.

    Returns:
        Dict containing:
        - magnitude_mean: Mean L2 norm of rows
        - magnitude_std: Std of L2 norms
        - frobenius_norm: Frobenius norm of matrix
        - sparsity: Fraction of near-zero entries
        - effective_rank: Number of significant singular values
        - top_singular_values: Top singular values
        - shape: Original shape
    """
    # Flatten batch dimension if present
    if matrix.ndim == 3:
        # [batch, slots, dim] -> [batch * slots, dim]
        matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    elif matrix.ndim == 2:
        matrix_2d = matrix
    else:
        raise ValueError(f"Unexpected matrix shape: {matrix.shape}")

    # Row-wise L2 norms (magnitude of each memory slot)
    row_norms = np.linalg.norm(matrix_2d, axis=1)
    magnitude_mean = float(np.mean(row_norms))
    magnitude_std = float(np.std(row_norms))

    # Frobenius norm (overall matrix magnitude)
    frobenius_norm = float(np.linalg.norm(matrix_2d, "fro"))

    # Sparsity (fraction of near-zero entries)
    abs_values = np.abs(matrix_2d)
    near_zero_count = np.sum(abs_values < SPARSITY_THRESHOLD)
    sparsity = float(near_zero_count / matrix_2d.size)

    # Effective rank via SVD
    # Sample for efficiency if matrix is large
    if matrix_2d.shape[0] > 1000:
        indices = np.random.choice(matrix_2d.shape[0], 1000, replace=False)
        sample = matrix_2d[indices]
    else:
        sample = matrix_2d

    try:
        # Compute singular values
        singular_values = linalg.svdvals(sample)

        # Normalize by largest singular value for threshold comparison
        if singular_values[0] > 0:
            normalized_sv = singular_values / singular_values[0]
        else:
            normalized_sv = singular_values

        # Effective rank: count singular values above threshold
        effective_rank = int(np.sum(normalized_sv > RANK_THRESHOLD))

        # Top singular values
        top_sv = singular_values[:TOP_SINGULAR_VALUES]
    except Exception as e:
        logger.warning(f"SVD failed for {name}: {e}")
        effective_rank = 0
        top_sv = np.zeros(TOP_SINGULAR_VALUES)

    return {
        "magnitude_mean": magnitude_mean,
        "magnitude_std": magnitude_std,
        "frobenius_norm": frobenius_norm,
        "sparsity": sparsity,
        "effective_rank": effective_rank,
        "top_singular_values": top_sv,
        "shape": matrix.shape,
    }


def analyze_layer_memory(
    layer_idx: int,
    m_matrix: NDArray[np.floating[Any]],
    s_matrix: NDArray[np.floating[Any]],
) -> LayerMemoryStats:
    """Analyze memory state for a single layer.

    Parameters:
        layer_idx: Layer index (0-indexed).
        m_matrix: M (memory content) matrix.
        s_matrix: S (memory state/salience) matrix.

    Returns:
        LayerMemoryStats with comprehensive layer analysis.
    """
    m_stats = compute_matrix_stats(m_matrix, f"layer_{layer_idx}_M")
    s_stats = compute_matrix_stats(s_matrix, f"layer_{layer_idx}_S")

    return LayerMemoryStats(
        layer_idx=layer_idx,
        m_magnitude_mean=m_stats["magnitude_mean"],
        m_magnitude_std=m_stats["magnitude_std"],
        m_frobenius_norm=m_stats["frobenius_norm"],
        m_sparsity=m_stats["sparsity"],
        m_effective_rank=m_stats["effective_rank"],
        m_top_singular_values=m_stats["top_singular_values"],
        s_magnitude_mean=s_stats["magnitude_mean"],
        s_magnitude_std=s_stats["magnitude_std"],
        s_frobenius_norm=s_stats["frobenius_norm"],
        s_sparsity=s_stats["sparsity"],
        s_effective_rank=s_stats["effective_rank"],
        s_top_singular_values=s_stats["top_singular_values"],
        m_shape=m_stats["shape"],
        s_shape=s_stats["shape"],
    )


def analyze_memory_states(
    memory_states: list[Any],
    epoch: int = 0,
    step: int = 0,
) -> MemoryEpisodeStats:
    """Analyze memory states from a checkpoint.

    Parameters:
        memory_states: List of memory states, either as dicts with 'M'/'S' keys
                      or as (M, S) tuples.
        epoch: Epoch number for metadata.
        step: Step number for metadata.

    Returns:
        MemoryEpisodeStats with complete analysis.
    """
    start_time = time.time()

    layer_stats = []
    total_m_bytes = 0
    total_s_bytes = 0

    for layer_idx, layer_state in enumerate(memory_states):
        # Extract M and S matrices
        if isinstance(layer_state, dict):
            m_matrix = _to_numpy(layer_state.get("M"))
            s_matrix = _to_numpy(layer_state.get("S"))
        elif isinstance(layer_state, tuple) and len(layer_state) == 2:
            m_matrix = _to_numpy(layer_state[0])
            s_matrix = _to_numpy(layer_state[1])
        else:
            logger.warning(f"Unexpected memory state format at layer {layer_idx}")
            continue

        if m_matrix is None or s_matrix is None:
            logger.warning(f"Missing M or S matrix at layer {layer_idx}")
            continue

        # Compute layer statistics
        stats = analyze_layer_memory(layer_idx, m_matrix, s_matrix)
        layer_stats.append(stats)

        # Track memory usage
        total_m_bytes += m_matrix.nbytes
        total_s_bytes += s_matrix.nbytes

    analysis_time = time.time() - start_time

    return MemoryEpisodeStats(
        epoch=epoch,
        step=step,
        layer_stats=layer_stats,
        num_layers=len(layer_stats),
        total_m_memory_bytes=total_m_bytes,
        total_s_memory_bytes=total_s_bytes,
        analysis_time_seconds=analysis_time,
        metadata={
            "sparsity_threshold": SPARSITY_THRESHOLD,
            "rank_threshold": RANK_THRESHOLD,
        },
    )


def _to_numpy(tensor: Any) -> NDArray[np.floating[Any]] | None:
    """Convert tensor to numpy array.

    Parameters:
        tensor: PyTorch tensor or numpy array.

    Returns:
        Numpy array or None if conversion fails.
    """
    if tensor is None:
        return None

    try:
        # Handle PyTorch tensors
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        # Already numpy
        return np.asarray(tensor)
    except Exception as e:
        logger.warning(f"Failed to convert tensor to numpy: {e}")
        return None


# ============================================================================
# Checkpoint Analysis
# ============================================================================


def analyze_memory_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> MemoryEpisodeStats:
    """Analyze memory states from a checkpoint file.

    Loads the checkpoint, extracts memory states, and computes
    comprehensive statistics. Checkpoint is unloaded after analysis.

    Parameters:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load checkpoint to (cpu recommended for analysis).

    Returns:
        MemoryEpisodeStats with complete analysis.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        ValueError: If checkpoint has no memory states.
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Analyzing memory states in {checkpoint_path.name}")

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)

    # Extract memory states
    memory_states = checkpoint.get("memory_states")
    if memory_states is None:
        raise ValueError(f"Checkpoint has no memory_states: {checkpoint_path}")

    # Analyze
    result = analyze_memory_states(memory_states, epoch=epoch, step=step)
    result.metadata["checkpoint_path"] = str(checkpoint_path)

    # Clean up
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# Cross-Epoch Analysis
# ============================================================================


def analyze_memory_evolution(
    checkpoint_dir: str | Path,
    epochs: list[int] | None = None,
    device: str = "cpu",
) -> MemoryEvolutionResult:
    """Analyze memory evolution across multiple epochs.

    Implements sequential checkpoint loading to avoid OOM.

    Parameters:
        checkpoint_dir: Directory containing checkpoint files.
        epochs: Specific epochs to analyze. None = analyze all available.
        device: Device to load checkpoints to.

    Returns:
        MemoryEvolutionResult with cross-epoch analysis.
    """

    checkpoint_dir = Path(checkpoint_dir)
    start_time = time.time()

    # Find checkpoints
    all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not all_checkpoints:
        all_checkpoints = sorted(checkpoint_dir.glob("*.pt"))

    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    logger.info(f"Found {len(all_checkpoints)} checkpoints")

    # Analyze each checkpoint
    epoch_stats: dict[int, MemoryEpisodeStats] = {}
    analyzed_epochs: list[int] = []

    for checkpoint_path in all_checkpoints:
        try:
            episode = analyze_memory_checkpoint(checkpoint_path, device=device)

            if epochs is None or episode.epoch in epochs:
                epoch_stats[episode.epoch] = episode
                analyzed_epochs.append(episode.epoch)
                logger.info(
                    f"Analyzed epoch {episode.epoch}: "
                    f"sparsity={episode.mean_m_sparsity:.3f}, "
                    f"rank={episode.mean_m_effective_rank:.1f}"
                )

            # Check if we have all requested epochs
            if epochs is not None and len(epoch_stats) == len(epochs):
                break

        except Exception as e:
            logger.warning(f"Failed to analyze {checkpoint_path}: {e}")
            continue

    analyzed_epochs = sorted(analyzed_epochs)

    # Compute trends
    sparsity_trend = 0.0
    rank_trend = 0.0
    magnitude_trend = 0.0
    outlier_epochs: list[int] = []

    if len(analyzed_epochs) >= MIN_EPOCHS_FOR_TREND:
        epoch_array = np.array(analyzed_epochs)
        sparsities = np.array([epoch_stats[e].mean_m_sparsity for e in analyzed_epochs])
        ranks = np.array([epoch_stats[e].mean_m_effective_rank for e in analyzed_epochs])
        magnitudes = np.array([epoch_stats[e].mean_m_magnitude for e in analyzed_epochs])

        # Pearson correlations
        if np.std(sparsities) > 0:
            sparsity_trend = float(pearsonr(epoch_array, sparsities).statistic)
        if np.std(ranks) > 0:
            rank_trend = float(pearsonr(epoch_array, ranks).statistic)
        if np.std(magnitudes) > 0:
            magnitude_trend = float(pearsonr(epoch_array, magnitudes).statistic)

        # Detect outliers (> 2 std from mean)
        sparsity_mean = np.mean(sparsities)
        sparsity_std = np.std(sparsities)
        for i, e in enumerate(analyzed_epochs):
            if abs(sparsities[i] - sparsity_mean) > 2 * sparsity_std:
                outlier_epochs.append(e)

    total_time = time.time() - start_time

    return MemoryEvolutionResult(
        epochs=analyzed_epochs,
        epoch_stats=epoch_stats,
        sparsity_trend=sparsity_trend,
        rank_trend=rank_trend,
        magnitude_trend=magnitude_trend,
        outlier_epochs=outlier_epochs,
        total_analysis_time_seconds=total_time,
        metadata={
            "checkpoint_dir": str(checkpoint_dir),
            "total_checkpoints": len(all_checkpoints),
            "min_epochs_for_trend": MIN_EPOCHS_FOR_TREND,
        },
    )


# ============================================================================
# Visualization Functions
# ============================================================================


def create_memory_heatmap(
    stats: MemoryEpisodeStats,
    output_path: str | Path | None = None,
) -> Any:
    """Create heatmap visualization of memory statistics across layers.

    Parameters:
        stats: MemoryEpisodeStats from analysis.
        output_path: Optional path to save the plot.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_layers = len(stats.layer_stats)
    if n_layers == 0:
        logger.warning("No layer stats to visualize")
        return None

    # Extract metrics per layer
    layers = [s.layer_idx for s in stats.layer_stats]
    m_sparsity = [s.m_sparsity for s in stats.layer_stats]
    s_sparsity = [s.s_sparsity for s in stats.layer_stats]
    m_rank = [s.m_effective_rank for s in stats.layer_stats]
    s_rank = [s.s_effective_rank for s in stats.layer_stats]
    # Note: magnitude data available via stats.layer_stats if needed

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "M Matrix Sparsity by Layer",
            "S Matrix Sparsity by Layer",
            "M Matrix Effective Rank",
            "S Matrix Effective Rank",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # M Sparsity
    fig.add_trace(
        go.Bar(
            x=layers,
            y=m_sparsity,
            name="M Sparsity",
            marker_color="blue",
        ),
        row=1,
        col=1,
    )

    # S Sparsity
    fig.add_trace(
        go.Bar(
            x=layers,
            y=s_sparsity,
            name="S Sparsity",
            marker_color="green",
        ),
        row=1,
        col=2,
    )

    # M Rank
    fig.add_trace(
        go.Bar(
            x=layers,
            y=m_rank,
            name="M Rank",
            marker_color="purple",
        ),
        row=2,
        col=1,
    )

    # S Rank
    fig.add_trace(
        go.Bar(
            x=layers,
            y=s_rank,
            name="S Rank",
            marker_color="orange",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Memory Statistics - Epoch {stats.epoch}",
        height=600,
        showlegend=False,
    )

    # Update axis labels
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=2)
    fig.update_yaxes(title_text="Sparsity", row=1, col=1)
    fig.update_yaxes(title_text="Sparsity", row=1, col=2)
    fig.update_yaxes(title_text="Effective Rank", row=2, col=1)
    fig.update_yaxes(title_text="Effective Rank", row=2, col=2)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if str(output_path).endswith(".html"):
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=1200, height=800, scale=2.5)
        logger.info(f"Saved memory heatmap to {output_path}")

    return fig


def create_evolution_plot(
    result: MemoryEvolutionResult,
    output_path: str | Path | None = None,
) -> Any:
    """Create line plot showing memory evolution across epochs.

    Parameters:
        result: MemoryEvolutionResult from analysis.
        output_path: Optional path to save the plot.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    epochs = result.epochs
    if not epochs:
        logger.warning("No epochs to visualize")
        return None

    # Extract metrics
    m_sparsity = [result.epoch_stats[e].mean_m_sparsity for e in epochs]
    s_sparsity = [result.epoch_stats[e].mean_s_sparsity for e in epochs]
    m_rank = [result.epoch_stats[e].mean_m_effective_rank for e in epochs]
    s_rank = [result.epoch_stats[e].mean_s_effective_rank for e in epochs]
    m_magnitude = [result.epoch_stats[e].mean_m_magnitude for e in epochs]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Memory Sparsity Over Training",
            "Effective Rank Over Training",
            "Memory Magnitude Over Training",
        ),
        vertical_spacing=0.1,
    )

    # Sparsity
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=m_sparsity,
            mode="lines+markers",
            name="M Sparsity",
            line={"color": "blue", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=s_sparsity,
            mode="lines+markers",
            name="S Sparsity",
            line={"color": "green", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    # Rank
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=m_rank,
            mode="lines+markers",
            name="M Rank",
            line={"color": "purple", "width": 2},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=s_rank,
            mode="lines+markers",
            name="S Rank",
            line={"color": "orange", "width": 2, "dash": "dash"},
        ),
        row=2,
        col=1,
    )

    # Magnitude
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=m_magnitude,
            mode="lines+markers",
            name="M Magnitude",
            line={"color": "red", "width": 2},
        ),
        row=3,
        col=1,
    )

    # Mark outlier epochs
    for outlier_epoch in result.outlier_epochs:
        if outlier_epoch in epochs:
            fig.add_vline(
                x=outlier_epoch,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"Outlier: {outlier_epoch}",
            )

    fig.update_layout(
        title=f"Memory Evolution (Sparsity trend: {result.sparsity_trend:.3f})",
        height=800,
        showlegend=True,
        legend={"x": 1.02, "y": 1},
    )

    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_yaxes(title_text="Sparsity", row=1, col=1)
    fig.update_yaxes(title_text="Effective Rank", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=3, col=1)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if str(output_path).endswith(".html"):
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=1000, height=800, scale=2.5)
        logger.info(f"Saved evolution plot to {output_path}")

    return fig


def create_singular_value_plot(
    stats: MemoryEpisodeStats,
    output_path: str | Path | None = None,
) -> Any:
    """Create plot showing singular value distributions across layers.

    Parameters:
        stats: MemoryEpisodeStats from analysis.
        output_path: Optional path to save the plot.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    n_layers = len(stats.layer_stats)
    if n_layers == 0:
        logger.warning("No layer stats to visualize")
        return None

    fig = go.Figure()

    for layer_stat in stats.layer_stats:
        sv = layer_stat.m_top_singular_values
        sv_indices = list(range(len(sv)))

        fig.add_trace(
            go.Scatter(
                x=sv_indices,
                y=sv,
                mode="lines+markers",
                name=f"Layer {layer_stat.layer_idx}",
            )
        )

    fig.update_layout(
        title=f"Top Singular Values of M Matrices - Epoch {stats.epoch}",
        xaxis_title="Singular Value Index",
        yaxis_title="Singular Value",
        yaxis_type="log",
        height=500,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if str(output_path).endswith(".html"):
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=800, height=500, scale=2.5)
        logger.info(f"Saved singular value plot to {output_path}")

    return fig


# ============================================================================
# MemoryTracer Class
# ============================================================================


class MemoryTracer:
    """High-level interface for memory episode tracing.

    Provides a convenient API for analyzing memory states across
    multiple checkpoints with caching and batch analysis support.

    Example:
        tracer = MemoryTracer()
        result = tracer.analyze_checkpoint("/path/to/checkpoint.pt")
        evolution = tracer.analyze_epochs([0, 50, 100, 186])
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        """Initialize the MemoryTracer.

        Parameters:
            checkpoint_dir: Default directory for checkpoints.
            device: Device to load checkpoints to.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = device
        self._cache: dict[str, MemoryEpisodeStats] = {}

    def analyze_checkpoint(
        self,
        checkpoint_path: str | Path,
        use_cache: bool = True,
    ) -> MemoryEpisodeStats:
        """Analyze a single checkpoint.

        Parameters:
            checkpoint_path: Path to checkpoint file.
            use_cache: Whether to use cached results.

        Returns:
            MemoryEpisodeStats for the checkpoint.
        """
        path_str = str(checkpoint_path)

        if use_cache and path_str in self._cache:
            return self._cache[path_str]

        result = analyze_memory_checkpoint(checkpoint_path, device=self.device)

        if use_cache:
            self._cache[path_str] = result

        return result

    def analyze_epochs(
        self,
        epochs: list[int] | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> MemoryEvolutionResult:
        """Analyze memory evolution across epochs.

        Parameters:
            epochs: Specific epochs to analyze. None = all available.
            checkpoint_dir: Override default checkpoint directory.

        Returns:
            MemoryEvolutionResult with cross-epoch analysis.
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = DEFAULT_CHECKPOINT_DIR

        return analyze_memory_evolution(
            checkpoint_dir,
            epochs=epochs,
            device=self.device,
        )

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface for memory tracing analysis."""
    parser = argparse.ArgumentParser(
        description="Memory Episode Tracing for Atlas model checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single checkpoint file to analyze",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=f"Directory containing checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Comma-separated list of epochs to analyze (e.g., '0,50,100,186')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (HTML or PNG)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output path for CSV metrics export",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoints to (default: cpu)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test with synthetic data (no checkpoints needed)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.test:
        _run_synthetic_test(args.output)
        return

    if args.checkpoint:
        # Single checkpoint analysis
        try:
            stats = analyze_memory_checkpoint(args.checkpoint, device=args.device)

            print("\n=== Memory Episode Statistics ===")
            print(f"Checkpoint: {args.checkpoint}")
            print(f"Epoch: {stats.epoch}, Step: {stats.step}")
            print(f"Layers: {stats.num_layers}")
            print(f"Overall Health: {stats.overall_health}")
            print("\nM Matrix Statistics:")
            print(f"  Mean Sparsity: {stats.mean_m_sparsity:.4f}")
            print(f"  Mean Effective Rank: {stats.mean_m_effective_rank:.1f}")
            print(f"  Mean Magnitude: {stats.mean_m_magnitude:.4f}")
            print("\nS Matrix Statistics:")
            print(f"  Mean Sparsity: {stats.mean_s_sparsity:.4f}")
            print(f"  Mean Effective Rank: {stats.mean_s_effective_rank:.1f}")
            print(f"  Mean Magnitude: {stats.mean_s_magnitude:.4f}")
            print(f"\nAnalysis time: {stats.analysis_time_seconds:.2f}s")
            print("Memory statistics computed")

            # Generate visualization if output specified
            if args.output:
                create_memory_heatmap(stats, args.output)

            # Export CSV if specified
            if args.output_csv:
                _export_stats_csv(stats, args.output_csv)

        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    elif args.epochs:
        # Cross-epoch analysis
        checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        epochs = [int(e.strip()) for e in args.epochs.split(",")]

        try:
            result = analyze_memory_evolution(
                checkpoint_dir,
                epochs=epochs,
                device=args.device,
            )

            print("\n=== Memory Evolution Analysis ===")
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"Epochs analyzed: {result.epochs}")
            print("\nTrends:")
            print(f"  Sparsity trend: {result.sparsity_trend:.3f}")
            print(f"  Rank trend: {result.rank_trend:.3f}")
            print(f"  Magnitude trend: {result.magnitude_trend:.3f}")
            if result.outlier_epochs:
                print(f"\nOutlier epochs: {result.outlier_epochs}")
            print(f"\nTotal analysis time: {result.total_analysis_time_seconds:.2f}s")
            print("Memory statistics computed")

            # Generate visualization
            if args.output:
                create_evolution_plot(result, args.output)

            # Export CSV
            if args.output_csv:
                _export_evolution_csv(result, args.output_csv)

        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.analysis.memory_tracing --checkpoint /path/to/checkpoint.pt")
        print("  python -m src.analysis.memory_tracing --epochs 0,50,100,186")
        print("  python -m src.analysis.memory_tracing --test")


def _run_synthetic_test(output_path: str | None = None) -> None:
    """Run test with synthetic memory data."""
    print("Running synthetic test (no checkpoints needed)...")

    np.random.seed(42)

    # Create synthetic memory states (4 layers)
    n_layers = 4
    batch_size = 8
    memory_slots = 100
    memory_dim = 64

    memory_states = []
    for layer in range(n_layers):
        # Add some structure: later layers are more sparse
        sparsity = 0.5 + 0.1 * layer
        m_matrix = np.random.randn(batch_size, memory_slots, memory_dim)
        m_matrix[np.random.rand(*m_matrix.shape) < sparsity] = 0

        s_matrix = np.random.randn(batch_size, memory_slots, memory_dim) * 0.5
        s_matrix[np.random.rand(*s_matrix.shape) < sparsity] = 0

        memory_states.append({"M": m_matrix, "S": s_matrix})

    print(f"Generated synthetic memory states for {n_layers} layers")

    # Analyze
    stats = analyze_memory_states(memory_states, epoch=100, step=5000)

    print("\n=== Synthetic Test Results ===")
    print(f"Epoch: {stats.epoch}, Step: {stats.step}")
    print(f"Layers: {stats.num_layers}")
    print(f"Overall Health: {stats.overall_health}")
    print("\nM Matrix Statistics:")
    print(f"  Mean Sparsity: {stats.mean_m_sparsity:.4f}")
    print(f"  Mean Effective Rank: {stats.mean_m_effective_rank:.1f}")
    print(f"  Mean Magnitude: {stats.mean_m_magnitude:.4f}")

    print("\nPer-layer breakdown:")
    for layer_stat in stats.layer_stats:
        print(f"  Layer {layer_stat.layer_idx}: "
              f"sparsity={layer_stat.m_sparsity:.3f}, "
              f"rank={layer_stat.m_effective_rank}, "
              f"health={layer_stat.health_status}")

    print(f"\nAnalysis time: {stats.analysis_time_seconds:.2f}s")
    print("Memory statistics computed")

    if output_path:
        print(f"\nGenerating visualization: {output_path}")
        create_memory_heatmap(stats, output_path)
    else:
        print("\nNo output path specified, skipping visualization")
        print("Use --output <path.html> to generate visualization")

    print("\nSynthetic test completed successfully!")


def _export_stats_csv(stats: MemoryEpisodeStats, output_path: str) -> None:
    """Export memory statistics to CSV."""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "m_sparsity", "s_sparsity",
            "m_rank", "s_rank",
            "m_magnitude", "s_magnitude",
            "health_status"
        ])

        for layer_stat in stats.layer_stats:
            writer.writerow([
                layer_stat.layer_idx,
                f"{layer_stat.m_sparsity:.6f}",
                f"{layer_stat.s_sparsity:.6f}",
                layer_stat.m_effective_rank,
                layer_stat.s_effective_rank,
                f"{layer_stat.m_magnitude_mean:.6f}",
                f"{layer_stat.s_magnitude_mean:.6f}",
                layer_stat.health_status,
            ])

    print(f"Exported statistics to {output_path}")


def _export_evolution_csv(result: MemoryEvolutionResult, output_path: str) -> None:
    """Export memory evolution to CSV."""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "step",
            "mean_m_sparsity", "mean_s_sparsity",
            "mean_m_rank", "mean_s_rank",
            "mean_m_magnitude", "mean_s_magnitude",
            "overall_health"
        ])

        for epoch in result.epochs:
            stats = result.epoch_stats[epoch]
            writer.writerow([
                epoch,
                stats.step,
                f"{stats.mean_m_sparsity:.6f}",
                f"{stats.mean_s_sparsity:.6f}",
                f"{stats.mean_m_effective_rank:.2f}",
                f"{stats.mean_s_effective_rank:.2f}",
                f"{stats.mean_m_magnitude:.6f}",
                f"{stats.mean_s_magnitude:.6f}",
                stats.overall_health,
            ])

    print(f"Exported evolution to {output_path}")


if __name__ == "__main__":
    main()
