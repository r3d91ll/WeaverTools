"""Layer Comparison Utilities for Hidden State Analysis.

This module provides utility functions for comparing and analyzing hidden states
across different layers of transformer models. These utilities enable researchers
to track how semantic information evolves through model layers and identify
bottlenecks in information flow.

MATHEMATICAL GROUNDING
======================
Layer-by-layer analysis leverages the Conveyance Hypothesis framework:

- D_eff (Effective Dimensionality): Measures the intrinsic dimensionality of
  the embedding space at each layer. Computed via PCA eigenvalue analysis.

- D_eff Trajectory: The sequence of D_eff values across layers reveals how
  information is compressed or expanded during forward propagation.

- Bottleneck Detection: Layers where D_eff drops significantly indicate
  information compression (potential loss of semantic diversity).

INTERPRETATION
==============
- Increasing D_eff trajectory: Model is expanding semantic representation
- Decreasing D_eff trajectory: Model is compressing/focusing representation
- Sharp drops in D_eff: Potential bottleneck layers (information loss)
- Multi-agent comparison: Identifies where different models process differently

Integration: Designed to work with TheLoom's HiddenStateResult class and
complements the conveyance_metrics module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .conveyance_metrics import calculate_d_eff, DEFAULT_VARIANCE_THRESHOLD
from ..extraction.hidden_states import HiddenStateResult

# ============================================================================
# Constants
# ============================================================================

# Minimum percentage drop in D_eff to be considered a bottleneck.
# 20% drop is chosen as a significant but not extreme threshold.
DEFAULT_BOTTLENECK_THRESHOLD = 0.20

# Minimum absolute D_eff drop to consider as bottleneck (avoids noise at low D_eff).
MIN_ABSOLUTE_DEFF_DROP = 2


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class LayerComparisonResult:
    """Results from comparing D_eff values between two sets of layers.

    Compares corresponding layers between two models/runs and provides
    metrics for understanding how their information processing differs.

    Interpretation:
    - mean_diff: Average D_eff difference across layers (positive = A > B)
    - max_diff_layer: Layer with largest absolute difference
    - correlation: How similarly the two trajectories evolve

    Use Cases:
    - Compare same model on different inputs
    - Compare different models on same input
    - Compare model before/after fine-tuning
    """

    layers_a: dict[int, int]  # Layer index -> D_eff for model/run A
    layers_b: dict[int, int]  # Layer index -> D_eff for model/run B
    layer_diffs: dict[int, int]  # Layer index -> D_eff(A) - D_eff(B)
    common_layers: list[int]  # Layers present in both
    mean_diff: float  # Mean of differences (A - B)
    abs_mean_diff: float  # Mean of absolute differences
    max_diff_layer: int | None  # Layer with maximum absolute difference
    max_diff_value: int  # Maximum absolute difference value
    correlation: float  # Pearson correlation of trajectories
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_similar(self) -> bool:
        """Check if the two layer trajectories are similar.

        Considers trajectories similar if correlation > 0.9 and
        mean absolute difference is < 10% of mean D_eff.
        """
        if not self.common_layers:
            return False
        mean_deff = np.mean(list(self.layers_a.values()) + list(self.layers_b.values()))
        if mean_deff == 0:
            return self.abs_mean_diff == 0
        return self.correlation > 0.9 and self.abs_mean_diff < 0.1 * mean_deff

    @property
    def divergence_quality(self) -> str:
        """Classify the level of divergence between trajectories.

        Categories:
        - identical: No differences
        - similar: High correlation, low mean difference
        - moderate: Some differences but same general pattern
        - divergent: Significantly different trajectories
        """
        if self.abs_mean_diff == 0:
            return "identical"
        elif self.is_similar:
            return "similar"
        elif self.correlation > 0.7:
            return "moderate"
        else:
            return "divergent"


@dataclass
class LayerTrajectoryResult:
    """Results from computing D_eff trajectory across layers.

    Tracks how effective dimensionality evolves from input to output,
    revealing information flow patterns through the transformer.

    Interpretation:
    - trajectory: Ordered D_eff values by layer
    - trend: "increasing", "decreasing", or "stable"
    - monotonicity: How consistently the trajectory moves in one direction
    """

    layer_indices: list[int]  # Sorted layer indices
    d_eff_values: list[int]  # D_eff at each layer (corresponding to indices)
    trajectory: dict[int, int]  # Layer index -> D_eff
    min_layer: int  # Layer with minimum D_eff
    max_layer: int  # Layer with maximum D_eff
    min_d_eff: int  # Minimum D_eff value
    max_d_eff: int  # Maximum D_eff value
    mean_d_eff: float  # Mean D_eff across layers
    std_d_eff: float  # Standard deviation of D_eff
    trend: str  # "increasing", "decreasing", "stable", or "variable"
    monotonicity: float  # Spearman correlation with layer index [-1, 1]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_compressive(self) -> bool:
        """Check if trajectory shows overall compression (decreasing D_eff)."""
        return self.trend == "decreasing" or self.monotonicity < -0.5

    @property
    def is_expansive(self) -> bool:
        """Check if trajectory shows overall expansion (increasing D_eff)."""
        return self.trend == "increasing" or self.monotonicity > 0.5

    @property
    def range_ratio(self) -> float:
        """Ratio of D_eff range to mean (higher = more variable)."""
        if self.mean_d_eff == 0:
            return 0.0
        return (self.max_d_eff - self.min_d_eff) / self.mean_d_eff


@dataclass
class BottleneckResult:
    """Results from identifying bottleneck layers.

    Bottleneck layers are those where D_eff drops significantly,
    indicating potential information compression or loss.

    Interpretation:
    - bottleneck_layers: Layer indices where significant drops occur
    - drop_magnitudes: How much D_eff dropped at each bottleneck
    - severity: Classification of how severe the bottlenecks are
    """

    bottleneck_layers: list[int]  # Layers identified as bottlenecks
    drop_magnitudes: dict[int, float]  # Layer -> relative drop percentage
    absolute_drops: dict[int, int]  # Layer -> absolute D_eff drop
    previous_layers: dict[int, int]  # Bottleneck layer -> preceding layer
    threshold_used: float  # Threshold used for detection
    n_bottlenecks: int  # Number of bottlenecks found
    max_drop_layer: int | None  # Layer with largest drop
    max_drop_magnitude: float  # Largest relative drop
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_bottlenecks(self) -> bool:
        """Check if any bottlenecks were detected."""
        return self.n_bottlenecks > 0

    @property
    def severity(self) -> str:
        """Classify the overall severity of bottlenecks.

        Categories:
        - none: No bottlenecks detected
        - mild: 1 bottleneck with < 30% drop
        - moderate: Multiple bottlenecks or single > 30% drop
        - severe: Multiple bottlenecks with > 40% average drop
        """
        if self.n_bottlenecks == 0:
            return "none"
        elif self.n_bottlenecks == 1 and self.max_drop_magnitude < 0.30:
            return "mild"
        elif self.n_bottlenecks > 2 or self.max_drop_magnitude > 0.40:
            return "severe"
        else:
            return "moderate"


# ============================================================================
# Layer Comparison Functions
# ============================================================================


def compare_layer_deff(
    layers_a: dict[int, HiddenStateResult | np.ndarray] | dict[int, int],
    layers_b: dict[int, HiddenStateResult | np.ndarray] | dict[int, int],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> LayerComparisonResult:
    """Compare D_eff values between two sets of layer hidden states.

    This function compares the effective dimensionality at corresponding
    layers between two models or runs, enabling analysis of how different
    models process information at each layer.

    Parameters:
        layers_a: dict[int, HiddenStateResult | np.ndarray] | dict[int, int]
            First set of layers. Can be:
            - dict mapping layer index to HiddenStateResult
            - dict mapping layer index to embeddings array (n_samples, dim)
            - dict mapping layer index to pre-computed D_eff int
        layers_b: dict[int, HiddenStateResult | np.ndarray] | dict[int, int]
            Second set of layers (same format options as layers_a).
        variance_threshold: float, default=0.90
            Variance threshold for D_eff calculation.

    Returns:
        LayerComparisonResult with comparison metrics including:
        - layer_diffs: Difference at each common layer
        - correlation: How similarly trajectories evolve
        - mean_diff: Average difference (A - B)

    Example:
        >>> # Compare two models on same input
        >>> result = compare_layer_deff(model_a_layers, model_b_layers)
        >>> print(f"Mean diff: {result.mean_diff:.2f}")
        >>> print(f"Correlation: {result.correlation:.4f}")

    Notes:
        - Only layers present in both sets are compared
        - Pre-computed D_eff values (int) bypass computation
        - Empty layer sets raise ValueError
    """
    if not layers_a or not layers_b:
        raise ValueError("Both layer sets must be non-empty")

    # Convert to D_eff values if needed
    deff_a = _compute_layer_deff_dict(layers_a, variance_threshold)
    deff_b = _compute_layer_deff_dict(layers_b, variance_threshold)

    # Find common layers
    common_layers = sorted(set(deff_a.keys()) & set(deff_b.keys()))

    if not common_layers:
        return LayerComparisonResult(
            layers_a=deff_a,
            layers_b=deff_b,
            layer_diffs={},
            common_layers=[],
            mean_diff=0.0,
            abs_mean_diff=0.0,
            max_diff_layer=None,
            max_diff_value=0,
            correlation=0.0,
            metadata={"warning": "no_common_layers"},
        )

    # Compute differences
    layer_diffs = {layer: deff_a[layer] - deff_b[layer] for layer in common_layers}

    # Statistics
    diffs_array = np.array([layer_diffs[layer] for layer in common_layers])
    mean_diff = float(np.mean(diffs_array))
    abs_mean_diff = float(np.mean(np.abs(diffs_array)))

    # Find max difference
    abs_diffs = {layer: abs(diff) for layer, diff in layer_diffs.items()}
    max_diff_layer = max(abs_diffs, key=abs_diffs.get)  # type: ignore
    max_diff_value = abs_diffs[max_diff_layer]

    # Compute correlation of trajectories
    vals_a = np.array([deff_a[layer] for layer in common_layers])
    vals_b = np.array([deff_b[layer] for layer in common_layers])

    if len(common_layers) < 2 or np.std(vals_a) == 0 or np.std(vals_b) == 0:
        correlation = 1.0 if np.allclose(vals_a, vals_b) else 0.0
    else:
        correlation = float(np.corrcoef(vals_a, vals_b)[0, 1])

    return LayerComparisonResult(
        layers_a=deff_a,
        layers_b=deff_b,
        layer_diffs=layer_diffs,
        common_layers=common_layers,
        mean_diff=mean_diff,
        abs_mean_diff=abs_mean_diff,
        max_diff_layer=max_diff_layer,
        max_diff_value=max_diff_value,
        correlation=correlation,
    )


def compute_layer_trajectory(
    layers: dict[int, HiddenStateResult | np.ndarray] | dict[int, int],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> LayerTrajectoryResult:
    """Compute D_eff trajectory across all layers.

    Analyzes how effective dimensionality evolves from early to late
    layers, revealing the model's information processing pattern.

    Parameters:
        layers: dict[int, HiddenStateResult | np.ndarray] | dict[int, int]
            Layers to analyze. Can be:
            - dict mapping layer index to HiddenStateResult
            - dict mapping layer index to embeddings array
            - dict mapping layer index to pre-computed D_eff int
        variance_threshold: float, default=0.90
            Variance threshold for D_eff calculation.

    Returns:
        LayerTrajectoryResult with trajectory analysis including:
        - d_eff_values: D_eff at each layer
        - trend: Overall trend direction
        - monotonicity: Consistency of trend direction

    Example:
        >>> trajectory = compute_layer_trajectory(all_layers)
        >>> print(f"Trend: {trajectory.trend}")
        >>> print(f"Min D_eff: {trajectory.min_d_eff} at layer {trajectory.min_layer}")
        >>> print(f"Max D_eff: {trajectory.max_d_eff} at layer {trajectory.max_layer}")

    Notes:
        - Layers are sorted by index for trajectory analysis
        - Trend is determined by Spearman correlation with layer index
        - Single-layer input returns "stable" trend
    """
    if not layers:
        raise ValueError("layers dict must be non-empty")

    # Convert to D_eff values if needed
    deff_dict = _compute_layer_deff_dict(layers, variance_threshold)

    # Sort by layer index
    layer_indices = sorted(deff_dict.keys())
    d_eff_values = [deff_dict[idx] for idx in layer_indices]

    # Basic statistics
    d_eff_array = np.array(d_eff_values)
    min_idx = int(np.argmin(d_eff_array))
    max_idx = int(np.argmax(d_eff_array))

    min_layer = layer_indices[min_idx]
    max_layer = layer_indices[max_idx]
    min_d_eff = d_eff_values[min_idx]
    max_d_eff = d_eff_values[max_idx]
    mean_d_eff = float(np.mean(d_eff_array))
    std_d_eff = float(np.std(d_eff_array))

    # Compute monotonicity (Spearman correlation with layer position)
    if len(layer_indices) < 2:
        monotonicity = 0.0
        trend = "stable"
    else:
        # Use scipy for proper Spearman if available, else simple correlation
        try:
            from scipy.stats import spearmanr
            monotonicity, _ = spearmanr(layer_indices, d_eff_values)
            monotonicity = float(monotonicity) if not np.isnan(monotonicity) else 0.0
        except ImportError:
            # Fallback: use Pearson correlation as approximation
            if np.std(d_eff_array) == 0:
                monotonicity = 0.0
            else:
                monotonicity = float(np.corrcoef(layer_indices, d_eff_values)[0, 1])
                if np.isnan(monotonicity):
                    monotonicity = 0.0

        # Determine trend from monotonicity
        if monotonicity > 0.5:
            trend = "increasing"
        elif monotonicity < -0.5:
            trend = "decreasing"
        elif mean_d_eff == 0:
            # Handle zero-mean case: use absolute threshold
            trend = "stable" if std_d_eff < 1e-8 else "variable"
        elif std_d_eff < 0.1 * mean_d_eff:
            trend = "stable"
        else:
            trend = "variable"

    return LayerTrajectoryResult(
        layer_indices=layer_indices,
        d_eff_values=d_eff_values,
        trajectory=deff_dict,
        min_layer=min_layer,
        max_layer=max_layer,
        min_d_eff=min_d_eff,
        max_d_eff=max_d_eff,
        mean_d_eff=mean_d_eff,
        std_d_eff=std_d_eff,
        trend=trend,
        monotonicity=monotonicity,
    )


def find_bottleneck_layers(
    deff_dict: dict[int, int] | dict[int, HiddenStateResult | np.ndarray],
    threshold: float = DEFAULT_BOTTLENECK_THRESHOLD,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> BottleneckResult:
    """Identify layers where D_eff drops significantly (bottlenecks).

    Bottleneck layers indicate potential information compression or loss,
    where the model significantly reduces the effective dimensionality
    of its representations.

    DETECTION ALGORITHM
    ===================
    For each consecutive layer pair (i, i+1):
    1. Compute relative drop: (D_eff_i - D_eff_{i+1}) / D_eff_i
    2. If drop > threshold AND absolute drop > MIN_ABSOLUTE_DEFF_DROP:
       Layer i+1 is marked as a bottleneck

    Parameters:
        deff_dict: dict[int, int] | dict[int, HiddenStateResult | np.ndarray]
            Mapping of layer indices to D_eff values (or data to compute from).
        threshold: float, default=0.20
            Minimum relative drop to be considered a bottleneck.
            E.g., 0.20 means D_eff must drop by at least 20%.
        variance_threshold: float, default=0.90
            Variance threshold for D_eff calculation (if computing from data).

    Returns:
        BottleneckResult with identified bottlenecks:
        - bottleneck_layers: List of bottleneck layer indices
        - drop_magnitudes: Relative drop at each bottleneck
        - severity: Overall severity classification

    Example:
        >>> bottlenecks = find_bottleneck_layers(deff_dict, threshold=0.25)
        >>> if bottlenecks.has_bottlenecks:
        ...     print(f"Bottlenecks at layers: {bottlenecks.bottleneck_layers}")
        ...     print(f"Severity: {bottlenecks.severity}")

    Notes:
        - Bottlenecks are computed on consecutive layers (sorted by index)
        - Very small D_eff values (< 5) may produce spurious bottlenecks
        - Use MIN_ABSOLUTE_DEFF_DROP to filter noise at low D_eff
    """
    if not deff_dict:
        return BottleneckResult(
            bottleneck_layers=[],
            drop_magnitudes={},
            absolute_drops={},
            previous_layers={},
            threshold_used=threshold,
            n_bottlenecks=0,
            max_drop_layer=None,
            max_drop_magnitude=0.0,
            metadata={"warning": "empty_input"},
        )

    # Convert to D_eff values if needed
    deff_values = _compute_layer_deff_dict(deff_dict, variance_threshold)

    # Sort layers
    sorted_layers = sorted(deff_values.keys())

    if len(sorted_layers) < 2:
        return BottleneckResult(
            bottleneck_layers=[],
            drop_magnitudes={},
            absolute_drops={},
            previous_layers={},
            threshold_used=threshold,
            n_bottlenecks=0,
            max_drop_layer=None,
            max_drop_magnitude=0.0,
            metadata={"warning": "insufficient_layers"},
        )

    # Find bottlenecks by comparing consecutive layers
    bottleneck_layers: list[int] = []
    drop_magnitudes: dict[int, float] = {}
    absolute_drops: dict[int, int] = {}
    previous_layers: dict[int, int] = {}

    for i in range(len(sorted_layers) - 1):
        prev_layer = sorted_layers[i]
        curr_layer = sorted_layers[i + 1]
        prev_deff = deff_values[prev_layer]
        curr_deff = deff_values[curr_layer]

        # Skip if previous D_eff is too small (avoid division issues)
        if prev_deff < MIN_ABSOLUTE_DEFF_DROP:
            continue

        absolute_drop = prev_deff - curr_deff
        relative_drop = absolute_drop / prev_deff

        # Check if this is a bottleneck
        if relative_drop >= threshold and absolute_drop >= MIN_ABSOLUTE_DEFF_DROP:
            bottleneck_layers.append(curr_layer)
            drop_magnitudes[curr_layer] = relative_drop
            absolute_drops[curr_layer] = absolute_drop
            previous_layers[curr_layer] = prev_layer

    # Find max drop
    if bottleneck_layers:
        max_drop_layer = max(drop_magnitudes, key=drop_magnitudes.get)  # type: ignore
        max_drop_magnitude = drop_magnitudes[max_drop_layer]
    else:
        max_drop_layer = None
        max_drop_magnitude = 0.0

    return BottleneckResult(
        bottleneck_layers=bottleneck_layers,
        drop_magnitudes=drop_magnitudes,
        absolute_drops=absolute_drops,
        previous_layers=previous_layers,
        threshold_used=threshold,
        n_bottlenecks=len(bottleneck_layers),
        max_drop_layer=max_drop_layer,
        max_drop_magnitude=max_drop_magnitude,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_layer_deff_dict(
    layers: dict[int, HiddenStateResult | np.ndarray] | dict[int, int],
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> dict[int, int]:
    """Convert layer data to D_eff values if not already computed.

    Handles three input formats:
    1. dict[int, int] - Already D_eff values, pass through
    2. dict[int, HiddenStateResult] - Extract vector and compute D_eff
    3. dict[int, np.ndarray] - Compute D_eff directly

    Parameters:
        layers: Layer data in any supported format.
        variance_threshold: Threshold for D_eff computation.

    Returns:
        dict[int, int] mapping layer indices to D_eff values.
    """
    result: dict[int, int] = {}

    for layer_idx, value in layers.items():
        if isinstance(value, int):
            # Already a D_eff value
            result[layer_idx] = value
        elif isinstance(value, HiddenStateResult):
            # Extract vector and compute D_eff
            embeddings = value.vector
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            result[layer_idx] = calculate_d_eff(embeddings, variance_threshold)
        elif isinstance(value, np.ndarray):
            # Compute D_eff from array
            if value.ndim == 1:
                value = value.reshape(1, -1)
            result[layer_idx] = calculate_d_eff(value, variance_threshold)
        else:
            raise TypeError(
                f"Unsupported type for layer {layer_idx}: {type(value)}. "
                f"Expected int, HiddenStateResult, or np.ndarray."
            )

    return result


def compute_layer_similarity_matrix(
    layers: dict[int, HiddenStateResult | np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Compute pairwise cosine similarity matrix between layer representations.

    Creates an NxN similarity matrix where entry (i,j) is the cosine similarity
    between the hidden states at layers i and j.

    Parameters:
        layers: dict mapping layer indices to hidden states.

    Returns:
        tuple of:
        - similarity_matrix: NxN numpy array of cosine similarities
        - layer_order: List of layer indices in matrix order

    Example:
        >>> sim_matrix, layer_order = compute_layer_similarity_matrix(layers)
        >>> print(f"Similarity between layers {layer_order[0]} and {layer_order[1]}: "
        ...       f"{sim_matrix[0, 1]:.4f}")

    Notes:
        - Diagonal entries are always 1.0 (self-similarity)
        - Matrix is symmetric
        - Zero-norm vectors result in 0.0 similarity
    """
    if not layers:
        raise ValueError("layers dict must be non-empty")

    layer_order = sorted(layers.keys())
    n_layers = len(layer_order)

    # Extract and flatten vectors
    vectors: list[np.ndarray] = []
    for layer_idx in layer_order:
        value = layers[layer_idx]
        if isinstance(value, HiddenStateResult):
            vec = value.vector.flatten()
        else:
            vec = np.asarray(value).flatten()
        vectors.append(vec)

    # Compute similarity matrix
    similarity_matrix = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        for j in range(i, n_layers):
            vec_i = vectors[i]
            vec_j = vectors[j]

            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)

            if norm_i == 0 or norm_j == 0:
                sim = 0.0
            else:
                sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))

            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric

    return similarity_matrix, layer_order


# ============================================================================
# Module Self-Test
# ============================================================================


if __name__ == "__main__":
    print("Testing Layer Comparison Utilities...")

    # Generate test data simulating transformer hidden states
    np.random.seed(42)

    # Create mock layer data with decreasing dimensionality (compression pattern)
    n_samples = 50
    hidden_dim = 768

    # Simulate compression: early layers use more dimensions, later layers fewer
    test_layers_a: dict[int, np.ndarray] = {}
    test_layers_b: dict[int, np.ndarray] = {}

    for layer in range(12):
        # Model A: More compression (lower D_eff in later layers)
        rank_a = max(10, 100 - layer * 7)  # Decreasing rank
        test_layers_a[layer] = np.random.randn(n_samples, rank_a) @ np.random.randn(rank_a, hidden_dim)

        # Model B: Less compression (more stable D_eff)
        rank_b = max(10, 80 - layer * 3)  # Less decrease
        test_layers_b[layer] = np.random.randn(n_samples, rank_b) @ np.random.randn(rank_b, hidden_dim)

    print("\n" + "=" * 60)
    print("Testing compute_layer_trajectory...")
    print("=" * 60)

    trajectory_a = compute_layer_trajectory(test_layers_a)
    print("\nModel A Trajectory:")
    print(f"  Layer indices: {trajectory_a.layer_indices}")
    print(f"  D_eff values: {trajectory_a.d_eff_values}")
    print(f"  Trend: {trajectory_a.trend}")
    print(f"  Monotonicity: {trajectory_a.monotonicity:.4f}")
    print(f"  Min D_eff: {trajectory_a.min_d_eff} at layer {trajectory_a.min_layer}")
    print(f"  Max D_eff: {trajectory_a.max_d_eff} at layer {trajectory_a.max_layer}")

    trajectory_b = compute_layer_trajectory(test_layers_b)
    print("\nModel B Trajectory:")
    print(f"  Trend: {trajectory_b.trend}")
    print(f"  Monotonicity: {trajectory_b.monotonicity:.4f}")

    print("\n" + "=" * 60)
    print("Testing compare_layer_deff...")
    print("=" * 60)

    comparison = compare_layer_deff(test_layers_a, test_layers_b)
    print("\nComparison Results:")
    print(f"  Common layers: {comparison.common_layers}")
    print(f"  Mean diff (A - B): {comparison.mean_diff:.2f}")
    print(f"  Abs mean diff: {comparison.abs_mean_diff:.2f}")
    print(f"  Correlation: {comparison.correlation:.4f}")
    print(f"  Max diff at layer {comparison.max_diff_layer}: {comparison.max_diff_value}")
    print(f"  Divergence quality: {comparison.divergence_quality}")

    print("\n" + "=" * 60)
    print("Testing find_bottleneck_layers...")
    print("=" * 60)

    # Create test data with obvious bottleneck
    bottleneck_test: dict[int, int] = {
        0: 100,
        1: 95,
        2: 90,
        3: 50,  # Bottleneck here (44% drop)
        4: 48,
        5: 45,
        6: 30,  # Another bottleneck (33% drop)
        7: 28,
    }

    bottlenecks = find_bottleneck_layers(bottleneck_test, threshold=0.20)
    print("\nBottleneck Analysis:")
    print(f"  Input D_eff: {bottleneck_test}")
    print(f"  Bottleneck layers: {bottlenecks.bottleneck_layers}")
    print(f"  Drop magnitudes: {bottlenecks.drop_magnitudes}")
    print(f"  Severity: {bottlenecks.severity}")
    print(f"  Max drop at layer {bottlenecks.max_drop_layer}: {bottlenecks.max_drop_magnitude:.2%}")

    # Test with pre-computed D_eff from trajectory
    bottlenecks_a = find_bottleneck_layers(trajectory_a.trajectory)
    print("\nModel A Bottlenecks:")
    print(f"  Has bottlenecks: {bottlenecks_a.has_bottlenecks}")
    print(f"  Bottleneck layers: {bottlenecks_a.bottleneck_layers}")
    print(f"  Severity: {bottlenecks_a.severity}")

    print("\n" + "=" * 60)
    print("Testing compute_layer_similarity_matrix...")
    print("=" * 60)

    # Use a smaller subset for similarity matrix
    small_layers = {i: test_layers_a[i] for i in range(4)}
    sim_matrix, layer_order = compute_layer_similarity_matrix(small_layers)
    print(f"\nSimilarity Matrix (layers {layer_order}):")
    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Matrix:\n{np.round(sim_matrix, 4)}")
    print(f"  Diagonal (self-similarity): {np.diag(sim_matrix)}")

    print("\n" + "=" * 60)
    print("All Layer Comparison Utility tests passed!")
    print("=" * 60)
