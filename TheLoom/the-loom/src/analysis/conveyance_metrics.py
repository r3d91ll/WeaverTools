"""Conveyance Metrics for Semantic Information Transfer Analysis.

This module implements quantitative metrics for measuring semantic information
transfer between agents in multi-agent conversations, based on the Conveyance
Hypothesis framework.

MATHEMATICAL GROUNDING
======================
- Effective Dimensionality (D_eff) via PCA: Directly grounded in the manifold
  hypothesis and established dimensionality estimation techniques. Uses PCA
  eigenvalue analysis to determine the number of dimensions required to
  capture a specified percentage of variance (default 90%).

  Reference: Whiteley et al. "Statistical exploration of the Manifold Hypothesis"
  (arXiv:2208.11665). The Latent Metric Model explains why data concentrates
  on low-dimensional manifolds, justifying PCA-based intrinsic dimension estimation.

IMPLEMENTATION NOTES
====================
- L2 normalization is applied BEFORE PCA to prevent magnitude artifacts from
  dominating the dimensionality estimate. This ensures we measure directional
  diversity rather than magnitude variation.

- Uses `np.linalg.eigvalsh` for symmetric covariance matrices (faster than
  generic `np.linalg.eig` and numerically more stable).

- Eigenvalues are sorted descending for cumulative variance computation.

Integration: Designed to work with TheLoom's HiddenStateResult class and
complements the kakeya_geometry module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Constants
# ============================================================================

# Minimum samples required for meaningful dimensionality analysis.
# With fewer samples, PCA produces degenerate or unreliable results.
MIN_SAMPLES_FOR_D_EFF = 2

# Default variance threshold for effective dimensionality calculation.
# 90% is the established standard per the Conveyance Hypothesis.
DEFAULT_VARIANCE_THRESHOLD = 0.90


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class EffectiveDimensionalityResult:
    """Results from effective dimensionality (D_eff) calculation.

    D_eff measures the intrinsic dimensionality of embedding space by finding
    the minimum number of principal components needed to explain a specified
    percentage of variance (default 90%).

    Interpretation:
    - D_eff / ambient_dim: Fraction of space utilized by embeddings
    - Low D_eff: Embeddings concentrate on low-dimensional manifold
    - High D_eff: Embeddings span many directions in the space
    - D_eff close to min(n_samples-1, ambient_dim): Near-full rank data

    Mathematical Basis:
    Given n embeddings in d-dimensional space:
    1. L2-normalize to remove magnitude effects
    2. Center the normalized embeddings
    3. Compute covariance matrix C = X^T X / (n-1)
    4. Find eigenvalues lambda_1 >= lambda_2 >= ... >= lambda_d
    5. Compute cumulative variance ratio: cumvar_k = sum(lambda_1..k) / sum(all)
    6. D_eff = argmin_k { cumvar_k >= threshold }
    """

    d_eff: int  # Effective dimensionality (dimensions for 90% variance)
    ambient_dim: int  # Original/ambient dimensionality
    n_samples: int  # Number of samples used for calculation
    variance_threshold: float  # Threshold used (default 0.90)
    eigenvalues: NDArray[np.floating[Any]]  # All eigenvalues (sorted descending, 1D)
    cumulative_variance: NDArray[np.floating[Any]]  # Cumulative variance ratios (1D)
    variance_ratio: float  # d_eff / ambient_dim
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_low_dimensional(self) -> bool:
        """Check if data concentrates on low-dimensional manifold."""
        return self.variance_ratio < 0.3

    @property
    def is_full_rank(self) -> bool:
        """Check if data is approximately full rank."""
        # Full rank if D_eff is close to min(n_samples-1, ambient_dim)
        max_possible = min(self.n_samples - 1, self.ambient_dim)
        return self.d_eff >= 0.9 * max_possible if max_possible > 0 else False

    @property
    def dimensionality_quality(self) -> str:
        """Classify dimensionality characteristics."""
        if self.variance_ratio < 0.1:
            return "degenerate"
        elif self.variance_ratio < 0.3:
            return "low_dimensional"
        elif self.variance_ratio < 0.6:
            return "moderate"
        else:
            return "high_dimensional"


# ============================================================================
# D_eff Calculation
# ============================================================================


def calculate_d_eff(
    embeddings: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    normalize: bool = True,
) -> int:
    """Calculate effective dimensionality (D_eff) of embedding space.

    Effective dimensionality is the number of principal components needed to
    explain the specified percentage of variance. This measures how many
    dimensions the embeddings actually utilize, regardless of ambient dimension.

    CRITICAL: L2 normalization is applied by default BEFORE PCA to prevent
    magnitude artifacts from dominating the dimensionality estimate.

    Parameters:
        embeddings: Array of shape (n_samples, n_features)
            The embedding vectors to analyze. Each row is one embedding.
        variance_threshold: float, default=0.90
            Cumulative variance threshold for determining D_eff.
            Standard value is 0.90 per Conveyance Hypothesis.
        normalize: bool, default=True
            Whether to L2-normalize embeddings before PCA.
            HIGHLY RECOMMENDED to prevent magnitude artifacts.

    Returns:
        d_eff: int
            The effective dimensionality (number of components for threshold).
            Always >= 1 for valid input.

    Raises:
        ValueError: If embeddings has invalid shape or variance_threshold out of range.

    Example:
        >>> import numpy as np
        >>> embeddings = np.random.randn(100, 768)
        >>> d_eff = calculate_d_eff(embeddings)
        >>> print(f"D_eff: {d_eff}")  # Likely ~90-100 for random data

    Notes:
        - For n_samples < ambient_dim, max possible D_eff is n_samples - 1
        - Single-point embeddings return D_eff = 1
        - Constant embeddings (all identical) return D_eff = 1
        - Random Gaussian embeddings typically have D_eff ~ 0.9 * min(n, d)
    """
    # Input validation
    embeddings = np.asarray(embeddings)

    if embeddings.ndim == 1:
        # Single embedding - reshape to (1, n_features)
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array, got shape {embeddings.shape}"
        )

    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}"
        )

    n_samples, n_features = embeddings.shape

    # Edge case: single sample
    if n_samples == 1:
        return 1

    # Edge case: single feature
    if n_features == 1:
        return 1

    # Step 1: L2 normalization (CRITICAL - prevents magnitude artifacts)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms > 0, norms, 1.0)
        normalized = embeddings / norms
    else:
        normalized = embeddings

    # Step 2: Center the (normalized) embeddings
    centered = normalized - normalized.mean(axis=0)

    # Step 3: Compute covariance matrix
    # Use n_samples - 1 for unbiased estimator (Bessel's correction)
    cov = centered.T @ centered / (n_samples - 1)

    # Step 4: Compute eigenvalues using eigvalsh (optimized for symmetric matrices)
    # eigvalsh returns eigenvalues in ascending order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues = np.linalg.eigvalsh(cov)

    # Sort descending (eigvalsh returns ascending)
    eigenvalues = eigenvalues[::-1]

    # Handle numerical issues: clamp small negative eigenvalues to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Edge case: all eigenvalues are zero (constant embeddings)
    total_variance = eigenvalues.sum()
    if total_variance < 1e-10:
        return 1

    # Step 5: Calculate cumulative variance ratio
    cumulative_variance = np.cumsum(eigenvalues) / total_variance

    # Step 6: Find D_eff as first index where cumvar >= threshold
    # searchsorted returns the insertion point, which is the count of
    # elements < threshold. We need the first element >= threshold.
    d_eff = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)

    # Clamp to valid range [1, min(n_samples-1, n_features)]
    max_possible = min(n_samples - 1, n_features)
    d_eff = max(1, min(d_eff, max_possible))

    return d_eff


def calculate_d_eff_detailed(
    embeddings: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    normalize: bool = True,
) -> EffectiveDimensionalityResult:
    """Calculate effective dimensionality with full diagnostic information.

    This is the detailed version of calculate_d_eff that returns complete
    results including eigenvalues, cumulative variance, and metadata.

    Parameters:
        embeddings: Array of shape (n_samples, n_features)
            The embedding vectors to analyze.
        variance_threshold: float, default=0.90
            Cumulative variance threshold for determining D_eff.
        normalize: bool, default=True
            Whether to L2-normalize embeddings before PCA.

    Returns:
        EffectiveDimensionalityResult with full diagnostic information.

    Example:
        >>> result = calculate_d_eff_detailed(embeddings)
        >>> print(f"D_eff: {result.d_eff}")
        >>> print(f"Quality: {result.dimensionality_quality}")
        >>> print(f"Top 5 eigenvalues: {result.eigenvalues[:5]}")
    """
    # Input validation
    embeddings = np.asarray(embeddings)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array, got shape {embeddings.shape}"
        )

    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}"
        )

    n_samples, n_features = embeddings.shape

    # Edge case: single sample
    if n_samples == 1:
        return EffectiveDimensionalityResult(
            d_eff=1,
            ambient_dim=n_features,
            n_samples=1,
            variance_threshold=variance_threshold,
            eigenvalues=np.array([1.0]),
            cumulative_variance=np.array([1.0]),
            variance_ratio=1.0 / n_features,
            metadata={"edge_case": "single_sample"},
        )

    # L2 normalization
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = embeddings / norms
    else:
        normalized = embeddings

    # Center the embeddings
    centered = normalized - normalized.mean(axis=0)

    # Compute covariance matrix
    cov = centered.T @ centered / (n_samples - 1)

    # Compute eigenvalues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues = np.linalg.eigvalsh(cov)

    # Sort descending
    eigenvalues = eigenvalues[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Handle constant embeddings
    total_variance = eigenvalues.sum()
    if total_variance < 1e-10:
        return EffectiveDimensionalityResult(
            d_eff=1,
            ambient_dim=n_features,
            n_samples=n_samples,
            variance_threshold=variance_threshold,
            eigenvalues=eigenvalues,
            cumulative_variance=np.ones_like(eigenvalues),
            variance_ratio=1.0 / n_features,
            metadata={"edge_case": "constant_embeddings"},
        )

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(eigenvalues) / total_variance

    # Find D_eff
    d_eff = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    max_possible = min(n_samples - 1, n_features)
    d_eff = max(1, min(d_eff, max_possible))

    return EffectiveDimensionalityResult(
        d_eff=d_eff,
        ambient_dim=n_features,
        n_samples=n_samples,
        variance_threshold=variance_threshold,
        eigenvalues=eigenvalues,
        cumulative_variance=cumulative_variance,
        variance_ratio=d_eff / n_features,
    )


# ============================================================================
# Module Self-Test
# ============================================================================


if __name__ == "__main__":
    # Quick test with random data
    print("Testing Conveyance Metrics - D_eff Calculation...")

    # Generate test embeddings (simulating transformer hidden states)
    np.random.seed(42)
    test_embeddings = np.random.randn(100, 768)  # 100 samples, 768-dim

    # Basic D_eff calculation
    d_eff = calculate_d_eff(test_embeddings)
    print(f"\nBasic D_eff: {d_eff}")

    # Detailed calculation
    result = calculate_d_eff_detailed(test_embeddings)
    print(f"\nDetailed Results:")
    print(f"  D_eff: {result.d_eff}")
    print(f"  Ambient Dim: {result.ambient_dim}")
    print(f"  Variance Ratio: {result.variance_ratio:.4f}")
    print(f"  Quality: {result.dimensionality_quality}")
    print(f"  Is Low Dimensional: {result.is_low_dimensional}")
    print(f"  Top 5 Eigenvalues: {result.eigenvalues[:5]}")

    # Edge case: single point
    single_point = np.random.randn(1, 768)
    d_eff_single = calculate_d_eff(single_point)
    print(f"\nSingle Point D_eff: {d_eff_single}")
    assert d_eff_single == 1, "Single point should have D_eff = 1"

    # Edge case: constant embeddings
    constant = np.ones((100, 768))
    d_eff_constant = calculate_d_eff(constant)
    print(f"Constant Embeddings D_eff: {d_eff_constant}")
    assert d_eff_constant == 1, "Constant embeddings should have D_eff = 1"

    # Edge case: low-rank data
    low_rank = np.random.randn(100, 5) @ np.random.randn(5, 768)
    d_eff_low_rank = calculate_d_eff(low_rank)
    print(f"Low Rank (5) Embeddings D_eff: {d_eff_low_rank}")
    assert d_eff_low_rank <= 10, "Low rank data should have low D_eff"

    print("\nAll tests passed!")
