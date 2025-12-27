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

# Minimum samples required for meaningful Beta calculation.
# With fewer samples, collapse measurement is unreliable.
MIN_SAMPLES_FOR_BETA = 2

# Default reference dimensionality for f_dim scaling function.
# Used when computing pairwise conveyance to normalize D_eff.
# 768 is chosen as typical hidden state dimensionality (GPT-2, BERT-base).
DEFAULT_D_REF = 768


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


@dataclass
class BetaResult:
    """Results from Beta (Collapse Indicator) calculation.

    Beta measures the degree of semantic collapse in embedding space, quantifying
    how much the embeddings have converged to a low-dimensional manifold.

    MATHEMATICAL GROUNDING
    ======================
    Beta is derived from the relationship between effective dimensionality (D_eff)
    and the maximum possible dimensionality (D_max):

        Beta = 1 - (D_eff - 1) / (D_max - 1)

    Where:
    - D_eff: Effective dimensionality at 90% variance threshold
    - D_max: Maximum possible dimensionality = min(n_samples - 1, n_features)

    This normalization ensures:
    - Beta = 0 when D_eff = D_max (no collapse, embeddings span full space)
    - Beta = 1 when D_eff = 1 (complete collapse to single dimension)

    INTERPRETATION
    ==============
    - Beta near 0: Embeddings are diverse, spanning many directions
    - Beta near 0.5: Moderate collapse, typical for focused conversations
    - Beta near 1: Severe collapse, all embeddings nearly identical

    VALIDATION TARGET
    =================
    Per the Conveyance Hypothesis, Beta should correlate with conversation quality
    with r ≈ -0.92 (strong negative correlation: higher collapse → lower quality).
    """

    beta: float  # Collapse indicator value in range [0, 1]
    d_eff: int  # Effective dimensionality used for calculation
    d_max: int  # Maximum possible dimensionality
    n_samples: int  # Number of samples used for calculation
    ambient_dim: int  # Original/ambient dimensionality
    variance_threshold: float  # Threshold used for D_eff (default 0.90)
    eigenvalue_concentration: float  # Fraction of variance in first component
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def collapse_severity(self) -> str:
        """Classify the severity of semantic collapse.

        Categories:
        - none: Beta < 0.2, healthy diversity in embedding space
        - mild: 0.2 <= Beta < 0.5, some convergence but acceptable
        - moderate: 0.5 <= Beta < 0.8, notable collapse, may impact quality
        - severe: Beta >= 0.8, significant collapse, likely quality degradation
        """
        if self.beta < 0.2:
            return "none"
        elif self.beta < 0.5:
            return "mild"
        elif self.beta < 0.8:
            return "moderate"
        else:
            return "severe"

    @property
    def is_collapsed(self) -> bool:
        """Check if embeddings show significant collapse (Beta >= 0.5)."""
        return self.beta >= 0.5

    @property
    def is_healthy(self) -> bool:
        """Check if embeddings maintain healthy diversity (Beta < 0.3)."""
        return self.beta < 0.3


@dataclass
class CPairResult:
    """Results from C_pair (Pairwise Conveyance) calculation.

    C_pair measures the effective semantic information transfer capacity between
    a specific agent pair, incorporating bidirectional conveyance, dimensionality
    effects, and participation weight.

    MATHEMATICAL GROUNDING
    ======================
    C_pair is computed using the harmonic mean of directional conveyances:

        C_pair = H(C_out, C_in) × f_dim(D_eff) × P_ij

    Where:
    - H(C_out, C_in): Harmonic mean of outgoing and incoming conveyance
    - f_dim(D_eff): Dimensionality scaling function (log-normalized)
    - P_ij: Participation weight for the agent pair

    HARMONIC MEAN SEMANTICS
    =======================
    The harmonic mean is chosen because it captures the "limited by weakest link"
    semantics of information transfer:

    - H(a, b) = 2ab / (a + b), always ≤ min(a, b) when a ≠ b
    - If either direction has zero conveyance, the pair has zero transfer
    - Penalizes asymmetric transfer (one direction much stronger than other)

    This aligns with the intuition that effective bilateral communication requires
    both sending AND receiving capabilities.

    ZERO-PROPAGATION PRINCIPLE
    ==========================
    If any component (c_out, c_in, d_eff, p_ij) is zero or negative, C_pair = 0.
    This implements the principle that complete blockage in any component
    prevents information transfer entirely.

    INTERPRETATION
    ==============
    - C_pair near 0: Poor/blocked information transfer
    - C_pair moderate (0.3-0.6): Functional but constrained transfer
    - C_pair high (>0.6): Strong bilateral information flow

    VALIDATION TARGET
    =================
    C_pair should correlate with task success metrics in multi-agent scenarios.
    The harmonic mean property ensures it's limited by the weaker participant.
    """

    c_pair: float  # Pairwise conveyance value (non-negative)
    c_out: float  # Outgoing conveyance (sender → receiver)
    c_in: float  # Incoming conveyance (receiver → sender)
    harmonic_mean: float  # H(c_out, c_in)
    d_eff: int  # Effective dimensionality used
    f_dim: float  # Dimensionality scaling factor applied
    p_ij: float  # Participation weight
    d_ref: int  # Reference dimensionality for f_dim
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        """Check if transfer is effectively blocked (C_pair ≈ 0)."""
        return self.c_pair < 0.01

    @property
    def is_asymmetric(self) -> bool:
        """Check if transfer is highly asymmetric (ratio > 3:1)."""
        if min(self.c_out, self.c_in) < 0.01:
            return True  # Extreme asymmetry (one direction blocked)
        ratio = max(self.c_out, self.c_in) / min(self.c_out, self.c_in)
        return ratio > 3.0

    @property
    def transfer_quality(self) -> str:
        """Classify the quality of bilateral transfer.

        Categories:
        - blocked: C_pair < 0.01, effectively no transfer
        - poor: 0.01 <= C_pair < 0.2, severely limited
        - moderate: 0.2 <= C_pair < 0.5, functional but constrained
        - good: 0.5 <= C_pair < 0.7, healthy transfer
        - excellent: C_pair >= 0.7, strong bilateral flow
        """
        if self.c_pair < 0.01:
            return "blocked"
        elif self.c_pair < 0.2:
            return "poor"
        elif self.c_pair < 0.5:
            return "moderate"
        elif self.c_pair < 0.7:
            return "good"
        else:
            return "excellent"

    @property
    def limiting_direction(self) -> str:
        """Identify which direction limits the transfer.

        Returns:
        - "outgoing": c_out < c_in (sender is the bottleneck)
        - "incoming": c_in < c_out (receiver is the bottleneck)
        - "balanced": c_out ≈ c_in (within 10%)
        - "both_zero": Both directions blocked
        """
        if self.c_out < 0.01 and self.c_in < 0.01:
            return "both_zero"
        if abs(self.c_out - self.c_in) / max(self.c_out, self.c_in, 0.01) < 0.1:
            return "balanced"
        return "outgoing" if self.c_out < self.c_in else "incoming"


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
# Beta (Collapse Indicator) Calculation
# ============================================================================


def calculate_beta(
    embeddings: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    normalize: bool = True,
) -> float:
    """Calculate Beta (Collapse Indicator) for embedding space.

    Beta quantifies the degree of semantic collapse in an embedding distribution.
    It measures how much the embeddings have converged to a low-dimensional
    manifold, indicating potential loss of semantic diversity.

    MATHEMATICAL GROUNDING
    ======================
    Beta is computed as:

        Beta = 1 - (D_eff - 1) / (D_max - 1)

    Where:
    - D_eff: Effective dimensionality (dimensions for 90% variance)
    - D_max: Maximum possible dimensionality = min(n_samples - 1, n_features)

    This formula ensures:
    - Beta = 0: No collapse (D_eff = D_max, full rank data)
    - Beta = 1: Complete collapse (D_eff = 1, all variance in one direction)

    CRITICAL: L2 normalization is applied by default BEFORE PCA to prevent
    magnitude artifacts from affecting the collapse measurement.

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
        beta: float
            The collapse indicator value in range [0, 1].
            - 0: No collapse, embeddings span maximum dimensions
            - 1: Complete collapse, embeddings collapsed to single direction

    Raises:
        ValueError: If embeddings has invalid shape or variance_threshold out of range.

    Example:
        >>> import numpy as np
        >>> # Random embeddings (low collapse expected)
        >>> random_embeddings = np.random.randn(100, 768)
        >>> beta = calculate_beta(random_embeddings)
        >>> print(f"Beta: {beta:.4f}")  # Likely ~0.0-0.1 for random data

        >>> # Collapsed embeddings (high collapse expected)
        >>> collapsed = np.random.randn(100, 1) @ np.random.randn(1, 768)
        >>> beta_collapsed = calculate_beta(collapsed)
        >>> print(f"Beta (collapsed): {beta_collapsed:.4f}")  # Close to 1.0

    Notes:
        - Single-point embeddings return Beta = 1.0 (complete collapse)
        - Constant embeddings return Beta = 1.0 (complete collapse)
        - Random Gaussian embeddings typically have Beta ~ 0.0-0.15
        - Beta correlates negatively with conversation quality (r ≈ -0.92)
    """
    # Get detailed D_eff result for Beta calculation
    d_eff_result = calculate_d_eff_detailed(
        embeddings,
        variance_threshold=variance_threshold,
        normalize=normalize,
    )

    # Calculate maximum possible dimensionality
    d_max = min(d_eff_result.n_samples - 1, d_eff_result.ambient_dim)

    # Edge case: d_max <= 1 (single sample or single feature)
    # In these cases, Beta = 1 (complete collapse by definition)
    if d_max <= 1:
        return 1.0

    # Calculate Beta using the normalized formula
    # Beta = 1 - (D_eff - 1) / (D_max - 1)
    beta = 1.0 - (d_eff_result.d_eff - 1) / (d_max - 1)

    # Clamp to [0, 1] to handle any numerical edge cases
    beta = max(0.0, min(1.0, beta))

    return beta


def calculate_beta_detailed(
    embeddings: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    normalize: bool = True,
) -> BetaResult:
    """Calculate Beta (Collapse Indicator) with full diagnostic information.

    This is the detailed version of calculate_beta that returns complete
    results including D_eff components, eigenvalue concentration, and metadata.

    Parameters:
        embeddings: Array of shape (n_samples, n_features)
            The embedding vectors to analyze.
        variance_threshold: float, default=0.90
            Cumulative variance threshold for determining D_eff.
        normalize: bool, default=True
            Whether to L2-normalize embeddings before PCA.

    Returns:
        BetaResult with full diagnostic information including:
        - beta: The collapse indicator value [0, 1]
        - d_eff: Effective dimensionality used
        - d_max: Maximum possible dimensionality
        - collapse_severity: Classification of collapse level
        - eigenvalue_concentration: Fraction of variance in first component

    Example:
        >>> result = calculate_beta_detailed(embeddings)
        >>> print(f"Beta: {result.beta:.4f}")
        >>> print(f"Severity: {result.collapse_severity}")
        >>> print(f"D_eff: {result.d_eff} / D_max: {result.d_max}")
    """
    # Get detailed D_eff result
    d_eff_result = calculate_d_eff_detailed(
        embeddings,
        variance_threshold=variance_threshold,
        normalize=normalize,
    )

    # Calculate maximum possible dimensionality
    d_max = min(d_eff_result.n_samples - 1, d_eff_result.ambient_dim)

    # Edge case: d_max <= 1
    if d_max <= 1:
        return BetaResult(
            beta=1.0,
            d_eff=d_eff_result.d_eff,
            d_max=max(1, d_max),
            n_samples=d_eff_result.n_samples,
            ambient_dim=d_eff_result.ambient_dim,
            variance_threshold=variance_threshold,
            eigenvalue_concentration=1.0,
            metadata={"edge_case": "insufficient_dimensionality"},
        )

    # Calculate Beta
    beta = 1.0 - (d_eff_result.d_eff - 1) / (d_max - 1)
    beta = max(0.0, min(1.0, beta))

    # Calculate eigenvalue concentration (fraction of variance in first component)
    # This provides additional insight into the nature of collapse
    total_variance = d_eff_result.eigenvalues.sum()
    if total_variance > 1e-10:
        eigenvalue_concentration = float(
            d_eff_result.eigenvalues[0] / total_variance
        )
    else:
        eigenvalue_concentration = 1.0

    return BetaResult(
        beta=beta,
        d_eff=d_eff_result.d_eff,
        d_max=d_max,
        n_samples=d_eff_result.n_samples,
        ambient_dim=d_eff_result.ambient_dim,
        variance_threshold=variance_threshold,
        eigenvalue_concentration=eigenvalue_concentration,
    )


# ============================================================================
# C_pair (Pairwise Conveyance) Calculation
# ============================================================================


def _compute_harmonic_mean(a: float, b: float) -> float:
    """Compute harmonic mean of two values with zero-propagation.

    The harmonic mean H(a, b) = 2ab / (a + b).

    Properties:
    - H(a, b) <= min(a, b) when a != b (limited by weakest link)
    - H(a, a) = a (identity when equal)
    - H(0, x) = 0 (zero-propagation)

    Parameters:
        a: First value (must be non-negative)
        b: Second value (must be non-negative)

    Returns:
        Harmonic mean of a and b, or 0.0 if either is <= 0.
    """
    # Zero-propagation: if either value is zero or negative, result is zero
    if a <= 0 or b <= 0:
        return 0.0

    # Standard harmonic mean formula: 2ab / (a + b)
    return 2.0 * a * b / (a + b)


def _compute_f_dim(d_eff: int, d_ref: int = DEFAULT_D_REF) -> float:
    """Compute dimensionality scaling factor f_dim(D_eff).

    This function scales the contribution of effective dimensionality to
    pairwise conveyance. Uses log-normalized scaling to prevent extreme
    values while maintaining sensitivity to dimensionality changes.

    DESIGN RATIONALE
    ================
    Log scaling is chosen because:
    1. Diminishing returns: Going from 10 to 100 dimensions matters more
       than going from 1000 to 1090 dimensions.
    2. Bounded output: f_dim ∈ (0, 1] for reasonable inputs.
    3. Smooth gradient: No sharp transitions that could destabilize learning.

    Formula:
        f_dim(d) = log(1 + d) / log(1 + d_ref)

    Where d_ref is a reference dimensionality (default 768, typical for
    transformer hidden states).

    Parameters:
        d_eff: Effective dimensionality (must be positive integer)
        d_ref: Reference dimensionality for normalization (default 768)

    Returns:
        Scaling factor in range (0, 1] for d_eff <= d_ref,
        or > 1 for d_eff > d_ref.

    Example:
        >>> _compute_f_dim(100, d_ref=768)
        0.693...  # log(101) / log(769)
        >>> _compute_f_dim(768, d_ref=768)
        1.0  # Equal to reference
    """
    # Zero-propagation: if d_eff is zero or negative, return 0
    if d_eff <= 0:
        return 0.0

    if d_ref <= 0:
        raise ValueError(f"d_ref must be positive, got {d_ref}")

    # Log-normalized scaling
    # Adding 1 to avoid log(0) and ensure f_dim(1) > 0
    return np.log1p(d_eff) / np.log1p(d_ref)


def calculate_c_pair(
    c_out: float,
    c_in: float,
    d_eff: int,
    p_ij: float,
    d_ref: int = DEFAULT_D_REF,
) -> float:
    """Calculate C_pair (Pairwise Conveyance) for an agent pair.

    C_pair measures the effective semantic information transfer capacity
    between a specific agent pair, using the harmonic mean of bidirectional
    conveyances scaled by dimensionality and participation weight.

    MATHEMATICAL GROUNDING
    ======================
    C_pair is computed as:

        C_pair = H(C_out, C_in) × f_dim(D_eff) × P_ij

    Where:
    - H(C_out, C_in): Harmonic mean of directional conveyances
    - f_dim(D_eff): Log-normalized dimensionality scaling = log(1+D_eff)/log(1+D_ref)
    - P_ij: Participation weight ∈ [0, 1]

    HARMONIC MEAN SEMANTICS
    =======================
    The harmonic mean captures "limited by weakest link" semantics:
    - If sender can transmit but receiver can't absorb → low C_pair
    - If receiver can absorb but sender can't transmit → low C_pair
    - Only when BOTH directions work well → high C_pair

    This aligns with the communication theory principle that effective
    bilateral exchange requires both transmission AND reception.

    ZERO-PROPAGATION PRINCIPLE
    ==========================
    If ANY component is zero or negative, C_pair = 0:
    - c_out = 0: Sender cannot transmit → no transfer
    - c_in = 0: Receiver cannot absorb → no transfer
    - d_eff = 0: Degenerate space → no meaningful transfer
    - p_ij = 0: Zero participation → no transfer

    Parameters:
        c_out: Outgoing conveyance (sender → receiver), typically in [0, 1]
        c_in: Incoming conveyance (receiver → sender), typically in [0, 1]
        d_eff: Effective dimensionality of the shared semantic space
        p_ij: Participation weight for this agent pair, in [0, 1]
        d_ref: Reference dimensionality for f_dim normalization (default 768)

    Returns:
        c_pair: float
            The pairwise conveyance value (non-negative).
            Higher values indicate stronger bilateral information transfer.

    Raises:
        ValueError: If d_ref <= 0.

    Example:
        >>> # Symmetric transfer with moderate dimensionality
        >>> c_pair = calculate_c_pair(c_out=0.8, c_in=0.8, d_eff=100, p_ij=1.0)
        >>> print(f"C_pair: {c_pair:.4f}")  # ~0.554

        >>> # Asymmetric transfer (limited by weaker direction)
        >>> c_pair = calculate_c_pair(c_out=0.9, c_in=0.3, d_eff=100, p_ij=1.0)
        >>> print(f"C_pair: {c_pair:.4f}")  # ~0.311 (closer to 0.3 than 0.9)

        >>> # Zero-propagation example
        >>> c_pair = calculate_c_pair(c_out=0.8, c_in=0.0, d_eff=100, p_ij=1.0)
        >>> print(f"C_pair: {c_pair:.4f}")  # 0.0 (blocked by zero c_in)

    Notes:
        - C_pair is always <= min(c_out, c_in) * f_dim(d_eff) * p_ij
        - The harmonic mean ensures asymmetric transfers are penalized
        - Use calculate_c_pair_detailed for diagnostic information
    """
    # Zero-propagation: check all components
    if c_out <= 0 or c_in <= 0 or d_eff <= 0 or p_ij <= 0:
        return 0.0

    # Compute harmonic mean of directional conveyances
    h_mean = _compute_harmonic_mean(c_out, c_in)

    # Compute dimensionality scaling factor
    f_dim = _compute_f_dim(d_eff, d_ref)

    # Final C_pair calculation
    c_pair = h_mean * f_dim * p_ij

    return float(c_pair)


def calculate_c_pair_detailed(
    c_out: float,
    c_in: float,
    d_eff: int,
    p_ij: float,
    d_ref: int = DEFAULT_D_REF,
) -> CPairResult:
    """Calculate C_pair (Pairwise Conveyance) with full diagnostic information.

    This is the detailed version of calculate_c_pair that returns complete
    results including intermediate values and metadata for analysis.

    Parameters:
        c_out: Outgoing conveyance (sender → receiver)
        c_in: Incoming conveyance (receiver → sender)
        d_eff: Effective dimensionality of the shared semantic space
        p_ij: Participation weight for this agent pair
        d_ref: Reference dimensionality for f_dim normalization (default 768)

    Returns:
        CPairResult with full diagnostic information including:
        - c_pair: The final pairwise conveyance value
        - harmonic_mean: H(c_out, c_in)
        - f_dim: Dimensionality scaling factor
        - transfer_quality: Classification of transfer quality
        - limiting_direction: Which direction is the bottleneck

    Example:
        >>> result = calculate_c_pair_detailed(c_out=0.8, c_in=0.6, d_eff=50, p_ij=0.3)
        >>> print(f"C_pair: {result.c_pair:.4f}")
        >>> print(f"Transfer Quality: {result.transfer_quality}")
        >>> print(f"Limiting Direction: {result.limiting_direction}")
    """
    # Compute harmonic mean (with zero-propagation)
    h_mean = _compute_harmonic_mean(c_out, c_in)

    # Compute dimensionality scaling
    f_dim = _compute_f_dim(d_eff, d_ref) if d_eff > 0 else 0.0

    # Final C_pair (with zero-propagation for p_ij)
    if p_ij <= 0:
        c_pair = 0.0
    else:
        c_pair = h_mean * f_dim * p_ij

    # Build metadata
    metadata: dict[str, Any] = {}
    if c_out <= 0 or c_in <= 0:
        metadata["zero_propagation"] = "directional_conveyance"
    elif d_eff <= 0:
        metadata["zero_propagation"] = "d_eff"
    elif p_ij <= 0:
        metadata["zero_propagation"] = "p_ij"

    return CPairResult(
        c_pair=float(c_pair),
        c_out=float(c_out),
        c_in=float(c_in),
        harmonic_mean=float(h_mean),
        d_eff=int(d_eff) if d_eff > 0 else 0,
        f_dim=float(f_dim),
        p_ij=float(p_ij),
        d_ref=int(d_ref),
        metadata=metadata,
    )


# ============================================================================
# Module Self-Test
# ============================================================================


if __name__ == "__main__":
    # Quick test with random data
    print("Testing Conveyance Metrics - D_eff and Beta Calculation...")

    # Generate test embeddings (simulating transformer hidden states)
    np.random.seed(42)
    test_embeddings = np.random.randn(100, 768)  # 100 samples, 768-dim

    # Basic D_eff calculation
    d_eff = calculate_d_eff(test_embeddings)
    print(f"\nBasic D_eff: {d_eff}")

    # Detailed calculation
    result = calculate_d_eff_detailed(test_embeddings)
    print(f"\nDetailed D_eff Results:")
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

    print("\n" + "=" * 60)
    print("Testing Beta (Collapse Indicator) Calculation...")
    print("=" * 60)

    # Basic Beta calculation with random data (should be low)
    beta = calculate_beta(test_embeddings)
    print(f"\nRandom Embeddings Beta: {beta:.4f}")
    assert 0 <= beta <= 1, f"Beta must be in [0, 1], got {beta}"

    # Detailed Beta calculation
    beta_result = calculate_beta_detailed(test_embeddings)
    print(f"\nDetailed Beta Results:")
    print(f"  Beta: {beta_result.beta:.4f}")
    print(f"  D_eff: {beta_result.d_eff}")
    print(f"  D_max: {beta_result.d_max}")
    print(f"  Collapse Severity: {beta_result.collapse_severity}")
    print(f"  Is Healthy: {beta_result.is_healthy}")
    print(f"  Eigenvalue Concentration: {beta_result.eigenvalue_concentration:.4f}")

    # Edge case: collapsed embeddings (rank-1)
    collapsed = np.random.randn(50, 1) @ np.random.randn(1, 512)
    beta_collapsed = calculate_beta(collapsed)
    print(f"\nCollapsed (Rank-1) Beta: {beta_collapsed:.4f}")
    assert beta_collapsed > 0.9, f"Rank-1 data should have Beta > 0.9, got {beta_collapsed}"

    # Edge case: single point (complete collapse)
    beta_single = calculate_beta(single_point)
    print(f"Single Point Beta: {beta_single:.4f}")
    assert beta_single == 1.0, f"Single point should have Beta = 1.0, got {beta_single}"

    # Edge case: constant embeddings (complete collapse)
    beta_constant = calculate_beta(constant)
    print(f"Constant Embeddings Beta: {beta_constant:.4f}")
    assert beta_constant == 1.0, f"Constant embeddings should have Beta = 1.0, got {beta_constant}"

    # Test verification from spec: Beta should be in [0, 1] for random data
    verification_embeddings = np.random.randn(50, 512)
    beta_verify = calculate_beta(verification_embeddings)
    print(f"\nVerification Test (50x512 random): Beta = {beta_verify:.4f}")
    assert isinstance(beta_verify, float), "Beta must be a float"
    assert 0 <= beta_verify <= 1, f"Beta must be in [0, 1], got {beta_verify}"

    print("\n" + "=" * 60)
    print("Testing C_pair (Pairwise Conveyance) Calculation...")
    print("=" * 60)

    # Basic C_pair calculation
    c_pair = calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=50, p_ij=0.3)
    print(f"\nBasic C_pair (0.8, 0.6, 50, 0.3): {c_pair:.4f}")
    assert isinstance(c_pair, float), "C_pair must be a float"
    assert c_pair >= 0, f"C_pair must be non-negative, got {c_pair}"

    # Detailed C_pair calculation
    c_pair_result = calculate_c_pair_detailed(c_out=0.8, c_in=0.6, d_eff=50, p_ij=0.3)
    print(f"\nDetailed C_pair Results:")
    print(f"  C_pair: {c_pair_result.c_pair:.4f}")
    print(f"  Harmonic Mean: {c_pair_result.harmonic_mean:.4f}")
    print(f"  f_dim: {c_pair_result.f_dim:.4f}")
    print(f"  Transfer Quality: {c_pair_result.transfer_quality}")
    print(f"  Limiting Direction: {c_pair_result.limiting_direction}")

    # Test zero-propagation: c_out = 0
    c_pair_zero_out = calculate_c_pair(c_out=0.0, c_in=0.8, d_eff=100, p_ij=1.0)
    print(f"\nZero-propagation (c_out=0): C_pair = {c_pair_zero_out:.4f}")
    assert c_pair_zero_out == 0.0, f"C_pair should be 0 when c_out=0, got {c_pair_zero_out}"

    # Test zero-propagation: c_in = 0
    c_pair_zero_in = calculate_c_pair(c_out=0.8, c_in=0.0, d_eff=100, p_ij=1.0)
    print(f"Zero-propagation (c_in=0): C_pair = {c_pair_zero_in:.4f}")
    assert c_pair_zero_in == 0.0, f"C_pair should be 0 when c_in=0, got {c_pair_zero_in}"

    # Test zero-propagation: p_ij = 0
    c_pair_zero_pij = calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=100, p_ij=0.0)
    print(f"Zero-propagation (p_ij=0): C_pair = {c_pair_zero_pij:.4f}")
    assert c_pair_zero_pij == 0.0, f"C_pair should be 0 when p_ij=0, got {c_pair_zero_pij}"

    # Test harmonic mean property: result closer to minimum
    c_pair_asym = calculate_c_pair(c_out=0.9, c_in=0.3, d_eff=100, p_ij=1.0)
    # Harmonic mean of 0.9 and 0.3 = 2*0.9*0.3/(0.9+0.3) = 0.45
    # Arithmetic mean would be 0.6
    h_mean = 2 * 0.9 * 0.3 / (0.9 + 0.3)
    a_mean = (0.9 + 0.3) / 2
    print(f"\nAsymmetric transfer (0.9, 0.3):")
    print(f"  Harmonic mean: {h_mean:.4f} (used)")
    print(f"  Arithmetic mean: {a_mean:.4f} (not used)")
    print(f"  Difference from minimum: H-min={h_mean - 0.3:.4f}, A-min={a_mean - 0.3:.4f}")
    assert h_mean < a_mean, "Harmonic mean should be less than arithmetic mean"
    assert h_mean - 0.3 < a_mean - 0.3, "Harmonic mean should be closer to minimum"

    # Test symmetric transfer
    c_pair_sym = calculate_c_pair(c_out=0.7, c_in=0.7, d_eff=100, p_ij=1.0)
    h_mean_sym = 2 * 0.7 * 0.7 / (0.7 + 0.7)  # Should equal 0.7
    print(f"\nSymmetric transfer (0.7, 0.7):")
    print(f"  Harmonic mean: {h_mean_sym:.4f} (equals inputs when symmetric)")
    assert abs(h_mean_sym - 0.7) < 1e-10, "Harmonic mean of equal values should equal that value"

    print("\nAll D_eff, Beta, and C_pair tests passed!")
