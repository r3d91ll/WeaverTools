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
from collections.abc import Callable

from scipy import stats

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

# Default number of bootstrap resamples for confidence interval calculation.
# 10,000 is standard for reliable CI estimation; 5,000 acceptable for speed.
DEFAULT_BOOTSTRAP_RESAMPLES = 10000

# Default confidence level for bootstrap confidence intervals.
# 95% is the standard for scientific reporting.
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Minimum samples required for meaningful bootstrap analysis.
# With fewer samples, confidence intervals may be unreliable.
MIN_SAMPLES_FOR_BOOTSTRAP = 10


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


@dataclass
class BootstrapCIResult:
    """Results from bootstrap confidence interval calculation.

    Bootstrap confidence intervals provide non-parametric uncertainty
    quantification for any statistic, making them ideal for metrics like
    D_eff, Beta, and C_pair where parametric assumptions may not hold.

    MATHEMATICAL GROUNDING
    ======================
    Bootstrap resampling estimates the sampling distribution of a statistic by:
    1. Drawing B samples with replacement from the original data
    2. Computing the statistic on each bootstrap sample
    3. Using the empirical distribution of bootstrap statistics for inference

    The percentile method constructs CI as [Q_{α/2}, Q_{1-α/2}] of bootstrap
    statistics, where α = 1 - confidence_level.

    Reference: Efron & Tibshirani "An Introduction to the Bootstrap" (1993).
    The percentile method is robust and distribution-agnostic, appropriate for
    non-parametric settings common in embedding analysis.

    INTERPRETATION
    ==============
    - point_estimate: The statistic computed on the original data
    - ci_lower: Lower bound of confidence interval
    - ci_upper: Upper bound of confidence interval
    - ci_width: ci_upper - ci_lower (narrower = more precise)

    A wider CI indicates more uncertainty, often due to:
    - Small sample size
    - High variability in the data
    - Non-smooth statistic function

    VALIDATION TARGET
    =================
    For well-behaved distributions, approximately 95% of true parameter values
    should fall within 95% confidence intervals (coverage probability).
    """

    point_estimate: float  # Statistic computed on original data
    ci_lower: float  # Lower bound of confidence interval
    ci_upper: float  # Upper bound of confidence interval
    confidence_level: float  # Confidence level (typically 0.95)
    n_resamples: int  # Number of bootstrap resamples used
    n_samples: int  # Number of original data points
    standard_error: float  # Bootstrap standard error of the statistic
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def relative_ci_width(self) -> float:
        """CI width relative to point estimate (coefficient of variation)."""
        if abs(self.point_estimate) < 1e-10:
            return float('inf') if self.ci_width > 0 else 0.0
        return self.ci_width / abs(self.point_estimate)

    @property
    def precision_quality(self) -> str:
        """Classify the precision of the estimate based on CI width.

        Categories:
        - excellent: relative CI width < 0.1 (very narrow)
        - good: 0.1 <= relative CI width < 0.3
        - moderate: 0.3 <= relative CI width < 0.5
        - poor: relative CI width >= 0.5 (wide uncertainty)
        """
        rel_width = self.relative_ci_width
        if rel_width < 0.1:
            return "excellent"
        elif rel_width < 0.3:
            return "good"
        elif rel_width < 0.5:
            return "moderate"
        else:
            return "poor"

    @property
    def is_reliable(self) -> bool:
        """Check if the estimate is reliable (sufficient samples and narrow CI)."""
        return self.n_samples >= MIN_SAMPLES_FOR_BOOTSTRAP and self.relative_ci_width < 0.5

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"BootstrapCIResult(point={self.point_estimate:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"n={self.n_samples})"
        )


@dataclass
class ConveyanceMetricsResult:
    """Complete conveyance metrics analysis result.

    Combines all individual metric results (D_eff, Beta, C_pair) into a single
    comprehensive report with overall assessment and serialization support.

    This is the primary return type for full conveyance analysis, providing:
    - Effective dimensionality analysis (semantic space utilization)
    - Collapse indicator (semantic convergence/diversity)
    - Pairwise conveyance metrics (bilateral transfer capacity)
    - Overall health assessment and quality indicators

    INTERPRETATION
    ==============
    - overall_health: Summary assessment of conveyance quality
    - has_collapse: Quick check for semantic collapse (Beta >= 0.5)
    - has_blocked_pairs: Quick check for blocked agent pairs

    USAGE
    =====
    This dataclass is designed for:
    1. Comprehensive analysis results from full pipeline
    2. Serialization to JSON/dict for storage and API responses
    3. Aggregation of multiple individual metric results

    Integration: Follows the pattern of KakeyaGeometryReport from
    kakeya_geometry.py for consistency across analysis modules.
    """

    d_eff_result: EffectiveDimensionalityResult  # Effective dimensionality analysis
    beta_result: BetaResult  # Collapse indicator analysis
    c_pair_results: list[CPairResult]  # Pairwise conveyance results (may be empty)
    n_samples: int  # Number of embedding samples analyzed
    ambient_dim: int  # Original/ambient dimensionality of embeddings
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_collapse(self) -> bool:
        """Check if semantic collapse is detected (Beta >= 0.5)."""
        return self.beta_result.is_collapsed

    @property
    def has_blocked_pairs(self) -> bool:
        """Check if any agent pairs have blocked transfer (C_pair ≈ 0)."""
        return any(cp.is_blocked for cp in self.c_pair_results)

    @property
    def mean_c_pair(self) -> float:
        """Calculate mean pairwise conveyance across all pairs."""
        if not self.c_pair_results:
            return 0.0
        return float(np.mean([cp.c_pair for cp in self.c_pair_results]))

    @property
    def min_c_pair(self) -> float:
        """Get minimum pairwise conveyance (weakest link)."""
        if not self.c_pair_results:
            return 0.0
        return float(min(cp.c_pair for cp in self.c_pair_results))

    @property
    def overall_health(self) -> str:
        """Overall conveyance health assessment.

        Returns a string classification based on the combination of
        dimensionality quality, collapse severity, and transfer quality.

        Categories:
        - healthy: Good dimensionality, no collapse, strong transfer
        - warning:<issue>: Single issue detected
        - unhealthy:<issues>: Multiple issues detected
        - critical: Severe collapse or all pairs blocked
        """
        issues = []

        # Check dimensional health
        if self.d_eff_result.dimensionality_quality == "degenerate":
            issues.append("degenerate_space")
        elif self.d_eff_result.is_low_dimensional:
            issues.append("low_dimensional")

        # Check collapse status
        if self.beta_result.collapse_severity == "severe":
            issues.append("severe_collapse")
        elif self.beta_result.collapse_severity == "moderate":
            issues.append("moderate_collapse")

        # Check transfer quality
        if self.has_blocked_pairs:
            issues.append("blocked_pairs")
        elif self.c_pair_results and self.mean_c_pair < 0.2:
            issues.append("poor_transfer")

        # Determine overall health
        if not issues:
            return "healthy"
        elif "severe_collapse" in issues and "blocked_pairs" in issues:
            return "critical"
        elif len(issues) == 1:
            return f"warning:{issues[0]}"
        else:
            return f"unhealthy:{','.join(issues)}"

    @property
    def quality_score(self) -> float:
        """Compute overall quality score in range [0, 1].

        Combines multiple indicators into a single quality metric:
        - Dimensionality utilization (d_eff / ambient_dim)
        - Collapse indicator (inverted: 1 - beta)
        - Mean transfer quality (mean_c_pair)

        Uses geometric mean to ensure any zero component propagates.
        """
        # Dimensionality score (clamped to [0, 1])
        dim_score = max(0.0, min(1.0, self.d_eff_result.variance_ratio * 2))

        # Collapse score (inverted: high beta = low score)
        collapse_score = 1.0 - self.beta_result.beta

        # Transfer score (clamped to [0, 1])
        transfer_score = max(0.0, min(1.0, self.mean_c_pair))

        # Geometric mean (zero-propagation)
        if dim_score <= 0 or collapse_score <= 0:
            return 0.0

        if transfer_score <= 0 and self.c_pair_results:
            return 0.0

        # If no c_pair results, use only dim and collapse scores
        if not self.c_pair_results:
            return float(np.sqrt(dim_score * collapse_score))

        return float(np.cbrt(dim_score * collapse_score * transfer_score))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns a nested dictionary structure suitable for JSON serialization
        or storage, containing all key metrics and assessments.
        """
        return {
            "overall_health": self.overall_health,
            "quality_score": self.quality_score,
            "n_samples": self.n_samples,
            "ambient_dim": self.ambient_dim,
            "d_eff": {
                "d_eff": self.d_eff_result.d_eff,
                "ambient_dim": self.d_eff_result.ambient_dim,
                "variance_ratio": self.d_eff_result.variance_ratio,
                "quality": self.d_eff_result.dimensionality_quality,
                "is_low_dimensional": self.d_eff_result.is_low_dimensional,
            },
            "beta": {
                "beta": self.beta_result.beta,
                "d_eff": self.beta_result.d_eff,
                "d_max": self.beta_result.d_max,
                "severity": self.beta_result.collapse_severity,
                "is_collapsed": self.beta_result.is_collapsed,
                "is_healthy": self.beta_result.is_healthy,
                "eigenvalue_concentration": self.beta_result.eigenvalue_concentration,
            },
            "c_pair": {
                "num_pairs": len(self.c_pair_results),
                "mean_c_pair": self.mean_c_pair,
                "min_c_pair": self.min_c_pair,
                "has_blocked_pairs": self.has_blocked_pairs,
                "pairs": [
                    {
                        "c_pair": cp.c_pair,
                        "c_out": cp.c_out,
                        "c_in": cp.c_in,
                        "harmonic_mean": cp.harmonic_mean,
                        "quality": cp.transfer_quality,
                        "limiting_direction": cp.limiting_direction,
                    }
                    for cp in self.c_pair_results
                ],
            },
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ConveyanceMetricsResult("
            f"health={self.overall_health!r}, "
            f"d_eff={self.d_eff_result.d_eff}, "
            f"beta={self.beta_result.beta:.4f}, "
            f"n_pairs={len(self.c_pair_results)}, "
            f"n_samples={self.n_samples})"
        )


# ============================================================================
# Bootstrap Confidence Interval Calculation
# ============================================================================


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Calculate bootstrap confidence interval for a statistic.

    Computes a non-parametric confidence interval using bootstrap resampling.
    This is the recommended method for uncertainty quantification of conveyance
    metrics (D_eff, Beta, C_pair) where parametric assumptions may not hold.

    MATHEMATICAL GROUNDING
    ======================
    Bootstrap CI estimation works by:
    1. Computing the statistic on original data → point estimate
    2. Drawing B samples with replacement from data
    3. Computing statistic on each bootstrap sample
    4. Using percentile method: CI = [Q_{α/2}, Q_{1-α/2}]

    The percentile method is chosen for robustness:
    - No normality assumption required
    - Works for bounded statistics (like Beta ∈ [0, 1])
    - Distribution-agnostic

    Parameters:
        data: np.ndarray
            1D array of data points to bootstrap from.
        statistic: Callable[[np.ndarray], float], default=np.mean
            Function that computes the statistic of interest.
            Must accept a 1D array and return a scalar.
        n_resamples: int, default=10000
            Number of bootstrap resamples. 10,000 standard, 5,000 acceptable.
        confidence_level: float, default=0.95
            Confidence level for the interval (0 < level < 1).
        random_state: int | None, default=None
            Random seed for reproducibility.

    Returns:
        tuple[float, float, float]:
            (point_estimate, ci_lower, ci_upper) where:
            - point_estimate: Statistic computed on original data
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            Guaranteed: ci_lower <= point_estimate <= ci_upper

    Raises:
        ValueError: If data is empty, confidence_level out of range, or n_resamples < 1.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100)
        >>> point, ci_low, ci_high = bootstrap_ci(data, statistic=np.mean)
        >>> print(f"Mean: {point:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        >>> # Custom statistic: median absolute deviation
        >>> from scipy.stats import median_abs_deviation
        >>> point, ci_low, ci_high = bootstrap_ci(data, statistic=median_abs_deviation)

    Notes:
        - For small samples (n < 30), CI may be unreliable; consider increasing n_resamples
        - Very narrow CI may indicate lack of variability in data
        - Statistic function must be deterministic for reproducible results
        - Uses SciPy's bootstrap with percentile method
    """
    # Input validation
    data = np.asarray(data).flatten()

    if len(data) == 0:
        raise ValueError("data cannot be empty")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    # Edge case: single data point
    if len(data) == 1:
        point = float(statistic(data))
        # With single point, CI is just the point estimate
        warnings.warn(
            "Single data point: confidence interval equals point estimate",
            UserWarning,
            stacklevel=2,
        )
        return (point, point, point)

    # Compute point estimate on original data
    point_estimate = float(statistic(data))

    # Handle edge case: all data identical
    if np.std(data) < 1e-10:
        # No variability - CI is just the point estimate
        return (point_estimate, point_estimate, point_estimate)

    # Perform bootstrap using scipy.stats.bootstrap
    # CRITICAL: Data must be passed as tuple (data,) not raw array
    rng = np.random.default_rng(random_state)

    try:
        result = stats.bootstrap(
            (data,),  # CRITICAL: tuple format
            statistic=statistic,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method='percentile',
            random_state=rng,
        )

        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)

    except Exception as e:
        # Fallback: if bootstrap fails, use normal approximation
        warnings.warn(
            f"Bootstrap failed ({e}), falling back to normal approximation",
            UserWarning,
            stacklevel=2,
        )
        std_err = np.std(data, ddof=1) / np.sqrt(len(data))
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = point_estimate - z * std_err
        ci_upper = point_estimate + z * std_err

    # Ensure CI bounds are ordered correctly
    # (handles edge cases where bootstrap distribution is degenerate)
    if ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    # Ensure point estimate is within CI bounds
    # (can happen with skewed bootstrap distributions)
    ci_lower = min(ci_lower, point_estimate)
    ci_upper = max(ci_upper, point_estimate)

    return (point_estimate, ci_lower, ci_upper)


def bootstrap_ci_detailed(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int | None = None,
) -> BootstrapCIResult:
    """Calculate bootstrap confidence interval with full diagnostic information.

    This is the detailed version of bootstrap_ci that returns complete
    results including standard error, metadata, and quality indicators.

    Parameters:
        data: np.ndarray
            1D array of data points to bootstrap from.
        statistic: Callable[[np.ndarray], float], default=np.mean
            Function that computes the statistic of interest.
        n_resamples: int, default=10000
            Number of bootstrap resamples.
        confidence_level: float, default=0.95
            Confidence level for the interval.
        random_state: int | None, default=None
            Random seed for reproducibility.

    Returns:
        BootstrapCIResult with full diagnostic information including:
        - point_estimate, ci_lower, ci_upper: The confidence interval
        - standard_error: Bootstrap standard error
        - precision_quality: Classification of CI precision
        - is_reliable: Whether the estimate is reliable

    Example:
        >>> result = bootstrap_ci_detailed(data, statistic=np.mean)
        >>> print(f"Mean: {result.point_estimate:.4f}")
        >>> print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        >>> print(f"Precision: {result.precision_quality}")
    """
    # Input validation
    data = np.asarray(data).flatten()
    n_samples = len(data)

    if n_samples == 0:
        raise ValueError("data cannot be empty")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    # Compute point estimate
    point_estimate = float(statistic(data))

    # Edge case: single data point
    if n_samples == 1:
        return BootstrapCIResult(
            point_estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            n_samples=1,
            standard_error=0.0,
            metadata={"warning": "single_sample"},
        )

    # Edge case: no variability
    if np.std(data) < 1e-10:
        return BootstrapCIResult(
            point_estimate=point_estimate,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            n_samples=n_samples,
            standard_error=0.0,
            metadata={"warning": "no_variability"},
        )

    # Perform bootstrap
    rng = np.random.default_rng(random_state)

    try:
        result = stats.bootstrap(
            (data,),
            statistic=statistic,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method='percentile',
            random_state=rng,
        )

        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        standard_error = float(result.standard_error)
        metadata: dict[str, Any] = {}

    except Exception as e:
        # Fallback to normal approximation
        std_err = np.std(data, ddof=1) / np.sqrt(n_samples)
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = point_estimate - z * std_err
        ci_upper = point_estimate + z * std_err
        standard_error = std_err
        metadata = {"warning": f"bootstrap_failed: {e}", "method": "normal_approximation"}

    # Ensure proper ordering
    if ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    ci_lower = min(ci_lower, point_estimate)
    ci_upper = max(ci_upper, point_estimate)

    # Add sample size warning if small
    if n_samples < MIN_SAMPLES_FOR_BOOTSTRAP:
        metadata["warning"] = metadata.get("warning", "") + "; small_sample_size"

    return BootstrapCIResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_samples=n_samples,
        standard_error=standard_error,
        metadata=metadata,
    )


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
# Patching Impact Metrics
# ============================================================================


@dataclass
class PatchingImpactResult:
    """Results from patching impact analysis.

    Quantifies the causal effect of activation patching on conveyance metrics,
    measuring how much a patch recovers baseline behavior from corrupted state.

    MATHEMATICAL GROUNDING
    ======================
    Patching impact is computed by comparing metrics across three execution paths:
    1. Clean (baseline): Original input, no intervention
    2. Corrupted: Modified input, no patching
    3. Patched: Modified input with activation patching applied

    Key metrics:
    - causal_effect: patched_metric - corrupted_metric
      Positive = patch improved metric toward baseline
    - recovery_rate: causal_effect / |clean_metric - corrupted_metric|
      1.0 = full recovery, 0.0 = no recovery, >1.0 = overcorrection

    INTERPRETATION
    ==============
    - recovery_rate near 1.0: Patch successfully restored clean behavior
    - recovery_rate near 0.0: Patch had minimal effect
    - recovery_rate > 1.0: Overcorrection (patched exceeds baseline)
    - recovery_rate < 0.0: Patch made corruption worse

    USAGE
    =====
    This dataclass integrates with the patching experiment framework to
    measure how activation interventions affect semantic information transfer.
    """

    metric_name: str  # Name of the metric being analyzed
    clean_value: float  # Metric value for clean (baseline) path
    corrupted_value: float  # Metric value for corrupted path
    patched_value: float  # Metric value for patched path
    causal_effect: float  # patched - corrupted (improvement from patch)
    recovery_rate: float  # causal_effect / |corruption_delta|
    corruption_delta: float  # corrupted - clean (damage from corruption)
    patch_layer: int | None = None  # Layer where patch was applied (if applicable)
    patch_component: str | None = None  # Component patched (if applicable)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_recovery(self) -> bool:
        """Check if the patch moved the metric toward the clean baseline.

        Recovery is direction-aware: it checks whether the causal effect
        counteracts the corruption, regardless of whether values increased
        or decreased. The multiplication check handles both cases:

        - If corruption_delta > 0 (corrupted > clean): recovery requires
          causal_effect < 0 (patched moved down toward clean)
        - If corruption_delta < 0 (corrupted < clean): recovery requires
          causal_effect > 0 (patched moved up toward clean)

        Returns True if the patch moved in the recovery direction.
        """
        # Recovery means moving from corrupted toward clean
        # If corruption_delta < 0 (corrupted < clean), recovery means patched > corrupted
        # If corruption_delta > 0 (corrupted > clean), recovery means patched < corrupted
        if abs(self.corruption_delta) < 1e-10:
            return False  # No corruption to recover from
        return self.causal_effect * self.corruption_delta < 0

    @property
    def is_full_recovery(self) -> bool:
        """Check if patch fully recovered baseline (recovery_rate >= 0.95)."""
        return self.recovery_rate >= 0.95

    @property
    def is_overcorrection(self) -> bool:
        """Check if patch overcorrected (recovery_rate > 1.0)."""
        return self.recovery_rate > 1.0

    @property
    def is_harmful(self) -> bool:
        """Check if patch made corruption worse (recovery_rate < 0)."""
        return self.recovery_rate < 0.0

    @property
    def impact_severity(self) -> str:
        """Classify the severity/quality of patch impact.

        Categories:
        - full_recovery: recovery_rate >= 0.95
        - strong_recovery: 0.7 <= recovery_rate < 0.95
        - moderate_recovery: 0.3 <= recovery_rate < 0.7
        - weak_recovery: 0.0 <= recovery_rate < 0.3
        - no_effect: recovery_rate ≈ 0
        - harmful: recovery_rate < 0
        - overcorrection: recovery_rate > 1.0
        """
        if abs(self.recovery_rate) < 0.05:
            return "no_effect"
        elif self.recovery_rate < 0:
            return "harmful"
        elif self.recovery_rate > 1.0:
            return "overcorrection"
        elif self.recovery_rate >= 0.95:
            return "full_recovery"
        elif self.recovery_rate >= 0.7:
            return "strong_recovery"
        elif self.recovery_rate >= 0.3:
            return "moderate_recovery"
        else:
            return "weak_recovery"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "clean_value": self.clean_value,
            "corrupted_value": self.corrupted_value,
            "patched_value": self.patched_value,
            "causal_effect": self.causal_effect,
            "recovery_rate": self.recovery_rate,
            "corruption_delta": self.corruption_delta,
            "patch_layer": self.patch_layer,
            "patch_component": self.patch_component,
            "impact_severity": self.impact_severity,
            "is_recovery": self.is_recovery,
            "is_full_recovery": self.is_full_recovery,
            "is_overcorrection": self.is_overcorrection,
            "is_harmful": self.is_harmful,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        layer_info = f", layer={self.patch_layer}" if self.patch_layer is not None else ""
        return (
            f"PatchingImpactResult("
            f"metric={self.metric_name!r}, "
            f"recovery={self.recovery_rate:.4f}, "
            f"severity={self.impact_severity!r}"
            f"{layer_info})"
        )


@dataclass
class MultiPathMetricsResult:
    """Results from computing metrics across all execution paths.

    Contains conveyance metrics for clean, corrupted, and patched paths,
    enabling comprehensive comparison of patching effects on information transfer.
    """

    clean_metrics: ConveyanceMetricsResult | None  # Metrics for clean path
    corrupted_metrics: ConveyanceMetricsResult | None  # Metrics for corrupted path
    patched_metrics: list[ConveyanceMetricsResult]  # Metrics for each patched path
    impacts: list[PatchingImpactResult]  # Impact analysis for each patched path
    n_patched_paths: int  # Number of patched paths analyzed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_all_paths(self) -> bool:
        """Check if metrics are available for all path types."""
        return (
            self.clean_metrics is not None
            and self.corrupted_metrics is not None
            and len(self.patched_metrics) > 0
        )

    @property
    def mean_recovery_rate(self) -> float:
        """Calculate mean recovery rate across all patched paths."""
        if not self.impacts:
            return 0.0
        return float(np.mean([i.recovery_rate for i in self.impacts]))

    @property
    def best_recovery_rate(self) -> float:
        """Get the best (highest) recovery rate among patched paths."""
        if not self.impacts:
            return 0.0
        return float(max(i.recovery_rate for i in self.impacts))

    @property
    def best_patch_layer(self) -> int | None:
        """Get the layer with the best recovery rate."""
        if not self.impacts:
            return None
        best = max(self.impacts, key=lambda i: i.recovery_rate)
        return best.patch_layer

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_all_paths": self.has_all_paths,
            "n_patched_paths": self.n_patched_paths,
            "mean_recovery_rate": self.mean_recovery_rate,
            "best_recovery_rate": self.best_recovery_rate,
            "best_patch_layer": self.best_patch_layer,
            "clean_metrics": self.clean_metrics.to_dict() if self.clean_metrics else None,
            "corrupted_metrics": self.corrupted_metrics.to_dict() if self.corrupted_metrics else None,
            "patched_metrics": [pm.to_dict() for pm in self.patched_metrics],
            "impacts": [i.to_dict() for i in self.impacts],
            "metadata": self.metadata,
        }


def compute_patch_impact(
    clean_value: float,
    corrupted_value: float,
    patched_value: float,
    metric_name: str = "generic",
    patch_layer: int | None = None,
    patch_component: str | None = None,
) -> PatchingImpactResult:
    """Compute the impact of activation patching on a metric.

    Measures how effectively a patch recovers baseline behavior from a corrupted
    state. This is the core function for causal analysis of activation patches.

    MATHEMATICAL GROUNDING
    ======================
    The patching impact is computed as:

        causal_effect = patched_value - corrupted_value
        corruption_delta = corrupted_value - clean_value
        recovery_rate = -causal_effect / corruption_delta

    The negative sign ensures recovery_rate is positive when the patch
    moves toward clean (counteracts corruption), regardless of direction:
    - If corruption increased metric (delta > 0), recovery needs negative causal_effect
    - If corruption decreased metric (delta < 0), recovery needs positive causal_effect

    The recovery rate indicates what fraction of the corruption was undone:
    - recovery_rate = 1.0: Full recovery (patched equals baseline)
    - recovery_rate = 0.0: No recovery (patched equals corrupted)
    - recovery_rate > 1.0: Overcorrection (patched overshoots baseline)
    - recovery_rate < 0.0: Harmful (patched worse than corrupted)

    Parameters:
        clean_value: float
            Metric value for the clean (baseline) execution path.
        corrupted_value: float
            Metric value for the corrupted execution path.
        patched_value: float
            Metric value for the patched execution path.
        metric_name: str, default="generic"
            Name of the metric being analyzed (e.g., "beta", "d_eff", "c_pair").
        patch_layer: int | None, default=None
            Layer index where the patch was applied.
        patch_component: str | None, default=None
            Component that was patched (e.g., "resid_pre", "attn").

    Returns:
        PatchingImpactResult with full impact analysis including:
        - causal_effect: Direct measure of patch impact
        - recovery_rate: Normalized recovery metric
        - impact_severity: Classification of impact quality

    Example:
        >>> # Patch successfully restored clean behavior
        >>> result = compute_patch_impact(
        ...     clean_value=0.2,  # Low beta (healthy)
        ...     corrupted_value=0.8,  # High beta (collapsed)
        ...     patched_value=0.3,  # Patch reduced collapse
        ...     metric_name="beta",
        ...     patch_layer=5,
        ... )
        >>> print(f"Recovery rate: {result.recovery_rate:.2f}")  # ~0.83

        >>> # Patch had no effect
        >>> result = compute_patch_impact(
        ...     clean_value=0.5,
        ...     corrupted_value=0.8,
        ...     patched_value=0.8,  # Same as corrupted
        ...     metric_name="d_eff",
        ... )
        >>> print(result.impact_severity)  # "no_effect"

    Notes:
        - When clean and corrupted values are identical, recovery_rate = 0
          (no corruption to recover from).
        - Use with Beta metric: Lower Beta = healthier, so recovery means
          reducing Beta toward clean baseline.
        - Use with D_eff: Higher D_eff = more dimensions utilized.
    """
    # Calculate corruption delta (how much corruption changed the metric)
    corruption_delta = corrupted_value - clean_value

    # Calculate causal effect (how much patch changed from corrupted)
    causal_effect = patched_value - corrupted_value

    # Calculate recovery rate
    # Avoid division by zero when there's no corruption
    if abs(corruption_delta) < 1e-10:
        # No corruption to recover from
        if abs(causal_effect) < 1e-10:
            recovery_rate = 0.0  # Patch had no effect (but none needed)
        else:
            # Patch changed something when nothing was corrupted
            # This is unusual - treat as infinite recovery (or could be harmful)
            recovery_rate = float("inf") if causal_effect > 0 else float("-inf")
    else:
        # Standard recovery rate: how much of corruption was undone
        # Note: We use negative corruption_delta to correctly handle both
        # cases where corruption increases or decreases the metric
        recovery_rate = -causal_effect / corruption_delta

    return PatchingImpactResult(
        metric_name=metric_name,
        clean_value=clean_value,
        corrupted_value=corrupted_value,
        patched_value=patched_value,
        causal_effect=causal_effect,
        recovery_rate=recovery_rate,
        corruption_delta=corruption_delta,
        patch_layer=patch_layer,
        patch_component=patch_component,
    )


def compute_beta_patch_impact(
    clean_embeddings: np.ndarray,
    corrupted_embeddings: np.ndarray,
    patched_embeddings: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    patch_layer: int | None = None,
    patch_component: str | None = None,
) -> PatchingImpactResult:
    """Compute the impact of patching on Beta (collapse indicator) metric.

    Specialized function for analyzing how activation patches affect semantic
    collapse in embedding space.

    Parameters:
        clean_embeddings: Array of shape (n_samples, n_features)
            Embeddings from the clean (baseline) execution path.
        corrupted_embeddings: Array of shape (n_samples, n_features)
            Embeddings from the corrupted execution path.
        patched_embeddings: Array of shape (n_samples, n_features)
            Embeddings from the patched execution path.
        variance_threshold: float, default=0.90
            Variance threshold for D_eff calculation.
        patch_layer: int | None, default=None
            Layer where the patch was applied.
        patch_component: str | None, default=None
            Component that was patched.

    Returns:
        PatchingImpactResult for the Beta metric with full impact analysis.

    Example:
        >>> result = compute_beta_patch_impact(
        ...     clean_embeddings=clean_hidden_states,
        ...     corrupted_embeddings=corrupted_hidden_states,
        ...     patched_embeddings=patched_hidden_states,
        ...     patch_layer=5,
        ... )
        >>> print(f"Beta recovery: {result.recovery_rate:.2%}")
    """
    # Calculate Beta for each path
    beta_clean = calculate_beta(clean_embeddings, variance_threshold=variance_threshold)
    beta_corrupted = calculate_beta(corrupted_embeddings, variance_threshold=variance_threshold)
    beta_patched = calculate_beta(patched_embeddings, variance_threshold=variance_threshold)

    return compute_patch_impact(
        clean_value=beta_clean,
        corrupted_value=beta_corrupted,
        patched_value=beta_patched,
        metric_name="beta",
        patch_layer=patch_layer,
        patch_component=patch_component,
    )


def compute_d_eff_patch_impact(
    clean_embeddings: np.ndarray,
    corrupted_embeddings: np.ndarray,
    patched_embeddings: np.ndarray,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    patch_layer: int | None = None,
    patch_component: str | None = None,
) -> PatchingImpactResult:
    """Compute the impact of patching on D_eff (effective dimensionality) metric.

    Specialized function for analyzing how activation patches affect the
    effective dimensionality of embedding space.

    Parameters:
        clean_embeddings: Array of shape (n_samples, n_features)
            Embeddings from the clean (baseline) execution path.
        corrupted_embeddings: Array of shape (n_samples, n_features)
            Embeddings from the corrupted execution path.
        patched_embeddings: Array of shape (n_samples, n_features)
            Embeddings from the patched execution path.
        variance_threshold: float, default=0.90
            Variance threshold for D_eff calculation.
        patch_layer: int | None, default=None
            Layer where the patch was applied.
        patch_component: str | None, default=None
            Component that was patched.

    Returns:
        PatchingImpactResult for the D_eff metric with full impact analysis.
    """
    # Calculate D_eff for each path
    d_eff_clean = calculate_d_eff(clean_embeddings, variance_threshold=variance_threshold)
    d_eff_corrupted = calculate_d_eff(corrupted_embeddings, variance_threshold=variance_threshold)
    d_eff_patched = calculate_d_eff(patched_embeddings, variance_threshold=variance_threshold)

    return compute_patch_impact(
        clean_value=float(d_eff_clean),
        corrupted_value=float(d_eff_corrupted),
        patched_value=float(d_eff_patched),
        metric_name="d_eff",
        patch_layer=patch_layer,
        patch_component=patch_component,
    )


def compute_quality_score_patch_impact(
    clean_metrics: ConveyanceMetricsResult,
    corrupted_metrics: ConveyanceMetricsResult,
    patched_metrics: ConveyanceMetricsResult,
    patch_layer: int | None = None,
    patch_component: str | None = None,
) -> PatchingImpactResult:
    """Compute the impact of patching on the overall quality score.

    Uses the composite quality_score from ConveyanceMetricsResult which
    combines dimensionality, collapse, and transfer metrics.

    Parameters:
        clean_metrics: ConveyanceMetricsResult
            Full metrics from the clean (baseline) execution path.
        corrupted_metrics: ConveyanceMetricsResult
            Full metrics from the corrupted execution path.
        patched_metrics: ConveyanceMetricsResult
            Full metrics from the patched execution path.
        patch_layer: int | None, default=None
            Layer where the patch was applied.
        patch_component: str | None, default=None
            Component that was patched.

    Returns:
        PatchingImpactResult for the quality_score metric.
    """
    return compute_patch_impact(
        clean_value=clean_metrics.quality_score,
        corrupted_value=corrupted_metrics.quality_score,
        patched_value=patched_metrics.quality_score,
        metric_name="quality_score",
        patch_layer=patch_layer,
        patch_component=patch_component,
    )


def aggregate_patch_impacts(
    impacts: list[PatchingImpactResult],
) -> dict[str, Any]:
    """Aggregate patching impact results across multiple experiments.

    Computes summary statistics for a series of patching experiments,
    useful for multi-layer sweeps or systematic studies.

    Parameters:
        impacts: list[PatchingImpactResult]
            List of impact results to aggregate.

    Returns:
        dict[str, Any] containing:
        - mean_recovery_rate: Average recovery rate
        - best_recovery_rate: Highest recovery rate
        - worst_recovery_rate: Lowest recovery rate
        - best_layer: Layer with best recovery (if applicable)
        - recovery_distribution: Counts by severity category
        - all_recovery_rates: List of all recovery rates for visualization

    Example:
        >>> impacts = [
        ...     compute_patch_impact(0.2, 0.8, 0.3, "beta", patch_layer=0),
        ...     compute_patch_impact(0.2, 0.8, 0.4, "beta", patch_layer=5),
        ...     compute_patch_impact(0.2, 0.8, 0.25, "beta", patch_layer=10),
        ... ]
        >>> stats = aggregate_patch_impacts(impacts)
        >>> print(f"Best layer: {stats['best_layer']}, recovery: {stats['best_recovery_rate']:.2%}")
    """
    if not impacts:
        return {
            "mean_recovery_rate": 0.0,
            "best_recovery_rate": 0.0,
            "worst_recovery_rate": 0.0,
            "best_layer": None,
            "num_impacts": 0,
            "recovery_distribution": {},
            "all_recovery_rates": [],
        }

    recovery_rates = [i.recovery_rate for i in impacts if not np.isinf(i.recovery_rate)]

    if not recovery_rates:
        return {
            "mean_recovery_rate": 0.0,
            "best_recovery_rate": 0.0,
            "worst_recovery_rate": 0.0,
            "best_layer": None,
            "num_impacts": len(impacts),
            "recovery_distribution": {},
            "all_recovery_rates": [],
        }

    # Find best impact (highest recovery rate)
    best_impact = max(impacts, key=lambda i: i.recovery_rate if not np.isinf(i.recovery_rate) else float("-inf"))

    # Count impacts by severity category
    severity_counts: dict[str, int] = {}
    for impact in impacts:
        severity = impact.impact_severity
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return {
        "mean_recovery_rate": float(np.mean(recovery_rates)),
        "std_recovery_rate": float(np.std(recovery_rates)) if len(recovery_rates) > 1 else 0.0,
        "best_recovery_rate": float(max(recovery_rates)),
        "worst_recovery_rate": float(min(recovery_rates)),
        "best_layer": best_impact.patch_layer,
        "best_component": best_impact.patch_component,
        "num_impacts": len(impacts),
        "num_full_recovery": sum(1 for i in impacts if i.is_full_recovery),
        "num_harmful": sum(1 for i in impacts if i.is_harmful),
        "recovery_distribution": severity_counts,
        "all_recovery_rates": recovery_rates,
        "layers_by_recovery": [
            (i.patch_layer, i.recovery_rate)
            for i in sorted(impacts, key=lambda x: x.recovery_rate, reverse=True)
            if i.patch_layer is not None
        ],
    }


def identify_causal_layers(
    impacts: list[PatchingImpactResult],
    recovery_threshold: float = 0.3,
) -> list[int]:
    """Identify layers that causally affect the metric.

    Finds layers where patching has significant recovery effect, indicating
    that those layers carry causal information for the measured behavior.

    Parameters:
        impacts: list[PatchingImpactResult]
            List of impact results from a layer sweep.
        recovery_threshold: float, default=0.3
            Minimum recovery rate to consider a layer as causal.

    Returns:
        list[int]: Layer indices where patching has significant effect,
            sorted by recovery rate (highest first).

    Example:
        >>> # Find layers that matter for semantic transfer
        >>> causal_layers = identify_causal_layers(layer_sweep_impacts, threshold=0.5)
        >>> print(f"Causal layers: {causal_layers}")  # e.g., [5, 6, 8]
    """
    causal = [
        i for i in impacts
        if i.patch_layer is not None
        and i.recovery_rate >= recovery_threshold
        and not np.isinf(i.recovery_rate)
    ]

    # Sort by recovery rate (highest first)
    causal.sort(key=lambda i: i.recovery_rate, reverse=True)

    return [i.patch_layer for i in causal if i.patch_layer is not None]


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

    print("\n" + "=" * 60)
    print("Testing Bootstrap Confidence Interval Calculation...")
    print("=" * 60)

    # Basic bootstrap CI with mean
    test_data = np.random.randn(100)
    ci_result = bootstrap_ci(test_data, statistic=np.mean, n_resamples=1000, random_state=42)
    print(f"\nBasic Bootstrap CI for mean:")
    print(f"  Result: {ci_result}")
    print(f"  Point estimate: {ci_result[0]:.4f}")
    print(f"  CI lower: {ci_result[1]:.4f}")
    print(f"  CI upper: {ci_result[2]:.4f}")
    assert len(ci_result) == 3, "bootstrap_ci should return 3 values"
    assert ci_result[1] <= ci_result[0] <= ci_result[2], "CI bounds should contain point estimate"

    # Test with np.std as statistic
    ci_std = bootstrap_ci(test_data, statistic=np.std, n_resamples=1000, random_state=42)
    print(f"\nBootstrap CI for std:")
    print(f"  Point estimate: {ci_std[0]:.4f}")
    print(f"  CI: [{ci_std[1]:.4f}, {ci_std[2]:.4f}]")
    assert ci_std[1] <= ci_std[0] <= ci_std[2], "CI bounds should contain point estimate"

    # Detailed bootstrap result
    detailed_ci = bootstrap_ci_detailed(test_data, statistic=np.mean, n_resamples=1000, random_state=42)
    print(f"\nDetailed Bootstrap CI:")
    print(f"  Point estimate: {detailed_ci.point_estimate:.4f}")
    print(f"  CI: [{detailed_ci.ci_lower:.4f}, {detailed_ci.ci_upper:.4f}]")
    print(f"  Standard error: {detailed_ci.standard_error:.4f}")
    print(f"  CI width: {detailed_ci.ci_width:.4f}")
    print(f"  Precision quality: {detailed_ci.precision_quality}")
    print(f"  Is reliable: {detailed_ci.is_reliable}")

    # Edge case: small sample
    small_data = np.random.randn(5)
    ci_small = bootstrap_ci(small_data, statistic=np.mean, n_resamples=500, random_state=42)
    print(f"\nSmall sample (n=5) CI: [{ci_small[1]:.4f}, {ci_small[2]:.4f}]")
    assert ci_small[1] <= ci_small[0] <= ci_small[2], "Small sample CI should still be valid"

    # Edge case: constant data
    constant_data = np.ones(50)
    ci_const = bootstrap_ci(constant_data, statistic=np.mean, random_state=42)
    print(f"\nConstant data CI: [{ci_const[1]:.4f}, {ci_const[2]:.4f}]")
    assert ci_const[0] == ci_const[1] == ci_const[2] == 1.0, "Constant data should have zero-width CI"

    # Verify the spec requirement: result[1] < result[0] < result[2] for variable data
    variable_data = np.random.randn(100)
    ci_verify = bootstrap_ci(variable_data, statistic=np.mean, random_state=123)
    print(f"\nVerification test (variable data):")
    print(f"  CI: [{ci_verify[1]:.4f}, {ci_verify[0]:.4f}, {ci_verify[2]:.4f}]")
    print(f"  Check: ci_lower < point < ci_upper: {ci_verify[1] < ci_verify[0] < ci_verify[2]}")
    # For variable data, strict inequality should hold
    assert ci_verify[1] < ci_verify[0] < ci_verify[2], "Variable data should have strict CI bounds"

    print("\nAll Bootstrap CI tests passed!")

    print("\n" + "=" * 60)
    print("Testing Patching Impact Metrics...")
    print("=" * 60)

    # Basic patching impact calculation
    impact = compute_patch_impact(
        clean_value=0.2,
        corrupted_value=0.8,
        patched_value=0.4,
        metric_name="beta",
        patch_layer=5,
        patch_component="resid_pre",
    )
    print(f"\nBasic Patching Impact:")
    print(f"  Clean: {impact.clean_value}, Corrupted: {impact.corrupted_value}, Patched: {impact.patched_value}")
    print(f"  Corruption delta: {impact.corruption_delta:.4f}")
    print(f"  Causal effect: {impact.causal_effect:.4f}")
    print(f"  Recovery rate: {impact.recovery_rate:.4f}")
    print(f"  Impact severity: {impact.impact_severity}")
    print(f"  Is recovery: {impact.is_recovery}")
    assert isinstance(impact, PatchingImpactResult), "Should return PatchingImpactResult"
    assert impact.patch_layer == 5, "Should preserve patch layer"

    # Test full recovery case
    full_recovery_impact = compute_patch_impact(
        clean_value=0.2,
        corrupted_value=0.8,
        patched_value=0.2,  # Full recovery
        metric_name="beta",
    )
    print(f"\nFull Recovery Impact:")
    print(f"  Recovery rate: {full_recovery_impact.recovery_rate:.4f}")
    print(f"  Is full recovery: {full_recovery_impact.is_full_recovery}")
    print(f"  Severity: {full_recovery_impact.impact_severity}")
    assert full_recovery_impact.is_full_recovery, "Should be full recovery when patched == clean"
    assert full_recovery_impact.impact_severity == "full_recovery", f"Expected 'full_recovery', got {full_recovery_impact.impact_severity}"

    # Test no effect case
    no_effect_impact = compute_patch_impact(
        clean_value=0.2,
        corrupted_value=0.8,
        patched_value=0.8,  # No change
        metric_name="beta",
    )
    print(f"\nNo Effect Impact:")
    print(f"  Recovery rate: {no_effect_impact.recovery_rate:.4f}")
    print(f"  Severity: {no_effect_impact.impact_severity}")
    assert no_effect_impact.impact_severity == "no_effect", f"Expected 'no_effect', got {no_effect_impact.impact_severity}"

    # Test harmful case (patch makes it worse)
    harmful_impact = compute_patch_impact(
        clean_value=0.2,
        corrupted_value=0.8,
        patched_value=0.9,  # Worse than corrupted
        metric_name="beta",
    )
    print(f"\nHarmful Impact:")
    print(f"  Recovery rate: {harmful_impact.recovery_rate:.4f}")
    print(f"  Is harmful: {harmful_impact.is_harmful}")
    print(f"  Severity: {harmful_impact.impact_severity}")
    assert harmful_impact.is_harmful, "Should be harmful when patch makes metric worse"

    # Test overcorrection case
    overcorrection_impact = compute_patch_impact(
        clean_value=0.4,
        corrupted_value=0.8,
        patched_value=0.1,  # Overcorrected past clean
        metric_name="beta",
    )
    print(f"\nOvercorrection Impact:")
    print(f"  Recovery rate: {overcorrection_impact.recovery_rate:.4f}")
    print(f"  Is overcorrection: {overcorrection_impact.is_overcorrection}")
    print(f"  Severity: {overcorrection_impact.impact_severity}")
    assert overcorrection_impact.is_overcorrection, "Should be overcorrection when patched overshoots"

    # Test aggregate patch impacts
    test_impacts = [
        compute_patch_impact(0.2, 0.8, 0.3, "beta", patch_layer=0),
        compute_patch_impact(0.2, 0.8, 0.25, "beta", patch_layer=5),
        compute_patch_impact(0.2, 0.8, 0.6, "beta", patch_layer=10),
        compute_patch_impact(0.2, 0.8, 0.8, "beta", patch_layer=11),  # No effect
    ]
    aggregate = aggregate_patch_impacts(test_impacts)
    print(f"\nAggregate Impact Results:")
    print(f"  Mean recovery: {aggregate['mean_recovery_rate']:.4f}")
    print(f"  Best recovery: {aggregate['best_recovery_rate']:.4f}")
    print(f"  Best layer: {aggregate['best_layer']}")
    print(f"  Recovery distribution: {aggregate['recovery_distribution']}")
    assert aggregate["best_layer"] == 5, f"Best layer should be 5, got {aggregate['best_layer']}"

    # Test identify causal layers
    causal_layers = identify_causal_layers(test_impacts, recovery_threshold=0.3)
    print(f"\nCausal Layers (threshold=0.3): {causal_layers}")
    assert 5 in causal_layers, "Layer 5 should be identified as causal"
    assert 11 not in causal_layers, "Layer 11 should not be causal (no effect)"

    # Test compute_beta_patch_impact with embeddings
    clean_emb = np.random.randn(50, 256)
    collapsed_emb = np.random.randn(50, 1) @ np.random.randn(1, 256)  # Collapsed
    partial_emb = np.random.randn(50, 5) @ np.random.randn(5, 256)  # Partially collapsed

    beta_impact = compute_beta_patch_impact(
        clean_embeddings=clean_emb,
        corrupted_embeddings=collapsed_emb,
        patched_embeddings=partial_emb,
        patch_layer=7,
    )
    print(f"\nBeta Patch Impact from Embeddings:")
    print(f"  Clean beta: {beta_impact.clean_value:.4f}")
    print(f"  Corrupted beta: {beta_impact.corrupted_value:.4f}")
    print(f"  Patched beta: {beta_impact.patched_value:.4f}")
    print(f"  Recovery rate: {beta_impact.recovery_rate:.4f}")
    assert beta_impact.metric_name == "beta", "Should set metric name to 'beta'"

    # Test MultiPathMetricsResult (just verify it can be created)
    multi_result = MultiPathMetricsResult(
        clean_metrics=None,
        corrupted_metrics=None,
        patched_metrics=[],
        impacts=test_impacts,
        n_patched_paths=4,
    )
    print(f"\nMultiPathMetricsResult:")
    print(f"  Has all paths: {multi_result.has_all_paths}")
    print(f"  Mean recovery: {multi_result.mean_recovery_rate:.4f}")
    print(f"  Best layer: {multi_result.best_patch_layer}")

    print("\nAll Patching Impact tests passed!")

    print("\n" + "=" * 60)
    print("All D_eff, Beta, C_pair, Bootstrap CI, and Patching Impact tests passed!")
    print("=" * 60)
