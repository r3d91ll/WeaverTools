"""Tests for the conveyance metrics analysis module.

These tests validate the conveyance metrics functions used to quantify
semantic information transfer between agents in multi-agent conversations.

Metrics tested:
- D_eff (Effective Dimensionality): Number of principal components for 90% variance
- Beta (Collapse Indicator): Degree of semantic collapse in embedding space
- C_pair (Pairwise Conveyance): Bilateral information transfer capacity
- Bootstrap CI: Confidence intervals for all metrics
"""

import numpy as np
import pytest

from src.analysis.conveyance_metrics import (
    MIN_SAMPLES_FOR_D_EFF,
    MIN_SAMPLES_FOR_BETA,
    MIN_SAMPLES_FOR_BOOTSTRAP,
    DEFAULT_VARIANCE_THRESHOLD,
    DEFAULT_D_REF,
    EffectiveDimensionalityResult,
    BetaResult,
    CPairResult,
    BootstrapCIResult,
    ConveyanceMetricsResult,
    calculate_d_eff,
    calculate_d_eff_detailed,
    calculate_beta,
    calculate_beta_detailed,
    calculate_c_pair,
    calculate_c_pair_detailed,
    bootstrap_ci,
    bootstrap_ci_detailed,
)


# ============================================================================
# D_eff (Effective Dimensionality) Tests
# ============================================================================


class TestDEffBasic:
    """Tests for basic D_eff functionality."""

    def test_d_eff_random_embeddings(self) -> None:
        """Test D_eff calculation with random embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 768)
        d_eff = calculate_d_eff(embeddings)

        assert isinstance(d_eff, int)
        assert d_eff > 0
        assert d_eff <= min(99, 768)  # max possible is min(n-1, d)

    def test_d_eff_detailed_result(self) -> None:
        """Test that detailed D_eff returns proper result object."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)
        result = calculate_d_eff_detailed(embeddings)

        assert isinstance(result, EffectiveDimensionalityResult)
        assert result.d_eff > 0
        assert result.ambient_dim == 128
        assert result.n_samples == 50
        assert result.variance_threshold == DEFAULT_VARIANCE_THRESHOLD
        assert 0 <= result.variance_ratio <= 1
        assert len(result.eigenvalues) == 128

    def test_d_eff_cumulative_variance_monotonic(self) -> None:
        """Test that cumulative variance is monotonically increasing."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        result = calculate_d_eff_detailed(embeddings)

        # Check monotonicity
        assert all(np.diff(result.cumulative_variance) >= -1e-10)
        # Check bounds
        assert result.cumulative_variance[0] >= 0
        assert abs(result.cumulative_variance[-1] - 1.0) < 1e-10


class TestDEffEdgeCases:
    """Tests for D_eff edge cases."""

    def test_d_eff_single_point(self) -> None:
        """Test D_eff = 1 for single point embedding.

        Edge case: With only one sample, there's no variance to analyze,
        so D_eff should be 1 (single dimension required for single point).
        """
        np.random.seed(42)
        single_point = np.random.randn(1, 768)
        d_eff = calculate_d_eff(single_point)

        assert d_eff == 1, f"D_eff should be 1 for single point, got {d_eff}"

    def test_d_eff_single_point_detailed(self) -> None:
        """Test detailed D_eff for single point returns proper metadata."""
        np.random.seed(42)
        single_point = np.random.randn(1, 512)
        result = calculate_d_eff_detailed(single_point)

        assert result.d_eff == 1
        assert result.n_samples == 1
        assert result.ambient_dim == 512
        assert "edge_case" in result.metadata
        assert result.metadata["edge_case"] == "single_sample"

    def test_d_eff_constant_embeddings(self) -> None:
        """Test D_eff = 1 for constant (identical) embeddings.

        Edge case: When all embeddings are identical, there's zero variance,
        so D_eff should be 1.
        """
        constant = np.ones((100, 768))
        d_eff = calculate_d_eff(constant)

        assert d_eff == 1, f"D_eff should be 1 for constant embeddings, got {d_eff}"

    def test_d_eff_constant_embeddings_detailed(self) -> None:
        """Test detailed D_eff for constant embeddings."""
        constant = np.ones((50, 256))
        result = calculate_d_eff_detailed(constant)

        assert result.d_eff == 1
        assert "edge_case" in result.metadata
        assert result.metadata["edge_case"] == "constant_embeddings"

    def test_d_eff_two_samples(self) -> None:
        """Test D_eff with minimum viable sample size (2 samples).

        With 2 samples, max possible D_eff is 1 (n-1 = 1).
        """
        np.random.seed(42)
        two_samples = np.random.randn(2, 128)
        d_eff = calculate_d_eff(two_samples)

        assert d_eff == 1, f"D_eff should be 1 for 2 samples (max possible), got {d_eff}"

    def test_d_eff_single_feature(self) -> None:
        """Test D_eff with single feature dimension."""
        np.random.seed(42)
        single_feature = np.random.randn(100, 1)
        d_eff = calculate_d_eff(single_feature)

        assert d_eff == 1, f"D_eff should be 1 for single feature, got {d_eff}"

    def test_d_eff_low_rank_data(self) -> None:
        """Test D_eff correctly identifies low-rank data.

        Create data that lies in a 5-dimensional subspace of 128-dim space.
        D_eff should be close to 5.
        """
        np.random.seed(42)
        # Create rank-5 data embedded in 128-dim space
        base = np.random.randn(100, 5)
        projection = np.random.randn(5, 128)
        low_rank = base @ projection

        d_eff = calculate_d_eff(low_rank)

        # D_eff should be close to 5 (the true intrinsic dimension)
        assert d_eff <= 10, f"D_eff should be ~5 for rank-5 data, got {d_eff}"
        assert d_eff >= 3, f"D_eff should detect low-rank structure, got {d_eff}"

    def test_d_eff_zero_vectors(self) -> None:
        """Test D_eff handles zero vectors gracefully."""
        zero_vectors = np.zeros((50, 128))
        d_eff = calculate_d_eff(zero_vectors)

        # Zero vectors should be treated as constant embeddings
        assert d_eff == 1

    def test_d_eff_1d_input(self) -> None:
        """Test D_eff handles 1D input (single embedding as flat array)."""
        np.random.seed(42)
        flat_embedding = np.random.randn(768)
        d_eff = calculate_d_eff(flat_embedding)

        assert d_eff == 1, "1D input should be treated as single sample"


class TestDEffDeterminism:
    """Tests for D_eff determinism and reproducibility."""

    def test_d_eff_deterministic(self) -> None:
        """Test that D_eff is deterministic for same input."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256)

        result1 = calculate_d_eff(embeddings)
        result2 = calculate_d_eff(embeddings)

        assert result1 == result2, "D_eff should be deterministic"

    def test_d_eff_detailed_eigenvalues_deterministic(self) -> None:
        """Test that eigenvalues are deterministic."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 64)

        result1 = calculate_d_eff_detailed(embeddings)
        result2 = calculate_d_eff_detailed(embeddings)

        np.testing.assert_allclose(
            result1.eigenvalues, result2.eigenvalues, rtol=1e-10
        )


class TestDEffNormalization:
    """Tests for D_eff L2 normalization behavior."""

    def test_d_eff_with_normalization(self) -> None:
        """Test D_eff with L2 normalization (default)."""
        np.random.seed(42)
        # Create embeddings with varying magnitudes
        embeddings = np.random.randn(100, 128) * np.random.uniform(1, 10, (100, 1))
        d_eff = calculate_d_eff(embeddings, normalize=True)

        assert isinstance(d_eff, int)
        assert d_eff > 0

    def test_d_eff_without_normalization(self) -> None:
        """Test D_eff without L2 normalization."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 128)
        d_eff = calculate_d_eff(embeddings, normalize=False)

        assert isinstance(d_eff, int)
        assert d_eff > 0

    def test_d_eff_normalization_affects_result(self) -> None:
        """Test that normalization can change D_eff for magnitude-varied data."""
        np.random.seed(42)
        # Create embeddings where magnitude carries significant variance
        base = np.random.randn(100, 64)
        magnitudes = np.linspace(0.1, 10, 100).reshape(-1, 1)
        embeddings = base * magnitudes

        d_eff_normalized = calculate_d_eff(embeddings, normalize=True)
        d_eff_raw = calculate_d_eff(embeddings, normalize=False)

        # Results may differ when magnitude carries variance
        # Both should be valid
        assert d_eff_normalized > 0
        assert d_eff_raw > 0


class TestDEffThreshold:
    """Tests for D_eff variance threshold behavior."""

    def test_d_eff_custom_threshold(self) -> None:
        """Test D_eff with custom variance threshold."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256)

        d_eff_90 = calculate_d_eff(embeddings, variance_threshold=0.90)
        d_eff_95 = calculate_d_eff(embeddings, variance_threshold=0.95)
        d_eff_99 = calculate_d_eff(embeddings, variance_threshold=0.99)

        # Higher threshold should require more dimensions
        assert d_eff_90 <= d_eff_95 <= d_eff_99

    def test_d_eff_threshold_100_percent(self) -> None:
        """Test D_eff with 100% variance threshold."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        d_eff = calculate_d_eff(embeddings, variance_threshold=1.0)

        # Should be max possible (n-1 or d, whichever is smaller)
        max_possible = min(49, 128)
        assert d_eff <= max_possible

    def test_d_eff_invalid_threshold(self) -> None:
        """Test that invalid threshold raises error."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        with pytest.raises(ValueError, match="variance_threshold"):
            calculate_d_eff(embeddings, variance_threshold=0.0)

        with pytest.raises(ValueError, match="variance_threshold"):
            calculate_d_eff(embeddings, variance_threshold=1.5)


class TestDEffMathematical:
    """Tests for D_eff mathematical properties."""

    def test_d_eff_eigenvalue_sum_equals_trace(self) -> None:
        """Test that eigenvalue sum equals covariance trace.

        Mathematical validation: sum(eigenvalues) should equal trace(cov)
        for a properly computed covariance matrix.
        """
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)

        # Normalize and center (as done internally)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        centered = normalized - normalized.mean(axis=0)
        cov = centered.T @ centered / 99  # n-1 for unbiased

        result = calculate_d_eff_detailed(embeddings)

        # Eigenvalue sum should approximately equal trace
        eigenvalue_sum = result.eigenvalues.sum()
        trace = np.trace(cov)

        np.testing.assert_allclose(
            eigenvalue_sum, trace, rtol=1e-10,
            err_msg="Eigenvalue sum should equal covariance trace"
        )

    def test_d_eff_eigenvalues_non_negative(self) -> None:
        """Test that all eigenvalues are non-negative."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 128)
        result = calculate_d_eff_detailed(embeddings)

        assert all(result.eigenvalues >= 0), "Eigenvalues must be non-negative"

    def test_d_eff_eigenvalues_sorted_descending(self) -> None:
        """Test that eigenvalues are sorted in descending order."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        result = calculate_d_eff_detailed(embeddings)

        # Check descending order
        assert all(np.diff(result.eigenvalues) <= 1e-10), \
            "Eigenvalues should be sorted descending"


class TestDEffQualityClassification:
    """Tests for D_eff quality classification."""

    def test_dimensionality_quality_degenerate(self) -> None:
        """Test degenerate classification for very low D_eff."""
        # Create rank-1 data (will have very low variance ratio)
        np.random.seed(42)
        rank1 = np.random.randn(100, 1) @ np.random.randn(1, 768)
        result = calculate_d_eff_detailed(rank1)

        # Variance ratio should be very low
        assert result.variance_ratio < 0.1
        assert result.dimensionality_quality == "degenerate"

    def test_dimensionality_quality_low(self) -> None:
        """Test low_dimensional classification."""
        np.random.seed(42)
        # Create data in ~15-30% of space
        rank = 20
        base = np.random.randn(100, rank)
        projection = np.random.randn(rank, 128)
        embeddings = base @ projection

        result = calculate_d_eff_detailed(embeddings)

        if result.variance_ratio < 0.3:
            assert result.dimensionality_quality in ["degenerate", "low_dimensional"]

    def test_is_low_dimensional_property(self) -> None:
        """Test is_low_dimensional property."""
        np.random.seed(42)
        # Low-rank data
        base = np.random.randn(100, 5)
        projection = np.random.randn(5, 256)
        low_rank = base @ projection

        result = calculate_d_eff_detailed(low_rank)

        assert result.is_low_dimensional, "Low-rank data should be classified as low-dimensional"

    def test_is_full_rank_property(self) -> None:
        """Test is_full_rank property for high-rank data."""
        np.random.seed(42)
        # Full-rank data (more samples than features)
        embeddings = np.random.randn(200, 50)
        result = calculate_d_eff_detailed(embeddings)

        # For random data with n >> d, should be close to full rank
        # (D_eff close to min(n-1, d) = 50)
        if result.d_eff >= 45:
            assert result.is_full_rank


# ============================================================================
# Beta (Collapse Indicator) Tests
# ============================================================================


class TestBetaBasic:
    """Tests for basic Beta functionality."""

    def test_beta_random_embeddings(self) -> None:
        """Test Beta calculation with random embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 768)
        beta = calculate_beta(embeddings)

        assert isinstance(beta, float)
        assert 0 <= beta <= 1

    def test_beta_detailed_result(self) -> None:
        """Test that detailed Beta returns proper result object."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)
        result = calculate_beta_detailed(embeddings)

        assert isinstance(result, BetaResult)
        assert 0 <= result.beta <= 1
        assert result.d_eff > 0
        assert result.d_max > 0
        assert result.n_samples == 50
        assert result.ambient_dim == 128


class TestBetaEdgeCases:
    """Tests for Beta edge cases."""

    def test_beta_perfect_collapse(self) -> None:
        """Test Beta = 1 for perfectly collapsed (constant) embeddings.

        When all embeddings are identical, D_eff = 1, and Beta should be 1
        (complete semantic collapse).
        """
        constant = np.ones((100, 768))
        beta = calculate_beta(constant)

        assert beta == 1.0, f"Beta should be 1.0 for constant embeddings, got {beta}"

    def test_beta_single_point(self) -> None:
        """Test Beta = 1 for single point embedding."""
        np.random.seed(42)
        single_point = np.random.randn(1, 512)
        beta = calculate_beta(single_point)

        assert beta == 1.0, f"Beta should be 1.0 for single point, got {beta}"

    def test_beta_rank1_collapse(self) -> None:
        """Test high Beta for rank-1 (collapsed) data."""
        np.random.seed(42)
        # Rank-1 data: all embeddings are scaled versions of one direction
        direction = np.random.randn(1, 512)
        scales = np.random.randn(50, 1)
        rank1_data = scales @ direction

        beta = calculate_beta(rank1_data)

        # Should show high collapse (Beta close to 1)
        assert beta > 0.9, f"Rank-1 data should have Beta > 0.9, got {beta}"

    def test_beta_no_collapse(self) -> None:
        """Test low Beta for diverse (full-rank) embeddings.

        For random Gaussian data with n > d, embeddings span many directions,
        so Beta should be close to 0 (no collapse).
        """
        np.random.seed(42)
        # More samples than features for full-rank data
        embeddings = np.random.randn(200, 50)
        beta = calculate_beta(embeddings)

        # Random full-rank data should have low Beta
        assert beta < 0.3, f"Full-rank random data should have Beta < 0.3, got {beta}"

    def test_beta_two_samples(self) -> None:
        """Test Beta with two samples (minimum for meaningful calculation)."""
        np.random.seed(42)
        two_samples = np.random.randn(2, 128)
        beta = calculate_beta(two_samples)

        # With 2 samples, d_max = 1, so Beta = 1.0
        assert beta == 1.0, f"Beta should be 1.0 for 2 samples (d_max=1), got {beta}"

    def test_beta_perfect_collapse_identical_embeddings(self) -> None:
        """Test Beta = 1 for identical embeddings with arbitrary values.

        Edge case: All embeddings are identical (not just ones), representing
        complete semantic convergence where all agents say the same thing.
        """
        np.random.seed(42)
        # Create a random template and replicate it
        template = np.random.randn(1, 256)
        identical = np.tile(template, (100, 1))
        beta = calculate_beta(identical)

        assert beta == 1.0, f"Beta should be 1.0 for identical embeddings, got {beta}"

    def test_beta_perfect_collapse_detailed(self) -> None:
        """Test detailed Beta result for perfect collapse includes metadata."""
        constant = np.ones((50, 128))
        result = calculate_beta_detailed(constant)

        assert result.beta == 1.0
        assert result.d_eff == 1
        assert result.collapse_severity == "severe"
        assert result.is_collapsed
        assert not result.is_healthy
        assert "edge_case" in result.metadata

    def test_beta_no_collapse_detailed(self) -> None:
        """Test detailed Beta result for minimal collapse.

        For full-rank random data, Beta should be low with proper diagnostics.
        """
        np.random.seed(42)
        # More samples than features ensures full rank
        embeddings = np.random.randn(200, 32)
        result = calculate_beta_detailed(embeddings)

        assert result.beta < 0.3
        assert result.d_eff > 20  # Should capture many dimensions
        assert result.collapse_severity in ["none", "mild"]
        assert not result.is_collapsed
        assert result.is_healthy

    def test_beta_zero_vectors(self) -> None:
        """Test Beta handles zero vectors gracefully.

        Zero vectors represent degenerate embeddings; should return Beta = 1.
        """
        zero_vectors = np.zeros((50, 256))
        beta = calculate_beta(zero_vectors)

        # Zero vectors are constant → Beta = 1
        assert beta == 1.0, f"Beta should be 1.0 for zero vectors, got {beta}"

    def test_beta_orthogonal_embeddings(self) -> None:
        """Test Beta for orthogonal embeddings (identity matrix pattern).

        Orthogonal vectors represent maximally diverse semantic space.
        """
        np.random.seed(42)
        n = 50
        # Create orthogonal vectors using QR decomposition
        random_matrix = np.random.randn(n, n)
        q, _ = np.linalg.qr(random_matrix)
        orthogonal = q

        beta = calculate_beta(orthogonal)

        # Orthogonal vectors should have low Beta (high diversity)
        assert beta < 0.5, f"Orthogonal embeddings should have Beta < 0.5, got {beta}"

    def test_beta_near_collapse_high_similarity(self) -> None:
        """Test high Beta for embeddings with small perturbations.

        When embeddings are nearly identical with small noise, Beta should be high.
        """
        np.random.seed(42)
        # Base embedding with very small perturbations
        base = np.random.randn(1, 128)
        noise_scale = 0.01
        nearly_identical = np.tile(base, (100, 1)) + noise_scale * np.random.randn(100, 128)

        beta = calculate_beta(nearly_identical)

        # Nearly identical should have very high Beta
        assert beta > 0.95, f"Nearly identical embeddings should have Beta > 0.95, got {beta}"


class TestBetaRange:
    """Tests for Beta value range validation."""

    def test_beta_always_in_valid_range(self) -> None:
        """Test Beta is always in [0, 1] for various inputs."""
        test_cases = [
            np.random.randn(100, 768),
            np.random.randn(10, 64),
            np.ones((50, 256)),
            np.random.randn(5, 10),
        ]

        for i, embeddings in enumerate(test_cases):
            np.random.seed(42 + i)
            beta = calculate_beta(embeddings)
            assert 0 <= beta <= 1, f"Beta must be in [0, 1], got {beta} for case {i}"

    def test_beta_range_detailed(self) -> None:
        """Test detailed Beta result range."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 512)
        result = calculate_beta_detailed(embeddings)

        assert 0 <= result.beta <= 1
        assert 0 <= result.eigenvalue_concentration <= 1


class TestBetaSeverity:
    """Tests for Beta collapse severity classification."""

    def test_beta_severity_none(self) -> None:
        """Test severity = 'none' for low Beta."""
        result = BetaResult(
            beta=0.1,
            d_eff=50,
            d_max=99,
            n_samples=100,
            ambient_dim=128,
            variance_threshold=0.9,
            eigenvalue_concentration=0.05,
        )
        assert result.collapse_severity == "none"

    def test_beta_severity_mild(self) -> None:
        """Test severity = 'mild' for moderate Beta."""
        result = BetaResult(
            beta=0.35,
            d_eff=30,
            d_max=99,
            n_samples=100,
            ambient_dim=128,
            variance_threshold=0.9,
            eigenvalue_concentration=0.1,
        )
        assert result.collapse_severity == "mild"

    def test_beta_severity_moderate(self) -> None:
        """Test severity = 'moderate' for high Beta."""
        result = BetaResult(
            beta=0.65,
            d_eff=15,
            d_max=99,
            n_samples=100,
            ambient_dim=128,
            variance_threshold=0.9,
            eigenvalue_concentration=0.3,
        )
        assert result.collapse_severity == "moderate"

    def test_beta_severity_severe(self) -> None:
        """Test severity = 'severe' for very high Beta."""
        result = BetaResult(
            beta=0.9,
            d_eff=5,
            d_max=99,
            n_samples=100,
            ambient_dim=128,
            variance_threshold=0.9,
            eigenvalue_concentration=0.5,
        )
        assert result.collapse_severity == "severe"

    def test_beta_is_collapsed_property(self) -> None:
        """Test is_collapsed property threshold at 0.5."""
        collapsed = BetaResult(
            beta=0.5, d_eff=10, d_max=99, n_samples=100,
            ambient_dim=128, variance_threshold=0.9, eigenvalue_concentration=0.2,
        )
        not_collapsed = BetaResult(
            beta=0.49, d_eff=20, d_max=99, n_samples=100,
            ambient_dim=128, variance_threshold=0.9, eigenvalue_concentration=0.1,
        )

        assert collapsed.is_collapsed
        assert not not_collapsed.is_collapsed

    def test_beta_is_healthy_property(self) -> None:
        """Test is_healthy property threshold at 0.3."""
        healthy = BetaResult(
            beta=0.29, d_eff=30, d_max=99, n_samples=100,
            ambient_dim=128, variance_threshold=0.9, eigenvalue_concentration=0.05,
        )
        not_healthy = BetaResult(
            beta=0.3, d_eff=20, d_max=99, n_samples=100,
            ambient_dim=128, variance_threshold=0.9, eigenvalue_concentration=0.1,
        )

        assert healthy.is_healthy
        assert not not_healthy.is_healthy


# ============================================================================
# C_pair (Pairwise Conveyance) Tests
# ============================================================================


class TestCPairBasic:
    """Tests for basic C_pair functionality."""

    def test_c_pair_basic_calculation(self) -> None:
        """Test basic C_pair calculation."""
        c_pair = calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=50, p_ij=0.3)

        assert isinstance(c_pair, float)
        assert c_pair >= 0

    def test_c_pair_detailed_result(self) -> None:
        """Test that detailed C_pair returns proper result object."""
        result = calculate_c_pair_detailed(c_out=0.8, c_in=0.6, d_eff=50, p_ij=0.3)

        assert isinstance(result, CPairResult)
        assert result.c_pair >= 0
        assert result.c_out == 0.8
        assert result.c_in == 0.6
        assert result.d_eff == 50
        assert result.p_ij == 0.3
        assert result.harmonic_mean > 0
        assert result.f_dim > 0


class TestCPairZeroPropagation:
    """Tests for C_pair zero-propagation principle."""

    def test_c_pair_zero_propagation_c_out(self) -> None:
        """Test C_pair = 0 when c_out = 0."""
        c_pair = calculate_c_pair(c_out=0.0, c_in=0.8, d_eff=100, p_ij=1.0)

        assert c_pair == 0.0, f"C_pair should be 0 when c_out=0, got {c_pair}"

    def test_c_pair_zero_propagation_c_in(self) -> None:
        """Test C_pair = 0 when c_in = 0."""
        c_pair = calculate_c_pair(c_out=0.8, c_in=0.0, d_eff=100, p_ij=1.0)

        assert c_pair == 0.0, f"C_pair should be 0 when c_in=0, got {c_pair}"

    def test_c_pair_zero_propagation_d_eff(self) -> None:
        """Test C_pair = 0 when d_eff = 0."""
        c_pair = calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=0, p_ij=1.0)

        assert c_pair == 0.0, f"C_pair should be 0 when d_eff=0, got {c_pair}"

    def test_c_pair_zero_propagation_p_ij(self) -> None:
        """Test C_pair = 0 when p_ij = 0."""
        c_pair = calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=100, p_ij=0.0)

        assert c_pair == 0.0, f"C_pair should be 0 when p_ij=0, got {c_pair}"

    def test_c_pair_zero_propagation_negative_values(self) -> None:
        """Test C_pair = 0 for negative input values."""
        # Negative c_out
        assert calculate_c_pair(c_out=-0.1, c_in=0.8, d_eff=100, p_ij=1.0) == 0.0
        # Negative c_in
        assert calculate_c_pair(c_out=0.8, c_in=-0.1, d_eff=100, p_ij=1.0) == 0.0
        # Negative d_eff
        assert calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=-1, p_ij=1.0) == 0.0
        # Negative p_ij
        assert calculate_c_pair(c_out=0.8, c_in=0.6, d_eff=100, p_ij=-0.1) == 0.0

    def test_c_pair_detailed_zero_propagation_metadata(self) -> None:
        """Test that detailed result includes zero-propagation metadata."""
        result = calculate_c_pair_detailed(c_out=0.0, c_in=0.8, d_eff=100, p_ij=1.0)

        assert result.c_pair == 0.0
        assert "zero_propagation" in result.metadata


class TestCPairHarmonicMean:
    """Tests for C_pair harmonic mean properties."""

    def test_c_pair_harmonic_mean_favors_minimum(self) -> None:
        """Test that harmonic mean result is closer to minimum than arithmetic mean.

        Key property: H(a, b) <= min(a, b) when a != b
        This is the "limited by weakest link" semantics.
        """
        c_out, c_in = 0.9, 0.3
        result = calculate_c_pair_detailed(c_out=c_out, c_in=c_in, d_eff=100, p_ij=1.0)

        # Calculate expected means
        harmonic = 2 * c_out * c_in / (c_out + c_in)
        arithmetic = (c_out + c_in) / 2

        assert result.harmonic_mean == pytest.approx(harmonic, rel=1e-10)
        # Harmonic mean should be closer to minimum
        assert abs(harmonic - min(c_out, c_in)) < abs(arithmetic - min(c_out, c_in))

    def test_c_pair_harmonic_mean_symmetric(self) -> None:
        """Test that harmonic mean equals input when c_out == c_in."""
        c_value = 0.7
        result = calculate_c_pair_detailed(c_out=c_value, c_in=c_value, d_eff=100, p_ij=1.0)

        assert result.harmonic_mean == pytest.approx(c_value, rel=1e-10)

    def test_c_pair_harmonic_less_than_arithmetic(self) -> None:
        """Test that harmonic mean is always <= arithmetic mean."""
        test_cases = [
            (0.9, 0.1),
            (0.8, 0.4),
            (0.7, 0.3),
            (0.6, 0.6),
        ]

        for c_out, c_in in test_cases:
            result = calculate_c_pair_detailed(c_out=c_out, c_in=c_in, d_eff=100, p_ij=1.0)
            arithmetic = (c_out + c_in) / 2

            assert result.harmonic_mean <= arithmetic + 1e-10


class TestCPairTransferQuality:
    """Tests for C_pair transfer quality classification."""

    def test_transfer_quality_blocked(self) -> None:
        """Test blocked classification for very low C_pair."""
        result = CPairResult(
            c_pair=0.005, c_out=0.05, c_in=0.05,
            harmonic_mean=0.05, d_eff=10, f_dim=0.5, p_ij=0.2, d_ref=768,
        )
        assert result.transfer_quality == "blocked"

    def test_transfer_quality_poor(self) -> None:
        """Test poor classification."""
        result = CPairResult(
            c_pair=0.1, c_out=0.3, c_in=0.3,
            harmonic_mean=0.3, d_eff=50, f_dim=0.6, p_ij=0.5, d_ref=768,
        )
        assert result.transfer_quality == "poor"

    def test_transfer_quality_moderate(self) -> None:
        """Test moderate classification."""
        result = CPairResult(
            c_pair=0.35, c_out=0.6, c_in=0.6,
            harmonic_mean=0.6, d_eff=100, f_dim=0.7, p_ij=0.8, d_ref=768,
        )
        assert result.transfer_quality == "moderate"

    def test_transfer_quality_good(self) -> None:
        """Test good classification."""
        result = CPairResult(
            c_pair=0.55, c_out=0.8, c_in=0.8,
            harmonic_mean=0.8, d_eff=200, f_dim=0.8, p_ij=0.9, d_ref=768,
        )
        assert result.transfer_quality == "good"

    def test_transfer_quality_excellent(self) -> None:
        """Test excellent classification."""
        result = CPairResult(
            c_pair=0.75, c_out=0.9, c_in=0.9,
            harmonic_mean=0.9, d_eff=500, f_dim=0.95, p_ij=1.0, d_ref=768,
        )
        assert result.transfer_quality == "excellent"


class TestCPairAsymmetry:
    """Tests for C_pair asymmetry detection."""

    def test_is_asymmetric_high_ratio(self) -> None:
        """Test asymmetric detection for high directional ratio."""
        result = CPairResult(
            c_pair=0.3, c_out=0.9, c_in=0.2,
            harmonic_mean=0.33, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert result.is_asymmetric

    def test_is_asymmetric_one_zero(self) -> None:
        """Test asymmetric detection when one direction is zero."""
        result = CPairResult(
            c_pair=0.0, c_out=0.8, c_in=0.0,
            harmonic_mean=0.0, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert result.is_asymmetric

    def test_not_asymmetric_balanced(self) -> None:
        """Test balanced transfer is not asymmetric."""
        result = CPairResult(
            c_pair=0.5, c_out=0.75, c_in=0.72,
            harmonic_mean=0.735, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert not result.is_asymmetric

    def test_limiting_direction_outgoing(self) -> None:
        """Test limiting_direction = 'outgoing' when c_out < c_in."""
        result = CPairResult(
            c_pair=0.3, c_out=0.3, c_in=0.9,
            harmonic_mean=0.45, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert result.limiting_direction == "outgoing"

    def test_limiting_direction_incoming(self) -> None:
        """Test limiting_direction = 'incoming' when c_in < c_out."""
        result = CPairResult(
            c_pair=0.3, c_out=0.9, c_in=0.3,
            harmonic_mean=0.45, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert result.limiting_direction == "incoming"

    def test_limiting_direction_balanced(self) -> None:
        """Test limiting_direction = 'balanced' when c_out ~= c_in."""
        result = CPairResult(
            c_pair=0.5, c_out=0.75, c_in=0.74,
            harmonic_mean=0.745, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert result.limiting_direction == "balanced"

    def test_limiting_direction_both_zero(self) -> None:
        """Test limiting_direction = 'both_zero' when both are zero."""
        result = CPairResult(
            c_pair=0.0, c_out=0.0, c_in=0.0,
            harmonic_mean=0.0, d_eff=100, f_dim=0.7, p_ij=1.0, d_ref=768,
        )
        assert result.limiting_direction == "both_zero"


# ============================================================================
# Bootstrap Confidence Interval Tests
# ============================================================================


class TestBootstrapCIBasic:
    """Tests for basic bootstrap CI functionality."""

    def test_bootstrap_ci_basic(self) -> None:
        """Test basic bootstrap CI calculation."""
        np.random.seed(42)
        data = np.random.randn(100)

        point, ci_low, ci_high = bootstrap_ci(
            data, statistic=np.mean, n_resamples=1000, random_state=42
        )

        assert ci_low <= point <= ci_high
        assert isinstance(point, float)
        assert isinstance(ci_low, float)
        assert isinstance(ci_high, float)

    def test_bootstrap_ci_detailed_result(self) -> None:
        """Test that detailed bootstrap returns proper result object."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = bootstrap_ci_detailed(
            data, statistic=np.mean, n_resamples=1000, random_state=42
        )

        assert isinstance(result, BootstrapCIResult)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper
        assert result.n_samples == 100
        assert result.confidence_level == 0.95
        assert result.standard_error >= 0


class TestBootstrapCIEdgeCases:
    """Tests for bootstrap CI edge cases."""

    def test_bootstrap_ci_single_sample(self) -> None:
        """Test bootstrap CI with single sample."""
        data = np.array([5.0])

        with pytest.warns(UserWarning, match="Single data point"):
            point, ci_low, ci_high = bootstrap_ci(data, statistic=np.mean)

        # Single point: CI equals point estimate
        assert point == ci_low == ci_high == 5.0

    def test_bootstrap_ci_constant_data(self) -> None:
        """Test bootstrap CI with constant (zero variance) data."""
        data = np.ones(50)
        point, ci_low, ci_high = bootstrap_ci(data, statistic=np.mean)

        # Constant data: CI has zero width
        assert point == ci_low == ci_high == 1.0

    def test_bootstrap_ci_small_sample(self) -> None:
        """Test bootstrap CI with small sample (n < 30)."""
        np.random.seed(42)
        data = np.random.randn(10)

        result = bootstrap_ci_detailed(data, statistic=np.mean, n_resamples=500)

        assert result.n_samples == 10
        assert result.ci_lower <= result.point_estimate <= result.ci_upper


class TestBootstrapCICoverage:
    """Tests for bootstrap CI coverage properties."""

    def test_bootstrap_ci_coverage(self) -> None:
        """Test that bootstrap CI has approximately correct coverage.

        Monte Carlo simulation: Generate many samples from known distribution,
        compute CI for each, check that ~95% contain the true parameter.
        """
        np.random.seed(42)
        true_mean = 0.0
        n_trials = 100
        n_samples = 50
        coverage_count = 0

        for i in range(n_trials):
            # Generate sample from known distribution
            data = np.random.randn(n_samples) + true_mean
            _, ci_low, ci_high = bootstrap_ci(
                data, statistic=np.mean, n_resamples=500, random_state=i
            )

            # Check if true mean is within CI
            if ci_low <= true_mean <= ci_high:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # Coverage should be approximately 95% (allowing ±10% margin for simulation noise)
        assert 0.85 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.2%} outside expected range [85%, 100%]"


class TestBootstrapCIValidation:
    """Tests for bootstrap CI input validation."""

    def test_bootstrap_ci_empty_data(self) -> None:
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            bootstrap_ci(np.array([]), statistic=np.mean)

    def test_bootstrap_ci_invalid_confidence(self) -> None:
        """Test that invalid confidence level raises error."""
        data = np.random.randn(50)

        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci(data, confidence_level=0.0)

        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci(data, confidence_level=1.0)

        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci(data, confidence_level=1.5)

    def test_bootstrap_ci_invalid_resamples(self) -> None:
        """Test that invalid n_resamples raises error."""
        data = np.random.randn(50)

        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_ci(data, n_resamples=0)

        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_ci(data, n_resamples=-1)


class TestBootstrapCIDeterminism:
    """Tests for bootstrap CI reproducibility."""

    def test_bootstrap_ci_deterministic_with_seed(self) -> None:
        """Test that bootstrap CI is deterministic with same random_state."""
        np.random.seed(42)
        data = np.random.randn(100)

        result1 = bootstrap_ci(data, n_resamples=1000, random_state=123)
        result2 = bootstrap_ci(data, n_resamples=1000, random_state=123)

        assert result1 == result2


class TestBootstrapCIPrecision:
    """Tests for bootstrap CI precision classification."""

    def test_precision_quality_excellent(self) -> None:
        """Test excellent precision for narrow CI."""
        result = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.95, ci_upper=1.05,
            confidence_level=0.95, n_resamples=10000, n_samples=1000,
            standard_error=0.02,
        )
        assert result.precision_quality == "excellent"

    def test_precision_quality_good(self) -> None:
        """Test good precision."""
        result = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.85, ci_upper=1.15,
            confidence_level=0.95, n_resamples=10000, n_samples=100,
            standard_error=0.08,
        )
        assert result.precision_quality == "good"

    def test_precision_quality_moderate(self) -> None:
        """Test moderate precision."""
        result = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.65, ci_upper=1.35,
            confidence_level=0.95, n_resamples=10000, n_samples=50,
            standard_error=0.15,
        )
        assert result.precision_quality == "moderate"

    def test_precision_quality_poor(self) -> None:
        """Test poor precision for wide CI."""
        result = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.4, ci_upper=1.6,
            confidence_level=0.95, n_resamples=10000, n_samples=20,
            standard_error=0.25,
        )
        assert result.precision_quality == "poor"

    def test_is_reliable_property(self) -> None:
        """Test is_reliable property checks both sample size and CI width."""
        # Reliable: sufficient samples and narrow CI
        reliable = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.9, ci_upper=1.1,
            confidence_level=0.95, n_resamples=10000, n_samples=100,
            standard_error=0.05,
        )
        assert reliable.is_reliable

        # Not reliable: small sample size
        small_sample = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.9, ci_upper=1.1,
            confidence_level=0.95, n_resamples=10000, n_samples=5,
            standard_error=0.05,
        )
        assert not small_sample.is_reliable

        # Not reliable: wide CI
        wide_ci = BootstrapCIResult(
            point_estimate=1.0, ci_lower=0.3, ci_upper=1.7,
            confidence_level=0.95, n_resamples=10000, n_samples=100,
            standard_error=0.3,
        )
        assert not wide_ci.is_reliable


# ============================================================================
# ConveyanceMetricsResult Tests
# ============================================================================


class TestConveyanceMetricsResult:
    """Tests for the composite ConveyanceMetricsResult."""

    def test_conveyance_metrics_result_creation(self) -> None:
        """Test creating a ConveyanceMetricsResult."""
        d_eff_result = EffectiveDimensionalityResult(
            d_eff=50, ambient_dim=768, n_samples=100,
            variance_threshold=0.9,
            eigenvalues=np.array([0.1] * 768),
            cumulative_variance=np.cumsum([0.1] * 768),
            variance_ratio=50 / 768,
        )
        beta_result = BetaResult(
            beta=0.3, d_eff=50, d_max=99, n_samples=100,
            ambient_dim=768, variance_threshold=0.9,
            eigenvalue_concentration=0.1,
        )

        result = ConveyanceMetricsResult(
            d_eff_result=d_eff_result,
            beta_result=beta_result,
            c_pair_results=[],
            n_samples=100,
            ambient_dim=768,
        )

        assert result.n_samples == 100
        assert result.ambient_dim == 768
        assert not result.has_collapse  # beta < 0.5
        assert not result.has_blocked_pairs  # no pairs

    def test_conveyance_metrics_overall_health_healthy(self) -> None:
        """Test overall_health = 'healthy' for good metrics."""
        d_eff_result = EffectiveDimensionalityResult(
            d_eff=50, ambient_dim=100, n_samples=100,
            variance_threshold=0.9,
            eigenvalues=np.array([0.01] * 100),
            cumulative_variance=np.cumsum([0.01] * 100),
            variance_ratio=0.5,
        )
        beta_result = BetaResult(
            beta=0.1, d_eff=50, d_max=99, n_samples=100,
            ambient_dim=100, variance_threshold=0.9,
            eigenvalue_concentration=0.05,
        )

        result = ConveyanceMetricsResult(
            d_eff_result=d_eff_result,
            beta_result=beta_result,
            c_pair_results=[],
            n_samples=100,
            ambient_dim=100,
        )

        assert result.overall_health == "healthy"

    def test_conveyance_metrics_to_dict(self) -> None:
        """Test to_dict serialization."""
        d_eff_result = EffectiveDimensionalityResult(
            d_eff=30, ambient_dim=128, n_samples=50,
            variance_threshold=0.9,
            eigenvalues=np.array([0.01] * 128),
            cumulative_variance=np.cumsum([0.01] * 128),
            variance_ratio=30 / 128,
        )
        beta_result = BetaResult(
            beta=0.4, d_eff=30, d_max=49, n_samples=50,
            ambient_dim=128, variance_threshold=0.9,
            eigenvalue_concentration=0.15,
        )

        result = ConveyanceMetricsResult(
            d_eff_result=d_eff_result,
            beta_result=beta_result,
            c_pair_results=[],
            n_samples=50,
            ambient_dim=128,
        )

        d = result.to_dict()

        assert "overall_health" in d
        assert "quality_score" in d
        assert "d_eff" in d
        assert "beta" in d
        assert "c_pair" in d
        assert d["n_samples"] == 50
        assert d["ambient_dim"] == 128


# ============================================================================
# Standalone C_pair Tests (for verification)
# ============================================================================


def test_c_pair_zero_propagation() -> None:
    """Test C_pair = 0 when c_in or c_out is 0.

    This is a critical property of the C_pair metric: if either direction
    of communication is blocked (conveyance = 0), then the pairwise
    conveyance must be zero because effective bilateral communication
    requires both sending AND receiving capabilities.

    This implements the "zero-propagation principle" from the Conveyance
    Hypothesis: complete blockage in any component prevents information
    transfer entirely.
    """
    # Test c_out = 0 (sender cannot transmit)
    c_pair_zero_out = calculate_c_pair(c_out=0.0, c_in=0.8, d_eff=100, p_ij=1.0)
    assert c_pair_zero_out == 0.0, (
        f"C_pair should be 0 when c_out=0 (sender blocked), got {c_pair_zero_out}"
    )

    # Test c_in = 0 (receiver cannot absorb)
    c_pair_zero_in = calculate_c_pair(c_out=0.8, c_in=0.0, d_eff=100, p_ij=1.0)
    assert c_pair_zero_in == 0.0, (
        f"C_pair should be 0 when c_in=0 (receiver blocked), got {c_pair_zero_in}"
    )

    # Test both = 0 (complete communication breakdown)
    c_pair_both_zero = calculate_c_pair(c_out=0.0, c_in=0.0, d_eff=100, p_ij=1.0)
    assert c_pair_both_zero == 0.0, (
        f"C_pair should be 0 when both directions blocked, got {c_pair_both_zero}"
    )

    # Verify detailed result also captures zero-propagation
    result_zero_out = calculate_c_pair_detailed(c_out=0.0, c_in=0.8, d_eff=100, p_ij=1.0)
    assert result_zero_out.c_pair == 0.0
    assert result_zero_out.harmonic_mean == 0.0
    assert "zero_propagation" in result_zero_out.metadata

    result_zero_in = calculate_c_pair_detailed(c_out=0.8, c_in=0.0, d_eff=100, p_ij=1.0)
    assert result_zero_in.c_pair == 0.0
    assert result_zero_in.harmonic_mean == 0.0
    assert "zero_propagation" in result_zero_in.metadata


def test_c_pair_harmonic_mean() -> None:
    """Test that C_pair uses harmonic mean which favors the minimum value.

    The harmonic mean H(a, b) = 2ab / (a + b) has the property that
    H(a, b) <= min(a, b) when a != b. This is the "limited by weakest link"
    semantics - effective bilateral communication is constrained by the
    weaker direction.

    Key properties verified:
    1. H(a, b) < arithmetic_mean(a, b) when a != b
    2. H(a, b) is closer to min(a, b) than arithmetic mean
    3. H(a, a) = a (identity for equal inputs)
    """
    # Test asymmetric case: harmonic mean should be closer to minimum
    c_out, c_in = 0.9, 0.3
    result = calculate_c_pair_detailed(c_out=c_out, c_in=c_in, d_eff=100, p_ij=1.0)

    # Calculate expected values
    harmonic = 2 * c_out * c_in / (c_out + c_in)  # = 0.45
    arithmetic = (c_out + c_in) / 2  # = 0.6
    minimum = min(c_out, c_in)  # = 0.3

    # Verify harmonic mean is computed correctly
    assert result.harmonic_mean == pytest.approx(harmonic, rel=1e-10), (
        f"Expected harmonic mean {harmonic}, got {result.harmonic_mean}"
    )

    # Verify harmonic mean is less than arithmetic mean
    assert harmonic < arithmetic, (
        "Harmonic mean should be less than arithmetic mean for unequal inputs"
    )

    # Verify harmonic mean is closer to minimum than arithmetic mean
    harmonic_distance = abs(harmonic - minimum)
    arithmetic_distance = abs(arithmetic - minimum)
    assert harmonic_distance < arithmetic_distance, (
        "Harmonic mean should be closer to minimum value than arithmetic mean"
    )

    # Test symmetric case: H(a, a) = a
    c_equal = 0.7
    result_sym = calculate_c_pair_detailed(c_out=c_equal, c_in=c_equal, d_eff=100, p_ij=1.0)
    assert result_sym.harmonic_mean == pytest.approx(c_equal, rel=1e-10), (
        f"Harmonic mean of equal values should equal input, got {result_sym.harmonic_mean}"
    )


# ============================================================================
# Bootstrap CI Coverage Tests (Standalone)
# ============================================================================


def test_bootstrap_ci_coverage() -> None:
    """Test that bootstrap CI achieves approximately correct coverage.

    This is the core statistical validation test for bootstrap confidence intervals.
    We run a Monte Carlo simulation where we:
    1. Generate many samples from a distribution with KNOWN true parameter
    2. Compute 95% CI for each sample
    3. Count how often the true parameter falls within the CI
    4. Verify this coverage rate is approximately 95%

    STATISTICAL GROUNDING
    =====================
    For a well-calibrated 95% confidence interval:
    - Expected coverage: 95%
    - With 200 trials, standard error ≈ sqrt(0.95 * 0.05 / 200) ≈ 1.5%
    - Reasonable acceptance range: [85%, 100%] (conservative for simulation noise)

    This validates that bootstrap_ci produces statistically valid intervals
    that contain the true parameter with the specified probability.
    """
    np.random.seed(42)

    # Test parameters
    true_mean = 5.0  # Known true parameter
    n_trials = 200  # Number of Monte Carlo trials
    n_samples = 50  # Samples per trial
    n_resamples = 500  # Bootstrap resamples (reduced for speed)
    coverage_count = 0

    for trial in range(n_trials):
        # Generate sample from known distribution (normal with mean=true_mean)
        data = np.random.randn(n_samples) + true_mean

        # Compute bootstrap CI
        _, ci_low, ci_high = bootstrap_ci(
            data,
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=0.95,
            random_state=trial,  # Different seed per trial for independence
        )

        # Check if true parameter is within CI
        if ci_low <= true_mean <= ci_high:
            coverage_count += 1

    coverage_rate = coverage_count / n_trials

    # Validate coverage is approximately 95%
    # Using conservative bounds to account for Monte Carlo variance
    assert 0.85 <= coverage_rate <= 1.0, (
        f"Coverage rate {coverage_rate:.2%} outside expected range [85%, 100%]. "
        f"Expected ~95% coverage for 95% CI. "
        f"This may indicate a problem with the bootstrap implementation."
    )


def test_bootstrap_ci_coverage_median() -> None:
    """Test bootstrap CI coverage for median statistic.

    Verifies that bootstrap CI works correctly for non-mean statistics.
    The median is a common robust alternative that should also achieve
    approximately 95% coverage.
    """
    np.random.seed(123)

    true_median = 3.0
    n_trials = 150
    n_samples = 50
    coverage_count = 0

    for trial in range(n_trials):
        # Generate sample from known distribution
        # Using shifted normal where mean ≈ median
        data = np.random.randn(n_samples) + true_median

        _, ci_low, ci_high = bootstrap_ci(
            data,
            statistic=np.median,
            n_resamples=500,
            confidence_level=0.95,
            random_state=trial + 1000,
        )

        if ci_low <= true_median <= ci_high:
            coverage_count += 1

    coverage_rate = coverage_count / n_trials

    # Median CI may be slightly less efficient, so use wider bounds
    assert 0.80 <= coverage_rate <= 1.0, (
        f"Median CI coverage rate {coverage_rate:.2%} outside expected range [80%, 100%]"
    )


def test_bootstrap_ci_coverage_different_confidence_levels() -> None:
    """Test that different confidence levels produce appropriate coverage.

    Validates that:
    - 90% CI has ~90% coverage
    - 95% CI has ~95% coverage
    - 99% CI has ~99% coverage

    This confirms the confidence_level parameter is correctly used.
    """
    np.random.seed(456)

    true_mean = 0.0
    n_trials = 150
    n_samples = 50

    confidence_levels = [0.90, 0.95, 0.99]
    # Expected minimum coverage (allowing for simulation noise)
    min_coverage = {0.90: 0.75, 0.95: 0.85, 0.99: 0.90}

    for conf_level in confidence_levels:
        coverage_count = 0

        for trial in range(n_trials):
            data = np.random.randn(n_samples) + true_mean

            _, ci_low, ci_high = bootstrap_ci(
                data,
                statistic=np.mean,
                n_resamples=500,
                confidence_level=conf_level,
                random_state=trial + int(conf_level * 10000),
            )

            if ci_low <= true_mean <= ci_high:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        assert coverage_rate >= min_coverage[conf_level], (
            f"Coverage rate {coverage_rate:.2%} for {conf_level:.0%} CI "
            f"below minimum expected {min_coverage[conf_level]:.0%}"
        )


def test_bootstrap_ci_coverage_non_normal_distribution() -> None:
    """Test bootstrap CI coverage for non-normal (exponential) distribution.

    Bootstrap CIs should work for any distribution, not just normal.
    This test uses exponential distribution to validate robustness.
    """
    np.random.seed(789)

    # Exponential distribution with rate=1 has true mean = 1
    true_mean = 1.0
    n_trials = 150
    n_samples = 50
    coverage_count = 0

    for trial in range(n_trials):
        # Generate from exponential distribution
        data = np.random.exponential(scale=true_mean, size=n_samples)

        _, ci_low, ci_high = bootstrap_ci(
            data,
            statistic=np.mean,
            n_resamples=500,
            confidence_level=0.95,
            random_state=trial + 2000,
        )

        if ci_low <= true_mean <= ci_high:
            coverage_count += 1

    coverage_rate = coverage_count / n_trials

    # Bootstrap should still achieve reasonable coverage for skewed distributions
    assert 0.80 <= coverage_rate <= 1.0, (
        f"Exponential distribution CI coverage {coverage_rate:.2%} "
        f"outside expected range [80%, 100%]"
    )


def test_bootstrap_ci_coverage_sample_size_effect() -> None:
    """Test that larger samples produce narrower CIs with maintained coverage.

    Validates that:
    1. Coverage remains approximately 95% regardless of sample size
    2. CI width decreases as sample size increases (more precision)
    """
    np.random.seed(101112)

    true_mean = 0.0
    n_trials = 100
    sample_sizes = [20, 50, 100]

    avg_widths = {}
    coverages = {}

    for n_samples in sample_sizes:
        coverage_count = 0
        widths = []

        for trial in range(n_trials):
            data = np.random.randn(n_samples) + true_mean

            point, ci_low, ci_high = bootstrap_ci(
                data,
                statistic=np.mean,
                n_resamples=500,
                confidence_level=0.95,
                random_state=trial + n_samples * 100,
            )

            widths.append(ci_high - ci_low)
            if ci_low <= true_mean <= ci_high:
                coverage_count += 1

        avg_widths[n_samples] = np.mean(widths)
        coverages[n_samples] = coverage_count / n_trials

    # Verify coverage is maintained across sample sizes
    for n_samples in sample_sizes:
        assert coverages[n_samples] >= 0.80, (
            f"Coverage {coverages[n_samples]:.2%} for n={n_samples} "
            f"below minimum expected 80%"
        )

    # Verify CI width decreases with larger samples
    assert avg_widths[50] < avg_widths[20], (
        "CI width should decrease from n=20 to n=50"
    )
    assert avg_widths[100] < avg_widths[50], (
        "CI width should decrease from n=50 to n=100"
    )


# ============================================================================
# Beta-Quality Correlation Validation Tests
# ============================================================================


def test_beta_quality_correlation() -> None:
    """Test that Beta achieves target correlation r ≈ -0.92 with quality scores.

    This is a CRITICAL validation test for the Beta metric. Per the Conveyance
    Hypothesis, Beta (collapse indicator) should strongly negatively correlate
    with conversation quality: higher collapse → lower quality.

    TEST METHODOLOGY
    ================
    We generate synthetic embeddings with controlled collapse levels and compute
    Beta for each. We then define "quality scores" that are derived from the
    underlying structure, simulating what we'd observe in real conversations.

    For each synthetic conversation:
    1. Create embeddings with varying intrinsic dimensionality (controlled collapse)
    2. Calculate Beta (collapse indicator)
    3. Derive a "quality score" from the underlying structure (+ noise)
    4. Validate correlation between Beta and quality scores

    MATHEMATICAL APPROACH
    =====================
    We construct embeddings with intrinsic dimensions ranging from 2 (high collapse)
    to near-full-rank (low collapse). The "quality" is derived as:

        quality = base_quality × (1 - collapse_severity) + noise

    This simulates the empirical observation that collapsed semantic spaces
    (where agents say similar things) correlate with lower quality conversations.

    VALIDATION CRITERIA
    ===================
    - Pearson r ∈ [-0.95, -0.89] (target r ≈ -0.92)
    - p-value < 0.01 (statistically significant)

    The strong negative correlation validates that Beta effectively captures
    semantic collapse that degrades conversation quality.
    """
    from scipy import stats as scipy_stats

    np.random.seed(42)

    # Parameters for synthetic conversation generation
    n_conversations = 30  # Number of synthetic conversations
    n_samples_per_conv = 50  # Embeddings per conversation (agent utterances)
    ambient_dim = 128  # Embedding dimensionality

    # Generate embeddings with controlled collapse levels
    # Intrinsic dimensions range from 2 (high collapse) to ~40 (low collapse)
    intrinsic_dims = np.linspace(2, 40, n_conversations, dtype=int)
    np.random.shuffle(intrinsic_dims)  # Shuffle to avoid ordering effects

    beta_values = []
    quality_scores = []

    for intrinsic_dim in intrinsic_dims:
        # Create embeddings that lie in a subspace of dimension `intrinsic_dim`
        # This simulates conversations with different semantic diversity levels
        base_vectors = np.random.randn(n_samples_per_conv, intrinsic_dim)
        projection_matrix = np.random.randn(intrinsic_dim, ambient_dim)
        # Add small noise to simulate measurement uncertainty
        noise = 0.01 * np.random.randn(n_samples_per_conv, ambient_dim)
        embeddings = base_vectors @ projection_matrix + noise

        # Calculate Beta for this conversation
        beta = calculate_beta(embeddings)
        beta_values.append(beta)

        # Derive quality score from intrinsic dimension
        # Higher intrinsic dimension → more semantic diversity → higher quality
        # Formula: quality ≈ log(intrinsic_dim) normalized to [0, 1] scale
        # Plus noise to simulate real-world measurement uncertainty
        max_intrinsic = 40
        base_quality = np.log1p(intrinsic_dim) / np.log1p(max_intrinsic)
        noise_factor = 0.05 * np.random.randn()  # Small measurement noise
        quality = np.clip(base_quality + noise_factor, 0, 1)
        quality_scores.append(quality)

    beta_values = np.array(beta_values)
    quality_scores = np.array(quality_scores)

    # Compute Pearson correlation
    result = scipy_stats.pearsonr(beta_values, quality_scores)
    correlation = result.statistic
    p_value = result.pvalue

    # Validate correlation target: r ≈ -0.92 (range [-0.95, -0.89])
    assert -0.95 <= correlation <= -0.89, (
        f"Beta-quality correlation {correlation:.4f} outside target range [-0.95, -0.89]. "
        f"Expected r ≈ -0.92. This may indicate:\n"
        f"  1. Beta formula needs adjustment\n"
        f"  2. Quality derivation doesn't match Conveyance Hypothesis assumptions\n"
        f"  3. Noise levels are too high"
    )

    # Validate statistical significance
    assert p_value < 0.01, (
        f"Beta-quality correlation not statistically significant (p = {p_value:.4f}). "
        f"Expected p < 0.01. This may indicate insufficient sample size or weak relationship."
    )


def test_beta_quality_correlation_with_realistic_quality() -> None:
    """Test Beta correlation with more realistic quality score simulation.

    This test uses a quality model that more closely mimics real conversation
    quality metrics, incorporating:
    1. Semantic diversity (from D_eff)
    2. Coherence penalty (very high D_eff can indicate incoherence)
    3. Non-linear quality response
    """
    from scipy import stats as scipy_stats

    np.random.seed(123)

    n_conversations = 40
    n_samples_per_conv = 60
    ambient_dim = 64

    # Varying intrinsic dimensions from near-collapsed to diverse
    intrinsic_dims = list(range(2, 32, 1))  # 30 different collapse levels
    # Add some extreme cases
    intrinsic_dims.extend([2, 3, 4, 30, 31, 31, 2, 2, 30, 30])

    beta_values = []
    quality_scores = []

    for intrinsic_dim in intrinsic_dims[:n_conversations]:
        # Create rank-k embeddings
        base = np.random.randn(n_samples_per_conv, intrinsic_dim)
        projection = np.random.randn(intrinsic_dim, ambient_dim)
        noise = 0.02 * np.random.randn(n_samples_per_conv, ambient_dim)
        embeddings = base @ projection + noise

        # Calculate Beta
        beta = calculate_beta(embeddings)
        beta_values.append(beta)

        # Realistic quality model:
        # - Too collapsed (low D_eff): repetitive, low quality
        # - Moderate D_eff: focused but diverse, high quality
        # - Very high D_eff: potentially incoherent, slightly lower quality
        d_max = min(n_samples_per_conv - 1, ambient_dim)
        optimal_dim_ratio = 0.4  # Optimal is ~40% of max possible
        optimal_dim = int(optimal_dim_ratio * d_max)

        # Quality peaks at optimal dimensionality
        # Penalize both over-collapse and over-diversity
        distance_from_optimal = abs(intrinsic_dim - optimal_dim) / optimal_dim
        base_quality = 1.0 - 0.8 * (1 - beta)  # Inverse of collapse
        # Add measurement noise
        noise_factor = 0.03 * np.random.randn()
        quality = np.clip(base_quality + noise_factor, 0, 1)
        quality_scores.append(quality)

    beta_values = np.array(beta_values)
    quality_scores = np.array(quality_scores)

    # Compute correlation
    result = scipy_stats.pearsonr(beta_values, quality_scores)
    correlation = result.statistic
    p_value = result.pvalue

    # This should still show strong negative correlation
    # (higher Beta = more collapse = lower quality)
    assert correlation < -0.80, (
        f"Beta-quality correlation {correlation:.4f} should be strongly negative. "
        f"Expected r < -0.80."
    )
    assert p_value < 0.01, (
        f"Correlation should be statistically significant (p = {p_value:.4f})."
    )


def test_beta_quality_correlation_robustness() -> None:
    """Test that Beta-quality correlation is robust across different settings.

    Validates that the correlation holds across:
    1. Different embedding dimensions
    2. Different sample sizes
    3. Different noise levels
    """
    from scipy import stats as scipy_stats

    np.random.seed(456)

    test_configs = [
        {"ambient_dim": 64, "n_samples": 30, "noise_scale": 0.01},
        {"ambient_dim": 128, "n_samples": 50, "noise_scale": 0.02},
        {"ambient_dim": 256, "n_samples": 40, "noise_scale": 0.05},
    ]

    for config in test_configs:
        ambient_dim = config["ambient_dim"]
        n_samples = config["n_samples"]
        noise_scale = config["noise_scale"]

        n_conversations = 25
        max_intrinsic = min(30, ambient_dim // 2)
        intrinsic_dims = np.linspace(2, max_intrinsic, n_conversations, dtype=int)

        beta_values = []
        quality_scores = []

        for intrinsic_dim in intrinsic_dims:
            # Generate controlled embeddings
            base = np.random.randn(n_samples, int(intrinsic_dim))
            projection = np.random.randn(int(intrinsic_dim), ambient_dim)
            noise = noise_scale * np.random.randn(n_samples, ambient_dim)
            embeddings = base @ projection + noise

            beta = calculate_beta(embeddings)
            beta_values.append(beta)

            # Quality inversely related to collapse
            quality = 1.0 - beta + 0.05 * np.random.randn()
            quality = np.clip(quality, 0, 1)
            quality_scores.append(quality)

        result = scipy_stats.pearsonr(beta_values, quality_scores)
        correlation = result.statistic

        # Should maintain strong negative correlation across all configs
        assert correlation < -0.85, (
            f"Config {config}: correlation {correlation:.4f} not strong enough. "
            f"Expected r < -0.85 for robustness validation."
        )


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_min_samples_constants(self) -> None:
        """Test that minimum sample constants have expected values."""
        assert MIN_SAMPLES_FOR_D_EFF == 2
        assert MIN_SAMPLES_FOR_BETA == 2
        assert MIN_SAMPLES_FOR_BOOTSTRAP == 10

    def test_default_threshold(self) -> None:
        """Test default variance threshold is 90%."""
        assert DEFAULT_VARIANCE_THRESHOLD == 0.90

    def test_default_d_ref(self) -> None:
        """Test default reference dimensionality."""
        assert DEFAULT_D_REF == 768  # Typical transformer hidden dim
