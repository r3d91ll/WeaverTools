"""Tests for the Kakeya geometry analysis module.

These tests validate the geometric analysis functions used to study
transformer hidden state representations.
"""

import numpy as np
import pytest

from src.analysis import (
    MIN_SAMPLES_FOR_ANALYSIS,
    BilateralGeometryResult,
    DirectionalCoverageResult,
    GrainAnalysisResult,
    HiddenStateProtocol,
    KakeyaGeometryReport,
    WolfAxiomResult,
    analyze_directional_coverage,
    analyze_grains,
    analyze_hidden_state_batch,
    analyze_kakeya_geometry,
    check_wolf_axioms,
    compare_bilateral_geometry,
    run_conveyance_experiment,
)


class TestAnalyzeKakeyaGeometry:
    """Tests for the main analyze_kakeya_geometry function."""

    def test_basic_analysis(self) -> None:
        """Test basic Kakeya geometry analysis with random vectors."""
        np.random.seed(42)
        vectors = np.random.randn(100, 768)
        report = analyze_kakeya_geometry(vectors)

        assert isinstance(report, KakeyaGeometryReport)
        # Health can be "healthy", "warning:*", or "unhealthy:*"
        assert report.overall_health.startswith(("healthy", "warning:", "unhealthy:"))
        assert 0 <= report.directional_coverage.coverage_ratio <= 1
        assert report.wolf_axiom.max_density_ratio > 0
        assert report.grain_analysis.num_grains >= 0

    def test_small_vectors(self) -> None:
        """Test analysis with small number of vectors."""
        np.random.seed(42)
        vectors = np.random.randn(10, 64)
        report = analyze_kakeya_geometry(vectors)

        assert isinstance(report, KakeyaGeometryReport)
        assert report.directional_coverage.ambient_dim == 64

    def test_normalized_input(self) -> None:
        """Test that analysis works with pre-normalized vectors."""
        np.random.seed(42)
        vectors = np.random.randn(50, 128)
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms

        report = analyze_kakeya_geometry(normalized)
        assert isinstance(report, KakeyaGeometryReport)

    def test_metadata_populated(self) -> None:
        """Test that report fields are populated."""
        np.random.seed(42)
        vectors = np.random.randn(50, 256)
        report = analyze_kakeya_geometry(vectors)

        # num_vectors and ambient_dim are top-level attributes
        assert report.num_vectors == 50
        assert report.ambient_dim == 256


class TestWolfAxiomAnalysis:
    """Tests for Wolf axiom density analysis."""

    def test_check_wolf_axioms_basic(self) -> None:
        """Test basic Wolf axiom checking."""
        np.random.seed(42)
        vectors = np.random.randn(100, 128)
        result = check_wolf_axioms(vectors)

        assert isinstance(result, WolfAxiomResult)
        assert result.max_density_ratio > 0
        assert 0 <= result.uniformity_p_value <= 1
        assert result.num_regions_tested > 0

    def test_wolf_axiom_severity_levels(self) -> None:
        """Test severity classification."""
        # Create result with known density ratio
        result = WolfAxiomResult(
            max_density_ratio=1.0,
            mean_density_ratio=1.0,
            density_ratios=[1.0],
            uniformity_p_value=0.5,
            violation_count=0,
            violation_threshold=2.5,
            num_regions_tested=10,
        )
        assert result.severity == "none"

        result.max_density_ratio = 2.0
        assert result.severity == "mild"

        result.max_density_ratio = 4.0
        assert result.severity == "moderate"

        result.max_density_ratio = 6.0
        assert result.severity == "severe"

    def test_wolf_axiom_deterministic(self) -> None:
        """Test that results are deterministic with same random state."""
        vectors = np.random.randn(50, 64)

        result1 = check_wolf_axioms(vectors, random_state=42)
        result2 = check_wolf_axioms(vectors, random_state=42)

        assert result1.max_density_ratio == result2.max_density_ratio
        assert result1.uniformity_p_value == result2.uniformity_p_value


class TestDirectionalCoverage:
    """Tests for directional coverage analysis."""

    def test_analyze_directional_coverage_basic(self) -> None:
        """Test basic directional coverage analysis."""
        np.random.seed(42)
        vectors = np.random.randn(100, 128)
        result = analyze_directional_coverage(vectors)

        assert isinstance(result, DirectionalCoverageResult)
        assert result.ambient_dim == 128
        assert result.effective_dim > 0
        assert result.effective_dim <= result.ambient_dim
        assert 0 <= result.coverage_ratio <= 1
        assert 0 <= result.spherical_uniformity <= 1

    def test_coverage_quality_classification(self) -> None:
        """Test coverage quality classification."""
        np.random.seed(42)
        vectors = np.random.randn(100, 128)
        result = analyze_directional_coverage(vectors)

        assert result.coverage_quality in ["degenerate", "sparse", "moderate", "full"]

    def test_degenerate_detection(self) -> None:
        """Test detection of degenerate (low-dimensional) distributions."""
        # Create vectors that lie in a low-dimensional subspace
        np.random.seed(42)
        base = np.random.randn(100, 5)  # 5-dim subspace
        projection = np.random.randn(5, 128)  # Project to 128-dim
        vectors = base @ projection

        result = analyze_directional_coverage(vectors)
        # Effective dim should be close to 5
        assert result.effective_dim <= 10  # Some tolerance


class TestGrainAnalysis:
    """Tests for grain (cluster) detection."""

    def test_analyze_grains_basic(self) -> None:
        """Test basic grain analysis."""
        np.random.seed(42)
        vectors = np.random.randn(100, 64)
        result = analyze_grains(vectors)

        assert isinstance(result, GrainAnalysisResult)
        assert result.num_grains >= 0
        assert 0 <= result.grain_coverage <= 1

    def test_clustered_data_finds_grains(self) -> None:
        """Test that clustered data produces grains."""
        np.random.seed(42)
        # Create clearly clustered data
        cluster1 = np.random.randn(30, 64) + np.array([5.0] * 64)
        cluster2 = np.random.randn(30, 64) + np.array([-5.0] * 64)
        cluster3 = np.random.randn(30, 64)
        vectors = np.vstack([cluster1, cluster2, cluster3])

        result = analyze_grains(vectors)
        # Should find at least 2 clusters
        assert result.num_grains >= 2


class TestBilateralComparison:
    """Tests for bilateral geometry comparison."""

    def test_bilateral_comparison_identical(self) -> None:
        """Test bilateral comparison with identical vectors."""
        np.random.seed(42)
        vectors = np.random.randn(50, 128)

        bilateral = compare_bilateral_geometry(vectors, vectors)

        assert isinstance(bilateral, BilateralGeometryResult)
        assert bilateral.directional_alignment == pytest.approx(1.0, abs=0.01)
        assert bilateral.subspace_overlap == pytest.approx(1.0, abs=0.01)

    def test_bilateral_comparison_similar(self) -> None:
        """Test bilateral comparison with similar vectors."""
        np.random.seed(42)
        sender = np.random.randn(50, 128)
        receiver = sender + np.random.randn(50, 128) * 0.1  # Small perturbation

        bilateral = compare_bilateral_geometry(sender, receiver)

        assert 0 <= bilateral.overall_alignment <= 1
        assert bilateral.overall_alignment > 0.5  # Should be high for similar vectors

    def test_bilateral_comparison_orthogonal(self) -> None:
        """Test bilateral comparison with orthogonal vectors."""
        np.random.seed(42)
        sender = np.random.randn(50, 128)
        # Create receiver in a different subspace
        receiver = np.random.randn(50, 128)
        # Orthogonalize receiver against sender mean
        sender_mean = sender.mean(axis=0)
        receiver = receiver - np.outer(receiver @ sender_mean, sender_mean) / np.dot(sender_mean, sender_mean)

        bilateral = compare_bilateral_geometry(sender, receiver)

        # Alignment should be lower for orthogonal vectors
        assert bilateral.overall_alignment < 0.8


class TestConveyanceExperiment:
    """Tests for the conveyance experiment runner."""

    def test_run_conveyance_experiment_basic(self) -> None:
        """Test basic conveyance experiment."""
        np.random.seed(42)
        # Create batches of vectors (need >= MIN_SAMPLES_FOR_ANALYSIS per batch)
        sender_states = [np.random.randn(5, 128) for _ in range(10)]
        receiver_states = [np.random.randn(5, 128) for _ in range(10)]
        task_success = [True, False, True, True, False, True, True, False, True, True]

        result = run_conveyance_experiment(sender_states, receiver_states, task_success)

        assert "n_interactions" in result
        assert "n_analyzed" in result
        assert "n_skipped" in result
        assert "alignment" in result
        assert "hypothesis_support" in result

    def test_run_conveyance_experiment_skips_small_samples(self) -> None:
        """Test that experiment skips interactions with too few samples."""
        np.random.seed(42)
        # Create single-vector inputs (should be skipped)
        sender_states = [np.random.randn(1, 128) for _ in range(5)]
        receiver_states = [np.random.randn(1, 128) for _ in range(5)]
        task_success = [True, False, True, True, False]

        with pytest.warns(UserWarning, match="Skipping interaction"):
            result = run_conveyance_experiment(sender_states, receiver_states, task_success)

        assert result["n_skipped"] == 5
        assert result["n_analyzed"] == 0

    def test_run_conveyance_experiment_length_mismatch(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        sender_states = [np.random.randn(5, 128) for _ in range(10)]
        receiver_states = [np.random.randn(5, 128) for _ in range(8)]  # Wrong length
        task_success = [True] * 10

        with pytest.raises(ValueError, match="same length"):
            run_conveyance_experiment(sender_states, receiver_states, task_success)


class TestHiddenStateBatch:
    """Tests for hidden state batch analysis."""

    def test_analyze_hidden_state_batch_with_arrays(self) -> None:
        """Test batch analysis with raw numpy arrays."""
        np.random.seed(42)
        arrays = [np.random.randn(128) for _ in range(20)]

        report = analyze_hidden_state_batch(arrays, normalize=True)

        assert isinstance(report, KakeyaGeometryReport)

    def test_protocol_conformance(self) -> None:
        """Test that HiddenStateProtocol works as expected."""
        # Create a mock object that conforms to the protocol
        class MockHiddenState:
            def __init__(self, vec: np.ndarray) -> None:
                self._vector = vec

            @property
            def vector(self) -> np.ndarray:
                return self._vector

            def l2_normalize(self) -> "MockHiddenState":
                norm = np.linalg.norm(self._vector)
                return MockHiddenState(self._vector / norm if norm > 0 else self._vector)

        # Check protocol conformance
        mock = MockHiddenState(np.random.randn(128))
        assert isinstance(mock, HiddenStateProtocol)


class TestMinSamplesConstant:
    """Tests for the MIN_SAMPLES_FOR_ANALYSIS constant."""

    def test_constant_value(self) -> None:
        """Test that the constant has expected value."""
        assert MIN_SAMPLES_FOR_ANALYSIS == 3

    def test_constant_used_in_experiment(self) -> None:
        """Test that constant is respected in experiment runner."""
        np.random.seed(42)
        # Create inputs with exactly MIN_SAMPLES_FOR_ANALYSIS - should work
        sender_states = [np.random.randn(MIN_SAMPLES_FOR_ANALYSIS, 64)]
        receiver_states = [np.random.randn(MIN_SAMPLES_FOR_ANALYSIS, 64)]
        task_success = [True]

        result = run_conveyance_experiment(sender_states, receiver_states, task_success)
        assert result["n_analyzed"] == 1
        assert result["n_skipped"] == 0
