"""Tests for the Atlas interpretability analysis modules.

These tests validate the analysis functions used to study Atlas model
training dynamics and memory evolution.

Test Coverage:
- AlignedPCA: Epoch 186 reference frame, component consistency, variance explained
- Memory Tracing: Matrix statistics, layer analysis, epoch tracking
- Atlas Statistics: Trend analysis, outlier detection, summary metrics
- Visualization: Configuration classes, export functions
- Batch Pipeline: Stage execution, result aggregation
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from typing import Any

from src.analysis import (
    # AlignedPCA module
    DEFAULT_REFERENCE_EPOCH,
    DEFAULT_N_COMPONENTS,
    VARIANCE_EXPLAINED_THRESHOLD,
    AlignedPCA,
    AlignedPCAFitResult,
    AlignedPCAResult,
    CrossEpochTrajectory,
    build_cross_epoch_trajectory,
    compute_epoch_distances,
    compute_convergence_curve,
    # Memory Tracing module
    SPARSITY_THRESHOLD,
    RANK_THRESHOLD,
    TOP_SINGULAR_VALUES,
    MIN_EPOCHS_FOR_TREND,
    LayerMemoryStats,
    MemoryEpisodeStats,
    MemoryEvolutionResult,
    compute_matrix_stats,
    analyze_layer_memory,
    analyze_memory_states,
    MemoryTracer,
    # Atlas Statistics module
    OUTLIER_Z_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    MIN_EPOCHS_FOR_STATS,
    EpochStatistics,
    TrendAnalysis,
    OutlierDetectionResult,
    AtlasStatisticsResult,
    compute_epoch_statistics,
    analyze_trends,
    detect_outliers,
    compute_summary_statistics,
    # Visualization module
    DEFAULT_AXIS_RANGE,
    DEFAULT_PNG_WIDTH,
    DEFAULT_PNG_HEIGHT,
    PNG_SCALE_FACTOR,
    MAX_HTML_SIZE_BYTES,
    VisualizationStyle,
    AnimationConfig,
    ExportConfig,
    Axis3DConfig,
    Landscape3DConfig,
    # Batch Pipeline module
    TOTAL_EPOCHS,
    GPU_MEMORY_BUDGET_MB,
    ValidationSummary,
    PipelineStageResult,
    BatchPipelineResult,
)


# ============================================================================
# AlignedPCA Tests
# ============================================================================


class TestAlignedPCAConstants:
    """Tests for AlignedPCA constants and defaults."""

    def test_default_reference_epoch_is_186(self) -> None:
        """Verify that epoch 186 is the default reference per PRD v1.1."""
        assert DEFAULT_REFERENCE_EPOCH == 186

    def test_default_n_components(self) -> None:
        """Test default number of PCA components."""
        assert DEFAULT_N_COMPONENTS == 50

    def test_variance_explained_threshold(self) -> None:
        """Test variance explained threshold for quality assessment."""
        assert VARIANCE_EXPLAINED_THRESHOLD == 0.80


class TestAlignedPCA:
    """Tests for the AlignedPCA class."""

    def test_default_initialization(self) -> None:
        """Test AlignedPCA initializes with epoch 186 as reference."""
        aligner = AlignedPCA()
        assert aligner.reference_epoch == 186
        assert aligner.n_components == DEFAULT_N_COMPONENTS
        assert aligner.reference_pca is None
        assert aligner.fit_result is None

    def test_custom_initialization(self) -> None:
        """Test AlignedPCA with custom parameters."""
        aligner = AlignedPCA(n_components=30, reference_epoch=100)
        assert aligner.reference_epoch == 100
        assert aligner.n_components == 30

    @pytest.mark.gpu
    def test_fit_reference_basic(self) -> None:
        """Test basic fit_reference functionality."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=10, reference_epoch=186)
        embeddings = np.random.randn(100, 128)

        fit_result = aligner.fit_reference(embeddings, epoch=186)

        assert isinstance(fit_result, AlignedPCAFitResult)
        assert fit_result.reference_epoch == 186
        assert fit_result.n_components == 10
        assert fit_result.components.shape == (10, 128)
        assert fit_result.mean.shape == (128,)
        assert len(fit_result.variance_explained) == 10
        assert 0 <= fit_result.total_variance_explained <= 1.0

    @pytest.mark.gpu
    def test_fit_reference_variance_explained_threshold(self) -> None:
        """Test that variance explained exceeds threshold for random data."""
        np.random.seed(42)
        # Generate data with strong principal components
        n_samples = 200
        n_features = 50
        # Create low-rank structure
        base = np.random.randn(n_samples, 10)
        projection = np.random.randn(10, n_features)
        embeddings = base @ projection + np.random.randn(n_samples, n_features) * 0.1

        aligner = AlignedPCA(n_components=10, reference_epoch=186)
        fit_result = aligner.fit_reference(embeddings, epoch=186)

        # With low-rank data, should capture >80% variance
        assert fit_result.total_variance_explained >= VARIANCE_EXPLAINED_THRESHOLD

    def test_fit_reference_warns_on_wrong_epoch(self) -> None:
        """Test warning when fitting on non-reference epoch."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=5, reference_epoch=186)
        embeddings = np.random.randn(50, 64)

        with pytest.warns(UserWarning, match="but reference_epoch is 186"):
            aligner.fit_reference(embeddings, epoch=100)

    @pytest.mark.gpu
    def test_transform_basic(self) -> None:
        """Test basic transform functionality."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=10, reference_epoch=186)

        ref_embeddings = np.random.randn(100, 64)
        aligner.fit_reference(ref_embeddings, epoch=186)

        other_embeddings = np.random.randn(80, 64)
        result = aligner.transform(other_embeddings, epoch=100)

        assert isinstance(result, AlignedPCAResult)
        assert result.epoch == 100
        assert result.transformed_embeddings.shape == (80, 10)
        assert result.n_samples == 80
        assert result.n_components == 10
        assert 0 <= result.total_variance_explained <= 1.0
        assert result.reconstruction_error >= 0

    def test_transform_without_fit_raises(self) -> None:
        """Test that transform without fit raises RuntimeError."""
        aligner = AlignedPCA()
        embeddings = np.random.randn(50, 64)

        with pytest.raises(RuntimeError, match="Must call fit_reference"):
            aligner.transform(embeddings)

    def test_transform_feature_mismatch_raises(self) -> None:
        """Test that transform with wrong feature dimension raises ValueError."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=5)
        aligner.fit_reference(np.random.randn(50, 64), epoch=186)

        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            aligner.transform(np.random.randn(30, 128))  # Wrong feature dim

    @pytest.mark.gpu
    def test_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=10, reference_epoch=186)
        embeddings = np.random.randn(100, 64)

        result = aligner.fit_transform(embeddings)

        assert isinstance(result, AlignedPCAResult)
        assert result.epoch == 186
        assert aligner.is_fitted()

    def test_is_fitted(self) -> None:
        """Test is_fitted method."""
        aligner = AlignedPCA()
        assert not aligner.is_fitted()

        np.random.seed(42)
        aligner.fit_reference(np.random.randn(50, 64))
        assert aligner.is_fitted()

    def test_get_components(self) -> None:
        """Test get_components method."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=5)
        embeddings = np.random.randn(50, 64)
        aligner.fit_reference(embeddings)

        components = aligner.get_components()
        assert components.shape == (5, 64)

    def test_get_components_not_fitted_raises(self) -> None:
        """Test get_components without fit raises RuntimeError."""
        aligner = AlignedPCA()

        with pytest.raises(RuntimeError, match="Must call fit_reference"):
            aligner.get_components()


class TestAlignedPCAResult:
    """Tests for AlignedPCAResult dataclass."""

    def test_quality_assessment_excellent(self) -> None:
        """Test quality assessment for excellent variance explained."""
        result = AlignedPCAResult(
            transformed_embeddings=np.random.randn(50, 10),
            epoch=100,
            variance_explained=np.array([0.5, 0.3, 0.1, 0.05, 0.05]),
            total_variance_explained=0.95,
            reconstruction_error=0.01,
        )
        assert result.quality_assessment == "excellent"

    def test_quality_assessment_good(self) -> None:
        """Test quality assessment for good variance explained."""
        result = AlignedPCAResult(
            transformed_embeddings=np.random.randn(50, 10),
            epoch=100,
            variance_explained=np.array([0.4, 0.25, 0.1, 0.1]),
            total_variance_explained=0.85,
            reconstruction_error=0.05,
        )
        assert result.quality_assessment == "good"

    def test_quality_assessment_moderate(self) -> None:
        """Test quality assessment for moderate variance explained."""
        result = AlignedPCAResult(
            transformed_embeddings=np.random.randn(50, 10),
            epoch=100,
            variance_explained=np.array([0.3, 0.2, 0.1]),
            total_variance_explained=0.65,
            reconstruction_error=0.1,
        )
        assert result.quality_assessment == "moderate"

    def test_quality_assessment_poor(self) -> None:
        """Test quality assessment for poor variance explained."""
        result = AlignedPCAResult(
            transformed_embeddings=np.random.randn(50, 10),
            epoch=100,
            variance_explained=np.array([0.2, 0.15, 0.1]),
            total_variance_explained=0.45,
            reconstruction_error=0.2,
        )
        assert result.quality_assessment == "poor"


class TestAlignedPCAFitResult:
    """Tests for AlignedPCAFitResult dataclass."""

    def test_effective_dimensionality(self) -> None:
        """Test effective dimensionality calculation."""
        # Create variance explained that sums to >95% at 3 components
        variance_explained = np.array([0.5, 0.3, 0.16, 0.02, 0.01, 0.01])
        result = AlignedPCAFitResult(
            reference_epoch=186,
            n_components=6,
            variance_explained=variance_explained,
            total_variance_explained=float(np.sum(variance_explained)),
            components=np.random.randn(6, 64),
            mean=np.random.randn(64),
            singular_values=np.array([10.0, 7.0, 5.0, 2.0, 1.0, 0.5]),
        )
        assert result.effective_dimensionality == 3

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = AlignedPCAFitResult(
            reference_epoch=186,
            n_components=5,
            variance_explained=np.array([0.5, 0.3, 0.15, 0.03, 0.02]),
            total_variance_explained=1.0,
            components=np.random.randn(5, 64),
            mean=np.random.randn(64),
            singular_values=np.array([10.0, 7.0, 5.0, 2.0, 1.0]),
        )
        d = result.to_dict()
        assert d["reference_epoch"] == 186
        assert d["n_components"] == 5
        assert "total_variance_explained" in d


class TestCrossEpochTrajectory:
    """Tests for CrossEpochTrajectory dataclass."""

    def test_trajectory_properties(self) -> None:
        """Test trajectory property accessors."""
        np.random.seed(42)
        trajectories = np.random.randn(4, 100, 10)  # 4 epochs, 100 samples, 10 components
        trajectory = CrossEpochTrajectory(
            epochs=[0, 50, 100, 186],
            trajectories=trajectories,
            sample_indices=list(range(100)),
            variance_explained_per_epoch=[0.9, 0.85, 0.88, 0.95],
        )

        assert trajectory.n_epochs == 4
        assert trajectory.n_samples == 100
        assert trajectory.n_components == 10

    def test_get_epoch_embeddings(self) -> None:
        """Test getting embeddings for specific epoch."""
        np.random.seed(42)
        trajectories = np.random.randn(3, 50, 10)
        trajectory = CrossEpochTrajectory(
            epochs=[0, 100, 186],
            trajectories=trajectories,
            sample_indices=list(range(50)),
            variance_explained_per_epoch=[0.8, 0.85, 0.9],
        )

        epoch_100_emb = trajectory.get_epoch_embeddings(100)
        assert epoch_100_emb.shape == (50, 10)
        np.testing.assert_array_equal(epoch_100_emb, trajectories[1])

    def test_compute_displacement(self) -> None:
        """Test displacement computation between epochs."""
        np.random.seed(42)
        trajectories = np.random.randn(2, 50, 10)
        trajectory = CrossEpochTrajectory(
            epochs=[0, 186],
            trajectories=trajectories,
            sample_indices=list(range(50)),
            variance_explained_per_epoch=[0.8, 0.9],
        )

        displacement = trajectory.compute_displacement(0, 186)
        expected = trajectories[1] - trajectories[0]
        np.testing.assert_array_almost_equal(displacement, expected)

    def test_compute_total_path_length(self) -> None:
        """Test total path length computation."""
        # Create trajectory with known path lengths
        np.random.seed(42)
        trajectories = np.zeros((3, 2, 2))  # 3 epochs, 2 samples, 2 dims
        trajectories[0] = [[0, 0], [0, 0]]
        trajectories[1] = [[1, 0], [0, 1]]  # Distance 1 from epoch 0
        trajectories[2] = [[2, 0], [0, 2]]  # Distance 1 from epoch 1

        trajectory = CrossEpochTrajectory(
            epochs=[0, 50, 100],
            trajectories=trajectories,
            sample_indices=[0, 1],
            variance_explained_per_epoch=[0.9, 0.9, 0.9],
        )

        path_lengths = trajectory.compute_total_path_length()
        np.testing.assert_array_almost_equal(path_lengths, [2.0, 2.0])


class TestBuildCrossEpochTrajectory:
    """Tests for build_cross_epoch_trajectory function."""

    @pytest.mark.gpu
    def test_build_trajectory_basic(self) -> None:
        """Test basic trajectory building."""
        np.random.seed(42)
        aligner = AlignedPCA(n_components=10, reference_epoch=186)

        # Create embeddings for multiple epochs
        n_samples = 50
        n_features = 64
        epoch_186 = np.random.randn(n_samples, n_features)
        epoch_100 = epoch_186 + np.random.randn(n_samples, n_features) * 0.3
        epoch_50 = epoch_186 + np.random.randn(n_samples, n_features) * 0.5

        aligner.fit_reference(epoch_186, epoch=186)

        embeddings_by_epoch = {50: epoch_50, 100: epoch_100, 186: epoch_186}
        trajectory = build_cross_epoch_trajectory(aligner, embeddings_by_epoch)

        assert isinstance(trajectory, CrossEpochTrajectory)
        assert trajectory.epochs == [50, 100, 186]  # Sorted
        assert trajectory.n_epochs == 3
        assert trajectory.n_samples == 50

    def test_build_trajectory_not_fitted_raises(self) -> None:
        """Test that trajectory building without fitted aligner raises."""
        aligner = AlignedPCA()
        embeddings_by_epoch = {0: np.random.randn(50, 64)}

        with pytest.raises(RuntimeError, match="must be fitted"):
            build_cross_epoch_trajectory(aligner, embeddings_by_epoch)


class TestComputeConvergenceCurve:
    """Tests for compute_convergence_curve function."""

    def test_convergence_curve_basic(self) -> None:
        """Test basic convergence curve computation."""
        np.random.seed(42)
        # Create trajectory converging to final epoch
        trajectories = np.zeros((4, 10, 5))
        trajectories[0] = np.random.randn(10, 5)  # Random start
        trajectories[1] = trajectories[0] * 0.7  # Closer
        trajectories[2] = trajectories[0] * 0.3  # Even closer
        trajectories[3] = np.zeros((10, 5))  # Final position

        trajectory = CrossEpochTrajectory(
            epochs=[0, 50, 100, 186],
            trajectories=trajectories,
            sample_indices=list(range(10)),
            variance_explained_per_epoch=[0.9, 0.9, 0.9, 0.95],
        )

        epochs, mean_distances = compute_convergence_curve(trajectory)

        assert epochs == [0, 50, 100, 186]
        assert len(mean_distances) == 4
        # Final epoch should have zero distance
        assert mean_distances[-1] == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Memory Tracing Tests
# ============================================================================


class TestMemoryTracingConstants:
    """Tests for Memory Tracing constants."""

    def test_sparsity_threshold(self) -> None:
        """Test sparsity threshold value."""
        assert SPARSITY_THRESHOLD == 1e-6

    def test_rank_threshold(self) -> None:
        """Test rank threshold value."""
        assert RANK_THRESHOLD == 1e-4

    def test_top_singular_values(self) -> None:
        """Test number of top singular values tracked."""
        assert TOP_SINGULAR_VALUES == 10

    def test_min_epochs_for_trend(self) -> None:
        """Test minimum epochs required for trend analysis."""
        assert MIN_EPOCHS_FOR_TREND == 3


class TestComputeMatrixStats:
    """Tests for compute_matrix_stats function."""

    def test_basic_2d_matrix(self) -> None:
        """Test stats computation for 2D matrix."""
        np.random.seed(42)
        matrix = np.random.randn(100, 64)
        stats = compute_matrix_stats(matrix)

        assert "magnitude_mean" in stats
        assert "magnitude_std" in stats
        assert "frobenius_norm" in stats
        assert "sparsity" in stats
        assert "effective_rank" in stats
        assert "top_singular_values" in stats
        assert "shape" in stats

        assert stats["shape"] == (100, 64)
        assert stats["magnitude_mean"] > 0
        assert 0 <= stats["sparsity"] <= 1

    def test_basic_3d_matrix(self) -> None:
        """Test stats computation for 3D matrix (batch, slots, dim)."""
        np.random.seed(42)
        matrix = np.random.randn(8, 100, 64)  # batch=8, slots=100, dim=64
        stats = compute_matrix_stats(matrix)

        assert stats["shape"] == (8, 100, 64)
        assert stats["magnitude_mean"] > 0

    def test_sparse_matrix(self) -> None:
        """Test stats computation for sparse matrix."""
        np.random.seed(42)
        matrix = np.random.randn(100, 64)
        # Make 90% of entries near-zero
        mask = np.random.rand(100, 64) < 0.9
        matrix[mask] = 0

        stats = compute_matrix_stats(matrix)

        assert stats["sparsity"] > 0.8  # Should be high sparsity


class TestLayerMemoryStats:
    """Tests for LayerMemoryStats dataclass."""

    def test_total_sparsity(self) -> None:
        """Test total sparsity calculation."""
        stats = LayerMemoryStats(
            layer_idx=0,
            m_magnitude_mean=1.0,
            m_magnitude_std=0.1,
            m_frobenius_norm=10.0,
            m_sparsity=0.3,
            m_effective_rank=50,
            m_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
            s_magnitude_mean=0.5,
            s_magnitude_std=0.05,
            s_frobenius_norm=5.0,
            s_sparsity=0.5,
            s_effective_rank=30,
            s_top_singular_values=np.ones(TOP_SINGULAR_VALUES) * 0.5,
            m_shape=(8, 100, 64),
            s_shape=(8, 100, 64),
        )
        assert stats.total_sparsity == pytest.approx(0.4)

    def test_memory_utilization(self) -> None:
        """Test memory utilization calculation."""
        stats = LayerMemoryStats(
            layer_idx=0,
            m_magnitude_mean=1.0,
            m_magnitude_std=0.1,
            m_frobenius_norm=10.0,
            m_sparsity=0.3,
            m_effective_rank=50,
            m_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
            s_magnitude_mean=0.5,
            s_magnitude_std=0.05,
            s_frobenius_norm=5.0,
            s_sparsity=0.5,
            s_effective_rank=30,
            s_top_singular_values=np.ones(TOP_SINGULAR_VALUES) * 0.5,
            m_shape=(8, 100, 64),
            s_shape=(8, 100, 64),
        )
        assert stats.memory_utilization == pytest.approx(0.6)

    def test_health_status_healthy(self) -> None:
        """Test health status for healthy memory."""
        stats = LayerMemoryStats(
            layer_idx=0,
            m_magnitude_mean=1.0,
            m_magnitude_std=0.1,
            m_frobenius_norm=10.0,
            m_sparsity=0.3,
            m_effective_rank=50,
            m_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
            s_magnitude_mean=0.5,
            s_magnitude_std=0.05,
            s_frobenius_norm=5.0,
            s_sparsity=0.3,
            s_effective_rank=30,
            s_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
            m_shape=(8, 100, 64),
            s_shape=(8, 100, 64),
        )
        assert stats.health_status == "healthy"

    def test_health_status_sparse(self) -> None:
        """Test health status for sparse memory."""
        stats = LayerMemoryStats(
            layer_idx=0,
            m_magnitude_mean=0.1,
            m_magnitude_std=0.01,
            m_frobenius_norm=1.0,
            m_sparsity=0.95,  # Very sparse
            m_effective_rank=50,
            m_top_singular_values=np.ones(TOP_SINGULAR_VALUES) * 0.1,
            s_magnitude_mean=0.05,
            s_magnitude_std=0.005,
            s_frobenius_norm=0.5,
            s_sparsity=0.95,  # Very sparse
            s_effective_rank=30,
            s_top_singular_values=np.ones(TOP_SINGULAR_VALUES) * 0.1,
            m_shape=(8, 100, 64),
            s_shape=(8, 100, 64),
        )
        assert stats.health_status == "sparse"

    def test_health_status_degenerate(self) -> None:
        """Test health status for degenerate memory (low rank)."""
        stats = LayerMemoryStats(
            layer_idx=0,
            m_magnitude_mean=1.0,
            m_magnitude_std=0.1,
            m_frobenius_norm=10.0,
            m_sparsity=0.3,
            m_effective_rank=2,  # Very low rank
            m_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
            s_magnitude_mean=0.5,
            s_magnitude_std=0.05,
            s_frobenius_norm=5.0,
            s_sparsity=0.3,
            s_effective_rank=2,  # Very low rank
            s_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
            m_shape=(8, 100, 64),
            s_shape=(8, 100, 64),
        )
        assert stats.health_status == "degenerate"


class TestAnalyzeLayerMemory:
    """Tests for analyze_layer_memory function."""

    @pytest.mark.gpu
    def test_basic_layer_analysis(self) -> None:
        """Test basic layer memory analysis."""
        np.random.seed(42)
        m_matrix = np.random.randn(8, 100, 64)
        s_matrix = np.random.randn(8, 100, 64) * 0.5

        stats = analyze_layer_memory(0, m_matrix, s_matrix)

        assert isinstance(stats, LayerMemoryStats)
        assert stats.layer_idx == 0
        assert stats.m_shape == (8, 100, 64)
        assert stats.s_shape == (8, 100, 64)
        assert stats.m_magnitude_mean > 0
        assert stats.m_effective_rank > 0


class TestAnalyzeMemoryStates:
    """Tests for analyze_memory_states function."""

    @pytest.mark.gpu
    def test_dict_format_memory_states(self) -> None:
        """Test analysis with dict format memory states."""
        np.random.seed(42)
        memory_states = [
            {"M": np.random.randn(8, 100, 64), "S": np.random.randn(8, 100, 64) * 0.5}
            for _ in range(4)
        ]

        result = analyze_memory_states(memory_states, epoch=50, step=2500)

        assert isinstance(result, MemoryEpisodeStats)
        assert result.epoch == 50
        assert result.step == 2500
        assert result.num_layers == 4
        assert len(result.layer_stats) == 4

    @pytest.mark.gpu
    def test_tuple_format_memory_states(self) -> None:
        """Test analysis with tuple format memory states."""
        np.random.seed(42)
        memory_states = [
            (np.random.randn(8, 100, 64), np.random.randn(8, 100, 64) * 0.5)
            for _ in range(4)
        ]

        result = analyze_memory_states(memory_states, epoch=100, step=5000)

        assert isinstance(result, MemoryEpisodeStats)
        assert result.num_layers == 4


class TestMemoryEpisodeStats:
    """Tests for MemoryEpisodeStats dataclass."""

    def test_mean_properties(self) -> None:
        """Test mean property calculations."""
        layer_stats = [
            LayerMemoryStats(
                layer_idx=i,
                m_magnitude_mean=1.0 + i * 0.1,
                m_magnitude_std=0.1,
                m_frobenius_norm=10.0,
                m_sparsity=0.2 + i * 0.05,
                m_effective_rank=50 - i * 5,
                m_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                s_magnitude_mean=0.5 + i * 0.05,
                s_magnitude_std=0.05,
                s_frobenius_norm=5.0,
                s_sparsity=0.25 + i * 0.05,
                s_effective_rank=30 - i * 3,
                s_top_singular_values=np.ones(TOP_SINGULAR_VALUES) * 0.5,
                m_shape=(8, 100, 64),
                s_shape=(8, 100, 64),
            )
            for i in range(4)
        ]

        stats = MemoryEpisodeStats(
            epoch=100,
            step=5000,
            layer_stats=layer_stats,
            num_layers=4,
            total_m_memory_bytes=1000000,
            total_s_memory_bytes=1000000,
            analysis_time_seconds=1.5,
        )

        assert stats.mean_m_sparsity == pytest.approx(0.275, abs=0.01)
        assert stats.mean_s_sparsity == pytest.approx(0.325, abs=0.01)

    def test_overall_health_healthy(self) -> None:
        """Test overall health assessment for healthy layers."""
        layer_stats = [
            LayerMemoryStats(
                layer_idx=i,
                m_magnitude_mean=1.0,
                m_magnitude_std=0.1,
                m_frobenius_norm=10.0,
                m_sparsity=0.3,
                m_effective_rank=50,
                m_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                s_magnitude_mean=0.5,
                s_magnitude_std=0.05,
                s_frobenius_norm=5.0,
                s_sparsity=0.3,
                s_effective_rank=30,
                s_top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                m_shape=(8, 100, 64),
                s_shape=(8, 100, 64),
            )
            for i in range(4)
        ]

        stats = MemoryEpisodeStats(
            epoch=100,
            step=5000,
            layer_stats=layer_stats,
            num_layers=4,
            total_m_memory_bytes=1000000,
            total_s_memory_bytes=1000000,
            analysis_time_seconds=1.5,
        )

        assert stats.overall_health == "healthy"


class TestMemoryEvolutionResult:
    """Tests for MemoryEvolutionResult dataclass."""

    def test_has_trend_property(self) -> None:
        """Test has_trend property."""
        result_with_trend = MemoryEvolutionResult(
            epochs=[0, 50, 100, 186],
            epoch_stats={},
            sparsity_trend=0.7,  # Strong trend
            rank_trend=0.3,
            magnitude_trend=0.2,
            outlier_epochs=[],
            total_analysis_time_seconds=10.0,
        )
        assert result_with_trend.has_trend is True

        result_no_trend = MemoryEvolutionResult(
            epochs=[0, 50, 100, 186],
            epoch_stats={},
            sparsity_trend=0.2,  # Weak trend
            rank_trend=0.3,
            magnitude_trend=0.1,
            outlier_epochs=[],
            total_analysis_time_seconds=10.0,
        )
        assert result_no_trend.has_trend is False


# ============================================================================
# Atlas Statistics Tests
# ============================================================================


class TestAtlasStatisticsConstants:
    """Tests for Atlas Statistics constants."""

    def test_outlier_z_threshold(self) -> None:
        """Test outlier z-score threshold."""
        assert OUTLIER_Z_THRESHOLD == 2.5

    def test_outlier_iqr_multiplier(self) -> None:
        """Test outlier IQR multiplier."""
        assert OUTLIER_IQR_MULTIPLIER == 1.5

    def test_min_epochs_for_stats(self) -> None:
        """Test minimum epochs for statistical analysis."""
        assert MIN_EPOCHS_FOR_STATS == 3


class TestTrendAnalysis:
    """Tests for TrendAnalysis dataclass."""

    def test_is_significant(self) -> None:
        """Test is_significant property."""
        significant_trend = TrendAnalysis(
            metric_name="sparsity",
            slope=0.01,
            intercept=0.3,
            r_squared=0.85,
            p_value=0.01,  # Significant
            correlation=0.92,
            direction="increasing",
        )
        assert significant_trend.is_significant is True

        non_significant = TrendAnalysis(
            metric_name="magnitude",
            slope=0.001,
            intercept=1.0,
            r_squared=0.1,
            p_value=0.15,  # Not significant
            correlation=0.3,
            direction="stable",
        )
        assert non_significant.is_significant is False


class TestOutlierDetectionResult:
    """Tests for OutlierDetectionResult dataclass."""

    def test_has_outliers_true(self) -> None:
        """Test has_outliers when outliers exist."""
        result = OutlierDetectionResult(
            outlier_epochs=[50, 100],
            outlier_metrics={50: ["magnitude"], 100: ["sparsity"]},
            z_scores={0: {"magnitude": 0.5}, 50: {"magnitude": 3.0}},
            iqr_outliers=[50],
            z_outliers=[100],
        )
        assert result.has_outliers is True

    def test_has_outliers_false(self) -> None:
        """Test has_outliers when no outliers."""
        result = OutlierDetectionResult(
            outlier_epochs=[],
            outlier_metrics={},
            z_scores={0: {"magnitude": 0.5}},
            iqr_outliers=[],
            z_outliers=[],
        )
        assert result.has_outliers is False


class TestComputeEpochStatistics:
    """Tests for compute_epoch_statistics function."""

    @pytest.mark.gpu
    def test_basic_epoch_statistics(self) -> None:
        """Test basic epoch statistics computation."""
        np.random.seed(42)
        memory_states = [
            {"M": np.random.randn(8, 100, 64), "S": np.random.randn(8, 100, 64) * 0.5}
            for _ in range(4)
        ]

        stats = compute_epoch_statistics(memory_states, epoch=100, step=5000)

        assert isinstance(stats, EpochStatistics)
        assert stats.epoch == 100
        assert stats.step == 5000
        assert stats.num_layers == 4
        assert stats.magnitude_mean > 0
        assert 0 <= stats.sparsity <= 1


class TestAnalyzeTrends:
    """Tests for analyze_trends function."""

    def test_increasing_trend_detection(self) -> None:
        """Test detection of increasing trend."""
        epoch_stats = {
            epoch: EpochStatistics(
                epoch=epoch,
                step=epoch * 50,
                magnitude_mean=1.0 - epoch / 500,  # Decreasing
                magnitude_std=0.1,
                magnitude_min=0.5,
                magnitude_max=1.5,
                sparsity=0.3 + epoch / 500,  # Increasing
                m_sparsity=0.3 + epoch / 500,
                s_sparsity=0.3 + epoch / 500,
                rank=50 - epoch / 10,  # Decreasing
                m_rank=50 - epoch / 10,
                s_rank=50 - epoch / 10,
                frobenius_norm=10.0,
                top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                num_layers=4,
                analysis_time_seconds=0.1,
            )
            for epoch in [0, 50, 100, 150]
        }

        trends = analyze_trends(epoch_stats)

        assert "sparsity" in trends
        assert trends["sparsity"].direction == "increasing"
        assert trends["magnitude_mean"].direction == "decreasing"

    def test_stable_trend_detection(self) -> None:
        """Test detection of stable (no) trend."""
        epoch_stats = {
            epoch: EpochStatistics(
                epoch=epoch,
                step=epoch * 50,
                magnitude_mean=1.0,  # Constant
                magnitude_std=0.1,
                magnitude_min=0.5,
                magnitude_max=1.5,
                sparsity=0.3,  # Constant
                m_sparsity=0.3,
                s_sparsity=0.3,
                rank=50,
                m_rank=50,
                s_rank=50,
                frobenius_norm=10.0,
                top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                num_layers=4,
                analysis_time_seconds=0.1,
            )
            for epoch in [0, 50, 100, 150]
        }

        trends = analyze_trends(epoch_stats)

        assert trends["sparsity"].direction == "stable"
        assert trends["magnitude_mean"].direction == "stable"


class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_no_outliers_uniform_data(self) -> None:
        """Test that uniform data has no outliers."""
        epoch_stats = {
            epoch: EpochStatistics(
                epoch=epoch,
                step=epoch * 50,
                magnitude_mean=1.0,
                magnitude_std=0.1,
                magnitude_min=0.5,
                magnitude_max=1.5,
                sparsity=0.3,
                m_sparsity=0.3,
                s_sparsity=0.3,
                rank=50,
                m_rank=50,
                s_rank=50,
                frobenius_norm=10.0,
                top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                num_layers=4,
                analysis_time_seconds=0.1,
            )
            for epoch in [0, 50, 100, 150]
        }

        result = detect_outliers(epoch_stats)

        assert result.has_outliers is False

    def test_outlier_detection_with_anomaly(self) -> None:
        """Test outlier detection with an anomalous epoch."""
        epoch_stats = {}
        for epoch in [0, 50, 100, 150]:
            # Normal values except epoch 100 has very different magnitude
            magnitude = 10.0 if epoch == 100 else 1.0

            epoch_stats[epoch] = EpochStatistics(
                epoch=epoch,
                step=epoch * 50,
                magnitude_mean=magnitude,
                magnitude_std=0.1,
                magnitude_min=0.5,
                magnitude_max=1.5,
                sparsity=0.3,
                m_sparsity=0.3,
                s_sparsity=0.3,
                rank=50,
                m_rank=50,
                s_rank=50,
                frobenius_norm=10.0,
                top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                num_layers=4,
                analysis_time_seconds=0.1,
            )

        result = detect_outliers(epoch_stats)

        assert result.has_outliers is True
        assert 100 in result.outlier_epochs


class TestComputeSummaryStatistics:
    """Tests for compute_summary_statistics function."""

    def test_summary_statistics_basic(self) -> None:
        """Test basic summary statistics computation."""
        epoch_stats = {
            epoch: EpochStatistics(
                epoch=epoch,
                step=epoch * 50,
                magnitude_mean=1.0 + epoch / 100,
                magnitude_std=0.1,
                magnitude_min=0.5,
                magnitude_max=1.5,
                sparsity=0.3 + epoch / 500,
                m_sparsity=0.3,
                s_sparsity=0.3,
                rank=50,
                m_rank=50,
                s_rank=50,
                frobenius_norm=10.0,
                top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                num_layers=4,
                analysis_time_seconds=0.1,
            )
            for epoch in [0, 50, 100, 150]
        }

        summary = compute_summary_statistics(epoch_stats)

        assert "n_epochs" in summary
        assert summary["n_epochs"] == 4
        assert "magnitude" in summary
        assert "mean" in summary["magnitude"]
        assert "std" in summary["magnitude"]


class TestAtlasStatisticsResult:
    """Tests for AtlasStatisticsResult dataclass."""

    def test_export_csv(self) -> None:
        """Test CSV export functionality."""
        epoch_stats = {
            epoch: EpochStatistics(
                epoch=epoch,
                step=epoch * 50,
                magnitude_mean=1.0,
                magnitude_std=0.1,
                magnitude_min=0.5,
                magnitude_max=1.5,
                sparsity=0.3,
                m_sparsity=0.3,
                s_sparsity=0.3,
                rank=50,
                m_rank=50,
                s_rank=50,
                frobenius_norm=10.0,
                top_singular_values=np.ones(TOP_SINGULAR_VALUES),
                num_layers=4,
                analysis_time_seconds=0.1,
            )
            for epoch in [0, 50, 100]
        }

        result = AtlasStatisticsResult(
            epochs=[0, 50, 100],
            epoch_statistics=epoch_stats,
            trends={},
            outliers=OutlierDetectionResult(
                outlier_epochs=[],
                outlier_metrics={},
                z_scores={},
                iqr_outliers=[],
                z_outliers=[],
            ),
            cluster_stability=[],
            summary_statistics={},
            total_analysis_time_seconds=1.0,
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            result.export_csv(output_path)
            assert output_path.exists()

            # Read and verify CSV
            import csv
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert "epoch" in rows[0]
            assert "magnitude_mean" in rows[0]
            assert "sparsity" in rows[0]
            assert "rank" in rows[0]
        finally:
            output_path.unlink(missing_ok=True)


# ============================================================================
# Visualization Tests
# ============================================================================


class TestVisualizationConstants:
    """Tests for Visualization constants."""

    def test_default_axis_range(self) -> None:
        """Test default axis range for 3D plots."""
        assert DEFAULT_AXIS_RANGE == (-10.0, 10.0)

    def test_default_png_dimensions(self) -> None:
        """Test default PNG dimensions."""
        assert DEFAULT_PNG_WIDTH == 1200
        assert DEFAULT_PNG_HEIGHT == 800

    def test_png_scale_factor(self) -> None:
        """Test PNG scale factor for 300 DPI."""
        assert PNG_SCALE_FACTOR == 2.5

    def test_max_html_size(self) -> None:
        """Test maximum HTML file size target."""
        assert MAX_HTML_SIZE_BYTES == 10 * 1024 * 1024  # 10MB


class TestVisualizationStyle:
    """Tests for VisualizationStyle configuration."""

    def test_default_values(self) -> None:
        """Test default style values."""
        style = VisualizationStyle()
        assert style.colorscale == "viridis"
        assert style.marker_size == 4
        assert style.marker_opacity == 0.8
        assert style.line_width == 2

    def test_custom_values(self) -> None:
        """Test custom style values."""
        style = VisualizationStyle(
            colorscale="plasma",
            marker_size=8,
            marker_opacity=0.6,
        )
        assert style.colorscale == "plasma"
        assert style.marker_size == 8
        assert style.marker_opacity == 0.6


class TestAnimationConfig:
    """Tests for AnimationConfig configuration."""

    def test_default_values(self) -> None:
        """Test default animation configuration."""
        config = AnimationConfig()
        assert config.duration_ms == 500
        assert config.redraw is True
        assert config.show_slider is True
        assert config.show_play_button is True


class TestExportConfig:
    """Tests for ExportConfig configuration."""

    def test_default_values(self) -> None:
        """Test default export configuration."""
        config = ExportConfig()
        assert config.html_path is None
        assert config.png_path is None
        assert config.png_width == DEFAULT_PNG_WIDTH
        assert config.png_height == DEFAULT_PNG_HEIGHT
        assert config.png_scale == PNG_SCALE_FACTOR


class TestAxis3DConfig:
    """Tests for Axis3DConfig configuration."""

    def test_default_range(self) -> None:
        """Test that axis config has fixed range."""
        config = Axis3DConfig()
        assert config.range == DEFAULT_AXIS_RANGE


class TestLandscape3DConfig:
    """Tests for Landscape3DConfig configuration."""

    def test_contains_axis_configs(self) -> None:
        """Test that landscape config contains axis configs."""
        config = Landscape3DConfig()
        assert hasattr(config, "x_axis")
        assert hasattr(config, "y_axis")
        assert hasattr(config, "z_axis")


# ============================================================================
# Batch Pipeline Tests
# ============================================================================


class TestBatchPipelineConstants:
    """Tests for Batch Pipeline constants."""

    def test_total_epochs(self) -> None:
        """Test total epochs constant."""
        assert TOTAL_EPOCHS == 186

    def test_gpu_memory_budget(self) -> None:
        """Test GPU memory budget in MB."""
        assert GPU_MEMORY_BUDGET_MB == 5000  # 5GB


class TestValidationSummary:
    """Tests for ValidationSummary dataclass."""

    def test_validation_summary_properties(self) -> None:
        """Test ValidationSummary properties."""
        summary = ValidationSummary(
            total_checkpoints=186,
            valid_checkpoints=186,
            invalid_checkpoints=0,
            validation_errors=[],
            checkpoint_paths=["/path/to/checkpoint.pt"],
        )

        assert summary.total_checkpoints == 186
        assert summary.valid_checkpoints == 186


class TestPipelineStageResult:
    """Tests for PipelineStageResult dataclass."""

    def test_stage_result_success(self) -> None:
        """Test successful stage result."""
        result = PipelineStageResult(
            stage_name="concept_landscape",
            success=True,
            duration_seconds=120.0,
            outputs={"html_path": "/tmp/output.html"},
            errors=[],
            metrics={"n_epochs": 4},
        )

        assert result.success is True
        assert result.stage_name == "concept_landscape"


class TestBatchPipelineResult:
    """Tests for BatchPipelineResult dataclass."""

    def test_pipeline_result_aggregation(self) -> None:
        """Test pipeline result aggregation."""
        result = BatchPipelineResult(
            stages={
                "validation": PipelineStageResult(
                    stage_name="validation",
                    success=True,
                    duration_seconds=10.0,
                    outputs={},
                    errors=[],
                    metrics={},
                ),
                "concept_landscape": PipelineStageResult(
                    stage_name="concept_landscape",
                    success=True,
                    duration_seconds=120.0,
                    outputs={},
                    errors=[],
                    metrics={},
                ),
            },
            total_duration_seconds=130.0,
            success=True,
            summary={},
        )

        assert result.success is True
        assert len(result.stages) == 2


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestAlignedPCAIntegration:
    """Integration tests for the AlignedPCA workflow."""

    @pytest.mark.gpu
    def test_full_pca_workflow(self) -> None:
        """Test full PCA workflow from fitting to trajectory building."""
        np.random.seed(42)

        # Simulate embedding evolution
        n_samples = 100
        n_features = 128

        # Final epoch embeddings (reference)
        epoch_186 = np.random.randn(n_samples, n_features)
        # Earlier epochs with increasing noise
        epoch_100 = epoch_186 + np.random.randn(n_samples, n_features) * 0.5
        epoch_50 = epoch_186 + np.random.randn(n_samples, n_features) * 1.0
        epoch_0 = np.random.randn(n_samples, n_features)

        # Create aligner with epoch 186 reference
        aligner = AlignedPCA(n_components=50, reference_epoch=186)

        # Fit on reference epoch
        fit_result = aligner.fit_reference(epoch_186, epoch=186)

        assert fit_result.reference_epoch == 186
        assert fit_result.total_variance_explained > 0.5  # Should capture significant variance

        # Transform other epochs
        result_100 = aligner.transform(epoch_100, epoch=100)
        result_50 = aligner.transform(epoch_50, epoch=50)
        result_0 = aligner.transform(epoch_0, epoch=0)

        # Variance explained should generally be highest for reference epoch
        # (though this depends on data structure)
        assert result_100.n_components == 50
        assert result_50.n_components == 50
        assert result_0.n_components == 50

        # Build trajectory
        embeddings_by_epoch = {
            0: epoch_0,
            50: epoch_50,
            100: epoch_100,
            186: epoch_186,
        }
        trajectory = build_cross_epoch_trajectory(aligner, embeddings_by_epoch)

        assert trajectory.n_epochs == 4
        assert trajectory.epochs == [0, 50, 100, 186]

        # Compute convergence
        epochs, distances = compute_convergence_curve(trajectory)
        assert len(epochs) == 4
        # Final epoch should have ~0 distance to itself
        assert distances[-1] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.integration
class TestMemoryTracingIntegration:
    """Integration tests for memory tracing workflow."""

    @pytest.mark.gpu
    def test_memory_evolution_synthetic(self) -> None:
        """Test memory evolution analysis with synthetic data."""
        np.random.seed(42)

        # Simulate memory states for multiple epochs
        epochs = [0, 50, 100, 186]
        epoch_stats = {}

        for epoch in epochs:
            n_layers = 4
            # Evolving characteristics
            base_sparsity = 0.3 + 0.3 * (epoch / 186)
            base_magnitude = 1.0 - 0.3 * (epoch / 186)

            memory_states = []
            for layer in range(n_layers):
                layer_sparsity = base_sparsity + 0.1 * layer
                m_matrix = np.random.randn(8, 100, 64) * base_magnitude
                m_matrix[np.random.rand(*m_matrix.shape) < layer_sparsity] = 0

                s_matrix = np.random.randn(8, 100, 64) * 0.5 * base_magnitude
                s_matrix[np.random.rand(*s_matrix.shape) < layer_sparsity] = 0

                memory_states.append({"M": m_matrix, "S": s_matrix})

            stats = analyze_memory_states(memory_states, epoch=epoch, step=epoch * 50)
            epoch_stats[epoch] = stats

        # Verify epoch statistics
        assert all(epoch_stats[e].num_layers == 4 for e in epochs)

        # Sparsity should increase over epochs
        sparsity_values = [epoch_stats[e].mean_m_sparsity for e in epochs]
        # Trend should be increasing (correlation with epoch should be positive)
        correlation = np.corrcoef(epochs, sparsity_values)[0, 1]
        assert correlation > 0.5


@pytest.mark.integration
class TestAtlasStatisticsIntegration:
    """Integration tests for Atlas statistics workflow."""

    def test_full_statistics_workflow(self) -> None:
        """Test full statistics workflow with synthetic data."""
        np.random.seed(42)

        # Create epoch statistics with known trends
        epochs = [0, 50, 100, 150]
        epoch_statistics = {}

        for epoch in epochs:
            # Create synthetic memory states
            memory_states = [
                {"M": np.random.randn(8, 100, 64), "S": np.random.randn(8, 100, 64) * 0.5}
                for _ in range(4)
            ]
            stats = compute_epoch_statistics(memory_states, epoch=epoch, step=epoch * 50)
            epoch_statistics[epoch] = stats

        # Analyze trends
        trends = analyze_trends(epoch_statistics)
        assert len(trends) > 0

        # Detect outliers
        outliers = detect_outliers(epoch_statistics)
        assert isinstance(outliers, OutlierDetectionResult)

        # Compute summary
        summary = compute_summary_statistics(epoch_statistics)
        assert summary["n_epochs"] == 4

        # Create full result
        result = AtlasStatisticsResult(
            epochs=epochs,
            epoch_statistics=epoch_statistics,
            trends=trends,
            outliers=outliers,
            cluster_stability=[],
            summary_statistics=summary,
            total_analysis_time_seconds=1.0,
        )

        assert result.n_epochs == 4

        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            result.export_csv(output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        finally:
            output_path.unlink(missing_ok=True)
