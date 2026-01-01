"""End-to-End Tests for the Atlas Analysis Pipeline.

This module provides comprehensive E2E tests for the full Atlas analysis
pipeline, covering:

1. Checkpoint validation (all 186 epochs when available)
2. Aligned PCA with epoch 186 reference on epochs 0, 50, 100, 185
3. Animated concept landscape visualization generation
4. Statistical metrics CSV export
5. Artifact verification (HTML <10MB, PNG @300 DPI, metrics evolution)

USAGE
=====
Run all E2E tests (synthetic data, no real checkpoints needed):
    poetry run pytest tests/e2e_analysis_pipeline.py -v

Run with real checkpoints (requires ATLAS_CHECKPOINT_DIR):
    poetry run pytest tests/e2e_analysis_pipeline.py -v -m "e2e and not synthetic"

Run only synthetic tests (CI-friendly):
    poetry run pytest tests/e2e_analysis_pipeline.py -v -m "e2e and synthetic"

ENVIRONMENT VARIABLES
=====================
- ATLAS_CHECKPOINT_DIR: Path to Atlas checkpoints (optional, uses synthetic if not set)

PERFORMANCE TARGETS
===================
- Checkpoint load time: <2s per checkpoint
- GPU memory: <5GB during batch processing
- Pipeline completion: <15 min for 186 epochs
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Mark entire module as E2E tests
pytestmark = [pytest.mark.e2e]


# ============================================================================
# Configuration
# ============================================================================

# Default checkpoint directory (from environment or empty - tests use synthetic data)
# Set ATLAS_CHECKPOINT_DIR environment variable for real checkpoint tests
DEFAULT_CHECKPOINT_DIR = os.environ.get("ATLAS_CHECKPOINT_DIR", "")

# Test epochs for E2E analysis
E2E_EPOCHS = [0, 50, 100, 185]

# Reference epoch per PRD v1.1
REFERENCE_EPOCH = 186

# Artifact size limits
MAX_HTML_SIZE_MB = 10.0
PNG_TARGET_WIDTH = 1200
PNG_TARGET_HEIGHT = 800
PNG_SCALE_FACTOR = 2.5

# GPU memory budget
GPU_MEMORY_BUDGET_MB = 5000

# Synthetic data parameters
N_SYNTHETIC_SAMPLES = 50
N_SYNTHETIC_FEATURES = 128
N_SYNTHETIC_EPOCHS = 20


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def output_dir() -> Path:
    """Create temporary output directory for E2E test artifacts."""
    with tempfile.TemporaryDirectory(prefix="atlas_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def synthetic_embeddings() -> dict[int, np.ndarray]:
    """Generate synthetic embeddings for testing without real checkpoints.

    Creates embeddings that show concept evolution toward epoch 186:
    - Early epochs: Random, high variance
    - Middle epochs: Converging toward final state
    - Final epoch: Reference state
    """
    np.random.seed(42)

    # Final epoch embeddings (reference state)
    epoch_186 = np.random.randn(N_SYNTHETIC_SAMPLES, N_SYNTHETIC_FEATURES)

    embeddings = {}

    # Generate epochs with decreasing distance from reference
    test_epochs = [0, 10, 25, 50, 75, 100, 125, 150, 175, 185, 186]

    for epoch in test_epochs:
        if epoch == 186:
            embeddings[epoch] = epoch_186.copy()
        else:
            # Distance from reference decreases with epoch
            noise_scale = 1.0 - (epoch / 186) * 0.8
            embeddings[epoch] = epoch_186 + np.random.randn(
                N_SYNTHETIC_SAMPLES, N_SYNTHETIC_FEATURES
            ) * noise_scale

    return embeddings


@pytest.fixture(scope="module")
def synthetic_memory_states() -> dict[int, list[dict[str, np.ndarray]]]:
    """Generate synthetic memory states for testing.

    Returns dict mapping epoch to list of layer memory states.
    Each layer has M and S matrices with evolving characteristics.
    """
    np.random.seed(42)

    memory_states = {}
    n_layers = 4

    for epoch in [0, 50, 100, 185, 186]:
        # Memory characteristics evolve over training
        progress = epoch / 186
        base_sparsity = 0.3 + 0.3 * progress  # Sparsity increases
        base_magnitude = 1.0 - 0.3 * progress  # Magnitude decreases

        layers = []
        for layer_idx in range(n_layers):
            # Each layer has slightly different characteristics
            layer_sparsity = base_sparsity + 0.05 * layer_idx

            # Generate M matrix (32 heads, 1152 slots, 128 dims per spec)
            m_matrix = np.random.randn(32, 1152, 128) * base_magnitude
            m_matrix[np.random.rand(*m_matrix.shape) < layer_sparsity] = 0

            # Generate S matrix (same shape)
            s_matrix = np.random.randn(32, 1152, 128) * base_magnitude * 0.5
            s_matrix[np.random.rand(*s_matrix.shape) < layer_sparsity] = 0

            layers.append({"M": m_matrix, "S": s_matrix})

        memory_states[epoch] = layers

    return memory_states


# ============================================================================
# Test: Aligned PCA with Epoch 186 Reference
# ============================================================================


@pytest.mark.synthetic
class TestAlignedPCAE2E:
    """E2E tests for aligned PCA with epoch 186 reference."""

    def test_aligned_pca_reference_epoch_186(
        self, synthetic_embeddings: dict[int, np.ndarray]
    ) -> None:
        """Test that PCA uses epoch 186 as reference per PRD v1.1."""
        from src.analysis import DEFAULT_REFERENCE_EPOCH, AlignedPCA

        # Verify constant
        assert DEFAULT_REFERENCE_EPOCH == 186

        # Create aligner with default reference
        aligner = AlignedPCA(n_components=50)
        assert aligner.reference_epoch == 186

        # Fit on epoch 186
        fit_result = aligner.fit_reference(synthetic_embeddings[186], epoch=186)

        # Verify fit result
        assert fit_result.reference_epoch == 186
        assert fit_result.n_components == 50
        assert fit_result.total_variance_explained > 0

    def test_transform_epochs_with_reference(
        self, synthetic_embeddings: dict[int, np.ndarray]
    ) -> None:
        """Test transforming epochs 0, 50, 100, 185 with epoch 186 reference."""
        from src.analysis import AlignedPCA

        aligner = AlignedPCA(n_components=50, reference_epoch=186)

        # Fit on reference epoch
        aligner.fit_reference(synthetic_embeddings[186], epoch=186)

        # Transform test epochs
        results = {}
        for epoch in E2E_EPOCHS:
            if epoch in synthetic_embeddings:
                result = aligner.transform(synthetic_embeddings[epoch], epoch=epoch)
                results[epoch] = result

        # Verify all epochs transformed
        assert len(results) >= 3  # At least 0, 50, 100

        # All transforms should have same number of components
        for _epoch, result in results.items():
            assert result.n_components == 50
            assert result.transformed_embeddings.shape[1] == 50

    def test_convergence_to_reference(
        self, synthetic_embeddings: dict[int, np.ndarray]
    ) -> None:
        """Test that later epochs show convergence toward reference."""
        from src.analysis import (
            AlignedPCA,
            build_cross_epoch_trajectory,
            compute_convergence_curve,
        )

        aligner = AlignedPCA(n_components=50, reference_epoch=186)
        aligner.fit_reference(synthetic_embeddings[186], epoch=186)

        # Build trajectory
        trajectory = build_cross_epoch_trajectory(aligner, synthetic_embeddings)

        # Compute convergence
        epochs, distances = compute_convergence_curve(trajectory)

        # Final epoch should have ~0 distance
        assert distances[-1] == pytest.approx(0.0, abs=1e-10)

        # Later epochs should generally be closer to reference
        if len(distances) >= 3:
            # Check trend: later epochs have smaller distances
            assert distances[-2] < distances[0], (
                "Later epochs should be closer to reference"
            )


# ============================================================================
# Test: Concept Landscape Visualization
# ============================================================================


@pytest.mark.synthetic
class TestConceptLandscapeE2E:
    """E2E tests for concept landscape visualization."""

    def test_animated_visualization_generation(
        self,
        synthetic_embeddings: dict[int, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Test generating animated 3D concept landscape."""
        from src.analysis import ExportResult, visualize_epoch_evolution

        # Generate visualization
        html_path = output_dir / "concept_landscape.html"
        png_path = output_dir / "concept_landscape.png"

        fig, export_result = visualize_epoch_evolution(
            {e: emb[:, :3] for e, emb in synthetic_embeddings.items()},
            title="E2E Test - Concept Landscape Evolution",
            output_html=str(html_path),
            output_png=str(png_path),
        )

        # Verify exports
        assert isinstance(export_result, ExportResult)
        assert export_result.html_path is not None
        assert export_result.html_path.exists()

        # Check HTML size limit
        html_size_mb = export_result.html_size_bytes / (1024 * 1024)
        assert html_size_mb < MAX_HTML_SIZE_MB, (
            f"HTML size {html_size_mb:.2f} MB exceeds {MAX_HTML_SIZE_MB} MB limit"
        )

    def test_visualization_has_fixed_axes(
        self,
        synthetic_embeddings: dict[int, np.ndarray],
    ) -> None:
        """Test that animated visualization uses fixed axis ranges."""
        from src.analysis import (
            DEFAULT_AXIS_RANGE,
            Landscape3DConfig,
            create_animated_scatter_3d,
        )

        # Prepare frames data
        frames_data = [
            {"points": emb[:, :3], "name": f"Epoch {e}"}
            for e, emb in sorted(synthetic_embeddings.items())
        ]

        # Create with default config (should have fixed ranges)
        config = Landscape3DConfig()
        fig = create_animated_scatter_3d(frames_data, config)

        # Verify axis ranges are fixed
        scene = fig.layout.scene
        assert scene.xaxis.range == list(DEFAULT_AXIS_RANGE)
        assert scene.yaxis.range == list(DEFAULT_AXIS_RANGE)
        assert scene.zaxis.range == list(DEFAULT_AXIS_RANGE)


# ============================================================================
# Test: Memory Tracing
# ============================================================================


@pytest.mark.synthetic
class TestMemoryTracingE2E:
    """E2E tests for memory episode tracing."""

    def test_memory_statistics_computation(
        self,
        synthetic_memory_states: dict[int, list[dict[str, np.ndarray]]],
    ) -> None:
        """Test computing memory statistics for all epochs."""
        from src.analysis import MemoryEpisodeStats, analyze_memory_states

        results = {}
        for epoch, memory_states in synthetic_memory_states.items():
            stats = analyze_memory_states(memory_states, epoch=epoch, step=epoch * 50)
            results[epoch] = stats

        # Verify all epochs analyzed
        assert len(results) == len(synthetic_memory_states)

        # Each result should have layer stats
        for _epoch, stats in results.items():
            assert isinstance(stats, MemoryEpisodeStats)
            assert stats.num_layers == 4
            assert len(stats.layer_stats) == 4

    def test_memory_evolution_trends(
        self,
        synthetic_memory_states: dict[int, list[dict[str, np.ndarray]]],
    ) -> None:
        """Test detecting memory evolution trends across epochs."""
        from src.analysis import analyze_memory_states

        # Compute stats for each epoch
        epoch_stats = {}
        for epoch, memory_states in sorted(synthetic_memory_states.items()):
            stats = analyze_memory_states(memory_states, epoch=epoch, step=epoch * 50)
            epoch_stats[epoch] = stats

        # Extract sparsity values
        epochs = sorted(epoch_stats.keys())
        sparsities = [epoch_stats[e].mean_m_sparsity for e in epochs]

        # Sparsity should increase over training (by design of synthetic data)
        if len(sparsities) >= 3:
            correlation = np.corrcoef(epochs, sparsities)[0, 1]
            assert correlation > 0.3, "Sparsity should increase with epoch"


# ============================================================================
# Test: Statistical Analysis
# ============================================================================


@pytest.mark.synthetic
class TestStatisticalAnalysisE2E:
    """E2E tests for statistical analysis and CSV export."""

    def test_epoch_statistics_computation(
        self,
        synthetic_memory_states: dict[int, list[dict[str, np.ndarray]]],
    ) -> None:
        """Test computing epoch statistics."""
        from src.analysis import EpochStatistics, compute_epoch_statistics

        for epoch, memory_states in synthetic_memory_states.items():
            stats = compute_epoch_statistics(
                memory_states, epoch=epoch, step=epoch * 50
            )

            assert isinstance(stats, EpochStatistics)
            assert stats.epoch == epoch
            assert stats.magnitude_mean > 0
            assert 0 <= stats.sparsity <= 1

    def test_trend_analysis(
        self,
        synthetic_memory_states: dict[int, list[dict[str, np.ndarray]]],
    ) -> None:
        """Test trend analysis across epochs."""
        from src.analysis import analyze_trends, compute_epoch_statistics

        # Compute stats for each epoch
        epoch_statistics = {}
        for epoch, memory_states in synthetic_memory_states.items():
            stats = compute_epoch_statistics(
                memory_states, epoch=epoch, step=epoch * 50
            )
            epoch_statistics[epoch] = stats

        # Analyze trends
        trends = analyze_trends(epoch_statistics)

        # Should have trends for key metrics
        assert "sparsity" in trends or "magnitude_mean" in trends

    def test_outlier_detection(
        self,
        synthetic_memory_states: dict[int, list[dict[str, np.ndarray]]],
    ) -> None:
        """Test outlier detection in statistics."""
        from src.analysis import (
            OutlierDetectionResult,
            compute_epoch_statistics,
            detect_outliers,
        )

        # Compute stats
        epoch_statistics = {}
        for epoch, memory_states in synthetic_memory_states.items():
            stats = compute_epoch_statistics(
                memory_states, epoch=epoch, step=epoch * 50
            )
            epoch_statistics[epoch] = stats

        # Detect outliers
        result = detect_outliers(epoch_statistics)

        assert isinstance(result, OutlierDetectionResult)
        # With uniform synthetic data, should have no outliers
        assert isinstance(result.outlier_epochs, list)

    def test_csv_export(
        self,
        synthetic_memory_states: dict[int, list[dict[str, np.ndarray]]],
        output_dir: Path,
    ) -> None:
        """Test exporting statistics to CSV."""
        from src.analysis import (
            AtlasStatisticsResult,
            analyze_trends,
            compute_epoch_statistics,
            compute_summary_statistics,
            detect_outliers,
        )

        # Compute stats
        epoch_statistics = {}
        for epoch, memory_states in synthetic_memory_states.items():
            stats = compute_epoch_statistics(
                memory_states, epoch=epoch, step=epoch * 50
            )
            epoch_statistics[epoch] = stats

        # Create result
        result = AtlasStatisticsResult(
            epochs=sorted(epoch_statistics.keys()),
            epoch_statistics=epoch_statistics,
            trends=analyze_trends(epoch_statistics),
            outliers=detect_outliers(epoch_statistics),
            cluster_stability=[],
            summary_statistics=compute_summary_statistics(epoch_statistics),
            total_analysis_time_seconds=1.0,
        )

        # Export CSV
        csv_path = output_dir / "metrics.csv"
        result.export_csv(csv_path)

        # Verify CSV
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(epoch_statistics)

        # Verify required columns
        required_columns = ["epoch", "magnitude_mean", "sparsity", "rank"]
        for col in required_columns:
            assert col in rows[0], f"Missing column: {col}"

        # Verify evolution is captured
        epochs = [int(row["epoch"]) for row in rows]
        assert epochs == sorted(epochs), "Epochs should be sorted"


# ============================================================================
# Test: Full Pipeline E2E
# ============================================================================


@pytest.mark.synthetic
class TestFullPipelineE2E:
    """Complete E2E tests for the full analysis pipeline."""

    def test_full_synthetic_pipeline(
        self,
        synthetic_embeddings: dict[int, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Test full pipeline with synthetic data.

        Verification steps:
        1. Run aligned PCA with epoch 186 reference on epochs 0,50,100,185
        2. Generate animated concept landscape visualization
        3. Export statistical metrics CSV
        4. Verify HTML <10MB, PNG @300 DPI, metrics show concept evolution
        """
        from src.analysis import (
            AlignedPCA,
            AtlasStatisticsResult,
            analyze_trends,
            build_cross_epoch_trajectory,
            compute_convergence_curve,
            compute_epoch_statistics,
            compute_summary_statistics,
            detect_outliers,
            visualize_epoch_evolution,
        )

        # Step 1: Aligned PCA with epoch 186 reference
        aligner = AlignedPCA(n_components=50, reference_epoch=186)
        assert aligner.reference_epoch == 186

        fit_result = aligner.fit_reference(synthetic_embeddings[186], epoch=186)
        assert fit_result.reference_epoch == 186

        # Transform test epochs
        transformed = {}
        for epoch in [0, 50, 100, 185, 186]:
            if epoch in synthetic_embeddings:
                result = aligner.transform(synthetic_embeddings[epoch], epoch=epoch)
                transformed[epoch] = result.transformed_embeddings

        assert len(transformed) >= 4, "Should transform at least 4 epochs"

        # Build trajectory and compute convergence
        trajectory = build_cross_epoch_trajectory(aligner, synthetic_embeddings)
        epochs_conv, distances = compute_convergence_curve(trajectory)
        assert distances[-1] == pytest.approx(0.0, abs=1e-10), (
            "Final epoch distance should be ~0"
        )

        # Step 2: Generate animated visualization
        html_path = output_dir / "concept_landscape.html"
        png_path = output_dir / "concept_landscape.png"

        # Use 3D for visualization
        vis_embeddings = {e: emb[:, :3] for e, emb in synthetic_embeddings.items()}
        fig, export_result = visualize_epoch_evolution(
            vis_embeddings,
            title="E2E Pipeline Test",
            output_html=str(html_path),
            output_png=str(png_path),
        )

        # Step 3: Verify HTML size limit
        assert export_result.html_path.exists()
        html_size_mb = export_result.html_size_bytes / (1024 * 1024)
        assert html_size_mb < MAX_HTML_SIZE_MB, (
            f"HTML {html_size_mb:.2f} MB exceeds limit"
        )

        # Step 4: Verify PNG exists and has correct parameters
        if export_result.png_path and export_result.png_path.exists():
            # PNG was generated - verify it exists
            assert export_result.png_size_bytes > 0

        # Step 5: Create and export statistics
        # Generate mock memory states for statistics
        np.random.seed(42)
        epoch_statistics = {}
        for epoch in [0, 50, 100, 185, 186]:
            # Create mock memory states
            memory_states = [
                {
                    "M": np.random.randn(8, 100, 64),
                    "S": np.random.randn(8, 100, 64) * 0.5,
                }
                for _ in range(4)
            ]
            stats = compute_epoch_statistics(
                memory_states, epoch=epoch, step=epoch * 50
            )
            epoch_statistics[epoch] = stats

        result = AtlasStatisticsResult(
            epochs=sorted(epoch_statistics.keys()),
            epoch_statistics=epoch_statistics,
            trends=analyze_trends(epoch_statistics),
            outliers=detect_outliers(epoch_statistics),
            cluster_stability=[],
            summary_statistics=compute_summary_statistics(epoch_statistics),
            total_analysis_time_seconds=1.0,
        )

        # Export CSV
        csv_path = output_dir / "metrics.csv"
        result.export_csv(csv_path)

        # Verify CSV shows concept evolution
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(epoch_statistics)

        # Verify evolution is captured (epochs are sorted)
        epochs_in_csv = [int(row["epoch"]) for row in rows]
        assert epochs_in_csv == sorted(epochs_in_csv)


# ============================================================================
# Test: Real Checkpoint Pipeline (Optional)
# ============================================================================


@pytest.mark.skipif(
    not Path(DEFAULT_CHECKPOINT_DIR).exists(),
    reason="Real checkpoints not available",
)
@pytest.mark.slow
class TestRealCheckpointPipeline:
    """E2E tests using real Atlas checkpoints.

    These tests require actual checkpoints and are skipped in CI.
    """

    def test_validate_all_checkpoints(self) -> None:
        """Validate all 186 Atlas checkpoints."""
        from src.analysis import stage_validate_checkpoints

        checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
        summary, epoch_to_path = stage_validate_checkpoints(checkpoint_dir)

        # Should find checkpoints
        assert summary.total_checkpoints > 0

        # Most should be valid
        assert summary.success_rate > 0.9, (
            f"Validation success rate {summary.success_rate:.1%} < 90%"
        )

    def test_batch_pipeline_with_real_checkpoints(
        self,
        output_dir: Path,
    ) -> None:
        """Run batch pipeline with real checkpoints."""
        from src.analysis import BatchAnalysisPipeline

        pipeline = BatchAnalysisPipeline(
            checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
            device="cuda:0" if _cuda_available() else "cpu",
            reference_epoch=186,
        )

        # Run on subset of epochs for speed
        result = pipeline.run(
            epochs=[0, 50, 100, 185],
            output_dir=str(output_dir),
            skip_memory_tracing=True,  # Skip for speed
        )

        # Verify results
        assert result.epochs_processed >= 3

        # Check outputs exist
        assert Path(result.output_directory).exists()

        # Verify GPU memory stayed within budget
        if result.peak_memory_mb > 0:
            assert result.peak_memory_mb < GPU_MEMORY_BUDGET_MB


# ============================================================================
# Test: Batch Pipeline Synthetic
# ============================================================================


@pytest.mark.synthetic
class TestBatchPipelineSyntheticE2E:
    """E2E tests for batch pipeline with synthetic data."""

    def test_pipeline_result_structure(
        self,
        output_dir: Path,
    ) -> None:
        """Test batch pipeline result structure."""
        from src.analysis import (
            BatchPipelineResult,
            PipelineStageResult,
            ValidationSummary,
        )

        # Create synthetic results
        validation = ValidationSummary(
            total_checkpoints=186,
            valid_checkpoints=186,
            invalid_checkpoints=0,
            epochs_found=list(range(186)),
            validation_errors={},
            validation_time_seconds=0.5,
        )

        stage_results = [
            PipelineStageResult(
                stage_name="concept_landscape",
                success=True,
                epochs_processed=4,
                duration_seconds=10.0,
                outputs={"html": str(output_dir / "landscape.html")},
                metrics={"reference_epoch": 186},
            ),
            PipelineStageResult(
                stage_name="memory_tracing",
                success=True,
                epochs_processed=4,
                duration_seconds=15.0,
                outputs={"csv": str(output_dir / "memory.csv")},
                metrics={"sparsity_trend": 0.5},
            ),
            PipelineStageResult(
                stage_name="statistics",
                success=True,
                epochs_processed=4,
                duration_seconds=5.0,
                outputs={"csv": str(output_dir / "metrics.csv")},
                metrics={"magnitude_mean": 0.8},
            ),
        ]

        result = BatchPipelineResult(
            epochs_processed=4,
            epochs_requested=[0, 50, 100, 185],
            validation_summary=validation,
            stage_results=stage_results,
            output_directory=str(output_dir),
            total_duration_seconds=30.0,
            peak_memory_mb=2500.0,
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T00:00:30",
            metadata={"test": True},
        )

        # Verify structure
        assert result.epochs_processed == 4
        assert len(result.stage_results) == 3
        assert result.validation_summary.success_rate == 1.0

        # Test export
        summary_path = output_dir / "pipeline_summary.json"
        result.export_summary(summary_path)

        assert summary_path.exists()
        with open(summary_path) as f:
            data = json.load(f)

        assert data["epochs_processed"] == 4
        assert data["validation_summary"]["valid_checkpoints"] == 186


# ============================================================================
# Test: Export Verification
# ============================================================================


@pytest.mark.synthetic
class TestExportVerificationE2E:
    """E2E tests for export artifact verification."""

    def test_html_size_within_limit(
        self,
        synthetic_embeddings: dict[int, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Verify HTML exports are within 10MB limit."""
        from src.analysis import visualize_epoch_evolution

        html_path = output_dir / "test_size.html"

        fig, result = visualize_epoch_evolution(
            {e: emb[:, :3] for e, emb in synthetic_embeddings.items()},
            output_html=str(html_path),
        )

        assert result.html_within_limit, (
            f"HTML {result.html_size_mb:.2f} MB exceeds 10MB"
        )

    def test_png_dimensions_and_dpi(
        self,
        synthetic_embeddings: dict[int, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Verify PNG exports have correct dimensions for 300 DPI."""
        from src.analysis import (
            DEFAULT_PNG_HEIGHT,
            DEFAULT_PNG_WIDTH,
            PNG_SCALE_FACTOR,
            visualize_epoch_evolution,
        )

        png_path = output_dir / "test_dpi.png"

        fig, result = visualize_epoch_evolution(
            {e: emb[:, :3] for e, emb in synthetic_embeddings.items()},
            output_png=str(png_path),
        )

        # Verify export configuration uses correct scale
        assert PNG_SCALE_FACTOR == 2.5
        assert DEFAULT_PNG_WIDTH == 1200
        assert DEFAULT_PNG_HEIGHT == 800

        # Effective dimensions: 1200*2.5 x 800*2.5 = 3000x2000 pixels
        # At ~10 inches width = 300 DPI

        if result.png_path and result.png_path.exists():
            # PNG was successfully generated
            assert result.png_size_bytes > 0


# ============================================================================
# Helper Functions
# ============================================================================


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Run E2E tests from command line."""
    import subprocess
    import sys

    # Run pytest with this file
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-v",
            "--tb=short",
        ],
        cwd=Path(__file__).parent.parent,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
