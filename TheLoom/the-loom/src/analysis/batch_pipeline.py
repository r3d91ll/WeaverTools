"""Batch Analysis Pipeline for Cross-Epoch Atlas Model Processing.

This module implements a comprehensive batch analysis pipeline that orchestrates
all Atlas model analysis components for cross-epoch processing. It provides
an efficient, memory-safe approach to analyzing all 186 training checkpoints.

PIPELINE STAGES
===============
1. Checkpoint Discovery & Validation
   - Find all checkpoints in directory
   - Validate structure and memory states
   - Report validation summary

2. Embedding Extraction (with aligned PCA)
   - Sequential load-extract-unload pattern
   - Apply aligned PCA with epoch 186 reference
   - Track convergence metrics

3. Memory Analysis
   - Extract M/S matrices from each checkpoint
   - Compute statistics (magnitude, sparsity, rank)
   - Detect trends and outliers

4. Visualization Generation
   - Concept landscape animations
   - Memory evolution plots
   - Convergence curves

5. Statistical Export
   - CSV metrics file
   - JSON detailed report
   - Summary statistics

MEMORY MANAGEMENT
=================
Atlas checkpoints are ~450MB each. Loading 186 checkpoints simultaneously
would require ~84GB memory. This pipeline implements sequential processing
with immediate unloading to maintain <5GB GPU memory:

    for checkpoint in checkpoints:
        load(checkpoint) -> extract_data() -> unload(checkpoint)

TARGET: Process 186 epochs in <15 minutes with GPU memory <5GB.

USAGE
=====
CLI:
    poetry run python -m src.analysis.batch_pipeline --epochs all --output /tmp/analysis_results/

Python:
    from src.analysis.batch_pipeline import BatchAnalysisPipeline

    pipeline = BatchAnalysisPipeline()
    result = pipeline.run(
        epochs="all",
        output_dir="/tmp/analysis_results/"
    )

Integration: Works with AtlasLoader, AlignedPCA, ConceptLandscape, MemoryTracer,
and AtlasStatisticsAnalyzer from TheLoom.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default checkpoint directory (from environment or empty)
# Set ATLAS_CHECKPOINT_DIR environment variable or pass --checkpoint-dir explicitly
DEFAULT_CHECKPOINT_DIR = os.environ.get("ATLAS_CHECKPOINT_DIR", "")

# Default reference epoch per PRD v1.1
DEFAULT_REFERENCE_EPOCH = 186

# Total expected epochs in Atlas training
TOTAL_EPOCHS = 186

# Memory budget for GPU operations (bytes)
GPU_MEMORY_BUDGET_MB = 5000

# Checkpoint processing timeout (seconds)
CHECKPOINT_TIMEOUT_SECONDS = 60

# Analysis batch sizes for memory efficiency
EMBEDDING_BATCH_SIZE = 10  # Epochs to process before memory cleanup
MEMORY_BATCH_SIZE = 20     # Memory checkpoints per batch

# Visualization settings
MAX_VISUALIZATION_EPOCHS = 20  # For animated visualizations
VISUALIZATION_SAMPLE_EPOCHS = [0, 10, 25, 50, 75, 100, 125, 150, 175, 185]


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class ValidationSummary:
    """Summary of checkpoint validation results.

    Note: When an `epochs` filter is provided to the pipeline, `valid_checkpoints`
    counts only checkpoints that are both structurally valid AND whose epoch
    is in the requested filter. This means `success_rate` reflects "valid and
    selected" checkpoints relative to total files found, not overall structural
    health. Use `total_checkpoints` for the full count of files found.

    Attributes:
        total_checkpoints: Total number of checkpoint files found.
        valid_checkpoints: Number of valid and selected checkpoints (filtered by epochs if specified).
        invalid_checkpoints: Number of invalid checkpoints.
        epochs_found: List of epoch numbers found (only those passing the filter).
        validation_errors: Dict mapping invalid paths to error messages.
        validation_time_seconds: Time taken for validation.
    """

    total_checkpoints: int
    valid_checkpoints: int
    invalid_checkpoints: int
    epochs_found: list[int]
    validation_errors: dict[str, str]
    validation_time_seconds: float

    @property
    def success_rate(self) -> float:
        """Fraction of checkpoints that are valid."""
        if self.total_checkpoints == 0:
            return 0.0
        return self.valid_checkpoints / self.total_checkpoints

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_checkpoints": self.total_checkpoints,
            "valid_checkpoints": self.valid_checkpoints,
            "invalid_checkpoints": self.invalid_checkpoints,
            "epochs_found": self.epochs_found,
            "success_rate": self.success_rate,
            "validation_errors": self.validation_errors,
            "validation_time_seconds": self.validation_time_seconds,
        }


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage.

    Attributes:
        stage_name: Name of the pipeline stage.
        success: Whether the stage completed successfully.
        epochs_processed: Number of epochs processed in this stage.
        duration_seconds: Time taken for the stage.
        error: Error message if stage failed.
        outputs: Dict of output file paths.
        metrics: Dict of computed metrics.
    """

    stage_name: str
    success: bool
    epochs_processed: int
    duration_seconds: float
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "epochs_processed": self.epochs_processed,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "outputs": self.outputs,
            "metrics": self.metrics,
        }


@dataclass
class BatchPipelineResult:
    """Complete results from the batch analysis pipeline.

    Attributes:
        epochs_processed: Total epochs successfully processed.
        epochs_requested: Epochs that were requested.
        validation_summary: Checkpoint validation results.
        stage_results: Results from each pipeline stage.
        output_directory: Path to output directory.
        total_duration_seconds: Total pipeline runtime.
        peak_memory_mb: Peak GPU memory usage (if tracked).
        start_time: Pipeline start timestamp.
        end_time: Pipeline end timestamp.
        metadata: Additional pipeline metadata.
    """

    epochs_processed: int
    epochs_requested: list[int]
    validation_summary: ValidationSummary
    stage_results: list[PipelineStageResult]
    output_directory: str
    total_duration_seconds: float
    peak_memory_mb: float
    start_time: str
    end_time: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epochs_processed": self.epochs_processed,
            "epochs_requested": self.epochs_requested,
            "validation_summary": self.validation_summary.to_dict(),
            "stage_results": [s.to_dict() for s in self.stage_results],
            "output_directory": self.output_directory,
            "total_duration_seconds": self.total_duration_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
        }

    def export_summary(self, output_path: str | Path) -> None:
        """Export pipeline summary to JSON.

        Parameters:
            output_path: Path for JSON output.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Pipeline summary exported to {output_path}")


# ============================================================================
# Pipeline Stage Implementations
# ============================================================================


def stage_validate_checkpoints(
    checkpoint_dir: Path,
    epochs: list[int] | None = None,
) -> tuple[ValidationSummary, dict[int, Path]]:
    """Stage 1: Discover and validate checkpoints.

    Parameters:
        checkpoint_dir: Directory containing checkpoint files.
        epochs: Specific epochs to validate, or None for all.

    Returns:
        Tuple of (ValidationSummary, epoch_to_path_mapping).
    """
    from ..loaders.atlas_loader import AtlasLoader

    start_time = time.time()

    # Find all checkpoints
    all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not all_checkpoints:
        all_checkpoints = sorted(checkpoint_dir.glob("*.pt"))

    logger.info(f"Found {len(all_checkpoints)} checkpoint files")

    loader = AtlasLoader()
    epoch_to_path: dict[int, Path] = {}
    validation_errors: dict[str, str] = {}
    valid_count = 0

    for checkpoint_path in all_checkpoints:
        try:
            result = loader.validate_checkpoint(checkpoint_path, strict=False)

            if result["valid"]:
                epoch = result.get("epoch", 0)

                # Filter by requested epochs if specified
                if epochs is None or epoch in epochs:
                    epoch_to_path[epoch] = checkpoint_path
                    valid_count += 1

                    if epochs is None or len(epoch_to_path) < 10:
                        logger.debug(f"Valid: {checkpoint_path.name} (epoch {epoch})")
            else:
                validation_errors[str(checkpoint_path)] = result.get("error", "Unknown")

        except Exception as e:
            validation_errors[str(checkpoint_path)] = str(e)

    validation_time = time.time() - start_time

    summary = ValidationSummary(
        total_checkpoints=len(all_checkpoints),
        valid_checkpoints=valid_count,
        invalid_checkpoints=len(validation_errors),
        epochs_found=sorted(epoch_to_path.keys()),
        validation_errors=validation_errors,
        validation_time_seconds=validation_time,
    )

    logger.info(
        f"Validation complete: {valid_count}/{len(all_checkpoints)} valid "
        f"in {validation_time:.2f}s"
    )

    return summary, epoch_to_path


def stage_concept_landscape(
    epoch_to_path: dict[int, Path],
    output_dir: Path,
    reference_epoch: int = DEFAULT_REFERENCE_EPOCH,
    sample_epochs: list[int] | None = None,
    device: str = "cuda:0",
) -> PipelineStageResult:
    """Stage 2: Concept Landscape Analysis with aligned PCA.

    Parameters:
        epoch_to_path: Mapping of epoch to checkpoint path.
        output_dir: Output directory for results.
        reference_epoch: PCA reference epoch (default: 186).
        sample_epochs: Specific epochs for visualization (default: sampled).
        device: Device for model execution.

    Returns:
        PipelineStageResult with stage outcomes.
    """
    from .concept_landscape import (
        DEFAULT_SAMPLE_TEXTS,
        VisualizationConfig,
        analyze_concept_landscape,
        create_convergence_plot,
        create_landscape_visualization,
    )

    start_time = time.time()
    outputs: dict[str, str] = {}
    metrics: dict[str, Any] = {}

    try:
        # Select epochs for visualization
        all_epochs = sorted(epoch_to_path.keys())

        if sample_epochs is None:
            # Sample epochs for visualization to avoid huge animations
            if len(all_epochs) > MAX_VISUALIZATION_EPOCHS:
                # Use predefined sample points plus reference
                sample_epochs = [e for e in VISUALIZATION_SAMPLE_EPOCHS if e in all_epochs]
                if reference_epoch not in sample_epochs and reference_epoch in all_epochs:
                    sample_epochs.append(reference_epoch)
                sample_epochs = sorted(set(sample_epochs))
            else:
                sample_epochs = all_epochs

        logger.info(f"Concept landscape analysis on {len(sample_epochs)} epochs")

        # Extract embeddings for sample epochs
        embeddings_by_epoch = _extract_embeddings_batch(
            epoch_to_path,
            sample_epochs,
            device=device,
        )

        if not embeddings_by_epoch:
            return PipelineStageResult(
                stage_name="concept_landscape",
                success=False,
                epochs_processed=0,
                duration_seconds=time.time() - start_time,
                error="No embeddings extracted",
            )

        # Find best reference epoch from available epochs
        if reference_epoch not in embeddings_by_epoch:
            reference_epoch = max(embeddings_by_epoch.keys())
            logger.warning(
                f"Reference epoch 186 not available, using {reference_epoch}"
            )

        # Run analysis
        result = analyze_concept_landscape(
            embeddings_by_epoch,
            reference_epoch=reference_epoch,
            n_components=50,
            compute_clusters=True,
        )
        result.sample_texts = DEFAULT_SAMPLE_TEXTS

        # Generate visualizations
        landscape_html = output_dir / "concept_landscape.html"
        landscape_png = output_dir / "concept_landscape.png"
        convergence_plot = output_dir / "convergence.html"

        config = VisualizationConfig(
            output_html=str(landscape_html),
            output_png=str(landscape_png),
        )
        create_landscape_visualization(result, config)
        create_convergence_plot(result, convergence_plot)

        outputs["concept_landscape_html"] = str(landscape_html)
        outputs["concept_landscape_png"] = str(landscape_png)
        outputs["convergence_plot"] = str(convergence_plot)

        # Record metrics
        metrics["epochs_analyzed"] = len(result.epochs)
        metrics["reference_epoch"] = result.reference_epoch
        metrics["variance_explained"] = {
            str(e): result.variance_explained_per_epoch[e]
            for e in result.epochs
        }
        metrics["convergence_distances"] = {
            str(e): result.convergence_distances[e]
            for e in result.epochs
        }

        duration = time.time() - start_time

        return PipelineStageResult(
            stage_name="concept_landscape",
            success=True,
            epochs_processed=len(result.epochs),
            duration_seconds=duration,
            outputs=outputs,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Concept landscape stage failed: {e}")
        return PipelineStageResult(
            stage_name="concept_landscape",
            success=False,
            epochs_processed=0,
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


def stage_memory_tracing(
    epoch_to_path: dict[int, Path],
    output_dir: Path,
    epochs: list[int] | None = None,
    device: str = "cpu",
) -> PipelineStageResult:
    """Stage 3: Memory Episode Tracing.

    Parameters:
        epoch_to_path: Mapping of epoch to checkpoint path.
        output_dir: Output directory for results.
        epochs: Specific epochs to analyze (default: all).
        device: Device to load checkpoints to.

    Returns:
        PipelineStageResult with stage outcomes.
    """
    from .memory_tracing import (
        MemoryTracer,
        create_evolution_plot,
    )

    start_time = time.time()
    outputs: dict[str, str] = {}
    metrics: dict[str, Any] = {}

    try:
        all_epochs = sorted(epoch_to_path.keys())
        if epochs is None:
            epochs = all_epochs

        logger.info(f"Memory tracing on {len(epochs)} epochs")

        # Create tracer
        tracer = MemoryTracer(device=device)

        # Analyze each epoch sequentially
        epoch_stats = {}
        for epoch in epochs:
            if epoch not in epoch_to_path:
                continue

            try:
                stats = tracer.analyze_checkpoint(epoch_to_path[epoch])
                epoch_stats[epoch] = stats

                if len(epoch_stats) % 20 == 0:
                    logger.info(f"Memory tracing: {len(epoch_stats)}/{len(epochs)} epochs")

            except Exception as e:
                logger.warning(f"Failed to analyze epoch {epoch}: {e}")
                continue

        if not epoch_stats:
            return PipelineStageResult(
                stage_name="memory_tracing",
                success=False,
                epochs_processed=0,
                duration_seconds=time.time() - start_time,
                error="No epochs successfully analyzed",
            )

        # Compute trends (reusing from memory_tracing)
        from scipy import stats as scipy_stats

        from .memory_tracing import MemoryEvolutionResult

        analyzed_epochs = sorted(epoch_stats.keys())
        epoch_array = np.array(analyzed_epochs)
        sparsities = np.array([epoch_stats[e].mean_m_sparsity for e in analyzed_epochs])
        ranks = np.array([epoch_stats[e].mean_m_effective_rank for e in analyzed_epochs])
        magnitudes = np.array([epoch_stats[e].mean_m_magnitude for e in analyzed_epochs])

        sparsity_trend = 0.0
        rank_trend = 0.0
        magnitude_trend = 0.0

        if len(analyzed_epochs) >= 3:
            if np.std(sparsities) > 0:
                sparsity_trend = float(scipy_stats.pearsonr(epoch_array, sparsities).statistic)
            if np.std(ranks) > 0:
                rank_trend = float(scipy_stats.pearsonr(epoch_array, ranks).statistic)
            if np.std(magnitudes) > 0:
                magnitude_trend = float(scipy_stats.pearsonr(epoch_array, magnitudes).statistic)

        evolution = MemoryEvolutionResult(
            epochs=analyzed_epochs,
            epoch_stats=epoch_stats,
            sparsity_trend=sparsity_trend,
            rank_trend=rank_trend,
            magnitude_trend=magnitude_trend,
            outlier_epochs=[],
            total_analysis_time_seconds=time.time() - start_time,
        )

        # Generate evolution plot
        evolution_html = output_dir / "memory_evolution.html"
        create_evolution_plot(evolution, evolution_html)
        outputs["memory_evolution_plot"] = str(evolution_html)

        # Export CSV
        memory_csv = output_dir / "memory_stats.csv"
        _export_memory_csv(epoch_stats, memory_csv)
        outputs["memory_stats_csv"] = str(memory_csv)

        # Record metrics
        metrics["epochs_analyzed"] = len(epoch_stats)
        metrics["sparsity_trend"] = sparsity_trend
        metrics["rank_trend"] = rank_trend
        metrics["magnitude_trend"] = magnitude_trend
        metrics["final_sparsity"] = float(sparsities[-1]) if len(sparsities) > 0 else 0
        metrics["final_rank"] = float(ranks[-1]) if len(ranks) > 0 else 0

        duration = time.time() - start_time

        return PipelineStageResult(
            stage_name="memory_tracing",
            success=True,
            epochs_processed=len(epoch_stats),
            duration_seconds=duration,
            outputs=outputs,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Memory tracing stage failed: {e}")
        return PipelineStageResult(
            stage_name="memory_tracing",
            success=False,
            epochs_processed=0,
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


def stage_statistics(
    epoch_to_path: dict[int, Path],
    output_dir: Path,
    epochs: list[int] | None = None,
    device: str = "cpu",
) -> PipelineStageResult:
    """Stage 4: Statistical Analysis.

    Parameters:
        epoch_to_path: Mapping of epoch to checkpoint path.
        output_dir: Output directory for results.
        epochs: Specific epochs to analyze (default: all).
        device: Device to load checkpoints to.

    Returns:
        PipelineStageResult with stage outcomes.
    """
    from .atlas_statistics import AtlasStatisticsAnalyzer

    start_time = time.time()
    outputs: dict[str, str] = {}
    metrics: dict[str, Any] = {}

    try:
        all_epochs = sorted(epoch_to_path.keys())
        if epochs is None:
            epochs = all_epochs

        logger.info(f"Statistical analysis on {len(epochs)} epochs")

        # Create analyzer
        analyzer = AtlasStatisticsAnalyzer(device=device)

        # Analyze each checkpoint
        epoch_statistics = {}
        for epoch in epochs:
            if epoch not in epoch_to_path:
                continue

            try:
                stats = analyzer.analyze_checkpoint(epoch_to_path[epoch])
                epoch_statistics[epoch] = stats

                if len(epoch_statistics) % 20 == 0:
                    logger.info(f"Statistics: {len(epoch_statistics)}/{len(epochs)} epochs")

            except Exception as e:
                logger.warning(f"Failed to compute stats for epoch {epoch}: {e}")
                continue

        if not epoch_statistics:
            return PipelineStageResult(
                stage_name="statistics",
                success=False,
                epochs_processed=0,
                duration_seconds=time.time() - start_time,
                error="No epochs successfully analyzed",
            )

        # Compute trends and summary
        from .atlas_statistics import (
            AtlasStatisticsResult,
            analyze_trends,
            compute_summary_statistics,
            detect_outliers,
        )

        trends = analyze_trends(epoch_statistics)
        outliers = detect_outliers(epoch_statistics)
        summary = compute_summary_statistics(epoch_statistics)

        result = AtlasStatisticsResult(
            epochs=sorted(epoch_statistics.keys()),
            epoch_statistics=epoch_statistics,
            trends=trends,
            outliers=outliers,
            cluster_stability=[],
            summary_statistics=summary,
            total_analysis_time_seconds=time.time() - start_time,
        )

        # Export CSV
        stats_csv = output_dir / "metrics.csv"
        result.export_csv(stats_csv)
        outputs["metrics_csv"] = str(stats_csv)

        # Export JSON
        stats_json = output_dir / "statistics.json"
        result.export_json(stats_json)
        outputs["statistics_json"] = str(stats_json)

        # Record metrics
        metrics["epochs_analyzed"] = len(epoch_statistics)
        if summary:
            metrics["magnitude_mean"] = summary.get("magnitude", {}).get("mean", 0)
            metrics["sparsity_mean"] = summary.get("sparsity", {}).get("mean", 0)
            metrics["rank_mean"] = summary.get("rank", {}).get("mean", 0)
        metrics["outlier_epochs"] = outliers.outlier_epochs

        duration = time.time() - start_time

        return PipelineStageResult(
            stage_name="statistics",
            success=True,
            epochs_processed=len(epoch_statistics),
            duration_seconds=duration,
            outputs=outputs,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Statistics stage failed: {e}")
        return PipelineStageResult(
            stage_name="statistics",
            success=False,
            epochs_processed=0,
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_embeddings_batch(
    epoch_to_path: dict[int, Path],
    epochs: list[int],
    device: str = "cuda:0",
) -> dict[int, NDArray[np.floating[Any]]]:
    """Extract embeddings from multiple epochs with memory management.

    Uses sequential load-extract-unload pattern.

    Parameters:
        epoch_to_path: Mapping of epoch to checkpoint path.
        epochs: Epochs to extract.
        device: Device for model execution.

    Returns:
        Dict mapping epoch to embeddings array.
    """
    import torch

    from ..loaders.atlas_loader import AtlasLoader
    from .concept_landscape import DEFAULT_SAMPLE_TEXTS

    loader = AtlasLoader()
    embeddings_by_epoch: dict[int, NDArray[np.floating[Any]]] = {}

    for epoch in epochs:
        if epoch not in epoch_to_path:
            continue

        checkpoint_path = epoch_to_path[epoch]

        try:
            logger.debug(f"Extracting embeddings for epoch {epoch}")

            # Load model
            loaded_model = loader.load(str(checkpoint_path), device=device)

            # Extract embeddings for each sample text
            embeddings_list = []
            with torch.no_grad():
                for text in DEFAULT_SAMPLE_TEXTS:
                    output = loader.embed(loaded_model, text, pooling="mean")
                    embeddings_list.append(output.embedding.numpy())

            embeddings = np.stack(embeddings_list)
            embeddings_by_epoch[epoch] = embeddings

            # Clean up - critical for memory management
            del loaded_model.model
            del loaded_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Failed to extract embeddings for epoch {epoch}: {e}")
            continue

    return embeddings_by_epoch


def _export_memory_csv(
    epoch_stats: dict[int, Any],
    output_path: Path,
) -> None:
    """Export memory statistics to CSV.

    Parameters:
        epoch_stats: Dict mapping epoch to MemoryEpisodeStats.
        output_path: Path for CSV output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "epoch",
        "step",
        "mean_m_sparsity",
        "mean_s_sparsity",
        "mean_m_rank",
        "mean_s_rank",
        "mean_m_magnitude",
        "mean_s_magnitude",
        "overall_health",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for epoch in sorted(epoch_stats.keys()):
            stats = epoch_stats[epoch]
            writer.writerow({
                "epoch": epoch,
                "step": stats.step,
                "mean_m_sparsity": f"{stats.mean_m_sparsity:.6f}",
                "mean_s_sparsity": f"{stats.mean_s_sparsity:.6f}",
                "mean_m_rank": f"{stats.mean_m_effective_rank:.2f}",
                "mean_s_rank": f"{stats.mean_s_effective_rank:.2f}",
                "mean_m_magnitude": f"{stats.mean_m_magnitude:.6f}",
                "mean_s_magnitude": f"{stats.mean_s_magnitude:.6f}",
                "overall_health": stats.overall_health,
            })


def _get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB.

    Returns:
        Current GPU memory usage in MB, or 0 if not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ============================================================================
# Main Pipeline Class
# ============================================================================


class BatchAnalysisPipeline:
    """Orchestrates the full batch analysis pipeline.

    Coordinates checkpoint validation, embedding extraction, memory analysis,
    and statistical computation across all training epochs.

    Example:
        pipeline = BatchAnalysisPipeline()
        result = pipeline.run(
            epochs="all",
            output_dir="/tmp/analysis_results/"
        )
        print(f"Processed {result.epochs_processed} epochs")
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "cuda:0",
        reference_epoch: int = DEFAULT_REFERENCE_EPOCH,
    ):
        """Initialize the pipeline.

        Parameters:
            checkpoint_dir: Directory containing checkpoints.
            device: Device for GPU operations.
            reference_epoch: PCA reference epoch (default: 186).
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(DEFAULT_CHECKPOINT_DIR)
        self.device = device
        self.reference_epoch = reference_epoch

    def run(
        self,
        epochs: Literal["all"] | list[int] = "all",
        output_dir: str | Path = "/tmp/analysis_results",
        skip_concept_landscape: bool = False,
        skip_memory_tracing: bool = False,
        skip_statistics: bool = False,
        profile_memory: bool = False,
    ) -> BatchPipelineResult:
        """Run the complete batch analysis pipeline.

        Parameters:
            epochs: "all" for all epochs, or list of specific epoch numbers.
            output_dir: Output directory for results.
            skip_concept_landscape: Skip concept landscape stage.
            skip_memory_tracing: Skip memory tracing stage.
            skip_statistics: Skip statistics stage.
            profile_memory: Track GPU memory usage.

        Returns:
            BatchPipelineResult with complete pipeline results.
        """
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stage_results: list[PipelineStageResult] = []
        peak_memory_mb = 0.0

        # Parse epochs
        if epochs == "all":
            requested_epochs = None  # Will discover from checkpoints
        else:
            # epochs is list[int] here due to Literal type narrowing
            requested_epochs = list(epochs)

        logger.info("Starting batch analysis pipeline")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Epochs: {'all' if requested_epochs is None else requested_epochs}")

        # Stage 1: Validation
        logger.info("=== Stage 1: Checkpoint Validation ===")
        validation_summary, epoch_to_path = stage_validate_checkpoints(
            self.checkpoint_dir,
            requested_epochs,
        )

        if not epoch_to_path:
            logger.error("No valid checkpoints found!")
            return BatchPipelineResult(
                epochs_processed=0,
                epochs_requested=requested_epochs or [],
                validation_summary=validation_summary,
                stage_results=[],
                output_directory=str(output_dir),
                total_duration_seconds=time.time() - start_time,
                peak_memory_mb=0,
                start_time=start_timestamp,
                end_time=datetime.now().isoformat(),
                metadata={"error": "No valid checkpoints found"},
            )

        epochs_to_process = sorted(epoch_to_path.keys())
        logger.info(f"Found {len(epochs_to_process)} valid epochs")

        # Track memory if requested
        if profile_memory:
            peak_memory_mb = max(peak_memory_mb, _get_gpu_memory_mb())

        # Stage 2: Concept Landscape
        if not skip_concept_landscape:
            logger.info("=== Stage 2: Concept Landscape Analysis ===")
            landscape_result = stage_concept_landscape(
                epoch_to_path,
                output_dir,
                reference_epoch=self.reference_epoch,
                device=self.device,
            )
            stage_results.append(landscape_result)

            if profile_memory:
                peak_memory_mb = max(peak_memory_mb, _get_gpu_memory_mb())

        # Stage 3: Memory Tracing
        if not skip_memory_tracing:
            logger.info("=== Stage 3: Memory Episode Tracing ===")
            memory_result = stage_memory_tracing(
                epoch_to_path,
                output_dir,
                epochs=epochs_to_process,
                device="cpu",  # Use CPU for memory-heavy operations
            )
            stage_results.append(memory_result)

            if profile_memory:
                peak_memory_mb = max(peak_memory_mb, _get_gpu_memory_mb())

        # Stage 4: Statistics
        if not skip_statistics:
            logger.info("=== Stage 4: Statistical Analysis ===")
            stats_result = stage_statistics(
                epoch_to_path,
                output_dir,
                epochs=epochs_to_process,
                device="cpu",
            )
            stage_results.append(stats_result)

            if profile_memory:
                peak_memory_mb = max(peak_memory_mb, _get_gpu_memory_mb())

        # Compute final metrics
        total_duration = time.time() - start_time
        end_timestamp = datetime.now().isoformat()

        epochs_processed = len(epochs_to_process)

        result = BatchPipelineResult(
            epochs_processed=epochs_processed,
            epochs_requested=requested_epochs or epochs_to_process,
            validation_summary=validation_summary,
            stage_results=stage_results,
            output_directory=str(output_dir),
            total_duration_seconds=total_duration,
            peak_memory_mb=peak_memory_mb,
            start_time=start_timestamp,
            end_time=end_timestamp,
            metadata={
                "checkpoint_dir": str(self.checkpoint_dir),
                "reference_epoch": self.reference_epoch,
                "device": self.device,
                "stages_run": [s.stage_name for s in stage_results],
            },
        )

        # Export summary
        summary_path = output_dir / "pipeline_summary.json"
        result.export_summary(summary_path)

        logger.info("=== Pipeline Complete ===")
        logger.info(f"Processed {epochs_processed} epochs")
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info(f"Output directory: {output_dir}")

        return result


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface for batch analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch Analysis Pipeline for Atlas model cross-epoch processing"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="all",
        help="Epochs to process: 'all' or comma-separated list (e.g., '0,50,100,185')",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=f"Directory containing checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for GPU operations (default: cuda:0)",
    )
    parser.add_argument(
        "--reference-epoch",
        type=int,
        default=DEFAULT_REFERENCE_EPOCH,
        help=f"PCA reference epoch (default: {DEFAULT_REFERENCE_EPOCH})",
    )
    parser.add_argument(
        "--skip-concept-landscape",
        action="store_true",
        help="Skip concept landscape analysis stage",
    )
    parser.add_argument(
        "--skip-memory-tracing",
        action="store_true",
        help="Skip memory tracing stage",
    )
    parser.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Skip statistics stage",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Track GPU memory usage during pipeline",
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
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.test:
        _run_synthetic_test(args.output)
        return

    # Parse epochs
    if args.epochs.lower() == "all":
        epochs = "all"
    else:
        epochs = [int(e.strip()) for e in args.epochs.split(",")]

    # Create and run pipeline
    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR

    pipeline = BatchAnalysisPipeline(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        reference_epoch=args.reference_epoch,
    )

    try:
        result = pipeline.run(
            epochs=epochs,
            output_dir=args.output,
            skip_concept_landscape=args.skip_concept_landscape,
            skip_memory_tracing=args.skip_memory_tracing,
            skip_statistics=args.skip_statistics,
            profile_memory=args.profile_memory,
        )

        # Print summary
        print("\n=== Batch Analysis Complete ===")
        print(f"Processed {result.epochs_processed} epochs")
        print(f"Duration: {result.total_duration_seconds:.2f}s")
        print(f"Output: {result.output_directory}")

        # Print stage results
        print("\nStage Results:")
        for stage in result.stage_results:
            status = "✓" if stage.success else "✗"
            print(f"  {status} {stage.stage_name}: {stage.epochs_processed} epochs in {stage.duration_seconds:.2f}s")
            if stage.error:
                print(f"      Error: {stage.error}")

        if result.peak_memory_mb > 0:
            print(f"\nPeak GPU memory: {result.peak_memory_mb:.1f}MB")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_synthetic_test(output_dir: str) -> None:
    """Run test with synthetic data (no checkpoints needed)."""
    print("Running synthetic test (no checkpoints needed)...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Simulate epoch data
    epochs = list(range(0, 186, 10))  # Every 10th epoch
    epochs.append(185)  # Include final epoch

    print(f"Simulating {len(epochs)} epochs: {epochs[:5]}...{epochs[-3:]}")

    # Create synthetic validation summary
    validation = ValidationSummary(
        total_checkpoints=186,
        valid_checkpoints=186,
        invalid_checkpoints=0,
        epochs_found=epochs,
        validation_errors={},
        validation_time_seconds=0.5,
    )

    # Create synthetic stage results
    stage_results = [
        PipelineStageResult(
            stage_name="concept_landscape",
            success=True,
            epochs_processed=len(epochs),
            duration_seconds=5.0,
            outputs={"concept_landscape_html": str(output_dir / "concept_landscape.html")},
            metrics={"reference_epoch": 186, "epochs_analyzed": len(epochs)},
        ),
        PipelineStageResult(
            stage_name="memory_tracing",
            success=True,
            epochs_processed=len(epochs),
            duration_seconds=10.0,
            outputs={"memory_evolution_plot": str(output_dir / "memory_evolution.html")},
            metrics={"sparsity_trend": 0.3, "rank_trend": -0.1},
        ),
        PipelineStageResult(
            stage_name="statistics",
            success=True,
            epochs_processed=len(epochs),
            duration_seconds=8.0,
            outputs={"metrics_csv": str(output_dir / "metrics.csv")},
            metrics={"magnitude_mean": 0.5, "sparsity_mean": 0.4},
        ),
    ]

    # Create synthetic result
    result = BatchPipelineResult(
        epochs_processed=len(epochs),
        epochs_requested=epochs,
        validation_summary=validation,
        stage_results=stage_results,
        output_directory=str(output_dir),
        total_duration_seconds=25.0,
        peak_memory_mb=2500.0,
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
        metadata={"source": "synthetic_test"},
    )

    # Export summary
    result.export_summary(output_dir / "pipeline_summary.json")

    # Create synthetic metrics CSV
    metrics_csv = output_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "magnitude_mean", "sparsity", "rank"])
        writer.writeheader()
        for epoch in epochs:
            # Simulate metrics changing over training
            progress = epoch / 185
            writer.writerow({
                "epoch": epoch,
                "magnitude_mean": f"{1.0 - 0.3 * progress:.6f}",
                "sparsity": f"{0.3 + 0.4 * progress:.6f}",
                "rank": f"{100 - 50 * progress:.2f}",
            })

    print("\n=== Synthetic Test Complete ===")
    print(f"Processed {result.epochs_processed} epochs")
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print(f"  - {output_dir / 'pipeline_summary.json'}")
    print(f"  - {output_dir / 'metrics.csv'}")

    print("\nSynthetic test completed successfully!")


if __name__ == "__main__":
    main()
