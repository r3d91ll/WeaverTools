"""Statistical Analysis for Atlas Model Memory Tracing.

This module implements comprehensive statistical analysis for the Atlas model's
memory states across training epochs, enabling quantitative comparison of memory
evolution and identification of significant training dynamics.

ANALYSIS CAPABILITIES
=====================
1. Cross-Epoch Statistics:
   - Memory magnitude (mean, std, min, max) per epoch
   - Sparsity evolution (fraction of near-zero entries)
   - Effective rank progression (via SVD analysis)
   - Concept cluster stability (clustering consistency)

2. Statistical Metrics:
   - Descriptive statistics (mean, std, percentiles)
   - Trend analysis (linear regression coefficients)
   - Outlier detection (z-score and IQR methods)
   - Correlation matrices between metrics

3. Export Capabilities:
   - CSV export with standardized columns
   - JSON export for full result preservation
   - Summary statistics for publication

4. Outlier Detection:
   - Epochs with unusual magnitude changes
   - Sparsity anomalies
   - Rank instability detection

INTEGRATION
===========
Works with AtlasLoader and MemoryTracer for checkpoint access and memory
extraction. Uses AlignedPCA for concept cluster analysis.

Usage:
    from src.analysis.atlas_statistics import AtlasStatisticsAnalyzer

    # Analyze specific epochs
    analyzer = AtlasStatisticsAnalyzer()
    result = analyzer.analyze_epochs([0, 50, 100, 185])

    # Export to CSV
    result.export_csv("/tmp/metrics.csv")

CLI:
    poetry run python -m src.analysis.atlas_statistics --epochs 0,50,100,185 --output /tmp/metrics.csv
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
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default checkpoint directory (from environment or current working directory)
# Set ATLAS_CHECKPOINT_DIR environment variable or pass --checkpoint-dir explicitly
DEFAULT_CHECKPOINT_DIR = os.environ.get("ATLAS_CHECKPOINT_DIR", "")

# Z-score threshold for outlier detection
OUTLIER_Z_THRESHOLD = 2.5

# IQR multiplier for outlier detection
OUTLIER_IQR_MULTIPLIER = 1.5

# Minimum epochs required for statistical trend analysis
MIN_EPOCHS_FOR_STATS = 3

# Sparsity threshold for memory entries (from memory_tracing)
SPARSITY_THRESHOLD = 1e-6

# Rank threshold for singular value significance (from memory_tracing)
RANK_THRESHOLD = 1e-4

# Number of top singular values to track
TOP_SINGULAR_VALUES = 10

# Default number of PCA components for cluster analysis
DEFAULT_CLUSTER_COMPONENTS = 10


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class EpochStatistics:
    """Statistics for a single epoch.

    Contains all computed metrics for one training epoch,
    aggregated across layers and memory matrices.

    Attributes:
        epoch: Epoch number.
        step: Training step number.
        magnitude_mean: Mean L2 magnitude of memory vectors.
        magnitude_std: Standard deviation of magnitudes.
        magnitude_min: Minimum magnitude.
        magnitude_max: Maximum magnitude.
        sparsity: Overall sparsity (fraction near-zero).
        m_sparsity: M matrix sparsity.
        s_sparsity: S matrix sparsity.
        rank: Average effective rank.
        m_rank: M matrix effective rank (mean across layers).
        s_rank: S matrix effective rank (mean across layers).
        frobenius_norm: Mean Frobenius norm.
        top_singular_values: Top singular values (averaged).
        num_layers: Number of layers analyzed.
        analysis_time_seconds: Time taken for analysis.
        metadata: Additional metadata.
    """

    epoch: int
    step: int
    magnitude_mean: float
    magnitude_std: float
    magnitude_min: float
    magnitude_max: float
    sparsity: float
    m_sparsity: float
    s_sparsity: float
    rank: float
    m_rank: float
    s_rank: float
    frobenius_norm: float
    top_singular_values: NDArray[np.floating[Any]]
    num_layers: int
    analysis_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def combined_rank(self) -> float:
        """Combined effective rank (average of M and S)."""
        return (self.m_rank + self.s_rank) / 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "magnitude_mean": self.magnitude_mean,
            "magnitude_std": self.magnitude_std,
            "magnitude_min": self.magnitude_min,
            "magnitude_max": self.magnitude_max,
            "sparsity": self.sparsity,
            "m_sparsity": self.m_sparsity,
            "s_sparsity": self.s_sparsity,
            "rank": self.rank,
            "m_rank": self.m_rank,
            "s_rank": self.s_rank,
            "frobenius_norm": self.frobenius_norm,
            "top_singular_values": self.top_singular_values.tolist(),
            "num_layers": self.num_layers,
            "analysis_time_seconds": self.analysis_time_seconds,
            "metadata": self.metadata,
        }

    def to_csv_row(self) -> dict[str, Any]:
        """Convert to CSV row dictionary with expected columns."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "magnitude_mean": f"{self.magnitude_mean:.6f}",
            "magnitude_std": f"{self.magnitude_std:.6f}",
            "sparsity": f"{self.sparsity:.6f}",
            "rank": f"{self.rank:.2f}",
            "m_rank": f"{self.m_rank:.2f}",
            "s_rank": f"{self.s_rank:.2f}",
            "m_sparsity": f"{self.m_sparsity:.6f}",
            "s_sparsity": f"{self.s_sparsity:.6f}",
            "frobenius_norm": f"{self.frobenius_norm:.6f}",
            "num_layers": self.num_layers,
        }


@dataclass
class ClusterStabilityMetrics:
    """Cluster stability metrics across epochs.

    Measures how consistently concepts cluster together
    as training progresses.

    Attributes:
        epoch_pair: Tuple of (epoch_a, epoch_b).
        cluster_similarity: Adjusted Rand Index between clusterings.
        centroid_drift: Mean distance between cluster centroids.
        membership_stability: Fraction of samples staying in same cluster.
        n_clusters_a: Number of clusters in epoch_a.
        n_clusters_b: Number of clusters in epoch_b.
    """

    epoch_pair: tuple[int, int]
    cluster_similarity: float
    centroid_drift: float
    membership_stability: float
    n_clusters_a: int
    n_clusters_b: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epoch_pair": list(self.epoch_pair),
            "cluster_similarity": self.cluster_similarity,
            "centroid_drift": self.centroid_drift,
            "membership_stability": self.membership_stability,
            "n_clusters_a": self.n_clusters_a,
            "n_clusters_b": self.n_clusters_b,
        }


@dataclass
class TrendAnalysis:
    """Statistical trend analysis results.

    Linear regression and correlation analysis for metric evolution.

    Attributes:
        metric_name: Name of the analyzed metric.
        slope: Linear regression slope (rate of change per epoch).
        intercept: Linear regression intercept.
        r_squared: Coefficient of determination.
        p_value: Statistical significance of trend.
        correlation: Pearson correlation with epoch.
        direction: 'increasing', 'decreasing', or 'stable'.
    """

    metric_name: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    correlation: float
    direction: str

    @property
    def is_significant(self) -> bool:
        """Check if trend is statistically significant (p < 0.05)."""
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "p_value": self.p_value,
            "correlation": self.correlation,
            "direction": self.direction,
            "is_significant": self.is_significant,
        }


@dataclass
class OutlierDetectionResult:
    """Results from outlier detection.

    Identifies epochs with unusual statistical properties.

    Attributes:
        outlier_epochs: List of epochs flagged as outliers.
        outlier_metrics: Dict mapping epoch to list of outlier metrics.
        z_scores: Dict mapping epoch to metric z-scores.
        iqr_outliers: Epochs flagged by IQR method.
        z_outliers: Epochs flagged by z-score method.
    """

    outlier_epochs: list[int]
    outlier_metrics: dict[int, list[str]]
    z_scores: dict[int, dict[str, float]]
    iqr_outliers: list[int]
    z_outliers: list[int]

    @property
    def has_outliers(self) -> bool:
        """Check if any outliers were detected."""
        return len(self.outlier_epochs) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "outlier_epochs": self.outlier_epochs,
            "outlier_metrics": {str(k): v for k, v in self.outlier_metrics.items()},
            "iqr_outliers": self.iqr_outliers,
            "z_outliers": self.z_outliers,
            "has_outliers": self.has_outliers,
        }


@dataclass
class AtlasStatisticsResult:
    """Complete statistical analysis result.

    Aggregates all statistical analysis for a set of epochs.

    Attributes:
        epochs: List of analyzed epochs.
        epoch_statistics: Dict mapping epoch to EpochStatistics.
        trends: Dict of TrendAnalysis for each metric.
        outliers: OutlierDetectionResult.
        cluster_stability: List of ClusterStabilityMetrics (if computed).
        summary_statistics: Aggregated summary statistics.
        total_analysis_time_seconds: Total analysis time.
        metadata: Additional metadata.
    """

    epochs: list[int]
    epoch_statistics: dict[int, EpochStatistics]
    trends: dict[str, TrendAnalysis]
    outliers: OutlierDetectionResult
    cluster_stability: list[ClusterStabilityMetrics]
    summary_statistics: dict[str, Any]
    total_analysis_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        """Number of epochs analyzed."""
        return len(self.epochs)

    def export_csv(self, output_path: str | Path) -> None:
        """Export epoch statistics to CSV.

        Creates CSV with columns: epoch, magnitude_mean, sparsity, rank
        (plus additional metrics).

        Parameters:
            output_path: Path for CSV output.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns (required columns first)
        columns = [
            "epoch",
            "magnitude_mean",
            "sparsity",
            "rank",
            "step",
            "magnitude_std",
            "m_sparsity",
            "s_sparsity",
            "m_rank",
            "s_rank",
            "frobenius_norm",
            "num_layers",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for epoch in sorted(self.epochs):
                stats = self.epoch_statistics[epoch]
                row = stats.to_csv_row()
                # Filter to only columns we want
                writer.writerow({k: row.get(k, "") for k in columns})

        logger.info("CSV exported with columns: epoch, magnitude_mean, sparsity, rank")

    def export_json(self, output_path: str | Path) -> None:
        """Export full results to JSON.

        Parameters:
            output_path: Path for JSON output.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "epochs": self.epochs,
            "n_epochs": self.n_epochs,
            "epoch_statistics": {
                str(e): s.to_dict() for e, s in self.epoch_statistics.items()
            },
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "outliers": self.outliers.to_dict(),
            "cluster_stability": [cs.to_dict() for cs in self.cluster_stability],
            "summary_statistics": self.summary_statistics,
            "total_analysis_time_seconds": self.total_analysis_time_seconds,
            "metadata": self.metadata,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON exported to {output_path}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epochs": self.epochs,
            "n_epochs": self.n_epochs,
            "epoch_statistics": {
                str(e): s.to_dict() for e, s in self.epoch_statistics.items()
            },
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "outliers": self.outliers.to_dict(),
            "cluster_stability": [cs.to_dict() for cs in self.cluster_stability],
            "summary_statistics": self.summary_statistics,
            "total_analysis_time_seconds": self.total_analysis_time_seconds,
            "metadata": self.metadata,
        }


# ============================================================================
# Core Analysis Functions
# ============================================================================


def compute_epoch_statistics(
    memory_states: list[Any],
    epoch: int = 0,
    step: int = 0,
) -> EpochStatistics:
    """Compute comprehensive statistics for a single epoch.

    Parameters:
        memory_states: List of memory states (dicts with 'M'/'S' or tuples).
        epoch: Epoch number.
        step: Training step number.

    Returns:
        EpochStatistics with all computed metrics.
    """
    start_time = time.time()

    # Collect metrics across layers
    m_magnitudes: list[float] = []
    s_magnitudes: list[float] = []
    m_sparsities: list[float] = []
    s_sparsities: list[float] = []
    m_ranks: list[int] = []
    s_ranks: list[int] = []
    m_frobenius: list[float] = []
    s_frobenius: list[float] = []
    all_top_sv: list[NDArray[np.floating[Any]]] = []

    for _layer_idx, layer_state in enumerate(memory_states):
        # Extract M and S matrices
        m_matrix, s_matrix = _extract_matrices(layer_state)
        if m_matrix is None or s_matrix is None:
            continue

        # Compute M matrix statistics
        m_stats = _compute_matrix_statistics(m_matrix)
        m_magnitudes.append(m_stats["magnitude_mean"])
        m_sparsities.append(m_stats["sparsity"])
        m_ranks.append(m_stats["effective_rank"])
        m_frobenius.append(m_stats["frobenius_norm"])
        all_top_sv.append(m_stats["top_singular_values"])

        # Compute S matrix statistics
        s_stats = _compute_matrix_statistics(s_matrix)
        s_magnitudes.append(s_stats["magnitude_mean"])
        s_sparsities.append(s_stats["sparsity"])
        s_ranks.append(s_stats["effective_rank"])
        s_frobenius.append(s_stats["frobenius_norm"])

    if not m_magnitudes:
        # Return empty statistics if no valid layers
        return EpochStatistics(
            epoch=epoch,
            step=step,
            magnitude_mean=0.0,
            magnitude_std=0.0,
            magnitude_min=0.0,
            magnitude_max=0.0,
            sparsity=1.0,
            m_sparsity=1.0,
            s_sparsity=1.0,
            rank=0.0,
            m_rank=0.0,
            s_rank=0.0,
            frobenius_norm=0.0,
            top_singular_values=np.zeros(TOP_SINGULAR_VALUES),
            num_layers=0,
            analysis_time_seconds=time.time() - start_time,
            metadata={"warning": "no_valid_layers"},
        )

    # Aggregate statistics
    # Note: Using M matrix as primary for magnitude metrics
    magnitude_mean = float(np.mean(m_magnitudes))
    magnitude_std = float(np.std(m_magnitudes))
    magnitude_min = float(np.min(m_magnitudes))
    magnitude_max = float(np.max(m_magnitudes))

    m_sparsity_mean = float(np.mean(m_sparsities))
    s_sparsity_mean = float(np.mean(s_sparsities))
    overall_sparsity = (m_sparsity_mean + s_sparsity_mean) / 2

    m_rank_mean = float(np.mean(m_ranks))
    s_rank_mean = float(np.mean(s_ranks))
    overall_rank = (m_rank_mean + s_rank_mean) / 2

    frobenius_mean = float(np.mean(m_frobenius + s_frobenius))

    # Average top singular values
    if all_top_sv:
        stacked_sv = np.stack(all_top_sv)
        avg_top_sv = np.mean(stacked_sv, axis=0)
    else:
        avg_top_sv = np.zeros(TOP_SINGULAR_VALUES)

    analysis_time = time.time() - start_time

    return EpochStatistics(
        epoch=epoch,
        step=step,
        magnitude_mean=magnitude_mean,
        magnitude_std=magnitude_std,
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
        sparsity=overall_sparsity,
        m_sparsity=m_sparsity_mean,
        s_sparsity=s_sparsity_mean,
        rank=overall_rank,
        m_rank=m_rank_mean,
        s_rank=s_rank_mean,
        frobenius_norm=frobenius_mean,
        top_singular_values=avg_top_sv,
        num_layers=len(m_magnitudes),
        analysis_time_seconds=analysis_time,
    )


def _extract_matrices(layer_state: Any) -> tuple[NDArray[np.floating[Any]] | None, NDArray[np.floating[Any]] | None]:
    """Extract M and S matrices from layer state.

    Parameters:
        layer_state: Dict with 'M'/'S' keys or (M, S) tuple.

    Returns:
        Tuple of (M_matrix, S_matrix) as numpy arrays.
    """
    m_matrix = None
    s_matrix = None

    if isinstance(layer_state, dict):
        m_tensor = layer_state.get("M")
        s_tensor = layer_state.get("S")
    elif isinstance(layer_state, tuple) and len(layer_state) == 2:
        m_tensor, s_tensor = layer_state
    else:
        return None, None

    m_matrix = _to_numpy(m_tensor)
    s_matrix = _to_numpy(s_tensor)

    return m_matrix, s_matrix


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


def _compute_matrix_statistics(matrix: NDArray[np.floating[Any]]) -> dict[str, Any]:
    """Compute statistics for a single matrix.

    Parameters:
        matrix: Memory matrix (any dimensionality).

    Returns:
        Dict with magnitude, sparsity, rank, etc.
    """
    # Flatten to 2D for analysis
    if matrix.ndim == 3:
        matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    elif matrix.ndim == 2:
        matrix_2d = matrix
    else:
        matrix_2d = matrix.reshape(-1, 1)

    # Row-wise L2 norms (magnitude)
    row_norms = np.linalg.norm(matrix_2d, axis=1)
    magnitude_mean = float(np.mean(row_norms))
    magnitude_std = float(np.std(row_norms))

    # Frobenius norm
    frobenius_norm = float(np.linalg.norm(matrix_2d, "fro"))

    # Sparsity (fraction near zero)
    abs_values = np.abs(matrix_2d)
    near_zero_count = np.sum(abs_values < SPARSITY_THRESHOLD)
    sparsity = float(near_zero_count / matrix_2d.size)

    # Effective rank via SVD
    try:
        # Sample for efficiency
        if matrix_2d.shape[0] > 1000:
            indices = np.random.choice(matrix_2d.shape[0], 1000, replace=False)
            sample = matrix_2d[indices]
        else:
            sample = matrix_2d

        from scipy import linalg
        singular_values = linalg.svdvals(sample)

        # Normalize by largest
        if singular_values[0] > 0:
            normalized_sv = singular_values / singular_values[0]
        else:
            normalized_sv = singular_values

        effective_rank = int(np.sum(normalized_sv > RANK_THRESHOLD))
        top_sv = np.zeros(TOP_SINGULAR_VALUES)
        top_sv[:min(len(singular_values), TOP_SINGULAR_VALUES)] = singular_values[:TOP_SINGULAR_VALUES]

    except Exception as e:
        logger.warning(f"SVD failed: {e}")
        effective_rank = 0
        top_sv = np.zeros(TOP_SINGULAR_VALUES)

    return {
        "magnitude_mean": magnitude_mean,
        "magnitude_std": magnitude_std,
        "frobenius_norm": frobenius_norm,
        "sparsity": sparsity,
        "effective_rank": effective_rank,
        "top_singular_values": top_sv,
    }


# ============================================================================
# Trend Analysis
# ============================================================================


def analyze_trends(
    epoch_statistics: dict[int, EpochStatistics],
) -> dict[str, TrendAnalysis]:
    """Analyze trends in metrics across epochs.

    Performs linear regression and correlation analysis for key metrics.

    Parameters:
        epoch_statistics: Dict mapping epoch to EpochStatistics.

    Returns:
        Dict mapping metric name to TrendAnalysis.
    """
    epochs = sorted(epoch_statistics.keys())
    if len(epochs) < MIN_EPOCHS_FOR_STATS:
        return {}

    epoch_array = np.array(epochs, dtype=float)

    metrics_to_analyze = [
        ("magnitude_mean", [epoch_statistics[e].magnitude_mean for e in epochs]),
        ("sparsity", [epoch_statistics[e].sparsity for e in epochs]),
        ("rank", [epoch_statistics[e].rank for e in epochs]),
        ("m_sparsity", [epoch_statistics[e].m_sparsity for e in epochs]),
        ("s_sparsity", [epoch_statistics[e].s_sparsity for e in epochs]),
        ("m_rank", [epoch_statistics[e].m_rank for e in epochs]),
        ("s_rank", [epoch_statistics[e].s_rank for e in epochs]),
        ("frobenius_norm", [epoch_statistics[e].frobenius_norm for e in epochs]),
    ]

    trends = {}
    for metric_name, values in metrics_to_analyze:
        values_array = np.array(values, dtype=float)

        # Skip if no variation
        if np.std(values_array) < 1e-10:
            trends[metric_name] = TrendAnalysis(
                metric_name=metric_name,
                slope=0.0,
                intercept=float(np.mean(values_array)),
                r_squared=0.0,
                p_value=1.0,
                correlation=0.0,
                direction="stable",
            )
            continue

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            epoch_array, values_array
        )

        # Determine direction
        if abs(slope) < 1e-8:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        trends[metric_name] = TrendAnalysis(
            metric_name=metric_name,
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_value ** 2),
            p_value=float(p_value),
            correlation=float(r_value),
            direction=direction,
        )

    return trends


# ============================================================================
# Outlier Detection
# ============================================================================


def detect_outliers(
    epoch_statistics: dict[int, EpochStatistics],
    z_threshold: float = OUTLIER_Z_THRESHOLD,
    iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
) -> OutlierDetectionResult:
    """Detect outlier epochs using z-score and IQR methods.

    Parameters:
        epoch_statistics: Dict mapping epoch to EpochStatistics.
        z_threshold: Z-score threshold for outlier detection.
        iqr_multiplier: IQR multiplier for outlier detection.

    Returns:
        OutlierDetectionResult with flagged epochs.
    """
    epochs = sorted(epoch_statistics.keys())
    if len(epochs) < MIN_EPOCHS_FOR_STATS:
        return OutlierDetectionResult(
            outlier_epochs=[],
            outlier_metrics={},
            z_scores={},
            iqr_outliers=[],
            z_outliers=[],
        )

    metrics_to_check = [
        ("magnitude_mean", [epoch_statistics[e].magnitude_mean for e in epochs]),
        ("sparsity", [epoch_statistics[e].sparsity for e in epochs]),
        ("rank", [epoch_statistics[e].rank for e in epochs]),
    ]

    z_scores: dict[int, dict[str, float]] = {e: {} for e in epochs}
    z_outliers_set: set[int] = set()
    iqr_outliers_set: set[int] = set()
    outlier_metrics: dict[int, list[str]] = {e: [] for e in epochs}

    for metric_name, values in metrics_to_check:
        values_array = np.array(values, dtype=float)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)

        # Z-score method
        if std_val > 1e-10:
            z_vals = (values_array - mean_val) / std_val
            for i, epoch in enumerate(epochs):
                z_scores[epoch][metric_name] = float(z_vals[i])
                if abs(z_vals[i]) > z_threshold:
                    z_outliers_set.add(epoch)
                    outlier_metrics[epoch].append(metric_name)

        # IQR method
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        for i, epoch in enumerate(epochs):
            if values_array[i] < lower_bound or values_array[i] > upper_bound:
                iqr_outliers_set.add(epoch)
                if metric_name not in outlier_metrics[epoch]:
                    outlier_metrics[epoch].append(metric_name)

    # Combine outliers
    all_outliers = sorted(z_outliers_set | iqr_outliers_set)

    # Filter out epochs with no outlier metrics
    outlier_metrics = {e: m for e, m in outlier_metrics.items() if m}

    return OutlierDetectionResult(
        outlier_epochs=all_outliers,
        outlier_metrics=outlier_metrics,
        z_scores=z_scores,
        iqr_outliers=sorted(iqr_outliers_set),
        z_outliers=sorted(z_outliers_set),
    )


# ============================================================================
# Summary Statistics
# ============================================================================


def compute_summary_statistics(
    epoch_statistics: dict[int, EpochStatistics],
) -> dict[str, Any]:
    """Compute summary statistics across all epochs.

    Parameters:
        epoch_statistics: Dict mapping epoch to EpochStatistics.

    Returns:
        Dict with summary statistics.
    """
    if not epoch_statistics:
        return {}

    epochs = sorted(epoch_statistics.keys())

    magnitude_means = [epoch_statistics[e].magnitude_mean for e in epochs]
    sparsities = [epoch_statistics[e].sparsity for e in epochs]
    ranks = [epoch_statistics[e].rank for e in epochs]

    def compute_stats(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    return {
        "n_epochs": len(epochs),
        "epoch_range": [min(epochs), max(epochs)],
        "magnitude": compute_stats(magnitude_means),
        "sparsity": compute_stats(sparsities),
        "rank": compute_stats(ranks),
    }


# ============================================================================
# AtlasStatisticsAnalyzer Class
# ============================================================================


class AtlasStatisticsAnalyzer:
    """High-level interface for Atlas statistical analysis.

    Provides convenient API for analyzing memory statistics across
    multiple epochs with caching and batch analysis support.

    Example:
        analyzer = AtlasStatisticsAnalyzer()
        result = analyzer.analyze_epochs([0, 50, 100, 185])
        result.export_csv("/tmp/metrics.csv")
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        """Initialize the analyzer.

        Parameters:
            checkpoint_dir: Default directory for checkpoints.
            device: Device to load checkpoints to.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = device
        self._cache: dict[int, EpochStatistics] = {}

    def analyze_checkpoint(
        self,
        checkpoint_path: str | Path,
        use_cache: bool = True,
    ) -> EpochStatistics:
        """Analyze a single checkpoint.

        Parameters:
            checkpoint_path: Path to checkpoint file.
            use_cache: Whether to use cached results.

        Returns:
            EpochStatistics for the checkpoint.
        """
        import torch

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)

        if use_cache and epoch in self._cache:
            return self._cache[epoch]

        # Extract and analyze memory states
        memory_states = checkpoint.get("memory_states")
        if memory_states is None:
            raise ValueError(f"Checkpoint has no memory_states: {checkpoint_path}")

        result = compute_epoch_statistics(memory_states, epoch=epoch, step=step)
        result.metadata["checkpoint_path"] = str(checkpoint_path)

        if use_cache:
            self._cache[epoch] = result

        # Clean up
        del checkpoint
        torch.cuda.empty_cache()

        return result

    def analyze_epochs(
        self,
        epochs: list[int],
        checkpoint_dir: str | Path | None = None,
    ) -> AtlasStatisticsResult:
        """Analyze statistics across multiple epochs.

        Parameters:
            epochs: List of epochs to analyze.
            checkpoint_dir: Override default checkpoint directory.

        Returns:
            AtlasStatisticsResult with complete analysis.
        """
        start_time = time.time()

        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
        else:
            checkpoint_dir = Path(checkpoint_dir)

        epoch_statistics: dict[int, EpochStatistics] = {}
        analyzed_epochs: list[int] = []

        # Find checkpoints
        all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        if not all_checkpoints:
            all_checkpoints = sorted(checkpoint_dir.glob("*.pt"))

        logger.info(f"Found {len(all_checkpoints)} checkpoints in {checkpoint_dir}")

        for checkpoint_path in all_checkpoints:
            try:
                stats = self.analyze_checkpoint(checkpoint_path)

                if stats.epoch in epochs:
                    epoch_statistics[stats.epoch] = stats
                    analyzed_epochs.append(stats.epoch)
                    logger.info(
                        f"Analyzed epoch {stats.epoch}: "
                        f"magnitude={stats.magnitude_mean:.4f}, "
                        f"sparsity={stats.sparsity:.4f}, "
                        f"rank={stats.rank:.1f}"
                    )

                # Stop if we have all requested epochs
                if len(epoch_statistics) == len(epochs):
                    break

            except Exception as e:
                logger.warning(f"Failed to analyze {checkpoint_path}: {e}")
                continue

        analyzed_epochs = sorted(analyzed_epochs)

        # Compute trends
        trends = analyze_trends(epoch_statistics)

        # Detect outliers
        outliers = detect_outliers(epoch_statistics)

        # Compute summary statistics
        summary = compute_summary_statistics(epoch_statistics)

        total_time = time.time() - start_time

        return AtlasStatisticsResult(
            epochs=analyzed_epochs,
            epoch_statistics=epoch_statistics,
            trends=trends,
            outliers=outliers,
            cluster_stability=[],  # Computed separately if needed
            summary_statistics=summary,
            total_analysis_time_seconds=total_time,
            metadata={
                "checkpoint_dir": str(checkpoint_dir),
                "requested_epochs": epochs,
                "device": self.device,
            },
        )

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface for Atlas statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Statistical Analysis for Atlas model memory tracing"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        required=False,
        help="Comma-separated list of epochs to analyze (e.g., '0,50,100,185')",
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
        required=False,
        help="Output path for CSV export",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output path for JSON export",
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

    if not args.epochs:
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.analysis.atlas_statistics --epochs 0,50,100,185 --output /tmp/metrics.csv")
        print("  python -m src.analysis.atlas_statistics --test")
        return

    # Parse epochs
    epochs = [int(e.strip()) for e in args.epochs.split(",")]
    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR

    try:
        analyzer = AtlasStatisticsAnalyzer(
            checkpoint_dir=checkpoint_dir,
            device=args.device,
        )

        result = analyzer.analyze_epochs(epochs)

        # Print summary
        print("\n=== Atlas Statistical Analysis ===")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Epochs analyzed: {result.epochs}")
        print("\nSummary Statistics:")
        if "magnitude" in result.summary_statistics:
            mag = result.summary_statistics["magnitude"]
            print(f"  Magnitude: mean={mag['mean']:.4f}, std={mag['std']:.4f}")
        if "sparsity" in result.summary_statistics:
            spar = result.summary_statistics["sparsity"]
            print(f"  Sparsity: mean={spar['mean']:.4f}, std={spar['std']:.4f}")
        if "rank" in result.summary_statistics:
            rk = result.summary_statistics["rank"]
            print(f"  Rank: mean={rk['mean']:.1f}, std={rk['std']:.1f}")

        # Print trends
        if result.trends:
            print("\nTrends:")
            for metric_name, trend in result.trends.items():
                sig = "*" if trend.is_significant else ""
                print(f"  {metric_name}: {trend.direction} (r={trend.correlation:.3f}{sig})")

        # Print outliers
        if result.outliers.has_outliers:
            print(f"\nOutlier epochs: {result.outliers.outlier_epochs}")
            for epoch, metrics in result.outliers.outlier_metrics.items():
                print(f"  Epoch {epoch}: {', '.join(metrics)}")

        print(f"\nTotal analysis time: {result.total_analysis_time_seconds:.2f}s")

        # Export CSV
        if args.output:
            result.export_csv(args.output)
            print("\nCSV exported with columns: epoch, magnitude_mean, sparsity, rank")

        # Export JSON
        if args.output_json:
            result.export_json(args.output_json)

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_synthetic_test(output_path: str | None = None) -> None:
    """Run test with synthetic memory data."""
    print("Running synthetic test (no checkpoints needed)...")

    np.random.seed(42)

    # Simulate 4 epochs with varying characteristics
    epochs = [0, 50, 100, 185]
    epoch_statistics: dict[int, EpochStatistics] = {}

    for epoch in epochs:
        # Simulate memory states (4 layers)
        n_layers = 4
        batch_size = 8
        memory_slots = 100
        memory_dim = 64

        # Varying characteristics by epoch
        base_sparsity = 0.3 + 0.3 * (epoch / 185)  # Increasing sparsity
        base_magnitude = 1.0 - 0.3 * (epoch / 185)  # Decreasing magnitude

        memory_states = []
        for layer in range(n_layers):
            layer_sparsity = base_sparsity + 0.1 * layer
            m_matrix = np.random.randn(batch_size, memory_slots, memory_dim) * base_magnitude
            m_matrix[np.random.rand(*m_matrix.shape) < layer_sparsity] = 0

            s_matrix = np.random.randn(batch_size, memory_slots, memory_dim) * 0.5 * base_magnitude
            s_matrix[np.random.rand(*s_matrix.shape) < layer_sparsity] = 0

            memory_states.append({"M": m_matrix, "S": s_matrix})

        stats = compute_epoch_statistics(memory_states, epoch=epoch, step=epoch * 50)
        epoch_statistics[epoch] = stats

        print(
            f"Epoch {epoch}: magnitude={stats.magnitude_mean:.4f}, "
            f"sparsity={stats.sparsity:.4f}, rank={stats.rank:.1f}"
        )

    # Compute trends and outliers
    trends = analyze_trends(epoch_statistics)
    outliers = detect_outliers(epoch_statistics)
    summary = compute_summary_statistics(epoch_statistics)

    # Create result
    result = AtlasStatisticsResult(
        epochs=epochs,
        epoch_statistics=epoch_statistics,
        trends=trends,
        outliers=outliers,
        cluster_stability=[],
        summary_statistics=summary,
        total_analysis_time_seconds=0.1,
        metadata={"source": "synthetic_test"},
    )

    print("\n=== Synthetic Test Results ===")
    print(f"Epochs analyzed: {result.epochs}")

    if result.trends:
        print("\nTrends:")
        for metric_name, trend in result.trends.items():
            sig = "*" if trend.is_significant else ""
            print(f"  {metric_name}: {trend.direction} (r={trend.correlation:.3f}{sig})")

    if result.outliers.has_outliers:
        print(f"\nOutlier epochs: {result.outliers.outlier_epochs}")

    if output_path:
        result.export_csv(output_path)
        print("\nCSV exported with columns: epoch, magnitude_mean, sparsity, rank")
    else:
        print("\nNo output path specified, skipping CSV export")
        print("Use --output <path.csv> to export CSV")

    print("\nSynthetic test completed successfully!")


if __name__ == "__main__":
    main()
