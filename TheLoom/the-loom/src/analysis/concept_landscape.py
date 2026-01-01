"""Concept Landscape Analysis for Cross-Epoch Embedding Evolution.

This module implements the Concept Landscape Analysis pipeline for tracking
how concept embeddings evolve during Atlas model training. It provides tools
for extracting, aligning, and visualizing embedding trajectories across epochs.

DESIGN RATIONALE
================
During training, embeddings evolve from random initializations toward their
final (epoch 186) configuration. The Concept Landscape captures this evolution
by:

1. Loading checkpoints sequentially (memory-efficient pattern)
2. Extracting embeddings from each checkpoint
3. Projecting all epochs into a consistent PCA space (epoch 186 reference)
4. Generating interactive 3D visualizations of concept trajectories

This enables researchers to:
- Observe how concepts "find their place" in the embedding space
- Identify epochs where major reorganization occurs
- Track cluster formation and stability over training
- Correlate embedding changes with training metrics

MEMORY MANAGEMENT
=================
Atlas checkpoints are ~450MB each. Loading 186 checkpoints simultaneously
would require ~84GB memory. This module implements sequential load-extract-
unload to maintain <5GB GPU memory:

1. Load checkpoint N
2. Extract embeddings (run forward pass on sample text)
3. Store extracted embeddings (~few MB)
4. Unload checkpoint (del + empty_cache)
5. Repeat for checkpoint N+1

MATHEMATICAL GROUNDING
======================
- Aligned PCA: Projects all epochs into a consistent coordinate system
  defined by epoch 186's principal components. See aligned_pca.py for details.
- Trajectory analysis: Euclidean distances in PCA space approximate
  geodesic distances on the embedding manifold for nearby points.
- Clustering: DBSCAN in PCA space detects concept clusters evolving over time.

Integration: Works with AtlasLoader and AlignedPCA from TheLoom.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default reference epoch as specified in PRD v1.1
DEFAULT_REFERENCE_EPOCH = 186

# Default checkpoint directory (from environment, no hardcoded fallback)
DEFAULT_CHECKPOINT_DIR = os.environ.get("ATLAS_CHECKPOINT_DIR", "")

# Number of PCA components for visualization (3 for 3D plots)
VISUALIZATION_COMPONENTS = 3

# Full PCA components for analysis
ANALYSIS_COMPONENTS = 50

# Sample texts for embedding extraction (representative of training distribution)
DEFAULT_SAMPLE_TEXTS = [
    "To be, or not to be, that is the question:",
    "All the world's a stage, and all the men and women merely players:",
    "What's in a name? That which we call a rose",
    "The quality of mercy is not strained",
    "Now is the winter of our discontent",
    "Friends, Romans, countrymen, lend me your ears;",
    "The fault, dear Brutus, is not in our stars,",
    "Some are born great, some achieve greatness,",
    "If music be the food of love, play on;",
    "We are such stuff as dreams are made on,",
]

# Minimum epochs required for meaningful landscape analysis
MIN_EPOCHS_FOR_ANALYSIS = 2


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class EpochEmbeddings:
    """Embeddings extracted from a single epoch checkpoint.

    Attributes:
        epoch: The epoch number this data came from.
        step: The training step within the epoch.
        embeddings: Raw embeddings before PCA. Shape (n_samples, n_features).
        sample_texts: The text samples used to generate embeddings.
        extraction_time_seconds: Time taken to extract embeddings.
        metadata: Additional information about the extraction.
    """

    epoch: int
    step: int
    embeddings: NDArray[np.floating[Any]]
    sample_texts: list[str]
    extraction_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of embedding samples."""
        return self.embeddings.shape[0]

    @property
    def n_features(self) -> int:
        """Embedding dimensionality."""
        return self.embeddings.shape[1]


@dataclass
class ConceptLandscapeResult:
    """Complete results from concept landscape analysis.

    Contains aligned embeddings across all epochs plus diagnostic
    information and visualization data.

    Attributes:
        epochs: List of epoch numbers analyzed, in order.
        aligned_embeddings: Dict mapping epoch to aligned embeddings.
            Each array has shape (n_samples, n_components).
        reference_epoch: The epoch used as PCA reference (186).
        variance_explained_per_epoch: Dict of variance explained by epoch.
        convergence_distances: Mean distance to final epoch per epoch.
        cluster_labels_per_epoch: Optional cluster assignments per epoch.
        sample_texts: The text samples used for embeddings.
        total_analysis_time_seconds: Total time for full analysis.
        metadata: Additional diagnostic information.
    """

    epochs: list[int]
    aligned_embeddings: dict[int, NDArray[np.floating[Any]]]
    reference_epoch: int
    variance_explained_per_epoch: dict[int, float]
    convergence_distances: dict[int, float]
    cluster_labels_per_epoch: dict[int, NDArray[np.int_]] | None
    sample_texts: list[str]
    total_analysis_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        """Number of epochs analyzed."""
        return len(self.epochs)

    @property
    def n_samples(self) -> int:
        """Number of samples per epoch."""
        if not self.aligned_embeddings:
            return 0
        first_epoch = next(iter(self.aligned_embeddings))
        return self.aligned_embeddings[first_epoch].shape[0]

    @property
    def n_components(self) -> int:
        """Number of PCA components."""
        if not self.aligned_embeddings:
            return 0
        first_epoch = next(iter(self.aligned_embeddings))
        return self.aligned_embeddings[first_epoch].shape[1]

    def get_trajectory(self, sample_idx: int) -> NDArray[np.floating[Any]]:
        """Get the trajectory of a single sample across all epochs.

        Parameters:
            sample_idx: Index of the sample to track.

        Returns:
            Array of shape (n_epochs, n_components) with the sample's position
            at each epoch.
        """
        trajectory = []
        for epoch in self.epochs:
            trajectory.append(self.aligned_embeddings[epoch][sample_idx])
        return np.stack(trajectory)

    def get_epoch_3d(self, epoch: int) -> NDArray[np.floating[Any]]:
        """Get first 3 PCA components for an epoch (for 3D visualization).

        Parameters:
            epoch: Epoch number.

        Returns:
            Array of shape (n_samples, 3).
        """
        return self.aligned_embeddings[epoch][:, :3]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (without large arrays)."""
        return {
            "epochs": self.epochs,
            "n_epochs": self.n_epochs,
            "n_samples": self.n_samples,
            "n_components": self.n_components,
            "reference_epoch": self.reference_epoch,
            "variance_explained_per_epoch": self.variance_explained_per_epoch,
            "convergence_distances": self.convergence_distances,
            "sample_texts": self.sample_texts,
            "total_analysis_time_seconds": self.total_analysis_time_seconds,
            "metadata": self.metadata,
        }


@dataclass
class VisualizationConfig:
    """Configuration for landscape visualization.

    Attributes:
        title: Title for the visualization.
        axis_range: Fixed axis range for consistent animation.
        marker_size: Size of scatter markers.
        colormap: Colormap for epoch coloring.
        animation_duration: Frame duration in milliseconds.
        show_trajectories: Whether to draw sample trajectories.
        output_html: Path for HTML output (interactive).
        output_png: Path for PNG output (static, 300 DPI).
    """

    title: str = "Concept Landscape Evolution"
    axis_range: tuple[float, float] = (-10.0, 10.0)
    marker_size: int = 4
    colormap: str = "viridis"
    animation_duration: int = 500
    show_trajectories: bool = False
    output_html: str | None = None
    output_png: str | None = None


# ============================================================================
# Embedding Extraction
# ============================================================================


def extract_embeddings_from_checkpoint(
    checkpoint_path: str | Path,
    sample_texts: list[str] | None = None,
    device: str = "cuda:0",
    pooling: str = "mean",
) -> EpochEmbeddings:
    """Extract embeddings from a single Atlas checkpoint.

    Loads the checkpoint, runs forward passes on sample texts, and
    extracts the final layer embeddings. Checkpoint is unloaded after
    extraction to conserve memory.

    Parameters:
        checkpoint_path: Path to the checkpoint file.
        sample_texts: Text samples to embed. Defaults to Shakespeare excerpts.
        device: Device for model execution.
        pooling: Pooling strategy (mean, last_token, first_token).

    Returns:
        EpochEmbeddings with extracted embeddings.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If embedding extraction fails.
    """
    # Lazy import to avoid circular dependencies and for optional GPU usage
    import torch

    from ..loaders.atlas_loader import AtlasLoader

    if sample_texts is None:
        sample_texts = DEFAULT_SAMPLE_TEXTS.copy()

    checkpoint_path = Path(checkpoint_path)
    start_time = time.time()

    logger.info(f"Loading checkpoint: {checkpoint_path.name}")

    # Load model
    loader = AtlasLoader()
    loaded_model = loader.load(str(checkpoint_path), device=device)

    epoch = loaded_model.metadata.get("epoch", 0)
    step = loaded_model.metadata.get("step", 0)

    logger.info(f"Extracting embeddings for epoch {epoch}, step {step}")

    # Extract embeddings
    embeddings_list = []
    with torch.no_grad():
        for text in sample_texts:
            embedding_output = loader.embed(loaded_model, text, pooling=pooling)
            # Move to CPU before converting to numpy (required for GPU tensors)
            embedding = embedding_output.embedding
            if embedding.is_cuda:
                embedding = embedding.cpu()
            embeddings_list.append(embedding.numpy())

    embeddings = np.stack(embeddings_list)

    # Clean up - critical for memory management
    del loaded_model.model
    del loaded_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extraction_time = time.time() - start_time

    return EpochEmbeddings(
        epoch=epoch,
        step=step,
        embeddings=embeddings,
        sample_texts=sample_texts,
        extraction_time_seconds=extraction_time,
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "device": device,
            "pooling": pooling,
        },
    )


def extract_embeddings_batch(
    checkpoint_dir: str | Path,
    epochs: list[int] | None = None,
    sample_texts: list[str] | None = None,
    device: str = "cuda:0",
    pooling: str = "mean",
) -> dict[int, EpochEmbeddings]:
    """Extract embeddings from multiple checkpoints sequentially.

    Implements the sequential load-extract-unload pattern to avoid OOM.
    Each checkpoint is fully unloaded before the next is loaded.

    Parameters:
        checkpoint_dir: Directory containing checkpoint files.
        epochs: Specific epochs to load. None = load all available.
        sample_texts: Text samples to embed.
        device: Device for model execution.
        pooling: Pooling strategy.

    Returns:
        Dict mapping epoch number to EpochEmbeddings.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoints
    all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not all_checkpoints:
        all_checkpoints = sorted(checkpoint_dir.glob("*.pt"))

    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    logger.info(f"Found {len(all_checkpoints)} checkpoints in {checkpoint_dir}")

    # If specific epochs requested, filter checkpoints
    # This requires loading each to check epoch - inefficient but necessary
    # when checkpoint filenames don't contain epoch numbers
    result: dict[int, EpochEmbeddings] = {}

    for checkpoint_path in all_checkpoints:
        try:
            epoch_embeddings = extract_embeddings_from_checkpoint(
                checkpoint_path,
                sample_texts=sample_texts,
                device=device,
                pooling=pooling,
            )

            if epochs is None or epoch_embeddings.epoch in epochs:
                result[epoch_embeddings.epoch] = epoch_embeddings
                logger.info(
                    f"Extracted epoch {epoch_embeddings.epoch} "
                    f"({len(result)}/{len(epochs) if epochs else '?'} complete)"
                )

            # Check if we have all requested epochs
            if epochs is not None and len(result) == len(epochs):
                logger.info("All requested epochs extracted")
                break

        except Exception as e:
            logger.warning(f"Failed to extract from {checkpoint_path}: {e}")
            continue

    return result


# ============================================================================
# Concept Landscape Analysis
# ============================================================================


def analyze_concept_landscape(
    embeddings_by_epoch: dict[int, NDArray[np.floating[Any]]],
    reference_epoch: int = DEFAULT_REFERENCE_EPOCH,
    n_components: int = ANALYSIS_COMPONENTS,
    compute_clusters: bool = True,
    cluster_eps: float | None = None,
    cluster_min_samples: int = 3,
) -> ConceptLandscapeResult:
    """Analyze concept landscape evolution across epochs.

    Applies aligned PCA with the reference epoch (186) defining the
    coordinate system, then computes convergence metrics and optional
    clustering.

    Parameters:
        embeddings_by_epoch: Dict mapping epoch to raw embeddings.
        reference_epoch: Epoch to use as PCA reference (default: 186).
        n_components: Number of PCA components to retain.
        compute_clusters: Whether to compute DBSCAN clusters per epoch.
        cluster_eps: DBSCAN epsilon (auto-computed if None).
        cluster_min_samples: DBSCAN min_samples.

    Returns:
        ConceptLandscapeResult with full analysis.

    Raises:
        ValueError: If reference epoch not in embeddings or <2 epochs.
    """
    from .aligned_pca import AlignedPCA

    start_time = time.time()

    epochs = sorted(embeddings_by_epoch.keys())
    if len(epochs) < MIN_EPOCHS_FOR_ANALYSIS:
        raise ValueError(
            f"Need at least {MIN_EPOCHS_FOR_ANALYSIS} epochs, got {len(epochs)}"
        )

    # Find reference epoch - use specified or latest available
    if reference_epoch not in embeddings_by_epoch:
        actual_reference = max(epochs)
        logger.warning(
            f"Reference epoch {reference_epoch} not in data, using {actual_reference}"
        )
        reference_epoch = actual_reference

    reference_embeddings = embeddings_by_epoch[reference_epoch]

    # Create and fit aligned PCA
    aligner = AlignedPCA(n_components=n_components, reference_epoch=reference_epoch)
    aligner.fit_reference(reference_embeddings, epoch=reference_epoch)

    logger.info(
        f"Fitted PCA on epoch {reference_epoch}: "
        f"{aligner.fit_result.total_variance_explained:.1%} variance explained"
    )

    # Transform all epochs
    aligned_embeddings = {}
    variance_explained_per_epoch = {}

    for epoch in epochs:
        result = aligner.transform(embeddings_by_epoch[epoch], epoch=epoch)
        aligned_embeddings[epoch] = result.transformed_embeddings
        variance_explained_per_epoch[epoch] = result.total_variance_explained

    # Compute convergence distances (distance to reference epoch)
    reference_aligned = aligned_embeddings[reference_epoch]
    convergence_distances = {}
    for epoch in epochs:
        distances = np.linalg.norm(
            aligned_embeddings[epoch] - reference_aligned, axis=1
        )
        convergence_distances[epoch] = float(np.mean(distances))

    # Compute clusters per epoch (optional)
    cluster_labels_per_epoch = None
    if compute_clusters:
        cluster_labels_per_epoch = _compute_clusters_per_epoch(
            aligned_embeddings, eps=cluster_eps, min_samples=cluster_min_samples
        )

    total_time = time.time() - start_time

    return ConceptLandscapeResult(
        epochs=epochs,
        aligned_embeddings=aligned_embeddings,
        reference_epoch=reference_epoch,
        variance_explained_per_epoch=variance_explained_per_epoch,
        convergence_distances=convergence_distances,
        cluster_labels_per_epoch=cluster_labels_per_epoch,
        sample_texts=[],  # Filled in by caller if using EpochEmbeddings
        total_analysis_time_seconds=total_time,
        metadata={
            "n_components": n_components,
            "compute_clusters": compute_clusters,
        },
    )


def _compute_clusters_per_epoch(
    aligned_embeddings: dict[int, NDArray[np.floating[Any]]],
    eps: float | None = None,
    min_samples: int = 3,
) -> dict[int, NDArray[np.int_]]:
    """Compute DBSCAN clusters for each epoch.

    Parameters:
        aligned_embeddings: Dict of aligned embeddings per epoch.
        eps: DBSCAN epsilon. Auto-computed if None.
        min_samples: DBSCAN minimum samples.

    Returns:
        Dict mapping epoch to cluster labels array.
    """
    from scipy.spatial.distance import cdist
    from sklearn.cluster import DBSCAN

    cluster_labels = {}

    for epoch, embeddings in aligned_embeddings.items():
        n_samples = embeddings.shape[0]

        # Auto-compute eps using k-NN heuristic
        local_eps = eps
        if local_eps is None and n_samples > min_samples:
            distances = cdist(embeddings, embeddings)
            np.fill_diagonal(distances, np.inf)
            k = min(min_samples, n_samples - 1)
            kth_distances = np.partition(distances, k, axis=1)[:, k]
            local_eps = float(np.median(kth_distances) * 1.5)

        if local_eps is None:
            local_eps = 1.0

        clustering = DBSCAN(eps=local_eps, min_samples=min_samples)
        labels = clustering.fit_predict(embeddings)
        cluster_labels[epoch] = labels

    return cluster_labels


# ============================================================================
# Visualization
# ============================================================================


def create_landscape_visualization(
    result: ConceptLandscapeResult,
    config: VisualizationConfig | None = None,
) -> Any:
    """Create interactive 3D Plotly visualization of concept landscape.

    Generates an animated 3D scatter plot showing how embeddings move
    through PCA space during training. Fixed axis ranges ensure smooth
    animation playback.

    Parameters:
        result: ConceptLandscapeResult from analysis.
        config: Visualization configuration.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    if config is None:
        config = VisualizationConfig()

    epochs = result.epochs
    n_samples = result.n_samples

    # Compute axis ranges from data if not specified
    if config.axis_range == (-10.0, 10.0):
        # Auto-compute from data
        all_coords = []
        for epoch in epochs:
            emb_3d = result.get_epoch_3d(epoch)
            all_coords.append(emb_3d)
        all_coords = np.concatenate(all_coords, axis=0)
        max_abs = np.max(np.abs(all_coords)) * 1.1  # 10% padding
        axis_range = (-max_abs, max_abs)
    else:
        axis_range = config.axis_range

    # Create sample labels based on text
    if result.sample_texts:
        labels = [f"'{t[:30]}...'" if len(t) > 30 else f"'{t}'" for t in result.sample_texts]
    else:
        labels = [f"Sample {i}" for i in range(n_samples)]

    # Create frames for animation
    frames = []
    for epoch in epochs:
        emb_3d = result.get_epoch_3d(epoch)
        variance_exp = result.variance_explained_per_epoch.get(epoch, 0)
        conv_dist = result.convergence_distances.get(epoch, 0)

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=emb_3d[:, 0],
                    y=emb_3d[:, 1],
                    z=emb_3d[:, 2],
                    mode="markers",
                    marker={
                        "size": config.marker_size,
                        "color": np.arange(n_samples),
                        "colorscale": config.colormap,
                        "opacity": 0.8,
                    },
                    text=labels,
                    hovertemplate=(
                        "%{text}<br>"
                        f"Epoch: {epoch}<br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        "PC3: %{z:.3f}<extra></extra>"
                    ),
                )
            ],
            name=str(epoch),
            layout=go.Layout(
                annotations=[
                    {
                        "text": (
                            f"Epoch {epoch} | "
                            f"Var: {variance_exp:.1%} | "
                            f"Dist to final: {conv_dist:.2f}"
                        ),
                        "x": 0.5,
                        "y": 1.05,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 14},
                    }
                ]
            ),
        )
        frames.append(frame)

    # Initial frame data
    initial_epoch = epochs[0]
    initial_emb = result.get_epoch_3d(initial_epoch)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=initial_emb[:, 0],
                y=initial_emb[:, 1],
                z=initial_emb[:, 2],
                mode="markers",
                marker={
                    "size": config.marker_size,
                    "color": np.arange(n_samples),
                    "colorscale": config.colormap,
                    "opacity": 0.8,
                },
                text=labels,
            )
        ],
        frames=frames,
        layout=go.Layout(
            title={
                "text": config.title,
                "x": 0.5,
                "font": {"size": 20},
            },
            scene={
                "xaxis": {
                    "range": list(axis_range),
                    "title": "PC1",
                    "showbackground": True,
                    "backgroundcolor": "rgb(230, 230, 230)",
                },
                "yaxis": {
                    "range": list(axis_range),
                    "title": "PC2",
                    "showbackground": True,
                    "backgroundcolor": "rgb(230, 230, 230)",
                },
                "zaxis": {
                    "range": list(axis_range),
                    "title": "PC3",
                    "showbackground": True,
                    "backgroundcolor": "rgb(230, 230, 230)",
                },
                "camera": {
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.5},
                },
            },
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0.1,
                    "xanchor": "left",
                    "yanchor": "top",
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": config.animation_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "Epoch: ",
                        "visible": True,
                        "xanchor": "center",
                    },
                    "pad": {"t": 50, "b": 10},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(epoch)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": str(epoch),
                            "method": "animate",
                        }
                        for epoch in epochs
                    ],
                }
            ],
        ),
    )

    # Export if paths specified
    if config.output_html:
        output_path = Path(config.output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs=True)
        logger.info(f"Saved HTML visualization to {output_path}")

    if config.output_png:
        output_path = Path(config.output_png)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(
            str(output_path),
            width=1200,
            height=800,
            scale=2.5,  # ~300 DPI
        )
        logger.info(f"Saved PNG visualization to {output_path}")

    return fig


def create_convergence_plot(
    result: ConceptLandscapeResult,
    output_path: str | Path | None = None,
) -> Any:
    """Create a 2D line plot showing convergence to final epoch.

    Parameters:
        result: ConceptLandscapeResult from analysis.
        output_path: Optional path to save the plot.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    epochs = result.epochs
    distances = [result.convergence_distances[e] for e in epochs]
    variance_exp = [result.variance_explained_per_epoch[e] for e in epochs]

    fig = go.Figure()

    # Convergence distance line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=distances,
            mode="lines+markers",
            name="Distance to Final",
            yaxis="y1",
            line={"color": "blue", "width": 2},
            marker={"size": 6},
        )
    )

    # Variance explained line (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=variance_exp,
            mode="lines+markers",
            name="Variance Explained",
            yaxis="y2",
            line={"color": "green", "width": 2, "dash": "dash"},
            marker={"size": 6},
        )
    )

    fig.update_layout(
        title="Concept Landscape Convergence",
        xaxis={"title": "Epoch"},
        yaxis={
            "title": "Mean Distance to Epoch 186",
            "titlefont": {"color": "blue"},
            "tickfont": {"color": "blue"},
        },
        yaxis2={
            "title": "Variance Explained",
            "titlefont": {"color": "green"},
            "tickfont": {"color": "green"},
            "anchor": "x",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
        },
        legend={"x": 0.5, "y": 1.15, "xanchor": "center", "orientation": "h"},
        hovermode="x unified",
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if str(output_path).endswith(".html"):
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=800, height=500, scale=2)
        logger.info(f"Saved convergence plot to {output_path}")

    return fig


# ============================================================================
# Pipeline Orchestration
# ============================================================================


def run_concept_landscape_pipeline(
    epochs: list[int],
    checkpoint_dir: str | Path | None = None,
    output_html: str | Path | None = None,
    output_png: str | Path | None = None,
    sample_texts: list[str] | None = None,
    device: str = "cuda:0",
    reference_epoch: int = DEFAULT_REFERENCE_EPOCH,
) -> ConceptLandscapeResult:
    """Run the complete concept landscape analysis pipeline.

    Orchestrates the full workflow:
    1. Load checkpoints for specified epochs (sequentially)
    2. Extract embeddings from each checkpoint
    3. Apply aligned PCA with epoch 186 reference
    4. Generate visualizations

    Parameters:
        epochs: List of epoch numbers to analyze.
        checkpoint_dir: Directory containing checkpoints.
        output_html: Path for HTML output (interactive).
        output_png: Path for PNG output (static, 300 DPI).
        sample_texts: Text samples to embed.
        device: Device for model execution.
        reference_epoch: PCA reference epoch.

    Returns:
        ConceptLandscapeResult with complete analysis.
    """
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    checkpoint_dir = Path(checkpoint_dir)

    logger.info(f"Starting concept landscape pipeline for epochs: {epochs}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    start_time = time.time()

    # Step 1: Extract embeddings from each epoch
    logger.info("Step 1: Extracting embeddings from checkpoints...")
    epoch_embeddings = extract_embeddings_batch(
        checkpoint_dir,
        epochs=epochs,
        sample_texts=sample_texts,
        device=device,
    )

    if not epoch_embeddings:
        raise RuntimeError(f"No embeddings extracted. Check checkpoint directory: {checkpoint_dir}")

    # Convert to simple dict of embeddings
    embeddings_by_epoch = {
        epoch: data.embeddings for epoch, data in epoch_embeddings.items()
    }

    # Get sample texts from first epoch
    first_epoch_data = next(iter(epoch_embeddings.values()))
    actual_sample_texts = first_epoch_data.sample_texts

    # Step 2: Run analysis
    logger.info("Step 2: Running aligned PCA analysis...")
    result = analyze_concept_landscape(
        embeddings_by_epoch,
        reference_epoch=reference_epoch,
    )
    result.sample_texts = actual_sample_texts

    # Step 3: Generate visualizations
    if output_html or output_png:
        logger.info("Step 3: Generating visualizations...")
        config = VisualizationConfig(
            output_html=str(output_html) if output_html else None,
            output_png=str(output_png) if output_png else None,
        )
        create_landscape_visualization(result, config)

    total_time = time.time() - start_time
    result.total_analysis_time_seconds = total_time

    logger.info(f"Pipeline completed in {total_time:.1f}s")
    logger.info(f"Analyzed {result.n_epochs} epochs, {result.n_samples} samples")

    return result


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface for concept landscape analysis."""
    parser = argparse.ArgumentParser(
        description="Concept Landscape Analysis for Atlas model training"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        required=True,
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
        default=None,
        help="Output path for HTML visualization",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="Output path for PNG visualization (300 DPI)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for model execution",
    )
    parser.add_argument(
        "--reference-epoch",
        type=int,
        default=DEFAULT_REFERENCE_EPOCH,
        help=f"PCA reference epoch (default: {DEFAULT_REFERENCE_EPOCH})",
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

    # Parse epochs
    try:
        epochs = [int(e.strip()) for e in args.epochs.split(",")]
    except ValueError as e:
        print(f"Error parsing epochs: {e}")
        sys.exit(1)

    if len(epochs) < MIN_EPOCHS_FOR_ANALYSIS:
        print(f"Error: Need at least {MIN_EPOCHS_FOR_ANALYSIS} epochs")
        sys.exit(1)

    # Run pipeline
    try:
        result = run_concept_landscape_pipeline(
            epochs=epochs,
            checkpoint_dir=args.checkpoint_dir,
            output_html=args.output,
            output_png=args.output_png,
            device=args.device,
            reference_epoch=args.reference_epoch,
        )

        print("\n=== Concept Landscape Analysis Results ===")
        print(f"Epochs analyzed: {result.epochs}")
        print(f"Samples: {result.n_samples}")
        print(f"PCA components: {result.n_components}")
        print(f"Reference epoch: {result.reference_epoch}")
        print(f"\nConvergence distances (to epoch {result.reference_epoch}):")
        for epoch in result.epochs:
            dist = result.convergence_distances[epoch]
            var = result.variance_explained_per_epoch[epoch]
            print(f"  Epoch {epoch:3d}: dist={dist:.4f}, var={var:.1%}")

        print(f"\nTotal time: {result.total_analysis_time_seconds:.1f}s")
        print("Generated concept landscape")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_synthetic_test(output_path: str | None = None) -> None:
    """Run test with synthetic data."""
    print("Running synthetic test (no checkpoints needed)...")

    np.random.seed(42)

    # Simulate embedding evolution: starts random, converges to final state
    n_samples = 10
    n_features = 128
    epochs = [0, 50, 100, 186]

    # Epoch 186 is the "final" configuration
    epoch_186 = np.random.randn(n_samples, n_features)

    # Earlier epochs: progressively more distant from final
    embeddings_by_epoch = {
        186: epoch_186,
        100: epoch_186 + np.random.randn(n_samples, n_features) * 0.5,
        50: epoch_186 + np.random.randn(n_samples, n_features) * 1.0,
        0: np.random.randn(n_samples, n_features),  # Random initial
    }

    print(f"Generated synthetic embeddings for epochs: {epochs}")

    # Run analysis
    result = analyze_concept_landscape(
        embeddings_by_epoch,
        reference_epoch=186,
        n_components=ANALYSIS_COMPONENTS,
    )
    result.sample_texts = DEFAULT_SAMPLE_TEXTS[:n_samples]

    print("\n=== Synthetic Test Results ===")
    print(f"Epochs: {result.epochs}")
    print(f"Samples: {result.n_samples}")
    print(f"Components: {result.n_components}")
    print(f"Reference epoch: {result.reference_epoch}")

    print("\nConvergence distances (to epoch 186):")
    for epoch in result.epochs:
        dist = result.convergence_distances[epoch]
        var = result.variance_explained_per_epoch[epoch]
        print(f"  Epoch {epoch:3d}: dist={dist:.4f}, var={var:.1%}")

    # Generate visualization if output path specified
    if output_path:
        print(f"\nGenerating visualization: {output_path}")
        config = VisualizationConfig(output_html=output_path)
        create_landscape_visualization(result, config)
        print(f"Generated concept landscape: {output_path}")
    else:
        print("\nNo output path specified, skipping visualization")
        print("Use --output <path.html> to generate visualization")

    print("\nSynthetic test completed successfully!")


if __name__ == "__main__":
    main()
