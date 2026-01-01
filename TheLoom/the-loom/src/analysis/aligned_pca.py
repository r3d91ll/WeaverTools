"""Aligned PCA for Cross-Epoch Embedding Comparison.

This module implements PCA-based dimensionality reduction with a fixed reference
frame, enabling consistent cross-epoch comparison of transformer embeddings.

DESIGN RATIONALE
================
When tracking how concept embeddings evolve during training, a key challenge is
that standard PCA computes different principal components for each epoch, making
direct comparison impossible. The "aligned PCA" approach solves this by:

1. Fitting PCA on a reference epoch (epoch 186, the final trained epoch)
2. Transforming all other epochs using the SAME principal components
3. This projects all epochs into a consistent coordinate system

This design choice is intentional: we use the FINAL epoch (186) as the reference
because it represents the mature, fully-trained embedding space. Earlier epochs
are projected into this final space to visualize how concepts evolve TOWARD
their final configuration.

Alternative (epoch 0 reference) would show how embeddings DEPART from their
initial configuration - a different but valid perspective. The PRD v1.1
specification requires epoch 186 as reference.

MATHEMATICAL GROUNDING
======================
- Standard PCA: Finds orthogonal directions of maximum variance
- Aligned PCA: Uses reference epoch's principal components for all epochs
- Variance explained: May be lower for non-reference epochs (expected)
- Distance preservation: Relative distances in PCA space are approximately
  preserved, enabling cross-epoch trajectory analysis

USAGE
=====
    from src.analysis.aligned_pca import AlignedPCA

    # Create aligner with reference epoch specification
    aligner = AlignedPCA(n_components=50, reference_epoch=186)

    # Fit on final (reference) epoch embeddings
    aligner.fit_reference(epoch_186_embeddings)

    # Transform any epoch using the reference components
    aligned_epoch_100 = aligner.transform(epoch_100_embeddings)
    aligned_epoch_50 = aligner.transform(epoch_50_embeddings)

    # All aligned embeddings share the same coordinate system
    # and can be directly compared or animated

Integration: Designed to work with AtlasLoader for checkpoint-based analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

# ============================================================================
# Constants
# ============================================================================

# Default reference epoch as specified in PRD v1.1
DEFAULT_REFERENCE_EPOCH = 186

# Default number of PCA components to retain
DEFAULT_N_COMPONENTS = 50

# Target variance explained threshold for quality assessment
VARIANCE_EXPLAINED_THRESHOLD = 0.80


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class AlignedPCAResult:
    """Results from an aligned PCA transformation.

    Contains the transformed embeddings along with diagnostic information
    about the transformation quality.

    Attributes:
        transformed_embeddings: The embeddings projected into the reference
            PCA space. Shape (n_samples, n_components).
        epoch: The epoch number these embeddings came from.
        variance_explained: Variance explained by each component when
            applied to this epoch's data. May differ from reference epoch.
        total_variance_explained: Total variance explained by all components.
            Expected to be highest for reference epoch.
        reconstruction_error: Mean squared reconstruction error.
        metadata: Additional diagnostic information.
    """

    transformed_embeddings: NDArray[np.floating[Any]]
    epoch: int
    variance_explained: NDArray[np.floating[Any]]
    total_variance_explained: float
    reconstruction_error: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples in the transformed embeddings."""
        return self.transformed_embeddings.shape[0]

    @property
    def n_components(self) -> int:
        """Number of PCA components used."""
        return self.transformed_embeddings.shape[1]

    @property
    def quality_assessment(self) -> str:
        """Assess transformation quality based on variance explained."""
        if self.total_variance_explained >= 0.90:
            return "excellent"
        elif self.total_variance_explained >= VARIANCE_EXPLAINED_THRESHOLD:
            return "good"
        elif self.total_variance_explained >= 0.60:
            return "moderate"
        else:
            return "poor"


@dataclass
class AlignedPCAFitResult:
    """Results from fitting the reference PCA.

    Contains information about the reference frame that will be used
    for all subsequent transformations.

    Attributes:
        reference_epoch: The epoch used as reference (should be 186).
        n_components: Number of principal components retained.
        variance_explained: Variance explained by each component.
        total_variance_explained: Total variance explained by all components.
        components: The principal component vectors. Shape (n_components, n_features).
        mean: The mean vector used for centering. Shape (n_features,).
        singular_values: The singular values from SVD.
        metadata: Additional diagnostic information.
    """

    reference_epoch: int
    n_components: int
    variance_explained: NDArray[np.floating[Any]]
    total_variance_explained: float
    components: NDArray[np.floating[Any]]
    mean: NDArray[np.floating[Any]]
    singular_values: NDArray[np.floating[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Original feature dimensionality."""
        return self.components.shape[1]

    @property
    def effective_dimensionality(self) -> int:
        """Number of components needed for 95% variance."""
        cumulative = np.cumsum(self.variance_explained)
        idx = int(np.searchsorted(cumulative, 0.95))
        # Clamp to valid range to avoid off-by-one when 95% is never reached
        idx = min(idx, len(cumulative) - 1)
        return idx + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reference_epoch": self.reference_epoch,
            "n_components": self.n_components,
            "total_variance_explained": float(self.total_variance_explained),
            "effective_dimensionality": self.effective_dimensionality,
            "n_features": self.n_features,
            "metadata": self.metadata,
        }


@dataclass
class CrossEpochTrajectory:
    """Trajectory of embeddings across multiple epochs.

    Represents how a set of concept embeddings (e.g., specific tokens)
    move through the aligned PCA space during training.

    Attributes:
        epochs: List of epoch numbers in trajectory order.
        trajectories: Array of shape (n_epochs, n_samples, n_components)
            containing the aligned embeddings for each epoch.
        sample_indices: Indices of the samples being tracked.
        variance_explained_per_epoch: Total variance explained for each epoch.
        metadata: Additional diagnostic information.
    """

    epochs: list[int]
    trajectories: NDArray[np.floating[Any]]
    sample_indices: list[int] | None
    variance_explained_per_epoch: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        """Number of epochs in the trajectory."""
        return len(self.epochs)

    @property
    def n_samples(self) -> int:
        """Number of samples being tracked."""
        return self.trajectories.shape[1]

    @property
    def n_components(self) -> int:
        """Number of PCA components."""
        return self.trajectories.shape[2]

    def get_epoch_embeddings(self, epoch: int) -> NDArray[np.floating[Any]]:
        """Get embeddings for a specific epoch."""
        idx = self.epochs.index(epoch)
        return self.trajectories[idx]

    def compute_displacement(
        self, from_epoch: int, to_epoch: int
    ) -> NDArray[np.floating[Any]]:
        """Compute displacement vectors between two epochs.

        Returns:
            Array of shape (n_samples, n_components) with displacement vectors.
        """
        from_emb = self.get_epoch_embeddings(from_epoch)
        to_emb = self.get_epoch_embeddings(to_epoch)
        return to_emb - from_emb

    def compute_total_path_length(self) -> NDArray[np.floating[Any]]:
        """Compute total path length for each sample across all epochs.

        Returns:
            Array of shape (n_samples,) with total path lengths.
        """
        total_length = np.zeros(self.n_samples)
        for i in range(1, self.n_epochs):
            displacement = np.linalg.norm(
                self.trajectories[i] - self.trajectories[i - 1], axis=1
            )
            total_length += displacement
        return total_length


# ============================================================================
# Main AlignedPCA Class
# ============================================================================


class AlignedPCA:
    """Aligned PCA for cross-epoch embedding comparison.

    This class implements PCA with a fixed reference frame, enabling consistent
    comparison of embeddings across training epochs. The reference epoch (186,
    the final trained epoch) defines the coordinate system.

    CRITICAL: Use epoch 186 (final) as reference per PRD v1.1, NOT epoch 0.
    This ensures all earlier epochs are projected into the mature embedding
    space, showing how concepts evolve TOWARD their final configuration.

    Attributes:
        n_components: Number of principal components to retain.
        reference_epoch: The epoch used as reference frame (default: 186).
        reference_pca: The fitted PCA object (None until fit_reference called).
        fit_result: Detailed fit results (None until fit_reference called).

    Example:
        >>> aligner = AlignedPCA(n_components=50, reference_epoch=186)
        >>> aligner.fit_reference(final_epoch_embeddings)
        >>> aligned = aligner.transform(earlier_epoch_embeddings)
        >>> print(f"Variance explained: {aligned.total_variance_explained:.2%}")
    """

    def __init__(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        reference_epoch: int = DEFAULT_REFERENCE_EPOCH,
    ) -> None:
        """Initialize AlignedPCA.

        Args:
            n_components: Number of principal components to retain.
                Defaults to 50 for balance of expressiveness and efficiency.
            reference_epoch: The epoch to use as reference frame.
                Defaults to 186 (final epoch) per PRD v1.1 specification.
        """
        self.n_components = n_components
        self.reference_epoch = reference_epoch
        self.reference_pca: PCA | None = None
        self.fit_result: AlignedPCAFitResult | None = None

    def fit_reference(
        self,
        embeddings: NDArray[np.floating[Any]],
        epoch: int | None = None,
    ) -> AlignedPCAFitResult:
        """Fit PCA on reference epoch embeddings to establish reference frame.

        This method fits standard PCA on the provided embeddings, which should
        be from the reference epoch (186 by default). The fitted components
        become the fixed coordinate system for all subsequent transformations.

        Args:
            embeddings: Reference epoch embeddings. Shape (n_samples, n_features).
                Should be from epoch 186 (final) for proper cross-epoch analysis.
            epoch: Optional epoch number for validation. If provided and differs
                from reference_epoch, a warning is issued.

        Returns:
            AlignedPCAFitResult with detailed fit information.

        Raises:
            ValueError: If embeddings have fewer samples than requested components.

        Note:
            Uses regular PCA (not IncrementalPCA) for the fit-once, transform-many
            pattern required by cross-epoch analysis.
        """
        if epoch is not None and epoch != self.reference_epoch:
            warnings.warn(
                f"Fitting on epoch {epoch} but reference_epoch is {self.reference_epoch}. "
                f"For proper cross-epoch analysis, fit on epoch {self.reference_epoch} (final).",
                UserWarning,
                stacklevel=2,
            )

        n_samples, n_features = embeddings.shape

        # Adjust n_components if needed
        actual_n_components = min(self.n_components, n_samples - 1, n_features)
        if actual_n_components < self.n_components:
            warnings.warn(
                f"Reducing n_components from {self.n_components} to {actual_n_components} "
                f"due to data shape ({n_samples} samples, {n_features} features).",
                UserWarning,
                stacklevel=2,
            )

        # Fit PCA - using standard PCA for fit-once, transform-many pattern
        self.reference_pca = PCA(n_components=actual_n_components)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reference_pca.fit(embeddings)

        # Create fit result
        self.fit_result = AlignedPCAFitResult(
            reference_epoch=epoch if epoch is not None else self.reference_epoch,
            n_components=actual_n_components,
            variance_explained=self.reference_pca.explained_variance_ratio_.copy(),
            total_variance_explained=float(
                np.sum(self.reference_pca.explained_variance_ratio_)
            ),
            components=self.reference_pca.components_.copy(),
            mean=self.reference_pca.mean_.copy(),
            singular_values=self.reference_pca.singular_values_.copy(),
            metadata={
                "n_samples_fitted": n_samples,
                "n_features": n_features,
            },
        )

        return self.fit_result

    def transform(
        self,
        embeddings: NDArray[np.floating[Any]],
        epoch: int = 0,
    ) -> AlignedPCAResult:
        """Transform embeddings using the reference PCA for consistent comparison.

        Projects the provided embeddings into the reference epoch's PCA space,
        enabling direct comparison with embeddings from other epochs.

        Args:
            embeddings: Embeddings to transform. Shape (n_samples, n_features).
            epoch: Epoch number for these embeddings (for metadata).

        Returns:
            AlignedPCAResult with transformed embeddings and diagnostics.

        Raises:
            RuntimeError: If fit_reference has not been called.
            ValueError: If embeddings have wrong number of features.
        """
        if self.reference_pca is None:
            raise RuntimeError(
                "Must call fit_reference() before transform(). "
                "Fit on epoch 186 (final) embeddings first."
            )

        n_samples, n_features = embeddings.shape

        if n_features != self.fit_result.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.fit_result.n_features}, "
                f"got {n_features}."
            )

        # Transform using reference PCA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = self.reference_pca.transform(embeddings)

        # Compute variance explained for this epoch's data
        # (will differ from reference epoch's variance explained)
        centered = embeddings - self.reference_pca.mean_
        total_var = np.var(centered, axis=0).sum()

        if total_var > 0:
            projected_var = np.var(transformed, axis=0)
            variance_explained = projected_var / total_var
            total_variance_explained = float(np.sum(projected_var) / total_var)
        else:
            variance_explained = np.zeros(transformed.shape[1])
            total_variance_explained = 0.0

        # Compute reconstruction error
        reconstructed = self.reference_pca.inverse_transform(transformed)
        reconstruction_error = float(np.mean((embeddings - reconstructed) ** 2))

        return AlignedPCAResult(
            transformed_embeddings=transformed,
            epoch=epoch,
            variance_explained=variance_explained,
            total_variance_explained=total_variance_explained,
            reconstruction_error=reconstruction_error,
            metadata={
                "reference_epoch": self.fit_result.reference_epoch,
                "n_samples": n_samples,
            },
        )

    def fit_transform(
        self,
        embeddings: NDArray[np.floating[Any]],
        epoch: int | None = None,
    ) -> AlignedPCAResult:
        """Fit on reference embeddings and return transformed result.

        Convenience method that combines fit_reference and transform.

        Args:
            embeddings: Reference epoch embeddings. Shape (n_samples, n_features).
            epoch: Epoch number (defaults to reference_epoch).

        Returns:
            AlignedPCAResult for the reference epoch.
        """
        actual_epoch = epoch if epoch is not None else self.reference_epoch
        self.fit_reference(embeddings, epoch=actual_epoch)
        return self.transform(embeddings, epoch=actual_epoch)

    def is_fitted(self) -> bool:
        """Check if the reference PCA has been fitted."""
        return self.reference_pca is not None

    def get_components(self) -> NDArray[np.floating[Any]]:
        """Get the principal component vectors.

        Returns:
            Principal components. Shape (n_components, n_features).

        Raises:
            RuntimeError: If not fitted.
        """
        if self.reference_pca is None:
            raise RuntimeError("Must call fit_reference() first.")
        return self.reference_pca.components_.copy()

    def get_mean(self) -> NDArray[np.floating[Any]]:
        """Get the mean vector used for centering.

        Returns:
            Mean vector. Shape (n_features,).

        Raises:
            RuntimeError: If not fitted.
        """
        if self.reference_pca is None:
            raise RuntimeError("Must call fit_reference() first.")
        return self.reference_pca.mean_.copy()


# ============================================================================
# Batch Processing Functions
# ============================================================================


def build_cross_epoch_trajectory(
    aligner: AlignedPCA,
    embeddings_by_epoch: dict[int, NDArray[np.floating[Any]]],
    sample_indices: list[int] | None = None,
) -> CrossEpochTrajectory:
    """Build trajectory of embeddings across multiple epochs.

    Given embeddings from multiple epochs, transforms them all into the
    aligned PCA space and organizes them as trajectories.

    Args:
        aligner: Fitted AlignedPCA instance.
        embeddings_by_epoch: Dictionary mapping epoch number to embeddings.
            Each embedding array should have shape (n_samples, n_features).
        sample_indices: Optional list of sample indices to track. If None,
            tracks all samples (requires all epochs to have same n_samples).

    Returns:
        CrossEpochTrajectory with aligned embeddings across epochs.

    Raises:
        RuntimeError: If aligner is not fitted.
        ValueError: If sample counts are inconsistent.

    Example:
        >>> aligner = AlignedPCA(reference_epoch=186)
        >>> aligner.fit_reference(epoch_186_embeddings)
        >>> trajectory = build_cross_epoch_trajectory(
        ...     aligner,
        ...     {0: emb_0, 50: emb_50, 100: emb_100, 186: emb_186}
        ... )
        >>> print(f"Tracking {trajectory.n_samples} samples across {trajectory.n_epochs} epochs")
    """
    if not aligner.is_fitted():
        raise RuntimeError(
            "AlignedPCA must be fitted before building trajectory. "
            "Call fit_reference() with epoch 186 embeddings first."
        )

    # Sort epochs for consistent ordering
    sorted_epochs = sorted(embeddings_by_epoch.keys())

    # Determine which samples to track
    if sample_indices is None:
        # Track all samples - verify consistent count
        n_samples_set = {emb.shape[0] for emb in embeddings_by_epoch.values()}
        if len(n_samples_set) > 1:
            raise ValueError(
                f"Inconsistent sample counts across epochs: {n_samples_set}. "
                "Provide sample_indices to track specific samples."
            )
        n_samples = n_samples_set.pop()
        sample_indices = list(range(n_samples))
    else:
        n_samples = len(sample_indices)

    # Transform each epoch and collect trajectories
    trajectories = []
    variance_explained_per_epoch = []

    for epoch in sorted_epochs:
        embeddings = embeddings_by_epoch[epoch]

        # Extract requested samples if needed
        if sample_indices is not None and len(sample_indices) < embeddings.shape[0]:
            embeddings = embeddings[sample_indices]

        result = aligner.transform(embeddings, epoch=epoch)
        trajectories.append(result.transformed_embeddings)
        variance_explained_per_epoch.append(result.total_variance_explained)

    return CrossEpochTrajectory(
        epochs=sorted_epochs,
        trajectories=np.stack(trajectories),
        sample_indices=sample_indices,
        variance_explained_per_epoch=variance_explained_per_epoch,
        metadata={
            "reference_epoch": aligner.reference_epoch,
            "n_components": aligner.n_components,
        },
    )


def compute_epoch_distances(
    trajectory: CrossEpochTrajectory,
    reference_epoch: int | None = None,
) -> dict[int, NDArray[np.floating[Any]]]:
    """Compute distances from each epoch to a reference epoch.

    For each epoch in the trajectory, computes the Euclidean distance
    from each sample to its position in the reference epoch.

    Args:
        trajectory: CrossEpochTrajectory with aligned embeddings.
        reference_epoch: Epoch to use as reference. Defaults to the
            last epoch in the trajectory.

    Returns:
        Dictionary mapping epoch number to array of distances.
        Each array has shape (n_samples,).
    """
    if reference_epoch is None:
        reference_epoch = trajectory.epochs[-1]

    ref_embeddings = trajectory.get_epoch_embeddings(reference_epoch)

    distances = {}
    for epoch in trajectory.epochs:
        epoch_embeddings = trajectory.get_epoch_embeddings(epoch)
        distances[epoch] = np.linalg.norm(epoch_embeddings - ref_embeddings, axis=1)

    return distances


def compute_convergence_curve(
    trajectory: CrossEpochTrajectory,
) -> tuple[list[int], NDArray[np.floating[Any]]]:
    """Compute convergence curve showing mean distance to final epoch.

    Useful for visualizing how quickly embeddings converge to their
    final configuration during training.

    Args:
        trajectory: CrossEpochTrajectory with aligned embeddings.

    Returns:
        Tuple of (epochs, mean_distances) where:
        - epochs: List of epoch numbers
        - mean_distances: Array of mean distances to final epoch
    """
    distances = compute_epoch_distances(trajectory, reference_epoch=trajectory.epochs[-1])
    mean_distances = np.array([np.mean(distances[e]) for e in trajectory.epochs])
    return trajectory.epochs, mean_distances


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface for aligned PCA analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aligned PCA for cross-epoch embedding analysis"
    )
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="Verify that epoch 186 is used as reference frame",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with synthetic data",
    )

    args = parser.parse_args()

    if args.verify_reference:
        print(f"Default reference epoch: {DEFAULT_REFERENCE_EPOCH}")
        aligner = AlignedPCA()
        print(f"AlignedPCA reference_epoch: {aligner.reference_epoch}")
        if aligner.reference_epoch == 186:
            print("VERIFIED: Epoch 186 is configured as reference frame (PRD v1.1 compliant)")
        else:
            print("WARNING: Reference epoch is not 186!")

    if args.test:
        print("\nRunning quick test with synthetic data...")

        # Generate synthetic embeddings
        np.random.seed(42)
        n_samples = 100
        n_features = 128

        # Simulate embedding evolution - starts random, converges to final state
        epoch_186 = np.random.randn(n_samples, n_features)
        epoch_100 = epoch_186 + np.random.randn(n_samples, n_features) * 0.5
        epoch_50 = epoch_186 + np.random.randn(n_samples, n_features) * 1.0
        epoch_0 = np.random.randn(n_samples, n_features)  # Random initial

        # Create aligner and fit on reference epoch
        aligner = AlignedPCA(n_components=50, reference_epoch=186)
        fit_result = aligner.fit_reference(epoch_186, epoch=186)

        print("\nFit Results (epoch 186 reference):")
        print(f"  Components: {fit_result.n_components}")
        print(f"  Variance explained: {fit_result.total_variance_explained:.2%}")
        print(f"  Effective dimensionality: {fit_result.effective_dimensionality}")

        # Transform other epochs
        for name, embeddings, epoch in [
            ("Epoch 186 (reference)", epoch_186, 186),
            ("Epoch 100", epoch_100, 100),
            ("Epoch 50", epoch_50, 50),
            ("Epoch 0", epoch_0, 0),
        ]:
            result = aligner.transform(embeddings, epoch=epoch)
            print(f"\n{name}:")
            print(f"  Variance explained: {result.total_variance_explained:.2%}")
            print(f"  Quality: {result.quality_assessment}")
            print(f"  Reconstruction error: {result.reconstruction_error:.4f}")

        # Build trajectory
        trajectory = build_cross_epoch_trajectory(
            aligner,
            {0: epoch_0, 50: epoch_50, 100: epoch_100, 186: epoch_186},
        )
        print(f"\nTrajectory built: {trajectory.n_epochs} epochs, {trajectory.n_samples} samples")

        # Compute convergence
        epochs, distances = compute_convergence_curve(trajectory)
        print("\nConvergence to epoch 186:")
        for e, d in zip(epochs, distances, strict=False):
            print(f"  Epoch {e:3d}: mean distance = {d:.4f}")

        print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
