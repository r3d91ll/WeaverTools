"""Kakeya Geometry Analysis for Hidden States.

This module implements geometric tests inspired by Kakeya set theory to analyze
whether transformer hidden states exhibit properties that might constrain or
enable effective information transfer (conveyance).

HYPOTHESIS UNDER INVESTIGATION:
If Kakeya-like geometric constraints govern semantic representation, then:
1. Hidden states should exhibit bounded density in convex regions (Wolf axioms)
2. Directional coverage should be "full" relative to effective dimensionality
3. "Grains" (semantic intersection regions) should have consistent structure

FALSIFICATION CRITERIA:
- If Wolf density violations correlate with BETTER task performance → reject
- If directional sparsity correlates with BETTER conveyance → reject
- If grain structure shows no relationship to transfer success → reject

Integration: Designed to work with TheLoom's HiddenStateResult class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.random_projection import GaussianRandomProjection
import warnings


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class WolfAxiomResult:
    """Results from Wolf axiom density analysis.

    The Wolf axioms in Kakeya theory constrain how tubes can be distributed
    in convex regions. We adapt this to embeddings: vectors shouldn't be
    overly concentrated in any convex subregion of the space.

    Interpretation:
    - max_density_ratio > 2.0: Potential concentration violation
    - max_density_ratio > 5.0: Strong violation (slab-like structure)
    - uniformity_p_value < 0.05: Statistically significant non-uniformity
    """

    max_density_ratio: float  # max(observed/expected) across regions
    mean_density_ratio: float  # average density ratio
    density_ratios: List[float]  # per-region ratios
    uniformity_p_value: float  # statistical test for uniform distribution
    violation_count: int  # regions exceeding threshold
    violation_threshold: float  # threshold used
    num_regions_tested: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_violation(self) -> bool:
        """Check if any region shows density violation."""
        return self.violation_count > 0

    @property
    def severity(self) -> str:
        """Classify violation severity."""
        if self.max_density_ratio < 1.5:
            return "none"
        elif self.max_density_ratio < 2.5:
            return "mild"
        elif self.max_density_ratio < 5.0:
            return "moderate"
        else:
            return "severe"


@dataclass
class DirectionalCoverageResult:
    """Results from directional coverage analysis.

    Kakeya sets contain unit line segments in EVERY direction while having
    measure zero. For embeddings, we ask: how completely do hidden states
    cover the available directions in the space?

    Interpretation:
    - effective_dim / ambient_dim: Fraction of space utilized
    - spherical_uniformity close to 1.0: Good directional coverage
    - coverage_gaps: Directions with unusually low representation
    """

    ambient_dim: int  # Original dimensionality
    effective_dim: int  # Dimensions needed for 95% variance
    effective_dim_99: int  # Dimensions needed for 99% variance
    variance_explained: np.ndarray  # Per-component variance
    spherical_uniformity: float  # 0-1, how uniform on unit sphere
    coverage_ratio: float  # effective_dim / ambient_dim
    principal_angles: np.ndarray  # Angles of top principal components
    isotropy_score: float  # How isotropic (vs elongated) the distribution is
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_degenerate(self) -> bool:
        """Check if distribution is degenerate (collapsed to subspace)."""
        return self.coverage_ratio < 0.1

    @property
    def coverage_quality(self) -> str:
        """Classify coverage quality."""
        if self.coverage_ratio < 0.1:
            return "degenerate"
        elif self.coverage_ratio < 0.3:
            return "sparse"
        elif self.coverage_ratio < 0.6:
            return "moderate"
        else:
            return "full"


@dataclass
class Grain:
    """A detected grain (semantic intersection region).

    In Kakeya theory, grains are where multiple tubes intersect - they have
    characteristic dimensions δ × δ × δ/ρ. In embedding space, we look for
    regions where multiple semantic directions converge.
    """

    center: np.ndarray  # Centroid of the grain
    members: List[int]  # Indices of vectors in this grain
    density: float  # Local density
    principal_directions: np.ndarray  # Local PCA directions
    local_dim: int  # Local effective dimensionality
    aspect_ratio: float  # Elongation (1.0 = spherical)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class GrainAnalysisResult:
    """Results from grain detection and analysis.

    Interpretation:
    - num_grains: How many distinct semantic intersection regions
    - grain_alignment: Do grains align across samples? (important for conveyance)
    - dimension_consistency: Is local_dim consistent across grains?
    """

    grains: List[Grain]
    num_grains: int
    mean_grain_size: float
    grain_coverage: float  # Fraction of vectors in grains
    dimension_consistency: float  # Std of local_dim across grains
    mean_local_dim: float
    mean_aspect_ratio: float
    singleton_count: int  # Vectors not in any grain
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_structure(self) -> bool:
        """Check if meaningful grain structure exists."""
        return self.num_grains > 1 and self.grain_coverage > 0.5


@dataclass
class KakeyaGeometryReport:
    """Complete Kakeya geometry analysis report.

    Combines all analyses into a single report with overall assessment.
    """

    wolf_axiom: WolfAxiomResult
    directional_coverage: DirectionalCoverageResult
    grain_analysis: GrainAnalysisResult
    num_vectors: int
    ambient_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_health(self) -> str:
        """Overall geometric health assessment."""
        issues = []

        if self.wolf_axiom.severity in ("moderate", "severe"):
            issues.append("density_violation")
        if self.directional_coverage.is_degenerate:
            issues.append("dimensional_collapse")
        if not self.grain_analysis.has_structure:
            issues.append("no_grain_structure")

        if not issues:
            return "healthy"
        elif len(issues) == 1:
            return f"warning:{issues[0]}"
        else:
            return f"unhealthy:{','.join(issues)}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_health": self.overall_health,
            "num_vectors": self.num_vectors,
            "ambient_dim": self.ambient_dim,
            "wolf_axiom": {
                "max_density_ratio": self.wolf_axiom.max_density_ratio,
                "severity": self.wolf_axiom.severity,
                "violation_count": self.wolf_axiom.violation_count,
                "uniformity_p_value": self.wolf_axiom.uniformity_p_value,
            },
            "directional_coverage": {
                "effective_dim": self.directional_coverage.effective_dim,
                "coverage_ratio": self.directional_coverage.coverage_ratio,
                "coverage_quality": self.directional_coverage.coverage_quality,
                "spherical_uniformity": self.directional_coverage.spherical_uniformity,
                "isotropy_score": self.directional_coverage.isotropy_score,
            },
            "grain_analysis": {
                "num_grains": self.grain_analysis.num_grains,
                "grain_coverage": self.grain_analysis.grain_coverage,
                "mean_local_dim": self.grain_analysis.mean_local_dim,
                "has_structure": self.grain_analysis.has_structure,
            },
            "metadata": self.metadata,
        }


# ============================================================================
# Wolf Axiom Analysis
# ============================================================================


def check_wolf_axioms(
    vectors: np.ndarray,
    num_regions: int = 50,
    violation_threshold: float = 2.5,
    random_state: Optional[int] = None,
) -> WolfAxiomResult:
    """Check Wolf-axiom-like density constraints on embedding distribution.

    The Wolf axioms (Castile-Wolf and Frostman-Wolf) constrain how Kakeya
    tubes can be distributed in convex regions. We adapt this to embeddings:

    For any convex region W:
        density(vectors in W) ≤ C × expected_density

    If vectors are uniformly distributed, density should be proportional to
    region volume. Violations indicate "slab-like" concentration.

    Parameters:
        vectors: Array of shape (n_samples, n_features), should be L2-normalized
        num_regions: Number of random convex regions to test
        violation_threshold: Ratio above which to flag violation
        random_state: For reproducibility

    Returns:
        WolfAxiomResult with density analysis

    FALSIFICATION: If high density_ratio correlates with BETTER performance,
    the Kakeya-constraint hypothesis is weakened.
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = vectors.shape

    # Normalize vectors to unit sphere for directional analysis
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
    normalized = vectors / norms

    density_ratios = []

    for _ in range(num_regions):
        # Generate random convex region via random halfspace intersection
        # This creates a "slice" through the embedding space
        region_vectors, expected_fraction = _sample_convex_region(
            normalized, n_features, rng
        )

        if expected_fraction < 0.01:  # Skip tiny regions
            continue

        observed_fraction = len(region_vectors) / n_samples

        if expected_fraction > 0:
            ratio = observed_fraction / expected_fraction
            density_ratios.append(ratio)

    if not density_ratios:
        # Fallback if all regions were too small
        return WolfAxiomResult(
            max_density_ratio=1.0,
            mean_density_ratio=1.0,
            density_ratios=[1.0],
            uniformity_p_value=1.0,
            violation_count=0,
            violation_threshold=violation_threshold,
            num_regions_tested=0,
            metadata={"warning": "insufficient_valid_regions"},
        )

    density_ratios_arr = np.array(density_ratios)

    # Statistical test for uniformity
    # Under uniformity, density ratios should be ~1.0
    # Use one-sample t-test against mean=1.0
    if len(density_ratios_arr) > 2:
        _, p_value = stats.ttest_1samp(density_ratios_arr, 1.0)
    else:
        p_value = 1.0

    violation_count = int(np.sum(density_ratios_arr > violation_threshold))

    return WolfAxiomResult(
        max_density_ratio=float(np.max(density_ratios_arr)),
        mean_density_ratio=float(np.mean(density_ratios_arr)),
        density_ratios=density_ratios_arr.tolist(),
        uniformity_p_value=float(p_value),
        violation_count=violation_count,
        violation_threshold=violation_threshold,
        num_regions_tested=len(density_ratios_arr),
    )


def _sample_convex_region(
    normalized_vectors: np.ndarray,
    n_features: int,
    rng: np.random.RandomState,
    num_halfspaces: int = 3,
) -> Tuple[np.ndarray, float]:
    """Sample a random convex region and return vectors inside it.

    Creates region by intersection of random halfspaces.
    Returns vectors in region and expected fraction under uniform distribution.
    """
    n_samples = len(normalized_vectors)

    # Generate random halfspaces
    # Each halfspace: {x : <x, normal> >= threshold}
    in_region = np.ones(n_samples, dtype=bool)
    expected_fraction = 1.0

    for _ in range(num_halfspaces):
        # Random direction on unit sphere
        normal = rng.randn(n_features)
        normal = normal / np.linalg.norm(normal)

        # Random threshold (bias toward keeping reasonable fraction)
        # For unit sphere, projection onto random direction is ~N(0, 1/d)
        threshold = rng.uniform(-0.5, 0.5) / np.sqrt(n_features)

        # Check which vectors satisfy this halfspace
        projections = normalized_vectors @ normal
        in_halfspace = projections >= threshold
        in_region = in_region & in_halfspace

        # Expected fraction under uniform spherical distribution
        # Approximate: P(projection >= t) ≈ 0.5 for t=0, less for t>0
        expected_fraction *= 0.5 * (1 - stats.norm.cdf(threshold * np.sqrt(n_features)))

    return normalized_vectors[in_region], expected_fraction


# ============================================================================
# Directional Coverage Analysis
# ============================================================================


def analyze_directional_coverage(
    vectors: np.ndarray,
    variance_threshold_95: float = 0.95,
    variance_threshold_99: float = 0.99,
) -> DirectionalCoverageResult:
    """Analyze how completely vectors cover the available directions.

    Kakeya sets are remarkable for containing directions to EVERY point
    while having measure zero. For embeddings, we ask: do hidden states
    achieve "full directional coverage" of the semantic space?

    Parameters:
        vectors: Array of shape (n_samples, n_features)
        variance_threshold_95: Variance threshold for effective_dim
        variance_threshold_99: Variance threshold for effective_dim_99

    Returns:
        DirectionalCoverageResult with coverage analysis

    FALSIFICATION: If low coverage_ratio correlates with BETTER conveyance,
    the Kakeya-coverage hypothesis is weakened.
    """
    n_samples, n_features = vectors.shape

    # Center the data
    centered = vectors - np.mean(vectors, axis=0)

    # PCA for effective dimensionality
    n_components = min(n_samples - 1, n_features)
    pca = PCA(n_components=n_components)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca.fit(centered)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    # Effective dimensionality at different thresholds
    effective_dim_95 = int(np.searchsorted(cumulative_variance, variance_threshold_95) + 1)
    effective_dim_99 = int(np.searchsorted(cumulative_variance, variance_threshold_99) + 1)

    # Spherical uniformity: compare to uniform distribution on sphere
    # Use L2-normalized vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    normalized = vectors / norms

    spherical_uniformity = _compute_spherical_uniformity(normalized)

    # Isotropy score: ratio of smallest to largest eigenvalue
    # 1.0 = perfectly isotropic, 0.0 = completely anisotropic
    eigenvalues = pca.explained_variance_
    if len(eigenvalues) > 0 and eigenvalues[0] > 0:
        # Use ratio of geometric mean to max
        log_eigenvalues = np.log(eigenvalues[eigenvalues > 0] + 1e-10)
        geo_mean = np.exp(np.mean(log_eigenvalues))
        isotropy_score = float(geo_mean / eigenvalues[0])
    else:
        isotropy_score = 0.0

    # Principal angles (directions of max variance)
    principal_angles = pca.components_[:min(10, len(pca.components_))]

    return DirectionalCoverageResult(
        ambient_dim=n_features,
        effective_dim=effective_dim_95,
        effective_dim_99=effective_dim_99,
        variance_explained=variance_explained,
        spherical_uniformity=spherical_uniformity,
        coverage_ratio=effective_dim_95 / n_features,
        principal_angles=principal_angles,
        isotropy_score=isotropy_score,
    )


def _compute_spherical_uniformity(normalized_vectors: np.ndarray) -> float:
    """Compute how uniformly vectors are distributed on the unit sphere.

    Uses pairwise cosine similarity distribution. For uniform distribution
    on high-dimensional sphere, pairwise cosines should be ~N(0, 1/d).

    Returns value in [0, 1] where 1 = perfectly uniform.
    """
    n_samples, n_features = normalized_vectors.shape

    if n_samples < 10:
        return 0.5  # Not enough samples to assess

    # Sample pairwise cosine similarities (avoid O(n²) for large n)
    max_pairs = 5000
    if n_samples * (n_samples - 1) // 2 > max_pairs:
        # Random sample of pairs
        idx1 = np.random.randint(0, n_samples, max_pairs)
        idx2 = np.random.randint(0, n_samples, max_pairs)
        # Ensure different indices
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        cosines = np.sum(normalized_vectors[idx1] * normalized_vectors[idx2], axis=1)
    else:
        # All pairs
        cosines = (normalized_vectors @ normalized_vectors.T)[np.triu_indices(n_samples, k=1)]

    # Under uniform distribution on d-sphere, cosines ~ N(0, 1/d)
    expected_std = 1.0 / np.sqrt(n_features)
    observed_std = np.std(cosines)
    observed_mean = np.mean(cosines)

    # Score based on how close to expected distribution
    mean_score = np.exp(-abs(observed_mean) * np.sqrt(n_features))  # Mean should be ~0
    std_score = np.exp(-abs(observed_std - expected_std) / expected_std)  # Std should match

    return float(0.5 * mean_score + 0.5 * std_score)


# ============================================================================
# Grain Analysis
# ============================================================================


def analyze_grains(
    vectors: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 3,
    local_pca_neighbors: int = 10,
) -> GrainAnalysisResult:
    """Detect and analyze grains (semantic intersection regions).

    In Kakeya theory, grains are where multiple tubes intersect. They have
    characteristic structure (δ × δ × δ/ρ). In embeddings, we look for
    analogous structures: regions where multiple semantic directions converge.

    Parameters:
        vectors: Array of shape (n_samples, n_features)
        eps: DBSCAN epsilon (auto-computed if None)
        min_samples: Minimum samples for a grain
        local_pca_neighbors: Neighbors for local dimensionality

    Returns:
        GrainAnalysisResult with grain structure analysis

    FALSIFICATION: If grain structure shows no correlation with
    conveyance success, the Kakeya-grain hypothesis is weakened.
    """
    n_samples, n_features = vectors.shape

    # Normalize for directional clustering
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    normalized = vectors / norms

    # Auto-compute eps if not provided
    if eps is None:
        # Use k-th nearest neighbor distance heuristic
        k = min(min_samples, n_samples - 1)
        distances = cdist(normalized, normalized)
        np.fill_diagonal(distances, np.inf)
        kth_distances = np.partition(distances, k, axis=1)[:, k]
        eps = float(np.median(kth_distances) * 1.5)

    # Cluster using DBSCAN (finds dense regions)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(normalized)

    # Analyze each grain
    grains = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    for label in unique_labels:
        member_indices = np.where(labels == label)[0]
        grain_vectors = normalized[member_indices]

        grain = _analyze_single_grain(
            grain_vectors,
            member_indices.tolist(),
            local_pca_neighbors
        )
        grains.append(grain)

    # Compute summary statistics
    singleton_count = int(np.sum(labels == -1))

    if grains:
        mean_grain_size = float(np.mean([g.size for g in grains]))
        grain_coverage = float(1 - singleton_count / n_samples)
        local_dims = [g.local_dim for g in grains]
        dimension_consistency = float(np.std(local_dims)) if len(local_dims) > 1 else 0.0
        mean_local_dim = float(np.mean(local_dims))
        mean_aspect_ratio = float(np.mean([g.aspect_ratio for g in grains]))
    else:
        mean_grain_size = 0.0
        grain_coverage = 0.0
        dimension_consistency = 0.0
        mean_local_dim = 0.0
        mean_aspect_ratio = 0.0

    return GrainAnalysisResult(
        grains=grains,
        num_grains=len(grains),
        mean_grain_size=mean_grain_size,
        grain_coverage=grain_coverage,
        dimension_consistency=dimension_consistency,
        mean_local_dim=mean_local_dim,
        mean_aspect_ratio=mean_aspect_ratio,
        singleton_count=singleton_count,
        metadata={"eps": eps, "min_samples": min_samples},
    )


def _analyze_single_grain(
    grain_vectors: np.ndarray,
    member_indices: List[int],
    n_neighbors: int,
) -> Grain:
    """Analyze structure of a single grain."""
    center = np.mean(grain_vectors, axis=0)
    center = center / (np.linalg.norm(center) + 1e-10)  # Normalize

    n_samples = len(grain_vectors)

    # Local PCA for dimensionality and shape
    if n_samples >= 3:
        centered = grain_vectors - np.mean(grain_vectors, axis=0)
        n_components = min(n_samples - 1, grain_vectors.shape[1], 10)

        pca = PCA(n_components=n_components)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca.fit(centered)

        # Local effective dimensionality
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        local_dim = int(np.searchsorted(cumvar, 0.95) + 1)

        # Aspect ratio (elongation)
        if len(pca.explained_variance_) >= 2 and pca.explained_variance_[0] > 0:
            aspect_ratio = float(
                np.sqrt(pca.explained_variance_[-1] / pca.explained_variance_[0])
            )
        else:
            aspect_ratio = 1.0

        principal_directions = pca.components_
    else:
        local_dim = n_samples
        aspect_ratio = 1.0
        principal_directions = np.eye(min(3, grain_vectors.shape[1]))

    # Local density (average distance to center)
    distances = np.linalg.norm(grain_vectors - center, axis=1)
    density = float(1.0 / (np.mean(distances) + 1e-10))

    return Grain(
        center=center,
        members=member_indices,
        density=density,
        principal_directions=principal_directions,
        local_dim=local_dim,
        aspect_ratio=aspect_ratio,
    )


# ============================================================================
# Bilateral Conveyance Geometry
# ============================================================================


@dataclass
class BilateralGeometryResult:
    """Geometric comparison between two sets of hidden states.

    For measuring conveyance: how well do geometric properties align
    between sender and receiver representations?
    """

    directional_alignment: float  # Cosine similarity of mean directions
    subspace_overlap: float  # Principal subspace overlap
    grain_alignment: float  # How well grains correspond
    density_similarity: float  # Wolf axiom profile similarity
    effective_dim_ratio: float  # Ratio of effective dimensions
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_alignment(self) -> float:
        """Weighted overall alignment score."""
        return (
            0.3 * self.directional_alignment +
            0.3 * self.subspace_overlap +
            0.2 * self.grain_alignment +
            0.2 * self.density_similarity
        )


def compare_bilateral_geometry(
    sender_vectors: np.ndarray,
    receiver_vectors: np.ndarray,
) -> BilateralGeometryResult:
    """Compare geometric properties between sender and receiver hidden states.

    This is the core conveyance measurement: do the geometric representations
    align in ways that predict successful information transfer?

    Parameters:
        sender_vectors: Hidden states from sending agent
        receiver_vectors: Hidden states from receiving agent

    Returns:
        BilateralGeometryResult with alignment metrics

    HYPOTHESIS: Higher alignment scores should correlate with better
    task performance / information transfer success.
    """
    # Normalize both sets
    def normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        result: np.ndarray = v / norms
        return result

    sender_norm = normalize(sender_vectors)
    receiver_norm = normalize(receiver_vectors)

    # 1. Directional alignment: cosine of mean directions
    sender_mean = np.mean(sender_norm, axis=0)
    sender_mean = sender_mean / (np.linalg.norm(sender_mean) + 1e-10)
    receiver_mean = np.mean(receiver_norm, axis=0)
    receiver_mean = receiver_mean / (np.linalg.norm(receiver_mean) + 1e-10)
    directional_alignment = float(np.dot(sender_mean, receiver_mean))

    # 2. Subspace overlap: principal component alignment
    subspace_overlap = _compute_subspace_overlap(sender_norm, receiver_norm)

    # 3. Grain alignment: do cluster structures correspond?
    grain_alignment = _compute_grain_alignment(sender_norm, receiver_norm)

    # 4. Density similarity: Wolf axiom profile comparison
    sender_wolf = check_wolf_axioms(sender_vectors, num_regions=30)
    receiver_wolf = check_wolf_axioms(receiver_vectors, num_regions=30)
    density_similarity = float(
        1.0 - abs(sender_wolf.mean_density_ratio - receiver_wolf.mean_density_ratio) /
        max(sender_wolf.mean_density_ratio, receiver_wolf.mean_density_ratio, 1.0)
    )

    # 5. Effective dimension ratio
    sender_cov = analyze_directional_coverage(sender_vectors)
    receiver_cov = analyze_directional_coverage(receiver_vectors)
    dim_ratio = min(sender_cov.effective_dim, receiver_cov.effective_dim) / \
                max(sender_cov.effective_dim, receiver_cov.effective_dim, 1)

    return BilateralGeometryResult(
        directional_alignment=directional_alignment,
        subspace_overlap=subspace_overlap,
        grain_alignment=grain_alignment,
        density_similarity=density_similarity,
        effective_dim_ratio=float(dim_ratio),
    )


def _compute_subspace_overlap(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
    n_components: int = 10,
) -> float:
    """Compute overlap between principal subspaces."""
    n_comp = min(n_components, vectors1.shape[1], vectors2.shape[1],
                 vectors1.shape[0] - 1, vectors2.shape[0] - 1)

    if n_comp < 1:
        return 0.0

    pca1 = PCA(n_components=n_comp)
    pca2 = PCA(n_components=n_comp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca1.fit(vectors1)
        pca2.fit(vectors2)

    # Compute principal angles between subspaces
    # Use SVD of V1.T @ V2
    V1 = pca1.components_.T  # (features, n_comp)
    V2 = pca2.components_.T

    cross = V1.T @ V2
    _, s, _ = np.linalg.svd(cross)

    # s contains cosines of principal angles
    # Overlap = mean of squared cosines
    return float(np.mean(s ** 2))


def _compute_grain_alignment(
    vectors1: np.ndarray,
    vectors2: np.ndarray,
) -> float:
    """Compute alignment between grain structures."""
    grains1 = analyze_grains(vectors1)
    grains2 = analyze_grains(vectors2)

    if grains1.num_grains == 0 or grains2.num_grains == 0:
        return 0.0

    # Compute similarity based on grain count and coverage
    count_sim = 1.0 - abs(grains1.num_grains - grains2.num_grains) / \
                max(grains1.num_grains, grains2.num_grains)
    coverage_sim = 1.0 - abs(grains1.grain_coverage - grains2.grain_coverage)
    dim_sim = 1.0 - abs(grains1.mean_local_dim - grains2.mean_local_dim) / \
              max(grains1.mean_local_dim, grains2.mean_local_dim, 1)

    return float(0.4 * count_sim + 0.3 * coverage_sim + 0.3 * dim_sim)


# ============================================================================
# Main Analysis Function
# ============================================================================


def analyze_kakeya_geometry(
    vectors: np.ndarray,
    wolf_num_regions: int = 50,
    wolf_violation_threshold: float = 2.5,
    grain_min_samples: int = 3,
    random_state: Optional[int] = None,
) -> KakeyaGeometryReport:
    """Complete Kakeya geometry analysis of hidden state vectors.

    This is the main entry point for analyzing whether embedding geometry
    exhibits Kakeya-like properties that might constrain/enable conveyance.

    Parameters:
        vectors: Array of shape (n_samples, n_features)
        wolf_num_regions: Number of regions for Wolf axiom testing
        wolf_violation_threshold: Threshold for density violations
        grain_min_samples: Minimum samples for grain detection
        random_state: For reproducibility

    Returns:
        KakeyaGeometryReport with complete analysis

    Usage:
        # From TheLoom HiddenStateResult
        hidden_states = [result.vector for result in extraction_results]
        vectors = np.stack(hidden_states)
        report = analyze_kakeya_geometry(vectors)
        print(report.overall_health)
    """
    n_samples, n_features = vectors.shape

    # Run all analyses
    wolf_result = check_wolf_axioms(
        vectors,
        num_regions=wolf_num_regions,
        violation_threshold=wolf_violation_threshold,
        random_state=random_state,
    )

    coverage_result = analyze_directional_coverage(vectors)

    grain_result = analyze_grains(
        vectors,
        min_samples=grain_min_samples,
    )

    return KakeyaGeometryReport(
        wolf_axiom=wolf_result,
        directional_coverage=coverage_result,
        grain_analysis=grain_result,
        num_vectors=n_samples,
        ambient_dim=n_features,
    )


# ============================================================================
# Integration with TheLoom
# ============================================================================


def analyze_hidden_state_batch(
    hidden_state_results: List[Any],  # List[HiddenStateResult]
    normalize: bool = True,
) -> KakeyaGeometryReport:
    """Analyze a batch of HiddenStateResult objects from TheLoom.

    Parameters:
        hidden_state_results: List of HiddenStateResult from TheLoom extraction
        normalize: Whether to L2-normalize before analysis

    Returns:
        KakeyaGeometryReport for the batch
    """
    # Extract vectors
    vectors = []
    for result in hidden_state_results:
        if normalize and hasattr(result, 'l2_normalize'):
            result = result.l2_normalize()

        if hasattr(result, 'vector'):
            vectors.append(result.vector.flatten())
        elif hasattr(result, 'to_list'):
            vectors.append(np.array(result.to_list()))
        else:
            vectors.append(np.array(result).flatten())

    vectors = np.stack(vectors)
    return analyze_kakeya_geometry(vectors)


# ============================================================================
# Experiment Utilities
# ============================================================================


def run_conveyance_experiment(
    sender_states: List[np.ndarray],
    receiver_states: List[np.ndarray],
    task_success: List[bool],
) -> dict:
    """Run a complete conveyance experiment correlating geometry with success.

    Parameters:
        sender_states: List of sender hidden state vectors per interaction
        receiver_states: List of receiver hidden state vectors per interaction
        task_success: Whether each interaction succeeded

    Returns:
        Dictionary with correlation analysis

    This is the key falsification test:
    - If geometric alignment predicts success → hypothesis supported
    - If geometric alignment uncorrelated with success → hypothesis weakened
    """
    if len(sender_states) != len(receiver_states) != len(task_success):
        raise ValueError("All lists must have same length")

    alignments = []
    wolf_violations_sender = []
    wolf_violations_receiver = []
    coverage_ratios_sender = []
    coverage_ratios_receiver = []

    for sender, receiver in zip(sender_states, receiver_states):
        # Bilateral comparison
        bilateral = compare_bilateral_geometry(
            sender.reshape(1, -1) if sender.ndim == 1 else sender,
            receiver.reshape(1, -1) if receiver.ndim == 1 else receiver,
        )
        alignments.append(bilateral.overall_alignment)

        # Individual analyses
        sender_report = analyze_kakeya_geometry(
            sender.reshape(1, -1) if sender.ndim == 1 else sender
        )
        receiver_report = analyze_kakeya_geometry(
            receiver.reshape(1, -1) if receiver.ndim == 1 else receiver
        )

        wolf_violations_sender.append(sender_report.wolf_axiom.max_density_ratio)
        wolf_violations_receiver.append(receiver_report.wolf_axiom.max_density_ratio)
        coverage_ratios_sender.append(sender_report.directional_coverage.coverage_ratio)
        coverage_ratios_receiver.append(receiver_report.directional_coverage.coverage_ratio)

    # Compute correlations with task success
    success_numeric = np.array(task_success, dtype=float)

    def safe_correlation(x: List[float], y: np.ndarray) -> Tuple[float, float]:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0, 1.0
        result = stats.pearsonr(x, y)
        return float(result.statistic), float(result.pvalue)

    alignment_corr, alignment_p = safe_correlation(alignments, success_numeric)
    wolf_sender_corr, wolf_sender_p = safe_correlation(wolf_violations_sender, success_numeric)
    coverage_sender_corr, coverage_sender_p = safe_correlation(coverage_ratios_sender, success_numeric)

    return {
        "n_interactions": len(task_success),
        "success_rate": float(np.mean(success_numeric)),
        "alignment": {
            "mean": float(np.mean(alignments)),
            "std": float(np.std(alignments)),
            "correlation_with_success": float(alignment_corr),
            "p_value": float(alignment_p),
        },
        "wolf_violations_sender": {
            "mean": float(np.mean(wolf_violations_sender)),
            "correlation_with_success": float(wolf_sender_corr),
            "p_value": float(wolf_sender_p),
        },
        "coverage_sender": {
            "mean": float(np.mean(coverage_ratios_sender)),
            "correlation_with_success": float(coverage_sender_corr),
            "p_value": float(coverage_sender_p),
        },
        "hypothesis_support": {
            "alignment_predicts_success": alignment_corr > 0.3 and alignment_p < 0.05,
            "wolf_violation_hurts": wolf_sender_corr < -0.2 and wolf_sender_p < 0.05,
            "coverage_helps": coverage_sender_corr > 0.2 and coverage_sender_p < 0.05,
        },
    }


if __name__ == "__main__":
    # Quick test with random data
    print("Testing Kakeya Geometry Analysis...")

    # Generate test vectors (simulating hidden states)
    np.random.seed(42)
    test_vectors = np.random.randn(100, 768)  # 100 samples, 768-dim (like GPT-2)

    report = analyze_kakeya_geometry(test_vectors)

    print(f"\nOverall Health: {report.overall_health}")
    print(f"\nWolf Axiom Analysis:")
    print(f"  Max Density Ratio: {report.wolf_axiom.max_density_ratio:.3f}")
    print(f"  Severity: {report.wolf_axiom.severity}")
    print(f"\nDirectional Coverage:")
    print(f"  Effective Dim: {report.directional_coverage.effective_dim} / {report.directional_coverage.ambient_dim}")
    print(f"  Coverage Ratio: {report.directional_coverage.coverage_ratio:.3f}")
    print(f"  Quality: {report.directional_coverage.coverage_quality}")
    print(f"\nGrain Analysis:")
    print(f"  Num Grains: {report.grain_analysis.num_grains}")
    print(f"  Coverage: {report.grain_analysis.grain_coverage:.3f}")

    # Test bilateral comparison
    print("\n\nTesting Bilateral Comparison...")
    sender = np.random.randn(50, 768)
    receiver = sender + np.random.randn(50, 768) * 0.1  # Similar to sender

    bilateral = compare_bilateral_geometry(sender, receiver)
    print(f"Overall Alignment: {bilateral.overall_alignment:.3f}")
    print(f"Directional: {bilateral.directional_alignment:.3f}")
    print(f"Subspace: {bilateral.subspace_overlap:.3f}")
