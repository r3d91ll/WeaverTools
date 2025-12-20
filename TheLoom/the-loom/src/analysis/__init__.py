"""Analysis modules for TheLoom hidden state extraction.

This package contains geometric and statistical analysis tools for
studying transformer hidden state representations.
"""

from .kakeya_geometry import (
    # Main analysis function
    analyze_kakeya_geometry,
    analyze_hidden_state_batch,
    # Bilateral comparison
    compare_bilateral_geometry,
    run_conveyance_experiment,
    # Individual analyses
    check_wolf_axioms,
    analyze_directional_coverage,
    analyze_grains,
    # Result types
    KakeyaGeometryReport,
    WolfAxiomResult,
    DirectionalCoverageResult,
    GrainAnalysisResult,
    Grain,
    BilateralGeometryResult,
)

__all__ = [
    # Main analysis function
    "analyze_kakeya_geometry",
    "analyze_hidden_state_batch",
    # Bilateral comparison
    "compare_bilateral_geometry",
    "run_conveyance_experiment",
    # Individual analyses
    "check_wolf_axioms",
    "analyze_directional_coverage",
    "analyze_grains",
    # Result types
    "KakeyaGeometryReport",
    "WolfAxiomResult",
    "DirectionalCoverageResult",
    "GrainAnalysisResult",
    "Grain",
    "BilateralGeometryResult",
]
