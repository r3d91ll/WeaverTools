"""Analysis modules for TheLoom hidden state extraction.

This package contains geometric and statistical analysis tools for
studying transformer hidden state representations.
"""

from .kakeya_geometry import (
    # Constants
    MIN_SAMPLES_FOR_ANALYSIS,
    # Protocols
    HiddenStateProtocol,
    # Result types
    BilateralGeometryResult,
    DirectionalCoverageResult,
    Grain,
    GrainAnalysisResult,
    KakeyaGeometryReport,
    WolfAxiomResult,
    # Main analysis functions
    analyze_directional_coverage,
    analyze_grains,
    analyze_hidden_state_batch,
    analyze_kakeya_geometry,
    check_wolf_axioms,
    compare_bilateral_geometry,
    run_conveyance_experiment,
)

from .conveyance_metrics import (
    # Result types
    ConveyanceMetricsResult,
    # Main conveyance metric functions
    bootstrap_ci,
    calculate_beta,
    calculate_c_pair,
    calculate_d_eff,
)

__all__ = [
    # Constants
    "MIN_SAMPLES_FOR_ANALYSIS",
    # Protocols
    "HiddenStateProtocol",
    # Main analysis functions
    "analyze_kakeya_geometry",
    "analyze_hidden_state_batch",
    "compare_bilateral_geometry",
    "run_conveyance_experiment",
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
    # Conveyance metrics functions
    "calculate_d_eff",
    "calculate_beta",
    "calculate_c_pair",
    "bootstrap_ci",
    # Conveyance metrics result types
    "ConveyanceMetricsResult",
]
