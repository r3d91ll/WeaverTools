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

from .layer_utils import (
    # Result types
    BottleneckResult,
    LayerComparisonResult,
    LayerTrajectoryResult,
    # Layer analysis functions
    compare_layer_deff,
    compute_layer_trajectory,
    find_bottleneck_layers,
    compute_layer_similarity_matrix,
)

from .agent_flow import (
    # Result types
    AgentAlignmentResult,
    AgentBottleneckResult,
    AgentFlowGraphResult,
    FlowComparisonResult,
    # Main agent flow analysis functions
    build_agent_flow_graph,
    compare_flow_patterns,
    find_agent_bottlenecks,
    visualize_agent_alignment,
)

from .aligned_pca import (
    # Constants
    DEFAULT_REFERENCE_EPOCH,
    DEFAULT_N_COMPONENTS,
    VARIANCE_EXPLAINED_THRESHOLD,
    # Main class
    AlignedPCA,
    # Result types
    AlignedPCAFitResult,
    AlignedPCAResult,
    CrossEpochTrajectory,
    # Utility functions
    build_cross_epoch_trajectory,
    compute_epoch_distances,
    compute_convergence_curve,
)

from .concept_landscape import (
    # Constants
    DEFAULT_SAMPLE_TEXTS,
    VISUALIZATION_COMPONENTS,
    ANALYSIS_COMPONENTS,
    MIN_EPOCHS_FOR_ANALYSIS,
    # Result types
    EpochEmbeddings,
    ConceptLandscapeResult,
    VisualizationConfig,
    # Extraction functions
    extract_embeddings_from_checkpoint,
    extract_embeddings_batch,
    # Analysis functions
    analyze_concept_landscape,
    # Visualization functions
    create_landscape_visualization,
    create_convergence_plot,
    # Pipeline orchestration
    run_concept_landscape_pipeline,
)

from .memory_tracing import (
    # Constants
    SPARSITY_THRESHOLD,
    RANK_THRESHOLD,
    TOP_SINGULAR_VALUES,
    MIN_EPOCHS_FOR_TREND,
    # Result types
    LayerMemoryStats,
    MemoryEpisodeStats,
    MemoryEvolutionResult,
    # Core analysis functions
    compute_matrix_stats,
    analyze_layer_memory,
    analyze_memory_states,
    analyze_memory_checkpoint,
    analyze_memory_evolution,
    # Visualization functions
    create_memory_heatmap,
    create_evolution_plot,
    create_singular_value_plot,
    # High-level interface
    MemoryTracer,
)

from .visualization import (
    # Constants
    DEFAULT_AXIS_RANGE,
    DEFAULT_PNG_WIDTH,
    DEFAULT_PNG_HEIGHT,
    PNG_SCALE_FACTOR,
    MAX_HTML_SIZE_BYTES,
    DEFAULT_ANIMATION_DURATION_MS,
    WEBGL_THRESHOLD_POINTS,
    # Configuration classes
    VisualizationStyle,
    AnimationConfig,
    ExportConfig,
    Axis3DConfig,
    Landscape3DConfig,
    # Result types
    ExportResult,
    # Core 3D visualization functions
    create_scatter_3d,
    create_animated_scatter_3d,
    create_trajectory_plot_3d,
    create_surface_3d,
    # Heatmap functions
    create_heatmap,
    create_multi_heatmap_grid,
    # Line plot functions
    create_line_plot,
    create_multi_line_subplot,
    # Export functions
    export_figure,
    export_for_publication,
    # High-level visualization functions
    visualize_epoch_evolution,
    visualize_memory_statistics,
)

from .atlas_statistics import (
    # Constants
    OUTLIER_Z_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    MIN_EPOCHS_FOR_STATS,
    DEFAULT_CLUSTER_COMPONENTS,
    # Result types
    EpochStatistics,
    ClusterStabilityMetrics,
    TrendAnalysis,
    OutlierDetectionResult,
    AtlasStatisticsResult,
    # Core analysis functions
    compute_epoch_statistics,
    analyze_trends,
    detect_outliers,
    compute_summary_statistics,
    # High-level interface
    AtlasStatisticsAnalyzer,
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
    # Layer analysis functions
    "compare_layer_deff",
    "compute_layer_trajectory",
    "find_bottleneck_layers",
    "compute_layer_similarity_matrix",
    # Layer analysis result types
    "BottleneckResult",
    "LayerComparisonResult",
    "LayerTrajectoryResult",
    # Agent flow analysis functions
    "build_agent_flow_graph",
    "compare_flow_patterns",
    "find_agent_bottlenecks",
    "visualize_agent_alignment",
    # Agent flow result types
    "AgentAlignmentResult",
    "AgentBottleneckResult",
    "AgentFlowGraphResult",
    "FlowComparisonResult",
    # Aligned PCA constants
    "DEFAULT_REFERENCE_EPOCH",
    "DEFAULT_N_COMPONENTS",
    "VARIANCE_EXPLAINED_THRESHOLD",
    # Aligned PCA main class
    "AlignedPCA",
    # Aligned PCA result types
    "AlignedPCAFitResult",
    "AlignedPCAResult",
    "CrossEpochTrajectory",
    # Aligned PCA utility functions
    "build_cross_epoch_trajectory",
    "compute_epoch_distances",
    "compute_convergence_curve",
    # Concept Landscape constants
    "DEFAULT_SAMPLE_TEXTS",
    "VISUALIZATION_COMPONENTS",
    "ANALYSIS_COMPONENTS",
    "MIN_EPOCHS_FOR_ANALYSIS",
    # Concept Landscape result types
    "EpochEmbeddings",
    "ConceptLandscapeResult",
    "VisualizationConfig",
    # Concept Landscape extraction functions
    "extract_embeddings_from_checkpoint",
    "extract_embeddings_batch",
    # Concept Landscape analysis functions
    "analyze_concept_landscape",
    # Concept Landscape visualization functions
    "create_landscape_visualization",
    "create_convergence_plot",
    # Concept Landscape pipeline
    "run_concept_landscape_pipeline",
    # Memory Tracing constants
    "SPARSITY_THRESHOLD",
    "RANK_THRESHOLD",
    "TOP_SINGULAR_VALUES",
    "MIN_EPOCHS_FOR_TREND",
    # Memory Tracing result types
    "LayerMemoryStats",
    "MemoryEpisodeStats",
    "MemoryEvolutionResult",
    # Memory Tracing analysis functions
    "compute_matrix_stats",
    "analyze_layer_memory",
    "analyze_memory_states",
    "analyze_memory_checkpoint",
    "analyze_memory_evolution",
    # Memory Tracing visualization functions
    "create_memory_heatmap",
    "create_evolution_plot",
    "create_singular_value_plot",
    # Memory Tracing high-level interface
    "MemoryTracer",
    # Visualization module constants
    "DEFAULT_AXIS_RANGE",
    "DEFAULT_PNG_WIDTH",
    "DEFAULT_PNG_HEIGHT",
    "PNG_SCALE_FACTOR",
    "MAX_HTML_SIZE_BYTES",
    "DEFAULT_ANIMATION_DURATION_MS",
    "WEBGL_THRESHOLD_POINTS",
    # Visualization configuration classes
    "VisualizationStyle",
    "AnimationConfig",
    "ExportConfig",
    "Axis3DConfig",
    "Landscape3DConfig",
    # Visualization result types
    "ExportResult",
    # Core 3D visualization functions
    "create_scatter_3d",
    "create_animated_scatter_3d",
    "create_trajectory_plot_3d",
    "create_surface_3d",
    # Heatmap visualization functions
    "create_heatmap",
    "create_multi_heatmap_grid",
    # Line plot visualization functions
    "create_line_plot",
    "create_multi_line_subplot",
    # Visualization export functions
    "export_figure",
    "export_for_publication",
    # High-level visualization functions
    "visualize_epoch_evolution",
    "visualize_memory_statistics",
    # Atlas Statistics constants
    "OUTLIER_Z_THRESHOLD",
    "OUTLIER_IQR_MULTIPLIER",
    "MIN_EPOCHS_FOR_STATS",
    "DEFAULT_CLUSTER_COMPONENTS",
    # Atlas Statistics result types
    "EpochStatistics",
    "ClusterStabilityMetrics",
    "TrendAnalysis",
    "OutlierDetectionResult",
    "AtlasStatisticsResult",
    # Atlas Statistics analysis functions
    "compute_epoch_statistics",
    "analyze_trends",
    "detect_outliers",
    "compute_summary_statistics",
    # Atlas Statistics high-level interface
    "AtlasStatisticsAnalyzer",
]
