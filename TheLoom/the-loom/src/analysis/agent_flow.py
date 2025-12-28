"""Cross-Agent Information Flow Analysis.

This module implements graph-based analysis tools for tracking semantic
information flow between agents in multi-turn conversations. It extends
TheLoom's interpretability framework from single-model analysis to the
multi-agent context.

MATHEMATICAL GROUNDING
======================
- Information Flow Graph: Directed graph G = (V, E) where:
  - V = agent states (nodes with D_eff and beta attributes)
  - E = information transfer edges (weighted by conveyance metrics)

- Bottleneck Detection: Uses betweenness centrality combined with D_eff
  trajectory analysis. A transition (agent_i, agent_j) is a bottleneck if:

      D_eff_drop(i→j) > threshold × mean(D_eff_drops)

  where D_eff_drop = D_eff_i - D_eff_j when D_eff_j < D_eff_i.

- Semantic Alignment: Measured via cosine similarity of hidden states:

      alignment(i, j) = cos(h_i, h_j) = (h_i · h_j) / (||h_i|| × ||h_j||)

  Note: scipy.spatial.distance.cosine returns DISTANCE (1 - similarity),
  so we convert: similarity = 1 - cosine_distance.

- Flow Pattern Comparison: Uses graph edit distance and D_eff trajectory
  correlation to quantify similarity between different configurations:

      similarity = correlation(D_eff_trajectory_1, D_eff_trajectory_2)

IMPLEMENTATION NOTES
====================
- Uses NetworkX's DiGraph for directed information flow (temporal ordering).
- Node attributes: agent name, turn index, D_eff, beta values.
- Edge attributes: conveyance weight (both 'weight' and 'conveyance' keys).
- Layout reproducibility: All NetworkX layout functions use seed=42.
- Visualization: Always call plt.close() after savefig() to prevent memory leaks.
- Imports D_eff and beta calculations from conveyance_metrics module (DRY principle).
- L2 normalization applied before D_eff calculation per conveyance_metrics pattern.

Integration: Designed to work with TheLoom's conveyance_metrics and layer_utils
modules. Complements single-model interpretability with multi-agent analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# Import D_eff and beta calculations from conveyance_metrics (DRY principle)
from .conveyance_metrics import calculate_beta, calculate_d_eff


# ============================================================================
# Constants
# ============================================================================

# Default random seed for reproducible graph layouts.
# Used in nx.spring_layout() and other layout functions.
DEFAULT_LAYOUT_SEED = 42

# Default figure DPI for publication-quality visualizations.
# 300 DPI is standard for print publications; >600 is unnecessary.
DEFAULT_FIGURE_DPI = 300

# Default figure size in inches (width, height).
# Sized for comfortable viewing and clear labels in publications.
DEFAULT_FIGURE_SIZE = (12, 10)

# Threshold for classifying D_eff drops as significant bottlenecks.
# A drop is significant if drop_ratio > mean_drop × threshold_multiplier.
DEFAULT_BOTTLENECK_THRESHOLD = 1.5

# Minimum number of nodes required for meaningful bottleneck analysis.
# With fewer nodes, betweenness centrality is undefined or trivial.
MIN_NODES_FOR_BOTTLENECK = 3

# Minimum number of edges required for meaningful flow analysis.
# Single-agent conversations have no edges.
MIN_EDGES_FOR_FLOW = 1

# Warning threshold for large graphs where layout computation becomes slow.
# Consider using kamada_kawai_layout instead of spring_layout above this.
LARGE_GRAPH_NODE_THRESHOLD = 1000

# Default k parameter for spring layout (optimal distance between nodes).
# Higher values spread nodes further apart.
DEFAULT_SPRING_LAYOUT_K = 0.5

# Minimum D_eff value to avoid division by zero in relative drop calculations.
# Absolute drops are used when either D_eff value is below this threshold.
MIN_D_EFF_FOR_RELATIVE = 1


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class AgentFlowGraphResult:
    """Results from constructing an agent information flow graph.

    Contains the directed graph representing information flow between agents
    in a multi-turn conversation, along with summary statistics and metrics.

    Interpretation:
    - graph: NetworkX DiGraph with nodes (agent states) and edges (information flow)
    - n_nodes: Total number of agent states (turns) in the conversation
    - n_edges: Number of information transfers between agents
    - agents: Set of unique agent names participating in the conversation

    Node Attributes:
    - agent: Name of the agent at this turn
    - turn: Turn index in the conversation
    - d_eff: Effective dimensionality of the hidden state
    - beta: Conveyance metric beta value

    Edge Attributes:
    - weight: Edge weight based on conveyance (for graph algorithms)
    - conveyance: Same as weight (explicit semantic name)

    Use Cases:
    - Visualize multi-agent conversation flow
    - Analyze information bottlenecks between agents
    - Compare conversation patterns across different agent configurations
    """

    graph: nx.DiGraph  # The information flow graph
    n_nodes: int  # Number of nodes (agent turns)
    n_edges: int  # Number of edges (information transfers)
    agents: set[str]  # Unique agent names in the graph
    turn_sequence: list[str]  # Agent names in turn order
    d_eff_by_turn: dict[int, int]  # Turn index -> D_eff value
    beta_by_turn: dict[int, float]  # Turn index -> beta value
    total_conveyance: float  # Sum of all edge weights
    mean_conveyance: float  # Average edge weight
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        """Number of unique agents in the conversation."""
        return len(self.agents)

    @property
    def is_multi_agent(self) -> bool:
        """Check if this is a multi-agent conversation (>1 agent)."""
        return self.n_agents > 1

    @property
    def is_connected(self) -> bool:
        """Check if the graph is weakly connected.

        A weakly connected graph has a path between any two nodes
        when edge directions are ignored.
        """
        if self.n_nodes == 0:
            return False
        return nx.is_weakly_connected(self.graph)

    @property
    def has_sufficient_data(self) -> bool:
        """Check if graph has enough data for meaningful analysis.

        Requires at least MIN_NODES_FOR_BOTTLENECK nodes and
        MIN_EDGES_FOR_FLOW edges.
        """
        return (
            self.n_nodes >= MIN_NODES_FOR_BOTTLENECK
            and self.n_edges >= MIN_EDGES_FOR_FLOW
        )

    @property
    def conversation_length(self) -> int:
        """Number of turns in the conversation."""
        return len(self.turn_sequence)

    @property
    def d_eff_trajectory(self) -> list[int]:
        """D_eff values in turn order."""
        return [self.d_eff_by_turn[i] for i in sorted(self.d_eff_by_turn.keys())]


@dataclass
class AgentBottleneckResult:
    """Results from identifying information bottlenecks in agent flow.

    Bottleneck transitions are agent-to-agent handoffs where D_eff drops
    significantly, indicating potential information compression or loss
    during the transfer.

    Interpretation:
    - bottleneck_edges: List of (source, target) node pairs that are bottlenecks
    - d_eff_drops: Magnitude of D_eff drop at each bottleneck
    - centrality_scores: Betweenness centrality of each node
    - severity: Classification of overall bottleneck severity

    Detection Algorithm:
    A transition (i, j) is a bottleneck if:
        D_eff_drop(i→j) > threshold × mean(D_eff_drops)
    where D_eff_drop = D_eff_i - D_eff_j when D_eff_j < D_eff_i.

    Use Cases:
    - Identify where information is lost between agents
    - Find critical transitions that need attention
    - Compare bottleneck patterns across configurations
    """

    bottleneck_edges: list[tuple[str, str]]  # (source_node, target_node) pairs
    d_eff_drops: dict[tuple[str, str], int]  # Edge -> absolute D_eff drop
    relative_drops: dict[tuple[str, str], float]  # Edge -> relative drop (0-1)
    centrality_scores: dict[str, float]  # Node -> betweenness centrality
    threshold_used: float  # Threshold multiplier used for detection
    n_bottlenecks: int  # Number of bottlenecks found
    max_drop_edge: tuple[str, str] | None  # Edge with largest D_eff drop
    max_drop_value: int  # Maximum absolute D_eff drop
    mean_d_eff_drop: float  # Mean D_eff drop across all transitions
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_bottlenecks(self) -> bool:
        """Check if any bottlenecks were detected."""
        return self.n_bottlenecks > 0

    @property
    def severity(self) -> str:
        """Classify the overall severity of bottlenecks.

        Categories:
        - none: No bottlenecks detected
        - mild: 1 bottleneck with < 30% relative drop
        - moderate: 2-3 bottlenecks or single > 30% drop
        - severe: >3 bottlenecks or maximum drop > 50%
        """
        if self.n_bottlenecks == 0:
            return "none"
        max_relative = max(self.relative_drops.values()) if self.relative_drops else 0.0
        if max_relative > 0.50 or self.n_bottlenecks > 3:
            return "severe"
        elif max_relative > 0.30 or self.n_bottlenecks > 1:
            return "moderate"
        else:
            return "mild"

    @property
    def bottleneck_agents(self) -> set[str]:
        """Get unique agents involved in bottleneck transitions.

        Extracts agent names from bottleneck edge node identifiers.
        Assumes node format "agent_turn" or similar.
        """
        agents: set[str] = set()
        for source, target in self.bottleneck_edges:
            # Nodes are typically formatted as "agent_turn" or just agent names
            for node in (source, target):
                # Try to extract agent name (before last underscore if present)
                if "_" in node:
                    agent = "_".join(node.split("_")[:-1])
                else:
                    agent = node
                agents.add(agent)
        return agents


@dataclass
class FlowComparisonResult:
    """Results from comparing two agent flow graphs.

    Compares information flow patterns between two different multi-agent
    conversations or configurations, quantifying structural and metric
    differences.

    Interpretation:
    - trajectory_correlation: How similarly D_eff evolves across turns
    - structural_similarity: Graph structure similarity (0-1)
    - agent_overlap: Fraction of agents common to both graphs
    - divergence_quality: Classification of overall difference

    Use Cases:
    - Compare same agents with different prompts
    - Compare different agent configurations
    - Track changes in flow patterns over time
    """

    graph_a_stats: dict[str, Any]  # Summary stats for graph A
    graph_b_stats: dict[str, Any]  # Summary stats for graph B
    common_agents: set[str]  # Agents present in both graphs
    agents_only_in_a: set[str]  # Agents only in graph A
    agents_only_in_b: set[str]  # Agents only in graph B
    trajectory_correlation: float  # Pearson correlation of D_eff trajectories
    d_eff_mean_diff: float  # Mean D_eff difference (A - B)
    d_eff_abs_mean_diff: float  # Mean absolute D_eff difference
    conveyance_ratio: float  # Ratio of total conveyance (A / B)
    structural_similarity: float  # Graph structure similarity (0-1)
    n_turns_a: int  # Number of turns in graph A
    n_turns_b: int  # Number of turns in graph B
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_overlap(self) -> float:
        """Fraction of agents that appear in both graphs.

        Computed as |common| / |union| (Jaccard similarity).
        Returns 1.0 if both graphs have the same agents.
        """
        all_agents = (
            self.common_agents | self.agents_only_in_a | self.agents_only_in_b
        )
        if not all_agents:
            return 0.0
        return len(self.common_agents) / len(all_agents)

    @property
    def is_similar(self) -> bool:
        """Check if the two flow patterns are similar.

        Considers patterns similar if:
        - trajectory_correlation > 0.8
        - structural_similarity > 0.7
        - agent_overlap > 0.8
        """
        return (
            self.trajectory_correlation > 0.8
            and self.structural_similarity > 0.7
            and self.agent_overlap > 0.8
        )

    @property
    def divergence_quality(self) -> str:
        """Classify the level of divergence between flow patterns.

        Categories:
        - identical: Perfect match in all metrics
        - similar: High correlation and structural similarity
        - moderate: Some differences but recognizable pattern
        - divergent: Significantly different patterns
        """
        if (
            self.trajectory_correlation > 0.99
            and self.structural_similarity > 0.99
            and self.agent_overlap == 1.0
        ):
            return "identical"
        elif self.is_similar:
            return "similar"
        elif self.trajectory_correlation > 0.5 and self.structural_similarity > 0.5:
            return "moderate"
        else:
            return "divergent"

    @property
    def length_ratio(self) -> float:
        """Ratio of conversation lengths (A / B).

        Values close to 1.0 indicate similar conversation lengths.
        """
        if self.n_turns_b == 0:
            return float("inf") if self.n_turns_a > 0 else 0.0
        return self.n_turns_a / self.n_turns_b


@dataclass
class AgentAlignmentResult:
    """Results from measuring semantic alignment between agents.

    Alignment is measured via cosine similarity of hidden states,
    indicating how similarly different agents represent information.

    Interpretation:
    - pairwise_alignment: Cosine similarity between each agent pair
    - mean_alignment: Average alignment across all pairs
    - alignment_matrix: Full NxN similarity matrix

    Alignment Values:
    - 1.0: Identical representations
    - 0.0: Orthogonal representations
    - -1.0: Opposite representations (rare in practice)

    Note: Uses cosine similarity (1 - cosine_distance), not distance.

    Use Cases:
    - Identify agents with similar internal representations
    - Detect semantic drift across conversation turns
    - Measure convergence/divergence of agent understanding
    """

    agent_names: list[str]  # Ordered list of agent names
    alignment_matrix: NDArray[np.floating[Any]]  # NxN cosine similarity matrix
    pairwise_alignment: dict[tuple[str, str], float]  # (agent_i, agent_j) -> similarity
    mean_alignment: float  # Mean of all pairwise alignments
    min_alignment: float  # Minimum pairwise alignment
    max_alignment: float  # Maximum pairwise alignment (excluding self)
    min_pair: tuple[str, str] | None  # Agent pair with minimum alignment
    max_pair: tuple[str, str] | None  # Agent pair with maximum alignment
    std_alignment: float  # Standard deviation of alignments
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        """Number of agents in the alignment analysis."""
        return len(self.agent_names)

    @property
    def is_well_aligned(self) -> bool:
        """Check if agents are generally well-aligned.

        Considers well-aligned if mean alignment > 0.7 and
        minimum alignment > 0.3.
        """
        return self.mean_alignment > 0.7 and self.min_alignment > 0.3

    @property
    def alignment_quality(self) -> str:
        """Classify the overall alignment quality.

        Categories:
        - excellent: Mean > 0.9, min > 0.7
        - good: Mean > 0.7, min > 0.5
        - moderate: Mean > 0.5, min > 0.2
        - poor: Mean <= 0.5 or min <= 0.2
        """
        if self.mean_alignment > 0.9 and self.min_alignment > 0.7:
            return "excellent"
        elif self.mean_alignment > 0.7 and self.min_alignment > 0.5:
            return "good"
        elif self.mean_alignment > 0.5 and self.min_alignment > 0.2:
            return "moderate"
        else:
            return "poor"

    @property
    def has_outlier(self) -> bool:
        """Check if there's an agent with unusually low alignment.

        Detects outliers where any pairwise alignment is more than
        2 standard deviations below the mean.
        """
        if self.std_alignment == 0:
            return False
        return self.min_alignment < (self.mean_alignment - 2 * self.std_alignment)

    def get_alignment(self, agent_a: str, agent_b: str) -> float:
        """Get alignment between two specific agents.

        Parameters:
            agent_a: Name of first agent.
            agent_b: Name of second agent.

        Returns:
            Cosine similarity between the two agents.

        Raises:
            KeyError: If either agent is not in the alignment result.
        """
        if agent_a == agent_b:
            return 1.0
        # Try both orderings since pairwise_alignment may have either
        if (agent_a, agent_b) in self.pairwise_alignment:
            return self.pairwise_alignment[(agent_a, agent_b)]
        elif (agent_b, agent_a) in self.pairwise_alignment:
            return self.pairwise_alignment[(agent_b, agent_a)]
        else:
            raise KeyError(
                f"Alignment not found for agents '{agent_a}' and '{agent_b}'"
            )


# ============================================================================
# Core Analysis Functions
# ============================================================================

# Core functions will be implemented in subtasks 2-3 through 2-6
