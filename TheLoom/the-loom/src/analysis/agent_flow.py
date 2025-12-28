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


def build_agent_flow_graph(
    agent_states: dict[str, dict[int, NDArray[np.floating[Any]]]],
    conveyance_edges: dict[tuple[str, str], float],
    variance_threshold: float = 0.90,
) -> AgentFlowGraphResult:
    """Build directed graph of agent information flow.

    Constructs a NetworkX DiGraph representing semantic information transfer
    between agents across conversation turns. Nodes represent agent states
    (with D_eff and beta attributes), edges represent information transfer
    (with conveyance weights).

    GRAPH STRUCTURE
    ===============
    - Nodes: Named "{agent}_t{turn}" (e.g., "assistant_t0", "user_t1")
    - Node attributes: agent (str), turn (int), d_eff (int), beta (float)
    - Edges: Connect consecutive turns in temporal order
    - Edge attributes: weight (float), conveyance (float)

    Parameters:
        agent_states: dict[str, dict[int, NDArray]]
            Mapping from agent name to turn-indexed hidden states.
            Format: {agent_name: {turn_index: hidden_state_array}}
            Each hidden_state_array should be 2D (n_samples, n_features).
        conveyance_edges: dict[tuple[str, str], float]
            Pre-computed conveyance weights for agent-to-agent transitions.
            Format: {(from_agent, to_agent): conveyance_value}
            If empty, edges are created with weight 1.0 between consecutive turns.
        variance_threshold: float, default=0.90
            Variance threshold for D_eff calculation.

    Returns:
        AgentFlowGraphResult containing:
        - graph: nx.DiGraph with nodes and edges
        - n_nodes: Number of nodes (agent turns)
        - n_edges: Number of edges (information transfers)
        - agents: Set of unique agent names
        - turn_sequence: Agent names in turn order
        - d_eff_by_turn: Turn index -> D_eff value
        - beta_by_turn: Turn index -> beta value
        - total_conveyance: Sum of all edge weights
        - mean_conveyance: Average edge weight

    Example:
        >>> import numpy as np
        >>> # Create sample hidden states for two agents
        >>> agent_states = {
        ...     "user": {0: np.random.randn(10, 768), 2: np.random.randn(10, 768)},
        ...     "assistant": {1: np.random.randn(10, 768), 3: np.random.randn(10, 768)},
        ... }
        >>> conveyance_edges = {("user", "assistant"): 0.8, ("assistant", "user"): 0.7}
        >>> result = build_agent_flow_graph(agent_states, conveyance_edges)
        >>> print(f"Nodes: {result.n_nodes}, Edges: {result.n_edges}")
        >>> print(f"Agents: {result.agents}")

    Notes:
        - Empty agent_states returns an empty graph (no error)
        - Single-agent conversations have no edges
        - Missing conveyance values default to 1.0
        - L2 normalization is applied before D_eff calculation
    """
    # Initialize the directed graph
    G = nx.DiGraph()

    # Handle empty input
    if not agent_states:
        return AgentFlowGraphResult(
            graph=G,
            n_nodes=0,
            n_edges=0,
            agents=set(),
            turn_sequence=[],
            d_eff_by_turn={},
            beta_by_turn={},
            total_conveyance=0.0,
            mean_conveyance=0.0,
            metadata={"warning": "empty_input"},
        )

    # Collect all turns with their agent names for temporal ordering
    # Format: list of (turn_index, agent_name, hidden_state)
    all_turns: list[tuple[int, str, NDArray[np.floating[Any]]]] = []

    for agent_name, turn_states in agent_states.items():
        for turn_idx, hidden_state in turn_states.items():
            all_turns.append((turn_idx, agent_name, hidden_state))

    # Sort by turn index to establish temporal order
    all_turns.sort(key=lambda x: x[0])

    # Track unique agents and turn sequence
    agents: set[str] = set()
    turn_sequence: list[str] = []
    d_eff_by_turn: dict[int, int] = {}
    beta_by_turn: dict[int, float] = {}

    # Add nodes with D_eff and beta attributes
    for turn_idx, agent_name, hidden_state in all_turns:
        node_id = f"{agent_name}_t{turn_idx}"

        # Ensure hidden_state is 2D for D_eff calculation
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)

        # Calculate D_eff and beta using conveyance_metrics module
        d_eff = calculate_d_eff(hidden_state, variance_threshold=variance_threshold)
        beta = calculate_beta(hidden_state, variance_threshold=variance_threshold)

        # Add node with attributes
        G.add_node(
            node_id,
            agent=agent_name,
            turn=turn_idx,
            d_eff=d_eff,
            beta=beta,
        )

        # Track metadata
        agents.add(agent_name)
        turn_sequence.append(agent_name)
        d_eff_by_turn[turn_idx] = d_eff
        beta_by_turn[turn_idx] = beta

    # Add edges between consecutive turns (temporal information flow)
    total_conveyance = 0.0
    n_edges = 0

    for i in range(len(all_turns) - 1):
        turn_idx_from, agent_from, _ = all_turns[i]
        turn_idx_to, agent_to, _ = all_turns[i + 1]

        node_from = f"{agent_from}_t{turn_idx_from}"
        node_to = f"{agent_to}_t{turn_idx_to}"

        # Look up conveyance weight from pre-computed edges
        # Try both (from, to) key format
        conveyance_key = (agent_from, agent_to)
        if conveyance_key in conveyance_edges:
            weight = conveyance_edges[conveyance_key]
        else:
            # Default weight if not specified
            weight = 1.0

        # Add edge with both 'weight' and 'conveyance' attributes
        G.add_edge(
            node_from,
            node_to,
            weight=weight,
            conveyance=weight,
        )

        total_conveyance += weight
        n_edges += 1

    # Calculate mean conveyance
    mean_conveyance = total_conveyance / n_edges if n_edges > 0 else 0.0

    return AgentFlowGraphResult(
        graph=G,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        agents=agents,
        turn_sequence=turn_sequence,
        d_eff_by_turn=d_eff_by_turn,
        beta_by_turn=beta_by_turn,
        total_conveyance=total_conveyance,
        mean_conveyance=mean_conveyance,
    )


def find_agent_bottlenecks(
    G: nx.DiGraph,
    threshold: float = DEFAULT_BOTTLENECK_THRESHOLD,
) -> AgentBottleneckResult:
    """Identify information bottlenecks in agent communication.

    Analyzes the agent flow graph to find transitions where D_eff drops
    significantly, indicating potential information compression or loss
    during agent-to-agent handoffs.

    DETECTION ALGORITHM
    ===================
    For each edge (source, target) in the flow graph:
    1. Compute D_eff drop: D_eff_source - D_eff_target (only if positive)
    2. Compute relative drop: drop / D_eff_source
    3. Calculate mean D_eff drop across all positive drops
    4. An edge is a bottleneck if:
       drop > threshold × mean_drop

    Parameters:
        G: nx.DiGraph
            Agent information flow graph with D_eff node attributes.
            Expected node attributes: 'd_eff' (int).
            Typically created by build_agent_flow_graph().
        threshold: float, default=1.5
            Threshold multiplier for bottleneck detection.
            A transition is a bottleneck if its D_eff drop exceeds
            threshold × mean(D_eff_drops) across all transitions.
            Higher values are more selective (fewer bottlenecks).

    Returns:
        AgentBottleneckResult with detected bottlenecks:
        - bottleneck_edges: List of (source, target) node pairs
        - d_eff_drops: Absolute D_eff drop at each bottleneck
        - relative_drops: Relative drop (0-1) at each bottleneck
        - centrality_scores: Betweenness centrality of each node
        - severity: Classification of overall bottleneck severity

    Example:
        >>> import networkx as nx
        >>> from src.analysis.agent_flow import find_agent_bottlenecks
        >>> # Create a flow graph with D_eff drops
        >>> G = nx.DiGraph()
        >>> G.add_node("agent_a_t0", d_eff=100)
        >>> G.add_node("agent_b_t1", d_eff=50)  # 50% drop - bottleneck!
        >>> G.add_node("agent_a_t2", d_eff=45)  # Small drop
        >>> G.add_edge("agent_a_t0", "agent_b_t1")
        >>> G.add_edge("agent_b_t1", "agent_a_t2")
        >>> result = find_agent_bottlenecks(G)
        >>> print(f"Bottlenecks found: {result.n_bottlenecks}")
        >>> print(f"Severity: {result.severity}")

    Notes:
        - Empty graphs return empty results (no error)
        - Graphs with fewer than MIN_NODES_FOR_BOTTLENECK nodes return empty results
        - Betweenness centrality is computed to identify critical nodes
        - Nodes missing 'd_eff' attribute are skipped with relative_drops=0
    """
    # Handle empty or insufficient graph
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < MIN_NODES_FOR_BOTTLENECK or n_edges < MIN_EDGES_FOR_FLOW:
        return AgentBottleneckResult(
            bottleneck_edges=[],
            d_eff_drops={},
            relative_drops={},
            centrality_scores={},
            threshold_used=threshold,
            n_bottlenecks=0,
            max_drop_edge=None,
            max_drop_value=0,
            mean_d_eff_drop=0.0,
            metadata={"warning": "insufficient_data", "n_nodes": n_nodes, "n_edges": n_edges},
        )

    # Compute betweenness centrality for all nodes
    # This helps identify nodes that are critical for information flow
    centrality_scores: dict[str, float] = nx.betweenness_centrality(G)

    # Collect D_eff drops for all edges
    all_drops: list[int] = []
    edge_drops: dict[tuple[str, str], int] = {}
    edge_relative_drops: dict[tuple[str, str], float] = {}

    for source, target in G.edges():
        # Get D_eff values from node attributes
        source_d_eff = G.nodes[source].get("d_eff", 0)
        target_d_eff = G.nodes[target].get("d_eff", 0)

        # Only consider positive drops (information compression)
        drop = source_d_eff - target_d_eff
        if drop > 0:
            all_drops.append(drop)
            edge_drops[(source, target)] = drop

            # Compute relative drop (avoid division by zero)
            if source_d_eff >= MIN_D_EFF_FOR_RELATIVE:
                relative_drop = drop / source_d_eff
            else:
                relative_drop = 0.0
            edge_relative_drops[(source, target)] = relative_drop

    # If no positive drops, return empty result
    if not all_drops:
        return AgentBottleneckResult(
            bottleneck_edges=[],
            d_eff_drops={},
            relative_drops={},
            centrality_scores=centrality_scores,
            threshold_used=threshold,
            n_bottlenecks=0,
            max_drop_edge=None,
            max_drop_value=0,
            mean_d_eff_drop=0.0,
            metadata={"info": "no_positive_drops"},
        )

    # Calculate mean D_eff drop for threshold comparison
    mean_d_eff_drop = float(np.mean(all_drops))

    # Identify bottleneck edges: drop > threshold × mean_drop
    bottleneck_threshold = threshold * mean_d_eff_drop
    bottleneck_edges: list[tuple[str, str]] = []
    bottleneck_d_eff_drops: dict[tuple[str, str], int] = {}
    bottleneck_relative_drops: dict[tuple[str, str], float] = {}

    for edge, drop in edge_drops.items():
        if drop > bottleneck_threshold:
            bottleneck_edges.append(edge)
            bottleneck_d_eff_drops[edge] = drop
            bottleneck_relative_drops[edge] = edge_relative_drops[edge]

    # Find max drop edge
    if bottleneck_d_eff_drops:
        max_drop_edge = max(bottleneck_d_eff_drops, key=bottleneck_d_eff_drops.get)  # type: ignore
        max_drop_value = bottleneck_d_eff_drops[max_drop_edge]
    else:
        max_drop_edge = None
        max_drop_value = 0

    return AgentBottleneckResult(
        bottleneck_edges=bottleneck_edges,
        d_eff_drops=bottleneck_d_eff_drops,
        relative_drops=bottleneck_relative_drops,
        centrality_scores=centrality_scores,
        threshold_used=threshold,
        n_bottlenecks=len(bottleneck_edges),
        max_drop_edge=max_drop_edge,
        max_drop_value=max_drop_value,
        mean_d_eff_drop=mean_d_eff_drop,
    )


def compare_flow_patterns(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
) -> FlowComparisonResult:
    """Compare information flow patterns between two agent graphs.

    Quantifies similarity between two multi-agent conversation flow graphs
    by comparing their D_eff trajectories, conveyance metrics, agent
    participation, and structural properties.

    COMPARISON METRICS
    ==================
    1. Trajectory Correlation: Pearson correlation of D_eff values across
       turns. High correlation indicates similar information compression
       patterns even if absolute values differ.

    2. Structural Similarity: Based on normalized graph edit distance
       considering node/edge counts. Measures how similar the graph
       topologies are.

    3. Agent Overlap: Jaccard similarity of agent sets. Indicates how
       much overlap in participating agents between conversations.

    4. Conveyance Ratio: Ratio of total edge weights (conveyance).
       Values near 1.0 indicate similar overall information transfer.

    Parameters:
        G1: nx.DiGraph
            First agent information flow graph.
            Expected node attributes: 'agent', 'turn', 'd_eff'.
            Expected edge attributes: 'weight' or 'conveyance'.
        G2: nx.DiGraph
            Second agent information flow graph (same format).

    Returns:
        FlowComparisonResult containing:
        - trajectory_correlation: Pearson correlation of D_eff trajectories
        - structural_similarity: Graph structure similarity (0-1)
        - agent_overlap: Jaccard similarity of agent sets (via property)
        - conveyance_ratio: Ratio of total conveyance (G1 / G2)
        - d_eff_mean_diff: Mean D_eff difference (G1 - G2)
        - divergence_quality: Classification ("identical", "similar", etc.)

    Example:
        >>> import networkx as nx
        >>> from src.analysis.agent_flow import compare_flow_patterns
        >>> # Create two similar flow graphs
        >>> G1 = nx.DiGraph()
        >>> G1.add_node("user_t0", agent="user", turn=0, d_eff=100)
        >>> G1.add_node("assistant_t1", agent="assistant", turn=1, d_eff=90)
        >>> G1.add_edge("user_t0", "assistant_t1", weight=0.8)
        >>> G2 = nx.DiGraph()
        >>> G2.add_node("user_t0", agent="user", turn=0, d_eff=95)
        >>> G2.add_node("assistant_t1", agent="assistant", turn=1, d_eff=85)
        >>> G2.add_edge("user_t0", "assistant_t1", weight=0.75)
        >>> result = compare_flow_patterns(G1, G2)
        >>> print(f"Trajectory correlation: {result.trajectory_correlation:.4f}")
        >>> print(f"Divergence: {result.divergence_quality}")

    Notes:
        - Empty graphs return zero correlation and similarity
        - Graphs of different lengths can still be compared via correlation
        - Missing 'd_eff' node attributes default to 0
        - Missing 'weight'/'conveyance' edge attributes default to 1.0
    """
    # Extract graph statistics
    graph_a_stats = _compute_graph_stats(G1)
    graph_b_stats = _compute_graph_stats(G2)

    # Extract agent sets
    agents_a = _get_agents_from_graph(G1)
    agents_b = _get_agents_from_graph(G2)

    common_agents = agents_a & agents_b
    agents_only_in_a = agents_a - agents_b
    agents_only_in_b = agents_b - agents_a

    # Extract D_eff trajectories (ordered by turn)
    trajectory_a = _get_d_eff_trajectory(G1)
    trajectory_b = _get_d_eff_trajectory(G2)

    # Compute trajectory correlation
    trajectory_correlation = _compute_trajectory_correlation(trajectory_a, trajectory_b)

    # Compute D_eff differences
    d_eff_mean_diff, d_eff_abs_mean_diff = _compute_d_eff_differences(
        trajectory_a, trajectory_b
    )

    # Compute conveyance ratio
    conveyance_a = _compute_total_conveyance(G1)
    conveyance_b = _compute_total_conveyance(G2)

    if conveyance_b > 0:
        conveyance_ratio = conveyance_a / conveyance_b
    elif conveyance_a > 0:
        conveyance_ratio = float("inf")
    else:
        conveyance_ratio = 1.0  # Both zero

    # Compute structural similarity
    structural_similarity = _compute_structural_similarity(G1, G2)

    # Get turn counts
    n_turns_a = G1.number_of_nodes()
    n_turns_b = G2.number_of_nodes()

    return FlowComparisonResult(
        graph_a_stats=graph_a_stats,
        graph_b_stats=graph_b_stats,
        common_agents=common_agents,
        agents_only_in_a=agents_only_in_a,
        agents_only_in_b=agents_only_in_b,
        trajectory_correlation=trajectory_correlation,
        d_eff_mean_diff=d_eff_mean_diff,
        d_eff_abs_mean_diff=d_eff_abs_mean_diff,
        conveyance_ratio=conveyance_ratio,
        structural_similarity=structural_similarity,
        n_turns_a=n_turns_a,
        n_turns_b=n_turns_b,
    )


# ============================================================================
# Helper Functions for Flow Comparison
# ============================================================================


def _compute_graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    """Compute summary statistics for an agent flow graph.

    Parameters:
        G: nx.DiGraph - Agent flow graph with node/edge attributes.

    Returns:
        dict with summary statistics including node count, edge count,
        agent count, total conveyance, and D_eff statistics.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Extract agents
    agents = _get_agents_from_graph(G)

    # Extract D_eff values
    d_eff_values = [G.nodes[node].get("d_eff", 0) for node in G.nodes()]

    # Compute D_eff statistics
    if d_eff_values:
        mean_d_eff = float(np.mean(d_eff_values))
        std_d_eff = float(np.std(d_eff_values))
        min_d_eff = int(min(d_eff_values))
        max_d_eff = int(max(d_eff_values))
    else:
        mean_d_eff = 0.0
        std_d_eff = 0.0
        min_d_eff = 0
        max_d_eff = 0

    # Compute total conveyance
    total_conveyance = _compute_total_conveyance(G)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_agents": len(agents),
        "agents": list(agents),
        "total_conveyance": total_conveyance,
        "mean_d_eff": mean_d_eff,
        "std_d_eff": std_d_eff,
        "min_d_eff": min_d_eff,
        "max_d_eff": max_d_eff,
    }


def _get_agents_from_graph(G: nx.DiGraph) -> set[str]:
    """Extract unique agent names from graph node attributes.

    Parameters:
        G: nx.DiGraph - Agent flow graph with 'agent' node attributes.

    Returns:
        Set of unique agent names found in the graph.
    """
    agents: set[str] = set()
    for node in G.nodes():
        agent = G.nodes[node].get("agent")
        if agent is not None:
            agents.add(str(agent))
    return agents


def _get_d_eff_trajectory(G: nx.DiGraph) -> list[int]:
    """Extract D_eff values ordered by turn index.

    Parameters:
        G: nx.DiGraph - Agent flow graph with 'turn' and 'd_eff' node attributes.

    Returns:
        List of D_eff values in turn order. Empty list if no nodes.
    """
    if G.number_of_nodes() == 0:
        return []

    # Collect (turn, d_eff) pairs
    turn_d_eff_pairs: list[tuple[int, int]] = []
    for node in G.nodes():
        turn = G.nodes[node].get("turn", 0)
        d_eff = G.nodes[node].get("d_eff", 0)
        turn_d_eff_pairs.append((turn, d_eff))

    # Sort by turn and extract D_eff values
    turn_d_eff_pairs.sort(key=lambda x: x[0])
    return [d_eff for _, d_eff in turn_d_eff_pairs]


def _compute_trajectory_correlation(
    trajectory_a: list[int],
    trajectory_b: list[int],
) -> float:
    """Compute Pearson correlation between two D_eff trajectories.

    For trajectories of different lengths, compares the overlapping portion
    (shorter trajectory determines comparison length).

    Parameters:
        trajectory_a: D_eff values in turn order for graph A.
        trajectory_b: D_eff values in turn order for graph B.

    Returns:
        Pearson correlation coefficient in [-1, 1].
        Returns 0.0 for empty trajectories.
        Returns 1.0 if both trajectories are identical constants.
    """
    # Handle empty trajectories
    if not trajectory_a or not trajectory_b:
        return 0.0

    # Use minimum length for comparison
    min_len = min(len(trajectory_a), len(trajectory_b))

    if min_len < 2:
        # Cannot compute correlation with < 2 points
        # Return 1.0 if values are equal, 0.0 otherwise
        if min_len == 1:
            return 1.0 if trajectory_a[0] == trajectory_b[0] else 0.0
        return 0.0

    vals_a = np.array(trajectory_a[:min_len], dtype=float)
    vals_b = np.array(trajectory_b[:min_len], dtype=float)

    # Handle constant trajectories
    std_a = np.std(vals_a)
    std_b = np.std(vals_b)

    if std_a == 0 and std_b == 0:
        # Both constant - identical if same value
        return 1.0 if vals_a[0] == vals_b[0] else 0.0
    elif std_a == 0 or std_b == 0:
        # One constant, one varying - undefined correlation
        return 0.0

    # Compute Pearson correlation
    correlation = float(np.corrcoef(vals_a, vals_b)[0, 1])

    # Handle NaN (shouldn't occur with std checks, but be safe)
    if np.isnan(correlation):
        return 0.0

    return correlation


def _compute_d_eff_differences(
    trajectory_a: list[int],
    trajectory_b: list[int],
) -> tuple[float, float]:
    """Compute mean and absolute mean D_eff differences.

    Parameters:
        trajectory_a: D_eff values in turn order for graph A.
        trajectory_b: D_eff values in turn order for graph B.

    Returns:
        tuple of (mean_diff, abs_mean_diff) where:
        - mean_diff: Average of (A - B) differences
        - abs_mean_diff: Average of absolute differences
        Returns (0.0, 0.0) for empty or mismatched-length trajectories.
    """
    if not trajectory_a or not trajectory_b:
        return 0.0, 0.0

    # Use minimum length for comparison
    min_len = min(len(trajectory_a), len(trajectory_b))

    vals_a = np.array(trajectory_a[:min_len], dtype=float)
    vals_b = np.array(trajectory_b[:min_len], dtype=float)

    diffs = vals_a - vals_b
    mean_diff = float(np.mean(diffs))
    abs_mean_diff = float(np.mean(np.abs(diffs)))

    return mean_diff, abs_mean_diff


def _compute_total_conveyance(G: nx.DiGraph) -> float:
    """Compute total conveyance (sum of edge weights) for a graph.

    Parameters:
        G: nx.DiGraph - Agent flow graph with 'weight' or 'conveyance' edge attributes.

    Returns:
        Sum of all edge weights. Returns 0.0 for empty graph.
    """
    total = 0.0
    for _, _, data in G.edges(data=True):
        # Try 'weight' first, then 'conveyance', default to 1.0
        weight = data.get("weight", data.get("conveyance", 1.0))
        total += float(weight)
    return total


def _compute_structural_similarity(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
) -> float:
    """Compute structural similarity between two graphs.

    Uses a combination of node count ratio, edge count ratio, and
    edge density comparison to approximate structural similarity
    without expensive graph edit distance computation.

    The similarity score is computed as:
        similarity = (node_sim + edge_sim + density_sim) / 3

    where each component is min(x, y) / max(x, y) for the respective metric.

    Parameters:
        G1: First graph.
        G2: Second graph.

    Returns:
        Structural similarity in [0, 1]. 1.0 means identical structure.
        Returns 1.0 if both graphs are empty.
    """
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    e1 = G1.number_of_edges()
    e2 = G2.number_of_edges()

    # Handle empty graphs
    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0

    # Node count similarity: min/max ratio
    node_sim = min(n1, n2) / max(n1, n2)

    # Edge count similarity: min/max ratio (handle 0 edges)
    if e1 == 0 and e2 == 0:
        edge_sim = 1.0
    elif e1 == 0 or e2 == 0:
        edge_sim = 0.0
    else:
        edge_sim = min(e1, e2) / max(e1, e2)

    # Density similarity: compare edge densities
    # Density = edges / (nodes * (nodes - 1)) for directed graph
    if n1 > 1:
        density1 = e1 / (n1 * (n1 - 1))
    else:
        density1 = 0.0

    if n2 > 1:
        density2 = e2 / (n2 * (n2 - 1))
    else:
        density2 = 0.0

    if density1 == 0 and density2 == 0:
        density_sim = 1.0
    elif density1 == 0 or density2 == 0:
        density_sim = 0.0
    else:
        density_sim = min(density1, density2) / max(density1, density2)

    # Average of all similarity components
    return (node_sim + edge_sim + density_sim) / 3.0


def visualize_agent_alignment(
    G: nx.DiGraph,
    output_path: str,
    figsize: tuple[int, int] = DEFAULT_FIGURE_SIZE,
    dpi: int = DEFAULT_FIGURE_DPI,
) -> AgentAlignmentResult | None:
    """Visualize semantic alignment between agents in a flow graph.

    Creates a publication-quality multi-panel visualization showing:
    1. Agent alignment matrix (heatmap of D_eff-based similarity)
    2. D_eff trajectory over conversation turns
    3. Graph layout with D_eff-colored nodes

    VISUALIZATION LAYOUT
    ====================
    The output is a 2x2 subplot grid:
    - Top-left: Alignment matrix heatmap (agents × agents)
    - Top-right: D_eff trajectory line plot (turn × D_eff)
    - Bottom-left: Network graph with spring layout
    - Bottom-right: Legend and summary statistics

    ALIGNMENT COMPUTATION
    =====================
    Agent alignment is computed based on D_eff similarity:

        alignment(i, j) = 1 - |D_eff_i - D_eff_j| / max(D_eff_i, D_eff_j)

    This measures how similarly two agents represent information
    dimensionally. Values range from 0 (very different) to 1 (identical).

    Parameters:
        G: nx.DiGraph
            Agent information flow graph with node attributes:
            - 'agent': Agent name (str)
            - 'turn': Turn index (int)
            - 'd_eff': Effective dimensionality (int)
            Typically created by build_agent_flow_graph().
        output_path: str
            Path to save the visualization (e.g., "alignment.png").
            Supports any format matplotlib can save (png, pdf, svg).
        figsize: tuple[int, int], default=(12, 10)
            Figure size in inches (width, height).
        dpi: int, default=300
            Dots per inch for the output image.
            300 is publication-quality; use 150 for drafts.

    Returns:
        AgentAlignmentResult with alignment metrics, or None if graph
        has insufficient data (< 2 nodes).

    Example:
        >>> import networkx as nx
        >>> from src.analysis.agent_flow import visualize_agent_alignment
        >>> # Create a flow graph
        >>> G = nx.DiGraph()
        >>> G.add_node("user_t0", agent="user", turn=0, d_eff=100)
        >>> G.add_node("assistant_t1", agent="assistant", turn=1, d_eff=90)
        >>> G.add_node("user_t2", agent="user", turn=2, d_eff=85)
        >>> G.add_edge("user_t0", "assistant_t1", weight=0.8)
        >>> G.add_edge("assistant_t1", "user_t2", weight=0.75)
        >>> result = visualize_agent_alignment(G, "alignment.png")
        >>> if result:
        ...     print(f"Mean alignment: {result.mean_alignment:.4f}")

    Notes:
        - Empty graphs produce a placeholder figure with warning text
        - Single-agent graphs produce alignment matrix with single entry
        - Always calls plt.close() after saving to prevent memory leaks
        - Uses seed=42 for reproducible graph layout
        - Applies plt.tight_layout() before saving to avoid clipped labels
    """
    import matplotlib.pyplot as plt

    # Handle empty graph
    if G.number_of_nodes() == 0:
        # Create placeholder figure with warning
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "No data to visualize\n(empty graph)",
            ha="center", va="center",
            fontsize=16, color="gray"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return None

    # Extract agent data from graph
    agents_data = _extract_agent_data(G)
    agent_names = sorted(agents_data.keys())

    # Single-node graph handling
    if G.number_of_nodes() == 1:
        node = list(G.nodes())[0]
        d_eff = G.nodes[node].get("d_eff", 0)
        agent = G.nodes[node].get("agent", "unknown")

        # Create simple figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            f"Single node: {agent}\nD_eff: {d_eff}",
            ha="center", va="center",
            fontsize=14
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        # Return minimal alignment result
        return AgentAlignmentResult(
            agent_names=[agent],
            alignment_matrix=np.array([[1.0]]),
            pairwise_alignment={},
            mean_alignment=1.0,
            min_alignment=1.0,
            max_alignment=1.0,
            min_pair=None,
            max_pair=None,
            std_alignment=0.0,
            metadata={"warning": "single_node"},
        )

    # Compute alignment matrix
    alignment_matrix, pairwise_alignment = _compute_alignment_matrix(
        agents_data, agent_names
    )

    # Calculate alignment statistics
    alignment_result = _create_alignment_result(
        agent_names, alignment_matrix, pairwise_alignment
    )

    # Create the visualization
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Alignment matrix heatmap (top-left)
    _plot_alignment_heatmap(axs[0, 0], alignment_matrix, agent_names)

    # Panel 2: D_eff trajectory (top-right)
    _plot_deff_trajectory(axs[0, 1], G)

    # Panel 3: Network graph (bottom-left)
    _plot_network_graph(axs[1, 0], G)

    # Panel 4: Summary statistics (bottom-right)
    _plot_summary_stats(axs[1, 1], alignment_result, G)

    # Apply tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()  # CRITICAL: Prevent memory leaks

    return alignment_result


# ============================================================================
# Helper Functions for Visualization
# ============================================================================


def _extract_agent_data(G: nx.DiGraph) -> dict[str, list[int]]:
    """Extract D_eff values grouped by agent name.

    Parameters:
        G: Agent flow graph with 'agent' and 'd_eff' node attributes.

    Returns:
        dict mapping agent name to list of D_eff values across turns.
    """
    agents_data: dict[str, list[int]] = {}

    for node in G.nodes():
        agent = G.nodes[node].get("agent", "unknown")
        d_eff = G.nodes[node].get("d_eff", 0)

        if agent not in agents_data:
            agents_data[agent] = []
        agents_data[agent].append(d_eff)

    return agents_data


def _compute_alignment_matrix(
    agents_data: dict[str, list[int]],
    agent_names: list[str],
) -> tuple[NDArray[np.floating[Any]], dict[tuple[str, str], float]]:
    """Compute pairwise alignment matrix based on D_eff similarity.

    Alignment between two agents is computed as:
        alignment = 1 - |mean_d_eff_i - mean_d_eff_j| / max(mean_d_eff_i, mean_d_eff_j)

    Parameters:
        agents_data: Agent name -> list of D_eff values.
        agent_names: Sorted list of agent names.

    Returns:
        tuple of:
        - alignment_matrix: NxN numpy array of alignments [0, 1]
        - pairwise_alignment: dict mapping (agent_i, agent_j) -> alignment
    """
    n = len(agent_names)
    alignment_matrix = np.zeros((n, n), dtype=np.float64)
    pairwise_alignment: dict[tuple[str, str], float] = {}

    # Compute mean D_eff for each agent
    mean_d_eff = {
        agent: float(np.mean(d_eff_list)) if d_eff_list else 0.0
        for agent, d_eff_list in agents_data.items()
    }

    for i, agent_i in enumerate(agent_names):
        for j, agent_j in enumerate(agent_names):
            if i == j:
                alignment_matrix[i, j] = 1.0
            else:
                d_i = mean_d_eff[agent_i]
                d_j = mean_d_eff[agent_j]

                # Compute alignment based on D_eff similarity
                if max(d_i, d_j) > 0:
                    alignment = 1.0 - abs(d_i - d_j) / max(d_i, d_j)
                else:
                    alignment = 1.0  # Both zero -> identical

                alignment_matrix[i, j] = alignment

                # Store pairwise (only store once per pair)
                if i < j:
                    pairwise_alignment[(agent_i, agent_j)] = alignment

    return alignment_matrix, pairwise_alignment


def _create_alignment_result(
    agent_names: list[str],
    alignment_matrix: NDArray[np.floating[Any]],
    pairwise_alignment: dict[tuple[str, str], float],
) -> AgentAlignmentResult:
    """Create AgentAlignmentResult from computed alignment data.

    Parameters:
        agent_names: Sorted list of agent names.
        alignment_matrix: NxN alignment matrix.
        pairwise_alignment: dict of (agent_i, agent_j) -> alignment.

    Returns:
        AgentAlignmentResult with all statistics computed.
    """
    # Extract off-diagonal alignments for statistics
    if pairwise_alignment:
        alignment_values = list(pairwise_alignment.values())
        mean_alignment = float(np.mean(alignment_values))
        min_alignment = float(min(alignment_values))
        max_alignment = float(max(alignment_values))
        std_alignment = float(np.std(alignment_values))

        # Find min and max pairs
        min_pair = min(pairwise_alignment, key=pairwise_alignment.get)  # type: ignore
        max_pair = max(pairwise_alignment, key=pairwise_alignment.get)  # type: ignore
    else:
        # Single agent case
        mean_alignment = 1.0
        min_alignment = 1.0
        max_alignment = 1.0
        std_alignment = 0.0
        min_pair = None
        max_pair = None

    return AgentAlignmentResult(
        agent_names=agent_names,
        alignment_matrix=alignment_matrix,
        pairwise_alignment=pairwise_alignment,
        mean_alignment=mean_alignment,
        min_alignment=min_alignment,
        max_alignment=max_alignment,
        min_pair=min_pair,
        max_pair=max_pair,
        std_alignment=std_alignment,
    )


def _plot_alignment_heatmap(
    ax: Any,
    alignment_matrix: NDArray[np.floating[Any]],
    agent_names: list[str],
) -> None:
    """Plot alignment matrix as a heatmap.

    Parameters:
        ax: Matplotlib axes to plot on.
        alignment_matrix: NxN alignment matrix.
        agent_names: Agent names for axis labels.
    """
    import matplotlib.pyplot as plt

    im = ax.imshow(alignment_matrix, cmap="RdYlGn", vmin=0, vmax=1)

    # Set axis labels
    ax.set_xticks(range(len(agent_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_yticklabels(agent_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Alignment")

    # Add value annotations
    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            text_color = "white" if alignment_matrix[i, j] < 0.5 else "black"
            ax.text(
                j, i, f"{alignment_matrix[i, j]:.2f}",
                ha="center", va="center", color=text_color, fontsize=8
            )

    ax.set_title("Agent Alignment Matrix")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")


def _plot_deff_trajectory(ax: Any, G: nx.DiGraph) -> None:
    """Plot D_eff trajectory over conversation turns.

    Parameters:
        ax: Matplotlib axes to plot on.
        G: Agent flow graph with 'turn' and 'd_eff' node attributes.
    """
    import matplotlib.pyplot as plt

    # Extract turn and D_eff data
    turn_data: list[tuple[int, int, str]] = []

    for node in G.nodes():
        turn = G.nodes[node].get("turn", 0)
        d_eff = G.nodes[node].get("d_eff", 0)
        agent = G.nodes[node].get("agent", "unknown")
        turn_data.append((turn, d_eff, agent))

    # Sort by turn
    turn_data.sort(key=lambda x: x[0])

    if not turn_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("D_eff Trajectory")
        return

    turns = [t[0] for t in turn_data]
    d_eff_values = [t[1] for t in turn_data]
    agents = [t[2] for t in turn_data]

    # Plot trajectory line
    ax.plot(turns, d_eff_values, "b-o", linewidth=2, markersize=6, label="D_eff")

    # Color points by agent
    unique_agents = sorted(set(agents))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_agents), 1)))  # type: ignore
    agent_colors = {agent: colors[i] for i, agent in enumerate(unique_agents)}

    for i, (turn, d_eff, agent) in enumerate(turn_data):
        ax.scatter([turn], [d_eff], c=[agent_colors[agent]], s=100, zorder=5)

    ax.set_xlabel("Turn")
    ax.set_ylabel("D_eff")
    ax.set_title("D_eff Trajectory Over Turns")
    ax.grid(True, alpha=0.3)

    # Add legend for agents
    for agent, color in agent_colors.items():
        ax.scatter([], [], c=[color], label=agent, s=50)
    ax.legend(loc="best", fontsize=8)


def _plot_network_graph(ax: Any, G: nx.DiGraph) -> None:
    """Plot network graph with D_eff-colored nodes.

    Parameters:
        ax: Matplotlib axes to plot on.
        G: Agent flow graph to visualize.
    """
    import matplotlib.pyplot as plt

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
        ax.set_title("Agent Flow Graph")
        return

    # Use spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=DEFAULT_LAYOUT_SEED, k=DEFAULT_SPRING_LAYOUT_K)

    # Extract D_eff values for node coloring
    d_eff_values = [G.nodes[node].get("d_eff", 0) for node in G.nodes()]

    # Draw the graph
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=d_eff_values,
        cmap=plt.cm.viridis,  # type: ignore
        node_size=500
    )
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    # Add colorbar for D_eff
    if nodes is not None and d_eff_values:
        plt.colorbar(nodes, ax=ax, label="D_eff")

    ax.set_title("Agent Flow Graph")
    ax.axis("off")


def _plot_summary_stats(
    ax: Any,
    alignment_result: AgentAlignmentResult,
    G: nx.DiGraph,
) -> None:
    """Plot summary statistics text.

    Parameters:
        ax: Matplotlib axes to plot on.
        alignment_result: Alignment analysis results.
        G: Agent flow graph for additional stats.
    """
    ax.axis("off")

    # Build summary text
    lines = [
        "Summary Statistics",
        "=" * 25,
        f"Agents: {alignment_result.n_agents}",
        f"Nodes: {G.number_of_nodes()}",
        f"Edges: {G.number_of_edges()}",
        "",
        "Alignment Metrics",
        "-" * 20,
        f"Mean Alignment: {alignment_result.mean_alignment:.4f}",
        f"Min Alignment: {alignment_result.min_alignment:.4f}",
        f"Max Alignment: {alignment_result.max_alignment:.4f}",
        f"Std Alignment: {alignment_result.std_alignment:.4f}",
        "",
        f"Quality: {alignment_result.alignment_quality}",
        f"Well-Aligned: {alignment_result.is_well_aligned}",
    ]

    if alignment_result.min_pair:
        lines.append(f"Min Pair: {alignment_result.min_pair}")
    if alignment_result.max_pair:
        lines.append(f"Max Pair: {alignment_result.max_pair}")

    # Plot text
    text = "\n".join(lines)
    ax.text(
        0.1, 0.9, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace"
    )
