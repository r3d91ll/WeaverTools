"""Tests for the cross-agent information flow analysis module.

These tests validate the graph-based analysis functions used to study
information flow between agents in multi-turn conversations.
"""

import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.analysis.agent_flow import (
    DEFAULT_BOTTLENECK_THRESHOLD,
    DEFAULT_FIGURE_DPI,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_LAYOUT_SEED,
    MIN_EDGES_FOR_FLOW,
    MIN_NODES_FOR_BOTTLENECK,
    AgentAlignmentResult,
    AgentBottleneckResult,
    AgentFlowGraphResult,
    FlowComparisonResult,
    build_agent_flow_graph,
    compare_flow_patterns,
    find_agent_bottlenecks,
    visualize_agent_alignment,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_agent_states() -> dict:
    """Create simple agent states for basic tests."""
    np.random.seed(42)
    return {
        "user": {0: np.random.randn(10, 128), 2: np.random.randn(10, 128)},
        "assistant": {1: np.random.randn(10, 128), 3: np.random.randn(10, 128)},
    }


@pytest.fixture
def simple_conveyance_edges() -> dict:
    """Create simple conveyance edges for basic tests."""
    return {
        ("user", "assistant"): 0.8,
        ("assistant", "user"): 0.7,
    }


@pytest.fixture
def multi_agent_states() -> dict:
    """Create multi-agent states for complex tests."""
    np.random.seed(42)
    return {
        "user": {0: np.random.randn(20, 256), 3: np.random.randn(20, 256)},
        "assistant": {1: np.random.randn(20, 256), 4: np.random.randn(20, 256)},
        "tool": {2: np.random.randn(20, 256)},
    }


@pytest.fixture
def sample_flow_graph() -> nx.DiGraph:
    """Create a sample flow graph for testing."""
    G = nx.DiGraph()
    G.add_node("user_t0", agent="user", turn=0, d_eff=100)
    G.add_node("assistant_t1", agent="assistant", turn=1, d_eff=80)
    G.add_node("user_t2", agent="user", turn=2, d_eff=85)
    G.add_node("assistant_t3", agent="assistant", turn=3, d_eff=40)  # Large drop
    G.add_edge("user_t0", "assistant_t1", weight=0.8, conveyance=0.8)
    G.add_edge("assistant_t1", "user_t2", weight=0.7, conveyance=0.7)
    G.add_edge("user_t2", "assistant_t3", weight=0.6, conveyance=0.6)
    return G


@pytest.fixture
def bottleneck_graph() -> nx.DiGraph:
    """Create a graph with clear bottlenecks for testing."""
    G = nx.DiGraph()
    # Node with high D_eff
    G.add_node("a_t0", agent="a", turn=0, d_eff=100)
    # Significant drop (50%) - bottleneck
    G.add_node("b_t1", agent="b", turn=1, d_eff=50)
    # Small drop
    G.add_node("a_t2", agent="a", turn=2, d_eff=45)
    # Another significant drop (60%) - bottleneck
    G.add_node("b_t3", agent="b", turn=3, d_eff=18)

    G.add_edge("a_t0", "b_t1", weight=0.8)
    G.add_edge("b_t1", "a_t2", weight=0.9)
    G.add_edge("a_t2", "b_t3", weight=0.5)
    return G


# ============================================================================
# Tests for build_agent_flow_graph
# ============================================================================


class TestBuildAgentFlowGraph:
    """Tests for the build_agent_flow_graph function."""

    @pytest.mark.gpu
    def test_build_agent_flow_graph(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test basic graph construction with simple inputs."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)

        assert isinstance(result, AgentFlowGraphResult)
        assert isinstance(result.graph, nx.DiGraph)
        assert result.n_nodes == 4
        assert result.n_edges == 3  # 4 nodes, 3 edges between consecutive turns
        assert result.agents == {"user", "assistant"}
        assert result.is_multi_agent

    @pytest.mark.gpu
    def test_build_with_empty_input(self) -> None:
        """Test graph construction with empty input."""
        result = build_agent_flow_graph({}, {})

        assert isinstance(result, AgentFlowGraphResult)
        assert result.n_nodes == 0
        assert result.n_edges == 0
        assert result.agents == set()
        assert "warning" in result.metadata
        assert result.metadata["warning"] == "empty_input"

    @pytest.mark.gpu
    def test_build_single_agent(self) -> None:
        """Test graph construction with single agent."""
        np.random.seed(42)
        agent_states = {
            "solo_agent": {0: np.random.randn(10, 64), 1: np.random.randn(10, 64)},
        }
        result = build_agent_flow_graph(agent_states, {})

        assert result.n_nodes == 2
        assert result.n_edges == 1
        assert result.agents == {"solo_agent"}
        assert not result.is_multi_agent
        assert result.n_agents == 1

    @pytest.mark.gpu
    def test_d_eff_and_beta_computed(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test that D_eff and beta values are computed for each turn."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)

        # Check that d_eff_by_turn has all turns
        assert len(result.d_eff_by_turn) == 4
        assert len(result.beta_by_turn) == 4

        # D_eff values should be positive integers
        for d_eff in result.d_eff_by_turn.values():
            assert isinstance(d_eff, int)
            assert d_eff >= 0

        # Beta values should be floats
        for beta in result.beta_by_turn.values():
            assert isinstance(beta, float)

    @pytest.mark.gpu
    def test_node_attributes(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test that graph nodes have correct attributes."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)
        G = result.graph

        for node in G.nodes():
            attrs = G.nodes[node]
            assert "agent" in attrs
            assert "turn" in attrs
            assert "d_eff" in attrs
            assert "beta" in attrs
            assert attrs["agent"] in {"user", "assistant"}
            assert isinstance(attrs["turn"], int)

    @pytest.mark.gpu
    def test_edge_attributes(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test that graph edges have correct attributes."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)
        G = result.graph

        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert "conveyance" in data
            # Weight and conveyance should be the same
            assert data["weight"] == data["conveyance"]

    @pytest.mark.gpu
    def test_conveyance_metrics(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test that conveyance metrics are correctly aggregated."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)

        assert result.total_conveyance > 0
        assert result.mean_conveyance > 0
        assert result.mean_conveyance == result.total_conveyance / result.n_edges

    @pytest.mark.gpu
    def test_turn_sequence(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test that turn sequence is correctly ordered."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)

        # Turn sequence should be in temporal order
        assert len(result.turn_sequence) == 4
        # User starts (turn 0), then alternates
        assert result.turn_sequence == ["user", "assistant", "user", "assistant"]

    @pytest.mark.gpu
    def test_d_eff_trajectory_property(
        self, simple_agent_states: dict, simple_conveyance_edges: dict
    ) -> None:
        """Test the d_eff_trajectory property."""
        result = build_agent_flow_graph(simple_agent_states, simple_conveyance_edges)

        trajectory = result.d_eff_trajectory
        assert len(trajectory) == 4
        assert all(isinstance(d, int) for d in trajectory)

    @pytest.mark.gpu
    def test_1d_hidden_states(self) -> None:
        """Test handling of 1D hidden state arrays."""
        np.random.seed(42)
        # 1D arrays should be reshaped to 2D
        agent_states = {
            "agent": {0: np.random.randn(128), 1: np.random.randn(128)},
        }
        result = build_agent_flow_graph(agent_states, {})

        assert result.n_nodes == 2
        assert len(result.d_eff_by_turn) == 2

    @pytest.mark.gpu
    def test_missing_conveyance_defaults(self) -> None:
        """Test that missing conveyance values default to 1.0."""
        np.random.seed(42)
        agent_states = {
            "user": {0: np.random.randn(10, 64)},
            "assistant": {1: np.random.randn(10, 64)},
        }
        # No conveyance edges provided
        result = build_agent_flow_graph(agent_states, {})

        G = result.graph
        for u, v, data in G.edges(data=True):
            assert data["weight"] == 1.0
            assert data["conveyance"] == 1.0


# ============================================================================
# Tests for find_agent_bottlenecks
# ============================================================================


class TestFindAgentBottlenecks:
    """Tests for the find_agent_bottlenecks function."""

    def test_find_bottlenecks_basic(self, bottleneck_graph: nx.DiGraph) -> None:
        """Test basic bottleneck detection."""
        result = find_agent_bottlenecks(bottleneck_graph)

        assert isinstance(result, AgentBottleneckResult)
        assert result.has_bottlenecks
        assert result.n_bottlenecks > 0
        assert len(result.bottleneck_edges) == result.n_bottlenecks

    def test_find_bottlenecks_empty_graph(self) -> None:
        """Test bottleneck detection on empty graph."""
        G = nx.DiGraph()
        result = find_agent_bottlenecks(G)

        assert isinstance(result, AgentBottleneckResult)
        assert not result.has_bottlenecks
        assert result.n_bottlenecks == 0
        assert "warning" in result.metadata
        assert result.metadata["warning"] == "insufficient_data"

    def test_find_bottlenecks_insufficient_nodes(self) -> None:
        """Test bottleneck detection with too few nodes."""
        G = nx.DiGraph()
        G.add_node("a", d_eff=100)
        G.add_node("b", d_eff=50)
        G.add_edge("a", "b")

        result = find_agent_bottlenecks(G)

        assert not result.has_bottlenecks
        assert "warning" in result.metadata

    def test_find_bottlenecks_no_drops(self) -> None:
        """Test bottleneck detection when D_eff only increases."""
        G = nx.DiGraph()
        G.add_node("a_t0", d_eff=50)
        G.add_node("b_t1", d_eff=60)
        G.add_node("a_t2", d_eff=70)
        G.add_node("b_t3", d_eff=80)
        G.add_edge("a_t0", "b_t1")
        G.add_edge("b_t1", "a_t2")
        G.add_edge("a_t2", "b_t3")

        result = find_agent_bottlenecks(G)

        assert not result.has_bottlenecks
        assert result.n_bottlenecks == 0
        # Should have "no_positive_drops" info
        assert result.metadata.get("info") == "no_positive_drops"

    def test_bottleneck_severity_classification(
        self, bottleneck_graph: nx.DiGraph
    ) -> None:
        """Test bottleneck severity classification."""
        result = find_agent_bottlenecks(bottleneck_graph)

        # Severity should be one of the valid values
        assert result.severity in ["none", "mild", "moderate", "severe"]

    def test_centrality_scores_computed(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test that betweenness centrality is computed."""
        result = find_agent_bottlenecks(sample_flow_graph)

        assert len(result.centrality_scores) == sample_flow_graph.number_of_nodes()
        for score in result.centrality_scores.values():
            assert 0 <= score <= 1

    def test_max_drop_edge_identified(self, bottleneck_graph: nx.DiGraph) -> None:
        """Test that maximum drop edge is correctly identified."""
        result = find_agent_bottlenecks(bottleneck_graph)

        if result.has_bottlenecks:
            assert result.max_drop_edge is not None
            assert result.max_drop_value > 0
            # Max drop should be the largest in d_eff_drops
            if result.d_eff_drops:
                max_from_dict = max(result.d_eff_drops.values())
                assert result.max_drop_value == max_from_dict

    def test_relative_drops_computed(self, bottleneck_graph: nx.DiGraph) -> None:
        """Test that relative drops are computed correctly."""
        result = find_agent_bottlenecks(bottleneck_graph)

        for edge, relative_drop in result.relative_drops.items():
            assert 0 <= relative_drop <= 1

    def test_custom_threshold(self, bottleneck_graph: nx.DiGraph) -> None:
        """Test bottleneck detection with custom threshold."""
        # Lower threshold should find more bottlenecks
        result_low = find_agent_bottlenecks(bottleneck_graph, threshold=0.5)
        result_high = find_agent_bottlenecks(bottleneck_graph, threshold=3.0)

        assert result_low.threshold_used == 0.5
        assert result_high.threshold_used == 3.0
        # Lower threshold should find >= as many bottlenecks
        assert result_low.n_bottlenecks >= result_high.n_bottlenecks


class TestAgentBottleneckResultProperties:
    """Tests for AgentBottleneckResult computed properties."""

    def test_severity_none(self) -> None:
        """Test severity 'none' when no bottlenecks."""
        result = AgentBottleneckResult(
            bottleneck_edges=[],
            d_eff_drops={},
            relative_drops={},
            centrality_scores={},
            threshold_used=1.5,
            n_bottlenecks=0,
            max_drop_edge=None,
            max_drop_value=0,
            mean_d_eff_drop=0.0,
        )
        assert result.severity == "none"

    def test_severity_mild(self) -> None:
        """Test severity 'mild' with small bottleneck."""
        result = AgentBottleneckResult(
            bottleneck_edges=[("a", "b")],
            d_eff_drops={("a", "b"): 20},
            relative_drops={("a", "b"): 0.2},  # 20% drop
            centrality_scores={},
            threshold_used=1.5,
            n_bottlenecks=1,
            max_drop_edge=("a", "b"),
            max_drop_value=20,
            mean_d_eff_drop=20.0,
        )
        assert result.severity == "mild"

    def test_severity_moderate(self) -> None:
        """Test severity 'moderate' with multiple bottlenecks."""
        result = AgentBottleneckResult(
            bottleneck_edges=[("a", "b"), ("c", "d")],
            d_eff_drops={("a", "b"): 20, ("c", "d"): 25},
            relative_drops={("a", "b"): 0.2, ("c", "d"): 0.25},
            centrality_scores={},
            threshold_used=1.5,
            n_bottlenecks=2,
            max_drop_edge=("c", "d"),
            max_drop_value=25,
            mean_d_eff_drop=22.5,
        )
        assert result.severity == "moderate"

    def test_severity_severe(self) -> None:
        """Test severity 'severe' with large bottleneck."""
        result = AgentBottleneckResult(
            bottleneck_edges=[("a", "b")],
            d_eff_drops={("a", "b"): 60},
            relative_drops={("a", "b"): 0.6},  # 60% drop
            centrality_scores={},
            threshold_used=1.5,
            n_bottlenecks=1,
            max_drop_edge=("a", "b"),
            max_drop_value=60,
            mean_d_eff_drop=60.0,
        )
        assert result.severity == "severe"

    def test_bottleneck_agents_extraction(self) -> None:
        """Test extraction of agents involved in bottlenecks."""
        result = AgentBottleneckResult(
            bottleneck_edges=[("user_t0", "assistant_t1"), ("assistant_t1", "tool_t2")],
            d_eff_drops={},
            relative_drops={},
            centrality_scores={},
            threshold_used=1.5,
            n_bottlenecks=2,
            max_drop_edge=None,
            max_drop_value=0,
            mean_d_eff_drop=0.0,
        )
        agents = result.bottleneck_agents
        assert "user" in agents
        assert "assistant" in agents
        assert "tool" in agents


# ============================================================================
# Tests for compare_flow_patterns
# ============================================================================


class TestCompareFlowPatterns:
    """Tests for the compare_flow_patterns function."""

    def test_compare_identical_graphs(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test comparison of identical graphs."""
        result = compare_flow_patterns(sample_flow_graph, sample_flow_graph)

        assert isinstance(result, FlowComparisonResult)
        assert result.trajectory_correlation == pytest.approx(1.0)
        assert result.structural_similarity == pytest.approx(1.0)
        assert result.agent_overlap == pytest.approx(1.0)

    def test_compare_empty_graphs(self) -> None:
        """Test comparison of empty graphs."""
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()
        result = compare_flow_patterns(G1, G2)

        assert isinstance(result, FlowComparisonResult)
        assert result.structural_similarity == pytest.approx(1.0)
        assert result.n_turns_a == 0
        assert result.n_turns_b == 0

    def test_compare_different_graphs(self) -> None:
        """Test comparison of different graphs."""
        G1 = nx.DiGraph()
        G1.add_node("user_t0", agent="user", turn=0, d_eff=100)
        G1.add_node("assistant_t1", agent="assistant", turn=1, d_eff=90)
        G1.add_edge("user_t0", "assistant_t1", weight=0.8)

        G2 = nx.DiGraph()
        G2.add_node("user_t0", agent="user", turn=0, d_eff=50)
        G2.add_node("assistant_t1", agent="assistant", turn=1, d_eff=40)
        G2.add_edge("user_t0", "assistant_t1", weight=0.5)

        result = compare_flow_patterns(G1, G2)

        # Graphs have same structure but different values
        assert result.structural_similarity == pytest.approx(1.0)
        # D_eff values differ
        assert result.d_eff_mean_diff != 0

    def test_compare_different_agents(self) -> None:
        """Test comparison of graphs with different agents."""
        G1 = nx.DiGraph()
        G1.add_node("user_t0", agent="user", turn=0, d_eff=100)
        G1.add_node("assistant_t1", agent="assistant", turn=1, d_eff=90)
        G1.add_edge("user_t0", "assistant_t1")

        G2 = nx.DiGraph()
        G2.add_node("admin_t0", agent="admin", turn=0, d_eff=100)
        G2.add_node("tool_t1", agent="tool", turn=1, d_eff=90)
        G2.add_edge("admin_t0", "tool_t1")

        result = compare_flow_patterns(G1, G2)

        assert result.agent_overlap == 0.0
        assert result.common_agents == set()
        assert "user" in result.agents_only_in_a
        assert "admin" in result.agents_only_in_b

    def test_conveyance_ratio(self) -> None:
        """Test conveyance ratio computation."""
        G1 = nx.DiGraph()
        G1.add_node("a_t0", agent="a", turn=0, d_eff=100)
        G1.add_node("b_t1", agent="b", turn=1, d_eff=90)
        G1.add_edge("a_t0", "b_t1", weight=2.0)

        G2 = nx.DiGraph()
        G2.add_node("a_t0", agent="a", turn=0, d_eff=100)
        G2.add_node("b_t1", agent="b", turn=1, d_eff=90)
        G2.add_edge("a_t0", "b_t1", weight=1.0)

        result = compare_flow_patterns(G1, G2)

        assert result.conveyance_ratio == pytest.approx(2.0)

    def test_graph_stats_populated(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test that graph statistics are populated."""
        G2 = nx.DiGraph()
        G2.add_node("x_t0", agent="x", turn=0, d_eff=50)
        result = compare_flow_patterns(sample_flow_graph, G2)

        assert "n_nodes" in result.graph_a_stats
        assert "n_edges" in result.graph_a_stats
        assert "n_agents" in result.graph_a_stats
        assert "mean_d_eff" in result.graph_a_stats


class TestFlowComparisonResultProperties:
    """Tests for FlowComparisonResult computed properties."""

    def test_divergence_quality_identical(self) -> None:
        """Test divergence quality 'identical'."""
        result = FlowComparisonResult(
            graph_a_stats={},
            graph_b_stats={},
            common_agents={"user", "assistant"},
            agents_only_in_a=set(),
            agents_only_in_b=set(),
            trajectory_correlation=1.0,
            d_eff_mean_diff=0.0,
            d_eff_abs_mean_diff=0.0,
            conveyance_ratio=1.0,
            structural_similarity=1.0,
            n_turns_a=5,
            n_turns_b=5,
        )
        assert result.divergence_quality == "identical"

    def test_divergence_quality_similar(self) -> None:
        """Test divergence quality 'similar'."""
        result = FlowComparisonResult(
            graph_a_stats={},
            graph_b_stats={},
            common_agents={"user", "assistant"},
            agents_only_in_a=set(),
            agents_only_in_b=set(),
            trajectory_correlation=0.85,
            d_eff_mean_diff=5.0,
            d_eff_abs_mean_diff=5.0,
            conveyance_ratio=1.1,
            structural_similarity=0.9,
            n_turns_a=5,
            n_turns_b=5,
        )
        assert result.divergence_quality == "similar"
        assert result.is_similar

    def test_divergence_quality_divergent(self) -> None:
        """Test divergence quality 'divergent'."""
        result = FlowComparisonResult(
            graph_a_stats={},
            graph_b_stats={},
            common_agents=set(),
            agents_only_in_a={"user"},
            agents_only_in_b={"admin"},
            trajectory_correlation=0.2,
            d_eff_mean_diff=50.0,
            d_eff_abs_mean_diff=50.0,
            conveyance_ratio=0.5,
            structural_similarity=0.3,
            n_turns_a=3,
            n_turns_b=10,
        )
        assert result.divergence_quality == "divergent"
        assert not result.is_similar

    def test_length_ratio(self) -> None:
        """Test length ratio computation."""
        result = FlowComparisonResult(
            graph_a_stats={},
            graph_b_stats={},
            common_agents=set(),
            agents_only_in_a=set(),
            agents_only_in_b=set(),
            trajectory_correlation=0.0,
            d_eff_mean_diff=0.0,
            d_eff_abs_mean_diff=0.0,
            conveyance_ratio=1.0,
            structural_similarity=1.0,
            n_turns_a=10,
            n_turns_b=5,
        )
        assert result.length_ratio == pytest.approx(2.0)


# ============================================================================
# Tests for visualize_agent_alignment
# ============================================================================


class TestVisualizeAgentAlignment:
    """Tests for the visualize_agent_alignment function."""

    def test_visualize_basic(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test basic visualization generation."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        result = visualize_agent_alignment(sample_flow_graph, output_path)

        assert result is not None
        assert isinstance(result, AgentAlignmentResult)
        assert Path(output_path).exists()
        # Cleanup
        Path(output_path).unlink()

    def test_visualize_empty_graph(self) -> None:
        """Test visualization of empty graph."""
        G = nx.DiGraph()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        result = visualize_agent_alignment(G, output_path)

        assert result is None
        assert Path(output_path).exists()  # Placeholder image created
        # Cleanup
        Path(output_path).unlink()

    def test_visualize_single_node(self) -> None:
        """Test visualization of single-node graph."""
        G = nx.DiGraph()
        G.add_node("agent_t0", agent="agent", turn=0, d_eff=100)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        result = visualize_agent_alignment(G, output_path)

        assert result is not None
        assert result.n_agents == 1
        assert result.mean_alignment == 1.0
        assert Path(output_path).exists()
        # Cleanup
        Path(output_path).unlink()

    def test_alignment_matrix_computed(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test that alignment matrix is computed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        result = visualize_agent_alignment(sample_flow_graph, output_path)

        assert result is not None
        # Should have 2 agents: user and assistant
        assert result.n_agents == 2
        assert result.alignment_matrix.shape == (2, 2)
        # Diagonal should be 1.0
        assert result.alignment_matrix[0, 0] == pytest.approx(1.0)
        assert result.alignment_matrix[1, 1] == pytest.approx(1.0)
        # Cleanup
        Path(output_path).unlink()

    def test_custom_figsize_and_dpi(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test visualization with custom figure size and DPI."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        result = visualize_agent_alignment(
            sample_flow_graph, output_path, figsize=(8, 6), dpi=150
        )

        assert result is not None
        assert Path(output_path).exists()
        # Cleanup
        Path(output_path).unlink()

    def test_pdf_output(self, sample_flow_graph: nx.DiGraph) -> None:
        """Test visualization output to PDF format."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        result = visualize_agent_alignment(sample_flow_graph, output_path)

        assert result is not None
        assert Path(output_path).exists()
        # Cleanup
        Path(output_path).unlink()


class TestAgentAlignmentResultProperties:
    """Tests for AgentAlignmentResult computed properties."""

    def test_is_well_aligned(self) -> None:
        """Test is_well_aligned property."""
        result = AgentAlignmentResult(
            agent_names=["a", "b"],
            alignment_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
            pairwise_alignment={("a", "b"): 0.8},
            mean_alignment=0.8,
            min_alignment=0.8,
            max_alignment=0.8,
            min_pair=("a", "b"),
            max_pair=("a", "b"),
            std_alignment=0.0,
        )
        assert result.is_well_aligned

    def test_is_not_well_aligned(self) -> None:
        """Test when agents are not well-aligned."""
        result = AgentAlignmentResult(
            agent_names=["a", "b"],
            alignment_matrix=np.array([[1.0, 0.2], [0.2, 1.0]]),
            pairwise_alignment={("a", "b"): 0.2},
            mean_alignment=0.2,
            min_alignment=0.2,
            max_alignment=0.2,
            min_pair=("a", "b"),
            max_pair=("a", "b"),
            std_alignment=0.0,
        )
        assert not result.is_well_aligned

    def test_alignment_quality_excellent(self) -> None:
        """Test alignment quality 'excellent'."""
        result = AgentAlignmentResult(
            agent_names=["a", "b"],
            alignment_matrix=np.array([[1.0, 0.95], [0.95, 1.0]]),
            pairwise_alignment={("a", "b"): 0.95},
            mean_alignment=0.95,
            min_alignment=0.95,
            max_alignment=0.95,
            min_pair=("a", "b"),
            max_pair=("a", "b"),
            std_alignment=0.0,
        )
        assert result.alignment_quality == "excellent"

    def test_alignment_quality_poor(self) -> None:
        """Test alignment quality 'poor'."""
        result = AgentAlignmentResult(
            agent_names=["a", "b"],
            alignment_matrix=np.array([[1.0, 0.1], [0.1, 1.0]]),
            pairwise_alignment={("a", "b"): 0.1},
            mean_alignment=0.1,
            min_alignment=0.1,
            max_alignment=0.1,
            min_pair=("a", "b"),
            max_pair=("a", "b"),
            std_alignment=0.0,
        )
        assert result.alignment_quality == "poor"

    def test_get_alignment(self) -> None:
        """Test get_alignment method."""
        result = AgentAlignmentResult(
            agent_names=["a", "b"],
            alignment_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
            pairwise_alignment={("a", "b"): 0.75},
            mean_alignment=0.75,
            min_alignment=0.75,
            max_alignment=0.75,
            min_pair=("a", "b"),
            max_pair=("a", "b"),
            std_alignment=0.0,
        )

        assert result.get_alignment("a", "b") == pytest.approx(0.75)
        assert result.get_alignment("b", "a") == pytest.approx(0.75)
        assert result.get_alignment("a", "a") == pytest.approx(1.0)

    def test_get_alignment_key_error(self) -> None:
        """Test get_alignment raises KeyError for unknown agents."""
        result = AgentAlignmentResult(
            agent_names=["a", "b"],
            alignment_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
            pairwise_alignment={("a", "b"): 0.75},
            mean_alignment=0.75,
            min_alignment=0.75,
            max_alignment=0.75,
            min_pair=("a", "b"),
            max_pair=("a", "b"),
            std_alignment=0.0,
        )

        with pytest.raises(KeyError):
            result.get_alignment("a", "unknown")

    def test_has_outlier(self) -> None:
        """Test has_outlier property."""
        # Result with an outlier (one value much lower than mean - 2*std)
        result = AgentAlignmentResult(
            agent_names=["a", "b", "c"],
            alignment_matrix=np.array([
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.1],
                [0.1, 0.1, 1.0],
            ]),
            pairwise_alignment={("a", "b"): 0.9, ("a", "c"): 0.1, ("b", "c"): 0.1},
            mean_alignment=0.37,
            min_alignment=0.1,
            max_alignment=0.9,
            min_pair=("a", "c"),
            max_pair=("a", "b"),
            std_alignment=0.4,  # Large std
        )
        # min_alignment (0.1) < mean (0.37) - 2*std (0.8) = -0.43
        # So 0.1 > -0.43, no outlier by this definition
        # Let's test with different values

        result2 = AgentAlignmentResult(
            agent_names=["a", "b", "c"],
            alignment_matrix=np.array([
                [1.0, 0.95, 0.2],
                [0.95, 1.0, 0.2],
                [0.2, 0.2, 1.0],
            ]),
            pairwise_alignment={("a", "b"): 0.95, ("a", "c"): 0.2, ("b", "c"): 0.2},
            mean_alignment=0.45,
            min_alignment=0.2,
            max_alignment=0.95,
            min_pair=("a", "c"),
            max_pair=("a", "b"),
            std_alignment=0.1,  # Small std, so min is outlier
        )
        # min (0.2) < mean (0.45) - 2*std (0.2) = 0.25
        assert result2.has_outlier


# ============================================================================
# Tests for AgentFlowGraphResult Properties
# ============================================================================


class TestAgentFlowGraphResultProperties:
    """Tests for AgentFlowGraphResult computed properties."""

    def test_is_connected(self) -> None:
        """Test is_connected property."""
        G = nx.DiGraph()
        G.add_node("a_t0")
        G.add_node("b_t1")
        G.add_edge("a_t0", "b_t1")

        result = AgentFlowGraphResult(
            graph=G,
            n_nodes=2,
            n_edges=1,
            agents={"a", "b"},
            turn_sequence=["a", "b"],
            d_eff_by_turn={0: 100, 1: 90},
            beta_by_turn={0: 0.5, 1: 0.6},
            total_conveyance=1.0,
            mean_conveyance=1.0,
        )
        assert result.is_connected

    def test_is_not_connected(self) -> None:
        """Test is_connected when graph is disconnected."""
        G = nx.DiGraph()
        G.add_node("a_t0")
        G.add_node("b_t1")
        # No edge - disconnected

        result = AgentFlowGraphResult(
            graph=G,
            n_nodes=2,
            n_edges=0,
            agents={"a", "b"},
            turn_sequence=["a", "b"],
            d_eff_by_turn={0: 100, 1: 90},
            beta_by_turn={0: 0.5, 1: 0.6},
            total_conveyance=0.0,
            mean_conveyance=0.0,
        )
        assert not result.is_connected

    def test_has_sufficient_data(self) -> None:
        """Test has_sufficient_data property."""
        G = nx.DiGraph()
        G.add_node("a_t0")
        G.add_node("b_t1")
        G.add_node("c_t2")
        G.add_edge("a_t0", "b_t1")
        G.add_edge("b_t1", "c_t2")

        result = AgentFlowGraphResult(
            graph=G,
            n_nodes=3,
            n_edges=2,
            agents={"a", "b", "c"},
            turn_sequence=["a", "b", "c"],
            d_eff_by_turn={0: 100, 1: 90, 2: 80},
            beta_by_turn={0: 0.5, 1: 0.6, 2: 0.7},
            total_conveyance=2.0,
            mean_conveyance=1.0,
        )
        assert result.has_sufficient_data

    def test_conversation_length(self) -> None:
        """Test conversation_length property."""
        G = nx.DiGraph()
        result = AgentFlowGraphResult(
            graph=G,
            n_nodes=5,
            n_edges=4,
            agents={"a", "b"},
            turn_sequence=["a", "b", "a", "b", "a"],
            d_eff_by_turn={},
            beta_by_turn={},
            total_conveyance=4.0,
            mean_conveyance=1.0,
        )
        assert result.conversation_length == 5


# ============================================================================
# Tests for Constants
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_layout_seed(self) -> None:
        """Test that default layout seed is 42."""
        assert DEFAULT_LAYOUT_SEED == 42

    def test_default_figure_dpi(self) -> None:
        """Test that default DPI is publication-quality."""
        assert DEFAULT_FIGURE_DPI == 300

    def test_default_figure_size(self) -> None:
        """Test default figure size."""
        assert DEFAULT_FIGURE_SIZE == (12, 10)

    def test_bottleneck_threshold(self) -> None:
        """Test default bottleneck threshold."""
        assert DEFAULT_BOTTLENECK_THRESHOLD == 1.5

    def test_min_nodes_for_bottleneck(self) -> None:
        """Test minimum nodes constant."""
        assert MIN_NODES_FOR_BOTTLENECK == 3

    def test_min_edges_for_flow(self) -> None:
        """Test minimum edges constant."""
        assert MIN_EDGES_FOR_FLOW == 1
