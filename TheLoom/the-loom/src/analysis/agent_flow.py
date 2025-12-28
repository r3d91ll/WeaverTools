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

# Dataclass result types will be implemented in subtask-2-2


# ============================================================================
# Core Analysis Functions
# ============================================================================

# Core functions will be implemented in subtasks 2-3 through 2-6
