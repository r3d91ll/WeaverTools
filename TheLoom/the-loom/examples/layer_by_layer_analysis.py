"""Layer-by-Layer Hidden State Analysis Tutorial.

This script demonstrates how to use The Loom for layer-by-layer hidden state
extraction and analysis. It covers:

1. Extracting hidden states from specific layers (e.g., [-1, -5, -11])
2. Computing D_eff (effective dimensionality) per layer
3. Visualizing D_eff evolution across transformer layers

This is the core capability that enables interpretability research:
understanding how semantic information evolves through model layers.

Requirements:
    - The Loom server running (docker or local)
    - numpy, matplotlib (for visualization)
    - httpx (for HTTP client)

Usage:
    # Start The Loom server first
    docker run -d --gpus all -p 8080:8080 tbucy/loom:latest

    # Then run this script
    python layer_by_layer_analysis.py

References:
    - D_eff (Effective Dimensionality): Whiteley et al. "Statistical exploration
      of the Manifold Hypothesis" (arXiv:2208.11665)
    - Conveyance Hypothesis: Measures semantic information transfer via D_eff
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports (when running as script)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import LoomClient
from src.analysis.conveyance_metrics import calculate_d_eff, calculate_d_eff_detailed


# =============================================================================
# Configuration
# =============================================================================

# The Loom server URL
LOOM_URL = "http://localhost:8080"

# Model to analyze (use a small model for quick testing)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Prompt for analysis
PROMPT = "The fundamental nature of reality is"

# Layers to extract for targeted analysis
# Negative indices: -1 = last layer, -5 = 5th from last, -11 = 11th from last
TARGETED_LAYERS = [-1, -5, -11]


# =============================================================================
# Helper Functions
# =============================================================================


def compute_d_eff_for_hidden_state(
    hidden_state_data: list[float],
    shape: list[int],
    variance_threshold: float = 0.90,
) -> int:
    """Compute D_eff for a single hidden state vector.

    For a single hidden state vector, D_eff is computed by treating the
    vector dimensions as samples of a 1-dimensional distribution.

    For meaningful D_eff computation, you typically need multiple hidden
    states (e.g., from a batch or sequence). This function provides a
    basic estimate for single vectors.

    Parameters:
        hidden_state_data: Flattened hidden state values
        shape: Original shape of the hidden state
        variance_threshold: Cumulative variance threshold (default 0.90)

    Returns:
        Estimated effective dimensionality
    """
    # Convert to numpy array and reshape
    vector = np.array(hidden_state_data).reshape(shape)

    # Squeeze batch dimension if present
    if vector.ndim > 1 and vector.shape[0] == 1:
        vector = vector.squeeze(0)

    # For single vector: treat dimensions as samples
    # This gives an estimate of how many dimensions are "active"
    if vector.ndim == 1:
        # Compute variance explained by each dimension
        # (treated as a 1D signal analysis)
        abs_values = np.abs(vector)
        sorted_vals = np.sort(abs_values)[::-1]

        # Compute cumulative contribution
        total = sorted_vals.sum()
        if total == 0:
            return 1

        cumsum = np.cumsum(sorted_vals) / total
        d_eff = int(np.searchsorted(cumsum, variance_threshold) + 1)
        return min(d_eff, len(vector))

    # For multiple samples: use proper PCA-based D_eff
    return calculate_d_eff(vector, variance_threshold)


def analyze_layer_hidden_states(
    client: LoomClient,
    model: str,
    prompt: str,
    layers: list[int] | str = "all",
) -> dict[int, dict[str, Any]]:
    """Extract and analyze hidden states from specified layers.

    Parameters:
        client: LoomClient instance
        model: Model ID to use
        prompt: Input prompt
        layers: List of layer indices or "all" for all layers

    Returns:
        Dictionary mapping layer index to analysis results
    """
    # Request hidden states for specified layers
    # API accepts "all" string directly, no need for double-request
    result = client.generate(
        model=model,
        prompt=prompt,
        max_tokens=20,  # Short generation for analysis
        return_hidden_states=True,
        hidden_state_layers=layers,  # API accepts "all" string or list[int]
    )

    hidden_states = result.get("hidden_states", {})
    analysis_results: dict[int, dict[str, Any]] = {}

    for layer_key, layer_data in hidden_states.items():
        layer_idx = int(layer_key)
        data = layer_data.get("data", [])
        shape = layer_data.get("shape", [])
        dtype = layer_data.get("dtype", "unknown")

        # Compute D_eff for this layer
        d_eff = compute_d_eff_for_hidden_state(data, shape)

        # Compute basic statistics
        vector = np.array(data)
        analysis_results[layer_idx] = {
            "layer": layer_idx,
            "shape": shape,
            "dtype": dtype,
            "d_eff": d_eff,
            "hidden_size": shape[-1] if shape else len(data),
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "l2_norm": float(np.linalg.norm(vector)),
            "sparsity": float(np.mean(np.abs(vector) < 1e-6)),
        }

    return analysis_results


def plot_d_eff_by_layer(
    layer_results: dict[int, dict[str, Any]],
    title: str = "D_eff by Layer",
    save_path: str | None = None,
) -> None:
    """Plot D_eff values across layers.

    Parameters:
        layer_results: Analysis results by layer
        title: Plot title
        save_path: If provided, save plot to this path
    """
    # Sort layers by index (negative to positive)
    sorted_layers = sorted(layer_results.keys())

    layers = [layer_results[idx]["layer"] for idx in sorted_layers]
    d_effs = [layer_results[idx]["d_eff"] for idx in sorted_layers]
    hidden_sizes = [layer_results[idx]["hidden_size"] for idx in sorted_layers]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute D_eff values
    ax1.plot(layers, d_effs, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("D_eff (Effective Dimensionality)", fontsize=12)
    ax1.set_title(f"{title} - Absolute Values", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add reference line for hidden size
    if hidden_sizes:
        avg_hidden = np.mean(hidden_sizes)
        ax1.axhline(y=avg_hidden, color="r", linestyle="--", alpha=0.5,
                    label=f"Hidden Size ({avg_hidden:.0f})")
        ax1.legend()

    # Plot 2: D_eff as fraction of hidden size (utilization ratio)
    utilization = [d_effs[i] / hidden_sizes[i] if hidden_sizes[i] > 0 else 0
                   for i in range(len(d_effs))]

    ax2.bar(layers, utilization, color="green", alpha=0.7)
    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("D_eff / Hidden Size (Utilization)", fontsize=12)
    ax2.set_title(f"{title} - Space Utilization", fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def print_layer_analysis(layer_results: dict[int, dict[str, Any]]) -> None:
    """Print a formatted summary of layer analysis results."""
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER HIDDEN STATE ANALYSIS")
    print("=" * 70)

    # Sort layers
    sorted_layers = sorted(layer_results.keys())

    print(f"\n{'Layer':>8} {'D_eff':>10} {'Hidden':>10} {'Ratio':>10} {'L2 Norm':>12}")
    print("-" * 70)

    for layer_idx in sorted_layers:
        result = layer_results[layer_idx]
        ratio = result["d_eff"] / result["hidden_size"] if result["hidden_size"] > 0 else 0
        print(
            f"{result['layer']:>8} "
            f"{result['d_eff']:>10} "
            f"{result['hidden_size']:>10} "
            f"{ratio:>10.3f} "
            f"{result['l2_norm']:>12.3f}"
        )

    # Summary statistics
    d_effs = [layer_results[idx]["d_eff"] for idx in sorted_layers]
    print("-" * 70)
    print("\nSummary Statistics:")
    print(f"  Mean D_eff:   {np.mean(d_effs):.2f}")
    print(f"  Std D_eff:    {np.std(d_effs):.2f}")
    print(f"  Min D_eff:    {np.min(d_effs)} (Layer {sorted_layers[np.argmin(d_effs)]})")
    print(f"  Max D_eff:    {np.max(d_effs)} (Layer {sorted_layers[np.argmax(d_effs)]})")


# =============================================================================
# Main Analysis
# =============================================================================


def main() -> None:
    """Run the layer-by-layer analysis demonstration."""
    print("Layer-by-Layer Hidden State Analysis")
    print("=" * 50)
    print(f"\nModel: {MODEL_ID}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Target Layers: {TARGETED_LAYERS}")

    # Connect to The Loom server
    print(f"\nConnecting to The Loom at {LOOM_URL}...")
    client = LoomClient(base_url=LOOM_URL)

    try:
        # Check server health
        health = client.health()
        print(f"Server healthy: {health.get('status', 'unknown')}")

        # =====================================================================
        # Part 1: Targeted Layer Analysis (specific layers)
        # =====================================================================
        print("\n" + "=" * 50)
        print("Part 1: Targeted Layer Analysis")
        print(f"Extracting layers: {TARGETED_LAYERS}")
        print("=" * 50)

        targeted_results = analyze_layer_hidden_states(
            client=client,
            model=MODEL_ID,
            prompt=PROMPT,
            layers=TARGETED_LAYERS,
        )

        print_layer_analysis(targeted_results)

        # =====================================================================
        # Part 2: Full Layer Analysis (all layers)
        # =====================================================================
        print("\n" + "=" * 50)
        print("Part 2: Full Layer Analysis (all layers)")
        print("=" * 50)

        # Request all layers by not specifying hidden_state_layers
        full_result = client.generate(
            model=MODEL_ID,
            prompt=PROMPT,
            max_tokens=20,
            return_hidden_states=True,
            # When hidden_state_layers is not specified, server returns all
        )

        all_hidden_states = full_result.get("hidden_states", {})

        if all_hidden_states:
            full_results = analyze_layer_hidden_states(
                client=client,
                model=MODEL_ID,
                prompt=PROMPT,
                layers=[int(k) for k in all_hidden_states.keys()],
            )

            print_layer_analysis(full_results)

            # Plot D_eff across all layers
            print("\nGenerating D_eff visualization...")
            plot_d_eff_by_layer(
                full_results,
                title=f"D_eff Analysis: {MODEL_ID}",
                save_path="layer_deff_analysis.png",
            )
        else:
            print("No hidden states returned. Ensure model is loaded and accessible.")

        # =====================================================================
        # Part 3: Interpretation Guide
        # =====================================================================
        print("\n" + "=" * 50)
        print("INTERPRETATION GUIDE")
        print("=" * 50)
        print("""
D_eff (Effective Dimensionality) measures how many dimensions are
actively used by the hidden state representations.

Key Interpretations:
  - High D_eff: Rich, diverse representations (many directions used)
  - Low D_eff: Compressed representations (information concentrated)

Typical Patterns:
  - Early layers: Often higher D_eff (processing raw input features)
  - Middle layers: May show compression (abstraction forming)
  - Final layers: Often task-specific (semantic compression)

D_eff / Hidden_Size Ratio:
  - Close to 1.0: Full utilization of available dimensions
  - Close to 0: Severe compression or collapse
  - 0.3-0.7: Healthy range for most transformer layers

For multi-agent comparison, differences in D_eff patterns can reveal:
  - Bottleneck layers where information is lost
  - Layers where models diverge in representation strategy
  - Optimal layers for feature extraction
        """)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure The Loom server is running:")
        print("  docker run -d --gpus all -p 8080:8080 tbucy/loom:latest")
        raise

    finally:
        client.close()

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
