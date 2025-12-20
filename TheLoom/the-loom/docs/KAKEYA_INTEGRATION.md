# Kakeya Geometry Integration with TheLoom

## Overview

This document describes how to integrate the `kakeya_geometry.py` module with TheLoom to test whether transformer hidden states exhibit Kakeya-like geometric properties that might constrain or enable information transfer (conveyance).

## Installation

The `kakeya_geometry.py` module is located in the analysis directory:

```
TheLoom/
└── the-loom/
    └── src/
        ├── extraction/
        │   └── hidden_states.py  # Existing
        ├── analysis/             # Analysis modules
        │   ├── __init__.py
        │   └── kakeya_geometry.py
        └── ...
```

Dependencies are included in pyproject.toml:
- scipy>=1.10.0
- scikit-learn>=1.2.0

## Basic Integration

### Analyzing Hidden States from TheLoom

```python
from the_loom.src.extraction.hidden_states import HiddenStateResult
from the_loom.src.analysis.kakeya_geometry import (
    analyze_kakeya_geometry,
    analyze_hidden_state_batch,
    KakeyaGeometryReport,
)

# After extracting hidden states from model
hidden_states: list[HiddenStateResult] = extraction_results

# Option 1: Use the batch analyzer (handles HiddenStateResult objects)
report = analyze_hidden_state_batch(hidden_states, normalize=True)

# Option 2: Manual extraction
import numpy as np
vectors = np.stack([hs.l2_normalize().vector for hs in hidden_states])
report = analyze_kakeya_geometry(vectors)

# Check results
print(f"Health: {report.overall_health}")
print(f"Effective dim: {report.directional_coverage.effective_dim}")
print(f"Wolf severity: {report.wolf_axiom.severity}")
```

### Adding to HTTP Response

Modify `transport/http.py` to include Kakeya analysis:

```python
from the_loom.src.analysis.kakeya_geometry import analyze_kakeya_geometry

class ChatCompletionResponse(BaseModel):
    # ... existing fields ...
    kakeya_geometry: dict | None = Field(
        default=None,
        description="Kakeya geometry analysis of hidden states"
    )

# In the generate endpoint:
if return_hidden_states and hidden_state_result:
    vectors = hidden_state_result.vector.reshape(1, -1)
    kakeya_report = analyze_kakeya_geometry(vectors)
    response.kakeya_geometry = kakeya_report.to_dict()
```

## Experiment Protocols

### Experiment 1: Baseline Geometry Characterization

**Question:** Do real hidden states differ from random in Kakeya properties?

```python
import numpy as np
from kakeya_geometry import analyze_kakeya_geometry

def experiment_1_baseline(model_loader, test_prompts: list[str]):
    """Compare real hidden states to random baseline."""

    results = {
        "real": [],
        "random": [],
    }

    for prompt in test_prompts:
        # Get real hidden state
        output = model_loader.generate(prompt, return_hidden_states=True)
        real_vector = output.hidden_states[-1]  # Last layer

        # Generate random baseline (same shape)
        random_vector = np.random.randn(*real_vector.shape)

        # Analyze both
        results["real"].append(analyze_kakeya_geometry(real_vector))
        results["random"].append(analyze_kakeya_geometry(random_vector))

    # Compare distributions
    real_coverage = [r.directional_coverage.coverage_ratio for r in results["real"]]
    random_coverage = [r.directional_coverage.coverage_ratio for r in results["random"]]

    real_wolf = [r.wolf_axiom.max_density_ratio for r in results["real"]]
    random_wolf = [r.wolf_axiom.max_density_ratio for r in results["random"]]

    print(f"Coverage ratio - Real: {np.mean(real_coverage):.3f} vs Random: {np.mean(random_coverage):.3f}")
    print(f"Wolf density - Real: {np.mean(real_wolf):.3f} vs Random: {np.mean(random_wolf):.3f}")

    # Statistical test
    from scipy import stats
    _, p_coverage = stats.mannwhitneyu(real_coverage, random_coverage)
    _, p_wolf = stats.mannwhitneyu(real_wolf, random_wolf)

    return {
        "coverage_different": p_coverage < 0.05,
        "wolf_different": p_wolf < 0.05,
        "real_healthier": np.mean(real_wolf) < np.mean(random_wolf),
    }
```

**Expected if hypothesis holds:** Real hidden states should show healthier Kakeya geometry than random vectors (lower density violations, appropriate coverage).

**Falsification:** If random vectors show equal or better Kakeya properties, the hypothesis that transformers learn Kakeya-constrained representations is weakened.

### Experiment 2: Layer-wise Geometry Evolution

**Question:** How do Kakeya properties evolve through transformer layers?

```python
def experiment_2_layerwise(model_loader, test_prompts: list[str]):
    """Track Kakeya geometry across layers."""

    layer_reports = {}  # layer_idx -> list of reports

    for prompt in test_prompts:
        # Get hidden states from ALL layers
        output = model_loader.generate(
            prompt,
            return_hidden_states=True,
            hidden_state_layers=list(range(model_loader.num_layers))
        )

        for layer_idx, hidden_state in output.hidden_states.items():
            if layer_idx not in layer_reports:
                layer_reports[layer_idx] = []

            report = analyze_kakeya_geometry(hidden_state.reshape(1, -1))
            layer_reports[layer_idx].append(report)

    # Aggregate by layer
    layer_summary = {}
    for layer_idx, reports in sorted(layer_reports.items()):
        layer_summary[layer_idx] = {
            "effective_dim": np.mean([r.directional_coverage.effective_dim for r in reports]),
            "coverage_ratio": np.mean([r.directional_coverage.coverage_ratio for r in reports]),
            "wolf_severity": np.mean([r.wolf_axiom.max_density_ratio for r in reports]),
            "isotropy": np.mean([r.directional_coverage.isotropy_score for r in reports]),
        }

    return layer_summary
```

**Expected patterns (if hypothesis holds):**
- Early layers: Lower effective dimension (generic features)
- Middle layers: Peak effective dimension (maximum expressiveness)
- Final layers: Moderate dimension, good Wolf properties (task-ready)

This mirrors the "hunchback" pattern observed in intrinsic dimension research.

### Experiment 3: Bilateral Conveyance Test

**Question:** Does geometric alignment predict successful information transfer?

```python
from kakeya_geometry import compare_bilateral_geometry, run_conveyance_experiment

def experiment_3_bilateral(weaver_session):
    """Test if Kakeya geometry alignment predicts conveyance success."""

    sender_states = []
    receiver_states = []
    task_success = []

    # Run multiple Senior → Junior interactions via Weaver
    for task in test_tasks:
        # Senior generates response
        senior_output = weaver_session.senior.generate(
            task.prompt,
            return_hidden_states=True
        )
        sender_states.append(senior_output.hidden_state.vector)

        # Junior receives and processes
        junior_output = weaver_session.junior.generate(
            senior_output.text,
            return_hidden_states=True
        )
        receiver_states.append(junior_output.hidden_state.vector)

        # Evaluate if transfer succeeded
        success = evaluate_task_completion(task, junior_output.text)
        task_success.append(success)

    # Run the conveyance experiment
    results = run_conveyance_experiment(
        sender_states=sender_states,
        receiver_states=receiver_states,
        task_success=task_success,
    )

    print(f"Alignment-Success Correlation: {results['alignment']['correlation_with_success']:.3f}")
    print(f"P-value: {results['alignment']['p_value']:.4f}")
    print(f"Hypothesis Supported: {results['hypothesis_support']['alignment_predicts_success']}")

    return results
```

**Key falsification test:** If `alignment_predicts_success` is False across multiple runs with sufficient data, the Kakeya-conveyance hypothesis is seriously weakened.

### Experiment 4: Perturbation Study

**Question:** Does artificially violating Kakeya properties hurt performance?

```python
def experiment_4_perturbation(model_loader, test_prompts: list[str]):
    """Test if violating Kakeya geometry degrades task performance."""

    results = {
        "original": [],
        "wolf_violated": [],
        "coverage_reduced": [],
    }

    for prompt in test_prompts:
        # Get original hidden state
        output = model_loader.generate(prompt, return_hidden_states=True)
        original_hidden = output.hidden_states[-1]

        # Create Wolf-violated version (concentrate in thin slab)
        wolf_violated = create_wolf_violation(original_hidden)

        # Create coverage-reduced version (project to lower dim)
        coverage_reduced = reduce_coverage(original_hidden, target_dim=50)

        # Continue generation from each hidden state
        for name, hidden in [
            ("original", original_hidden),
            ("wolf_violated", wolf_violated),
            ("coverage_reduced", coverage_reduced)
        ]:
            # Inject modified hidden state and continue
            continuation = model_loader.continue_from_hidden(hidden, max_tokens=100)

            # Evaluate quality
            quality_score = evaluate_generation_quality(prompt, continuation)
            results[name].append(quality_score)

    # Compare
    from scipy import stats

    orig_vs_wolf = stats.mannwhitneyu(results["original"], results["wolf_violated"])
    orig_vs_coverage = stats.mannwhitneyu(results["original"], results["coverage_reduced"])

    return {
        "wolf_violation_hurts": np.mean(results["original"]) > np.mean(results["wolf_violated"]),
        "wolf_p_value": orig_vs_wolf.pvalue,
        "coverage_reduction_hurts": np.mean(results["original"]) > np.mean(results["coverage_reduced"]),
        "coverage_p_value": orig_vs_coverage.pvalue,
    }

def create_wolf_violation(hidden_state: np.ndarray) -> np.ndarray:
    """Concentrate vectors into thin slab (violate Wolf density)."""
    # Project onto first few principal components only
    pca = PCA(n_components=5)
    reduced = pca.fit_transform(hidden_state.reshape(1, -1))
    reconstructed = pca.inverse_transform(reduced)
    return reconstructed

def reduce_coverage(hidden_state: np.ndarray, target_dim: int) -> np.ndarray:
    """Reduce effective dimensionality."""
    # Zero out components beyond target_dim
    u, s, vt = np.linalg.svd(hidden_state.reshape(1, -1), full_matrices=False)
    s[target_dim:] = 0
    return (u @ np.diag(s) @ vt).reshape(hidden_state.shape)
```

**Note:** Experiment 4 requires hidden state injection, which may not be supported by all model architectures. This is an advanced test.

## Metrics Reference

### Wolf Axiom Metrics

| Metric | Healthy Range | Warning | Critical |
|--------|---------------|---------|----------|
| `max_density_ratio` | < 1.5 | 1.5 - 2.5 | > 2.5 |
| `violation_count` | 0 | 1-3 | > 3 |
| `uniformity_p_value` | > 0.05 | 0.01 - 0.05 | < 0.01 |

### Directional Coverage Metrics

| Metric | Interpretation |
|--------|----------------|
| `effective_dim` | Dimensions for 95% variance |
| `coverage_ratio` | effective_dim / ambient_dim |
| `spherical_uniformity` | 0-1, how uniform on sphere |
| `isotropy_score` | 0-1, how spherical (vs elongated) |

Coverage quality thresholds:
- `degenerate`: coverage_ratio < 0.1
- `sparse`: 0.1 - 0.3
- `moderate`: 0.3 - 0.6
- `full`: > 0.6

### Bilateral Alignment Metrics

| Metric | What It Measures |
|--------|------------------|
| `directional_alignment` | Cosine of mean directions |
| `subspace_overlap` | Principal component alignment |
| `grain_alignment` | Cluster structure correspondence |
| `density_similarity` | Wolf profile similarity |
| `overall_alignment` | Weighted combination (0-1) |

## Decision Framework

After running experiments, use this framework:

```
IF baseline_experiment shows real ≠ random geometry:
    → Transformers learn structured representations (continue)
ELSE:
    → Kakeya structure not learned (weaken hypothesis)

IF layerwise_experiment shows evolution pattern:
    → Processing transforms geometry systematically (continue)
ELSE:
    → Geometry random across layers (weaken hypothesis)

IF bilateral_experiment shows alignment predicts success:
    → Kakeya geometry governs conveyance (SUPPORT hypothesis)
ELSE:
    → Geometry uncorrelated with transfer (REJECT hypothesis)

IF perturbation_experiment shows violations hurt performance:
    → Kakeya constraints are causal (STRONG SUPPORT)
ELSE:
    → Correlation without causation (moderate support at best)
```

## Logging & Analysis

Integrate with your existing telemetry:

```python
# In Weaver session logging
import json

def log_kakeya_metrics(session_id: str, report: KakeyaGeometryReport):
    """Log Kakeya metrics for later analysis."""
    metrics = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "kakeya": report.to_dict(),
    }

    # Append to JSONL file
    with open("kakeya_metrics.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")
```

Then analyze with:

```python
import pandas as pd

# Load metrics
metrics = pd.read_json("kakeya_metrics.jsonl", lines=True)

# Expand nested structure
metrics = pd.concat([
    metrics,
    pd.json_normalize(metrics["kakeya"])
], axis=1)

# Correlate with outcomes
correlation = metrics.groupby("task_type").apply(
    lambda g: g["wolf_axiom.max_density_ratio"].corr(g["success"])
)
```

## Next Steps

1. **Implement** `kakeya_geometry.py` in TheLoom
2. **Run Experiment 1** (baseline) on multiple models
3. **If baseline passes**, run Experiment 2 (layerwise)
4. **If patterns emerge**, run Experiment 3 (bilateral) with Weaver
5. **Document findings** - positive OR negative
6. **Share results** regardless of whether they support the hypothesis

Remember: **Negative results are valuable.** If Kakeya geometry doesn't correlate with conveyance, that's important scientific information that saves future researchers from pursuing a dead end.
