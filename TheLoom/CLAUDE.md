# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**TheLoom** is a research repository for measuring semantic information transfer between AI agents. It contains:

1. **The Conveyance Hypothesis (v4.1)** - Mathematical framework extending Shannon's information theory to include semantic effectiveness
2. **The Loom** - Python/FastAPI server exposing hidden states for geometric analysis

## Structure

```
TheLoom/
├── conveyance-hypothesis-v4.1.md           # Theoretical framework
└── the-loom/                               # Implementation
    ├── src/                                # Python source
    └── tests/                              # Test suite
```

## The Loom

Hidden state extraction for transformer models - the capability that production inference servers don't provide.

```
Input → [Transformer Layers] → Hidden State → [lm_head] → Logits → Tokens
                                    ↑
                          THE LOOM EXPOSES THIS
```

## Commands (the-loom)

```bash
cd the-loom

# Install
pip install -e .
pip install -e ".[dev]"  # With dev dependencies

# Run server
loom                        # Default (port 8080)
loom --port 9000            # Custom port
loom --preload model-id     # Preload model

# Development
pytest                      # Run tests
mypy src --pretty           # Type check
ruff format src tests       # Format
ruff check src tests        # Lint
```

## Core Concepts

### The Conveyance Framework

The primary equation measures bilateral information transfer effectiveness:

```
C_pair(i ↔ j) = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij

Where:
C_out = W × R × H / T  (sender capacity)
C_in  = W × R × H / T  (receiver capacity)
```

**Key Variables:**
| Symbol | Name | Meaning |
|--------|------|---------|
| W | Recognition | Signal vs noise distinction |
| R | Relational Discovery | Geometric positioning in embedding space |
| H | Computational Frame | Processing throughput (includes hardware) |
| T | Temporal Investment | Processing time allocated |
| D_eff | Effective Dimensionality | Semantic richness (PCA-based, 90% variance) |
| β | Collapse Indicator | Diagnostic for dimensional compression (target: < 2.0) |
| P_ij | Protocol Compatibility | Interface match between agents [0,1] |

### Boundary Objects

Low-dimensional representations that externalize high-dimensional semantic states for transmission between bounded networks. The hidden state before `lm_head` projection is the critical "boundary object" this framework aims to capture.

### The Loom's Purpose

Exposes the final hidden state that production servers (Ollama, vLLM) hide:

```json
POST /generate
{
  "model": "model-name",
  "prompt": "...",
  "return_hidden_states": true
}

Response includes:
{
  "text": "...",
  "hidden_states": {
    "-1": {
      "data": [0.123, -0.456, ...],
      "shape": [4096],
      "dtype": "float32"
    }
  }
}
```

## Key Design Principles

1. **Research-first, not production-first** - Expose internals that production servers hide
2. **Harmonic mean constraint** - Transfer limited by the weaker participant (sender or receiver)
3. **Multiplicative structure** - Any zero component produces zero conveyance (zero-propagation principle)
4. **D_eff is primary metric** - Effective dimensionality measures semantic preservation
5. **β is diagnostic only** - Never optimize for β directly; it indicates collapse

## Falsification Criteria

The hypothesis would be falsified by:
- β showing positive correlation with performance (current: r ≈ -0.92 negative)
- High-dimensional (512D+) consistently outperforming 128-256D representations
- Boundary scaffolding outperforming attention-only mechanisms
- Bilateral asymmetry having no predictive value for alignment

## Related Documentation

- `conveyance-hypothesis-v4.1.md` - Full theoretical framework with citations
- `the-loom/README.md` - Server usage and API documentation
