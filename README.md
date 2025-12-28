# WeaverTools

Multi-agent AI orchestration ecosystem for conveyance hypothesis research.

## Overview

WeaverTools is a collection of Go and Python modules that enable multi-agent AI conversations with hidden state extraction for measuring semantic information transfer (conveyance).

```
┌─────────────────────────────────────────────────────────────┐
│                        WEAVER                                │
│                    (Orchestrator CLI)                        │
│                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │   Senior    │    │   Junior    │    │ Conversant  │    │
│   │ (Claude)    │    │   (Loom)    │    │   (Loom)    │    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │
│          │                  │                  │            │
│          │     ┌────────────┴────────────┐     │            │
│          │     │         YARN            │     │            │
│          │     │  (Conversations &       │     │            │
│          │     │   Measurements)         │     │            │
│          │     └─────────────────────────┘     │            │
│          │                                     │            │
│          │     ┌─────────────────────────┐     │            │
│          │     │         WOOL            │     │            │
│          │     │   (Agent Definitions)   │     │            │
│          │     └─────────────────────────┘     │            │
└──────────┼─────────────────────────────────────┼────────────┘
           │                                     │
           ▼                                     ▼
    ┌─────────────┐                      ┌─────────────┐
    │ Claude Code │                      │  The Loom   │
    │ (Subprocess)│                      │  (Server)   │
    └─────────────┘                      └─────────────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │Hidden States│
                                         │ (Geometry)  │
                                         └─────────────┘
```

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| **Wool** | Go | Agent role definitions and capabilities |
| **Yarn** | Go | Conversations, messages, measurements, sessions |
| **Weaver** | Go | Orchestrator CLI with backends and interactive shell |
| **TheLoom** | Python | Hidden state extraction server for transformer models |

## The Weaving Metaphor

- **Wool** - The raw material; defines WHAT an agent IS (roles, capabilities)
- **Yarn** - The thread; carries information between agents (messages, measurements)
- **Weaver** - The craftsperson; orchestrates agents to create conversations
- **Loom** - The machine; processes the raw compute (model inference + hidden states)

## Quick Start

```bash
# Start The Loom (hidden state server)
cd TheLoom/the-loom
pip install -e .
loom

# Build and run Weaver
cd Weaver
go build -o build/weaver ./cmd/weaver
./build/weaver
```

## Configuration

Edit `Weaver/config.yaml` to configure agents:

```yaml
agents:
  senior:
    role: senior
    backend: claudecode
    active: true

  junior:
    role: junior
    backend: loom
    model: Qwen/Qwen2.5-Coder-7B-Instruct
    gpu: auto
    active: true

  conversant1:
    role: conversant
    backend: loom
    model: Qwen/Qwen2.5-Coder-7B-Instruct
    gpu: "0"
    active: false
```

## Version Pinning

WeaverTools follows [Semantic Versioning](https://semver.org/) with a 6-month minimum deprecation policy for all public APIs. This ensures research reproducibility - experiments can be re-run with identical dependencies months or years later.

### Using WeaverTools as a Dependency

To use WeaverTools modules in your Go project with version pinning:

```bash
# Add a specific version of Yarn to your project
go get github.com/r3d91ll/yarn@v1.0.0
```

This adds the following to your `go.mod`:

```go
// go.mod - single module
require github.com/r3d91ll/yarn v1.0.0

// go.mod - multiple modules
require (
    github.com/r3d91ll/yarn v1.0.0   // Conversation and measurement types
    github.com/r3d91ll/wool v1.0.0   // Agent role definitions
    github.com/r3d91ll/weaver v1.0.0 // Orchestration utilities
)
```

### Version Pinning Best Practices

1. **For long-running experiments**: Pin to a specific patch version (e.g., `v1.0.0`)
2. **For active development**: Use minor version constraints (e.g., `v1.0.x`)
3. **Check compatibility**: See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for version matrix

### Version Guarantees

| Change Type | Version Bump | Compatibility |
|------------|--------------|---------------|
| Bug fixes | Patch (1.0.x) | Fully backward compatible |
| New features | Minor (1.x.0) | Backward compatible |
| Breaking changes | Major (x.0.0) | Migration guide provided |

For detailed versioning policy and deprecation procedures, see [docs/VERSIONING.md](docs/VERSIONING.md).

## Conveyance Hypothesis

This ecosystem is designed to validate the Conveyance Hypothesis - that semantic information transfer between AI agents can be measured geometrically through hidden state analysis.

Key metrics:
- **D_eff** (Effective Dimensionality) - Semantic richness via PCA
- **β** (Collapse Indicator) - Dimensional compression diagnostic
- **C_pair** - Bilateral conveyance between agent pairs

## License

MIT
