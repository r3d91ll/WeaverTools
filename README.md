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

## Conveyance Hypothesis

This ecosystem is designed to validate the Conveyance Hypothesis - that semantic information transfer between AI agents can be measured geometrically through hidden state analysis.

Key metrics:
- **D_eff** (Effective Dimensionality) - Semantic richness via PCA
- **β** (Collapse Indicator) - Dimensional compression diagnostic
- **C_pair** - Bilateral conveyance between agent pairs

## License

MIT
