# Conveyance Model Loader - Integration Context Document

**Purpose:** This document provides context for building a research-focused model loader that will integrate with WeaverCode. Use this as the design foundation when starting the model loader project.

**Status:** WeaverCode 2.0 development is ON HOLD pending this model loader.

---

## Part 1: What is WeaverCode?

### Current State

WeaverCode is a Go-based CLI tool that orchestrates AI agents for software engineering tasks:

```
User → WeaverCode (Go) → Senior Engineer (Claude Code)
                      → Junior Engineer (Local LLM via Ollama/LM Studio)
```

**Repository:** `git@github.com:r3d91ll/weaver-code.git`
**Location:** `/home/todd/olympus/git-repos/weaver-code/go-port/`

### Current Architecture

```
go-port/
├── cmd/weaver/main.go           # CLI entry point
├── internal/
│   ├── senior/                  # Claude Code subprocess wrapper
│   ├── junior/                  # OpenAI-compatible HTTP client
│   │   └── model.go             # Talks to Ollama/LM Studio/vLLM
│   ├── orchestrator/            # Message routing, delegation
│   ├── loader/                  # Service detection (Ollama, LM Studio, etc.)
│   ├── assessment/              # Junior capability evaluation
│   ├── context/                 # Conversation history management
│   ├── memory/                  # Shared notepad between agents
│   ├── telemetry/               # OpenTelemetry/Phoenix tracing
│   └── tools/                   # Sandboxed tool execution
```

### How WeaverCode Talks to Local Models Today

```go
// internal/junior/model.go - OpenAI-compatible HTTP client
POST http://localhost:11434/v1/chat/completions
{
  "model": "devstral-weaver-v2",
  "messages": [...],
  "temperature": 0.15,
  "max_tokens": 16384
}

Response:
{
  "choices": [{"message": {"content": "..."}}],
  "usage": {"prompt_tokens": 100, "completion_tokens": 50}
}
```

**Problem:** This API returns ONLY text and token counts. No geometric data.

---

## Part 2: What WeaverCode Needs (Future State)

### The Vision: Conveyance Measurement Platform

WeaverCode 2.0 will be a multi-agent research platform for validating the Conveyance Hypothesis through geometric analysis of agent interactions.

### The Critical Missing Data Point

```
Input → [Transformer Layers] → Final Hidden State → [lm_head] → Logits → Tokens
                                      ↑
                              THIS IS THE BOUNDARY OBJECT

The geometric representation of meaning BEFORE it becomes text.
This is what we need to capture for conveyance measurement.
```

### What WeaverCode Will Request from Model Loader

```json
POST http://localhost:8000/v1/generate
{
  "model": "devstral-weaver-v2",
  "messages": [...],
  "return_hidden_states": true
}

Response:
{
  "text": "Here is the solution...",
  "usage": {"prompt_tokens": 100, "completion_tokens": 50},
  "hidden_state": {
    "final": [0.123, -0.456, ...],  // Last layer, last token (e.g., 4096 dims)
    "shape": [1, 4096],
    "layer": -1
  },
  "generation_metadata": {
    "latency_ms": 1234,
    "tokens_per_second": 45.2
  }
}
```

### Metrics WeaverCode Will Compute from Hidden States

| Metric | Description | TCF Reference |
|--------|-------------|---------------|
| D_eff | Effective dimensionality via PCA | Dimensional preservation |
| β | Collapse indicator | Overfitting detection |
| Geometric Alignment | Cosine similarity between agent states | Bilateral conveyance |
| Boundary Quality | Variance in hidden state | C_ext measurement |

---

## Part 3: Model Loader Design Philosophy

### THIS IS NOT OLLAMA/VLLM/LM STUDIO

**What they optimize for:**

- Thousands of concurrent users
- Production deployment at scale
- Maximum throughput (tokens/second)
- API compatibility for app developers

**What WE optimize for:**

- Small research team (1-5 users)
- Fast iteration and experimentation
- Data extraction for analysis
- Simplicity over scalability

### Design Principles for the Model Loader

1. **Research-First, Not Production-First**
   - Expose internals that production servers hide
   - Prioritize data richness over throughput
   - Simple deployment (single process, no orchestration)

2. **Fast Iteration**
   - Quick model swapping without restart
   - Hot-reload configurations
   - Minimal ceremony to try new things

3. **Observable by Default**
   - Hidden states always available (opt-out, not opt-in)
   - Built-in metrics export
   - Easy integration with analysis tools

4. **Single Team Scale**
   - No need for load balancing
   - No need for request queuing
   - No need for multi-tenant isolation
   - One GPU, one model, one researcher (or small team)

### What We DON'T Need (Yet)

- [ ] Horizontal scaling
- [ ] Request batching for throughput
- [ ] Multi-model serving
- [ ] Production hardening
- [ ] Kubernetes deployment
- [ ] API rate limiting

### What We DO Need

- [x] Final hidden state extraction
- [x] Simple HTTP API
- [x] Fast model loading
- [x] GPU memory management
- [x] Basic health checks
- [x] Clean shutdown

---

## Part 4: Integration Interface

### API Contract Between WeaverCode and Model Loader

**Endpoint:** `POST /v1/generate`

```python
# Request
{
  "model": str,                    # Model identifier
  "messages": [                    # Chat format
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "max_tokens": int,               # Max generation length
  "temperature": float,            # Sampling temperature
  "return_hidden_states": bool,    # Enable geometric data (default: true)
  "stream": bool                   # Streaming (future, optional)
}

# Response
{
  "text": str,                     # Generated text
  "usage": {
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int
  },
  "hidden_state": {                # Only if return_hidden_states=true
    "final": list[float],          # Final hidden state vector
    "shape": list[int],            # Tensor shape [batch, hidden_dim]
    "layer": int,                  # Which layer (-1 = last)
    "dtype": str                   # "float32" or "float16"
  },
  "metadata": {
    "model": str,
    "latency_ms": float,
    "tokens_per_second": float
  }
}
```

**Endpoint:** `GET /v1/health`

```json
{
  "status": "healthy",
  "model_loaded": "devstral-weaver-v2",
  "gpu_memory_used_gb": 12.4,
  "gpu_memory_total_gb": 24.0
}
```

**Endpoint:** `POST /v1/models/load`

```json
{
  "model": "path/to/model/or/hf-id",
  "dtype": "float16",
  "device": "cuda:0"
}
```

### WeaverCode Integration Point

```go
// internal/junior/model.go - Extended for conveyance loader
type ConveyanceResponse struct {
    Text        string        `json:"text"`
    Usage       Usage         `json:"usage"`
    HiddenState *HiddenState  `json:"hidden_state,omitempty"`
    Metadata    Metadata      `json:"metadata"`
}

type HiddenState struct {
    Final []float64 `json:"final"`
    Shape []int     `json:"shape"`
    Layer int       `json:"layer"`
    DType string    `json:"dtype"`
}
```

---

## Part 5: Technical Requirements for Model Loader

### Language & Framework

- **Python** (PyTorch + Transformers ecosystem)
- **FastAPI** for HTTP server (async, fast, simple)
- **Single-file or minimal structure** (research tool, not enterprise)

### Core Functionality

```python
# Pseudocode for what we need
class ConveyanceModelServer:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            output_hidden_states=True,  # THIS IS KEY
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, messages: list, max_tokens: int, temperature: float):
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # Extract final hidden state (last layer, last token, before lm_head)
        final_hidden = outputs.hidden_states[-1][-1][:, -1, :].cpu().numpy()

        text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return {
            "text": text,
            "hidden_state": {
                "final": final_hidden.tolist(),
                "shape": list(final_hidden.shape),
                "layer": -1
            }
        }
```

### GPU Requirements

- Single GPU operation (RTX A6000 48GB or similar)
- Support for 7B-24B parameter models
- FP16/BF16 inference
- No multi-GPU complexity initially

---

## Part 6: Future Integration Path

### Phase 1: Basic Model Loader (Separate Project)

1. Build standalone Python model server
2. Implement `/v1/generate` with hidden state extraction
3. Test with single model on local GPU
4. Validate hidden state data is correct

### Phase 2: WeaverCode Integration

1. Add `conveyance` backend type to WeaverCode
2. Implement client for model loader API
3. Add `GeometricMeasurement` struct to capture hidden states
4. Store measurements alongside conversation history

### Phase 3: Conveyance Analysis

1. Implement D_eff calculation (PCA on hidden states)
2. Implement β collapse detection
3. Build bilateral conveyance measurement between agents
4. Export data for analysis

### Phase 4: Scale (Later, If Needed)

- Multi-model support
- Request batching
- Remote deployment options

---

## Part 7: Project Boundaries

### Model Loader Project (NEW, SEPARATE)

**Repository:** TBD (e.g., `conveyance-loader` or `geometric-serve`)
**Language:** Python
**Scope:**

- Load models via HuggingFace transformers
- Expose HTTP API with hidden state extraction
- Single GPU, single model, research-focused

### WeaverCode Project (EXISTING, ON HOLD)

**Repository:** `git@github.com:r3d91ll/weaver-code.git`
**Language:** Go
**Scope:**

- Multi-agent orchestration
- Conversation management
- Conveyance measurement (after model loader exists)
- CLI interface

### Clean Integration

When model loader is ready:

1. WeaverCode adds new backend type (`conveyance`)
2. WeaverCode calls model loader via HTTP
3. No Python code in WeaverCode repo
4. No Go code in model loader repo
5. Clean API boundary between projects

---

## Summary

**What we're building:** A research-focused model loader that exposes the final hidden state (the geometric boundary object) before text generation.

**Why:** To enable WeaverCode to measure conveyance - the effectiveness of information transfer between AI agents through geometric analysis.

**Design philosophy:** Simple, fast-to-iterate, research-first. Not a production server.

**Integration:** HTTP API returning hidden states alongside generated text. WeaverCode consumes this API.

**Next step:** Build the model loader as a separate Python project, then return to WeaverCode 2.0 development.
