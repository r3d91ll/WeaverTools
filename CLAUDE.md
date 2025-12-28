# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**WeaverTools** is a multi-agent AI orchestration ecosystem with four main components:

```
WeaverTools/
├── Weaver/               # AI orchestration CLI (Go)
├── Wool/                 # Common types and interfaces (Go)
├── Yarn/                 # Shared utilities (Go)
├── TheLoom/              # Hidden state extraction server (Python)
│   └── the-loom/         # Server implementation
└── weaver-testing/       # Local workspace for testing
```

## The Weaver Ecosystem

```
                           ┌─────────────────────────┐
 YOU ──────► WEAVER ──────►│  SENIOR (Claude Code)   │
              (CLI)        │  - Complex tasks        │
               │           │  - Architecture         │
               │           │  - Review Junior's work │
               │           └──────────┬──────────────┘
               │                      │ /local <task>
               │                      ▼
               │           ┌─────────────────────────┐
               │           │  JUNIOR (Local Model)   │
               └──────────►│  - Simple tasks         │
                           │  - File operations      │
                           │  - Tests/linting        │
                           └──────────┬──────────────┘
                                      │
                           ┌──────────▼──────────────┐
                           │  THE LOOM (Hidden State)│
                           │  - Embedding extraction │
                           │  - Conveyance metrics   │
                           └─────────────────────────┘
```

## Development Commands

### Makefile (Recommended)

The repository includes a Makefile for standardized development workflows across all Go modules (Weaver, Wool, Yarn). **Use these commands instead of manual go commands.**

```bash
# From repository root

make build          # Build all modules (compile check)
make build-weaver   # Build the weaver binary (output: Weaver/weaver)

make test           # Run tests for all modules
make test-verbose   # Run tests with verbose output
make test-coverage  # Run tests with coverage report

make lint           # Run golangci-lint on all modules
make lint-fix       # Run golangci-lint with auto-fix

make vet            # Run go vet on all modules
make fmt            # Format all code with gofmt
make fmt-check      # Check if code is formatted (CI-friendly)

make check          # Run ALL quality checks (fmt-check, vet, lint, test)

make clean          # Remove build artifacts
make deps           # Download and verify dependencies
make deps-tidy      # Tidy dependencies (go mod tidy)

make help           # Show all available targets
```

**Note:** `golangci-lint` must be installed for lint commands:
```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

### Weaver (Go CLI) - Direct Commands

For running the Weaver CLI (after building):

```bash
./Weaver/weaver                      # Run interactive mode
./Weaver/weaver -m "message"         # Single query
./Weaver/weaver --trace proj         # With Phoenix tracing
```

### Go Libraries (Wool, Yarn) - Direct Commands

While `make` targets are preferred, you can still run commands directly:

```bash
# Wool - Common types
cd Wool
go build ./...
go test -v ./...
go vet ./...

# Yarn - Utilities
cd Yarn
go build ./...
go test -v ./...
go vet ./...
```

### The Loom (Hidden State Server)

```bash
cd TheLoom/the-loom

poetry install                        # Install dependencies
poetry run loom                       # Start server (port 8080)
poetry run loom --transport unix      # Unix socket mode
poetry run pytest                     # Run tests
poetry run pytest -m "not slow"       # Skip slow tests
poetry run mypy src --pretty
poetry run ruff format src tests
poetry run ruff check src tests
```

## Architecture

### Weaver

```
Weaver/
├── cmd/weaver/main.go           # CLI entry point
├── internal/
│   ├── senior/                  # Claude Code subprocess wrapper
│   │   ├── claude.go            # Claude CLI integration
│   │   └── adapter.go           # Senior provider interface
│   ├── junior/                  # Local model HTTP client
│   │   ├── model.go             # OpenAI-compatible client
│   │   └── mistral.go           # Mistral-specific handling
│   ├── orchestrator/            # Main coordination
│   │   ├── weaver.go            # Routing + delegation
│   │   └── prompts.go           # System prompts
│   ├── loader/                  # Model service detection
│   │   ├── services.go          # Service definitions
│   │   ├── ollama.go            # Ollama operations
│   │   └── lmstudio.go          # LM Studio operations
│   ├── assessment/              # Junior model evaluation
│   │   ├── challenges.go        # Coding challenges
│   │   ├── assessment.go        # Runner
│   │   └── report.go            # CLAUDE.md generation
│   ├── telemetry/               # Phoenix/OTEL tracing
│   ├── context/                 # Context window management
│   ├── memory/                  # Shared notepad
│   └── tools/                   # Tool executor for Junior
└── go.mod
```

### Wool (Common Types)

```
Wool/
├── types.go                     # Shared type definitions
└── go.mod
```

### Yarn (Utilities)

```
Yarn/
├── utils.go                     # Shared utilities
└── go.mod
```

### The Loom

```
the-loom/src/
├── server.py                    # CLI entry point (`loom`)
├── client.py                    # Python client library
├── config.py                    # Pydantic configuration
├── loaders/                     # Model loading
│   ├── base.py                  # ABC and LoadedModel
│   ├── registry.py              # Auto-detection
│   ├── transformers_loader.py   # HuggingFace (~80%)
│   ├── sentence_transformers_loader.py  # Embeddings (~15%)
│   └── custom_loader.py         # Edge cases (~5%)
├── transport/
│   └── http.py                  # FastAPI server
├── extraction/
│   └── hidden_states.py         # D_eff, beta metrics
└── utils/
    ├── gpu.py                   # GPU management
    └── serialization.py         # Tensor to JSON
```

## Key Concepts

### Senior/Junior Delegation

All user messages go to Claude (Senior). Claude decides when to delegate simple tasks to the local model (Junior). Junior's responses always return to Claude for review.

**Junior-appropriate tasks:**
- File searches, grep operations
- Running tests, linting
- Simple code generation (boilerplate, utilities)
- Reading file contents

**Senior-only tasks:**
- Architecture decisions
- Security-sensitive code
- Complex debugging
- Multi-step reasoning

### Junior Assessment

Evaluate local model capabilities with `/junior-assessment` command. Results saved to `CLAUDE.md` in working directory with:
- Scores across 6 categories (algorithms, data structures, code quality, real-world, tool use, problem solving)
- Delegation guidelines based on strengths/weaknesses
- Individual challenge details

### Conveyance Framework (The Loom)

The Loom exposes hidden states for measuring semantic information transfer:

```
C_pair(i ↔ j) = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij
```

**Key metrics:**
- **D_eff** (Effective Dimensionality): Semantic richness via PCA (90% variance)
- **β** (Collapse Indicator): Dimensional compression diagnostic (target: < 2.0)

### Service Auto-Detection

Weaver auto-detects running local model services:

| Service | Default URL | Detection |
|---------|-------------|-----------|
| LM Studio | localhost:1234 | `/v1/models` |
| Ollama | localhost:11434 | `/api/tags` |
| vLLM | localhost:8000 | `/v1/models` |
| LocalAI | localhost:8080 | `/v1/models` |

## Testing Patterns

### Go Components

**Using Makefile (recommended):**
```bash
make test           # Run all tests across all modules
make test-verbose   # Run tests with verbose output
make test-coverage  # Run tests with coverage report
```

**Direct commands (for specific packages/tests):**
```bash
# Weaver
cd Weaver
go test -v ./...                           # All tests
go test -v ./internal/assessment/...       # Specific package
go test -v -run TestName ./...             # Single test

# Wool / Yarn
cd Wool && go test -v ./...
cd Yarn && go test -v ./...
```

### The Loom (Python)

```bash
cd TheLoom/the-loom
poetry run pytest                          # All tests
poetry run pytest tests/test_file.py       # Single file
poetry run pytest -k "test_name"           # By name pattern
poetry run pytest -v                       # Verbose
poetry run pytest -m "not slow"            # Skip slow tests
```

## Configuration

### Weaver Shared Memory
- Location: `~/.weaver/shared.json`
- Used by both Senior and Junior for coordination

### The Loom Configuration
- Priority: Environment (`LOOM_*`) > Config file > Defaults
- Config location: `~/.config/loom/config.yaml`
- Transport modes: `http`, `unix`, `both`

## Common Workflows

### Adding a New Junior Challenge

1. Edit `Weaver/internal/assessment/challenges.go`
2. Add challenge to appropriate category
3. Add to `extendedChallenges` map with name, prompt, max_points
4. Run `go test ./...` to verify

### Adding a New Model Loader (The Loom)

1. Create loader in `TheLoom/the-loom/src/loaders/`
2. Implement `ModelLoader` ABC from `base.py`
3. Register patterns in `LoaderRegistry` (`registry.py`)
4. Add tests in `tests/test_loaders.py`

### Phoenix Tracing Setup

```bash
# Start Phoenix
docker run -d -p 6006:6006 arizephoenix/phoenix:latest

# Run Weaver with tracing
cd Weaver
./weaver --trace my-project-name

# View traces at http://localhost:6006
```

## Dependencies

### Go Components (Weaver, Wool, Yarn)
- Go 1.21+
- Claude CLI installed and authenticated

### Python (TheLoom)
- Python 3.10+
- torch, transformers, sentence-transformers
- FastAPI, uvicorn
- Optional: bitsandbytes (quantization), prometheus-client (metrics)

## Security & Credential Handling

### Important: Never Commit Credentials

**API tokens, passwords, and secrets must NEVER be committed to version control.** This includes:
- GitHub tokens (PATs: `ghp_*`, OAuth: `gho_*`, Fine-grained: `github_pat_*`)
- API keys (Anthropic: `sk-ant-*`, OpenAI: `sk-*`)
- Database passwords
- Private keys

If a credential is accidentally committed, consider it **compromised** and revoke it immediately.

### Using Environment Files

This repository uses `.env` files for credential storage. These files are gitignored and should never be committed.

**Setup:**
```bash
# Navigate to the auto-claude directory
cd .auto-claude

# Copy the template
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or your preferred editor
```

**The `.env.example` template** is safely committed and shows required variables without real values. Always use it as reference when setting up credentials.

### Credential Best Practices

1. **Use environment variables** - Store credentials in `.env` files or system environment variables
2. **Never hardcode secrets** - Don't put real tokens in source code, even temporarily
3. **Use minimal scopes** - When creating tokens, grant only the permissions needed
4. **Rotate regularly** - Periodically regenerate tokens and update your `.env` files
5. **Different tokens per environment** - Use separate tokens for development/staging/production

### Token Revocation

If you suspect a token has been exposed, **act immediately**:

1. **Revoke the token** on GitHub (see table below for locations)
2. **Generate a new token** with minimal required scopes
3. **Update your `.env` file** with the new token
4. **Audit usage** - Check [GitHub's security log](https://github.com/settings/security-log) for unauthorized access

| Token Type | Prefix | Revocation Location |
|------------|--------|---------------------|
| Personal Access Token (Classic) | `ghp_` | [Settings > Developer settings > Tokens (classic)](https://github.com/settings/tokens) |
| Fine-grained Token | `github_pat_` | [Settings > Developer settings > Fine-grained tokens](https://github.com/settings/personal-access-tokens) |
| OAuth Token | `gho_` | [Settings > Applications > Authorized OAuth Apps](https://github.com/settings/applications) |

For detailed step-by-step instructions, see **[docs/security/TOKEN-REVOCATION.md](docs/security/TOKEN-REVOCATION.md)**.

### Pre-commit Hook (Optional)

A pre-commit hook is available to scan for potential secrets before commits. This adds an extra layer of protection against accidental credential exposure.

**Installation:**
```bash
./hooks/install.sh           # Install the hook
./hooks/install.sh --remove  # Remove the hook
```

The hook scans staged files for:
- GitHub tokens (OAuth: `gho_*`, PAT: `ghp_*`, Fine-grained: `github_pat_*`)
- Anthropic/OpenAI API keys
- AWS access keys
- Private key files
- Generic secret patterns

To bypass the hook in exceptional cases: `git commit --no-verify` (NOT recommended)
