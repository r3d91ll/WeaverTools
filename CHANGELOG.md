# Changelog

All notable changes to WeaverTools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [1.0.0] - 2024-12-28

### Added

- **Yarn**: Core message handling library for agentic workflows
  - Message struct with role and content fields
  - JSON marshaling/unmarshaling support
  - Thread-safe message operations

- **Wool**: Memory and state management library
  - Memory provider interface for pluggable backends
  - State tracking for agent conversations
  - Context windowing utilities

- **Weaver**: CLI orchestrator and API server
  - Command-line interface for agent orchestration
  - HTTP API server with backend routes
  - LLM backend integration (OpenAI, Claude, Ollama)
  - GPU configuration and status endpoints

- **TheLoom** (Python): Model server for layer-by-layer analysis
  - Hidden state extraction capabilities
  - GPU-accelerated inference support
  - FastAPI-based REST endpoints

- **Semantic Versioning**: Version management infrastructure
  - Git tag-based versioning (v1.0.0 format)
  - 6-month deprecation policy
  - Compatibility tracking

### Changed

- Standardized Go module paths across all services:
  - `github.com/r3d91ll/yarn`
  - `github.com/r3d91ll/wool`
  - `github.com/r3d91ll/weaver`

### Deprecated

- None (initial release)

### Removed

- None (initial release)

### Fixed

- None (initial release)

### Security

- Removed exposed GitHub token from version control
- Added pre-commit hook for secret detection
- Comprehensive .gitignore patterns for credential files

---

## Version History Summary

| Version | Date       | Highlights                           |
|---------|------------|--------------------------------------|
| 1.0.0   | 2024-12-28 | Initial stable release of all Go modules |

## Versioning Policy

WeaverTools follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

### Deprecation Policy

All deprecated APIs receive a minimum **6-month notice** before removal:

1. Deprecated features are marked with `// Deprecated:` godoc comments
2. Deprecated features appear in the CHANGELOG under "Deprecated"
3. Removal is announced at least one major version in advance
4. See [VERSIONING.md](docs/VERSIONING.md) for full deprecation policy

[Unreleased]: https://github.com/r3d91ll/WeaverTools/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/r3d91ll/WeaverTools/releases/tag/v1.0.0
