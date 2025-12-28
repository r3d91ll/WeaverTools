# WeaverTools Versioning Policy

This document outlines the semantic versioning strategy, deprecation policy, and release processes for WeaverTools. Our commitment to API stability ensures that researchers can run multi-month or multi-year experiments with confidence.

## Table of Contents

- [Semantic Versioning](#semantic-versioning)
- [Deprecation Policy](#deprecation-policy)
- [Godoc Deprecation Format](#godoc-deprecation-format)
- [Git Tagging Workflow](#git-tagging-workflow)
- [Breaking Change Guidelines](#breaking-change-guidelines)
- [Version Pinning](#version-pinning)
- [Release Checklist](#release-checklist)

## Semantic Versioning

WeaverTools follows [Semantic Versioning 2.0.0](https://semver.org/) for all public APIs. Version numbers use the format `vMAJOR.MINOR.PATCH`:

| Component | When to Increment | Example |
|-----------|------------------|---------|
| **MAJOR** | Breaking changes to public API | `v1.0.0` → `v2.0.0` |
| **MINOR** | New features, backward compatible | `v1.0.0` → `v1.1.0` |
| **PATCH** | Bug fixes, backward compatible | `v1.0.0` → `v1.0.1` |

### Version Guarantees

- **Same MAJOR version**: Full backward compatibility guaranteed
- **Same MINOR version**: Bug fixes only, no new features removed
- **Pre-1.0 (v0.x.x)**: API may change; minor version bumps may include breaking changes

### Go Module Versioning

Go modules use Git tags for version management. The version is **not** declared in `go.mod`:

```go
// go.mod - module declaration (NO version here)
module github.com/r3d91ll/yarn

go 1.23.4

// Versions are declared via Git tags only:
// git tag -a v1.0.0 -m "Initial release"
```

**Important**: For major version 2 and above, Go requires the import path to include the major version:

```go
// v1.x.x - standard import
import "github.com/r3d91ll/yarn"

// v2.x.x - import path must include /v2
import "github.com/r3d91ll/yarn/v2"
```

## Deprecation Policy

WeaverTools enforces a **minimum 6-month deprecation window** for all public APIs. This ensures researchers have adequate time to migrate their code.

### Deprecation Timeline

| Stage | Action | Timeline |
|-------|--------|----------|
| **Announcement** | API marked deprecated with godoc comment | Day 0 |
| **Documentation** | Added to DEPRECATIONS.md tracking document | Day 0 |
| **CHANGELOG** | Noted in "Deprecated" section | Day 0 |
| **Grace Period** | API remains functional | Months 1-6 |
| **Removal Eligible** | API may be removed in next MAJOR version | Month 6+ |
| **Removal** | API removed, MAJOR version incremented | After Month 6 |

### Deprecation Rules

1. **No surprise removals**: Every deprecated API must have at least 6 months notice
2. **Clear migration path**: Deprecation notices must include replacement API or migration guide
3. **Version tracking**: Deprecation notice must specify version deprecated and removal version
4. **Date tracking**: Deprecation notice must include removal eligibility date
5. **Quarterly audits**: Deprecations are reviewed quarterly to ensure timely removal

### Example Timeline

```text
January 2025:  DoOldThing() deprecated in v1.5.0
               Removal eligible: July 2025 (6 months)

July 2025:     v2.0.0 released with DoOldThing() removed
               Breaking change documented in CHANGELOG.md
```

## Godoc Deprecation Format

Since Go lacks built-in deprecation warnings, we use standardized godoc comments. This format is recognized by IDEs, linters, and documentation tools.

### Standard Format

```go
// Deprecated: DoOldThing is deprecated as of v1.5.0 and will be removed in v2.0.0.
// Use DoNewThing instead. Migration guide: https://docs.weavertools.dev/migration/v2
//
// Removal scheduled for: 2025-07-01 (6 months from deprecation)
func DoOldThing() error {
    // Implementation remains functional until removal
    return nil
}

// DoNewThing replaces DoOldThing with improved error handling.
func DoNewThing() error {
    // New implementation
    return nil
}
```

### Key Elements

| Element | Required | Description |
|---------|----------|-------------|
| `// Deprecated:` | Yes | Must be first word on line for tool recognition |
| Function name | Yes | Clarify which function is deprecated |
| Version deprecated | Yes | When the deprecation started (e.g., "as of v1.5.0") |
| Removal version | Yes | When the API will be removed (e.g., "will be removed in v2.0.0") |
| Replacement | Yes | What to use instead (e.g., "Use DoNewThing instead") |
| Migration guide | Recommended | Link to detailed migration documentation |
| Removal date | Yes | Specific date, minimum 6 months out |

### Type and Package Deprecation

For deprecated types:

```go
// Deprecated: OldConfig is deprecated as of v1.3.0 and will be removed in v2.0.0.
// Use NewConfig instead, which supports additional validation options.
//
// Removal scheduled for: 2025-09-01
type OldConfig struct {
    // Fields...
}
```

For deprecated packages (in doc.go):

```go
// Deprecated: Package oldutils is deprecated as of v1.4.0.
// Use github.com/r3d91ll/yarn/utils instead.
//
// This package will be removed in v2.0.0 (scheduled: 2025-10-01).
package oldutils
```

### Linting for Deprecated APIs

WeaverTools uses `staticcheck` to detect usage of deprecated APIs:

```yaml
# .golangci.yml
linters:
  enable:
    - staticcheck

linters-settings:
  staticcheck:
    checks:
      - "SA1019"  # Warns when using deprecated functions/types
```

## Git Tagging Workflow

All versions are managed exclusively through Git tags. This section describes how to create releases.

### Creating a Release Tag

```bash
# 1. Ensure you're on the main branch with latest changes
git checkout main
git pull origin main

# 2. Verify all tests pass
go test ./...

# 3. Create an annotated tag (REQUIRED - never use lightweight tags)
git tag -a v1.2.3 -m "Release v1.2.3

Summary of changes:
- Added new feature X
- Fixed bug Y
- Deprecated API Z (removal: v2.0.0, date: 2025-07-01)

Breaking changes: None

Full changelog: https://github.com/r3d91ll/WeaverTools/blob/main/CHANGELOG.md"

# 4. Push the tag to remote
git push origin v1.2.3
```

### Tag Requirements

| Requirement | Correct | Incorrect |
|-------------|---------|-----------|
| Format | `v1.2.3` | `1.2.3`, `version-1.2.3` |
| Type | Annotated (`git tag -a`) | Lightweight (`git tag`) |
| Message | Includes changelog summary | Empty or minimal |
| Signed | Optional but recommended | N/A |

### Verifying Tags

```bash
# List all version tags
git tag -l 'v*'

# Verify a tag is annotated (output should be 'tag', not 'commit')
git for-each-ref refs/tags/v1.0.0 --format='%(objecttype)'

# View tag details
git show v1.0.0 --no-patch

# Verify tag matches semver pattern
git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$'
```

### Monorepo Tagging

WeaverTools uses a monorepo structure. A single tag covers all Go modules:

```text
v1.0.0 applies to:
├── Yarn   (github.com/r3d91ll/yarn)
├── Wool   (github.com/r3d91ll/wool)
└── Weaver (github.com/r3d91ll/weaver)
```

**Note**: TheLoom (Python) uses independent versioning via Poetry/pyproject.toml.

## Breaking Change Guidelines

A breaking change is any modification that requires consumers to update their code. All breaking changes require a MAJOR version bump.

### What Constitutes a Breaking Change

| Category | Breaking | Not Breaking |
|----------|----------|--------------|
| **Function Removal** | Removing any exported function | Adding new exported functions |
| **Signature Change** | Changing parameter types or order | Adding optional parameters |
| **Return Type** | Changing return type | Adding additional return values |
| **Struct Fields** | Removing or renaming exported fields | Adding new optional fields |
| **Interface** | Adding new required methods | Adding new types implementing interface |
| **Behavior** | Changing documented behavior | Fixing bugs to match documentation |
| **Error Types** | Changing error types | Adding new error types |

### Breaking Change Process

1. **Document the need**: Create an issue explaining why the change is necessary
2. **Deprecate first**: Mark the old API deprecated with 6-month notice
3. **Provide migration guide**: Document how to migrate to the new API
4. **Update CHANGELOG**: Add to "Deprecated" section immediately
5. **Wait for grace period**: Minimum 6 months before removal
6. **Remove in MAJOR release**: Only remove when incrementing MAJOR version
7. **Document removal**: Update CHANGELOG "Removed" section

### Migration Guide Template

Create migration guides at `docs/migration/vX.md`:

```markdown
# Migration Guide: v1.x to v2.0

## Breaking Changes Summary

1. `DoOldThing()` removed - use `DoNewThing()` instead
2. `OldConfig` type removed - use `NewConfig` instead

## Detailed Migration Steps

### DoOldThing → DoNewThing

**Before (v1.x):**
```go
result := DoOldThing(param)
```

**After (v2.0):**
```go
result, err := DoNewThing(param)
if err != nil {
    // Handle error
}
```
```

## Version Pinning

Consumers should pin to specific versions for reproducible builds.

### Pinning in go.mod

```go
module github.com/example/my-research

go 1.23

require (
    github.com/r3d91ll/yarn v1.0.0    // Pin to exact version
    github.com/r3d91ll/wool v1.0.0
    github.com/r3d91ll/weaver v1.0.0
)
```

### Version Constraints

| Constraint | Meaning | Use Case |
|------------|---------|----------|
| `v1.0.0` | Exact version | Research reproducibility |
| `v1.0` | Latest patch of v1.0.x | Accept bug fixes only |
| `v1` | Latest of v1.x.x | Accept new features |

### Updating Pinned Versions

```bash
# Update to specific version
go get github.com/r3d91ll/yarn@v1.1.0

# Update to latest within major version
go get github.com/r3d91ll/yarn@v1

# Check for available updates
go list -m -u github.com/r3d91ll/yarn
```

## Release Checklist

Use this checklist before creating any release:

### Pre-Release

- [ ] All tests pass (`go test ./...`)
- [ ] All services build (`go build ./...`)
- [ ] CHANGELOG.md updated with all changes
- [ ] Deprecated APIs have 6+ months notice
- [ ] Migration guides written for breaking changes
- [ ] Version number follows semver correctly
- [ ] Compatibility matrix updated

### Creating the Release

- [ ] Tag is annotated (not lightweight)
- [ ] Tag follows `vX.Y.Z` format
- [ ] Tag message includes changelog summary
- [ ] Tag pushed to remote

### Post-Release

- [ ] GitHub release created with full notes
- [ ] DEPRECATIONS.md updated if applicable
- [ ] Compatibility matrix reflects new version
- [ ] Team notified of release

## Quick Reference

### Semver Decision Tree

```text
Is this a bug fix with no API changes?
├── Yes → PATCH (v1.0.0 → v1.0.1)
└── No
    Is this a new feature without breaking changes?
    ├── Yes → MINOR (v1.0.0 → v1.1.0)
    └── No
        Does this remove, rename, or change any public API?
        ├── Yes → MAJOR (v1.0.0 → v2.0.0)
        └── No → Reassess: likely MINOR or PATCH
```

### Key Commands

```bash
# Create annotated release tag
git tag -a v1.2.3 -m "Release message"

# Push tag to remote
git push origin v1.2.3

# Verify tag is annotated
git for-each-ref refs/tags/v1.2.3 --format='%(objecttype)'

# Run changelog generator
make changelog
```

### Related Documents

- [CHANGELOG.md](../CHANGELOG.md) - Version history and changes
- [COMPATIBILITY.md](./COMPATIBILITY.md) - Cross-version compatibility matrix
- [DEPRECATIONS.md](./DEPRECATIONS.md) - Active deprecation tracker
