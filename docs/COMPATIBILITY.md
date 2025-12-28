# WeaverTools Compatibility Matrix

This document tracks version compatibility across WeaverTools components. Use this matrix to determine which versions of Yarn, Wool, Weaver, and TheLoom work together.

## Table of Contents

- [Compatibility Matrix](#compatibility-matrix)
- [Legend](#legend)
- [Version Notes](#version-notes)
- [TheLoom (Python) Versioning](#theloom-python-versioning)
- [Cross-Version Compatibility Guidelines](#cross-version-compatibility-guidelines)
- [Upgrade Recommendations](#upgrade-recommendations)
- [Reporting Compatibility Issues](#reporting-compatibility-issues)

## Compatibility Matrix

### Go Services (Yarn, Wool, Weaver)

| Version | Yarn | Wool | Weaver | Release Date | Status |
|---------|------|------|--------|--------------|--------|
| v1.0.0  | ✓    | ✓    | ✓      | 2024-12-28   | Current Stable |

### Go Services + TheLoom (Python)

| Go Version | TheLoom Version | Status | Notes |
|------------|-----------------|--------|-------|
| v1.0.0     | 0.2.0           | ✓      | Initial stable pairing |

## Legend

| Symbol | Meaning | Description |
|--------|---------|-------------|
| ✓      | Fully Compatible | All features work correctly; recommended for production |
| ⚠️      | Partial Compatibility | Core features work, but some edge cases may have issues; see notes |
| ✗      | Breaking Changes | Incompatible versions; do not use together |
| —      | Not Tested | Compatibility not verified; use at your own risk |

## Version Notes

### v1.0.0 (Current)

**Release Date:** 2024-12-28

**Go Modules:**
- `github.com/r3d91ll/yarn` v1.0.0
- `github.com/r3d91ll/wool` v1.0.0
- `github.com/r3d91ll/weaver` v1.0.0

**Compatibility:**
- All three Go modules are fully compatible with each other
- Tested with Go 1.23.4
- TheLoom 0.2.0 integration verified

**Known Issues:**
- None reported

## TheLoom (Python) Versioning

TheLoom is the Python model server component of WeaverTools. It uses **independent versioning** from the Go services due to fundamental differences in the Python ecosystem.

### Why Independent Versioning?

| Aspect | Go Services | TheLoom (Python) |
|--------|-------------|------------------|
| Package Manager | go mod | Poetry (pyproject.toml) |
| Version Format | vMAJOR.MINOR.PATCH (Git tags) | MAJOR.MINOR.PATCH (pyproject.toml) |
| Release Cadence | Synchronized across Yarn/Wool/Weaver | Independent release cycle |
| Ecosystem | Go modules | PyPI |

### Current TheLoom Version

| Version | Go Compatibility | Notes |
|---------|------------------|-------|
| 0.2.0   | v1.0.0          | Current development version; pre-1.0 API may change |

### TheLoom Versioning Strategy

TheLoom follows [PEP 440](https://peps.python.org/pep-0440/) versioning via Poetry:

- **0.x.x versions**: Development phase; breaking changes may occur in minor versions
- **1.0.0+**: Will follow strict semver with 6-month deprecation policy
- **Future synchronization**: Cross-language version coordination planned for post-1.0

### Using TheLoom with Go Services

To ensure compatibility:

```python
# In your requirements or pyproject.toml
theloom>=0.2.0,<0.3.0  # Pin to minor version during 0.x development
```

Check this compatibility matrix before upgrading TheLoom to verify compatibility with your Go service versions.

## Cross-Version Compatibility Guidelines

### Guaranteed Compatibility

Within the same **major version** of Go services:

| If you use... | Compatible with... |
|---------------|-------------------|
| Yarn v1.x.x   | Wool v1.x.x, Weaver v1.x.x |
| Yarn v2.x.x   | Wool v2.x.x, Weaver v2.x.x |

### Mixed Major Versions

Mixing major versions is **not recommended**:

```go
// NOT RECOMMENDED - may cause subtle incompatibilities
require (
    github.com/r3d91ll/yarn v1.0.0    // v1.x
    github.com/r3d91ll/wool v2.0.0    // v2.x - potential issues!
    github.com/r3d91ll/weaver v1.0.0  // v1.x
)

// RECOMMENDED - keep all services on same major version
require (
    github.com/r3d91ll/yarn v2.0.0
    github.com/r3d91ll/wool v2.0.0
    github.com/r3d91ll/weaver v2.0.0
)
```

### Minor Version Flexibility

Minor version differences within the same major version are safe:

```go
// SAFE - minor versions are backward compatible
require (
    github.com/r3d91ll/yarn v1.0.0    // Older minor version
    github.com/r3d91ll/wool v1.2.0    // Newer minor version
    github.com/r3d91ll/weaver v1.1.0  // Different minor version
)
```

## Upgrade Recommendations

### Upgrading Go Services

1. **Check this matrix** for compatibility with your TheLoom version
2. **Read the [CHANGELOG](../CHANGELOG.md)** for breaking changes
3. **Update all Go services together** to the same major version
4. **Run tests** before deploying to production

### Upgrading TheLoom

1. **Check the TheLoom compatibility row** in the matrix above
2. **Review TheLoom release notes** for Python API changes
3. **Test integration** with your Go service configuration
4. **Pin to minor version** during 0.x development phase

### Upgrade Command Reference

```bash
# Update Go services to specific version
go get github.com/r3d91ll/yarn@v1.1.0
go get github.com/r3d91ll/wool@v1.1.0
go get github.com/r3d91ll/weaver@v1.1.0

# Update TheLoom (via Poetry)
cd TheLoom
poetry update theloom

# Or with pip
pip install --upgrade theloom>=0.2.0,<0.3.0
```

## Reporting Compatibility Issues

If you encounter compatibility issues not documented in this matrix:

1. **Check existing issues**: [GitHub Issues](https://github.com/r3d91ll/WeaverTools/issues)
2. **Create a new issue** with:
   - Versions of all WeaverTools components
   - Go/Python versions
   - Error messages or unexpected behavior
   - Minimal reproduction steps
3. **Label the issue** with `compatibility`

### Compatibility Issue Template

```markdown
## Compatibility Issue

**Component Versions:**
- Yarn: v1.x.x
- Wool: v1.x.x
- Weaver: v1.x.x
- TheLoom: 0.x.x

**Environment:**
- Go version: 1.23.x
- Python version: 3.x.x
- OS: Linux/macOS/Windows

**Issue:**
[Description of incompatibility]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Reproduction Steps:**
1. ...
2. ...
```

## Related Documents

- [VERSIONING.md](./VERSIONING.md) - Versioning policy and deprecation process
- [CHANGELOG.md](../CHANGELOG.md) - Version history and release notes
- [DEPRECATIONS.md](./DEPRECATIONS.md) - Active deprecation tracker
