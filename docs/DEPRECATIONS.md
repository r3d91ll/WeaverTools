# WeaverTools Deprecation Tracker

This document maintains an active list of all deprecated APIs in WeaverTools. It serves as the central reference for planned API removals, ensuring researchers have adequate notice to migrate their code.

## Table of Contents

- [Active Deprecations](#active-deprecations)
- [Quarterly Audit Process](#quarterly-audit-process)
- [Audit Schedule](#audit-schedule)
- [Past Removals](#past-removals)
- [Requesting Deprecation Extensions](#requesting-deprecation-extensions)

## Active Deprecations

The following APIs are currently deprecated and scheduled for removal. All deprecated APIs maintain a **minimum 6-month notice period** before removal.

| API | Service | Version Deprecated | Removal Version | Removal Date | Migration Guide |
|-----|---------|-------------------|-----------------|--------------|-----------------|
| `FormatMessage()` | Yarn | v1.5.0 | v2.0.0 | 2026-06-01 | [See FormatMessageWithOptions](#formatmessage-migration) |

### Deprecation Details

#### FormatMessage Migration

**Status**: Deprecated as of v1.5.0

**Replacement**: `FormatMessageWithOptions()`

**Reason for Deprecation**: The original `FormatMessage()` function lacks flexibility for advanced formatting needs. The new `FormatMessageWithOptions()` provides configurable whitespace trimming, Unicode normalization, and length limits.

**Before (Deprecated)**:
```go
import "github.com/r3d91ll/yarn"

result := yarn.FormatMessage(content)
```

**After (Recommended)**:
```go
import "github.com/r3d91ll/yarn"

// With default options (equivalent to FormatMessage)
result := yarn.FormatMessageWithOptions(content, nil)

// With custom options
opts := &yarn.FormatOptions{
    TrimWhitespace:   true,
    NormalizeUnicode: true,
    MaxLength:        1000,
}
result := yarn.FormatMessageWithOptions(content, opts)
```

**Timeline**:
- December 2024: Deprecated in v1.5.0
- June 2026: Eligible for removal
- v2.0.0: Will be removed in next major version

## Quarterly Audit Process

To prevent forgotten deprecations and ensure timely API evolution, WeaverTools conducts quarterly deprecation audits.

### Audit Objectives

1. **Review all active deprecations** for upcoming removal eligibility
2. **Identify APIs approaching removal date** (within 3 months)
3. **Update documentation** to reflect any changes
4. **Communicate with users** about imminent removals
5. **Plan major version releases** that include removals

### Audit Checklist

Each quarterly audit should complete the following:

- [ ] Review each entry in the Active Deprecations table
- [ ] Check if any deprecations have passed their 6-month grace period
- [ ] Verify replacement APIs are documented and working
- [ ] Confirm migration guides are complete and accurate
- [ ] Update CHANGELOG.md with deprecation status changes
- [ ] Plan timing for next MAJOR release if removals are pending
- [ ] Send notification to mailing list/GitHub Discussions for imminent removals
- [ ] Archive removed deprecations to "Past Removals" section

### Audit Actions by Timeline

| Days Until Removal | Action Required |
|-------------------|-----------------|
| 90+ days | Monitor only - no action needed |
| 60-90 days | Draft removal announcement; verify migration guides |
| 30-60 days | Send user notification; finalize MAJOR version plan |
| 0-30 days | Prepare MAJOR release; final migration support |
| Past due | Schedule removal in next MAJOR release |

### Audit Report Template

```markdown
## Q[N] [YEAR] Deprecation Audit Report

**Audit Date**: YYYY-MM-DD
**Auditor**: [Name]

### Summary
- Total active deprecations: [N]
- Deprecations ready for removal: [N]
- New deprecations this quarter: [N]

### Deprecations Reviewed

| API | Status | Action Taken |
|-----|--------|--------------|
| [API Name] | Ready for removal / Still in grace period | [Action] |

### Upcoming Removals (Next Quarter)
- [List any APIs that will be eligible for removal]

### Notes
- [Any special considerations or user feedback]
```

## Audit Schedule

| Quarter | Audit Window | Responsible Team |
|---------|-------------|------------------|
| Q1 | January 1-15 | Core maintainers |
| Q2 | April 1-15 | Core maintainers |
| Q3 | July 1-15 | Core maintainers |
| Q4 | October 1-15 | Core maintainers |

### Next Scheduled Audit

**Q1 2025**: January 1-15, 2025

## Past Removals

APIs that have completed the deprecation lifecycle and been removed.

| API | Service | Version Removed | Removal Date | Replacement |
|-----|---------|----------------|--------------|-------------|
| *No removals yet* | - | - | - | - |

*This table will be populated as APIs complete the deprecation lifecycle and are removed in major version releases.*

## Requesting Deprecation Extensions

In exceptional cases, the deprecation timeline may be extended. Valid reasons include:

1. **Critical dependency**: The deprecated API is used by a widely-adopted package
2. **Complex migration**: The migration path requires significant refactoring
3. **Active research**: Ongoing research projects cannot migrate mid-experiment

### Extension Request Process

1. Open a GitHub issue with the `deprecation-extension` label
2. Provide:
   - API(s) requiring extension
   - Current research/project using the API
   - Estimated migration completion date
   - Justification for extension
3. Core team reviews within 14 days
4. If approved, removal date updated in this document

### Extension Limits

- Maximum extension: 6 additional months (12 months total from deprecation)
- Extensions do not delay MAJOR version releases
- Deprecated APIs may be removed from main release but maintained in LTS branch

## Adding New Deprecations

When deprecating an API, maintainers must:

1. Add the `// Deprecated:` godoc comment (see [VERSIONING.md](./VERSIONING.md#godoc-deprecation-format))
2. Add an entry to the Active Deprecations table above
3. Create a migration section with before/after examples
4. Update CHANGELOG.md "Deprecated" section
5. Commit with message: `deprecate: [API name] - [brief reason]`

### Entry Template

```markdown
| `FunctionName()` | Service | vX.Y.0 | vX+1.0.0 | YYYY-MM-DD | [See migration](#function-migration) |
```

## Related Documents

- [VERSIONING.md](./VERSIONING.md) - Versioning policy and deprecation format
- [COMPATIBILITY.md](./COMPATIBILITY.md) - Cross-version compatibility matrix
- [CHANGELOG.md](../CHANGELOG.md) - Version history with deprecated/removed sections
