# golangci-lint Analysis Report

**Date:** 2025-12-26
**Status:** Manual analysis (sandbox environment restricts running golangci-lint, make, and go commands)

## Instructions for Running golangci-lint

Since the sandbox environment cannot run golangci-lint directly, run the following commands in a terminal with access to the repository:

```bash
# Ensure golangci-lint is installed
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run lint on all modules
cd Weaver && golangci-lint run ./... 2>&1 | tee ../lint-weaver.txt
cd ../Wool && golangci-lint run ./... 2>&1 | tee ../lint-wool.txt
cd ../Yarn && golangci-lint run ./... 2>&1 | tee ../lint-yarn.txt

# Or use the Makefile
make lint
```

---

## Manual Code Analysis

Based on the .golangci.yml configuration and manual code review, here are potential issues categorized by linter:

### 1. gofmt / goimports / gci (Style - Import Ordering)

**Files likely to have import ordering issues:**

| File | Issue | Category |
|------|-------|----------|
| Weaver/cmd/weaver/main.go | Mixed stdlib/3rd-party/local imports | Style preference |
| Weaver/pkg/backend/claudecode.go | Mixed import grouping | Style preference |
| Weaver/pkg/shell/shell.go | Many imports, may need reordering | Style preference |

**Recommendation:** Let `make lint-fix` auto-fix these. Import ordering is purely style.

### 2. gosec (Security)

**Potential security issues to review:**

| File | Line | Issue | Category |
|------|------|-------|----------|
| Weaver/pkg/backend/claudecode.go | 128 | Uses `--dangerously-skip-permissions` | Real concern - but documented as intentional |
| Weaver/pkg/backend/claudecode.go | 57 | `exec.CommandContext` with "claude" | May trigger G204 (subprocess call) |
| Weaver/pkg/config/config.go | 165 | `os.MkdirAll(dir, 0755)` | G301 - directory permissions |
| Weaver/pkg/config/config.go | 174 | `os.WriteFile(path, data, 0644)` | G306 - file permissions |
| Yarn/session.go | 152 | `os.MkdirAll(exportDir, 0755)` | G301 - directory permissions |
| Yarn/session.go | 162 | `os.WriteFile(sessionFile, sessionData, 0644)` | G306 - file permissions |
| Weaver/pkg/concepts/store.go | 215 | `os.MkdirAll(dir, 0755)` | G301 - directory permissions |
| Weaver/pkg/concepts/store.go | 221 | `os.WriteFile(path, data, 0644)` | G306 - file permissions |

**Recommendation:**
- The file permission warnings (0755/0644) are expected patterns for Go. Consider excluding G301/G306 if too noisy.
- The `exec.CommandContext` calls should be reviewed but are intentional for CLI wrapping.
- The `--dangerously-skip-permissions` flag has an inline comment explaining its necessity.

### 3. govet (Suspicious Constructs)

**No issues detected in manual review:**
- Error checking patterns look correct
- No obvious struct tag issues
- No suspicious bool/string comparisons
- Proper use of sync.Mutex patterns

### 4. errcheck (Unchecked Errors)

**Potential unchecked errors (covered by exclusions):**

| File | Line | Pattern | Status |
|------|------|---------|--------|
| Weaver/cmd/weaver/main.go | 217 | `homeDir, _ := os.UserHomeDir()` | May trigger - error ignored |
| Weaver/pkg/backend/claudecode.go | 158 | `stdin.Close()` after defer | Covered by exclusion |
| Yarn/session.go | 183-186 | `f.Write/WriteString` | Covered by exclusion (bytes.Buffer pattern) |

**Recommendation:**
- The `os.UserHomeDir()` error in main.go should probably be handled gracefully rather than ignored.
- Most other patterns are covered by the errcheck exclusions in .golangci.yml.

### 5. revive (Best Practices)

**Potential issues based on enabled rules:**

| Rule | Files | Issue | Category |
|------|-------|-------|----------|
| exported | All | Missing package comments on some packages | Style |
| dot-imports | None | No dot imports found | OK |
| error-strings | All | Error strings appear correct (lowercase, no punctuation) | OK |
| context-as-argument | All | Context is first param where used | OK |
| receiver-naming | All | Consistent single-letter receivers | OK |

**Recommendation:**
- `package-comments` is disabled in config (intentionally) - enable later when docs are added
- Current code follows revive conventions well

### 6. gocritic (Style/Performance)

**Potential issues:**

| File | Issue | Category |
|------|-------|----------|
| Wool/agent.go L98-104 | Loop could use `slices.Contains` (performance tag) | Style preference |
| Weaver/pkg/shell/shell.go | Long switch statement - could use map dispatch | Style preference |

**Recommendation:** These are minor style preferences, not correctness issues.

### 7. staticcheck (Static Analysis)

**No issues detected in manual review:**
- No deprecated function usage observed
- No obvious unreachable code
- No ineffectual assignments visible

### 8. unused / ineffassign

**Potential issues:**

| File | Line | Variable | Issue |
|------|------|----------|-------|
| Weaver/pkg/config/config.go | 13 | `float64Ptr` | May be marked unused if not called |

**Recommendation:** Verify with actual lint run. The function appears to be used in Default() config.

### 9. typecheck

**Status:** All code appears to compile (no type errors detected in manual review)

---

## Summary by Category

### Real Problems (Should Fix)
1. **main.go:217** - Ignored error from `os.UserHomeDir()` should be handled
2. **Security flags** - The `--dangerously-skip-permissions` usage is intentional but worth documenting prominently

### Style Preferences (Can Auto-Fix)
1. **Import ordering** - All files likely need gci/goimports ordering
2. **Minor gocritic suggestions** - Performance optimizations

### False Positives Expected
1. **gosec G301/G306** - Standard Go file permissions (0755/0644) are fine
2. **gosec G204** - Subprocess execution is intentional for CLI wrapping
3. **errcheck on fmt.Print*** - Already excluded in config
4. **errcheck on Close()** - Already excluded in config

---

## Recommended Configuration Adjustments

Based on this analysis, consider these adjustments to `.golangci.yml` for subtask 3.2:

1. **Add gosec exclusions for file permission rules if too noisy:**
   ```yaml
   gosec:
     excludes:
       - G301  # Dir permissions
       - G306  # File permissions
   ```

2. **Verify errcheck exclusions cover all intentional patterns**

3. **Consider enabling more revive rules after initial cleanup:**
   - `package-comments` when docs are added
   - `unused-parameter` after interface review

---

## Next Steps

1. Run `make lint` in an environment with full tool access
2. Capture actual output and update this analysis
3. Adjust .golangci.yml based on real findings (subtask 3.2)
4. Fix any real issues identified
5. Run `make lint-fix` to auto-fix style issues
