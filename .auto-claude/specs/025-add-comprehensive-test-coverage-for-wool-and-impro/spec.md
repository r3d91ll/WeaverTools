# Specification: Comprehensive Test Coverage for Wool, Weaver Spinner, TheLoom, and E2E

## Overview

This task addresses four critical test coverage gaps identified in the bug hunt report (bughunt_main_261225.md). The work includes creating a new test suite for the Wool package (currently untested), fixing terminal detection issues in Weaver's spinner tests, separating GPU-dependent tests from unit tests in TheLoom to enable CI execution, and automating the manual E2E testing workflow into a comprehensive test harness.

## Workflow Type

**Type**: feature

**Rationale**: This task creates new test infrastructure and automation rather than fixing existing bugs or refactoring code. Each component (Wool tests, TheLoom test separation, E2E harness) represents net-new testing capabilities being added to the project.

## Task Scope

### Services Involved
- **Wool** (primary) - Create complete test suite from scratch
- **Weaver** (primary) - Fix 3 failing spinner tests
- **TheLoom** (primary) - Separate GPU tests from unit tests
- **All Services** (integration) - E2E test harness automation

### This Task Will:
- [x] Create comprehensive test suite for Wool package (agent definitions, roles, capabilities, validation)
- [x] Fix 3 failing Weaver spinner tests related to terminal color detection
- [x] Separate TheLoom GPU-dependent tests from unit tests to enable CI execution
- [x] Automate manual E2E testing workflow from bug hunt section 5
- [x] Achieve >80% test coverage for Wool package
- [x] Enable all non-GPU tests to run successfully in CI

### Out of Scope:
- Performance optimization of existing tests
- Adding new functionality to Wool, Weaver, or TheLoom packages
- GPU provisioning or CI infrastructure changes
- Refactoring existing test structures outside the 4 identified areas

## Service Context

### Wool (Go Package)

**Tech Stack:**
- Language: Go
- Framework: Standard library
- Key directories: `/Wool`

**Entry Point:** `Wool/agent.go`, `Wool/role.go`

**How to Run:**
```bash
cd Wool
go test -v ./...
go test -race -cover ./...
```

**Current State:** No test suite exists (0% coverage)

### Weaver (Go CLI)

**Tech Stack:**
- Language: Go
- Framework: Standard library
- Key directories: `/Weaver/pkg/spinner`

**How to Run:**
```bash
cd Weaver
go test -v ./pkg/spinner
go test -race ./...
```

**Current Test Status:** 3 failing tests in pkg/spinner

### TheLoom (Python/FastAPI)

**Tech Stack:**
- Language: Python
- Framework: FastAPI, pytest
- Key directories: `/TheLoom/the-loom/tests`

**How to Run:**
```bash
cd TheLoom/the-loom
poetry run pytest -v
poetry run pytest -v -m "not gpu"  # Non-GPU tests only
```

**Current Test Status:** 137 passing, 33 failures, 14 errors (GPU-related)

### E2E Testing (Cross-Service)

**Components Tested:**
- Weaver CLI orchestration
- TheLoom server integration
- Claude Code backend integration
- Agent delegation and measurement workflow

**Entry Point:** Manual steps from `bughunt_main_261225.md` section "Full-Stack Testing Plan"

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `Wool/agent_test.go` | Wool | Create new file - comprehensive agent validation tests |
| `Wool/role_test.go` | Wool | Create new file - role capability and state requirement tests |
| `Weaver/pkg/spinner/spinner_test.go` | Weaver | Fix color detection in TestSuccess, TestFail, TestRenderWithElapsedTime |
| `TheLoom/the-loom/tests/test_server.py` | TheLoom | Add GPU markers for GPU-dependent tests |
| `TheLoom/the-loom/tests/test_analysis.py` | TheLoom | Add GPU markers for geometry endpoint tests |
| `TheLoom/the-loom/tests/test_integration.py` | TheLoom | Add GPU markers or create mocks for integration tests |
| `TheLoom/the-loom/pyproject.toml` | TheLoom | Add pytest markers configuration for GPU tests |
| `tests/e2e/test_delegation.go` | E2E | Create new file - Senior/Junior delegation tests |
| `tests/e2e/test_measurement.go` | E2E | Create new file - Conveyance measurement workflow tests |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `Yarn/message_test.go` | Table-driven test pattern with subtests for Go validation testing |
| `Yarn/session_test.go` | Comprehensive validation and edge case testing in Go |
| `Weaver/pkg/spinner/spinner_test.go` | Terminal detection and output verification patterns |
| `TheLoom/the-loom/tests/test_server.py` | Pytest fixture pattern with mocks for server testing |
| `TheLoom/the-loom/tests/test_config.py` | Unit test patterns for configuration validation |
| `bughunt_main_261225.md` (lines 143-291) | Manual E2E testing steps to automate |

## Patterns to Follow

### Pattern 1: Table-Driven Go Tests

From `Yarn/message_test.go`:

```go
func TestMessageRoleIsValid(t *testing.T) {
    tests := []struct {
        name  string
        role  MessageRole
        valid bool
    }{
        {"system role is valid", RoleSystem, true},
        {"user role is valid", RoleUser, true},
        {"empty role is invalid", MessageRole(""), false},
        {"unknown role is invalid", MessageRole("unknown"), false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if got := tt.role.IsValid(); got != tt.valid {
                t.Errorf("MessageRole(%q).IsValid() = %v, want %v", tt.role, got, tt.valid)
            }
        })
    }
}
```

**Key Points:**
- Use table-driven tests with descriptive test case names
- Each test case is a subtest with t.Run()
- Test both positive and negative cases
- Use clear error messages with actual vs expected values

### Pattern 2: Validation Testing

From `Wool/agent.go` (lines 86-123):

**What to Test:**
- Required fields (name, role, backend)
- Valid backend values ("loom", "claudecode")
- Role-capability consistency (tools only for Senior/Junior)
- Inference parameter ranges (temperature 0-2.0, top_p 0-1.0)
- ValidationError structure and messages

### Pattern 3: Terminal Detection Abstraction

From `Weaver/pkg/spinner/spinner_test.go` (lines 251-279, 301-329):

**Issue:** Tests check for color codes but terminal detection varies by environment

**Solution:**
- Use explicit IsTTY configuration in test setup
- Test both TTY and non-TTY modes separately
- Verify behavior changes based on terminal capability
- Don't assume color code presence in all test environments

```go
// Force IsTTY mode for color tests
s := NewWithConfig(Config{
    Message: "test",
    Writer:  &buf,
    IsTTY:   boolPtr(true),  // Explicit TTY mode
})
```

### Pattern 4: Pytest GPU Markers

From TheLoom testing requirements:

```python
import pytest

@pytest.mark.gpu
def test_server_geometry_endpoint():
    """Test geometry analysis (requires GPU)."""
    # Test implementation
    pass

def test_config_validation():
    """Test configuration validation (no GPU needed)."""
    # Test implementation
    pass
```

**pyproject.toml configuration:**
```toml
[tool.pytest.ini_options]
markers = [
    "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')",
    "integration: marks tests as integration tests",
]
```

## Requirements

### Functional Requirements

#### 1. Wool Test Suite Creation

**Description:** Create comprehensive test coverage for Wool package covering all agent definitions, role configurations, and validation logic.

**Acceptance Criteria:**
- `agent_test.go` tests Agent struct validation, default constructors, and JSON/YAML serialization
- `role_test.go` tests all Role methods (IsValid, RequiresHiddenStates, SupportsTools, CanGenerateResponses, Description)
- Test coverage >80% measured by `go test -cover`
- All edge cases covered: empty fields, invalid values, boundary conditions
- Validation error messages tested for correctness

#### 2. Weaver Spinner Test Fixes

**Description:** Fix 3 failing spinner tests by properly handling terminal color detection across different environments.

**Acceptance Criteria:**
- `TestSuccess` passes reliably by explicitly setting IsTTY mode
- `TestFail` passes reliably by explicitly setting IsTTY mode
- `TestRenderWithElapsedTime` passes by verifying elapsed time format regardless of color codes
- All existing passing tests remain passing
- Tests work in both CI and local development environments

#### 3. TheLoom GPU Test Separation

**Description:** Separate GPU-dependent tests from unit tests to enable CI execution without GPU hardware.

**Acceptance Criteria:**
- All GPU-dependent tests marked with `@pytest.mark.gpu`
- `pytest -m "not gpu"` runs successfully without GPU hardware
- Server geometry endpoint tests (6 failures) properly marked
- Streaming SSE tests (9 failures) either mocked or marked as GPU
- Integration tests (14 errors) either mocked or marked as GPU
- pyproject.toml configured with pytest markers
- Documentation updated with GPU vs non-GPU test instructions

#### 4. E2E Test Harness Automation

**Description:** Automate the manual testing steps from bug hunt section 5 into a comprehensive automated test suite.

**Acceptance Criteria:**
- Senior/Junior delegation workflow automated
- Hidden state extraction validation automated
- Conveyance measurement calculation tests implemented
- Tests can run without manual intervention
- Tests verify actual backend integration (not just mocks)
- Test output includes measurement data validation (D_eff, β calculations)

### Edge Cases

1. **Wool Validation Edge Cases**
   - Empty/nil values for required fields
   - Invalid backend names (typos, case sensitivity)
   - Tools enabled for roles that don't support tools (Conversant, Subject, Observer)
   - Out-of-range inference parameters (negative temperature, top_p > 1.0)

2. **Spinner Terminal Detection Edge Cases**
   - Running in CI environment without TTY
   - Running in IDE test runner
   - Running in terminal with color support disabled
   - Concurrent spinner operations with different TTY states

3. **TheLoom GPU Detection Edge Cases**
   - GPU available but CUDA not initialized
   - Multiple GPUs with different capabilities
   - GPU memory exhausted during test
   - Mock model loading for unit tests

4. **E2E Integration Edge Cases**
   - TheLoom server not responding
   - Claude CLI not authenticated
   - Network timeouts during streaming
   - Measurement data missing or malformed

## Implementation Notes

### DO

**Wool Tests:**
- Follow the table-driven test pattern from `Yarn/message_test.go`
- Test all public methods on Agent and Role structs
- Test JSON/YAML marshaling/unmarshaling (use Agent as Config type)
- Verify ValidationError contains correct Field and Message values
- Test all default constructors: DefaultSenior(), DefaultJunior(), DefaultSubject()

**Weaver Spinner Fixes:**
- Use `IsTTY: boolPtr(true)` in test config to force TTY mode for color tests
- Create separate test cases for TTY vs non-TTY behavior
- Don't rely on automatic terminal detection in tests
- Keep all existing test structure and coverage

**TheLoom GPU Separation:**
- Add `import pytest` to test files that need GPU markers
- Mark entire test classes with `@pytest.mark.gpu` if all methods need GPU
- Create mock fixtures for server geometry endpoints to test API contract without GPU
- Add clear comments explaining why each test needs GPU
- Update CI documentation with `pytest -m "not gpu"` command

**E2E Test Harness:**
- Use Go's testing package for E2E orchestration
- Start TheLoom server as subprocess in test setup
- Verify health endpoint before running tests
- Clean up processes in test teardown (use `defer`)
- Test both success and failure paths

### DON'T

- Don't modify Wool package source code (only add tests)
- Don't change spinner behavior (only fix tests to handle terminal detection)
- Don't remove GPU tests from TheLoom (only mark them appropriately)
- Don't skip E2E tests in CI if they can run without GPU
- Don't create new testing frameworks (use existing: Go testing, pytest)

## Development Environment

### Start Services

**For Wool and Weaver Testing:**
```bash
cd Wool
go test -v -race -cover ./...

cd ../Weaver
go test -v -race ./pkg/spinner
```

**For TheLoom Testing:**
```bash
cd TheLoom/the-loom
poetry install
poetry run pytest -v                    # All tests (requires GPU)
poetry run pytest -v -m "not gpu"       # Unit tests only (no GPU needed)
```

**For E2E Testing:**
```bash
# Start TheLoom server
cd TheLoom/the-loom
poetry run loom --port 8080 &

# Verify health
curl http://localhost:8080/health

# Run E2E tests
cd ../../tests/e2e
go test -v ./...
```

### Service URLs
- TheLoom: http://localhost:8080
- Weaver CLI: Local binary execution

### Required Environment Variables
- `CLAUDE_API_KEY`: For Claude Code backend (E2E tests only)
- `CUDA_VISIBLE_DEVICES`: GPU selection for TheLoom (optional, defaults to auto)

## Success Criteria

The task is complete when:

1. [x] Wool package test coverage >80% (measured by `go test -cover`)
2. [x] All Weaver spinner tests passing (`go test ./pkg/spinner` exits 0)
3. [x] TheLoom non-GPU tests run successfully (`pytest -m "not gpu"` exits 0)
4. [x] E2E test harness runs without manual intervention
5. [x] No console errors during test execution
6. [x] All existing passing tests remain passing (Yarn 321/321)
7. [x] CI can run all non-GPU tests successfully

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests

#### Wool Package Tests
| Test | File | What to Verify |
|------|------|----------------|
| `TestAgentValidate_*` | `Wool/agent_test.go` | All validation rules enforced correctly |
| `TestDefaultSenior` | `Wool/agent_test.go` | Default Senior agent has correct role, backend="claudecode", tools enabled |
| `TestDefaultJunior` | `Wool/agent_test.go` | Default Junior agent has correct role, backend="loom", tools enabled |
| `TestDefaultSubject` | `Wool/agent_test.go` | Default Subject agent has correct role, tools disabled |
| `TestAgentValidate_InvalidBackend` | `Wool/agent_test.go` | Rejects backends other than "loom" or "claudecode" |
| `TestAgentValidate_ToolsEnabled` | `Wool/agent_test.go` | Rejects tools_enabled=true for roles that don't support tools |
| `TestAgentValidate_TemperatureRange` | `Wool/agent_test.go` | Rejects temperature < 0 or > 2.0 |
| `TestAgentValidate_TopPRange` | `Wool/agent_test.go` | Rejects top_p < 0 or > 1.0 |
| `TestRole_IsValid_*` | `Wool/role_test.go` | All valid roles return true, invalid roles return false |
| `TestRole_RequiresHiddenStates` | `Wool/role_test.go` | Conversant, Subject return true; Senior returns false; Junior returns true (optional) |
| `TestRole_SupportsTools` | `Wool/role_test.go` | Senior, Junior return true; Conversant, Subject, Observer return false |
| `TestRole_CanGenerateResponses` | `Wool/role_test.go` | All roles except Observer return true |
| `TestRole_Description` | `Wool/role_test.go` | Each role returns appropriate human-readable description |

#### Weaver Spinner Tests
| Test | File | What to Verify |
|------|------|----------------|
| `TestSuccess` | `Weaver/pkg/spinner/spinner_test.go` | Success displays green color in TTY mode, no color in non-TTY |
| `TestFail` | `Weaver/pkg/spinner/spinner_test.go` | Fail displays red color in TTY mode, no color in non-TTY |
| `TestRenderWithElapsedTime` | `Weaver/pkg/spinner/spinner_test.go` | Elapsed time displayed in format "(X.Xs)" or "(Xm Ys)" |

#### TheLoom Unit Tests (Non-GPU)
| Test | File | What to Verify |
|------|------|----------------|
| `test_config_validation` | `tests/test_config.py` | Config validation works without GPU |
| `test_serialization` | `tests/test_serialization.py` | Data structures serialize correctly |
| `test_client_*` (non-GPU) | `tests/test_client.py` | Client logic testable with mocks |

### Integration Tests

#### TheLoom GPU Tests (Marked)
| Test | Services | What to Verify |
|------|----------|----------------|
| `test_server_geometry_*` | TheLoom + GPU | Marked with `@pytest.mark.gpu`, skipped in `pytest -m "not gpu"` |
| `test_streaming_sse_*` | TheLoom + GPU | Marked with `@pytest.mark.gpu`, skipped in `pytest -m "not gpu"` |
| `test_integration_*` | TheLoom + GPU | Marked with `@pytest.mark.gpu`, skipped in `pytest -m "not gpu"` |

### End-to-End Tests

#### E2E Test Harness
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Senior Response | 1. Start Weaver 2. Send message to senior 3. Verify response | Claude Code subprocess responds successfully |
| Junior Delegation | 1. Start TheLoom server 2. Start Weaver 3. Send @junior message 4. Verify response + hidden states | TheLoom responds with hidden state data |
| Session Export | 1. Complete conversation 2. Run /export command 3. Verify file | Session JSON exported to experiments/ directory |
| Measurement Workflow | 1. Configure conversant agents 2. Run bilateral exchange 3. Verify measurements | D_eff and β values populated in session export |

### Browser Verification (if frontend)
N/A - No browser components in this task

### Database Verification (if applicable)
N/A - No database changes in this task

### QA Sign-off Requirements
- [x] All Wool unit tests pass (321+ tests)
- [x] All Weaver spinner tests pass (previously 3 failures, now 0)
- [x] TheLoom non-GPU tests pass (`pytest -m "not gpu"`)
- [x] TheLoom GPU tests properly marked and skippable
- [x] E2E test harness executes without manual intervention
- [x] No regressions in existing functionality (Yarn 321/321 still passing)
- [x] Code follows established test patterns (table-driven tests, pytest fixtures)
- [x] Test coverage for Wool package >80%
- [x] No security vulnerabilities introduced
- [x] CI can execute all non-GPU tests successfully

### Coverage Verification

**Run these commands to verify coverage:**

```bash
# Wool coverage (must be >80%)
cd Wool
go test -cover ./... | grep coverage

# Weaver spinner tests (must be 0 failures)
cd ../Weaver
go test -v ./pkg/spinner | grep FAIL

# TheLoom non-GPU tests (must pass)
cd ../TheLoom/the-loom
poetry run pytest -v -m "not gpu" --tb=short

# E2E tests (must run without manual intervention)
cd ../../tests/e2e
go test -v ./...

# Yarn regression check (must still be 321/321)
cd ../../Yarn
go test ./... | grep -E "PASS|FAIL"
```

### Test Execution Time Expectations

- Wool tests: <5 seconds
- Weaver spinner tests: <10 seconds (includes timing-sensitive tests)
- TheLoom non-GPU tests: <30 seconds
- E2E tests: <2 minutes (includes server startup)
- Total test suite (non-GPU): <3 minutes

### Known Test Environment Issues

1. **Spinner color tests:** Require explicit TTY configuration to pass in all environments
2. **TheLoom GPU tests:** Cannot run in standard CI - must be marked and skipped
3. **E2E tests:** Require Claude CLI authentication - may need to be skipped in some CI environments
4. **Timing-sensitive tests:** Spinner tests include sleep statements - may need adjustment for slow CI runners

## Reference Documentation

### Key Source Files
- `Wool/agent.go` - Agent struct definition and validation logic
- `Wool/role.go` - Role type and capability methods
- `Weaver/pkg/spinner/spinner_test.go` - Existing spinner tests (some failing)
- `bughunt_main_261225.md` - Complete bug hunt report with test status and manual E2E steps
- `Yarn/message_test.go` - Reference for Go table-driven test pattern
- `TheLoom/the-loom/tests/test_server.py` - Reference for pytest patterns

### Testing Command Reference

```bash
# Run specific test file
go test -v -run TestAgentValidate ./Wool

# Run with race detection
go test -race ./...

# Run with coverage report
go test -cover -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run specific pytest markers
poetry run pytest -v -m gpu          # Only GPU tests
poetry run pytest -v -m "not gpu"    # Only non-GPU tests
poetry run pytest -v -k "test_config"  # Only tests matching pattern

# Run with verbose output
go test -v ./...
poetry run pytest -vv
```

### Bug Hunt Reference Sections

- **Lines 87-123**: Weaver spinner test failures (3 tests)
- **Lines 124-139**: TheLoom test status (47 failures)
- **Lines 143-291**: Full-stack testing plan (manual E2E steps to automate)
- **Lines 293-324**: Full test checklist
- **Lines 360-385**: Known issues and priorities
