// Package errors tests for structured error types.
package errors

import (
	"errors"
	"fmt"
	"testing"
)

// -----------------------------------------------------------------------------
// WeaverError Construction Tests
// -----------------------------------------------------------------------------

func TestNew(t *testing.T) {
	we := New("TEST_ERROR", CategoryConfig, "test message")

	if we.Code != "TEST_ERROR" {
		t.Errorf("expected Code 'TEST_ERROR', got %q", we.Code)
	}
	if we.Category != CategoryConfig {
		t.Errorf("expected Category CategoryConfig, got %v", we.Category)
	}
	if we.Message != "test message" {
		t.Errorf("expected Message 'test message', got %q", we.Message)
	}
	if we.Context == nil {
		t.Error("expected Context map to be initialized, got nil")
	}
	if we.Cause != nil {
		t.Errorf("expected Cause to be nil, got %v", we.Cause)
	}
	if we.Suggestions != nil {
		t.Errorf("expected Suggestions to be nil, got %v", we.Suggestions)
	}
}

func TestWeaverError_Error(t *testing.T) {
	tests := []struct {
		name     string
		setup    func() *WeaverError
		expected string
	}{
		{
			name: "without cause",
			setup: func() *WeaverError {
				return New("CONFIG_NOT_FOUND", CategoryConfig, "configuration file not found")
			},
			expected: "CONFIG_NOT_FOUND: configuration file not found",
		},
		{
			name: "with cause",
			setup: func() *WeaverError {
				return New("FILE_READ_ERROR", CategoryIO, "failed to read file").
					WithCause(fmt.Errorf("permission denied"))
			},
			expected: "FILE_READ_ERROR: failed to read file: permission denied",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			we := tt.setup()
			if got := we.Error(); got != tt.expected {
				t.Errorf("Error() = %q, want %q", got, tt.expected)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Builder Pattern Tests
// -----------------------------------------------------------------------------

func TestWithContext(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithContext("file", "/path/to/config.yaml").
		WithContext("line", "42")

	if we.Context["file"] != "/path/to/config.yaml" {
		t.Errorf("expected file context '/path/to/config.yaml', got %q", we.Context["file"])
	}
	if we.Context["line"] != "42" {
		t.Errorf("expected line context '42', got %q", we.Context["line"])
	}
}

func TestWithContextMap(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithContextMap(map[string]string{
			"file":   "/path/to/config.yaml",
			"line":   "42",
			"column": "10",
		})

	if len(we.Context) != 3 {
		t.Errorf("expected 3 context entries, got %d", len(we.Context))
	}
	if we.Context["file"] != "/path/to/config.yaml" {
		t.Errorf("expected file context '/path/to/config.yaml', got %q", we.Context["file"])
	}
}

func TestWithContext_NilMap(t *testing.T) {
	// Test that WithContext handles nil Context gracefully
	we := &WeaverError{
		Code:     "TEST",
		Category: CategoryConfig,
		Message:  "test",
		Context:  nil, // explicitly nil
	}
	we.WithContext("key", "value")

	if we.Context == nil {
		t.Error("expected Context to be initialized")
	}
	if we.Context["key"] != "value" {
		t.Errorf("expected key 'value', got %q", we.Context["key"])
	}
}

func TestWithContextMap_NilMap(t *testing.T) {
	we := &WeaverError{
		Code:     "TEST",
		Category: CategoryConfig,
		Message:  "test",
		Context:  nil,
	}
	we.WithContextMap(map[string]string{"key": "value"})

	if we.Context == nil {
		t.Error("expected Context to be initialized")
	}
}

func TestWithCause(t *testing.T) {
	cause := fmt.Errorf("underlying error")
	we := New("TEST", CategoryIO, "test").WithCause(cause)

	if we.Cause != cause {
		t.Errorf("expected Cause to be set, got %v", we.Cause)
	}
}

func TestWithSuggestion(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithSuggestion("Run 'weaver --init' to create a config file").
		WithSuggestion("Check if the config path is correct")

	if len(we.Suggestions) != 2 {
		t.Errorf("expected 2 suggestions, got %d", len(we.Suggestions))
	}
	if we.Suggestions[0] != "Run 'weaver --init' to create a config file" {
		t.Errorf("unexpected first suggestion: %q", we.Suggestions[0])
	}
}

func TestWithSuggestions(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithSuggestions(
			"First suggestion",
			"Second suggestion",
			"Third suggestion",
		)

	if len(we.Suggestions) != 3 {
		t.Errorf("expected 3 suggestions, got %d", len(we.Suggestions))
	}
}

func TestBuilderChaining(t *testing.T) {
	cause := fmt.Errorf("network timeout")
	we := New("BACKEND_UNAVAILABLE", CategoryBackend, "backend is not responding").
		WithContext("backend", "claudecode").
		WithContext("timeout", "30s").
		WithCause(cause).
		WithSuggestion("Check if Claude CLI is installed").
		WithSuggestion("Verify network connectivity")

	if we.Code != "BACKEND_UNAVAILABLE" {
		t.Error("Code not preserved after chaining")
	}
	if len(we.Context) != 2 {
		t.Errorf("expected 2 context entries, got %d", len(we.Context))
	}
	if we.Cause != cause {
		t.Error("Cause not preserved after chaining")
	}
	if len(we.Suggestions) != 2 {
		t.Errorf("expected 2 suggestions, got %d", len(we.Suggestions))
	}
}

// -----------------------------------------------------------------------------
// Unwrap and Error Chain Tests
// -----------------------------------------------------------------------------

func TestUnwrap(t *testing.T) {
	cause := fmt.Errorf("original error")
	we := New("TEST", CategoryIO, "wrapper").WithCause(cause)

	unwrapped := we.Unwrap()
	if unwrapped != cause {
		t.Errorf("Unwrap() = %v, want %v", unwrapped, cause)
	}
}

func TestUnwrap_NilCause(t *testing.T) {
	we := New("TEST", CategoryIO, "no cause")

	if we.Unwrap() != nil {
		t.Error("expected Unwrap() to return nil for error without cause")
	}
}

func TestIs_SameCode(t *testing.T) {
	err1 := New("CONFIG_NOT_FOUND", CategoryConfig, "config not found")
	err2 := New("CONFIG_NOT_FOUND", CategoryConfig, "different message")

	if !err1.Is(err2) {
		t.Error("expected Is() to return true for same error code")
	}
}

func TestIs_DifferentCode(t *testing.T) {
	err1 := New("CONFIG_NOT_FOUND", CategoryConfig, "config not found")
	err2 := New("CONFIG_PARSE_ERROR", CategoryConfig, "parse error")

	if err1.Is(err2) {
		t.Error("expected Is() to return false for different error codes")
	}
}

func TestIs_StandardError(t *testing.T) {
	we := New("TEST", CategoryConfig, "test")
	stdErr := fmt.Errorf("standard error")

	if we.Is(stdErr) {
		t.Error("expected Is() to return false for non-WeaverError")
	}
}

func TestErrorsIs_WithWrapping(t *testing.T) {
	target := New("FILE_NOT_FOUND", CategoryIO, "file not found")
	wrapped := New("CONFIG_LOAD_FAILED", CategoryConfig, "failed to load config").
		WithCause(target)

	if !errors.Is(wrapped, target) {
		t.Error("expected errors.Is() to find wrapped WeaverError")
	}
}

// -----------------------------------------------------------------------------
// Helper and Utility Function Tests
// -----------------------------------------------------------------------------

func TestHasContext(t *testing.T) {
	tests := []struct {
		name     string
		setup    func() *WeaverError
		expected bool
	}{
		{
			name: "with context",
			setup: func() *WeaverError {
				return New("TEST", CategoryConfig, "test").WithContext("key", "value")
			},
			expected: true,
		},
		{
			name: "without context",
			setup: func() *WeaverError {
				return New("TEST", CategoryConfig, "test")
			},
			expected: false,
		},
		{
			name: "empty context map",
			setup: func() *WeaverError {
				we := New("TEST", CategoryConfig, "test")
				we.Context = make(map[string]string)
				return we
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			we := tt.setup()
			if got := we.HasContext(); got != tt.expected {
				t.Errorf("HasContext() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestHasSuggestions(t *testing.T) {
	tests := []struct {
		name     string
		setup    func() *WeaverError
		expected bool
	}{
		{
			name: "with suggestions",
			setup: func() *WeaverError {
				return New("TEST", CategoryConfig, "test").WithSuggestion("fix it")
			},
			expected: true,
		},
		{
			name: "without suggestions",
			setup: func() *WeaverError {
				return New("TEST", CategoryConfig, "test")
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			we := tt.setup()
			if got := we.HasSuggestions(); got != tt.expected {
				t.Errorf("HasSuggestions() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestContextString(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithContext("file", "/path/to/config.yaml")

	ctxStr := we.ContextString()
	if ctxStr != `file="/path/to/config.yaml"` {
		t.Errorf("ContextString() = %q, expected file=\"/path/to/config.yaml\"", ctxStr)
	}
}

func TestContextString_Empty(t *testing.T) {
	we := New("TEST", CategoryConfig, "test")

	if we.ContextString() != "" {
		t.Error("expected empty string for no context")
	}
}

func TestWrap(t *testing.T) {
	cause := fmt.Errorf("original error")
	we := Wrap(cause, "WRAPPED_ERROR", CategoryIO, "wrapped message")

	if we.Cause != cause {
		t.Error("expected cause to be wrapped")
	}
	if we.Code != "WRAPPED_ERROR" {
		t.Errorf("expected code 'WRAPPED_ERROR', got %q", we.Code)
	}
	if we.Category != CategoryIO {
		t.Errorf("expected CategoryIO, got %v", we.Category)
	}
}

func TestAsWeaverError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantOK   bool
		wantCode string
	}{
		{
			name:     "WeaverError",
			err:      New("TEST", CategoryConfig, "test"),
			wantOK:   true,
			wantCode: "TEST",
		},
		{
			name:   "standard error",
			err:    fmt.Errorf("standard error"),
			wantOK: false,
		},
		{
			name:   "nil error",
			err:    nil,
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			we, ok := AsWeaverError(tt.err)
			if ok != tt.wantOK {
				t.Errorf("AsWeaverError() ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && we.Code != tt.wantCode {
				t.Errorf("AsWeaverError() code = %q, want %q", we.Code, tt.wantCode)
			}
		})
	}
}

func TestIsCategory(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		category Category
		expected bool
	}{
		{
			name:     "matching category",
			err:      New("TEST", CategoryConfig, "test"),
			category: CategoryConfig,
			expected: true,
		},
		{
			name:     "non-matching category",
			err:      New("TEST", CategoryConfig, "test"),
			category: CategoryAgent,
			expected: false,
		},
		{
			name:     "standard error",
			err:      fmt.Errorf("standard error"),
			category: CategoryConfig,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsCategory(tt.err, tt.category); got != tt.expected {
				t.Errorf("IsCategory() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestIsCode(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		code     string
		expected bool
	}{
		{
			name:     "matching code",
			err:      New("CONFIG_NOT_FOUND", CategoryConfig, "test"),
			code:     "CONFIG_NOT_FOUND",
			expected: true,
		},
		{
			name:     "non-matching code",
			err:      New("CONFIG_NOT_FOUND", CategoryConfig, "test"),
			code:     "CONFIG_PARSE_ERROR",
			expected: false,
		},
		{
			name:     "standard error",
			err:      fmt.Errorf("standard error"),
			code:     "ANY_CODE",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsCode(tt.err, tt.code); got != tt.expected {
				t.Errorf("IsCode() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Category-Specific Constructor Tests
// -----------------------------------------------------------------------------

func TestConfigError(t *testing.T) {
	we := ConfigError("CONFIG_NOT_FOUND", "configuration file not found")

	if we.Category != CategoryConfig {
		t.Errorf("expected CategoryConfig, got %v", we.Category)
	}
	if we.Code != "CONFIG_NOT_FOUND" {
		t.Errorf("expected code 'CONFIG_NOT_FOUND', got %q", we.Code)
	}
}

func TestConfigErrorf(t *testing.T) {
	we := ConfigErrorf("CONFIG_NOT_FOUND", "file %q not found", "/path/to/config.yaml")

	if we.Message != `file "/path/to/config.yaml" not found` {
		t.Errorf("unexpected message: %q", we.Message)
	}
	if we.Category != CategoryConfig {
		t.Errorf("expected CategoryConfig, got %v", we.Category)
	}
}

func TestAgentError(t *testing.T) {
	we := AgentError("AGENT_NOT_FOUND", "agent not found")
	if we.Category != CategoryAgent {
		t.Errorf("expected CategoryAgent, got %v", we.Category)
	}
}

func TestAgentErrorf(t *testing.T) {
	we := AgentErrorf("AGENT_NOT_FOUND", "agent %q not found", "myAgent")
	if we.Message != `agent "myAgent" not found` {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

func TestBackendError(t *testing.T) {
	we := BackendError("BACKEND_UNAVAILABLE", "backend unavailable")
	if we.Category != CategoryBackend {
		t.Errorf("expected CategoryBackend, got %v", we.Category)
	}
}

func TestBackendErrorf(t *testing.T) {
	we := BackendErrorf("BACKEND_TIMEOUT", "backend %q timed out after %ds", "claudecode", 30)
	if we.Message != `backend "claudecode" timed out after 30s` {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

func TestCommandError(t *testing.T) {
	we := CommandError("UNKNOWN_COMMAND", "unknown command")
	if we.Category != CategoryCommand {
		t.Errorf("expected CategoryCommand, got %v", we.Category)
	}
}

func TestCommandErrorf(t *testing.T) {
	we := CommandErrorf("UNKNOWN_COMMAND", "unknown command: %q", "/help")
	if we.Message != `unknown command: "/help"` {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

func TestValidationError(t *testing.T) {
	we := ValidationError("INVALID_INPUT", "invalid input")
	if we.Category != CategoryValidation {
		t.Errorf("expected CategoryValidation, got %v", we.Category)
	}
}

func TestValidationErrorf(t *testing.T) {
	we := ValidationErrorf("INVALID_RANGE", "value %d is out of range [%d, %d]", 150, 0, 100)
	if we.Message != "value 150 is out of range [0, 100]" {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

func TestNetworkError(t *testing.T) {
	we := NetworkError("CONNECTION_REFUSED", "connection refused")
	if we.Category != CategoryNetwork {
		t.Errorf("expected CategoryNetwork, got %v", we.Category)
	}
}

func TestNetworkErrorf(t *testing.T) {
	we := NetworkErrorf("DNS_LOOKUP_FAILED", "failed to resolve %q", "api.example.com")
	if we.Message != `failed to resolve "api.example.com"` {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

func TestIOError(t *testing.T) {
	we := IOError("FILE_NOT_FOUND", "file not found")
	if we.Category != CategoryIO {
		t.Errorf("expected CategoryIO, got %v", we.Category)
	}
}

func TestIOErrorf(t *testing.T) {
	we := IOErrorf("PERMISSION_DENIED", "cannot access %q: permission denied", "/etc/shadow")
	if we.Message != `cannot access "/etc/shadow": permission denied` {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

func TestInternalError(t *testing.T) {
	we := InternalError("UNEXPECTED_STATE", "unexpected state")
	if we.Category != CategoryInternal {
		t.Errorf("expected CategoryInternal, got %v", we.Category)
	}
}

func TestInternalErrorf(t *testing.T) {
	we := InternalErrorf("NIL_POINTER", "unexpected nil pointer in %s", "processRequest")
	if we.Message != "unexpected nil pointer in processRequest" {
		t.Errorf("unexpected message: %q", we.Message)
	}
}

// -----------------------------------------------------------------------------
// Category-Specific Wrap Helper Tests
// -----------------------------------------------------------------------------

func TestWrapConfig(t *testing.T) {
	cause := fmt.Errorf("yaml: unmarshal error")
	we := WrapConfig(cause, "CONFIG_PARSE_ERROR", "failed to parse config file")

	if we.Category != CategoryConfig {
		t.Errorf("expected CategoryConfig, got %v", we.Category)
	}
	if we.Cause != cause {
		t.Error("expected cause to be wrapped")
	}
}

func TestWrapAgent(t *testing.T) {
	cause := fmt.Errorf("agent init failed")
	we := WrapAgent(cause, "AGENT_INIT_ERROR", "failed to initialize agent")

	if we.Category != CategoryAgent {
		t.Errorf("expected CategoryAgent, got %v", we.Category)
	}
}

func TestWrapBackend(t *testing.T) {
	cause := fmt.Errorf("connection refused")
	we := WrapBackend(cause, "BACKEND_CONNECTION_ERROR", "failed to connect to backend")

	if we.Category != CategoryBackend {
		t.Errorf("expected CategoryBackend, got %v", we.Category)
	}
}

func TestWrapCommand(t *testing.T) {
	cause := fmt.Errorf("parse error")
	we := WrapCommand(cause, "COMMAND_PARSE_ERROR", "failed to parse command")

	if we.Category != CategoryCommand {
		t.Errorf("expected CategoryCommand, got %v", we.Category)
	}
}

func TestWrapValidation(t *testing.T) {
	cause := fmt.Errorf("schema violation")
	we := WrapValidation(cause, "SCHEMA_ERROR", "input validation failed")

	if we.Category != CategoryValidation {
		t.Errorf("expected CategoryValidation, got %v", we.Category)
	}
}

func TestWrapNetwork(t *testing.T) {
	cause := fmt.Errorf("timeout exceeded")
	we := WrapNetwork(cause, "TIMEOUT", "network operation timed out")

	if we.Category != CategoryNetwork {
		t.Errorf("expected CategoryNetwork, got %v", we.Category)
	}
}

func TestWrapIO(t *testing.T) {
	cause := fmt.Errorf("disk full")
	we := WrapIO(cause, "DISK_FULL", "failed to write file")

	if we.Category != CategoryIO {
		t.Errorf("expected CategoryIO, got %v", we.Category)
	}
}

func TestWrapInternal(t *testing.T) {
	cause := fmt.Errorf("invariant violated")
	we := WrapInternal(cause, "INVARIANT_ERROR", "internal error occurred")

	if we.Category != CategoryInternal {
		t.Errorf("expected CategoryInternal, got %v", we.Category)
	}
}

// -----------------------------------------------------------------------------
// Category Constant Tests
// -----------------------------------------------------------------------------

func TestCategories(t *testing.T) {
	// Verify all category constants are defined correctly
	categories := []struct {
		category Category
		expected string
	}{
		{CategoryConfig, "config"},
		{CategoryAgent, "agent"},
		{CategoryBackend, "backend"},
		{CategoryCommand, "command"},
		{CategoryValidation, "validation"},
		{CategoryNetwork, "network"},
		{CategoryIO, "io"},
		{CategoryInternal, "internal"},
	}

	for _, c := range categories {
		if string(c.category) != c.expected {
			t.Errorf("expected category %q, got %q", c.expected, string(c.category))
		}
	}
}
