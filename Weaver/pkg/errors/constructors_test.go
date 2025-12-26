package errors

import (
	"errors"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Basic Constructor Tests
// -----------------------------------------------------------------------------

func TestConfig(t *testing.T) {
	err := Config(ErrConfigNotFound, "config file missing")

	if err.Code != ErrConfigNotFound {
		t.Errorf("Code = %q, want %q", err.Code, ErrConfigNotFound)
	}
	if err.Category != CategoryConfig {
		t.Errorf("Category = %q, want %q", err.Category, CategoryConfig)
	}
	if err.Message != "config file missing" {
		t.Errorf("Message = %q, want %q", err.Message, "config file missing")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestConfigf(t *testing.T) {
	err := Configf(ErrConfigParseFailed, "failed to parse %s", "config.yaml")

	if err.Code != ErrConfigParseFailed {
		t.Errorf("Code = %q, want %q", err.Code, ErrConfigParseFailed)
	}
	if err.Message != "failed to parse config.yaml" {
		t.Errorf("Message = %q, want %q", err.Message, "failed to parse config.yaml")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestConfigWrap(t *testing.T) {
	cause := errors.New("yaml: invalid syntax")
	err := ConfigWrap(cause, ErrConfigParseFailed, "configuration error")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
	if !errors.Is(err, cause) {
		t.Error("Expected errors.Is to match cause")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestConfigWrapf(t *testing.T) {
	cause := errors.New("yaml: invalid syntax")
	err := ConfigWrapf(cause, ErrConfigParseFailed, "failed at line %d", 42)

	if err.Message != "failed at line 42" {
		t.Errorf("Message = %q, want %q", err.Message, "failed at line 42")
	}
	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

func TestBackend(t *testing.T) {
	err := Backend(ErrBackendUnavailable, "all backends down")

	if err.Code != ErrBackendUnavailable {
		t.Errorf("Code = %q, want %q", err.Code, ErrBackendUnavailable)
	}
	if err.Category != CategoryBackend {
		t.Errorf("Category = %q, want %q", err.Category, CategoryBackend)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestBackendf(t *testing.T) {
	err := Backendf(ErrBackendNotFound, "backend %q not registered", "loom")

	if err.Message != `backend "loom" not registered` {
		t.Errorf("Message = %q, want %q", err.Message, `backend "loom" not registered`)
	}
}

func TestBackendWrap(t *testing.T) {
	cause := errors.New("connection refused")
	err := BackendWrap(cause, ErrBackendConnectionFailed, "failed to connect")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestAgent(t *testing.T) {
	err := Agent(ErrAgentNotFound, "agent not found")

	if err.Category != CategoryAgent {
		t.Errorf("Category = %q, want %q", err.Category, CategoryAgent)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestAgentf(t *testing.T) {
	err := Agentf(ErrAgentNotFound, "agent %s does not exist", "assistant")

	if err.Message != "agent assistant does not exist" {
		t.Errorf("Message = %q, want %q", err.Message, "agent assistant does not exist")
	}
}

func TestAgentWrap(t *testing.T) {
	cause := errors.New("backend error")
	err := AgentWrap(cause, ErrAgentCreationFailed, "failed to create agent")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestCommand(t *testing.T) {
	err := Command(ErrCommandNotFound, "unknown command")

	if err.Category != CategoryCommand {
		t.Errorf("Category = %q, want %q", err.Category, CategoryCommand)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestCommandf(t *testing.T) {
	err := Commandf(ErrCommandInvalidArg, "invalid argument: %s", "foo")

	if err.Message != "invalid argument: foo" {
		t.Errorf("Message = %q, want %q", err.Message, "invalid argument: foo")
	}
}

func TestCommandWrap(t *testing.T) {
	cause := errors.New("parse error")
	err := CommandWrap(cause, ErrCommandInvalidSyntax, "syntax error")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestValidation(t *testing.T) {
	err := Validation(ErrValidationRequired, "field required")

	if err.Category != CategoryValidation {
		t.Errorf("Category = %q, want %q", err.Category, CategoryValidation)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestValidationf(t *testing.T) {
	err := Validationf(ErrValidationInvalidValue, "invalid value for field %s", "name")

	if err.Message != "invalid value for field name" {
		t.Errorf("Message = %q, want %q", err.Message, "invalid value for field name")
	}
}

func TestValidationWrap(t *testing.T) {
	cause := errors.New("strconv: invalid syntax")
	err := ValidationWrap(cause, ErrValidationTypeMismatch, "type mismatch")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

func TestNetwork(t *testing.T) {
	err := Network(ErrNetworkTimeout, "request timed out")

	if err.Category != CategoryNetwork {
		t.Errorf("Category = %q, want %q", err.Category, CategoryNetwork)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestNetworkf(t *testing.T) {
	err := Networkf(ErrNetworkConnectionRefused, "connection to %s:%d refused", "localhost", 8080)

	if err.Message != "connection to localhost:8080 refused" {
		t.Errorf("Message = %q, want %q", err.Message, "connection to localhost:8080 refused")
	}
}

func TestNetworkWrap(t *testing.T) {
	cause := errors.New("dial tcp: connection refused")
	err := NetworkWrap(cause, ErrNetworkConnectionRefused, "connection failed")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

func TestIO(t *testing.T) {
	err := IO(ErrIOFileNotFound, "file not found")

	if err.Category != CategoryIO {
		t.Errorf("Category = %q, want %q", err.Category, CategoryIO)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestIOf(t *testing.T) {
	err := IOf(ErrIOPermissionDenied, "cannot write to %s", "/etc/config")

	if err.Message != "cannot write to /etc/config" {
		t.Errorf("Message = %q, want %q", err.Message, "cannot write to /etc/config")
	}
}

func TestIOWrap(t *testing.T) {
	cause := errors.New("permission denied")
	err := IOWrap(cause, ErrIOPermissionDenied, "cannot access file")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

func TestInternal(t *testing.T) {
	err := Internal(ErrInternalError, "unexpected error")

	if err.Category != CategoryInternal {
		t.Errorf("Category = %q, want %q", err.Category, CategoryInternal)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestInternalf(t *testing.T) {
	err := Internalf(ErrInternalNilPointer, "nil pointer at %s", "line 42")

	if err.Message != "nil pointer at line 42" {
		t.Errorf("Message = %q, want %q", err.Message, "nil pointer at line 42")
	}
}

func TestInternalWrap(t *testing.T) {
	cause := errors.New("runtime error")
	err := InternalWrap(cause, ErrInternalPanic, "panic recovered")

	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

// -----------------------------------------------------------------------------
// Context-Aware Constructor Tests
// -----------------------------------------------------------------------------

func TestConfigWithContext(t *testing.T) {
	ctx := map[string]string{"path": "/home/user/config.yaml"}
	err := ConfigWithContext(ErrConfigNotFound, "config not found", ctx)

	if err.Context["path"] != "/home/user/config.yaml" {
		t.Errorf("Context[path] = %q, want %q", err.Context["path"], "/home/user/config.yaml")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestBackendWithContext(t *testing.T) {
	ctx := map[string]string{ContextBackend: BackendClaudeCode}
	err := BackendWithContext(ErrBackendNotInstalled, "claude cli not found", ctx)

	if err.Context[ContextBackend] != BackendClaudeCode {
		t.Errorf("Context[backend] = %q, want %q", err.Context[ContextBackend], BackendClaudeCode)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
	// Should have Claude-specific suggestions
	found := false
	for _, s := range err.Suggestions {
		if strings.Contains(s, "claude") || strings.Contains(s, "Claude") {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected Claude-specific suggestion for BackendClaudeCode context")
	}
}

func TestAgentWithContext(t *testing.T) {
	ctx := map[string]string{"agent": "researcher"}
	err := AgentWithContext(ErrAgentNotFound, "agent not found", ctx)

	if err.Context["agent"] != "researcher" {
		t.Errorf("Context[agent] = %q, want %q", err.Context["agent"], "researcher")
	}
}

func TestCommandWithContext(t *testing.T) {
	ctx := map[string]string{"command": "/extract"}
	err := CommandWithContext(ErrCommandInvalidSyntax, "invalid syntax", ctx)

	if err.Context["command"] != "/extract" {
		t.Errorf("Context[command] = %q, want %q", err.Context["command"], "/extract")
	}
}

func TestValidationWithContext(t *testing.T) {
	ctx := map[string]string{"field": "temperature", "min": "0", "max": "2"}
	err := ValidationWithContext(ErrValidationOutOfRange, "value out of range", ctx)

	if err.Context["field"] != "temperature" {
		t.Errorf("Context[field] = %q, want %q", err.Context["field"], "temperature")
	}
}

func TestNetworkWithContext(t *testing.T) {
	ctx := map[string]string{"host": "api.example.com", "port": "443"}
	err := NetworkWithContext(ErrNetworkTimeout, "connection timed out", ctx)

	if err.Context["host"] != "api.example.com" {
		t.Errorf("Context[host] = %q, want %q", err.Context["host"], "api.example.com")
	}
}

func TestIOWithContext(t *testing.T) {
	ctx := map[string]string{"path": "/var/log/app.log"}
	err := IOWithContext(ErrIOPermissionDenied, "permission denied", ctx)

	if err.Context["path"] != "/var/log/app.log" {
		t.Errorf("Context[path] = %q, want %q", err.Context["path"], "/var/log/app.log")
	}
}

func TestInternalWithContext(t *testing.T) {
	ctx := map[string]string{"function": "processRequest", "line": "42"}
	err := InternalWithContext(ErrInternalInvariantViolation, "invariant violated", ctx)

	if err.Context["function"] != "processRequest" {
		t.Errorf("Context[function] = %q, want %q", err.Context["function"], "processRequest")
	}
}

// -----------------------------------------------------------------------------
// Quick Constructor Tests
// -----------------------------------------------------------------------------

func TestConfigNotFound(t *testing.T) {
	err := ConfigNotFound("/home/user/.config/weaver/config.yaml")

	if err.Code != ErrConfigNotFound {
		t.Errorf("Code = %q, want %q", err.Code, ErrConfigNotFound)
	}
	if err.Context["path"] != "/home/user/.config/weaver/config.yaml" {
		t.Errorf("Context[path] = %q, want correct path", err.Context["path"])
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestConfigParseError(t *testing.T) {
	cause := errors.New("yaml: line 5: unexpected indentation")
	err := ConfigParseError("/home/user/config.yaml", cause)

	if err.Code != ErrConfigParseFailed {
		t.Errorf("Code = %q, want %q", err.Code, ErrConfigParseFailed)
	}
	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
	if err.Context["path"] != "/home/user/config.yaml" {
		t.Errorf("Context[path] = %q, want correct path", err.Context["path"])
	}
}

func TestBackendNotAvailable(t *testing.T) {
	err := BackendNotAvailable()

	if err.Code != ErrBackendUnavailable {
		t.Errorf("Code = %q, want %q", err.Code, ErrBackendUnavailable)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestBackendNotFoundQuick(t *testing.T) {
	err := BackendNotFound("loom")

	if err.Code != ErrBackendNotFound {
		t.Errorf("Code = %q, want %q", err.Code, ErrBackendNotFound)
	}
	if err.Context["backend"] != "loom" {
		t.Errorf("Context[backend] = %q, want %q", err.Context["backend"], "loom")
	}
}

func TestBackendNotInstalledError(t *testing.T) {
	err := BackendNotInstalledError(BackendClaudeCode)

	if err.Code != ErrBackendNotInstalled {
		t.Errorf("Code = %q, want %q", err.Code, ErrBackendNotInstalled)
	}
	if err.Context[ContextBackend] != BackendClaudeCode {
		t.Errorf("Context[backend] = %q, want %q", err.Context[ContextBackend], BackendClaudeCode)
	}
	// Should have backend-specific suggestions
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestBackendConnectionError(t *testing.T) {
	cause := errors.New("connection refused")
	err := BackendConnectionError("loom", cause)

	if err.Code != ErrBackendConnectionFailed {
		t.Errorf("Code = %q, want %q", err.Code, ErrBackendConnectionFailed)
	}
	if err.Context["backend"] != "loom" {
		t.Errorf("Context[backend] = %q, want %q", err.Context["backend"], "loom")
	}
	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

func TestBackendTimeoutError(t *testing.T) {
	err := BackendTimeoutError("claudecode")

	if err.Code != ErrBackendTimeout {
		t.Errorf("Code = %q, want %q", err.Code, ErrBackendTimeout)
	}
	if err.Context["backend"] != "claudecode" {
		t.Errorf("Context[backend] = %q, want %q", err.Context["backend"], "claudecode")
	}
}

func TestAgentNotFoundQuick(t *testing.T) {
	err := AgentNotFound("researcher")

	if err.Code != ErrAgentNotFound {
		t.Errorf("Code = %q, want %q", err.Code, ErrAgentNotFound)
	}
	if err.Context["agent"] != "researcher" {
		t.Errorf("Context[agent] = %q, want %q", err.Context["agent"], "researcher")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestAgentCreationError(t *testing.T) {
	cause := errors.New("backend unavailable")
	err := AgentCreationError("assistant", cause)

	if err.Code != ErrAgentCreationFailed {
		t.Errorf("Code = %q, want %q", err.Code, ErrAgentCreationFailed)
	}
	if err.Context["agent"] != "assistant" {
		t.Errorf("Context[agent] = %q, want %q", err.Context["agent"], "assistant")
	}
	if err.Cause != cause {
		t.Error("Expected cause to be wrapped")
	}
}

func TestAgentChatError(t *testing.T) {
	cause := errors.New("stream closed")
	err := AgentChatError("researcher", cause)

	if err.Code != ErrAgentChatFailed {
		t.Errorf("Code = %q, want %q", err.Code, ErrAgentChatFailed)
	}
	if err.Context["agent"] != "researcher" {
		t.Errorf("Context[agent] = %q, want %q", err.Context["agent"], "researcher")
	}
}

func TestCommandNotFoundQuick(t *testing.T) {
	err := CommandNotFound("/foo")

	if err.Code != ErrCommandNotFound {
		t.Errorf("Code = %q, want %q", err.Code, ErrCommandNotFound)
	}
	if err.Context["command"] != "/foo" {
		t.Errorf("Context[command] = %q, want %q", err.Context["command"], "/foo")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

func TestCommandMissingArgsQuick(t *testing.T) {
	err := CommandMissingArgs("/extract", "/extract <count>")

	if err.Code != ErrCommandMissingArgs {
		t.Errorf("Code = %q, want %q", err.Code, ErrCommandMissingArgs)
	}
	if err.Context["command"] != "/extract" {
		t.Errorf("Context[command] = %q, want %q", err.Context["command"], "/extract")
	}
	if err.Context["usage"] != "/extract <count>" {
		t.Errorf("Context[usage] = %q, want %q", err.Context["usage"], "/extract <count>")
	}
}

func TestCommandInvalidArgQuick(t *testing.T) {
	err := CommandInvalidArg("abc", "integer")

	if err.Code != ErrCommandInvalidArg {
		t.Errorf("Code = %q, want %q", err.Code, ErrCommandInvalidArg)
	}
	if err.Context["argument"] != "abc" {
		t.Errorf("Context[argument] = %q, want %q", err.Context["argument"], "abc")
	}
	if err.Context["expected"] != "integer" {
		t.Errorf("Context[expected] = %q, want %q", err.Context["expected"], "integer")
	}
}

func TestValidationRequiredQuick(t *testing.T) {
	err := ValidationRequired("name")

	if err.Code != ErrValidationRequired {
		t.Errorf("Code = %q, want %q", err.Code, ErrValidationRequired)
	}
	if err.Context["field"] != "name" {
		t.Errorf("Context[field] = %q, want %q", err.Context["field"], "name")
	}
}

func TestValidationInvalidQuick(t *testing.T) {
	err := ValidationInvalid("email", "not-an-email", "must be a valid email address")

	if err.Code != ErrValidationInvalidValue {
		t.Errorf("Code = %q, want %q", err.Code, ErrValidationInvalidValue)
	}
	if err.Context["field"] != "email" {
		t.Errorf("Context[field] = %q, want %q", err.Context["field"], "email")
	}
	if err.Context["value"] != "not-an-email" {
		t.Errorf("Context[value] = %q, want %q", err.Context["value"], "not-an-email")
	}
}

func TestValidationOutOfRangeQuick(t *testing.T) {
	err := ValidationOutOfRange("temperature", 2.5, 0.0, 2.0)

	if err.Code != ErrValidationOutOfRange {
		t.Errorf("Code = %q, want %q", err.Code, ErrValidationOutOfRange)
	}
	if err.Context["field"] != "temperature" {
		t.Errorf("Context[field] = %q, want %q", err.Context["field"], "temperature")
	}
	if err.Context["min"] != "0" {
		t.Errorf("Context[min] = %q, want %q", err.Context["min"], "0")
	}
	if err.Context["max"] != "2" {
		t.Errorf("Context[max] = %q, want %q", err.Context["max"], "2")
	}
}

func TestNetworkTimeoutQuick(t *testing.T) {
	err := NetworkTimeout("api.anthropic.com")

	if err.Code != ErrNetworkTimeout {
		t.Errorf("Code = %q, want %q", err.Code, ErrNetworkTimeout)
	}
	if err.Context["host"] != "api.anthropic.com" {
		t.Errorf("Context[host] = %q, want %q", err.Context["host"], "api.anthropic.com")
	}
}

func TestNetworkConnectionRefusedQuick(t *testing.T) {
	err := NetworkConnectionRefused("localhost", 8080)

	if err.Code != ErrNetworkConnectionRefused {
		t.Errorf("Code = %q, want %q", err.Code, ErrNetworkConnectionRefused)
	}
	if err.Context["host"] != "localhost" {
		t.Errorf("Context[host] = %q, want %q", err.Context["host"], "localhost")
	}
	if err.Context["port"] != "8080" {
		t.Errorf("Context[port] = %q, want %q", err.Context["port"], "8080")
	}
}

func TestIOFileNotFoundQuick(t *testing.T) {
	err := IOFileNotFound("/home/user/data.json")

	if err.Code != ErrIOFileNotFound {
		t.Errorf("Code = %q, want %q", err.Code, ErrIOFileNotFound)
	}
	if err.Context["path"] != "/home/user/data.json" {
		t.Errorf("Context[path] = %q, want %q", err.Context["path"], "/home/user/data.json")
	}
}

func TestIOPermissionDeniedQuick(t *testing.T) {
	err := IOPermissionDenied("/etc/secrets")

	if err.Code != ErrIOPermissionDenied {
		t.Errorf("Code = %q, want %q", err.Code, ErrIOPermissionDenied)
	}
	if err.Context["path"] != "/etc/secrets" {
		t.Errorf("Context[path] = %q, want %q", err.Context["path"], "/etc/secrets")
	}
}

func TestInternalPanicQuick(t *testing.T) {
	err := InternalPanic("runtime error: invalid memory address")

	if err.Code != ErrInternalPanic {
		t.Errorf("Code = %q, want %q", err.Code, ErrInternalPanic)
	}
	if !strings.Contains(err.Message, "runtime error") {
		t.Errorf("Message should contain panic info, got %q", err.Message)
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions to be auto-attached")
	}
}

// -----------------------------------------------------------------------------
// Chaining Tests
// -----------------------------------------------------------------------------

func TestConstructorChaining(t *testing.T) {
	err := Config(ErrConfigInvalid, "invalid config").
		WithContext("field", "backend").
		WithContext("value", "unknown").
		WithSuggestion("Use 'claudecode' or 'loom' as backend value")

	if err.Code != ErrConfigInvalid {
		t.Errorf("Code = %q, want %q", err.Code, ErrConfigInvalid)
	}
	if err.Context["field"] != "backend" {
		t.Errorf("Context[field] = %q, want %q", err.Context["field"], "backend")
	}
	// Should have both auto-attached and manual suggestions
	if len(err.Suggestions) < 2 {
		t.Errorf("Expected at least 2 suggestions (auto + manual), got %d", len(err.Suggestions))
	}
}

func TestWrapperChaining(t *testing.T) {
	cause := errors.New("json: unexpected token")
	err := ConfigWrap(cause, ErrConfigParseFailed, "parse error").
		WithContext("path", "/home/user/config.yaml").
		WithContext("line", "15")

	if err.Cause != cause {
		t.Error("Expected cause to be preserved after chaining")
	}
	if err.Context["path"] != "/home/user/config.yaml" {
		t.Errorf("Context[path] = %q, want correct path", err.Context["path"])
	}
}

// -----------------------------------------------------------------------------
// Error Interface Tests
// -----------------------------------------------------------------------------

func TestConstructorErrorInterface(t *testing.T) {
	var err error = Config(ErrConfigNotFound, "not found")

	// Should implement error interface
	if err.Error() == "" {
		t.Error("Expected non-empty error message")
	}

	// Should be convertible to WeaverError
	we, ok := AsWeaverError(err)
	if !ok {
		t.Error("Expected AsWeaverError to return true")
	}
	if we.Code != ErrConfigNotFound {
		t.Errorf("Code = %q, want %q", we.Code, ErrConfigNotFound)
	}
}

func TestConstructorErrorsIs(t *testing.T) {
	cause := errors.New("original error")
	err := BackendWrap(cause, ErrBackendConnectionFailed, "connection failed")

	// Should match the wrapped cause
	if !errors.Is(err, cause) {
		t.Error("errors.Is should match wrapped cause")
	}

	// Should match the same error code
	target := New(ErrBackendConnectionFailed, CategoryBackend, "different message")
	if !errors.Is(err, target) {
		t.Error("errors.Is should match same error code")
	}
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

func TestConstructorWithDisplay(t *testing.T) {
	err := ConfigNotFound("/home/user/.config/weaver/config.yaml")

	// Format without color for testing
	output := Sprint(err)

	// Should contain error code
	if !strings.Contains(output, "CONFIG_NOT_FOUND") {
		t.Error("Output should contain error code")
	}

	// Should contain context
	if !strings.Contains(output, "path") {
		t.Error("Output should contain context key")
	}

	// Should contain suggestions
	if !strings.Contains(output, "weaver --init") || !strings.Contains(output, "config.yaml") {
		t.Error("Output should contain suggestions")
	}
}

func TestBackendWithContextDisplay(t *testing.T) {
	err := BackendNotInstalledError(BackendClaudeCode)

	// Format without color
	output := Sprint(err)

	// Should contain error code
	if !strings.Contains(output, "BACKEND_NOT_INSTALLED") {
		t.Error("Output should contain error code")
	}

	// Should have Claude-specific suggestion
	hasClaudeSuggestion := strings.Contains(output, "claude") ||
		strings.Contains(output, "npm install") ||
		strings.Contains(output, "Claude")
	if !hasClaudeSuggestion {
		t.Error("Output should contain Claude-specific suggestion")
	}
}

// -----------------------------------------------------------------------------
// Nil Handling Tests
// -----------------------------------------------------------------------------

func TestWrapWithNilCause(t *testing.T) {
	err := ConfigWrap(nil, ErrConfigNotFound, "config not found")

	if err.Cause != nil {
		t.Error("Expected nil cause to remain nil")
	}
	if !err.HasSuggestions() {
		t.Error("Expected suggestions even with nil cause")
	}
}
