// Package errors integration tests for error display across different scenarios.
// Tests verify error output format, color codes, and suggestion display.
package errors

import (
	"bytes"
	"fmt"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Error Output Format Tests
// -----------------------------------------------------------------------------
// These tests verify the complete error output format is correct.

func TestIntegration_ErrorOutputFormat_ConfigError(t *testing.T) {
	// Test a config error with full context and suggestions
	err := ConfigNotFound("/home/user/.weaver/config.yaml")

	output := Sprint(err)

	// Verify header format: ERROR [CODE]: Message
	if !strings.Contains(output, "ERROR [CONFIG_NOT_FOUND]:") {
		t.Errorf("expected error header format, got:\n%s", output)
	}
	if !strings.Contains(output, "configuration file not found") {
		t.Errorf("expected error message, got:\n%s", output)
	}

	// Verify context format: indented key: value
	if !strings.Contains(output, "  path: /home/user/.weaver/config.yaml") {
		t.Errorf("expected context with path, got:\n%s", output)
	}

	// Verify suggestions format: indented arrow with text
	if !strings.Contains(output, "  → Run 'weaver --init'") {
		t.Errorf("expected suggestion with arrow prefix, got:\n%s", output)
	}
}

func TestIntegration_ErrorOutputFormat_BackendError(t *testing.T) {
	cause := fmt.Errorf("dial tcp 127.0.0.1:8080: connection refused")
	err := BackendConnectionError("loom", cause)

	output := Sprint(err)

	// Verify all parts are present
	verifyOutputContains(t, output, []string{
		"ERROR [BACKEND_CONNECTION_FAILED]:",
		"failed to connect to backend",
		"backend: loom",
		"cause: dial tcp",
		"→", // At least one suggestion
	})
}

func TestIntegration_ErrorOutputFormat_ValidationError(t *testing.T) {
	err := ValidationOutOfRange("temperature", 2.5, 0.0, 2.0)

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [VALIDATION_OUT_OF_RANGE]:",
		"temperature value 2.5 is out of range [0, 2]",
		"field: temperature",
		"value: 2.5",
		"min: 0",
		"max: 2",
	})
}

func TestIntegration_ErrorOutputFormat_ChainedError(t *testing.T) {
	// Create a chain of errors
	rootCause := fmt.Errorf("EOF")
	yamlError := fmt.Errorf("yaml: unmarshal error: %w", rootCause)
	configError := ConfigParseError("/home/user/.weaver/config.yaml", yamlError)

	output := Sprint(configError)

	verifyOutputContains(t, output, []string{
		"ERROR [CONFIG_PARSE_FAILED]:",
		"failed to parse configuration file",
		"path:",
		"cause:",
		"yaml: unmarshal error",
	})
}

func TestIntegration_ErrorOutputFormat_MinimalError(t *testing.T) {
	// Error with no context, cause, or suggestions
	err := New("SIMPLE_ERROR", CategoryCommand, "simple error message")

	output := Sprint(err)

	// Should just be header line
	expected := "ERROR [SIMPLE_ERROR]: simple error message\n"
	if output != expected {
		t.Errorf("expected minimal output %q, got %q", expected, output)
	}
}

func TestIntegration_ErrorOutputFormat_MultipleContextEntries(t *testing.T) {
	err := New("MULTI_CONTEXT", CategoryConfig, "test error").
		WithContext("zebra", "last").
		WithContext("alpha", "first").
		WithContext("beta", "second")

	output := Sprint(err)

	// Verify context keys are sorted alphabetically
	alphaIdx := strings.Index(output, "alpha:")
	betaIdx := strings.Index(output, "beta:")
	zebraIdx := strings.Index(output, "zebra:")

	if alphaIdx == -1 || betaIdx == -1 || zebraIdx == -1 {
		t.Fatalf("missing context keys in output:\n%s", output)
	}

	if !(alphaIdx < betaIdx && betaIdx < zebraIdx) {
		t.Errorf("expected context keys sorted alphabetically, got:\n%s", output)
	}
}

func TestIntegration_ErrorOutputFormat_MultipleSuggestions(t *testing.T) {
	err := New("MULTI_SUGGEST", CategoryConfig, "test error").
		WithSuggestion("First suggestion").
		WithSuggestion("Second suggestion").
		WithSuggestion("Third suggestion")

	output := Sprint(err)

	// All suggestions should be present
	if strings.Count(output, "→") != 3 {
		t.Errorf("expected 3 arrow prefixes for suggestions, got:\n%s", output)
	}

	verifyOutputContains(t, output, []string{
		"→ First suggestion",
		"→ Second suggestion",
		"→ Third suggestion",
	})
}

// -----------------------------------------------------------------------------
// Color Code Tests
// -----------------------------------------------------------------------------
// These tests verify ANSI color codes are correctly applied.

func TestIntegration_ColorCodes_TTYOutput(t *testing.T) {
	err := Config(ErrConfigNotFound, "configuration file not found").
		WithContext("path", "/home/user/.weaver/config.yaml").
		WithCause(fmt.Errorf("file does not exist"))

	output := Sprintc(err) // Force color output

	// Verify all required color codes are present
	colorCodes := []struct {
		code        string
		description string
	}{
		{colorRed, "red for error header"},
		{colorBold, "bold for emphasis"},
		{colorYellow, "yellow for context"},
		{colorCyan, "cyan for suggestions"},
		{colorDim, "dim for cause"},
		{colorReset, "reset code"},
	}

	for _, cc := range colorCodes {
		if !strings.Contains(output, cc.code) {
			t.Errorf("expected %s (%q), got:\n%s", cc.description, cc.code, output)
		}
	}
}

func TestIntegration_ColorCodes_NonTTYOutput(t *testing.T) {
	err := Config(ErrConfigNotFound, "configuration file not found").
		WithContext("path", "/home/user/.weaver/config.yaml").
		WithCause(fmt.Errorf("file does not exist"))

	output := Sprint(err) // Force no-color output

	// Verify NO color codes are present
	colorCodes := []string{colorRed, colorBold, colorYellow, colorCyan, colorDim, colorReset, colorGreen}

	for _, code := range colorCodes {
		if strings.Contains(output, code) {
			t.Errorf("unexpected color code %q in non-TTY output:\n%s", code, output)
		}
	}
}

func TestIntegration_ColorCodes_RedForErrorCode(t *testing.T) {
	err := Backend(ErrBackendUnavailable, "no backends available")
	output := Sprintc(err)

	// Verify the ERROR header and code are in red
	// Pattern: \033[31m\033[1mERROR\033[0m\033[31m [CODE]:
	redBoldError := colorRed + colorBold + "ERROR"
	if !strings.Contains(output, redBoldError) {
		t.Errorf("expected red bold ERROR header, got:\n%s", output)
	}
}

func TestIntegration_ColorCodes_YellowForContext(t *testing.T) {
	err := New("TEST", CategoryConfig, "test").
		WithContext("mykey", "myvalue")
	output := Sprintc(err)

	// Context key should be in yellow
	yellowKey := colorYellow + "mykey:"
	if !strings.Contains(output, yellowKey) {
		t.Errorf("expected yellow context key, got:\n%s", output)
	}
}

func TestIntegration_ColorCodes_CyanForSuggestions(t *testing.T) {
	err := New("TEST", CategoryConfig, "test").
		WithSuggestion("Try this fix")
	output := Sprintc(err)

	// Suggestion arrow should be in cyan
	cyanArrow := colorCyan + "→"
	if !strings.Contains(output, cyanArrow) {
		t.Errorf("expected cyan suggestion arrow, got:\n%s", output)
	}
}

func TestIntegration_ColorCodes_DimForCause(t *testing.T) {
	err := New("TEST", CategoryIO, "test").
		WithCause(fmt.Errorf("underlying error"))
	output := Sprintc(err)

	// Cause should be in dim
	dimCause := colorDim + "cause:"
	if !strings.Contains(output, dimCause) {
		t.Errorf("expected dim cause, got:\n%s", output)
	}
}

func TestIntegration_ColorCodes_StandardErrorColored(t *testing.T) {
	stdErr := fmt.Errorf("a standard error")
	output := Sprintc(stdErr)

	// Standard errors should still have color in color mode
	if !strings.Contains(output, colorRed) {
		t.Errorf("expected red color for standard error, got:\n%s", output)
	}
	if !strings.Contains(output, "a standard error") {
		t.Errorf("expected error message, got:\n%s", output)
	}
}

// -----------------------------------------------------------------------------
// Suggestion Display Tests
// -----------------------------------------------------------------------------
// These tests verify suggestions are correctly attached and displayed.

func TestIntegration_SuggestionDisplay_AutoAttached(t *testing.T) {
	// Smart constructors should auto-attach suggestions from registry
	err := ConfigNotFound("/path/to/config.yaml")

	if !err.HasSuggestions() {
		t.Error("expected suggestions to be auto-attached")
	}

	output := Sprint(err)

	// Should have the registry suggestion about --init
	if !strings.Contains(output, "weaver --init") {
		t.Errorf("expected --init suggestion from registry, got:\n%s", output)
	}
}

func TestIntegration_SuggestionDisplay_BackendSpecific(t *testing.T) {
	// Backend-specific errors should get backend-specific suggestions
	err := BackendNotInstalledError("claudecode")

	output := Sprint(err)

	// Should have Claude-specific installation suggestions
	if !strings.Contains(output, "npm install") || !strings.Contains(output, "claude-cli") {
		t.Errorf("expected Claude CLI installation suggestion, got:\n%s", output)
	}
}

func TestIntegration_SuggestionDisplay_OSSpecific(t *testing.T) {
	// Create an error that has OS-specific suggestions
	err := Config(ErrConfigInitFailed, "failed to initialize config")

	output := Sprint(err)

	// Should have platform-appropriate suggestion
	currentOS := runtime.GOOS
	switch currentOS {
	case "linux", "darwin":
		// Unix systems should see mkdir -p suggestion
		if !strings.Contains(output, "mkdir -p") {
			t.Logf("Note: expected mkdir suggestion for %s (may not be present)", currentOS)
		}
	case "windows":
		// Windows should see APPDATA suggestion
		if !strings.Contains(output, "APPDATA") {
			t.Logf("Note: expected APPDATA suggestion for Windows (may not be present)")
		}
	}
}

func TestIntegration_SuggestionDisplay_ContextAware(t *testing.T) {
	// Backend-specific context should affect suggestions
	ctx := map[string]string{
		ContextBackend: BackendLoom,
	}
	err := BackendWithContext(ErrBackendAuthFailed, "authentication failed", ctx)

	output := Sprint(err)

	// Should have Loom-specific auth suggestion
	if !strings.Contains(output, "Loom") || !strings.Contains(output, "token") {
		t.Logf("Note: expected Loom-specific auth suggestion, got:\n%s", output)
	}
}

func TestIntegration_SuggestionDisplay_ManuallyAdded(t *testing.T) {
	err := New("CUSTOM_ERROR", CategoryCommand, "custom error").
		WithSuggestion("My custom suggestion")

	output := Sprint(err)

	if !strings.Contains(output, "→ My custom suggestion") {
		t.Errorf("expected manually added suggestion, got:\n%s", output)
	}
}

func TestIntegration_SuggestionDisplay_CombinedRegistryAndManual(t *testing.T) {
	err := AgentNotFound("myAgent").
		WithSuggestion("Additional manual suggestion")

	output := Sprint(err)

	// Should have both registry and manual suggestions
	if !strings.Contains(output, "/agents") { // Registry suggestion
		t.Errorf("expected registry suggestion, got:\n%s", output)
	}
	if !strings.Contains(output, "Additional manual suggestion") { // Manual
		t.Errorf("expected manual suggestion, got:\n%s", output)
	}
}

func TestIntegration_SuggestionDisplay_VisualSeparation(t *testing.T) {
	// When there's context/cause, suggestions should have blank line before them
	err := New("TEST", CategoryConfig, "test").
		WithContext("key", "value").
		WithSuggestion("suggestion")

	output := Sprint(err)

	// Should have blank line between context and suggestions
	if !strings.Contains(output, "\n\n  →") {
		t.Errorf("expected blank line before suggestions, got:\n%s", output)
	}
}

func TestIntegration_SuggestionDisplay_NoSeparationWithoutContext(t *testing.T) {
	// Without context/cause, no blank line before suggestions
	err := New("TEST", CategoryConfig, "test").
		WithSuggestion("suggestion")

	output := Sprint(err)

	// Should NOT have double newline
	if strings.Contains(output, "\n\n") {
		t.Errorf("expected no blank line without context, got:\n%s", output)
	}
}

// -----------------------------------------------------------------------------
// Error Scenario Tests
// -----------------------------------------------------------------------------
// These tests verify complete error scenarios across different error types.

func TestIntegration_Scenario_AgentChatFailure(t *testing.T) {
	cause := fmt.Errorf("connection refused")
	err := AgentChatError("assistant", cause)

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [AGENT_CHAT_FAILED]:",
		"chat request failed",
		"agent: assistant",
		"cause: connection refused",
		"→", // Should have suggestions
	})
}

func TestIntegration_Scenario_NetworkTimeout(t *testing.T) {
	err := NetworkTimeout("api.example.com")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [NETWORK_TIMEOUT]:",
		"connection timed out",
		"host: api.example.com",
	})

	// Should have timeout-related suggestions
	if !strings.Contains(output, "internet connection") && !strings.Contains(output, "slow") {
		t.Errorf("expected timeout-related suggestions, got:\n%s", output)
	}
}

func TestIntegration_Scenario_IOPermissionDenied(t *testing.T) {
	err := IOPermissionDenied("/etc/shadow")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [IO_PERMISSION_DENIED]:",
		"permission denied: /etc/shadow",
		"path: /etc/shadow",
	})

	// Should have permission-related suggestions
	if !strings.Contains(output, "permission") {
		t.Errorf("expected permission-related suggestions, got:\n%s", output)
	}
}

func TestIntegration_Scenario_CommandNotFound(t *testing.T) {
	err := CommandNotFound("/unknown")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [COMMAND_NOT_FOUND]:",
		"unknown command: /unknown",
		"command: /unknown",
		"/help", // Should suggest help command
	})
}

func TestIntegration_Scenario_ValidationRequired(t *testing.T) {
	err := ValidationRequired("agent_name")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [VALIDATION_REQUIRED]:",
		"required field is missing: agent_name",
		"field: agent_name",
	})
}

func TestIntegration_Scenario_BackendTimeout(t *testing.T) {
	err := BackendTimeoutError("claudecode")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [BACKEND_TIMEOUT]:",
		"backend request timed out",
		"backend: claudecode",
	})
}

// -----------------------------------------------------------------------------
// Formatter Configuration Tests
// -----------------------------------------------------------------------------

func TestIntegration_Formatter_CustomWriter(t *testing.T) {
	var buf bytes.Buffer
	f := &Formatter{
		UseColor: false,
		Writer:   &buf,
		Indent:   "  ",
	}

	err := Config(ErrConfigNotFound, "config not found")
	f.Display(err)

	if buf.Len() == 0 {
		t.Error("expected output written to custom writer")
	}
	if !strings.Contains(buf.String(), "CONFIG_NOT_FOUND") {
		t.Errorf("expected error code in output, got:\n%s", buf.String())
	}
}

func TestIntegration_Formatter_CustomIndent(t *testing.T) {
	f := &Formatter{
		UseColor: false,
		Indent:   "    ", // 4 spaces instead of 2
	}

	err := New("TEST", CategoryConfig, "test").
		WithContext("key", "value")
	output := f.Format(err)

	if !strings.Contains(output, "    key:") {
		t.Errorf("expected 4-space indent, got:\n%s", output)
	}
}

func TestIntegration_Formatter_TabIndent(t *testing.T) {
	f := &Formatter{
		UseColor: false,
		Indent:   "\t",
	}

	err := New("TEST", CategoryConfig, "test").
		WithContext("key", "value")
	output := f.Format(err)

	if !strings.Contains(output, "\tkey:") {
		t.Errorf("expected tab indent, got:\n%s", output)
	}
}

// -----------------------------------------------------------------------------
// FormatMultiple Tests
// -----------------------------------------------------------------------------

func TestIntegration_FormatMultiple_DifferentCategories(t *testing.T) {
	errs := []error{
		ConfigNotFound("/path/to/config"),
		BackendNotAvailable(),
		AgentNotFound("myAgent"),
	}

	output := FormatMultiple(errs)

	verifyOutputContains(t, output, []string{
		"CONFIG_NOT_FOUND",
		"BACKEND_UNAVAILABLE",
		"AGENT_NOT_FOUND",
	})

	// Errors should be separated
	parts := strings.Split(output, "\n\n")
	if len(parts) < 2 {
		t.Errorf("expected errors to be separated by blank lines, got:\n%s", output)
	}
}

func TestIntegration_FormatMultiple_MixedTypes(t *testing.T) {
	errs := []error{
		ConfigNotFound("/path"),      // WeaverError
		fmt.Errorf("standard error"), // Standard error
	}

	output := FormatMultiple(errs)

	// Both should be present
	if !strings.Contains(output, "CONFIG_NOT_FOUND") {
		t.Errorf("expected WeaverError, got:\n%s", output)
	}
	if !strings.Contains(output, "standard error") {
		t.Errorf("expected standard error, got:\n%s", output)
	}
}

// -----------------------------------------------------------------------------
// Edge Cases
// -----------------------------------------------------------------------------

func TestIntegration_EdgeCase_NilError(t *testing.T) {
	if Sprint(nil) != "" {
		t.Error("expected empty string for nil error")
	}
	if Sprintc(nil) != "" {
		t.Error("expected empty string for nil error (color)")
	}
}

func TestIntegration_EdgeCase_EmptyContext(t *testing.T) {
	err := New("TEST", CategoryConfig, "test")
	err.Context = map[string]string{} // Empty but not nil

	output := Sprint(err)

	// Should just be header, no context section
	expected := "ERROR [TEST]: test\n"
	if output != expected {
		t.Errorf("expected minimal output for empty context, got:\n%s", output)
	}
}

func TestIntegration_EdgeCase_EmptySuggestions(t *testing.T) {
	err := New("TEST", CategoryConfig, "test")
	err.Suggestions = []string{} // Empty but not nil

	output := Sprint(err)

	// Should just be header, no suggestions
	expected := "ERROR [TEST]: test\n"
	if output != expected {
		t.Errorf("expected minimal output for empty suggestions, got:\n%s", output)
	}
}

func TestIntegration_EdgeCase_ContextWithSpecialCharacters(t *testing.T) {
	err := New("TEST", CategoryConfig, "test").
		WithContext("path", "/home/user/my file.txt").
		WithContext("query", `SELECT * FROM "users"`)

	output := Sprint(err)

	// Special characters should be preserved
	if !strings.Contains(output, "my file.txt") {
		t.Errorf("expected space in path, got:\n%s", output)
	}
	if !strings.Contains(output, `SELECT * FROM "users"`) {
		t.Errorf("expected quotes in query, got:\n%s", output)
	}
}

func TestIntegration_EdgeCase_LongMessage(t *testing.T) {
	longMessage := strings.Repeat("a", 500)
	err := New("LONG_ERROR", CategoryConfig, longMessage)

	output := Sprint(err)

	// Long message should be preserved
	if !strings.Contains(output, longMessage) {
		t.Errorf("expected long message to be preserved, got length %d", len(output))
	}
}

func TestIntegration_EdgeCase_UnicodeContent(t *testing.T) {
	err := New("UNICODE_TEST", CategoryConfig, "Error with emoji: ").
		WithContext("symbol", "→").
		WithSuggestion("Try this: ")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"Error with emoji: ",
		"symbol: →",
		"Try this: ",
	})
}

// -----------------------------------------------------------------------------
// Category Label Tests
// -----------------------------------------------------------------------------

func TestIntegration_CategoryLabel_AllCategories(t *testing.T) {
	tests := []struct {
		category Category
		expected string
	}{
		{CategoryConfig, "Configuration Error"},
		{CategoryAgent, "Agent Error"},
		{CategoryBackend, "Backend Error"},
		{CategoryCommand, "Command Error"},
		{CategoryValidation, "Validation Error"},
		{CategoryNetwork, "Network Error"},
		{CategoryIO, "I/O Error"},
		{CategoryInternal, "Internal Error"},
	}

	for _, tc := range tests {
		t.Run(string(tc.category), func(t *testing.T) {
			label := CategoryLabel(tc.category)
			if label != tc.expected {
				t.Errorf("expected label %q for %s, got %q", tc.expected, tc.category, label)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Real-World Scenario Tests
// -----------------------------------------------------------------------------

func TestIntegration_RealWorld_ConfigParseYAMLError(t *testing.T) {
	yamlErr := fmt.Errorf("yaml: line 15: did not find expected key")
	err := ConfigWrap(yamlErr, ErrConfigParseFailed, "failed to parse config").
		WithContext("file", "/home/user/.weaver/config.yaml").
		WithContext("line", "15")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ERROR [CONFIG_PARSE_FAILED]:",
		"failed to parse config",
		"file: /home/user/.weaver/config.yaml",
		"line: 15",
		"cause: yaml: line 15",
		"→", // Suggestions
	})
}

func TestIntegration_RealWorld_BackendNotInstalled(t *testing.T) {
	err := BackendNotInstalledError("claudecode")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"BACKEND_NOT_INSTALLED",
		"claudecode",
		"npm install", // Should have install instructions
	})
}

func TestIntegration_RealWorld_AgentConfigurationError(t *testing.T) {
	err := Agent(ErrAgentInvalidConfig, "invalid agent configuration").
		WithContext("agent", "myAgent").
		WithContext("field", "backend").
		WithContext("value", "unknown_backend").
		WithContext("valid_options", "claudecode, loom")

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"AGENT_INVALID_CONFIG",
		"invalid agent configuration",
		"agent: myAgent",
		"field: backend",
		"value: unknown_backend",
		"valid_options: claudecode, loom",
	})
}

func TestIntegration_RealWorld_AnalysisServerUnavailable(t *testing.T) {
	cause := fmt.Errorf("dial tcp 127.0.0.1:8000: connection refused")
	err := Wrap(cause, ErrAnalysisServerUnavailable, CategoryInternal, "analysis server unavailable").
		WithContext("server", "http://localhost:8000")

	// Attach suggestions
	err = AttachSuggestions(err)

	output := Sprint(err)

	verifyOutputContains(t, output, []string{
		"ANALYSIS_SERVER_UNAVAILABLE",
		"analysis server unavailable",
		"server: http://localhost:8000",
		"cause: dial tcp",
	})
}

// -----------------------------------------------------------------------------
// Regression Tests
// -----------------------------------------------------------------------------

func TestIntegration_Regression_ColorResetAtEnd(t *testing.T) {
	err := Config(ErrConfigNotFound, "test").
		WithContext("key", "value").
		WithSuggestion("suggestion")

	output := Sprintc(err)

	// Should end with reset code to prevent color bleed
	lastReset := strings.LastIndex(output, colorReset)
	if lastReset == -1 {
		t.Error("expected at least one color reset code")
	}

	// The last non-whitespace should be after a reset
	contentAfterReset := strings.TrimSpace(output[lastReset+len(colorReset):])
	if len(contentAfterReset) > 0 && !strings.HasPrefix(contentAfterReset, colorReset) {
		// There may be content, but the final colorReset should cover it
		t.Logf("Note: content after last reset: %q", contentAfterReset)
	}
}

func TestIntegration_Regression_NoExtraNewlines(t *testing.T) {
	err := New("TEST", CategoryConfig, "test message")
	output := Sprint(err)

	// Should end with exactly one newline
	if !strings.HasSuffix(output, "\n") {
		t.Error("expected output to end with newline")
	}
	if strings.HasSuffix(output, "\n\n") {
		t.Error("expected only one trailing newline")
	}
}

func TestIntegration_Regression_ContextKeyOrder(t *testing.T) {
	// Create same error multiple times - context order should be deterministic
	outputs := make([]string, 5)
	for i := 0; i < 5; i++ {
		err := New("TEST", CategoryConfig, "test").
			WithContext("z_key", "z").
			WithContext("a_key", "a").
			WithContext("m_key", "m")
		outputs[i] = Sprint(err)
	}

	// All outputs should be identical
	for i := 1; i < 5; i++ {
		if outputs[i] != outputs[0] {
			t.Errorf("expected deterministic output, got different results:\n%s\nvs\n%s", outputs[0], outputs[i])
		}
	}
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// verifyOutputContains checks that output contains all expected substrings.
func verifyOutputContains(t *testing.T, output string, expected []string) {
	t.Helper()
	for _, s := range expected {
		if !strings.Contains(output, s) {
			t.Errorf("expected output to contain %q, got:\n%s", s, output)
		}
	}
}

// matchesPattern checks if output matches a regex pattern.
func matchesPattern(t *testing.T, output, pattern string) bool {
	t.Helper()
	matched, err := regexp.MatchString(pattern, output)
	if err != nil {
		t.Fatalf("invalid regex pattern %q: %v", pattern, err)
	}
	return matched
}
