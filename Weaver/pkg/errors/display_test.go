// Package errors tests for error formatting and display.
package errors

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Formatter Tests
// -----------------------------------------------------------------------------

func TestDefaultFormatter(t *testing.T) {
	f := DefaultFormatter()

	if f == nil {
		t.Fatal("DefaultFormatter() returned nil")
	}
	if f.Writer == nil {
		t.Error("expected Writer to be set")
	}
	if f.Indent != "  " {
		t.Errorf("expected Indent '  ', got %q", f.Indent)
	}
}

func TestFormatter_Format_NilError(t *testing.T) {
	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(nil)

	if result != "" {
		t.Errorf("expected empty string for nil error, got %q", result)
	}
}

func TestFormatter_Format_StandardError(t *testing.T) {
	tests := []struct {
		name     string
		useColor bool
		err      error
		contains []string
	}{
		{
			name:     "no color",
			useColor: false,
			err:      fmt.Errorf("something went wrong"),
			contains: []string{"Error:", "something went wrong"},
		},
		{
			name:     "with color",
			useColor: true,
			err:      fmt.Errorf("something went wrong"),
			contains: []string{colorRed, "Error:", "something went wrong", colorReset},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &Formatter{UseColor: tt.useColor, Indent: "  "}
			result := f.Format(tt.err)

			for _, substr := range tt.contains {
				if !strings.Contains(result, substr) {
					t.Errorf("expected output to contain %q, got %q", substr, result)
				}
			}
		})
	}
}

func TestFormatter_Format_WeaverError_NoColor(t *testing.T) {
	we := New("CONFIG_NOT_FOUND", CategoryConfig, "configuration file not found").
		WithContext("path", "/home/user/.weaver/config.yaml").
		WithCause(fmt.Errorf("file does not exist")).
		WithSuggestion("Run 'weaver --init' to create a config file").
		WithSuggestion("Check if the path is correct")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	// Check for error header
	if !strings.Contains(result, "ERROR [CONFIG_NOT_FOUND]:") {
		t.Errorf("expected error header, got %q", result)
	}
	if !strings.Contains(result, "configuration file not found") {
		t.Errorf("expected message, got %q", result)
	}

	// Check for context
	if !strings.Contains(result, "path:") {
		t.Errorf("expected context key 'path:', got %q", result)
	}
	if !strings.Contains(result, "/home/user/.weaver/config.yaml") {
		t.Errorf("expected context value, got %q", result)
	}

	// Check for cause
	if !strings.Contains(result, "cause:") {
		t.Errorf("expected cause label, got %q", result)
	}
	if !strings.Contains(result, "file does not exist") {
		t.Errorf("expected cause message, got %q", result)
	}

	// Check for suggestions
	if !strings.Contains(result, "Run 'weaver --init' to create a config file") {
		t.Errorf("expected first suggestion, got %q", result)
	}
	if !strings.Contains(result, "Check if the path is correct") {
		t.Errorf("expected second suggestion, got %q", result)
	}
}

func TestFormatter_Format_WeaverError_WithColor(t *testing.T) {
	we := New("BACKEND_ERROR", CategoryBackend, "backend unavailable").
		WithContext("backend", "claudecode").
		WithSuggestion("Check if Claude CLI is installed")

	f := &Formatter{UseColor: true, Indent: "  "}
	result := f.Format(we)

	// Check for ANSI color codes
	if !strings.Contains(result, colorRed) {
		t.Error("expected red color code in output")
	}
	if !strings.Contains(result, colorBold) {
		t.Error("expected bold code in output")
	}
	if !strings.Contains(result, colorYellow) {
		t.Error("expected yellow color code for context")
	}
	if !strings.Contains(result, colorCyan) {
		t.Error("expected cyan color code for suggestions")
	}
	if !strings.Contains(result, colorReset) {
		t.Error("expected reset code in output")
	}
}

func TestFormatter_Format_WeaverError_MinimalError(t *testing.T) {
	// Test error with no context, no cause, no suggestions
	we := New("SIMPLE_ERROR", CategoryCommand, "simple error message")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	expected := "ERROR [SIMPLE_ERROR]: simple error message\n"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestFormatter_Format_ContextSorted(t *testing.T) {
	// Context keys should be sorted for consistent output
	we := New("TEST", CategoryConfig, "test").
		WithContext("zebra", "z").
		WithContext("apple", "a").
		WithContext("mango", "m")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	// Find positions of context keys
	applePos := strings.Index(result, "apple:")
	mangoPos := strings.Index(result, "mango:")
	zebraPos := strings.Index(result, "zebra:")

	if applePos == -1 || mangoPos == -1 || zebraPos == -1 {
		t.Fatalf("missing context keys in output: %q", result)
	}

	if !(applePos < mangoPos && mangoPos < zebraPos) {
		t.Errorf("context keys not sorted alphabetically in output: %q", result)
	}
}

func TestFormatter_Format_SuggestionSeparator(t *testing.T) {
	// Suggestions should have a blank line separator when there's context or cause
	we := New("TEST", CategoryConfig, "test").
		WithContext("key", "value").
		WithSuggestion("suggestion")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	// Check for blank line before suggestion
	if !strings.Contains(result, "\n\n  ") {
		t.Errorf("expected blank line before suggestion, got %q", result)
	}
}

func TestFormatter_Format_NoSuggestionSeparatorWithoutContext(t *testing.T) {
	// No blank line when there's no context or cause
	we := New("TEST", CategoryConfig, "test").
		WithSuggestion("suggestion")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	lines := strings.Split(result, "\n")
	// Should have: header line, suggestion line (no blank line)
	if len(lines) != 2 || lines[1] == "" {
		// Account for final newline not being added for suggestions
		if !(len(lines) == 2 && strings.Contains(lines[1], "suggestion")) {
			t.Errorf("unexpected format without context: %q", result)
		}
	}
}

// -----------------------------------------------------------------------------
// Display Function Tests
// -----------------------------------------------------------------------------

func TestFormatter_Display_NilError(t *testing.T) {
	var buf bytes.Buffer
	f := &Formatter{UseColor: false, Writer: &buf, Indent: "  "}

	f.Display(nil)

	if buf.Len() != 0 {
		t.Errorf("expected no output for nil error, got %q", buf.String())
	}
}

func TestFormatter_Display_WritesToWriter(t *testing.T) {
	var buf bytes.Buffer
	f := &Formatter{UseColor: false, Writer: &buf, Indent: "  "}

	we := New("TEST", CategoryConfig, "test message")
	f.Display(we)

	if buf.Len() == 0 {
		t.Error("expected output to be written")
	}
	if !strings.Contains(buf.String(), "TEST") {
		t.Errorf("expected error code in output, got %q", buf.String())
	}
}

// -----------------------------------------------------------------------------
// Sprint and Sprintc Tests
// -----------------------------------------------------------------------------

func TestSprint(t *testing.T) {
	we := New("TEST_ERROR", CategoryConfig, "test message").
		WithContext("key", "value").
		WithSuggestion("try this")

	result := Sprint(we)

	// Should NOT contain color codes
	if strings.Contains(result, colorRed) {
		t.Error("Sprint() should not contain color codes")
	}
	if strings.Contains(result, colorReset) {
		t.Error("Sprint() should not contain color reset codes")
	}

	// Should contain error info
	if !strings.Contains(result, "ERROR [TEST_ERROR]:") {
		t.Errorf("expected error header, got %q", result)
	}
	if !strings.Contains(result, "key:") {
		t.Errorf("expected context, got %q", result)
	}
}

func TestSprintc(t *testing.T) {
	we := New("TEST_ERROR", CategoryConfig, "test message").
		WithContext("key", "value").
		WithSuggestion("try this")

	result := Sprintc(we)

	// Should contain color codes
	if !strings.Contains(result, colorRed) {
		t.Error("Sprintc() should contain color codes")
	}
	if !strings.Contains(result, colorReset) {
		t.Error("Sprintc() should contain color reset codes")
	}

	// Should contain error info
	if !strings.Contains(result, "TEST_ERROR") {
		t.Errorf("expected error code, got %q", result)
	}
}

func TestSprint_StandardError(t *testing.T) {
	err := fmt.Errorf("standard error")
	result := Sprint(err)

	if !strings.Contains(result, "Error:") {
		t.Errorf("expected 'Error:' prefix, got %q", result)
	}
	if !strings.Contains(result, "standard error") {
		t.Errorf("expected error message, got %q", result)
	}
}

func TestSprintc_StandardError(t *testing.T) {
	err := fmt.Errorf("standard error")
	result := Sprintc(err)

	if !strings.Contains(result, colorRed) {
		t.Error("expected color codes for standard error")
	}
	if !strings.Contains(result, "standard error") {
		t.Errorf("expected error message, got %q", result)
	}
}

func TestSprint_NilError(t *testing.T) {
	result := Sprint(nil)

	if result != "" {
		t.Errorf("expected empty string for nil, got %q", result)
	}
}

func TestSprintc_NilError(t *testing.T) {
	result := Sprintc(nil)

	if result != "" {
		t.Errorf("expected empty string for nil, got %q", result)
	}
}

// -----------------------------------------------------------------------------
// FormatMultiple Tests
// -----------------------------------------------------------------------------

func TestFormatMultiple_Empty(t *testing.T) {
	result := FormatMultiple([]error{})

	if result != "" {
		t.Errorf("expected empty string for empty slice, got %q", result)
	}
}

func TestFormatMultiple_SingleError(t *testing.T) {
	errs := []error{
		New("ERROR_1", CategoryConfig, "first error"),
	}
	result := FormatMultiple(errs)

	if !strings.Contains(result, "ERROR_1") {
		t.Errorf("expected error code, got %q", result)
	}
}

func TestFormatMultiple_MultipleErrors(t *testing.T) {
	errs := []error{
		New("ERROR_1", CategoryConfig, "first error"),
		New("ERROR_2", CategoryAgent, "second error"),
		New("ERROR_3", CategoryBackend, "third error"),
	}
	result := FormatMultiple(errs)

	// Check all errors are present
	if !strings.Contains(result, "ERROR_1") {
		t.Errorf("expected ERROR_1, got %q", result)
	}
	if !strings.Contains(result, "ERROR_2") {
		t.Errorf("expected ERROR_2, got %q", result)
	}
	if !strings.Contains(result, "ERROR_3") {
		t.Errorf("expected ERROR_3, got %q", result)
	}

	// Check they're separated
	parts := strings.Split(result, "\n\n")
	// At least should be able to distinguish between different error blocks
	if len(parts) < 2 {
		t.Errorf("expected errors to be separated, got %q", result)
	}
}

func TestFormatMultiple_WithNilErrors(t *testing.T) {
	errs := []error{
		New("ERROR_1", CategoryConfig, "first error"),
		nil, // should be skipped
		New("ERROR_2", CategoryAgent, "second error"),
	}
	result := FormatMultiple(errs)

	// Should contain both non-nil errors
	if !strings.Contains(result, "ERROR_1") {
		t.Errorf("expected ERROR_1, got %q", result)
	}
	if !strings.Contains(result, "ERROR_2") {
		t.Errorf("expected ERROR_2, got %q", result)
	}
}

func TestFormatMultiple_AllNil(t *testing.T) {
	errs := []error{nil, nil, nil}
	result := FormatMultiple(errs)

	if result != "" {
		t.Errorf("expected empty string for all-nil slice, got %q", result)
	}
}

func TestFormatMultiple_MixedErrorTypes(t *testing.T) {
	errs := []error{
		New("WEAVER_ERROR", CategoryConfig, "weaver error"),
		fmt.Errorf("standard error"),
	}
	result := FormatMultiple(errs)

	// Both should be formatted
	if !strings.Contains(result, "WEAVER_ERROR") {
		t.Errorf("expected weaver error, got %q", result)
	}
	if !strings.Contains(result, "standard error") {
		t.Errorf("expected standard error, got %q", result)
	}
}

// -----------------------------------------------------------------------------
// CategoryLabel Tests
// -----------------------------------------------------------------------------

func TestCategoryLabel(t *testing.T) {
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
		{Category("unknown"), "Error"}, // unknown category
	}

	for _, tt := range tests {
		t.Run(string(tt.category), func(t *testing.T) {
			if got := CategoryLabel(tt.category); got != tt.expected {
				t.Errorf("CategoryLabel(%q) = %q, want %q", tt.category, got, tt.expected)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// IsTTY Tests
// -----------------------------------------------------------------------------

func TestIsTTY_NilFile(t *testing.T) {
	if IsTTY(nil) {
		t.Error("IsTTY(nil) should return false")
	}
}

// Note: We can't easily test IsTTY with actual TTYs in unit tests,
// but we can verify it doesn't panic and handles edge cases.

// -----------------------------------------------------------------------------
// Top-level Function Tests
// -----------------------------------------------------------------------------

func TestFormat_TopLevel(t *testing.T) {
	we := New("TEST", CategoryConfig, "test message")
	result := Format(we)

	// Should return formatted string
	if !strings.Contains(result, "TEST") {
		t.Errorf("expected error code in output, got %q", result)
	}
}

// -----------------------------------------------------------------------------
// Edge Cases and Error Scenarios
// -----------------------------------------------------------------------------

func TestFormatter_Format_ErrorWithOnlyCause(t *testing.T) {
	we := New("ERROR_WITH_CAUSE", CategoryIO, "operation failed").
		WithCause(fmt.Errorf("underlying issue"))

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	if !strings.Contains(result, "cause:") {
		t.Errorf("expected cause in output, got %q", result)
	}
	if !strings.Contains(result, "underlying issue") {
		t.Errorf("expected cause message, got %q", result)
	}
}

func TestFormatter_Format_ErrorWithOnlySuggestions(t *testing.T) {
	we := New("ERROR_WITH_SUGGESTIONS", CategoryCommand, "command failed").
		WithSuggestions("Try option A", "Try option B")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	if !strings.Contains(result, "Try option A") {
		t.Errorf("expected first suggestion, got %q", result)
	}
	if !strings.Contains(result, "Try option B") {
		t.Errorf("expected second suggestion, got %q", result)
	}
}

func TestFormatter_Format_ErrorWithOnlyContext(t *testing.T) {
	we := New("ERROR_WITH_CONTEXT", CategoryValidation, "validation failed").
		WithContext("field", "email").
		WithContext("reason", "invalid format")

	f := &Formatter{UseColor: false, Indent: "  "}
	result := f.Format(we)

	if !strings.Contains(result, "field:") {
		t.Errorf("expected field context, got %q", result)
	}
	if !strings.Contains(result, "reason:") {
		t.Errorf("expected reason context, got %q", result)
	}
}

func TestFormatter_CustomIndent(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithContext("key", "value")

	f := &Formatter{UseColor: false, Indent: "    "} // 4 spaces
	result := f.Format(we)

	if !strings.Contains(result, "    key:") {
		t.Errorf("expected custom indent, got %q", result)
	}
}

func TestFormatter_EmptyIndent(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithContext("key", "value")

	f := &Formatter{UseColor: false, Indent: ""}
	result := f.Format(we)

	// Should still work with no indent
	if !strings.Contains(result, "key:") {
		t.Errorf("expected context without indent, got %q", result)
	}
}

// -----------------------------------------------------------------------------
// Suggestion Arrow Format Tests
// -----------------------------------------------------------------------------

func TestSuggestionArrowFormat(t *testing.T) {
	we := New("TEST", CategoryConfig, "test").
		WithSuggestion("Do something")

	// No color
	noColor := Sprint(we)
	if !strings.Contains(noColor, "→ Do something") {
		t.Errorf("expected arrow prefix in suggestion, got %q", noColor)
	}

	// With color
	withColor := Sprintc(we)
	if !strings.Contains(withColor, colorCyan) {
		t.Error("expected cyan color for suggestion")
	}
}

// -----------------------------------------------------------------------------
// Full Integration Scenarios
// -----------------------------------------------------------------------------

func TestIntegration_CompleteErrorDisplay(t *testing.T) {
	// Create a realistic error scenario
	we := ConfigError("CONFIG_PARSE_ERROR", "failed to parse configuration file").
		WithContext("file", "/home/user/.weaver/config.yaml").
		WithContext("line", "15").
		WithContext("column", "8").
		WithCause(fmt.Errorf("yaml: field 'agents' has invalid type: expected array, got string")).
		WithSuggestions(
			"Check the YAML syntax at line 15",
			"The 'agents' field should be a list, not a single value",
			"Example: agents: [agent1, agent2]",
		)

	// Test no-color output
	noColor := Sprint(we)

	// Verify all components are present
	mustContain := []string{
		"ERROR [CONFIG_PARSE_ERROR]:",
		"failed to parse configuration file",
		"file:",
		"/home/user/.weaver/config.yaml",
		"line:",
		"15",
		"cause:",
		"yaml:",
		"Check the YAML syntax",
		"agents",
		"Example:",
	}

	for _, s := range mustContain {
		if !strings.Contains(noColor, s) {
			t.Errorf("expected output to contain %q, got:\n%s", s, noColor)
		}
	}

	// Test color output
	withColor := Sprintc(we)

	// Verify color codes are present
	colorCodes := []string{colorRed, colorYellow, colorCyan, colorDim, colorReset}
	for _, code := range colorCodes {
		if !strings.Contains(withColor, code) {
			t.Errorf("expected output to contain color code %q", code)
		}
	}
}

func TestIntegration_NetworkErrorScenario(t *testing.T) {
	we := NetworkError("CONNECTION_TIMEOUT", "connection to backend timed out").
		WithContext("backend", "loom").
		WithContext("host", "localhost:8080").
		WithContext("timeout", "30s").
		WithCause(fmt.Errorf("dial tcp: connection refused")).
		WithSuggestion("Check if the Loom server is running").
		WithSuggestion("Verify the server address is correct").
		WithSuggestion("Try increasing the timeout with --timeout flag")

	result := Sprint(we)

	// Should have network error structure
	if !strings.Contains(result, "CONNECTION_TIMEOUT") {
		t.Error("expected error code")
	}
	if !strings.Contains(result, "connection to backend timed out") {
		t.Error("expected error message")
	}
	// Verify context is sorted
	if strings.Index(result, "backend:") > strings.Index(result, "host:") {
		t.Error("context should be sorted alphabetically")
	}
}

func TestIntegration_ValidationErrorScenario(t *testing.T) {
	we := ValidationError("INVALID_AGENT_CONFIG", "agent configuration is invalid").
		WithContext("agent", "myAgent").
		WithContext("field", "model").
		WithContext("value", "invalid-model-name").
		WithSuggestion("Valid models: gpt-4, claude-3, llama-2")

	result := Sprint(we)

	if !strings.Contains(result, "INVALID_AGENT_CONFIG") {
		t.Error("expected error code")
	}
	if !strings.Contains(result, "myAgent") {
		t.Error("expected agent name in context")
	}
	if !strings.Contains(result, "Valid models:") {
		t.Error("expected suggestion with valid options")
	}
}
