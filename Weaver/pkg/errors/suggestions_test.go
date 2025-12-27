package errors

import (
	"runtime"
	"testing"
)

// -----------------------------------------------------------------------------
// Suggestion Matching Tests
// -----------------------------------------------------------------------------

func TestSuggestion_Matches_NoConditions(t *testing.T) {
	s := Suggestion{Text: "test suggestion"}

	// No conditions should match any context
	if !s.Matches(nil) {
		t.Error("suggestion with no conditions should match nil context")
	}
	if !s.Matches(map[string]string{}) {
		t.Error("suggestion with no conditions should match empty context")
	}
	if !s.Matches(map[string]string{ContextOS: OSLinux}) {
		t.Error("suggestion with no conditions should match any context")
	}
}

func TestSuggestion_Matches_WithConditions(t *testing.T) {
	s := Suggestion{
		Text:       "Linux-specific suggestion",
		Conditions: map[string]string{ContextOS: OSLinux},
	}

	tests := []struct {
		name    string
		ctx     map[string]string
		want    bool
	}{
		{
			name: "matching condition",
			ctx:  map[string]string{ContextOS: OSLinux},
			want: true,
		},
		{
			name: "non-matching condition",
			ctx:  map[string]string{ContextOS: OSDarwin},
			want: false,
		},
		{
			name: "missing key in context",
			ctx:  map[string]string{ContextBackend: BackendClaudeCode},
			want: false,
		},
		{
			name: "nil context",
			ctx:  nil,
			want: false,
		},
		{
			name: "empty context",
			ctx:  map[string]string{},
			want: false,
		},
		{
			name: "matching with extra context",
			ctx:  map[string]string{ContextOS: OSLinux, ContextBackend: BackendLoom},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := s.Matches(tt.ctx); got != tt.want {
				t.Errorf("Matches(%v) = %v, want %v", tt.ctx, got, tt.want)
			}
		})
	}
}

func TestSuggestion_Matches_MultipleConditions(t *testing.T) {
	s := Suggestion{
		Text: "Claude on macOS suggestion",
		Conditions: map[string]string{
			ContextOS:      OSDarwin,
			ContextBackend: BackendClaudeCode,
		},
	}

	tests := []struct {
		name string
		ctx  map[string]string
		want bool
	}{
		{
			name: "all conditions match",
			ctx:  map[string]string{ContextOS: OSDarwin, ContextBackend: BackendClaudeCode},
			want: true,
		},
		{
			name: "only OS matches",
			ctx:  map[string]string{ContextOS: OSDarwin, ContextBackend: BackendLoom},
			want: false,
		},
		{
			name: "only backend matches",
			ctx:  map[string]string{ContextOS: OSLinux, ContextBackend: BackendClaudeCode},
			want: false,
		},
		{
			name: "neither matches",
			ctx:  map[string]string{ContextOS: OSLinux, ContextBackend: BackendLoom},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := s.Matches(tt.ctx); got != tt.want {
				t.Errorf("Matches(%v) = %v, want %v", tt.ctx, got, tt.want)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Registry Tests
// -----------------------------------------------------------------------------

func TestRegistry_Register(t *testing.T) {
	r := NewRegistry()
	r.Register("TEST_CODE", "First suggestion")
	r.Register("TEST_CODE", "Second suggestion")

	suggestions := r.Get("TEST_CODE", nil)
	if len(suggestions) != 2 {
		t.Errorf("expected 2 suggestions, got %d", len(suggestions))
	}
	if suggestions[0] != "First suggestion" {
		t.Errorf("expected 'First suggestion', got %q", suggestions[0])
	}
	if suggestions[1] != "Second suggestion" {
		t.Errorf("expected 'Second suggestion', got %q", suggestions[1])
	}
}

func TestRegistry_RegisterWithCondition(t *testing.T) {
	r := NewRegistry()
	r.RegisterWithCondition("TEST_CODE", "Linux fix", map[string]string{ContextOS: OSLinux})
	r.RegisterWithCondition("TEST_CODE", "macOS fix", map[string]string{ContextOS: OSDarwin})
	r.Register("TEST_CODE", "General fix")

	// Test Linux context
	linuxSuggestions := r.Get("TEST_CODE", map[string]string{ContextOS: OSLinux})
	if len(linuxSuggestions) != 2 {
		t.Errorf("expected 2 suggestions for Linux, got %d", len(linuxSuggestions))
	}
	containsLinux := false
	containsGeneral := false
	for _, s := range linuxSuggestions {
		if s == "Linux fix" {
			containsLinux = true
		}
		if s == "General fix" {
			containsGeneral = true
		}
	}
	if !containsLinux {
		t.Error("expected Linux-specific suggestion")
	}
	if !containsGeneral {
		t.Error("expected general suggestion")
	}

	// Test macOS context
	macSuggestions := r.Get("TEST_CODE", map[string]string{ContextOS: OSDarwin})
	if len(macSuggestions) != 2 {
		t.Errorf("expected 2 suggestions for macOS, got %d", len(macSuggestions))
	}
	containsMac := false
	for _, s := range macSuggestions {
		if s == "macOS fix" {
			containsMac = true
		}
	}
	if !containsMac {
		t.Error("expected macOS-specific suggestion")
	}

	// Test Windows context (only general)
	winSuggestions := r.Get("TEST_CODE", map[string]string{ContextOS: OSWindows})
	if len(winSuggestions) != 1 {
		t.Errorf("expected 1 suggestion for Windows, got %d", len(winSuggestions))
	}
	if winSuggestions[0] != "General fix" {
		t.Errorf("expected 'General fix', got %q", winSuggestions[0])
	}
}

func TestRegistry_RegisterWithPriority(t *testing.T) {
	r := NewRegistry()
	r.Register("TEST_CODE", "Low priority")
	r.RegisterWithPriority("TEST_CODE", "High priority", 10)
	r.RegisterWithPriority("TEST_CODE", "Medium priority", 5)

	suggestions := r.Get("TEST_CODE", nil)
	if len(suggestions) != 3 {
		t.Fatalf("expected 3 suggestions, got %d", len(suggestions))
	}

	// Should be sorted by priority (highest first)
	if suggestions[0] != "High priority" {
		t.Errorf("expected first suggestion 'High priority', got %q", suggestions[0])
	}
	if suggestions[1] != "Medium priority" {
		t.Errorf("expected second suggestion 'Medium priority', got %q", suggestions[1])
	}
	if suggestions[2] != "Low priority" {
		t.Errorf("expected third suggestion 'Low priority', got %q", suggestions[2])
	}
}

func TestRegistry_RegisterSuggestion(t *testing.T) {
	r := NewRegistry()
	r.RegisterSuggestion("TEST_CODE", Suggestion{
		Text:       "Complex suggestion",
		Conditions: map[string]string{ContextOS: OSLinux},
		Priority:   5,
	})

	// Should match Linux context
	suggestions := r.Get("TEST_CODE", map[string]string{ContextOS: OSLinux})
	if len(suggestions) != 1 {
		t.Errorf("expected 1 suggestion, got %d", len(suggestions))
	}

	// Should not match macOS context
	suggestions = r.Get("TEST_CODE", map[string]string{ContextOS: OSDarwin})
	if len(suggestions) != 0 {
		t.Errorf("expected 0 suggestions, got %d", len(suggestions))
	}
}

func TestRegistry_Get_NonexistentCode(t *testing.T) {
	r := NewRegistry()
	suggestions := r.Get("NONEXISTENT", nil)
	if suggestions != nil {
		t.Errorf("expected nil for nonexistent code, got %v", suggestions)
	}
}

func TestRegistry_GetAll(t *testing.T) {
	r := NewRegistry()
	r.Register("TEST_CODE", "Suggestion 1")
	r.RegisterWithCondition("TEST_CODE", "Suggestion 2", map[string]string{ContextOS: OSLinux})

	all := r.GetAll("TEST_CODE")
	if len(all) != 2 {
		t.Errorf("expected 2 suggestions, got %d", len(all))
	}
}

func TestRegistry_HasSuggestions(t *testing.T) {
	r := NewRegistry()

	if r.HasSuggestions("TEST_CODE") {
		t.Error("expected no suggestions for unregistered code")
	}

	r.Register("TEST_CODE", "A suggestion")

	if !r.HasSuggestions("TEST_CODE") {
		t.Error("expected suggestions for registered code")
	}
}

func TestRegistry_Codes(t *testing.T) {
	r := NewRegistry()
	r.Register("CODE_A", "Suggestion A")
	r.Register("CODE_B", "Suggestion B")
	r.Register("CODE_C", "Suggestion C")

	codes := r.Codes()
	if len(codes) != 3 {
		t.Errorf("expected 3 codes, got %d", len(codes))
	}

	// Check all codes are present (order may vary)
	codeMap := make(map[string]bool)
	for _, code := range codes {
		codeMap[code] = true
	}
	if !codeMap["CODE_A"] || !codeMap["CODE_B"] || !codeMap["CODE_C"] {
		t.Errorf("missing expected codes in %v", codes)
	}
}

func TestRegistry_ChainedRegistration(t *testing.T) {
	r := NewRegistry()
	r.Register("CODE_A", "A1").
		Register("CODE_A", "A2").
		Register("CODE_B", "B1")

	if len(r.Get("CODE_A", nil)) != 2 {
		t.Error("expected 2 suggestions for CODE_A")
	}
	if len(r.Get("CODE_B", nil)) != 1 {
		t.Error("expected 1 suggestion for CODE_B")
	}
}

// -----------------------------------------------------------------------------
// Platform Detection Tests
// -----------------------------------------------------------------------------

func TestCurrentOS(t *testing.T) {
	os := CurrentOS()
	if os != runtime.GOOS {
		t.Errorf("CurrentOS() = %q, want %q", os, runtime.GOOS)
	}
}

func TestCurrentArch(t *testing.T) {
	arch := CurrentArch()
	if arch != runtime.GOARCH {
		t.Errorf("CurrentArch() = %q, want %q", arch, runtime.GOARCH)
	}
}

func TestDefaultContext(t *testing.T) {
	ctx := DefaultContext()

	if ctx[ContextOS] != runtime.GOOS {
		t.Errorf("DefaultContext()[ContextOS] = %q, want %q", ctx[ContextOS], runtime.GOOS)
	}
	if ctx[ContextArch] != runtime.GOARCH {
		t.Errorf("DefaultContext()[ContextArch] = %q, want %q", ctx[ContextArch], runtime.GOARCH)
	}
}

func TestMergeContext(t *testing.T) {
	ctx1 := map[string]string{"a": "1", "b": "2"}
	ctx2 := map[string]string{"b": "3", "c": "4"}

	merged := MergeContext(ctx1, ctx2)

	expected := map[string]string{"a": "1", "b": "3", "c": "4"}
	if len(merged) != len(expected) {
		t.Errorf("expected %d keys, got %d", len(expected), len(merged))
	}
	for k, v := range expected {
		if merged[k] != v {
			t.Errorf("merged[%q] = %q, want %q", k, merged[k], v)
		}
	}
}

func TestMergeContext_Empty(t *testing.T) {
	merged := MergeContext()
	if len(merged) != 0 {
		t.Errorf("expected empty map, got %v", merged)
	}

	merged = MergeContext(nil, nil)
	if len(merged) != 0 {
		t.Errorf("expected empty map, got %v", merged)
	}
}

// -----------------------------------------------------------------------------
// Default Registry Tests
// -----------------------------------------------------------------------------

func TestDefaultRegistry_HasConfigSuggestions(t *testing.T) {
	configCodes := []string{
		ErrConfigNotFound,
		ErrConfigParseFailed,
		ErrConfigInvalid,
		ErrConfigInitFailed,
		ErrConfigReadFailed,
		ErrConfigWriteFailed,
	}

	for _, code := range configCodes {
		if !defaultRegistry.HasSuggestions(code) {
			t.Errorf("expected suggestions for %s", code)
		}
	}
}

func TestDefaultRegistry_HasBackendSuggestions(t *testing.T) {
	backendCodes := []string{
		ErrBackendUnavailable,
		ErrBackendNotFound,
		ErrBackendConnectionFailed,
		ErrBackendTimeout,
		ErrBackendAPIError,
		ErrBackendAuthFailed,
		ErrBackendNotInstalled,
		ErrBackendStreamFailed,
		ErrBackendAlreadyRegistered,
	}

	for _, code := range backendCodes {
		if !defaultRegistry.HasSuggestions(code) {
			t.Errorf("expected suggestions for %s", code)
		}
	}
}

func TestDefaultRegistry_HasAgentSuggestions(t *testing.T) {
	agentCodes := []string{
		ErrAgentNotFound,
		ErrAgentAlreadyExists,
		ErrAgentCreationFailed,
		ErrAgentNotReady,
		ErrAgentChatFailed,
		ErrAgentInvalidConfig,
		ErrAgentNoHiddenState,
	}

	for _, code := range agentCodes {
		if !defaultRegistry.HasSuggestions(code) {
			t.Errorf("expected suggestions for %s", code)
		}
	}
}

func TestDefaultRegistry_OSSpecificSuggestions(t *testing.T) {
	// Test that OS-specific suggestions are properly filtered
	linuxCtx := map[string]string{ContextOS: OSLinux}
	darwinCtx := map[string]string{ContextOS: OSDarwin}
	windowsCtx := map[string]string{ContextOS: OSWindows}

	// CONFIG_NOT_FOUND should have macOS-specific suggestion
	darwinSuggestions := defaultRegistry.Get(ErrConfigNotFound, darwinCtx)
	linuxSuggestions := defaultRegistry.Get(ErrConfigNotFound, linuxCtx)

	foundMacSpecific := false
	for _, s := range darwinSuggestions {
		if contains(s, "Library/Application Support") {
			foundMacSpecific = true
			break
		}
	}
	if !foundMacSpecific {
		t.Error("expected macOS-specific config path suggestion")
	}

	// Linux should not have the macOS suggestion
	for _, s := range linuxSuggestions {
		if contains(s, "Library/Application Support") {
			t.Error("Linux should not have macOS-specific suggestion")
		}
	}

	// IO_PERMISSION_DENIED should have different suggestions per OS
	linuxPerm := defaultRegistry.Get(ErrIOPermissionDenied, linuxCtx)
	windowsPerm := defaultRegistry.Get(ErrIOPermissionDenied, windowsCtx)

	foundSudo := false
	for _, s := range linuxPerm {
		if contains(s, "sudo") {
			foundSudo = true
			break
		}
	}
	if !foundSudo {
		t.Error("expected sudo suggestion for Linux")
	}

	foundAdmin := false
	for _, s := range windowsPerm {
		if contains(s, "Administrator") {
			foundAdmin = true
			break
		}
	}
	if !foundAdmin {
		t.Error("expected Administrator suggestion for Windows")
	}
}

func TestDefaultRegistry_BackendSpecificSuggestions(t *testing.T) {
	claudeCtx := map[string]string{ContextBackend: BackendClaudeCode}
	loomCtx := map[string]string{ContextBackend: BackendLoom}

	// BACKEND_NOT_INSTALLED should have different suggestions per backend
	claudeSuggestions := defaultRegistry.Get(ErrBackendNotInstalled, claudeCtx)
	loomSuggestions := defaultRegistry.Get(ErrBackendNotInstalled, loomCtx)

	foundClaudeCLI := false
	for _, s := range claudeSuggestions {
		if contains(s, "claude-cli") || contains(s, "npm install") {
			foundClaudeCLI = true
			break
		}
	}
	if !foundClaudeCLI {
		t.Error("expected Claude CLI installation suggestion")
	}

	foundLoom := false
	for _, s := range loomSuggestions {
		if contains(s, "Loom") {
			foundLoom = true
			break
		}
	}
	if !foundLoom {
		t.Error("expected Loom-specific suggestion")
	}
}

// -----------------------------------------------------------------------------
// GetSuggestions API Tests
// -----------------------------------------------------------------------------

func TestGetSuggestions(t *testing.T) {
	suggestions := GetSuggestions(ErrConfigNotFound)
	if len(suggestions) == 0 {
		t.Error("expected suggestions for CONFIG_NOT_FOUND")
	}
}

func TestGetSuggestionsWithContext(t *testing.T) {
	ctx := map[string]string{
		ContextOS:      OSLinux,
		ContextBackend: BackendClaudeCode,
	}
	suggestions := GetSuggestionsWithContext(ErrBackendNotInstalled, ctx)
	if len(suggestions) == 0 {
		t.Error("expected suggestions for BACKEND_NOT_INSTALLED with context")
	}
}

// -----------------------------------------------------------------------------
// AttachSuggestions Tests
// -----------------------------------------------------------------------------

func TestAttachSuggestions(t *testing.T) {
	err := New(ErrConfigNotFound, CategoryConfig, "Config file not found")

	// Should have no suggestions initially
	if len(err.Suggestions) != 0 {
		t.Errorf("expected 0 initial suggestions, got %d", len(err.Suggestions))
	}

	// Attach suggestions
	AttachSuggestions(err)

	// Should now have suggestions
	if len(err.Suggestions) == 0 {
		t.Error("expected suggestions after AttachSuggestions")
	}
}

func TestAttachSuggestions_NilError(t *testing.T) {
	result := AttachSuggestions(nil)
	if result != nil {
		t.Error("AttachSuggestions(nil) should return nil")
	}
}

func TestAttachSuggestions_WithContext(t *testing.T) {
	err := New(ErrBackendNotInstalled, CategoryBackend, "Backend not installed").
		WithContext(ContextBackend, BackendClaudeCode)

	AttachSuggestions(err)

	// Should have Claude-specific suggestions
	foundClaudeCLI := false
	for _, s := range err.Suggestions {
		if contains(s, "claude-cli") || contains(s, "npm install") {
			foundClaudeCLI = true
			break
		}
	}
	if !foundClaudeCLI {
		t.Error("expected Claude-specific suggestion when backend context is set")
	}
}

func TestAttachSuggestions_PreservesExisting(t *testing.T) {
	err := New(ErrConfigNotFound, CategoryConfig, "Config file not found").
		WithSuggestion("Custom suggestion")

	AttachSuggestions(err)

	// Should have both custom and registry suggestions
	foundCustom := false
	for _, s := range err.Suggestions {
		if s == "Custom suggestion" {
			foundCustom = true
			break
		}
	}
	if !foundCustom {
		t.Error("AttachSuggestions should preserve existing suggestions")
	}

	// Should have more than just the custom suggestion
	if len(err.Suggestions) <= 1 {
		t.Error("expected registry suggestions to be added")
	}
}

func TestNewWithSuggestions(t *testing.T) {
	err := NewWithSuggestions(ErrConfigNotFound, CategoryConfig, "Config file not found")

	if len(err.Suggestions) == 0 {
		t.Error("NewWithSuggestions should attach suggestions")
	}
}

func TestWrapWithSuggestions(t *testing.T) {
	cause := New(ErrIOFileNotFound, CategoryIO, "file not found")
	err := WrapWithSuggestions(cause, ErrConfigReadFailed, CategoryConfig, "Failed to read config")

	if len(err.Suggestions) == 0 {
		t.Error("WrapWithSuggestions should attach suggestions")
	}
	if err.Cause != cause {
		t.Error("WrapWithSuggestions should set cause")
	}
}

// -----------------------------------------------------------------------------
// FormatSuggestionList Tests
// -----------------------------------------------------------------------------

func TestFormatSuggestionList(t *testing.T) {
	suggestions := []string{"First suggestion", "Second suggestion"}
	formatted := FormatSuggestionList(suggestions)

	expected := "→ First suggestion\n→ Second suggestion"
	if formatted != expected {
		t.Errorf("FormatSuggestionList = %q, want %q", formatted, expected)
	}
}

func TestFormatSuggestionList_Empty(t *testing.T) {
	formatted := FormatSuggestionList(nil)
	if formatted != "" {
		t.Errorf("FormatSuggestionList(nil) = %q, want empty string", formatted)
	}

	formatted = FormatSuggestionList([]string{})
	if formatted != "" {
		t.Errorf("FormatSuggestionList([]) = %q, want empty string", formatted)
	}
}

func TestFormatSuggestionList_Single(t *testing.T) {
	formatted := FormatSuggestionList([]string{"Only suggestion"})
	expected := "→ Only suggestion"
	if formatted != expected {
		t.Errorf("FormatSuggestionList = %q, want %q", formatted, expected)
	}
}

// -----------------------------------------------------------------------------
// Integration Tests
// -----------------------------------------------------------------------------

func TestSuggestions_IntegrationWithErrorDisplay(t *testing.T) {
	// Create an error, attach suggestions, and format it
	err := New(ErrBackendNotInstalled, CategoryBackend, "Claude CLI not found").
		WithContext(ContextBackend, BackendClaudeCode).
		WithContext(ContextOS, OSDarwin)

	AttachSuggestions(err)

	// Format the error
	formatted := Sprint(err)

	// Should contain the error message
	if !contains(formatted, "Claude CLI not found") {
		t.Error("formatted output should contain error message")
	}

	// Should contain suggestions
	if !contains(formatted, "→") {
		t.Error("formatted output should contain suggestion arrows")
	}
}

func TestSuggestions_AllErrorCodesHaveSuggestions(t *testing.T) {
	// List of all error codes that should have suggestions
	codes := []string{
		// Config
		ErrConfigNotFound, ErrConfigParseFailed, ErrConfigInvalid,
		ErrConfigInitFailed, ErrConfigReadFailed, ErrConfigWriteFailed,
		// Backend
		ErrBackendUnavailable, ErrBackendNotFound, ErrBackendConnectionFailed,
		ErrBackendAlreadyRegistered, ErrBackendTimeout, ErrBackendAPIError,
		ErrBackendAuthFailed, ErrBackendNotInstalled, ErrBackendStreamFailed,
		// Agent
		ErrAgentNotFound, ErrAgentAlreadyExists, ErrAgentCreationFailed,
		ErrAgentNotReady, ErrAgentChatFailed, ErrAgentInvalidConfig,
		ErrAgentNoHiddenState,
		// Command
		ErrCommandInvalidSyntax, ErrCommandMissingArgs, ErrCommandInvalidArg,
		ErrCommandNotFound, ErrCommandExecutionFailed, ErrCommandEmptyInput,
		// Validation
		ErrValidationRequired, ErrValidationInvalidValue, ErrValidationOutOfRange,
		ErrValidationTypeMismatch, ErrValidationInvalidFormat,
		// Network
		ErrNetworkTimeout, ErrNetworkConnectionRefused, ErrNetworkDNSFailed,
		ErrNetworkUnreachable, ErrNetworkTLSFailed,
		// IO
		ErrIOReadFailed, ErrIOWriteFailed, ErrIOPermissionDenied,
		ErrIOFileNotFound, ErrIODirNotFound, ErrIODiskFull,
		ErrIOMarshalFailed, ErrIOUnmarshalFailed,
		// Internal
		ErrInternalError, ErrInternalInvariantViolation,
		ErrInternalNilPointer, ErrInternalPanic,
		// Concepts
		ErrConceptsNoHiddenState, ErrConceptsInsufficientSamples,
		ErrConceptsNotFound, ErrConceptsExtractionFailed,
		ErrAnalysisFailed, ErrAnalysisServerUnavailable, ErrAnalysisInvalidResponse,
		// Session
		ErrSessionNotFound, ErrSessionExportFailed, ErrSessionLoadFailed,
		// Shell
		ErrShellInitFailed, ErrShellHistoryFailed, ErrShellReadlineFailed,
	}

	for _, code := range codes {
		if !defaultRegistry.HasSuggestions(code) {
			t.Errorf("missing suggestions for error code: %s", code)
		}
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		len(s) > 0 && len(substr) > 0 &&
			findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
