package errors

import "testing"

// TestErrorCodeConstants verifies that all error code constants are non-empty.
func TestErrorCodeConstants(t *testing.T) {
	// Configuration error codes
	configCodes := []string{
		ErrConfigNotFound,
		ErrConfigParseFailed,
		ErrConfigInvalid,
		ErrConfigInitFailed,
		ErrConfigReadFailed,
		ErrConfigWriteFailed,
	}
	for _, code := range configCodes {
		if code == "" {
			t.Error("Config error code should not be empty")
		}
	}

	// Backend error codes
	backendCodes := []string{
		ErrBackendUnavailable,
		ErrBackendNotFound,
		ErrBackendConnectionFailed,
		ErrBackendAlreadyRegistered,
		ErrBackendTimeout,
		ErrBackendAPIError,
		ErrBackendAuthFailed,
		ErrBackendNotInstalled,
		ErrBackendStreamFailed,
	}
	for _, code := range backendCodes {
		if code == "" {
			t.Error("Backend error code should not be empty")
		}
	}

	// Agent error codes
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
		if code == "" {
			t.Error("Agent error code should not be empty")
		}
	}

	// Command error codes
	commandCodes := []string{
		ErrCommandInvalidSyntax,
		ErrCommandMissingArgs,
		ErrCommandInvalidArg,
		ErrCommandNotFound,
		ErrCommandExecutionFailed,
		ErrCommandEmptyInput,
	}
	for _, code := range commandCodes {
		if code == "" {
			t.Error("Command error code should not be empty")
		}
	}

	// Validation error codes
	validationCodes := []string{
		ErrValidationRequired,
		ErrValidationInvalidValue,
		ErrValidationOutOfRange,
		ErrValidationTypeMismatch,
		ErrValidationInvalidFormat,
	}
	for _, code := range validationCodes {
		if code == "" {
			t.Error("Validation error code should not be empty")
		}
	}

	// Network error codes
	networkCodes := []string{
		ErrNetworkTimeout,
		ErrNetworkConnectionRefused,
		ErrNetworkDNSFailed,
		ErrNetworkUnreachable,
		ErrNetworkTLSFailed,
	}
	for _, code := range networkCodes {
		if code == "" {
			t.Error("Network error code should not be empty")
		}
	}

	// I/O error codes
	ioCodes := []string{
		ErrIOReadFailed,
		ErrIOWriteFailed,
		ErrIOPermissionDenied,
		ErrIOFileNotFound,
		ErrIODirNotFound,
		ErrIODiskFull,
		ErrIOMarshalFailed,
		ErrIOUnmarshalFailed,
	}
	for _, code := range ioCodes {
		if code == "" {
			t.Error("I/O error code should not be empty")
		}
	}

	// Internal error codes
	internalCodes := []string{
		ErrInternalError,
		ErrInternalInvariantViolation,
		ErrInternalNilPointer,
		ErrInternalPanic,
	}
	for _, code := range internalCodes {
		if code == "" {
			t.Error("Internal error code should not be empty")
		}
	}

	// Concepts/Analysis error codes
	conceptsCodes := []string{
		ErrConceptsNoHiddenState,
		ErrConceptsInsufficientSamples,
		ErrConceptsNotFound,
		ErrConceptsExtractionFailed,
		ErrAnalysisFailed,
		ErrAnalysisServerUnavailable,
		ErrAnalysisInvalidResponse,
	}
	for _, code := range conceptsCodes {
		if code == "" {
			t.Error("Concepts/Analysis error code should not be empty")
		}
	}

	// Session error codes
	sessionCodes := []string{
		ErrSessionNotFound,
		ErrSessionExportFailed,
		ErrSessionLoadFailed,
	}
	for _, code := range sessionCodes {
		if code == "" {
			t.Error("Session error code should not be empty")
		}
	}

	// Shell error codes
	shellCodes := []string{
		ErrShellInitFailed,
		ErrShellHistoryFailed,
		ErrShellReadlineFailed,
	}
	for _, code := range shellCodes {
		if code == "" {
			t.Error("Shell error code should not be empty")
		}
	}
}

// TestCodeCategory verifies that CodeCategory returns correct categories.
func TestCodeCategory(t *testing.T) {
	tests := []struct {
		code     string
		expected Category
	}{
		// Config codes
		{ErrConfigNotFound, CategoryConfig},
		{ErrConfigParseFailed, CategoryConfig},
		{ErrConfigInvalid, CategoryConfig},
		{ErrConfigInitFailed, CategoryConfig},
		{ErrConfigReadFailed, CategoryConfig},
		{ErrConfigWriteFailed, CategoryConfig},

		// Backend codes
		{ErrBackendUnavailable, CategoryBackend},
		{ErrBackendNotFound, CategoryBackend},
		{ErrBackendConnectionFailed, CategoryBackend},
		{ErrBackendAlreadyRegistered, CategoryBackend},
		{ErrBackendTimeout, CategoryBackend},
		{ErrBackendAPIError, CategoryBackend},
		{ErrBackendAuthFailed, CategoryBackend},
		{ErrBackendNotInstalled, CategoryBackend},
		{ErrBackendStreamFailed, CategoryBackend},

		// Agent codes
		{ErrAgentNotFound, CategoryAgent},
		{ErrAgentAlreadyExists, CategoryAgent},
		{ErrAgentCreationFailed, CategoryAgent},
		{ErrAgentNotReady, CategoryAgent},
		{ErrAgentChatFailed, CategoryAgent},
		{ErrAgentInvalidConfig, CategoryAgent},
		{ErrAgentNoHiddenState, CategoryAgent},

		// Command codes
		{ErrCommandInvalidSyntax, CategoryCommand},
		{ErrCommandMissingArgs, CategoryCommand},
		{ErrCommandInvalidArg, CategoryCommand},
		{ErrCommandNotFound, CategoryCommand},
		{ErrCommandExecutionFailed, CategoryCommand},
		{ErrCommandEmptyInput, CategoryCommand},

		// Validation codes
		{ErrValidationRequired, CategoryValidation},
		{ErrValidationInvalidValue, CategoryValidation},
		{ErrValidationOutOfRange, CategoryValidation},
		{ErrValidationTypeMismatch, CategoryValidation},
		{ErrValidationInvalidFormat, CategoryValidation},

		// Network codes
		{ErrNetworkTimeout, CategoryNetwork},
		{ErrNetworkConnectionRefused, CategoryNetwork},
		{ErrNetworkDNSFailed, CategoryNetwork},
		{ErrNetworkUnreachable, CategoryNetwork},
		{ErrNetworkTLSFailed, CategoryNetwork},

		// IO codes
		{ErrIOReadFailed, CategoryIO},
		{ErrIOWriteFailed, CategoryIO},
		{ErrIOPermissionDenied, CategoryIO},
		{ErrIOFileNotFound, CategoryIO},
		{ErrIODirNotFound, CategoryIO},
		{ErrIODiskFull, CategoryIO},
		{ErrIOMarshalFailed, CategoryIO},
		{ErrIOUnmarshalFailed, CategoryIO},

		// Internal codes
		{ErrInternalError, CategoryInternal},
		{ErrInternalInvariantViolation, CategoryInternal},
		{ErrInternalNilPointer, CategoryInternal},
		{ErrInternalPanic, CategoryInternal},

		// Unknown code should return CategoryInternal
		{"UNKNOWN_CODE", CategoryInternal},
		{"", CategoryInternal},
	}

	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			got := CodeCategory(tt.code)
			if got != tt.expected {
				t.Errorf("CodeCategory(%q) = %v, want %v", tt.code, got, tt.expected)
			}
		})
	}
}

// TestCategoryHelpers verifies the category helper functions.
func TestCategoryHelpers(t *testing.T) {
	// Test IsConfigCode
	if !IsConfigCode(ErrConfigNotFound) {
		t.Error("IsConfigCode should return true for ErrConfigNotFound")
	}
	if IsConfigCode(ErrAgentNotFound) {
		t.Error("IsConfigCode should return false for ErrAgentNotFound")
	}

	// Test IsBackendCode
	if !IsBackendCode(ErrBackendUnavailable) {
		t.Error("IsBackendCode should return true for ErrBackendUnavailable")
	}
	if IsBackendCode(ErrAgentNotFound) {
		t.Error("IsBackendCode should return false for ErrAgentNotFound")
	}

	// Test IsAgentCode
	if !IsAgentCode(ErrAgentNotFound) {
		t.Error("IsAgentCode should return true for ErrAgentNotFound")
	}
	if IsAgentCode(ErrBackendNotFound) {
		t.Error("IsAgentCode should return false for ErrBackendNotFound")
	}

	// Test IsCommandCode
	if !IsCommandCode(ErrCommandInvalidSyntax) {
		t.Error("IsCommandCode should return true for ErrCommandInvalidSyntax")
	}
	if IsCommandCode(ErrAgentNotFound) {
		t.Error("IsCommandCode should return false for ErrAgentNotFound")
	}

	// Test IsValidationCode
	if !IsValidationCode(ErrValidationRequired) {
		t.Error("IsValidationCode should return true for ErrValidationRequired")
	}
	if IsValidationCode(ErrAgentNotFound) {
		t.Error("IsValidationCode should return false for ErrAgentNotFound")
	}

	// Test IsNetworkCode
	if !IsNetworkCode(ErrNetworkTimeout) {
		t.Error("IsNetworkCode should return true for ErrNetworkTimeout")
	}
	if IsNetworkCode(ErrAgentNotFound) {
		t.Error("IsNetworkCode should return false for ErrAgentNotFound")
	}

	// Test IsIOCode
	if !IsIOCode(ErrIOReadFailed) {
		t.Error("IsIOCode should return true for ErrIOReadFailed")
	}
	if IsIOCode(ErrAgentNotFound) {
		t.Error("IsIOCode should return false for ErrAgentNotFound")
	}

	// Test IsInternalCode
	if !IsInternalCode(ErrInternalError) {
		t.Error("IsInternalCode should return true for ErrInternalError")
	}
	if IsInternalCode(ErrAgentNotFound) {
		t.Error("IsInternalCode should return false for ErrAgentNotFound")
	}
}

// TestCodeUniqueness ensures all error codes are unique.
func TestCodeUniqueness(t *testing.T) {
	allCodes := []string{
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
		// Concepts/Analysis
		ErrConceptsNoHiddenState, ErrConceptsInsufficientSamples,
		ErrConceptsNotFound, ErrConceptsExtractionFailed,
		ErrAnalysisFailed, ErrAnalysisServerUnavailable, ErrAnalysisInvalidResponse,
		// Session
		ErrSessionNotFound, ErrSessionExportFailed, ErrSessionLoadFailed,
		// Shell
		ErrShellInitFailed, ErrShellHistoryFailed, ErrShellReadlineFailed,
	}

	seen := make(map[string]bool)
	for _, code := range allCodes {
		if seen[code] {
			t.Errorf("Duplicate error code: %s", code)
		}
		seen[code] = true
	}
}

// TestCodeFormat ensures all error codes follow the expected format.
func TestCodeFormat(t *testing.T) {
	allCodes := []string{
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
		// Concepts/Analysis
		ErrConceptsNoHiddenState, ErrConceptsInsufficientSamples,
		ErrConceptsNotFound, ErrConceptsExtractionFailed,
		ErrAnalysisFailed, ErrAnalysisServerUnavailable, ErrAnalysisInvalidResponse,
		// Session
		ErrSessionNotFound, ErrSessionExportFailed, ErrSessionLoadFailed,
		// Shell
		ErrShellInitFailed, ErrShellHistoryFailed, ErrShellReadlineFailed,
	}

	for _, code := range allCodes {
		// Codes should be UPPER_SNAKE_CASE
		for _, c := range code {
			if !((c >= 'A' && c <= 'Z') || c == '_') {
				t.Errorf("Error code %q contains invalid character %q (expected UPPER_SNAKE_CASE)", code, string(c))
				break
			}
		}

		// Codes should not start or end with underscore
		if len(code) > 0 && (code[0] == '_' || code[len(code)-1] == '_') {
			t.Errorf("Error code %q should not start or end with underscore", code)
		}
	}
}

// TestErrorCodesWithWeaverError verifies codes work with WeaverError.
func TestErrorCodesWithWeaverError(t *testing.T) {
	// Test creating a WeaverError with a defined code
	err := New(ErrConfigNotFound, CategoryConfig, "configuration file not found")

	if err.Code != ErrConfigNotFound {
		t.Errorf("Expected code %q, got %q", ErrConfigNotFound, err.Code)
	}
	if err.Category != CategoryConfig {
		t.Errorf("Expected category %q, got %q", CategoryConfig, err.Category)
	}

	// Test IsCode function
	if !IsCode(err, ErrConfigNotFound) {
		t.Error("IsCode should return true for matching code")
	}
	if IsCode(err, ErrConfigParseFailed) {
		t.Error("IsCode should return false for non-matching code")
	}

	// Test IsCategory function
	if !IsCategory(err, CategoryConfig) {
		t.Error("IsCategory should return true for matching category")
	}
	if IsCategory(err, CategoryBackend) {
		t.Error("IsCategory should return false for non-matching category")
	}
}

// TestErrorCodesIntegration verifies codes integrate well with error helpers.
func TestErrorCodesIntegration(t *testing.T) {
	// Test ConfigError helper
	err := ConfigError(ErrConfigNotFound, "config file not found")
	if err.Code != ErrConfigNotFound {
		t.Errorf("Expected code %q, got %q", ErrConfigNotFound, err.Code)
	}
	if err.Category != CategoryConfig {
		t.Errorf("Expected category %q, got %q", CategoryConfig, err.Category)
	}

	// Test BackendError helper
	err = BackendError(ErrBackendUnavailable, "no backends available")
	if err.Code != ErrBackendUnavailable {
		t.Errorf("Expected code %q, got %q", ErrBackendUnavailable, err.Code)
	}
	if err.Category != CategoryBackend {
		t.Errorf("Expected category %q, got %q", CategoryBackend, err.Category)
	}

	// Test AgentError helper
	err = AgentError(ErrAgentNotFound, "agent not found")
	if err.Code != ErrAgentNotFound {
		t.Errorf("Expected code %q, got %q", ErrAgentNotFound, err.Code)
	}
	if err.Category != CategoryAgent {
		t.Errorf("Expected category %q, got %q", CategoryAgent, err.Category)
	}

	// Test CommandError helper
	err = CommandError(ErrCommandInvalidSyntax, "invalid command syntax")
	if err.Code != ErrCommandInvalidSyntax {
		t.Errorf("Expected code %q, got %q", ErrCommandInvalidSyntax, err.Code)
	}
	if err.Category != CategoryCommand {
		t.Errorf("Expected category %q, got %q", CategoryCommand, err.Category)
	}

	// Test ValidationError helper
	err = ValidationError(ErrValidationRequired, "required field missing")
	if err.Code != ErrValidationRequired {
		t.Errorf("Expected code %q, got %q", ErrValidationRequired, err.Code)
	}
	if err.Category != CategoryValidation {
		t.Errorf("Expected category %q, got %q", CategoryValidation, err.Category)
	}

	// Test NetworkError helper
	err = NetworkError(ErrNetworkTimeout, "connection timed out")
	if err.Code != ErrNetworkTimeout {
		t.Errorf("Expected code %q, got %q", ErrNetworkTimeout, err.Code)
	}
	if err.Category != CategoryNetwork {
		t.Errorf("Expected category %q, got %q", CategoryNetwork, err.Category)
	}

	// Test IOError helper
	err = IOError(ErrIOReadFailed, "failed to read file")
	if err.Code != ErrIOReadFailed {
		t.Errorf("Expected code %q, got %q", ErrIOReadFailed, err.Code)
	}
	if err.Category != CategoryIO {
		t.Errorf("Expected category %q, got %q", CategoryIO, err.Category)
	}

	// Test InternalError helper
	err = InternalError(ErrInternalError, "unexpected error")
	if err.Code != ErrInternalError {
		t.Errorf("Expected code %q, got %q", ErrInternalError, err.Code)
	}
	if err.Category != CategoryInternal {
		t.Errorf("Expected category %q, got %q", CategoryInternal, err.Category)
	}
}
