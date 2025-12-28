// Package errors provides error code constants for Weaver.
// Error codes are organized by category for consistent handling and lookup.
package errors

// -----------------------------------------------------------------------------
// Configuration Error Codes
// -----------------------------------------------------------------------------
// Use these codes for errors related to config file loading, parsing,
// and validation.

const (
	// ErrConfigNotFound indicates the configuration file does not exist.
	ErrConfigNotFound = "CONFIG_NOT_FOUND"

	// ErrConfigParseFailed indicates the configuration file could not be parsed.
	// Usually a YAML syntax error or invalid structure.
	ErrConfigParseFailed = "CONFIG_PARSE_FAILED"

	// ErrConfigInvalid indicates configuration values are invalid.
	// Field values don't meet validation requirements.
	ErrConfigInvalid = "CONFIG_INVALID"

	// ErrConfigInitFailed indicates config initialization failed.
	// Unable to create config file or directory.
	ErrConfigInitFailed = "CONFIG_INIT_FAILED"

	// ErrConfigReadFailed indicates the config file could not be read.
	// File exists but is not readable (permissions, etc).
	ErrConfigReadFailed = "CONFIG_READ_FAILED"

	// ErrConfigWriteFailed indicates the config file could not be written.
	ErrConfigWriteFailed = "CONFIG_WRITE_FAILED"
)

// -----------------------------------------------------------------------------
// Backend Error Codes
// -----------------------------------------------------------------------------
// Use these codes for errors related to backend communication and availability.

const (
	// ErrBackendUnavailable indicates no backends are available.
	// All configured backends are unreachable or disabled.
	ErrBackendUnavailable = "BACKEND_UNAVAILABLE"

	// ErrBackendNotFound indicates the requested backend is not registered.
	ErrBackendNotFound = "BACKEND_NOT_FOUND"

	// ErrBackendConnectionFailed indicates a connection to the backend failed.
	ErrBackendConnectionFailed = "BACKEND_CONNECTION_FAILED"

	// ErrBackendAlreadyRegistered indicates a backend with this name already exists.
	ErrBackendAlreadyRegistered = "BACKEND_ALREADY_REGISTERED"

	// ErrBackendTimeout indicates a backend request timed out.
	ErrBackendTimeout = "BACKEND_TIMEOUT"

	// ErrBackendAPIError indicates the backend API returned an error.
	ErrBackendAPIError = "BACKEND_API_ERROR"

	// ErrBackendAuthFailed indicates authentication with the backend failed.
	ErrBackendAuthFailed = "BACKEND_AUTH_FAILED"

	// ErrBackendNotInstalled indicates the backend CLI/service is not installed.
	ErrBackendNotInstalled = "BACKEND_NOT_INSTALLED"

	// ErrBackendStreamFailed indicates streaming from the backend failed.
	ErrBackendStreamFailed = "BACKEND_STREAM_FAILED"
)

// -----------------------------------------------------------------------------
// Agent Error Codes
// -----------------------------------------------------------------------------
// Use these codes for errors related to agent creation and runtime.

const (
	// ErrAgentNotFound indicates the requested agent does not exist.
	ErrAgentNotFound = "AGENT_NOT_FOUND"

	// ErrAgentAlreadyExists indicates an agent with this name already exists.
	ErrAgentAlreadyExists = "AGENT_ALREADY_EXISTS"

	// ErrAgentCreationFailed indicates agent creation failed.
	ErrAgentCreationFailed = "AGENT_CREATION_FAILED"

	// ErrAgentNotReady indicates the agent is not ready for requests.
	ErrAgentNotReady = "AGENT_NOT_READY"

	// ErrAgentChatFailed indicates a chat request to the agent failed.
	ErrAgentChatFailed = "AGENT_CHAT_FAILED"

	// ErrAgentInvalidConfig indicates the agent configuration is invalid.
	ErrAgentInvalidConfig = "AGENT_INVALID_CONFIG"

	// ErrAgentNoHiddenState indicates the agent doesn't support hidden state.
	ErrAgentNoHiddenState = "AGENT_NO_HIDDEN_STATE"
)

// -----------------------------------------------------------------------------
// Command Error Codes
// -----------------------------------------------------------------------------
// Use these codes for errors related to shell command parsing and execution.

const (
	// ErrCommandInvalidSyntax indicates the command has invalid syntax.
	ErrCommandInvalidSyntax = "COMMAND_INVALID_SYNTAX"

	// ErrCommandMissingArgs indicates required arguments are missing.
	ErrCommandMissingArgs = "COMMAND_MISSING_ARGS"

	// ErrCommandInvalidArg indicates an argument value is invalid.
	ErrCommandInvalidArg = "COMMAND_INVALID_ARG"

	// ErrCommandNotFound indicates the command does not exist.
	ErrCommandNotFound = "COMMAND_NOT_FOUND"

	// ErrCommandExecutionFailed indicates command execution failed.
	ErrCommandExecutionFailed = "COMMAND_EXECUTION_FAILED"

	// ErrCommandEmptyInput indicates no input was provided.
	ErrCommandEmptyInput = "COMMAND_EMPTY_INPUT"
)

// -----------------------------------------------------------------------------
// Validation Error Codes
// -----------------------------------------------------------------------------
// Use these codes for input validation errors.

const (
	// ErrValidationRequired indicates a required field is missing.
	ErrValidationRequired = "VALIDATION_REQUIRED"

	// ErrValidationInvalidValue indicates a value is invalid.
	ErrValidationInvalidValue = "VALIDATION_INVALID_VALUE"

	// ErrValidationOutOfRange indicates a value is outside allowed range.
	ErrValidationOutOfRange = "VALIDATION_OUT_OF_RANGE"

	// ErrValidationTypeMismatch indicates a type mismatch.
	ErrValidationTypeMismatch = "VALIDATION_TYPE_MISMATCH"

	// ErrValidationInvalidFormat indicates an invalid format.
	ErrValidationInvalidFormat = "VALIDATION_INVALID_FORMAT"
)

// -----------------------------------------------------------------------------
// Network Error Codes
// -----------------------------------------------------------------------------
// Use these codes for network-related errors.

const (
	// ErrNetworkTimeout indicates a network operation timed out.
	ErrNetworkTimeout = "NETWORK_TIMEOUT"

	// ErrNetworkConnectionRefused indicates the connection was refused.
	ErrNetworkConnectionRefused = "NETWORK_CONNECTION_REFUSED"

	// ErrNetworkDNSFailed indicates DNS resolution failed.
	ErrNetworkDNSFailed = "NETWORK_DNS_FAILED"

	// ErrNetworkUnreachable indicates the network/host is unreachable.
	ErrNetworkUnreachable = "NETWORK_UNREACHABLE"

	// ErrNetworkTLSFailed indicates TLS/SSL handshake failed.
	ErrNetworkTLSFailed = "NETWORK_TLS_FAILED"
)

// -----------------------------------------------------------------------------
// I/O Error Codes
// -----------------------------------------------------------------------------
// Use these codes for file and I/O related errors.

const (
	// ErrIOReadFailed indicates a file read operation failed.
	ErrIOReadFailed = "IO_READ_FAILED"

	// ErrIOWriteFailed indicates a file write operation failed.
	ErrIOWriteFailed = "IO_WRITE_FAILED"

	// ErrIOPermissionDenied indicates a permission error.
	ErrIOPermissionDenied = "IO_PERMISSION_DENIED"

	// ErrIOFileNotFound indicates a file was not found.
	ErrIOFileNotFound = "IO_FILE_NOT_FOUND"

	// ErrIODirNotFound indicates a directory was not found.
	ErrIODirNotFound = "IO_DIR_NOT_FOUND"

	// ErrIODiskFull indicates the disk is full.
	ErrIODiskFull = "IO_DISK_FULL"

	// ErrIOMarshalFailed indicates data marshaling failed.
	ErrIOMarshalFailed = "IO_MARSHAL_FAILED"

	// ErrIOUnmarshalFailed indicates data unmarshaling failed.
	ErrIOUnmarshalFailed = "IO_UNMARSHAL_FAILED"
)

// -----------------------------------------------------------------------------
// Internal Error Codes
// -----------------------------------------------------------------------------
// Use these codes for internal/unexpected errors.

const (
	// ErrInternalError indicates an unexpected internal error.
	ErrInternalError = "INTERNAL_ERROR"

	// ErrInternalInvariantViolation indicates a programming invariant was violated.
	ErrInternalInvariantViolation = "INTERNAL_INVARIANT_VIOLATION"

	// ErrInternalNilPointer indicates an unexpected nil pointer.
	ErrInternalNilPointer = "INTERNAL_NIL_POINTER"

	// ErrInternalPanic indicates a panic was recovered.
	ErrInternalPanic = "INTERNAL_PANIC"
)

// -----------------------------------------------------------------------------
// Concepts/Analysis Error Codes
// -----------------------------------------------------------------------------
// Use these codes for concept extraction and analysis errors.

const (
	// ErrConceptsNoHiddenState indicates no agent supports hidden state extraction.
	ErrConceptsNoHiddenState = "CONCEPTS_NO_HIDDEN_STATE"

	// ErrConceptsInsufficientSamples indicates not enough samples for analysis.
	ErrConceptsInsufficientSamples = "CONCEPTS_INSUFFICIENT_SAMPLES"

	// ErrConceptsNotFound indicates the requested concept was not found.
	ErrConceptsNotFound = "CONCEPTS_NOT_FOUND"

	// ErrConceptsExtractionFailed indicates concept extraction failed.
	ErrConceptsExtractionFailed = "CONCEPTS_EXTRACTION_FAILED"

	// ErrAnalysisFailed indicates analysis failed.
	ErrAnalysisFailed = "ANALYSIS_FAILED"

	// ErrAnalysisServerUnavailable indicates the analysis server is unavailable.
	ErrAnalysisServerUnavailable = "ANALYSIS_SERVER_UNAVAILABLE"

	// ErrAnalysisInvalidResponse indicates the analysis server returned invalid data.
	ErrAnalysisInvalidResponse = "ANALYSIS_INVALID_RESPONSE"
)

// -----------------------------------------------------------------------------
// Session Error Codes
// -----------------------------------------------------------------------------
// Use these codes for session-related errors.

const (
	// ErrSessionNotFound indicates the session was not found.
	ErrSessionNotFound = "SESSION_NOT_FOUND"

	// ErrSessionExportFailed indicates session export failed.
	ErrSessionExportFailed = "SESSION_EXPORT_FAILED"

	// ErrSessionLoadFailed indicates session loading failed.
	ErrSessionLoadFailed = "SESSION_LOAD_FAILED"
)

// -----------------------------------------------------------------------------
// Export Error Codes
// -----------------------------------------------------------------------------
// Use these codes for academic format export errors.

const (
	// ErrExportFailed indicates a general export failure.
	ErrExportFailed = "EXPORT_FAILED"

	// ErrExportNoData indicates no data available to export.
	ErrExportNoData = "EXPORT_NO_DATA"

	// ErrExportDirCreateFailed indicates directory creation failed.
	ErrExportDirCreateFailed = "EXPORT_DIR_CREATE_FAILED"

	// ErrExportWriteFailed indicates file write failed during export.
	ErrExportWriteFailed = "EXPORT_WRITE_FAILED"

	// ErrExportInvalidFormat indicates an invalid export format was specified.
	ErrExportInvalidFormat = "EXPORT_INVALID_FORMAT"

	// ErrExportPermissionDenied indicates permission denied during export.
	ErrExportPermissionDenied = "EXPORT_PERMISSION_DENIED"

	// ErrExportDiskFull indicates disk is full during export.
	ErrExportDiskFull = "EXPORT_DISK_FULL"

	// ErrExportReadOnly indicates the target filesystem is read-only.
	ErrExportReadOnly = "EXPORT_READ_ONLY"

	// ErrExportPathTooLong indicates the export path exceeds system limits.
	ErrExportPathTooLong = "EXPORT_PATH_TOO_LONG"

	// ErrExportInvalidPath indicates the export path is invalid.
	ErrExportInvalidPath = "EXPORT_INVALID_PATH"
)

// -----------------------------------------------------------------------------
// Shell Error Codes
// -----------------------------------------------------------------------------
// Use these codes for shell-specific errors.

const (
	// ErrShellInitFailed indicates shell initialization failed.
	ErrShellInitFailed = "SHELL_INIT_FAILED"

	// ErrShellHistoryFailed indicates history file operations failed.
	ErrShellHistoryFailed = "SHELL_HISTORY_FAILED"

	// ErrShellReadlineFailed indicates readline initialization failed.
	ErrShellReadlineFailed = "SHELL_READLINE_FAILED"
)

// -----------------------------------------------------------------------------
// Error Code Lookup Helpers
// -----------------------------------------------------------------------------

// CodeCategory returns the category for a given error code.
// Returns CategoryInternal if the code is not recognized.
func CodeCategory(code string) Category {
	switch code {
	// Config codes
	case ErrConfigNotFound, ErrConfigParseFailed, ErrConfigInvalid,
		ErrConfigInitFailed, ErrConfigReadFailed, ErrConfigWriteFailed:
		return CategoryConfig

	// Backend codes
	case ErrBackendUnavailable, ErrBackendNotFound, ErrBackendConnectionFailed,
		ErrBackendAlreadyRegistered, ErrBackendTimeout, ErrBackendAPIError,
		ErrBackendAuthFailed, ErrBackendNotInstalled, ErrBackendStreamFailed:
		return CategoryBackend

	// Agent codes
	case ErrAgentNotFound, ErrAgentAlreadyExists, ErrAgentCreationFailed,
		ErrAgentNotReady, ErrAgentChatFailed, ErrAgentInvalidConfig,
		ErrAgentNoHiddenState:
		return CategoryAgent

	// Command codes
	case ErrCommandInvalidSyntax, ErrCommandMissingArgs, ErrCommandInvalidArg,
		ErrCommandNotFound, ErrCommandExecutionFailed, ErrCommandEmptyInput:
		return CategoryCommand

	// Validation codes
	case ErrValidationRequired, ErrValidationInvalidValue, ErrValidationOutOfRange,
		ErrValidationTypeMismatch, ErrValidationInvalidFormat:
		return CategoryValidation

	// Network codes
	case ErrNetworkTimeout, ErrNetworkConnectionRefused, ErrNetworkDNSFailed,
		ErrNetworkUnreachable, ErrNetworkTLSFailed:
		return CategoryNetwork

	// IO codes
	case ErrIOReadFailed, ErrIOWriteFailed, ErrIOPermissionDenied,
		ErrIOFileNotFound, ErrIODirNotFound, ErrIODiskFull,
		ErrIOMarshalFailed, ErrIOUnmarshalFailed:
		return CategoryIO

	// Concepts/Analysis codes
	case ErrConceptsNoHiddenState, ErrConceptsInsufficientSamples,
		ErrConceptsNotFound, ErrConceptsExtractionFailed,
		ErrAnalysisFailed, ErrAnalysisServerUnavailable, ErrAnalysisInvalidResponse:
		return CategoryInternal // These are internal to Weaver's analysis system

	// Session codes
	case ErrSessionNotFound, ErrSessionExportFailed, ErrSessionLoadFailed:
		return CategoryIO // Sessions are file-based

	// Export codes
	case ErrExportFailed, ErrExportNoData, ErrExportDirCreateFailed,
		ErrExportWriteFailed, ErrExportInvalidFormat, ErrExportPermissionDenied,
		ErrExportDiskFull, ErrExportReadOnly, ErrExportPathTooLong,
		ErrExportInvalidPath:
		return CategoryIO // Export is file-based

	// Shell codes
	case ErrShellInitFailed, ErrShellHistoryFailed, ErrShellReadlineFailed:
		return CategoryCommand // Shell is part of command interface

	// Internal codes
	case ErrInternalError, ErrInternalInvariantViolation,
		ErrInternalNilPointer, ErrInternalPanic:
		return CategoryInternal

	default:
		return CategoryInternal
	}
}

// IsConfigCode returns true if the code is a configuration error code.
func IsConfigCode(code string) bool {
	return CodeCategory(code) == CategoryConfig
}

// IsBackendCode returns true if the code is a backend error code.
func IsBackendCode(code string) bool {
	return CodeCategory(code) == CategoryBackend
}

// IsAgentCode returns true if the code is an agent error code.
func IsAgentCode(code string) bool {
	return CodeCategory(code) == CategoryAgent
}

// IsCommandCode returns true if the code is a command error code.
func IsCommandCode(code string) bool {
	return CodeCategory(code) == CategoryCommand
}

// IsValidationCode returns true if the code is a validation error code.
func IsValidationCode(code string) bool {
	return CodeCategory(code) == CategoryValidation
}

// IsNetworkCode returns true if the code is a network error code.
func IsNetworkCode(code string) bool {
	return CodeCategory(code) == CategoryNetwork
}

// IsIOCode returns true if the code is an I/O error code.
func IsIOCode(code string) bool {
	return CodeCategory(code) == CategoryIO
}

// IsInternalCode returns true if the code is an internal error code.
func IsInternalCode(code string) bool {
	return CodeCategory(code) == CategoryInternal
}

// IsExportCode returns true if the code is an export error code.
func IsExportCode(code string) bool {
	switch code {
	case ErrExportFailed, ErrExportNoData, ErrExportDirCreateFailed,
		ErrExportWriteFailed, ErrExportInvalidFormat, ErrExportPermissionDenied,
		ErrExportDiskFull, ErrExportReadOnly, ErrExportPathTooLong,
		ErrExportInvalidPath:
		return true
	default:
		return false
	}
}
