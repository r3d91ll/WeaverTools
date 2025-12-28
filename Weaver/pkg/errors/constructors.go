// Package errors provides smart error constructors that auto-attach suggestions.
// These constructors combine error creation with suggestion lookup for convenience.
package errors

import "fmt"

// -----------------------------------------------------------------------------
// Smart Constructors with Auto-Attached Suggestions
// -----------------------------------------------------------------------------
// These constructors create WeaverErrors and automatically attach appropriate
// suggestions from the global registry based on the error code and context.
// Use these for creating user-facing errors that need remediation guidance.

// Config creates a configuration error with auto-attached suggestions.
// Use for config file parsing, missing files, or invalid configuration values.
// The error code should be one of the ErrConfig* constants.
func Config(code, message string) *WeaverError {
	err := New(code, CategoryConfig, message)
	return AttachSuggestions(err)
}

// Configf creates a configuration error with a formatted message and auto-attached suggestions.
func Configf(code, format string, args ...interface{}) *WeaverError {
	return Config(code, fmt.Sprintf(format, args...))
}

// ConfigWrap wraps an error as a configuration error with auto-attached suggestions.
func ConfigWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryConfig, message)
	return AttachSuggestions(err)
}

// ConfigWrapf wraps an error as a configuration error with formatted message and suggestions.
func ConfigWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return ConfigWrap(cause, code, fmt.Sprintf(format, args...))
}

// Backend creates a backend communication error with auto-attached suggestions.
// Use for backend unavailable, API errors, or connection issues.
// The error code should be one of the ErrBackend* constants.
func Backend(code, message string) *WeaverError {
	err := New(code, CategoryBackend, message)
	return AttachSuggestions(err)
}

// Backendf creates a backend error with a formatted message and auto-attached suggestions.
func Backendf(code, format string, args ...interface{}) *WeaverError {
	return Backend(code, fmt.Sprintf(format, args...))
}

// BackendWrap wraps an error as a backend error with auto-attached suggestions.
func BackendWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryBackend, message)
	return AttachSuggestions(err)
}

// BackendWrapf wraps an error as a backend error with formatted message and suggestions.
func BackendWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return BackendWrap(cause, code, fmt.Sprintf(format, args...))
}

// Agent creates an agent-related error with auto-attached suggestions.
// Use for agent creation, runtime, or communication issues.
// The error code should be one of the ErrAgent* constants.
func Agent(code, message string) *WeaverError {
	err := New(code, CategoryAgent, message)
	return AttachSuggestions(err)
}

// Agentf creates an agent error with a formatted message and auto-attached suggestions.
func Agentf(code, format string, args ...interface{}) *WeaverError {
	return Agent(code, fmt.Sprintf(format, args...))
}

// AgentWrap wraps an error as an agent error with auto-attached suggestions.
func AgentWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryAgent, message)
	return AttachSuggestions(err)
}

// AgentWrapf wraps an error as an agent error with formatted message and suggestions.
func AgentWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return AgentWrap(cause, code, fmt.Sprintf(format, args...))
}

// Command creates a shell command error with auto-attached suggestions.
// Use for command parsing, execution, or argument validation issues.
// The error code should be one of the ErrCommand* constants.
func Command(code, message string) *WeaverError {
	err := New(code, CategoryCommand, message)
	return AttachSuggestions(err)
}

// Commandf creates a command error with a formatted message and auto-attached suggestions.
func Commandf(code, format string, args ...interface{}) *WeaverError {
	return Command(code, fmt.Sprintf(format, args...))
}

// CommandWrap wraps an error as a command error with auto-attached suggestions.
func CommandWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryCommand, message)
	return AttachSuggestions(err)
}

// CommandWrapf wraps an error as a command error with formatted message and suggestions.
func CommandWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return CommandWrap(cause, code, fmt.Sprintf(format, args...))
}

// Validation creates a validation error with auto-attached suggestions.
// Use for input validation, schema validation, or constraint violations.
// The error code should be one of the ErrValidation* constants.
func Validation(code, message string) *WeaverError {
	err := New(code, CategoryValidation, message)
	return AttachSuggestions(err)
}

// Validationf creates a validation error with a formatted message and auto-attached suggestions.
func Validationf(code, format string, args ...interface{}) *WeaverError {
	return Validation(code, fmt.Sprintf(format, args...))
}

// ValidationWrap wraps an error as a validation error with auto-attached suggestions.
func ValidationWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryValidation, message)
	return AttachSuggestions(err)
}

// ValidationWrapf wraps an error as a validation error with formatted message and suggestions.
func ValidationWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return ValidationWrap(cause, code, fmt.Sprintf(format, args...))
}

// Network creates a network/connectivity error with auto-attached suggestions.
// Use for connection failures, timeouts, or DNS issues.
// The error code should be one of the ErrNetwork* constants.
func Network(code, message string) *WeaverError {
	err := New(code, CategoryNetwork, message)
	return AttachSuggestions(err)
}

// Networkf creates a network error with a formatted message and auto-attached suggestions.
func Networkf(code, format string, args ...interface{}) *WeaverError {
	return Network(code, fmt.Sprintf(format, args...))
}

// NetworkWrap wraps an error as a network error with auto-attached suggestions.
func NetworkWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryNetwork, message)
	return AttachSuggestions(err)
}

// NetworkWrapf wraps an error as a network error with formatted message and suggestions.
func NetworkWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return NetworkWrap(cause, code, fmt.Sprintf(format, args...))
}

// IO creates a file/IO error with auto-attached suggestions.
// Use for file read/write failures, permission issues, or disk errors.
// The error code should be one of the ErrIO* constants.
func IO(code, message string) *WeaverError {
	err := New(code, CategoryIO, message)
	return AttachSuggestions(err)
}

// IOf creates an IO error with a formatted message and auto-attached suggestions.
func IOf(code, format string, args ...interface{}) *WeaverError {
	return IO(code, fmt.Sprintf(format, args...))
}

// IOWrap wraps an error as an IO error with auto-attached suggestions.
func IOWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryIO, message)
	return AttachSuggestions(err)
}

// IOWrapf wraps an error as an IO error with formatted message and suggestions.
func IOWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return IOWrap(cause, code, fmt.Sprintf(format, args...))
}

// Internal creates an internal/unexpected error with auto-attached suggestions.
// Use for programming errors, invariant violations, or unexpected states.
// The error code should be one of the ErrInternal* constants.
func Internal(code, message string) *WeaverError {
	err := New(code, CategoryInternal, message)
	return AttachSuggestions(err)
}

// Internalf creates an internal error with a formatted message and auto-attached suggestions.
func Internalf(code, format string, args ...interface{}) *WeaverError {
	return Internal(code, fmt.Sprintf(format, args...))
}

// InternalWrap wraps an error as an internal error with auto-attached suggestions.
func InternalWrap(cause error, code, message string) *WeaverError {
	err := Wrap(cause, code, CategoryInternal, message)
	return AttachSuggestions(err)
}

// InternalWrapf wraps an error as an internal error with formatted message and suggestions.
func InternalWrapf(cause error, code, format string, args ...interface{}) *WeaverError {
	return InternalWrap(cause, code, fmt.Sprintf(format, args...))
}

// -----------------------------------------------------------------------------
// Context-Aware Constructors
// -----------------------------------------------------------------------------
// These constructors allow specifying additional context that affects which
// suggestions are attached. Use these when you have specific context information
// that should influence the error's suggestions.

// ConfigWithContext creates a configuration error with context-aware suggestions.
// The context map is merged with the default platform context for suggestion lookup.
func ConfigWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryConfig, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// BackendWithContext creates a backend error with context-aware suggestions.
// The context map should include ContextBackend to get backend-specific suggestions.
func BackendWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryBackend, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// AgentWithContext creates an agent error with context-aware suggestions.
func AgentWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryAgent, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// CommandWithContext creates a command error with context-aware suggestions.
func CommandWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryCommand, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// ValidationWithContext creates a validation error with context-aware suggestions.
func ValidationWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryValidation, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// NetworkWithContext creates a network error with context-aware suggestions.
func NetworkWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryNetwork, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// IOWithContext creates an IO error with context-aware suggestions.
func IOWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryIO, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// InternalWithContext creates an internal error with context-aware suggestions.
func InternalWithContext(code, message string, ctx map[string]string) *WeaverError {
	err := New(code, CategoryInternal, message).WithContextMap(ctx)
	return AttachSuggestions(err)
}

// -----------------------------------------------------------------------------
// Quick Constructors for Common Error Codes
// -----------------------------------------------------------------------------
// These are convenience functions for the most commonly used error codes.
// They provide a simpler API when you just need to create a common error.

// ConfigNotFound creates a CONFIG_NOT_FOUND error with auto-attached suggestions.
func ConfigNotFound(path string) *WeaverError {
	return Configf(ErrConfigNotFound, "configuration file not found").
		WithContext("path", path)
}

// ConfigParseError creates a CONFIG_PARSE_FAILED error with auto-attached suggestions.
func ConfigParseError(path string, cause error) *WeaverError {
	return ConfigWrap(cause, ErrConfigParseFailed, "failed to parse configuration file").
		WithContext("path", path)
}

// BackendNotAvailable creates a BACKEND_UNAVAILABLE error with auto-attached suggestions.
func BackendNotAvailable() *WeaverError {
	return Backend(ErrBackendUnavailable, "no backends available")
}

// BackendNotFound creates a BACKEND_NOT_FOUND error with auto-attached suggestions.
func BackendNotFound(name string) *WeaverError {
	return Backendf(ErrBackendNotFound, "backend not found: %s", name).
		WithContext("backend", name)
}

// BackendNotInstalledError creates a BACKEND_NOT_INSTALLED error with backend-specific suggestions.
func BackendNotInstalledError(backendType string) *WeaverError {
	return BackendWithContext(
		ErrBackendNotInstalled,
		fmt.Sprintf("%s backend is not installed", backendType),
		map[string]string{ContextBackend: backendType},
	)
}

// BackendConnectionError creates a BACKEND_CONNECTION_FAILED error with auto-attached suggestions.
func BackendConnectionError(backend string, cause error) *WeaverError {
	return BackendWrap(cause, ErrBackendConnectionFailed, "failed to connect to backend").
		WithContext("backend", backend)
}

// BackendTimeoutError creates a BACKEND_TIMEOUT error with auto-attached suggestions.
func BackendTimeoutError(backend string) *WeaverError {
	return Backendf(ErrBackendTimeout, "backend request timed out").
		WithContext("backend", backend)
}

// AgentNotFound creates an AGENT_NOT_FOUND error with auto-attached suggestions.
func AgentNotFound(name string) *WeaverError {
	return Agentf(ErrAgentNotFound, "agent not found: %s", name).
		WithContext("agent", name)
}

// AgentCreationError creates an AGENT_CREATION_FAILED error with auto-attached suggestions.
func AgentCreationError(name string, cause error) *WeaverError {
	return AgentWrap(cause, ErrAgentCreationFailed, "failed to create agent").
		WithContext("agent", name)
}

// AgentChatError creates an AGENT_CHAT_FAILED error with auto-attached suggestions.
func AgentChatError(agent string, cause error) *WeaverError {
	return AgentWrap(cause, ErrAgentChatFailed, "chat request failed").
		WithContext("agent", agent)
}

// CommandNotFound creates a COMMAND_NOT_FOUND error with auto-attached suggestions.
func CommandNotFound(cmd string) *WeaverError {
	return Commandf(ErrCommandNotFound, "unknown command: %s", cmd).
		WithContext("command", cmd)
}

// CommandMissingArgs creates a COMMAND_MISSING_ARGS error with auto-attached suggestions.
func CommandMissingArgs(cmd, usage string) *WeaverError {
	return Commandf(ErrCommandMissingArgs, "missing required arguments for %s", cmd).
		WithContext("command", cmd).
		WithContext("usage", usage)
}

// CommandInvalidArg creates a COMMAND_INVALID_ARG error with auto-attached suggestions.
func CommandInvalidArg(arg, expected string) *WeaverError {
	return Commandf(ErrCommandInvalidArg, "invalid argument: %s", arg).
		WithContext("argument", arg).
		WithContext("expected", expected)
}

// ValidationRequired creates a VALIDATION_REQUIRED error with auto-attached suggestions.
func ValidationRequired(field string) *WeaverError {
	return Validationf(ErrValidationRequired, "required field is missing: %s", field).
		WithContext("field", field)
}

// ValidationInvalid creates a VALIDATION_INVALID_VALUE error with auto-attached suggestions.
func ValidationInvalid(field, value, reason string) *WeaverError {
	return Validationf(ErrValidationInvalidValue, "invalid value for %s: %s", field, reason).
		WithContext("field", field).
		WithContext("value", value)
}

// ValidationOutOfRange creates a VALIDATION_OUT_OF_RANGE error with auto-attached suggestions.
func ValidationOutOfRange(field string, value, min, max interface{}) *WeaverError {
	return Validationf(ErrValidationOutOfRange, "%s value %v is out of range [%v, %v]", field, value, min, max).
		WithContext("field", field).
		WithContext("value", fmt.Sprintf("%v", value)).
		WithContext("min", fmt.Sprintf("%v", min)).
		WithContext("max", fmt.Sprintf("%v", max))
}

// NetworkTimeout creates a NETWORK_TIMEOUT error with auto-attached suggestions.
func NetworkTimeout(host string) *WeaverError {
	return Networkf(ErrNetworkTimeout, "connection timed out").
		WithContext("host", host)
}

// NetworkConnectionRefused creates a NETWORK_CONNECTION_REFUSED error with auto-attached suggestions.
func NetworkConnectionRefused(host string, port int) *WeaverError {
	return Networkf(ErrNetworkConnectionRefused, "connection refused").
		WithContext("host", host).
		WithContext("port", fmt.Sprintf("%d", port))
}

// IOFileNotFound creates an IO_FILE_NOT_FOUND error with auto-attached suggestions.
func IOFileNotFound(path string) *WeaverError {
	return IOf(ErrIOFileNotFound, "file not found: %s", path).
		WithContext("path", path)
}

// IOPermissionDenied creates an IO_PERMISSION_DENIED error with auto-attached suggestions.
func IOPermissionDenied(path string) *WeaverError {
	return IOf(ErrIOPermissionDenied, "permission denied: %s", path).
		WithContext("path", path)
}

// InternalPanic creates an INTERNAL_PANIC error for recovered panics.
func InternalPanic(recovered interface{}) *WeaverError {
	return Internalf(ErrInternalPanic, "panic recovered: %v", recovered)
}

// -----------------------------------------------------------------------------
// Export Error Quick Constructors
// -----------------------------------------------------------------------------
// These are convenience functions for creating export-related errors.

// ExportFailed creates an EXPORT_FAILED error for general export failures.
func ExportFailed(command, format string, cause error) *WeaverError {
	return IOWrap(cause, ErrExportFailed, "export failed").
		WithContext("command", command).
		WithContext("format", format)
}

// ExportNoData creates an EXPORT_NO_DATA error when no data is available to export.
func ExportNoData(command, format string) *WeaverError {
	return IO(ErrExportNoData, "no data available to export").
		WithContext("command", command).
		WithContext("format", format).
		WithSuggestion("Run some analyses first to generate data").
		WithSuggestion("Use /extract and /analyze commands to create measurements")
}

// ExportDirCreateFailed creates an EXPORT_DIR_CREATE_FAILED error.
func ExportDirCreateFailed(path string, cause error) *WeaverError {
	return IOWrap(cause, ErrExportDirCreateFailed, "failed to create export directory").
		WithContext("path", path).
		WithSuggestion("Check if the parent directory exists").
		WithSuggestion("Verify write permissions for the directory")
}

// ExportWriteFailed creates an EXPORT_WRITE_FAILED error.
func ExportWriteFailed(path, format string, cause error) *WeaverError {
	return IOWrap(cause, ErrExportWriteFailed, "failed to write export file").
		WithContext("path", path).
		WithContext("format", format)
}

// ExportPermissionDenied creates an EXPORT_PERMISSION_DENIED error.
func ExportPermissionDenied(path string) *WeaverError {
	return IO(ErrExportPermissionDenied, "permission denied for export path").
		WithContext("path", path).
		WithSuggestion("Check write permissions for the export directory").
		WithSuggestion("Try exporting to a different location")
}

// ExportDiskFull creates an EXPORT_DISK_FULL error.
func ExportDiskFull(path string) *WeaverError {
	return IO(ErrExportDiskFull, "disk is full, cannot write export").
		WithContext("path", path).
		WithSuggestion("Free up disk space").
		WithSuggestion("Try exporting to a different disk or location")
}

// ExportReadOnly creates an EXPORT_READ_ONLY error.
func ExportReadOnly(path string) *WeaverError {
	return IO(ErrExportReadOnly, "filesystem is read-only").
		WithContext("path", path).
		WithSuggestion("The target location is on a read-only filesystem").
		WithSuggestion("Try exporting to a different location")
}

// ExportPathTooLong creates an EXPORT_PATH_TOO_LONG error.
func ExportPathTooLong(path string) *WeaverError {
	return IO(ErrExportPathTooLong, "export path exceeds system limits").
		WithContext("path", path).
		WithContext("path_length", fmt.Sprintf("%d", len(path))).
		WithSuggestion("Use a shorter export path").
		WithSuggestion("Configure a different export directory in session settings")
}

// ExportInvalidPath creates an EXPORT_INVALID_PATH error.
func ExportInvalidPath(path, reason string) *WeaverError {
	return IO(ErrExportInvalidPath, "invalid export path").
		WithContext("path", path).
		WithContext("reason", reason).
		WithSuggestion("Check the export path for invalid characters").
		WithSuggestion("Ensure the path is properly formatted")
}

// ExportInvalidFormat creates an EXPORT_INVALID_FORMAT error.
func ExportInvalidFormat(format string, validFormats []string) *WeaverError {
	err := IO(ErrExportInvalidFormat, fmt.Sprintf("invalid export format: %s", format)).
		WithContext("format", format)

	if len(validFormats) > 0 {
		err.WithContext("valid_formats", fmt.Sprintf("%v", validFormats))
		err.WithSuggestion("Valid formats: " + fmt.Sprintf("%v", validFormats))
	}

	return err
}
