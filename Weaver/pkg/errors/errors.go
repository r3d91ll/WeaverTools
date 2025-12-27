// Package errors provides structured error types for Weaver.
// Errors include context, causes, and actionable suggestions.
package errors

import (
	"fmt"
	"strings"
)

// Category classifies errors for consistent handling and display.
type Category string

const (
	CategoryConfig     Category = "config"     // Configuration loading/parsing errors
	CategoryAgent      Category = "agent"      // Agent creation/runtime errors
	CategoryBackend    Category = "backend"    // Backend communication errors
	CategoryCommand    Category = "command"    // Shell command errors
	CategoryValidation Category = "validation" // Input validation errors
	CategoryNetwork    Category = "network"    // Network/connectivity errors
	CategoryIO         Category = "io"         // File/IO errors
	CategoryInternal   Category = "internal"   // Internal/unexpected errors
)

// WeaverError is a structured error with context and suggestions.
// It implements the error interface and supports error wrapping.
type WeaverError struct {
	// Code is a unique identifier for this error type (e.g., "CONFIG_NOT_FOUND")
	Code string

	// Category classifies this error for consistent handling
	Category Category

	// Message is the primary error message describing what went wrong
	Message string

	// Context provides additional key-value details about the error
	Context map[string]string

	// Cause is the underlying error that triggered this error (for wrapping)
	Cause error

	// Suggestions are actionable remediation steps for the user
	Suggestions []string
}

// Error implements the error interface.
// Returns a simple string representation for compatibility with standard error handling.
func (e *WeaverError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying cause for error chain inspection.
// This enables errors.Is() and errors.As() to work with WeaverError.
func (e *WeaverError) Unwrap() error {
	return e.Cause
}

// Is reports whether e matches target for errors.Is() checks.
// Two WeaverErrors match if they have the same Code.
func (e *WeaverError) Is(target error) bool {
	if t, ok := target.(*WeaverError); ok {
		return e.Code == t.Code
	}
	return false
}

// New creates a new WeaverError with the given code, category, and message.
func New(code string, category Category, message string) *WeaverError {
	return &WeaverError{
		Code:     code,
		Category: category,
		Message:  message,
		Context:  make(map[string]string),
	}
}

// WithContext adds a context key-value pair and returns the error for chaining.
func (e *WeaverError) WithContext(key, value string) *WeaverError {
	if e.Context == nil {
		e.Context = make(map[string]string)
	}
	e.Context[key] = value
	return e
}

// WithContextMap adds multiple context key-value pairs.
func (e *WeaverError) WithContextMap(ctx map[string]string) *WeaverError {
	if e.Context == nil {
		e.Context = make(map[string]string)
	}
	for k, v := range ctx {
		e.Context[k] = v
	}
	return e
}

// WithCause wraps an underlying error and returns the error for chaining.
func (e *WeaverError) WithCause(cause error) *WeaverError {
	e.Cause = cause
	return e
}

// WithSuggestion adds a remediation suggestion and returns the error for chaining.
func (e *WeaverError) WithSuggestion(suggestion string) *WeaverError {
	e.Suggestions = append(e.Suggestions, suggestion)
	return e
}

// WithSuggestions adds multiple remediation suggestions.
func (e *WeaverError) WithSuggestions(suggestions ...string) *WeaverError {
	e.Suggestions = append(e.Suggestions, suggestions...)
	return e
}

// HasContext returns true if the error has context information.
func (e *WeaverError) HasContext() bool {
	return len(e.Context) > 0
}

// HasSuggestions returns true if the error has suggestions.
func (e *WeaverError) HasSuggestions() bool {
	return len(e.Suggestions) > 0
}

// ContextString returns a formatted string of all context entries.
func (e *WeaverError) ContextString() string {
	if len(e.Context) == 0 {
		return ""
	}
	var parts []string
	for k, v := range e.Context {
		parts = append(parts, fmt.Sprintf("%s=%q", k, v))
	}
	return strings.Join(parts, ", ")
}

// Wrap wraps an existing error with a WeaverError.
// This is a convenience function for common error wrapping patterns.
func Wrap(err error, code string, category Category, message string) *WeaverError {
	return New(code, category, message).WithCause(err)
}

// AsWeaverError attempts to convert an error to a WeaverError.
// Returns the WeaverError and true if successful, nil and false otherwise.
func AsWeaverError(err error) (*WeaverError, bool) {
	if err == nil {
		return nil, false
	}
	if we, ok := err.(*WeaverError); ok {
		return we, true
	}
	return nil, false
}

// IsCategory checks if an error is a WeaverError with the given category.
func IsCategory(err error, category Category) bool {
	if we, ok := AsWeaverError(err); ok {
		return we.Category == category
	}
	return false
}

// IsCode checks if an error is a WeaverError with the given code.
func IsCode(err error, code string) bool {
	if we, ok := AsWeaverError(err); ok {
		return we.Code == code
	}
	return false
}

// -----------------------------------------------------------------------------
// Helper Constructors for Common Error Types
// -----------------------------------------------------------------------------

// ConfigError creates a new configuration error.
// Use for config file parsing, missing files, or invalid configuration values.
func ConfigError(code, message string) *WeaverError {
	return New(code, CategoryConfig, message)
}

// ConfigErrorf creates a new configuration error with formatted message.
func ConfigErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryConfig, fmt.Sprintf(format, args...))
}

// AgentError creates a new agent-related error.
// Use for agent creation, runtime, or communication issues.
func AgentError(code, message string) *WeaverError {
	return New(code, CategoryAgent, message)
}

// AgentErrorf creates a new agent error with formatted message.
func AgentErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryAgent, fmt.Sprintf(format, args...))
}

// BackendError creates a new backend communication error.
// Use for backend unavailable, API errors, or connection issues.
func BackendError(code, message string) *WeaverError {
	return New(code, CategoryBackend, message)
}

// BackendErrorf creates a new backend error with formatted message.
func BackendErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryBackend, fmt.Sprintf(format, args...))
}

// CommandError creates a new shell command error.
// Use for command parsing, execution, or argument validation issues.
func CommandError(code, message string) *WeaverError {
	return New(code, CategoryCommand, message)
}

// CommandErrorf creates a new command error with formatted message.
func CommandErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryCommand, fmt.Sprintf(format, args...))
}

// ValidationError creates a new validation error.
// Use for input validation, schema validation, or constraint violations.
func ValidationError(code, message string) *WeaverError {
	return New(code, CategoryValidation, message)
}

// ValidationErrorf creates a new validation error with formatted message.
func ValidationErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryValidation, fmt.Sprintf(format, args...))
}

// NetworkError creates a new network/connectivity error.
// Use for connection failures, timeouts, or DNS issues.
func NetworkError(code, message string) *WeaverError {
	return New(code, CategoryNetwork, message)
}

// NetworkErrorf creates a new network error with formatted message.
func NetworkErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryNetwork, fmt.Sprintf(format, args...))
}

// IOError creates a new file/IO error.
// Use for file read/write failures, permission issues, or disk errors.
func IOError(code, message string) *WeaverError {
	return New(code, CategoryIO, message)
}

// IOErrorf creates a new IO error with formatted message.
func IOErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryIO, fmt.Sprintf(format, args...))
}

// InternalError creates a new internal/unexpected error.
// Use for programming errors, invariant violations, or unexpected states.
func InternalError(code, message string) *WeaverError {
	return New(code, CategoryInternal, message)
}

// InternalErrorf creates a new internal error with formatted message.
func InternalErrorf(code, format string, args ...interface{}) *WeaverError {
	return New(code, CategoryInternal, fmt.Sprintf(format, args...))
}

// -----------------------------------------------------------------------------
// Wrapping Helpers for Common Error Types
// -----------------------------------------------------------------------------

// WrapConfig wraps an error as a configuration error.
func WrapConfig(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryConfig, message)
}

// WrapAgent wraps an error as an agent error.
func WrapAgent(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryAgent, message)
}

// WrapBackend wraps an error as a backend error.
func WrapBackend(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryBackend, message)
}

// WrapCommand wraps an error as a command error.
func WrapCommand(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryCommand, message)
}

// WrapValidation wraps an error as a validation error.
func WrapValidation(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryValidation, message)
}

// WrapNetwork wraps an error as a network error.
func WrapNetwork(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryNetwork, message)
}

// WrapIO wraps an error as an IO error.
func WrapIO(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryIO, message)
}

// WrapInternal wraps an error as an internal error.
func WrapInternal(err error, code, message string) *WeaverError {
	return Wrap(err, code, CategoryInternal, message)
}
