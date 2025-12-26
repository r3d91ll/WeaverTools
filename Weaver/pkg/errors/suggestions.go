// Package errors provides a suggestions registry for error remediation.
// Maps error codes to context-aware suggestions that help users fix issues.
package errors

import (
	"runtime"
	"strings"
)

// -----------------------------------------------------------------------------
// Context Keys for Conditional Suggestions
// -----------------------------------------------------------------------------

// Context keys used to select appropriate suggestions.
const (
	// ContextOS is the operating system (e.g., "linux", "darwin", "windows")
	ContextOS = "os"

	// ContextBackend is the backend type (e.g., "claudecode", "loom")
	ContextBackend = "backend"

	// ContextArch is the CPU architecture (e.g., "amd64", "arm64")
	ContextArch = "arch"

	// ContextShell is the shell type (e.g., "bash", "zsh", "fish")
	ContextShell = "shell"
)

// OS values for platform-specific suggestions.
const (
	OSLinux   = "linux"
	OSDarwin  = "darwin"
	OSWindows = "windows"
)

// Backend values for backend-specific suggestions.
const (
	BackendClaudeCode = "claudecode"
	BackendLoom       = "loom"
)

// -----------------------------------------------------------------------------
// Suggestion Type
// -----------------------------------------------------------------------------

// Suggestion represents a remediation suggestion with optional conditions.
// Conditions allow context-aware suggestions (e.g., OS-specific fixes).
type Suggestion struct {
	// Text is the suggestion message displayed to the user.
	Text string

	// Conditions are optional key-value pairs that must match the error context.
	// If empty, the suggestion applies to all contexts.
	// If multiple conditions are specified, ALL must match.
	Conditions map[string]string

	// Priority determines order when multiple suggestions apply.
	// Higher priority suggestions are shown first.
	Priority int
}

// Matches returns true if this suggestion's conditions match the given context.
// Empty conditions match any context.
func (s *Suggestion) Matches(ctx map[string]string) bool {
	if len(s.Conditions) == 0 {
		return true
	}
	for key, value := range s.Conditions {
		if ctx[key] != value {
			return false
		}
	}
	return true
}

// -----------------------------------------------------------------------------
// Suggestions Registry
// -----------------------------------------------------------------------------

// Registry maps error codes to their remediation suggestions.
// Suggestions can be conditional based on context (OS, backend, etc).
type Registry struct {
	suggestions map[string][]Suggestion
}

// NewRegistry creates a new suggestion registry.
func NewRegistry() *Registry {
	return &Registry{
		suggestions: make(map[string][]Suggestion),
	}
}

// Register adds a suggestion for an error code.
func (r *Registry) Register(code, text string) *Registry {
	r.suggestions[code] = append(r.suggestions[code], Suggestion{
		Text: text,
	})
	return r
}

// RegisterWithCondition adds a conditional suggestion for an error code.
// The suggestion only applies when the context matches the conditions.
func (r *Registry) RegisterWithCondition(code, text string, conditions map[string]string) *Registry {
	r.suggestions[code] = append(r.suggestions[code], Suggestion{
		Text:       text,
		Conditions: conditions,
	})
	return r
}

// RegisterWithPriority adds a suggestion with explicit priority.
func (r *Registry) RegisterWithPriority(code, text string, priority int) *Registry {
	r.suggestions[code] = append(r.suggestions[code], Suggestion{
		Text:     text,
		Priority: priority,
	})
	return r
}

// RegisterSuggestion adds a complete Suggestion struct.
func (r *Registry) RegisterSuggestion(code string, suggestion Suggestion) *Registry {
	r.suggestions[code] = append(r.suggestions[code], suggestion)
	return r
}

// Get returns all suggestions for an error code that match the given context.
// Returns suggestions sorted by priority (highest first).
func (r *Registry) Get(code string, ctx map[string]string) []string {
	allSuggestions, ok := r.suggestions[code]
	if !ok {
		return nil
	}

	var matching []Suggestion
	for _, s := range allSuggestions {
		if s.Matches(ctx) {
			matching = append(matching, s)
		}
	}

	// Sort by priority (descending)
	sortByPriority(matching)

	// Extract text
	result := make([]string, len(matching))
	for i, s := range matching {
		result[i] = s.Text
	}
	return result
}

// GetAll returns all suggestions for an error code (ignoring conditions).
func (r *Registry) GetAll(code string) []Suggestion {
	return r.suggestions[code]
}

// HasSuggestions returns true if any suggestions exist for the error code.
func (r *Registry) HasSuggestions(code string) bool {
	return len(r.suggestions[code]) > 0
}

// Codes returns all error codes that have registered suggestions.
func (r *Registry) Codes() []string {
	codes := make([]string, 0, len(r.suggestions))
	for code := range r.suggestions {
		codes = append(codes, code)
	}
	return codes
}

// sortByPriority sorts suggestions by priority in descending order.
func sortByPriority(suggestions []Suggestion) {
	for i := 0; i < len(suggestions)-1; i++ {
		for j := i + 1; j < len(suggestions); j++ {
			if suggestions[j].Priority > suggestions[i].Priority {
				suggestions[i], suggestions[j] = suggestions[j], suggestions[i]
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Platform Detection
// -----------------------------------------------------------------------------

// CurrentOS returns the current operating system identifier.
func CurrentOS() string {
	return runtime.GOOS
}

// CurrentArch returns the current CPU architecture.
func CurrentArch() string {
	return runtime.GOARCH
}

// DefaultContext returns a context map with current platform information.
func DefaultContext() map[string]string {
	return map[string]string{
		ContextOS:   CurrentOS(),
		ContextArch: CurrentArch(),
	}
}

// MergeContext combines multiple context maps into one.
// Later maps override earlier ones for duplicate keys.
func MergeContext(contexts ...map[string]string) map[string]string {
	result := make(map[string]string)
	for _, ctx := range contexts {
		for k, v := range ctx {
			result[k] = v
		}
	}
	return result
}

// -----------------------------------------------------------------------------
// Global Default Registry
// -----------------------------------------------------------------------------

// defaultRegistry is the global registry with built-in suggestions.
var defaultRegistry = NewRegistry()

// GetSuggestions returns suggestions for an error code using the default registry.
// Uses the current platform context for conditional suggestions.
func GetSuggestions(code string) []string {
	return defaultRegistry.Get(code, DefaultContext())
}

// GetSuggestionsWithContext returns suggestions with custom context.
func GetSuggestionsWithContext(code string, ctx map[string]string) []string {
	return defaultRegistry.Get(code, ctx)
}

// DefaultRegistry returns the global default registry.
func DefaultRegistry() *Registry {
	return defaultRegistry
}

// -----------------------------------------------------------------------------
// Built-in Suggestions
// -----------------------------------------------------------------------------

func init() {
	registerConfigSuggestions()
	registerBackendSuggestions()
	registerAgentSuggestions()
	registerCommandSuggestions()
	registerValidationSuggestions()
	registerNetworkSuggestions()
	registerIOSuggestions()
	registerInternalSuggestions()
	registerConceptsSuggestions()
	registerSessionSuggestions()
	registerShellSuggestions()
}

// registerConfigSuggestions adds suggestions for config-related errors.
func registerConfigSuggestions() {
	// CONFIG_NOT_FOUND
	defaultRegistry.Register(ErrConfigNotFound,
		"Run 'weaver --init' to create a default configuration file")
	defaultRegistry.Register(ErrConfigNotFound,
		"Check that ~/.config/weaver/config.yaml exists")
	defaultRegistry.RegisterWithCondition(ErrConfigNotFound,
		"On macOS, config may be at ~/Library/Application Support/weaver/config.yaml",
		map[string]string{ContextOS: OSDarwin})

	// CONFIG_PARSE_FAILED
	defaultRegistry.Register(ErrConfigParseFailed,
		"Check your config file for YAML syntax errors")
	defaultRegistry.Register(ErrConfigParseFailed,
		"Validate YAML at https://yamlchecker.com or with 'yamllint'")
	defaultRegistry.Register(ErrConfigParseFailed,
		"Common issues: incorrect indentation, missing colons, or unquoted special characters")

	// CONFIG_INVALID
	defaultRegistry.Register(ErrConfigInvalid,
		"Review the error context for which field is invalid")
	defaultRegistry.Register(ErrConfigInvalid,
		"Check the Weaver documentation for valid configuration options")
	defaultRegistry.Register(ErrConfigInvalid,
		"Run 'weaver --init' to see an example configuration")

	// CONFIG_INIT_FAILED
	defaultRegistry.Register(ErrConfigInitFailed,
		"Check that the config directory is writable")
	defaultRegistry.RegisterWithCondition(ErrConfigInitFailed,
		"Try: mkdir -p ~/.config/weaver && chmod 755 ~/.config/weaver",
		map[string]string{ContextOS: OSLinux})
	defaultRegistry.RegisterWithCondition(ErrConfigInitFailed,
		"Try: mkdir -p ~/.config/weaver && chmod 755 ~/.config/weaver",
		map[string]string{ContextOS: OSDarwin})
	defaultRegistry.RegisterWithCondition(ErrConfigInitFailed,
		"Check that %APPDATA%\\weaver exists and is writable",
		map[string]string{ContextOS: OSWindows})

	// CONFIG_READ_FAILED
	defaultRegistry.Register(ErrConfigReadFailed,
		"Check file permissions on the config file")
	defaultRegistry.RegisterWithCondition(ErrConfigReadFailed,
		"Try: chmod 644 ~/.config/weaver/config.yaml",
		map[string]string{ContextOS: OSLinux})
	defaultRegistry.RegisterWithCondition(ErrConfigReadFailed,
		"Try: chmod 644 ~/.config/weaver/config.yaml",
		map[string]string{ContextOS: OSDarwin})

	// CONFIG_WRITE_FAILED
	defaultRegistry.Register(ErrConfigWriteFailed,
		"Check file and directory permissions")
	defaultRegistry.Register(ErrConfigWriteFailed,
		"Ensure you have write access to the config directory")
}

// registerBackendSuggestions adds suggestions for backend-related errors.
func registerBackendSuggestions() {
	// BACKEND_UNAVAILABLE
	defaultRegistry.Register(ErrBackendUnavailable,
		"Ensure at least one backend is installed and configured")
	defaultRegistry.Register(ErrBackendUnavailable,
		"Check your config file to enable backends")

	// BACKEND_NOT_FOUND
	defaultRegistry.Register(ErrBackendNotFound,
		"Check the backend name in your configuration")
	defaultRegistry.Register(ErrBackendNotFound,
		"Available backends: claudecode, loom")

	// BACKEND_CONNECTION_FAILED
	defaultRegistry.Register(ErrBackendConnectionFailed,
		"Check your network connection")
	defaultRegistry.Register(ErrBackendConnectionFailed,
		"Verify the backend service is running")

	// BACKEND_TIMEOUT
	defaultRegistry.Register(ErrBackendTimeout,
		"The backend is taking too long to respond")
	defaultRegistry.Register(ErrBackendTimeout,
		"Try again or check the backend service status")
	defaultRegistry.Register(ErrBackendTimeout,
		"Consider increasing the timeout in config")

	// BACKEND_API_ERROR
	defaultRegistry.Register(ErrBackendAPIError,
		"Check the backend service logs for details")
	defaultRegistry.Register(ErrBackendAPIError,
		"Verify your API credentials are correct")

	// BACKEND_AUTH_FAILED
	defaultRegistry.Register(ErrBackendAuthFailed,
		"Check your API key or credentials")
	defaultRegistry.RegisterWithCondition(ErrBackendAuthFailed,
		"For Claude CLI: Run 'claude auth login' to authenticate",
		map[string]string{ContextBackend: BackendClaudeCode})
	defaultRegistry.RegisterWithCondition(ErrBackendAuthFailed,
		"For Loom: Check your API token configuration",
		map[string]string{ContextBackend: BackendLoom})

	// BACKEND_NOT_INSTALLED
	defaultRegistry.RegisterWithCondition(ErrBackendNotInstalled,
		"Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-cli",
		map[string]string{ContextBackend: BackendClaudeCode})
	defaultRegistry.RegisterWithCondition(ErrBackendNotInstalled,
		"On macOS, you can also use: brew install anthropic/tap/claude-cli",
		map[string]string{ContextBackend: BackendClaudeCode, ContextOS: OSDarwin})
	defaultRegistry.RegisterWithCondition(ErrBackendNotInstalled,
		"Loom server not found. Ensure the Loom service is installed and running",
		map[string]string{ContextBackend: BackendLoom})
	defaultRegistry.Register(ErrBackendNotInstalled,
		"Check that the backend is in your PATH")

	// BACKEND_STREAM_FAILED
	defaultRegistry.Register(ErrBackendStreamFailed,
		"The response stream was interrupted")
	defaultRegistry.Register(ErrBackendStreamFailed,
		"Try the request again")
	defaultRegistry.Register(ErrBackendStreamFailed,
		"Check your network connection stability")

	// BACKEND_ALREADY_REGISTERED
	defaultRegistry.Register(ErrBackendAlreadyRegistered,
		"A backend with this name is already registered")
	defaultRegistry.Register(ErrBackendAlreadyRegistered,
		"Choose a different name or unregister the existing backend first")
}

// registerAgentSuggestions adds suggestions for agent-related errors.
func registerAgentSuggestions() {
	// AGENT_NOT_FOUND
	defaultRegistry.Register(ErrAgentNotFound,
		"Check the agent name is correct")
	defaultRegistry.Register(ErrAgentNotFound,
		"List available agents with '/agents' command")
	defaultRegistry.Register(ErrAgentNotFound,
		"Verify the agent is defined in your configuration")

	// AGENT_ALREADY_EXISTS
	defaultRegistry.Register(ErrAgentAlreadyExists,
		"An agent with this name already exists")
	defaultRegistry.Register(ErrAgentAlreadyExists,
		"Choose a different name or update the existing agent")

	// AGENT_CREATION_FAILED
	defaultRegistry.Register(ErrAgentCreationFailed,
		"Check the agent configuration for errors")
	defaultRegistry.Register(ErrAgentCreationFailed,
		"Ensure the specified backend is available")
	defaultRegistry.Register(ErrAgentCreationFailed,
		"Review the error context for specific issues")

	// AGENT_NOT_READY
	defaultRegistry.Register(ErrAgentNotReady,
		"The agent is still initializing")
	defaultRegistry.Register(ErrAgentNotReady,
		"Wait a moment and try again")

	// AGENT_CHAT_FAILED
	defaultRegistry.Register(ErrAgentChatFailed,
		"The chat request could not be completed")
	defaultRegistry.Register(ErrAgentChatFailed,
		"Check the backend connection")
	defaultRegistry.Register(ErrAgentChatFailed,
		"Try again or use a different agent")

	// AGENT_INVALID_CONFIG
	defaultRegistry.Register(ErrAgentInvalidConfig,
		"The agent configuration is invalid")
	defaultRegistry.Register(ErrAgentInvalidConfig,
		"Check required fields: name, backend, and system_prompt")
	defaultRegistry.Register(ErrAgentInvalidConfig,
		"See the documentation for valid agent options")

	// AGENT_NO_HIDDEN_STATE
	defaultRegistry.Register(ErrAgentNoHiddenState,
		"This agent's backend does not support hidden state extraction")
	defaultRegistry.Register(ErrAgentNoHiddenState,
		"Use an agent with a Loom backend for hidden state features")
	defaultRegistry.Register(ErrAgentNoHiddenState,
		"Hidden state is only available with backends that expose internal representations")
}

// registerCommandSuggestions adds suggestions for command-related errors.
func registerCommandSuggestions() {
	// COMMAND_INVALID_SYNTAX
	defaultRegistry.Register(ErrCommandInvalidSyntax,
		"Check the command syntax")
	defaultRegistry.Register(ErrCommandInvalidSyntax,
		"Use '/help' to see available commands")

	// COMMAND_MISSING_ARGS
	defaultRegistry.Register(ErrCommandMissingArgs,
		"This command requires additional arguments")
	defaultRegistry.Register(ErrCommandMissingArgs,
		"Use '/help <command>' for usage information")

	// COMMAND_INVALID_ARG
	defaultRegistry.Register(ErrCommandInvalidArg,
		"One or more arguments are invalid")
	defaultRegistry.Register(ErrCommandInvalidArg,
		"Check the expected argument format")

	// COMMAND_NOT_FOUND
	defaultRegistry.Register(ErrCommandNotFound,
		"This command does not exist")
	defaultRegistry.Register(ErrCommandNotFound,
		"Use '/help' to see available commands")
	defaultRegistry.Register(ErrCommandNotFound,
		"Commands start with '/' (e.g., /help, /agents)")

	// COMMAND_EXECUTION_FAILED
	defaultRegistry.Register(ErrCommandExecutionFailed,
		"The command could not be executed")
	defaultRegistry.Register(ErrCommandExecutionFailed,
		"Check the error details for more information")

	// COMMAND_EMPTY_INPUT
	defaultRegistry.Register(ErrCommandEmptyInput,
		"No input was provided")
	defaultRegistry.Register(ErrCommandEmptyInput,
		"Type a message or use a command starting with '/'")
}

// registerValidationSuggestions adds suggestions for validation errors.
func registerValidationSuggestions() {
	// VALIDATION_REQUIRED
	defaultRegistry.Register(ErrValidationRequired,
		"A required field is missing")
	defaultRegistry.Register(ErrValidationRequired,
		"Check the error context for which field is required")

	// VALIDATION_INVALID_VALUE
	defaultRegistry.Register(ErrValidationInvalidValue,
		"The provided value is not valid")
	defaultRegistry.Register(ErrValidationInvalidValue,
		"Check the expected format or allowed values")

	// VALIDATION_OUT_OF_RANGE
	defaultRegistry.Register(ErrValidationOutOfRange,
		"The value is outside the allowed range")
	defaultRegistry.Register(ErrValidationOutOfRange,
		"Check the minimum and maximum values in the error context")

	// VALIDATION_TYPE_MISMATCH
	defaultRegistry.Register(ErrValidationTypeMismatch,
		"The value type is incorrect")
	defaultRegistry.Register(ErrValidationTypeMismatch,
		"Expected type is shown in the error context")

	// VALIDATION_INVALID_FORMAT
	defaultRegistry.Register(ErrValidationInvalidFormat,
		"The format is invalid")
	defaultRegistry.Register(ErrValidationInvalidFormat,
		"Check the expected format pattern")
}

// registerNetworkSuggestions adds suggestions for network errors.
func registerNetworkSuggestions() {
	// NETWORK_TIMEOUT
	defaultRegistry.Register(ErrNetworkTimeout,
		"The network request timed out")
	defaultRegistry.Register(ErrNetworkTimeout,
		"Check your internet connection")
	defaultRegistry.Register(ErrNetworkTimeout,
		"The server may be slow or overloaded - try again later")

	// NETWORK_CONNECTION_REFUSED
	defaultRegistry.Register(ErrNetworkConnectionRefused,
		"The connection was refused by the server")
	defaultRegistry.Register(ErrNetworkConnectionRefused,
		"Check that the service is running")
	defaultRegistry.Register(ErrNetworkConnectionRefused,
		"Verify the port and host are correct")

	// NETWORK_DNS_FAILED
	defaultRegistry.Register(ErrNetworkDNSFailed,
		"Could not resolve the hostname")
	defaultRegistry.Register(ErrNetworkDNSFailed,
		"Check your internet connection")
	defaultRegistry.Register(ErrNetworkDNSFailed,
		"Verify the hostname is correct")

	// NETWORK_UNREACHABLE
	defaultRegistry.Register(ErrNetworkUnreachable,
		"The network or host is unreachable")
	defaultRegistry.Register(ErrNetworkUnreachable,
		"Check your internet connection")
	defaultRegistry.Register(ErrNetworkUnreachable,
		"Verify firewall settings are not blocking the connection")

	// NETWORK_TLS_FAILED
	defaultRegistry.Register(ErrNetworkTLSFailed,
		"TLS/SSL handshake failed")
	defaultRegistry.Register(ErrNetworkTLSFailed,
		"Check that the server certificate is valid")
	defaultRegistry.Register(ErrNetworkTLSFailed,
		"Ensure your system's CA certificates are up to date")
	defaultRegistry.RegisterWithCondition(ErrNetworkTLSFailed,
		"On macOS, update certificates via Keychain Access or 'security trust-settings-import'",
		map[string]string{ContextOS: OSDarwin})
	defaultRegistry.RegisterWithCondition(ErrNetworkTLSFailed,
		"On Linux, update CA certificates with 'update-ca-certificates' (Debian/Ubuntu) or 'update-ca-trust' (RHEL/CentOS)",
		map[string]string{ContextOS: OSLinux})
}

// registerIOSuggestions adds suggestions for I/O errors.
func registerIOSuggestions() {
	// IO_READ_FAILED
	defaultRegistry.Register(ErrIOReadFailed,
		"Could not read the file")
	defaultRegistry.Register(ErrIOReadFailed,
		"Check file permissions")

	// IO_WRITE_FAILED
	defaultRegistry.Register(ErrIOWriteFailed,
		"Could not write to the file")
	defaultRegistry.Register(ErrIOWriteFailed,
		"Check file and directory permissions")
	defaultRegistry.Register(ErrIOWriteFailed,
		"Ensure there is sufficient disk space")

	// IO_PERMISSION_DENIED
	defaultRegistry.Register(ErrIOPermissionDenied,
		"Permission denied")
	defaultRegistry.RegisterWithCondition(ErrIOPermissionDenied,
		"Try running with elevated permissions (sudo) if appropriate",
		map[string]string{ContextOS: OSLinux})
	defaultRegistry.RegisterWithCondition(ErrIOPermissionDenied,
		"Try running with elevated permissions (sudo) if appropriate",
		map[string]string{ContextOS: OSDarwin})
	defaultRegistry.RegisterWithCondition(ErrIOPermissionDenied,
		"Try running as Administrator",
		map[string]string{ContextOS: OSWindows})
	defaultRegistry.Register(ErrIOPermissionDenied,
		"Check file ownership and permissions")

	// IO_FILE_NOT_FOUND
	defaultRegistry.Register(ErrIOFileNotFound,
		"The file does not exist")
	defaultRegistry.Register(ErrIOFileNotFound,
		"Check the file path is correct")

	// IO_DIR_NOT_FOUND
	defaultRegistry.Register(ErrIODirNotFound,
		"The directory does not exist")
	defaultRegistry.Register(ErrIODirNotFound,
		"Create the directory or check the path is correct")

	// IO_DISK_FULL
	defaultRegistry.Register(ErrIODiskFull,
		"The disk is full")
	defaultRegistry.Register(ErrIODiskFull,
		"Free up disk space and try again")
	defaultRegistry.RegisterWithCondition(ErrIODiskFull,
		"Use 'df -h' to check disk usage",
		map[string]string{ContextOS: OSLinux})
	defaultRegistry.RegisterWithCondition(ErrIODiskFull,
		"Use 'df -h' to check disk usage",
		map[string]string{ContextOS: OSDarwin})

	// IO_MARSHAL_FAILED
	defaultRegistry.Register(ErrIOMarshalFailed,
		"Could not serialize data")
	defaultRegistry.Register(ErrIOMarshalFailed,
		"This may be an internal error - please report it")

	// IO_UNMARSHAL_FAILED
	defaultRegistry.Register(ErrIOUnmarshalFailed,
		"Could not deserialize data")
	defaultRegistry.Register(ErrIOUnmarshalFailed,
		"The file may be corrupted or in an unexpected format")
}

// registerInternalSuggestions adds suggestions for internal errors.
func registerInternalSuggestions() {
	// INTERNAL_ERROR
	defaultRegistry.Register(ErrInternalError,
		"An unexpected error occurred")
	defaultRegistry.Register(ErrInternalError,
		"This may be a bug - please report it with the error details")

	// INTERNAL_INVARIANT_VIOLATION
	defaultRegistry.Register(ErrInternalInvariantViolation,
		"An internal consistency check failed")
	defaultRegistry.Register(ErrInternalInvariantViolation,
		"This is likely a bug - please report it")

	// INTERNAL_NIL_POINTER
	defaultRegistry.Register(ErrInternalNilPointer,
		"An unexpected nil value was encountered")
	defaultRegistry.Register(ErrInternalNilPointer,
		"This is likely a bug - please report it")

	// INTERNAL_PANIC
	defaultRegistry.Register(ErrInternalPanic,
		"A panic was recovered")
	defaultRegistry.Register(ErrInternalPanic,
		"This is a critical error - please report it with the stack trace")
}

// registerConceptsSuggestions adds suggestions for concepts/analysis errors.
func registerConceptsSuggestions() {
	// CONCEPTS_NO_HIDDEN_STATE
	defaultRegistry.Register(ErrConceptsNoHiddenState,
		"No agent with hidden state support is available")
	defaultRegistry.Register(ErrConceptsNoHiddenState,
		"Use a Loom backend agent for hidden state extraction")

	// CONCEPTS_INSUFFICIENT_SAMPLES
	defaultRegistry.Register(ErrConceptsInsufficientSamples,
		"Not enough samples for reliable analysis")
	defaultRegistry.Register(ErrConceptsInsufficientSamples,
		"Collect more concept samples with '/extract' and try again")
	defaultRegistry.Register(ErrConceptsInsufficientSamples,
		"Minimum sample count is shown in the error context")

	// CONCEPTS_NOT_FOUND
	defaultRegistry.Register(ErrConceptsNotFound,
		"The requested concept was not found")
	defaultRegistry.Register(ErrConceptsNotFound,
		"Use '/concepts' to list available concepts")
	defaultRegistry.Register(ErrConceptsNotFound,
		"Extract concepts first with '/extract'")

	// CONCEPTS_EXTRACTION_FAILED
	defaultRegistry.Register(ErrConceptsExtractionFailed,
		"Could not extract concepts from the agent")
	defaultRegistry.Register(ErrConceptsExtractionFailed,
		"Try again or check the agent's backend connection")

	// ANALYSIS_FAILED
	defaultRegistry.Register(ErrAnalysisFailed,
		"The analysis could not be completed")
	defaultRegistry.Register(ErrAnalysisFailed,
		"Check the error details for specific issues")

	// ANALYSIS_SERVER_UNAVAILABLE
	defaultRegistry.Register(ErrAnalysisServerUnavailable,
		"The analysis server is not available")
	defaultRegistry.Register(ErrAnalysisServerUnavailable,
		"Check your network connection")
	defaultRegistry.Register(ErrAnalysisServerUnavailable,
		"Verify the analysis server URL in your configuration")

	// ANALYSIS_INVALID_RESPONSE
	defaultRegistry.Register(ErrAnalysisInvalidResponse,
		"The analysis server returned invalid data")
	defaultRegistry.Register(ErrAnalysisInvalidResponse,
		"This may be a server-side issue - try again later")
}

// registerSessionSuggestions adds suggestions for session-related errors.
func registerSessionSuggestions() {
	// SESSION_NOT_FOUND
	defaultRegistry.Register(ErrSessionNotFound,
		"The session was not found")
	defaultRegistry.Register(ErrSessionNotFound,
		"Use '/sessions' to list available sessions")

	// SESSION_EXPORT_FAILED
	defaultRegistry.Register(ErrSessionExportFailed,
		"Could not export the session")
	defaultRegistry.Register(ErrSessionExportFailed,
		"Check write permissions on the output location")

	// SESSION_LOAD_FAILED
	defaultRegistry.Register(ErrSessionLoadFailed,
		"Could not load the session")
	defaultRegistry.Register(ErrSessionLoadFailed,
		"The session file may be corrupted")
	defaultRegistry.Register(ErrSessionLoadFailed,
		"Check the file path is correct")
}

// registerShellSuggestions adds suggestions for shell-related errors.
func registerShellSuggestions() {
	// SHELL_INIT_FAILED
	defaultRegistry.Register(ErrShellInitFailed,
		"Could not initialize the shell")
	defaultRegistry.Register(ErrShellInitFailed,
		"Check terminal compatibility")

	// SHELL_HISTORY_FAILED
	defaultRegistry.Register(ErrShellHistoryFailed,
		"Could not access shell history")
	defaultRegistry.Register(ErrShellHistoryFailed,
		"Check permissions on ~/.config/weaver/history")
	defaultRegistry.RegisterWithCondition(ErrShellHistoryFailed,
		"Try: touch ~/.config/weaver/history && chmod 600 ~/.config/weaver/history",
		map[string]string{ContextOS: OSLinux})
	defaultRegistry.RegisterWithCondition(ErrShellHistoryFailed,
		"Try: touch ~/.config/weaver/history && chmod 600 ~/.config/weaver/history",
		map[string]string{ContextOS: OSDarwin})

	// SHELL_READLINE_FAILED
	defaultRegistry.Register(ErrShellReadlineFailed,
		"Readline initialization failed")
	defaultRegistry.Register(ErrShellReadlineFailed,
		"Try running in a different terminal emulator")
	defaultRegistry.RegisterWithCondition(ErrShellReadlineFailed,
		"On Linux, ensure libreadline is installed",
		map[string]string{ContextOS: OSLinux})
}

// -----------------------------------------------------------------------------
// Suggestion Helpers for WeaverError
// -----------------------------------------------------------------------------

// AttachSuggestions adds suggestions from the registry to a WeaverError.
// Uses the error's context for conditional suggestion matching.
func AttachSuggestions(err *WeaverError) *WeaverError {
	if err == nil {
		return nil
	}

	ctx := MergeContext(DefaultContext(), err.Context)
	suggestions := defaultRegistry.Get(err.Code, ctx)
	if len(suggestions) > 0 {
		err.Suggestions = append(err.Suggestions, suggestions...)
	}
	return err
}

// NewWithSuggestions creates a new WeaverError and attaches registry suggestions.
func NewWithSuggestions(code string, category Category, message string) *WeaverError {
	err := New(code, category, message)
	return AttachSuggestions(err)
}

// WrapWithSuggestions wraps an error and attaches registry suggestions.
func WrapWithSuggestions(cause error, code string, category Category, message string) *WeaverError {
	err := Wrap(cause, code, category, message)
	return AttachSuggestions(err)
}

// -----------------------------------------------------------------------------
// Text Formatting Helpers
// -----------------------------------------------------------------------------

// FormatSuggestionList formats a list of suggestions for display.
func FormatSuggestionList(suggestions []string) string {
	if len(suggestions) == 0 {
		return ""
	}
	var sb strings.Builder
	for i, s := range suggestions {
		sb.WriteString("→ ")
		sb.WriteString(s)
		if i < len(suggestions)-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}
