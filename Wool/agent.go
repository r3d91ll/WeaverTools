package wool

import (
	"fmt"
	"strings"
)

// Documentation links for agent configuration
const (
	DocLinkAgentConfig = "https://github.com/WeaverAI/docs/agent-configuration"
	DocLinkRoles       = "https://github.com/WeaverAI/docs/roles"
	DocLinkBackends    = "https://github.com/WeaverAI/docs/backends"
	DocLinkParameters  = "https://github.com/WeaverAI/docs/inference-parameters"
)

// ValidRoles returns all valid role values.
func ValidRoles() []string {
	return []string{
		string(RoleSenior),
		string(RoleJunior),
		string(RoleConversant),
		string(RoleSubject),
		string(RoleObserver),
	}
}

// Agent defines an agent's identity and capabilities.
// This is the specification - Weaver creates runtime instances from these.
type Agent struct {
	ID           string   `json:"id" yaml:"id"`
	Name         string   `json:"name" yaml:"name"`
	Role         Role     `json:"role" yaml:"role"`
	Backend      string   `json:"backend" yaml:"backend"`           // "loom" or "claudecode"
	Model        string   `json:"model,omitempty" yaml:"model"`     // Model ID for the backend
	SystemPrompt string   `json:"system_prompt" yaml:"system_prompt"`
	Tools        []string `json:"tools,omitempty" yaml:"tools"`     // Tool names this agent can use
	ToolsEnabled bool     `json:"tools_enabled" yaml:"tools_enabled"`
	Active       bool     `json:"active" yaml:"active"`             // Whether agent is active for this session

	// Inference parameters (for Loom backend)
	MaxTokens     int     `json:"max_tokens,omitempty" yaml:"max_tokens"`
	Temperature   float64 `json:"temperature,omitempty" yaml:"temperature"`
	ContextLength int     `json:"context_length,omitempty" yaml:"context_length"`
	TopP          float64 `json:"top_p,omitempty" yaml:"top_p"`
	TopK          int     `json:"top_k,omitempty" yaml:"top_k"`

	// GPU assignment (for Loom backend)
	// "auto" = let Loom decide, "0" = cuda:0, "1" = cuda:1, etc.
	GPU string `json:"gpu,omitempty" yaml:"gpu"`

	// Capabilities this agent has
	Capabilities []Capability `json:"capabilities,omitempty" yaml:"capabilities"`
}

// Config is an alias for Agent for YAML configuration loading.
type Config = Agent

// Capability defines what an agent can do.
type Capability struct {
	Name        string `json:"name" yaml:"name"`
	Description string `json:"description" yaml:"description"`
	Enabled     bool   `json:"enabled" yaml:"enabled"`
}

// DefaultSenior returns a default Senior agent configuration.
func DefaultSenior() Agent {
	return Agent{
		Name:    "senior",
		Role:    RoleSenior,
		Backend: "claudecode",
		SystemPrompt: `You are the Senior Engineer in a multi-agent AI research system.
Your role is to handle complex reasoning, architecture decisions, and code review.
You can delegate routine tasks to Junior agents using @junior <task>.`,
		ToolsEnabled: true,
		Active:       true,
	}
}

// DefaultJunior returns a default Junior agent configuration.
func DefaultJunior() Agent {
	return Agent{
		Name:    "junior",
		Role:    RoleJunior,
		Backend: "loom",
		SystemPrompt: `You are the Junior Engineer in a multi-agent AI research system.
Your role is to handle implementation tasks, file operations, and routine work.
Report results back to Senior for review.`,
		ToolsEnabled: true,
		Active:       true,
	}
}

// DefaultSubject returns a default Subject agent configuration for experiments.
func DefaultSubject() Agent {
	return Agent{
		Name:    "subject",
		Role:    RoleSubject,
		Backend: "loom",
		SystemPrompt: `You are a Subject in a conveyance measurement experiment.
Respond naturally to prompts. Your hidden states will be analyzed.`,
		ToolsEnabled: false,
		Active:       true,
	}
}

// ValidBackends defines the allowed backend values.
var ValidBackends = []string{"loom", "claudecode"}

// Validate checks if the agent configuration is valid.
// Returns a ValidationError with detailed context, valid options, examples, and doc links.
func (a *Agent) Validate() error {
	// Validate required name field
	if a.Name == "" {
		return NewRequiredFieldError("name", "my-agent", DocLinkAgentConfig)
	}

	// Validate name format (no special characters that could cause issues)
	if strings.ContainsAny(a.Name, " \t\n\r@#$%^&*(){}[]|\\<>") {
		return &ValidationError{
			Field:   "name",
			Message: "name contains invalid characters",
			Value:   a.Name,
			Example: "my-agent or myAgent",
			Context: map[string]string{
				"note": "Names should use alphanumeric characters, hyphens, or underscores",
			},
			DocLink: DocLinkAgentConfig,
		}
	}

	// Validate role
	if !a.Role.IsValid() {
		return NewInvalidValueError("role", string(a.Role), ValidRoles(), DocLinkRoles).
			WithContext("note", getRoleDescriptions())
	}

	// Validate required backend field
	if a.Backend == "" {
		return NewRequiredFieldError("backend", "loom", DocLinkBackends)
	}

	// Validate backend against allowed values
	validBackend := false
	for _, b := range ValidBackends {
		if a.Backend == b {
			validBackend = true
			break
		}
	}
	if !validBackend {
		return NewInvalidValueError("backend", a.Backend, ValidBackends, DocLinkBackends).
			WithContext("note", "Use 'loom' for local models with hidden state access, or 'claudecode' for Claude API")
	}

	// Validate role-capability consistency
	if a.ToolsEnabled && !a.Role.SupportsTools() {
		toolSupportingRoles := []string{string(RoleSenior), string(RoleJunior)}
		return NewIncompatibleError(
			"tools_enabled",
			fmt.Sprintf("role '%s' does not support tools", a.Role),
			fmt.Sprintf("Only %s roles can use tools. Either disable tools_enabled or change the role.", formatOptions(toolSupportingRoles)),
			DocLinkRoles,
		)
	}

	// Validate inference parameter ranges
	if a.Temperature < 0 || a.Temperature > 2.0 {
		return NewOutOfRangeError("temperature", a.Temperature, 0.0, 2.0, "0.7").
			WithContext("note", "Lower values (0.0-0.5) are more deterministic, higher values (1.0-2.0) are more creative")
	}
	if a.TopP < 0 || a.TopP > 1.0 {
		return NewOutOfRangeError("top_p", a.TopP, 0.0, 1.0, "0.9").
			WithContext("note", "Controls nucleus sampling; 1.0 considers all tokens, lower values focus on likely tokens")
	}

	// Validate max_tokens if set
	if a.MaxTokens < 0 {
		return &ValidationError{
			Field:   "max_tokens",
			Message: "max_tokens must be a non-negative integer",
			Value:   fmt.Sprintf("%d", a.MaxTokens),
			Example: "4096",
			DocLink: DocLinkParameters,
		}
	}

	// Validate context_length if set
	if a.ContextLength < 0 {
		return &ValidationError{
			Field:   "context_length",
			Message: "context_length must be a non-negative integer",
			Value:   fmt.Sprintf("%d", a.ContextLength),
			Example: "8192",
			DocLink: DocLinkParameters,
		}
	}

	// Validate top_k if set (must be non-negative)
	if a.TopK < 0 {
		return &ValidationError{
			Field:   "top_k",
			Message: "top_k must be a non-negative integer",
			Value:   fmt.Sprintf("%d", a.TopK),
			Example: "40",
			Context: map[string]string{
				"note": "0 means disabled; higher values consider more token options",
			},
			DocLink: DocLinkParameters,
		}
	}

	// Validate GPU assignment if set
	if a.GPU != "" && a.GPU != "auto" {
		// GPU should be "auto" or a numeric value like "0", "1", etc.
		if a.GPU != "auto" && !isNumericGPU(a.GPU) {
			return &ValidationError{
				Field:        "gpu",
				Message:      "invalid GPU assignment",
				Value:        a.GPU,
				ValidOptions: []string{"auto", "0", "1", "2", "..."},
				Example:      "auto",
				Context: map[string]string{
					"note": "Use 'auto' to let Loom decide, or specify GPU index (0, 1, 2, ...)",
				},
				DocLink: DocLinkBackends,
			}
		}
	}

	return nil
}

// isNumericGPU checks if a string is a valid numeric GPU index.
func isNumericGPU(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// getRoleDescriptions returns a summary of what each role does.
func getRoleDescriptions() string {
	return fmt.Sprintf("%s: %s | %s: %s | %s: %s | %s: %s | %s: %s",
		RoleSenior, "orchestration & architecture",
		RoleJunior, "implementation & tools",
		RoleConversant, "bilateral conveyance",
		RoleSubject, "single-agent study",
		RoleObserver, "passive monitoring")
}

// ValidationError represents a validation failure with rich context.
// Includes valid options, example values, and documentation links to help users fix issues.
type ValidationError struct {
	// Field is the name of the field that failed validation
	Field string

	// Message describes what went wrong
	Message string

	// Value is the actual value that was provided (if any)
	Value string

	// ValidOptions lists allowed values for enumerated fields
	ValidOptions []string

	// Example shows a valid example value for the field
	Example string

	// DocLink points to documentation for more information
	DocLink string

	// Context provides additional information about the validation failure
	Context map[string]string
}

// Error returns a detailed error message with suggestions.
func (e *ValidationError) Error() string {
	msg := e.Field + ": " + e.Message

	// Add the invalid value if provided
	if e.Value != "" {
		msg += " (got: " + e.Value + ")"
	}

	// Add valid options if available
	if len(e.ValidOptions) > 0 {
		msg += "\n  Valid options: " + formatOptions(e.ValidOptions)
	}

	// Add example if available
	if e.Example != "" {
		msg += "\n  Example: " + e.Example
	}

	// Add context entries
	for key, value := range e.Context {
		msg += "\n  " + key + ": " + value
	}

	// Add doc link if available
	if e.DocLink != "" {
		msg += "\n  See: " + e.DocLink
	}

	return msg
}

// WithContext adds a context key-value pair and returns the error for chaining.
func (e *ValidationError) WithContext(key, value string) *ValidationError {
	if e.Context == nil {
		e.Context = make(map[string]string)
	}
	e.Context[key] = value
	return e
}

// formatOptions formats a list of options for display.
func formatOptions(options []string) string {
	if len(options) == 0 {
		return ""
	}
	if len(options) == 1 {
		return "'" + options[0] + "'"
	}
	result := ""
	for i, opt := range options {
		if i > 0 {
			if i == len(options)-1 {
				result += ", or "
			} else {
				result += ", "
			}
		}
		result += "'" + opt + "'"
	}
	return result
}

// ValidationErrors holds multiple validation errors.
type ValidationErrors struct {
	Errors []*ValidationError
}

// Error returns a combined error message from all validation errors.
func (e *ValidationErrors) Error() string {
	if len(e.Errors) == 0 {
		return "validation failed"
	}
	if len(e.Errors) == 1 {
		return e.Errors[0].Error()
	}

	msg := "multiple validation errors:"
	for _, err := range e.Errors {
		msg += "\n  - " + err.Error()
	}
	return msg
}

// Add appends a validation error.
func (e *ValidationErrors) Add(err *ValidationError) {
	e.Errors = append(e.Errors, err)
}

// HasErrors returns true if there are any validation errors.
func (e *ValidationErrors) HasErrors() bool {
	return len(e.Errors) > 0
}

// NewValidationError creates a simple validation error.
func NewValidationError(field, message string) *ValidationError {
	return &ValidationError{
		Field:   field,
		Message: message,
	}
}

// NewRequiredFieldError creates an error for a missing required field.
func NewRequiredFieldError(field, example, docLink string) *ValidationError {
	return &ValidationError{
		Field:   field,
		Message: field + " is required",
		Example: example,
		DocLink: docLink,
	}
}

// NewInvalidValueError creates an error for an invalid value with valid options.
func NewInvalidValueError(field, value string, validOptions []string, docLink string) *ValidationError {
	return &ValidationError{
		Field:        field,
		Message:      "invalid " + field,
		Value:        value,
		ValidOptions: validOptions,
		DocLink:      docLink,
	}
}

// NewOutOfRangeError creates an error for a value outside its valid range.
func NewOutOfRangeError(field string, value interface{}, min, max interface{}, example string) *ValidationError {
	return &ValidationError{
		Field:   field,
		Message: field + " must be between " + formatValue(min) + " and " + formatValue(max),
		Value:   formatValue(value),
		Example: example,
	}
}

// NewIncompatibleError creates an error for incompatible field combinations.
func NewIncompatibleError(field, message, reason, docLink string) *ValidationError {
	err := &ValidationError{
		Field:   field,
		Message: message,
		DocLink: docLink,
	}
	if reason != "" {
		err.Context = map[string]string{"reason": reason}
	}
	return err
}

// formatValue converts a value to a display string.
func formatValue(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case float64:
		return fmt.Sprintf("%.1f", val)
	case float32:
		return fmt.Sprintf("%.1f", val)
	case int:
		return fmt.Sprintf("%d", val)
	case int64:
		return fmt.Sprintf("%d", val)
	default:
		return fmt.Sprintf("%v", val)
	}
}
