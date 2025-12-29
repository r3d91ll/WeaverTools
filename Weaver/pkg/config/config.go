// Package config handles Weaver configuration loading.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
	"gopkg.in/yaml.v3"
)

// float64Ptr returns a pointer to the given float64 value.
func float64Ptr(v float64) *float64 { return &v }

// Config is the root configuration structure.
type Config struct {
	Backends BackendsConfig         `yaml:"backends"`
	Agents   map[string]AgentConfig `yaml:"agents"`
	Session  SessionConfig          `yaml:"session"`
}

// BackendsConfig holds backend settings.
type BackendsConfig struct {
	ClaudeCode ClaudeCodeConfig `yaml:"claudecode"`
	Loom       LoomConfig       `yaml:"loom"`
}

// ClaudeCodeConfig holds Claude Code backend settings.
type ClaudeCodeConfig struct {
	Enabled bool `yaml:"enabled"`
}

// LoomConfig holds The Loom backend settings.
type LoomConfig struct {
	Enabled   bool   `yaml:"enabled"`
	URL       string `yaml:"url"`
	Path      string `yaml:"path"`       // Path to TheLoom directory (for auto-start)
	AutoStart bool   `yaml:"auto_start"` // Start TheLoom if not running
	Port      int    `yaml:"port"`       // Port for TheLoom server
	GPUs      []int  `yaml:"gpus"`       // GPU device IDs to use (e.g., [0, 1]). Empty = auto-detect all
}

// AgentConfig holds agent settings.
type AgentConfig struct {
	Role         string   `yaml:"role"`
	Backend      string   `yaml:"backend"`
	Model        string   `yaml:"model"`
	SystemPrompt string   `yaml:"system_prompt"`
	Tools        []string `yaml:"tools"`
	ToolsEnabled bool     `yaml:"tools_enabled"`
	Active       bool     `yaml:"active"` // Whether agent is active for this session

	// Inference parameters (for Loom backend)
	// Temperature and TopP are pointers to distinguish "not set" from "explicitly 0"
	MaxTokens     int      `yaml:"max_tokens"`
	Temperature   *float64 `yaml:"temperature"`
	ContextLength int      `yaml:"context_length"`
	TopP          *float64 `yaml:"top_p"`
	TopK          int      `yaml:"top_k"`

	// GPU assignment (for Loom backend)
	// "auto" = let Loom decide, "0" = cuda:0, "1" = cuda:1, etc.
	GPU string `yaml:"gpu"`
}

// InferenceDefaults returns sensible defaults for inference parameters.
func (a *AgentConfig) InferenceDefaults() {
	if a.MaxTokens == 0 {
		a.MaxTokens = 2048
	}
	if a.Temperature == nil {
		defaultTemp := 0.7
		a.Temperature = &defaultTemp
	}
	if a.ContextLength == 0 {
		a.ContextLength = 32768
	}
	if a.TopP == nil {
		defaultTopP := 0.9
		a.TopP = &defaultTopP
	}
}

// SessionConfig holds session settings.
type SessionConfig struct {
	MeasurementMode string `yaml:"measurement_mode"`
	AutoExport      bool   `yaml:"auto_export"`
	ExportPath      string `yaml:"export_path"`
}

// Default returns the default configuration.
func Default() *Config {
	return &Config{
		Backends: BackendsConfig{
			ClaudeCode: ClaudeCodeConfig{
				Enabled:   true,
			},
			Loom: LoomConfig{
				Enabled:   true,
				URL:       "http://localhost:8080",
				Path:      "../TheLoom/the-loom",
				AutoStart: true,
				Port:      8080,
			},
		},
		Agents: map[string]AgentConfig{
			"senior": {
				Role:    "senior",
				Backend: "claudecode",
				Active:  true,
				SystemPrompt: `You are the Senior Engineer in a multi-agent AI research system.
Your role is to handle complex reasoning, architecture decisions, and orchestration.
You can interact with other agents using @agent <message>.`,
				ToolsEnabled:   true,
			},
			"junior": {
				Role:    "junior",
				Backend: "loom",
				Model:   "Qwen/Qwen2.5-Coder-7B-Instruct",
				Active:  true,
				SystemPrompt: `You are the Junior Engineer in a multi-agent AI research system.
Your role is to handle implementation tasks, file operations, and routine work.
You have access to tools for file manipulation and command execution.`,
				Tools:         []string{"read_file", "write_file", "list_directory", "execute_command", "search_files", "context_read", "context_write"},
				ToolsEnabled:  true,
				MaxTokens:     2048,
				Temperature:   float64Ptr(0.7),
				ContextLength: 32768,
				TopP:          float64Ptr(0.9),
			},
		},
		Session: SessionConfig{
			MeasurementMode: "active",
			AutoExport:      true,
			ExportPath:      "./experiments",
		},
	}
}

// Load loads configuration from a file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, createConfigReadError(path, err)
	}

	cfg := Default()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, createYAMLParseError(path, data, err)
	}

	// Validate configuration values
	if err := cfg.validate(path); err != nil {
		return nil, err
	}

	return cfg, nil
}

// LoadOrDefault loads config from path, or returns default if not found.
func LoadOrDefault(path string) (*Config, error) {
	if path == "" {
		return Default(), nil
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		return Default(), nil
	}

	return Load(path)
}

// Save saves configuration to a file.
func (c *Config) Save(path string) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	data, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}
	return nil
}

// DefaultConfigPath returns the default config file path.
// Config is application-level, stored with the application.
func DefaultConfigPath() string {
	// First check for config in current working directory
	if _, err := os.Stat("config.yaml"); err == nil {
		return "config.yaml"
	}
	// Then check for config/ subdirectory
	if _, err := os.Stat("config/config.yaml"); err == nil {
		return "config/config.yaml"
	}
	// Default to config.yaml in current directory
	return "config.yaml"
}

// InitConfig creates a default config file if it doesn't exist.
func InitConfig(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil // Already exists
	}

	cfg := Default()
	return cfg.Save(path)
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

// validBackends is the list of supported backend names.
var validBackends = []string{"claudecode", "loom"}

// validRoles is the list of supported agent roles.
var validRoles = []string{"senior", "junior", "analyst", "architect", "reviewer"}

// validMeasurementModes is the list of supported measurement modes.
var validMeasurementModes = []string{"active", "passive", "disabled"}

// validate checks the configuration for invalid values.
func (c *Config) validate(path string) error {
	// Validate agent configurations
	for name, agent := range c.Agents {
		// Check backend
		if agent.Backend != "" && !isValidOption(agent.Backend, validBackends) {
			return createInvalidValueError(path, "agents."+name+".backend", agent.Backend, validBackends)
		}

		// Check role
		if agent.Role != "" && !isValidOption(agent.Role, validRoles) {
			return createInvalidValueError(path, "agents."+name+".role", agent.Role, validRoles)
		}

		// Check temperature range
		if agent.Temperature != nil {
			temp := *agent.Temperature
			if temp < 0.0 || temp > 2.0 {
				return createOutOfRangeError(path, "agents."+name+".temperature", temp, 0.0, 2.0)
			}
		}

		// Check top_p range
		if agent.TopP != nil {
			topP := *agent.TopP
			if topP < 0.0 || topP > 1.0 {
				return createOutOfRangeError(path, "agents."+name+".top_p", topP, 0.0, 1.0)
			}
		}

		// Check max_tokens (positive)
		if agent.MaxTokens < 0 {
			return createOutOfRangeError(path, "agents."+name+".max_tokens", agent.MaxTokens, 0, 1000000)
		}

		// Check context_length (positive)
		if agent.ContextLength < 0 {
			return createOutOfRangeError(path, "agents."+name+".context_length", agent.ContextLength, 0, 1000000)
		}

		// Check top_k (non-negative)
		if agent.TopK < 0 {
			return createOutOfRangeError(path, "agents."+name+".top_k", agent.TopK, 0, 1000)
		}
	}

	// Validate session configuration
	if c.Session.MeasurementMode != "" && !isValidOption(c.Session.MeasurementMode, validMeasurementModes) {
		return createInvalidValueError(path, "session.measurement_mode", c.Session.MeasurementMode, validMeasurementModes)
	}

	// Validate Loom backend URL if enabled
	if c.Backends.Loom.Enabled && c.Backends.Loom.URL == "" {
		return createMissingRequiredFieldError(path, "backends.loom.url", "http://localhost:8080")
	}

	return nil
}

// isValidOption checks if a value is in the list of valid options.
func isValidOption(value string, validOptions []string) bool {
	for _, opt := range validOptions {
		if value == opt {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------------
// Error Helpers
// -----------------------------------------------------------------------------

// createConfigReadError creates a structured error for config file read failures.
func createConfigReadError(path string, err error) *werrors.WeaverError {
	errStr := strings.ToLower(err.Error())

	// File not found
	if os.IsNotExist(err) {
		return werrors.ConfigNotFound(path).
			WithSuggestion("Run 'weaver --init' to create a default configuration").
			WithSuggestion("Check that the path is correct: " + path)
	}

	// Permission denied
	if os.IsPermission(err) {
		return werrors.Config(werrors.ErrConfigReadFailed, "permission denied reading configuration file").
			WithContext("path", path).
			WithCause(err).
			WithSuggestion("Check file permissions: chmod 644 " + path).
			WithSuggestion("Verify you own the file: ls -la " + path)
	}

	// Directory instead of file
	if strings.Contains(errStr, "is a directory") {
		return werrors.Config(werrors.ErrConfigReadFailed, "path is a directory, not a file").
			WithContext("path", path).
			WithCause(err).
			WithSuggestion("Specify a file path, not a directory").
			WithSuggestion("Example: weaver --config ./config/config.yaml")
	}

	// Generic read error
	return werrors.ConfigWrap(err, werrors.ErrConfigReadFailed, "failed to read configuration file").
		WithContext("path", path).
		WithSuggestion("Check that the file exists and is readable").
		WithSuggestion("Run 'weaver --init' to create a default configuration")
}

// createYAMLParseError creates a structured error for YAML parse failures with line numbers.
func createYAMLParseError(path string, data []byte, err error) *werrors.WeaverError {
	errStr := err.Error()

	// Try to extract line number from yaml error
	lineNum, col := extractYAMLErrorLocation(errStr)

	werrBuilder := werrors.ConfigWrap(err, werrors.ErrConfigParseFailed, "failed to parse YAML configuration").
		WithContext("path", path)

	if lineNum > 0 {
		werrBuilder.WithContext("line", strconv.Itoa(lineNum))
		if col > 0 {
			werrBuilder.WithContext("column", strconv.Itoa(col))
		}

		// Add context from the file
		lines := strings.Split(string(data), "\n")
		if lineNum <= len(lines) {
			problemLine := strings.TrimRight(lines[lineNum-1], "\r\n")
			werrBuilder.WithContext("content", problemLine)
		}
	}

	// Provide specific suggestions based on error type
	switch {
	case strings.Contains(errStr, "found character that cannot start"):
		werrBuilder.WithSuggestion("Check for invalid characters at the indicated position")
		werrBuilder.WithSuggestion("Ensure proper YAML indentation (use spaces, not tabs)")
	case strings.Contains(errStr, "mapping values are not allowed"):
		werrBuilder.WithSuggestion("Check for missing colons after keys")
		werrBuilder.WithSuggestion("Ensure proper indentation for nested values")
	case strings.Contains(errStr, "did not find expected key"):
		werrBuilder.WithSuggestion("Check for unclosed quotes or brackets")
		werrBuilder.WithSuggestion("Verify proper YAML structure")
	case strings.Contains(errStr, "could not find expected"):
		werrBuilder.WithSuggestion("Check for missing closing brackets or quotes")
		werrBuilder.WithSuggestion("Verify YAML syntax at the indicated location")
	case strings.Contains(errStr, "cannot unmarshal"):
		// Type mismatch error
		fieldType := extractExpectedType(errStr)
		if fieldType != "" {
			werrBuilder.WithContext("expected_type", fieldType)
		}
		werrBuilder.WithSuggestion("Check that the value matches the expected type")
		werrBuilder.WithSuggestion("Refer to example config: weaver --init")
	default:
		werrBuilder.WithSuggestion("Check YAML syntax with: yamllint " + path)
		werrBuilder.WithSuggestion("Regenerate config with: weaver --init")
	}

	return werrBuilder
}

// extractYAMLErrorLocation extracts line and column numbers from a YAML error message.
func extractYAMLErrorLocation(errStr string) (line, col int) {
	// yaml.v3 formats errors like: "yaml: line 5: mapping values are not allowed here"
	// or "yaml: unmarshal errors:\n  line 5: cannot unmarshal..."
	linePattern := regexp.MustCompile(`line (\d+)(?::(\d+))?`)
	matches := linePattern.FindStringSubmatch(errStr)
	if len(matches) >= 2 {
		line, _ = strconv.Atoi(matches[1])
		if len(matches) >= 3 && matches[2] != "" {
			col, _ = strconv.Atoi(matches[2])
		}
	}
	return
}

// extractExpectedType extracts the expected type from a yaml unmarshal error.
func extractExpectedType(errStr string) string {
	// Patterns like "cannot unmarshal !!str into *float64"
	typePattern := regexp.MustCompile(`into \*?(\w+)`)
	matches := typePattern.FindStringSubmatch(errStr)
	if len(matches) >= 2 {
		return matches[1]
	}
	return ""
}

// createInvalidValueError creates a structured error for invalid configuration values.
func createInvalidValueError(path, field, value string, validOptions []string) *werrors.WeaverError {
	return werrors.Config(werrors.ErrConfigInvalid, "invalid configuration value").
		WithContext("path", path).
		WithContext("field", field).
		WithContext("value", value).
		WithContext("valid_options", strings.Join(validOptions, ", ")).
		WithSuggestion("Valid options are: " + strings.Join(validOptions, ", ")).
		WithSuggestion("Check the field name and value in your config file")
}

// createOutOfRangeError creates a structured error for out-of-range configuration values.
func createOutOfRangeError(path, field string, value, min, max interface{}) *werrors.WeaverError {
	return werrors.Config(werrors.ErrConfigInvalid, "configuration value out of range").
		WithContext("path", path).
		WithContext("field", field).
		WithContext("value", fmt.Sprintf("%v", value)).
		WithContext("valid_range", fmt.Sprintf("%v - %v", min, max)).
		WithSuggestion(fmt.Sprintf("Value must be between %v and %v", min, max)).
		WithSuggestion("Check the documentation for valid ranges")
}

// createMissingRequiredFieldError creates a structured error for missing required fields.
func createMissingRequiredFieldError(path, field, example string) *werrors.WeaverError {
	werr := werrors.Config(werrors.ErrConfigInvalid, "required configuration field is missing").
		WithContext("path", path).
		WithContext("field", field).
		WithSuggestion("Add the required field to your configuration file")

	if example != "" {
		werr.WithContext("example", example)
		werr.WithSuggestion(fmt.Sprintf("Example: %s: %s", field, example))
	}

	return werr
}
