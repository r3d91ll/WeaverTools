// Package config handles Weaver configuration loading.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

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
	Enabled bool   `yaml:"enabled"`
	URL     string `yaml:"url"`
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
				Enabled: true,
			},
			Loom: LoomConfig{
				Enabled: true,
				URL:     "http://localhost:8080",
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
				ToolsEnabled: true,
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
				Temperature:   0.7,
				ContextLength: 32768,
				TopP:          0.9,
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
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	cfg := Default()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
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
