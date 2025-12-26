// Package config tests for configuration loading and structured error handling.
package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// -----------------------------------------------------------------------------
// Load Tests with Structured Errors
// -----------------------------------------------------------------------------

func TestLoad_FileNotFound(t *testing.T) {
	_, err := Load("/nonexistent/path/to/config.yaml")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}

	// Should be a WeaverError with CONFIG_NOT_FOUND code
	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigNotFound {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigNotFound, werr.Code)
	}

	if werr.Category != werrors.CategoryConfig {
		t.Errorf("expected category %v, got %v", werrors.CategoryConfig, werr.Category)
	}

	// Should have suggestions
	if len(werr.Suggestions) == 0 {
		t.Error("expected suggestions to be attached")
	}

	// Should mention --init in suggestions
	foundInit := false
	for _, s := range werr.Suggestions {
		if strings.Contains(s, "--init") {
			foundInit = true
			break
		}
	}
	if !foundInit {
		t.Error("expected suggestion to mention '--init'")
	}
}

func TestLoad_YAMLParseError(t *testing.T) {
	// Create a temp file with invalid YAML
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "bad.yaml")

	invalidYAML := `backends:
  claudecode:
    enabled: true
    invalid_indent
  loom:
    enabled: true
`
	if err := os.WriteFile(configPath, []byte(invalidYAML), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Fatal("expected error for invalid YAML")
	}

	// Should be a WeaverError with CONFIG_PARSE_FAILED code
	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigParseFailed {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigParseFailed, werr.Code)
	}

	// Should have path context
	if werr.Context["path"] != configPath {
		t.Errorf("expected path context %q, got %q", configPath, werr.Context["path"])
	}

	// Should have cause (the original yaml error)
	if werr.Cause == nil {
		t.Error("expected cause to be set")
	}

	// Should have suggestions
	if len(werr.Suggestions) == 0 {
		t.Error("expected suggestions to be attached")
	}
}

func TestLoad_InvalidBackend(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	invalidConfig := `agents:
  test-agent:
    backend: invalid_backend
    role: senior
`
	if err := os.WriteFile(configPath, []byte(invalidConfig), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Fatal("expected error for invalid backend")
	}

	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigInvalid {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
	}

	// Should have context about the invalid field and value
	if !strings.Contains(werr.Context["field"], "backend") {
		t.Errorf("expected field context to mention 'backend', got %q", werr.Context["field"])
	}

	if werr.Context["value"] != "invalid_backend" {
		t.Errorf("expected value context 'invalid_backend', got %q", werr.Context["value"])
	}

	// Should mention valid options
	if !strings.Contains(werr.Context["valid_options"], "claudecode") {
		t.Error("expected valid_options to include 'claudecode'")
	}
}

func TestLoad_InvalidRole(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	invalidConfig := `agents:
  test-agent:
    backend: claudecode
    role: invalid_role
`
	if err := os.WriteFile(configPath, []byte(invalidConfig), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Fatal("expected error for invalid role")
	}

	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigInvalid {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
	}

	// Should have context about the invalid role
	if werr.Context["value"] != "invalid_role" {
		t.Errorf("expected value context 'invalid_role', got %q", werr.Context["value"])
	}

	// Should mention valid options (senior, junior, etc.)
	validOpts := werr.Context["valid_options"]
	if !strings.Contains(validOpts, "senior") || !strings.Contains(validOpts, "junior") {
		t.Error("expected valid_options to include valid roles")
	}
}

func TestLoad_TemperatureOutOfRange(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{"too low", "-0.5"},
		{"too high", "3.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			configPath := filepath.Join(tmpDir, "config.yaml")

			config := `agents:
  test-agent:
    backend: loom
    role: junior
    temperature: ` + tt.value + `
`
			if err := os.WriteFile(configPath, []byte(config), 0644); err != nil {
				t.Fatalf("failed to write temp file: %v", err)
			}

			_, err := Load(configPath)
			if err == nil {
				t.Fatal("expected error for out of range temperature")
			}

			werr, ok := err.(*werrors.WeaverError)
			if !ok {
				t.Fatalf("expected *werrors.WeaverError, got %T", err)
			}

			if werr.Code != werrors.ErrConfigInvalid {
				t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
			}

			// Should mention valid range
			if !strings.Contains(werr.Context["valid_range"], "0") || !strings.Contains(werr.Context["valid_range"], "2") {
				t.Error("expected valid_range to mention 0-2 range")
			}
		})
	}
}

func TestLoad_TopPOutOfRange(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{"too low", "-0.1"},
		{"too high", "1.5"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			configPath := filepath.Join(tmpDir, "config.yaml")

			config := `agents:
  test-agent:
    backend: loom
    role: junior
    top_p: ` + tt.value + `
`
			if err := os.WriteFile(configPath, []byte(config), 0644); err != nil {
				t.Fatalf("failed to write temp file: %v", err)
			}

			_, err := Load(configPath)
			if err == nil {
				t.Fatal("expected error for out of range top_p")
			}

			werr, ok := err.(*werrors.WeaverError)
			if !ok {
				t.Fatalf("expected *werrors.WeaverError, got %T", err)
			}

			if werr.Code != werrors.ErrConfigInvalid {
				t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
			}
		})
	}
}

func TestLoad_NegativeMaxTokens(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	config := `agents:
  test-agent:
    backend: loom
    role: junior
    max_tokens: -100
`
	if err := os.WriteFile(configPath, []byte(config), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Fatal("expected error for negative max_tokens")
	}

	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigInvalid {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
	}
}

func TestLoad_InvalidMeasurementMode(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	config := `session:
  measurement_mode: invalid_mode
`
	if err := os.WriteFile(configPath, []byte(config), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Fatal("expected error for invalid measurement_mode")
	}

	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigInvalid {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
	}

	// Should mention valid options (active, passive, disabled)
	validOpts := werr.Context["valid_options"]
	if !strings.Contains(validOpts, "active") || !strings.Contains(validOpts, "passive") {
		t.Error("expected valid_options to include valid measurement modes")
	}
}

func TestLoad_MissingLoomURL(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	config := `backends:
  loom:
    enabled: true
    url: ""
`
	if err := os.WriteFile(configPath, []byte(config), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := Load(configPath)
	if err == nil {
		t.Fatal("expected error for missing Loom URL")
	}

	werr, ok := err.(*werrors.WeaverError)
	if !ok {
		t.Fatalf("expected *werrors.WeaverError, got %T", err)
	}

	if werr.Code != werrors.ErrConfigInvalid {
		t.Errorf("expected code %q, got %q", werrors.ErrConfigInvalid, werr.Code)
	}

	// Should mention the missing field
	if !strings.Contains(werr.Context["field"], "url") {
		t.Errorf("expected field context to mention 'url', got %q", werr.Context["field"])
	}
}

func TestLoad_ValidConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	validConfig := `backends:
  claudecode:
    enabled: true
  loom:
    enabled: true
    url: "http://localhost:8080"
agents:
  test-agent:
    backend: claudecode
    role: senior
session:
  measurement_mode: active
`
	if err := os.WriteFile(configPath, []byte(validConfig), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("unexpected error loading valid config: %v", err)
	}

	if cfg == nil {
		t.Fatal("expected config to be returned")
	}

	if !cfg.Backends.ClaudeCode.Enabled {
		t.Error("expected ClaudeCode to be enabled")
	}

	if _, ok := cfg.Agents["test-agent"]; !ok {
		t.Error("expected test-agent to be present")
	}
}

// -----------------------------------------------------------------------------
// LoadOrDefault Tests
// -----------------------------------------------------------------------------

func TestLoadOrDefault_EmptyPath(t *testing.T) {
	cfg, err := LoadOrDefault("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg == nil {
		t.Fatal("expected default config")
	}
}

func TestLoadOrDefault_FileNotFound(t *testing.T) {
	cfg, err := LoadOrDefault("/nonexistent/config.yaml")
	if err != nil {
		t.Fatalf("unexpected error for missing file: %v", err)
	}

	if cfg == nil {
		t.Fatal("expected default config")
	}

	// Should have default values
	if !cfg.Backends.ClaudeCode.Enabled {
		t.Error("expected ClaudeCode to be enabled by default")
	}
}

// -----------------------------------------------------------------------------
// Error Helper Tests
// -----------------------------------------------------------------------------

func TestExtractYAMLErrorLocation(t *testing.T) {
	tests := []struct {
		name        string
		errStr      string
		expectedLn  int
		expectedCol int
	}{
		{
			name:       "yaml v3 line only",
			errStr:     "yaml: line 5: mapping values are not allowed here",
			expectedLn: 5,
		},
		{
			name:        "yaml with line and column",
			errStr:      "yaml: line 10:5: found character that cannot start any token",
			expectedLn:  10,
			expectedCol: 5,
		},
		{
			name:       "unmarshal error with line",
			errStr:     "yaml: unmarshal errors:\n  line 3: cannot unmarshal !!str into int",
			expectedLn: 3,
		},
		{
			name:       "no line number",
			errStr:     "yaml: some generic error",
			expectedLn: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line, col := extractYAMLErrorLocation(tt.errStr)
			if line != tt.expectedLn {
				t.Errorf("expected line %d, got %d", tt.expectedLn, line)
			}
			if col != tt.expectedCol {
				t.Errorf("expected col %d, got %d", tt.expectedCol, col)
			}
		})
	}
}

func TestExtractExpectedType(t *testing.T) {
	tests := []struct {
		name     string
		errStr   string
		expected string
	}{
		{
			name:     "float64 pointer",
			errStr:   "cannot unmarshal !!str into *float64",
			expected: "float64",
		},
		{
			name:     "int type",
			errStr:   "cannot unmarshal !!str into int",
			expected: "int",
		},
		{
			name:     "no type",
			errStr:   "some other error",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractExpectedType(tt.errStr)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestIsValidOption(t *testing.T) {
	validOptions := []string{"a", "b", "c"}

	if !isValidOption("a", validOptions) {
		t.Error("expected 'a' to be valid")
	}
	if !isValidOption("b", validOptions) {
		t.Error("expected 'b' to be valid")
	}
	if isValidOption("d", validOptions) {
		t.Error("expected 'd' to be invalid")
	}
	if isValidOption("", validOptions) {
		t.Error("expected empty string to be invalid")
	}
}

// -----------------------------------------------------------------------------
// Default Config Tests
// -----------------------------------------------------------------------------

func TestDefault(t *testing.T) {
	cfg := Default()

	if cfg == nil {
		t.Fatal("expected default config")
	}

	// Check default backends
	if !cfg.Backends.ClaudeCode.Enabled {
		t.Error("expected ClaudeCode to be enabled by default")
	}
	if !cfg.Backends.Loom.Enabled {
		t.Error("expected Loom to be enabled by default")
	}
	if cfg.Backends.Loom.URL != "http://localhost:8080" {
		t.Errorf("expected default Loom URL, got %q", cfg.Backends.Loom.URL)
	}

	// Check default agents exist
	if _, ok := cfg.Agents["senior"]; !ok {
		t.Error("expected senior agent by default")
	}
	if _, ok := cfg.Agents["junior"]; !ok {
		t.Error("expected junior agent by default")
	}

	// Check session defaults
	if cfg.Session.MeasurementMode != "active" {
		t.Errorf("expected measurement_mode 'active', got %q", cfg.Session.MeasurementMode)
	}
}

// -----------------------------------------------------------------------------
// Save and Init Tests
// -----------------------------------------------------------------------------

func TestSave(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "saved_config.yaml")

	cfg := Default()
	if err := cfg.Save(configPath); err != nil {
		t.Fatalf("failed to save config: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(configPath); err != nil {
		t.Fatalf("config file not created: %v", err)
	}

	// Verify we can load it back
	loaded, err := Load(configPath)
	if err != nil {
		t.Fatalf("failed to load saved config: %v", err)
	}

	if loaded.Backends.ClaudeCode.Enabled != cfg.Backends.ClaudeCode.Enabled {
		t.Error("loaded config doesn't match saved config")
	}
}

func TestInitConfig_CreatesFile(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "init_config.yaml")

	if err := InitConfig(configPath); err != nil {
		t.Fatalf("failed to init config: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(configPath); err != nil {
		t.Fatalf("config file not created: %v", err)
	}
}

func TestInitConfig_SkipsExisting(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "existing.yaml")

	// Create existing file with custom content
	customContent := "# Custom config\n"
	if err := os.WriteFile(configPath, []byte(customContent), 0644); err != nil {
		t.Fatalf("failed to write existing file: %v", err)
	}

	// InitConfig should not overwrite
	if err := InitConfig(configPath); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify content wasn't changed
	content, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	if string(content) != customContent {
		t.Error("InitConfig overwrote existing file")
	}
}

// -----------------------------------------------------------------------------
// AgentConfig Tests
// -----------------------------------------------------------------------------

func TestAgentConfig_InferenceDefaults(t *testing.T) {
	agent := AgentConfig{}
	agent.InferenceDefaults()

	if agent.MaxTokens != 2048 {
		t.Errorf("expected MaxTokens 2048, got %d", agent.MaxTokens)
	}
	if agent.Temperature == nil {
		t.Fatal("expected Temperature to be set")
	}
	if *agent.Temperature != 0.7 {
		t.Errorf("expected Temperature 0.7, got %f", *agent.Temperature)
	}
	if agent.ContextLength != 32768 {
		t.Errorf("expected ContextLength 32768, got %d", agent.ContextLength)
	}
	if agent.TopP == nil {
		t.Fatal("expected TopP to be set")
	}
	if *agent.TopP != 0.9 {
		t.Errorf("expected TopP 0.9, got %f", *agent.TopP)
	}
}

func TestAgentConfig_InferenceDefaults_PreservesExisting(t *testing.T) {
	temp := 0.5
	topP := 0.8
	agent := AgentConfig{
		MaxTokens:     1000,
		Temperature:   &temp,
		ContextLength: 16384,
		TopP:          &topP,
	}
	agent.InferenceDefaults()

	// Should preserve existing values
	if agent.MaxTokens != 1000 {
		t.Error("MaxTokens should be preserved")
	}
	if *agent.Temperature != 0.5 {
		t.Error("Temperature should be preserved")
	}
	if agent.ContextLength != 16384 {
		t.Error("ContextLength should be preserved")
	}
	if *agent.TopP != 0.8 {
		t.Error("TopP should be preserved")
	}
}
