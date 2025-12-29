package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// Helper function to create a temporary config file for testing.
func createTempConfigFile(t *testing.T, content string) string {
	t.Helper()

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	if content != "" {
		if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
			t.Fatalf("Failed to create temp config file: %v", err)
		}
	}

	return configPath
}

// parseAPIResponse parses an APIResponse from the response body.
func parseAPIResponse(t *testing.T, body io.Reader) *APIResponse {
	t.Helper()

	var resp APIResponse
	if err := json.NewDecoder(body).Decode(&resp); err != nil {
		t.Fatalf("Failed to parse API response: %v", err)
	}

	return &resp
}

// -----------------------------------------------------------------------------
// ConfigHandler Tests
// -----------------------------------------------------------------------------

func TestNewConfigHandler(t *testing.T) {
	t.Run("with custom path", func(t *testing.T) {
		h := NewConfigHandler("/custom/path/config.yaml")
		if h.ConfigPath() != "/custom/path/config.yaml" {
			t.Errorf("Expected custom path, got %s", h.ConfigPath())
		}
	})

	t.Run("with empty path uses default", func(t *testing.T) {
		h := NewConfigHandler("")
		// Default path is determined by config.DefaultConfigPath()
		if h.ConfigPath() == "" {
			t.Error("Expected non-empty default path")
		}
	})
}

func TestConfigHandler_GetConfig(t *testing.T) {
	t.Run("returns default config when file does not exist", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "nonexistent.yaml")

		h := NewConfigHandler(configPath)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/config", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		// Check that data is present
		if resp.Data == nil {
			t.Error("Expected data in response")
		}

		// Verify it contains expected structure
		data, ok := resp.Data.(map[string]interface{})
		if !ok {
			t.Fatal("Expected data to be a map")
		}

		if _, exists := data["backends"]; !exists {
			t.Error("Expected backends in response")
		}
		if _, exists := data["agents"]; !exists {
			t.Error("Expected agents in response")
		}
		if _, exists := data["session"]; !exists {
			t.Error("Expected session in response")
		}
	})

	t.Run("returns config from existing file", func(t *testing.T) {
		configContent := `
backends:
  claudecode:
    enabled: true
  loom:
    enabled: false
    url: "http://localhost:8080"
agents:
  test-agent:
    role: junior
    backend: claudecode
    active: true
session:
  measurement_mode: active
  auto_export: false
  export_path: "./exports"
`
		configPath := createTempConfigFile(t, configContent)

		h := NewConfigHandler(configPath)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/config", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		// Verify the config values
		data := resp.Data.(map[string]interface{})
		backends := data["backends"].(map[string]interface{})
		claudeCode := backends["claudeCode"].(map[string]interface{})

		if enabled, ok := claudeCode["enabled"].(bool); !ok || !enabled {
			t.Error("Expected claudeCode.enabled to be true")
		}

		session := data["session"].(map[string]interface{})
		if mode, ok := session["measurementMode"].(string); !ok || mode != "active" {
			t.Errorf("Expected measurementMode 'active', got %v", mode)
		}
	})
}

func TestConfigHandler_PutConfig(t *testing.T) {
	t.Run("saves valid config", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")

		h := NewConfigHandler(configPath)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom: LoomRequest{
					Enabled:   true,
					URL:       "http://localhost:9090",
					Path:      "/path/to/loom",
					AutoStart: false,
					Port:      9090,
				},
			},
			Agents: map[string]AgentRequest{
				"my-agent": {
					Role:         "senior",
					Backend:      "claudecode",
					SystemPrompt: "You are a helpful assistant.",
					ToolsEnabled: true,
					Active:       true,
				},
			},
			Session: SessionRequest{
				MeasurementMode: "passive",
				AutoExport:      true,
				ExportPath:      "./my-exports",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPut, "/api/config", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		// Verify file was written
		if _, err := os.Stat(configPath); os.IsNotExist(err) {
			t.Error("Expected config file to be created")
		}

		// Read back and verify
		content, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config file: %v", err)
		}

		if !bytes.Contains(content, []byte("http://localhost:9090")) {
			t.Error("Expected saved config to contain the new Loom URL")
		}
	})

	t.Run("returns error for invalid JSON", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")

		h := NewConfigHandler(configPath)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPut, "/api/config", bytes.NewReader([]byte("not valid json")))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "invalid_json" {
			t.Error("Expected error code 'invalid_json'")
		}
	})

	t.Run("handles empty request body", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")

		h := NewConfigHandler(configPath)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPut, "/api/config", bytes.NewReader([]byte{}))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		// Empty body should result in bad request
		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}
	})
}

func TestConfigHandler_ValidateConfig(t *testing.T) {
	t.Run("validates correct config", func(t *testing.T) {
		h := NewConfigHandler("")
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom: LoomRequest{
					Enabled:   true,
					URL:       "http://localhost:8080",
					Path:      "/path/to/loom",
					AutoStart: true,
					Port:      8080,
				},
			},
			Agents: map[string]AgentRequest{
				"test-agent": {
					Role:         "junior",
					Backend:      "loom",
					SystemPrompt: "Test prompt",
					ToolsEnabled: true,
					Active:       true,
				},
			},
			Session: SessionRequest{
				MeasurementMode: "active",
				AutoExport:      true,
				ExportPath:      "./exports",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/config/validate", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		if valid, ok := data["valid"].(bool); !ok || !valid {
			t.Error("Expected valid to be true")
		}
	})

	t.Run("returns errors for invalid backend", func(t *testing.T) {
		h := NewConfigHandler("")
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom: LoomRequest{
					Enabled: false,
					URL:     "http://localhost:8080",
				},
			},
			Agents: map[string]AgentRequest{
				"test-agent": {
					Role:    "junior",
					Backend: "invalid-backend", // Invalid backend
					Active:  true,
				},
			},
			Session: SessionRequest{
				MeasurementMode: "active",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/config/validate", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		if valid, ok := data["valid"].(bool); !ok || valid {
			t.Error("Expected valid to be false")
		}

		errors := data["errors"].([]interface{})
		if len(errors) == 0 {
			t.Error("Expected validation errors")
		}
	})

	t.Run("returns errors for invalid role", func(t *testing.T) {
		h := NewConfigHandler("")
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom:       LoomRequest{Enabled: false},
			},
			Agents: map[string]AgentRequest{
				"test-agent": {
					Role:    "invalid-role", // Invalid role
					Backend: "claudecode",
					Active:  true,
				},
			},
			Session: SessionRequest{
				MeasurementMode: "active",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/config/validate", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		if valid := data["valid"].(bool); valid {
			t.Error("Expected valid to be false for invalid role")
		}
	})

	t.Run("returns errors for temperature out of range", func(t *testing.T) {
		h := NewConfigHandler("")
		router := NewRouter()
		h.RegisterRoutes(router)

		invalidTemp := 3.0 // Out of range (0.0 - 2.0)
		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom:       LoomRequest{Enabled: false},
			},
			Agents: map[string]AgentRequest{
				"test-agent": {
					Role:        "junior",
					Backend:     "claudecode",
					Active:      true,
					Temperature: &invalidTemp,
				},
			},
			Session: SessionRequest{
				MeasurementMode: "active",
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/config/validate", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		if valid := data["valid"].(bool); valid {
			t.Error("Expected valid to be false for out-of-range temperature")
		}
	})

	t.Run("returns errors for invalid measurement mode", func(t *testing.T) {
		h := NewConfigHandler("")
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom:       LoomRequest{Enabled: false},
			},
			Agents: map[string]AgentRequest{},
			Session: SessionRequest{
				MeasurementMode: "invalid-mode", // Invalid mode
			},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/config/validate", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		if valid := data["valid"].(bool); valid {
			t.Error("Expected valid to be false for invalid measurement mode")
		}
	})

	t.Run("returns error for loom enabled without URL", func(t *testing.T) {
		h := NewConfigHandler("")
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom: LoomRequest{
					Enabled: true,
					URL:     "", // Missing URL when enabled
				},
			},
			Agents:  map[string]AgentRequest{},
			Session: SessionRequest{MeasurementMode: "active"},
		}

		bodyBytes, _ := json.Marshal(requestBody)
		req := httptest.NewRequest(http.MethodPost, "/api/config/validate", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		if valid := data["valid"].(bool); valid {
			t.Error("Expected valid to be false when loom is enabled without URL")
		}
	})
}

// -----------------------------------------------------------------------------
// Conversion Function Tests
// -----------------------------------------------------------------------------

func TestConfigToResponse(t *testing.T) {
	temp := 0.8
	topP := 0.95

	// Create a sample config
	cfg := requestToConfig(&ConfigRequest{
		Backends: BackendsRequest{
			ClaudeCode: ClaudeCodeRequest{Enabled: true},
			Loom: LoomRequest{
				Enabled:   true,
				URL:       "http://localhost:8080",
				Path:      "/path/to/loom",
				AutoStart: true,
				Port:      8080,
			},
		},
		Agents: map[string]AgentRequest{
			"test-agent": {
				Role:          "senior",
				Backend:       "claudecode",
				Model:         "claude-3",
				SystemPrompt:  "Test prompt",
				Tools:         []string{"read_file", "write_file"},
				ToolsEnabled:  true,
				Active:        true,
				MaxTokens:     4096,
				Temperature:   &temp,
				ContextLength: 32768,
				TopP:          &topP,
				TopK:          40,
				GPU:           "auto",
			},
		},
		Session: SessionRequest{
			MeasurementMode: "active",
			AutoExport:      true,
			ExportPath:      "./exports",
		},
	})

	resp := configToResponse(cfg)

	// Verify backends
	if !resp.Backends.ClaudeCode.Enabled {
		t.Error("Expected ClaudeCode to be enabled")
	}
	if !resp.Backends.Loom.Enabled {
		t.Error("Expected Loom to be enabled")
	}
	if resp.Backends.Loom.URL != "http://localhost:8080" {
		t.Errorf("Expected Loom URL 'http://localhost:8080', got %s", resp.Backends.Loom.URL)
	}

	// Verify agent
	agent, exists := resp.Agents["test-agent"]
	if !exists {
		t.Fatal("Expected test-agent in response")
	}
	if agent.Role != "senior" {
		t.Errorf("Expected role 'senior', got %s", agent.Role)
	}
	if agent.Backend != "claudecode" {
		t.Errorf("Expected backend 'claudecode', got %s", agent.Backend)
	}
	if agent.MaxTokens != 4096 {
		t.Errorf("Expected maxTokens 4096, got %d", agent.MaxTokens)
	}
	if agent.Temperature == nil || *agent.Temperature != 0.8 {
		t.Error("Expected temperature 0.8")
	}

	// Verify session
	if resp.Session.MeasurementMode != "active" {
		t.Errorf("Expected measurementMode 'active', got %s", resp.Session.MeasurementMode)
	}
}

func TestRequestToConfig(t *testing.T) {
	temp := 0.7
	topP := 0.9

	req := &ConfigRequest{
		Backends: BackendsRequest{
			ClaudeCode: ClaudeCodeRequest{Enabled: false},
			Loom: LoomRequest{
				Enabled:   true,
				URL:       "http://custom:9090",
				Path:      "/custom/path",
				AutoStart: false,
				Port:      9090,
			},
		},
		Agents: map[string]AgentRequest{
			"custom-agent": {
				Role:          "junior",
				Backend:       "loom",
				Model:         "custom-model",
				SystemPrompt:  "Custom prompt",
				Tools:         []string{"custom_tool"},
				ToolsEnabled:  false,
				Active:        true,
				MaxTokens:     2048,
				Temperature:   &temp,
				ContextLength: 16384,
				TopP:          &topP,
				TopK:          20,
				GPU:           "0",
			},
		},
		Session: SessionRequest{
			MeasurementMode: "passive",
			AutoExport:      false,
			ExportPath:      "/custom/exports",
		},
	}

	cfg := requestToConfig(req)

	// Verify backends
	if cfg.Backends.ClaudeCode.Enabled {
		t.Error("Expected ClaudeCode to be disabled")
	}
	if !cfg.Backends.Loom.Enabled {
		t.Error("Expected Loom to be enabled")
	}
	if cfg.Backends.Loom.Port != 9090 {
		t.Errorf("Expected Loom port 9090, got %d", cfg.Backends.Loom.Port)
	}

	// Verify agent
	agent, exists := cfg.Agents["custom-agent"]
	if !exists {
		t.Fatal("Expected custom-agent in config")
	}
	if agent.Model != "custom-model" {
		t.Errorf("Expected model 'custom-model', got %s", agent.Model)
	}
	if len(agent.Tools) != 1 || agent.Tools[0] != "custom_tool" {
		t.Errorf("Expected tools ['custom_tool'], got %v", agent.Tools)
	}
	if agent.GPU != "0" {
		t.Errorf("Expected GPU '0', got %s", agent.GPU)
	}

	// Verify session
	if cfg.Session.MeasurementMode != "passive" {
		t.Errorf("Expected measurement_mode 'passive', got %s", cfg.Session.MeasurementMode)
	}
	if cfg.Session.AutoExport {
		t.Error("Expected auto_export to be false")
	}
}

// -----------------------------------------------------------------------------
// Validation Function Tests
// -----------------------------------------------------------------------------

func TestValidateConfig(t *testing.T) {
	t.Run("valid config returns no errors", func(t *testing.T) {
		cfg := requestToConfig(&ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom: LoomRequest{
					Enabled: true,
					URL:     "http://localhost:8080",
				},
			},
			Agents: map[string]AgentRequest{
				"agent1": {
					Role:    "senior",
					Backend: "claudecode",
					Active:  true,
				},
			},
			Session: SessionRequest{
				MeasurementMode: "active",
			},
		})

		errors := validateConfig(cfg)
		if len(errors) != 0 {
			t.Errorf("Expected no errors, got %v", errors)
		}
	})

	t.Run("multiple validation errors", func(t *testing.T) {
		invalidTemp := -1.0
		cfg := requestToConfig(&ConfigRequest{
			Backends: BackendsRequest{
				ClaudeCode: ClaudeCodeRequest{Enabled: true},
				Loom: LoomRequest{
					Enabled: true,
					URL:     "", // Error: missing URL
				},
			},
			Agents: map[string]AgentRequest{
				"agent1": {
					Role:        "invalid",      // Error: invalid role
					Backend:     "unknown",      // Error: invalid backend
					Active:      true,
					Temperature: &invalidTemp, // Error: out of range
				},
			},
			Session: SessionRequest{
				MeasurementMode: "unknown", // Error: invalid mode
			},
		})

		errors := validateConfig(cfg)
		if len(errors) < 4 {
			t.Errorf("Expected at least 4 errors, got %d: %v", len(errors), errors)
		}
	})
}

func TestIsValidOption(t *testing.T) {
	validOptions := []string{"one", "two", "three"}

	tests := []struct {
		value    string
		expected bool
	}{
		{"one", true},
		{"two", true},
		{"three", true},
		{"four", false},
		{"", false},
		{"One", false}, // Case sensitive
	}

	for _, tt := range tests {
		t.Run(tt.value, func(t *testing.T) {
			result := isValidOption(tt.value, validOptions)
			if result != tt.expected {
				t.Errorf("isValidOption(%q) = %v, expected %v", tt.value, result, tt.expected)
			}
		})
	}
}

func TestJoinOptions(t *testing.T) {
	tests := []struct {
		options  []string
		expected string
	}{
		{[]string{}, ""},
		{[]string{"one"}, "one"},
		{[]string{"one", "two"}, "one, two"},
		{[]string{"a", "b", "c"}, "a, b, c"},
	}

	for _, tt := range tests {
		result := joinOptions(tt.options)
		if result != tt.expected {
			t.Errorf("joinOptions(%v) = %q, expected %q", tt.options, result, tt.expected)
		}
	}
}

// -----------------------------------------------------------------------------
// Route Registration Tests
// -----------------------------------------------------------------------------

func TestConfigHandler_RegisterRoutes(t *testing.T) {
	h := NewConfigHandler("")
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/config"},
		{http.MethodPut, "/api/config"},
		{http.MethodPost, "/api/config/validate"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			var body io.Reader
			if tt.method == http.MethodPut || tt.method == http.MethodPost {
				body = bytes.NewReader([]byte("{}"))
			}

			req := httptest.NewRequest(tt.method, tt.path, body)
			if body != nil {
				req.Header.Set("Content-Type", "application/json")
			}
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			// Should not get 404
			if rec.Code == http.StatusNotFound {
				t.Errorf("Route %s %s not found", tt.method, tt.path)
			}
		})
	}
}
