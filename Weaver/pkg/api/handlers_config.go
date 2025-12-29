// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"net/http"
	"strings"
	"sync"

	"github.com/r3d91ll/weaver/pkg/config"
)

// ConfigHandler handles configuration-related API requests.
type ConfigHandler struct {
	// configPath is the path to the configuration file
	configPath string

	// mu protects concurrent access to the configuration file
	mu sync.RWMutex
}

// NewConfigHandler creates a new ConfigHandler with the given config file path.
func NewConfigHandler(configPath string) *ConfigHandler {
	if configPath == "" {
		configPath = config.DefaultConfigPath()
	}
	return &ConfigHandler{
		configPath: configPath,
	}
}

// ConfigPath returns the current configuration file path.
func (h *ConfigHandler) ConfigPath() string {
	return h.configPath
}

// RegisterRoutes registers the configuration API routes on the router.
func (h *ConfigHandler) RegisterRoutes(router *Router) {
	router.GET("/api/config", h.GetConfig)
	router.PUT("/api/config", h.PutConfig)
	router.POST("/api/config/validate", h.ValidateConfig)
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// ConfigResponse is the JSON response for configuration endpoints.
// It wraps config.Config with proper JSON tags for camelCase serialization.
type ConfigResponse struct {
	Backends BackendsResponse         `json:"backends"`
	Agents   map[string]AgentResponse `json:"agents"`
	Session  SessionResponse          `json:"session"`
}

// BackendsResponse is the JSON representation of backends configuration.
type BackendsResponse struct {
	ClaudeCode ClaudeCodeResponse `json:"claudeCode"`
	Loom       LoomResponse       `json:"loom"`
}

// ClaudeCodeResponse is the JSON representation of Claude Code backend config.
type ClaudeCodeResponse struct {
	Enabled bool `json:"enabled"`
}

// LoomResponse is the JSON representation of The Loom backend config.
type LoomResponse struct {
	Enabled   bool   `json:"enabled"`
	URL       string `json:"url"`
	Path      string `json:"path"`
	AutoStart bool   `json:"autoStart"`
	Port      int    `json:"port"`
	GPUs      []int  `json:"gpus,omitempty"`
}

// AgentResponse is the JSON representation of agent configuration.
type AgentResponse struct {
	Role          string   `json:"role"`
	Backend       string   `json:"backend"`
	Model         string   `json:"model,omitempty"`
	SystemPrompt  string   `json:"systemPrompt"`
	Tools         []string `json:"tools,omitempty"`
	ToolsEnabled  bool     `json:"toolsEnabled"`
	Active        bool     `json:"active"`
	MaxTokens     int      `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	ContextLength int      `json:"contextLength,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	TopK          int      `json:"topK,omitempty"`
	GPU           string   `json:"gpu,omitempty"`
}

// SessionResponse is the JSON representation of session configuration.
type SessionResponse struct {
	MeasurementMode string `json:"measurementMode"`
	AutoExport      bool   `json:"autoExport"`
	ExportPath      string `json:"exportPath"`
}

// ConfigRequest is the expected JSON body for PUT /api/config.
// It mirrors ConfigResponse to accept the same structure.
type ConfigRequest struct {
	Backends BackendsRequest         `json:"backends"`
	Agents   map[string]AgentRequest `json:"agents"`
	Session  SessionRequest          `json:"session"`
}

// BackendsRequest is the JSON input for backends configuration.
type BackendsRequest struct {
	ClaudeCode ClaudeCodeRequest `json:"claudeCode"`
	Loom       LoomRequest       `json:"loom"`
}

// ClaudeCodeRequest is the JSON input for Claude Code backend config.
type ClaudeCodeRequest struct {
	Enabled bool `json:"enabled"`
}

// LoomRequest is the JSON input for The Loom backend config.
type LoomRequest struct {
	Enabled   bool   `json:"enabled"`
	URL       string `json:"url"`
	Path      string `json:"path"`
	AutoStart bool   `json:"autoStart"`
	Port      int    `json:"port"`
	GPUs      []int  `json:"gpus,omitempty"`
}

// AgentRequest is the JSON input for agent configuration.
type AgentRequest struct {
	Role          string   `json:"role"`
	Backend       string   `json:"backend"`
	Model         string   `json:"model,omitempty"`
	SystemPrompt  string   `json:"systemPrompt"`
	Tools         []string `json:"tools,omitempty"`
	ToolsEnabled  bool     `json:"toolsEnabled"`
	Active        bool     `json:"active"`
	MaxTokens     int      `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	ContextLength int      `json:"contextLength,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	TopK          int      `json:"topK,omitempty"`
	GPU           string   `json:"gpu,omitempty"`
}

// SessionRequest is the JSON input for session configuration.
type SessionRequest struct {
	MeasurementMode string `json:"measurementMode"`
	AutoExport      bool   `json:"autoExport"`
	ExportPath      string `json:"exportPath"`
}

// ValidationResult is the JSON response for config validation.
type ValidationResult struct {
	Valid  bool     `json:"valid"`
	Errors []string `json:"errors,omitempty"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// GetConfig handles GET /api/config.
// It returns the current configuration loaded from the config file.
func (h *ConfigHandler) GetConfig(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Load configuration from file
	cfg, err := config.LoadOrDefault(h.configPath)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, "config_load_error",
			"Failed to load configuration: "+err.Error())
		return
	}

	// Convert to API response format
	response := configToResponse(cfg)
	WriteJSON(w, http.StatusOK, response)
}

// PutConfig handles PUT /api/config.
// It updates the configuration file with the provided JSON data.
func (h *ConfigHandler) PutConfig(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Parse request body
	var req ConfigRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Convert request to internal config type
	cfg := requestToConfig(&req)

	// Save configuration to file
	if err := cfg.Save(h.configPath); err != nil {
		WriteError(w, http.StatusInternalServerError, "config_save_error",
			"Failed to save configuration: "+err.Error())
		return
	}

	// Return the saved configuration
	response := configToResponse(cfg)
	WriteJSON(w, http.StatusOK, response)
}

// ValidateConfig handles POST /api/config/validate.
// It validates the provided configuration without saving it.
func (h *ConfigHandler) ValidateConfig(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req ConfigRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Convert request to internal config type
	cfg := requestToConfig(&req)

	// Validate configuration using the internal validation
	// We create a temporary config and check for validation errors
	validationErrors := validateConfig(cfg)

	result := ValidationResult{
		Valid:  len(validationErrors) == 0,
		Errors: validationErrors,
	}

	WriteJSON(w, http.StatusOK, result)
}

// -----------------------------------------------------------------------------
// Conversion Functions
// -----------------------------------------------------------------------------

// configToResponse converts a config.Config to a ConfigResponse for JSON output.
func configToResponse(cfg *config.Config) *ConfigResponse {
	agents := make(map[string]AgentResponse)
	for name, agent := range cfg.Agents {
		agents[name] = AgentResponse{
			Role:          agent.Role,
			Backend:       agent.Backend,
			Model:         agent.Model,
			SystemPrompt:  agent.SystemPrompt,
			Tools:         agent.Tools,
			ToolsEnabled:  agent.ToolsEnabled,
			Active:        agent.Active,
			MaxTokens:     agent.MaxTokens,
			Temperature:   agent.Temperature,
			ContextLength: agent.ContextLength,
			TopP:          agent.TopP,
			TopK:          agent.TopK,
			GPU:           agent.GPU,
		}
	}

	return &ConfigResponse{
		Backends: BackendsResponse{
			ClaudeCode: ClaudeCodeResponse{
				Enabled: cfg.Backends.ClaudeCode.Enabled,
			},
			Loom: LoomResponse{
				Enabled:   cfg.Backends.Loom.Enabled,
				URL:       cfg.Backends.Loom.URL,
				Path:      cfg.Backends.Loom.Path,
				AutoStart: cfg.Backends.Loom.AutoStart,
				Port:      cfg.Backends.Loom.Port,
				GPUs:      cfg.Backends.Loom.GPUs,
			},
		},
		Agents: agents,
		Session: SessionResponse{
			MeasurementMode: cfg.Session.MeasurementMode,
			AutoExport:      cfg.Session.AutoExport,
			ExportPath:      cfg.Session.ExportPath,
		},
	}
}

// requestToConfig converts a ConfigRequest to a config.Config for saving.
func requestToConfig(req *ConfigRequest) *config.Config {
	agents := make(map[string]config.AgentConfig)
	for name, agent := range req.Agents {
		agents[name] = config.AgentConfig{
			Role:          agent.Role,
			Backend:       agent.Backend,
			Model:         agent.Model,
			SystemPrompt:  agent.SystemPrompt,
			Tools:         agent.Tools,
			ToolsEnabled:  agent.ToolsEnabled,
			Active:        agent.Active,
			MaxTokens:     agent.MaxTokens,
			Temperature:   agent.Temperature,
			ContextLength: agent.ContextLength,
			TopP:          agent.TopP,
			TopK:          agent.TopK,
			GPU:           agent.GPU,
		}
	}

	return &config.Config{
		Backends: config.BackendsConfig{
			ClaudeCode: config.ClaudeCodeConfig{
				Enabled: req.Backends.ClaudeCode.Enabled,
			},
			Loom: config.LoomConfig{
				Enabled:   req.Backends.Loom.Enabled,
				URL:       req.Backends.Loom.URL,
				Path:      req.Backends.Loom.Path,
				AutoStart: req.Backends.Loom.AutoStart,
				Port:      req.Backends.Loom.Port,
				GPUs:      req.Backends.Loom.GPUs,
			},
		},
		Agents: agents,
		Session: config.SessionConfig{
			MeasurementMode: req.Session.MeasurementMode,
			AutoExport:      req.Session.AutoExport,
			ExportPath:      req.Session.ExportPath,
		},
	}
}

// validateConfig validates a config and returns a list of validation errors.
// This mimics the internal validation from config.Config.validate().
func validateConfig(cfg *config.Config) []string {
	var errors []string

	// Valid options for validation
	validBackends := []string{"claudecode", "loom"}
	validRoles := []string{"senior", "junior", "analyst", "architect", "reviewer", "conversant", "subject"}
	validMeasurementModes := []string{"active", "passive", "disabled"}

	// Validate agent configurations
	for name, agent := range cfg.Agents {
		// Check backend
		if agent.Backend != "" && !isValidOption(agent.Backend, validBackends) {
			errors = append(errors, "agents."+name+".backend: invalid value '"+agent.Backend+"', must be one of: "+joinOptions(validBackends))
		}

		// Check role
		if agent.Role != "" && !isValidOption(agent.Role, validRoles) {
			errors = append(errors, "agents."+name+".role: invalid value '"+agent.Role+"', must be one of: "+joinOptions(validRoles))
		}

		// Check temperature range
		if agent.Temperature != nil {
			temp := *agent.Temperature
			if temp < 0.0 || temp > 2.0 {
				errors = append(errors, "agents."+name+".temperature: value out of range, must be between 0.0 and 2.0")
			}
		}

		// Check top_p range
		if agent.TopP != nil {
			topP := *agent.TopP
			if topP < 0.0 || topP > 1.0 {
				errors = append(errors, "agents."+name+".topP: value out of range, must be between 0.0 and 1.0")
			}
		}

		// Check max_tokens (positive)
		if agent.MaxTokens < 0 {
			errors = append(errors, "agents."+name+".maxTokens: must be non-negative")
		}

		// Check context_length (positive)
		if agent.ContextLength < 0 {
			errors = append(errors, "agents."+name+".contextLength: must be non-negative")
		}

		// Check top_k (non-negative)
		if agent.TopK < 0 {
			errors = append(errors, "agents."+name+".topK: must be non-negative")
		}
	}

	// Validate session configuration
	if cfg.Session.MeasurementMode != "" && !isValidOption(cfg.Session.MeasurementMode, validMeasurementModes) {
		errors = append(errors, "session.measurementMode: invalid value '"+cfg.Session.MeasurementMode+"', must be one of: "+joinOptions(validMeasurementModes))
	}

	// Validate Loom backend URL if enabled
	if cfg.Backends.Loom.Enabled && cfg.Backends.Loom.URL == "" {
		errors = append(errors, "backends.loom.url: required when loom backend is enabled")
	}

	return errors
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

// joinOptions joins a slice of options with commas.
func joinOptions(options []string) string {
	return strings.Join(options, ", ")
}
