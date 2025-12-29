// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"context"
	"net/http"
	"sort"
	"sync"
)

// ModelProvider is the interface for accessing model information and control.
// This interface allows for dependency injection and testing.
type ModelProvider interface {
	// ListModels returns information about all available models.
	ListModels(ctx context.Context) ([]ModelInfo, error)
	// GetModel returns information about a specific model.
	GetModel(ctx context.Context, name string) (*ModelInfo, error)
	// LoadModel loads the specified model into memory.
	LoadModel(ctx context.Context, name string) error
	// UnloadModel unloads the specified model from memory.
	UnloadModel(ctx context.Context, name string) error
}

// ModelInfo represents information about a model.
type ModelInfo struct {
	Name       string `json:"name"`
	Loaded     bool   `json:"loaded"`
	Size       int64  `json:"size,omitempty"`       // Size in bytes
	MemoryUsed int64  `json:"memoryUsed,omitempty"` // Memory used when loaded
	Backend    string `json:"backend,omitempty"`    // Which backend owns this model
}

// ModelsHandler handles model-related API requests.
type ModelsHandler struct {
	// provider is the model provider (loom manager or mock)
	provider ModelProvider

	// mu protects concurrent access
	mu sync.RWMutex
}

// NewModelsHandler creates a new ModelsHandler with the given provider.
func NewModelsHandler(provider ModelProvider) *ModelsHandler {
	return &ModelsHandler{
		provider: provider,
	}
}

// RegisterRoutes registers the model API routes on the router.
func (h *ModelsHandler) RegisterRoutes(router *Router) {
	router.GET("/api/models", h.ListModels)
	router.GET("/api/models/:name", h.GetModel)
	router.POST("/api/models/:name/load", h.LoadModel)
	router.POST("/api/models/:name/unload", h.UnloadModel)
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// ModelListResponse is the JSON response for GET /api/models.
type ModelListResponse struct {
	Models []ModelInfo `json:"models"`
}

// ModelLoadRequest is the optional JSON request body for POST /api/models/:name/load.
type ModelLoadRequest struct {
	// Device specifies the device to load the model on (e.g., "cuda:0", "cpu")
	Device string `json:"device,omitempty"`
}

// ModelActionResponse is the JSON response for model load/unload operations.
type ModelActionResponse struct {
	Name    string `json:"name"`
	Action  string `json:"action"`
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ListModels handles GET /api/models.
// It returns a list of all available models with their status.
func (h *ModelsHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	provider := h.provider
	h.mu.RUnlock()

	if provider == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_provider",
			"Model provider is not available")
		return
	}

	ctx := r.Context()
	models, err := provider.ListModels(ctx)
	if err != nil {
		WriteError(w, http.StatusInternalServerError, "list_failed",
			"Failed to list models: "+err.Error())
		return
	}

	// Sort by name for consistent ordering
	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	response := ModelListResponse{
		Models: models,
	}

	WriteJSON(w, http.StatusOK, response)
}

// GetModel handles GET /api/models/:name.
// It returns details for a specific model.
func (h *ModelsHandler) GetModel(w http.ResponseWriter, r *http.Request) {
	modelName := PathParam(r, "name")
	if modelName == "" {
		WriteError(w, http.StatusBadRequest, "missing_model_name",
			"Model name is required in the URL path")
		return
	}

	h.mu.RLock()
	provider := h.provider
	h.mu.RUnlock()

	if provider == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_provider",
			"Model provider is not available")
		return
	}

	ctx := r.Context()
	model, err := provider.GetModel(ctx, modelName)
	if err != nil {
		WriteError(w, http.StatusNotFound, "model_not_found",
			"Model '"+modelName+"' not found: "+err.Error())
		return
	}

	WriteJSON(w, http.StatusOK, model)
}

// LoadModel handles POST /api/models/:name/load.
// It loads the specified model into memory.
func (h *ModelsHandler) LoadModel(w http.ResponseWriter, r *http.Request) {
	modelName := PathParam(r, "name")
	if modelName == "" {
		WriteError(w, http.StatusBadRequest, "missing_model_name",
			"Model name is required in the URL path")
		return
	}

	h.mu.RLock()
	provider := h.provider
	h.mu.RUnlock()

	if provider == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_provider",
			"Model provider is not available")
		return
	}

	// Parse optional request body for device specification
	var loadReq ModelLoadRequest
	if r.Body != nil && r.ContentLength > 0 {
		if err := ReadJSON(r, &loadReq); err != nil {
			// Ignore parse errors - device is optional
			_ = loadReq
		}
	}

	ctx := r.Context()
	if err := provider.LoadModel(ctx, modelName); err != nil {
		WriteError(w, http.StatusInternalServerError, "load_failed",
			"Failed to load model '"+modelName+"': "+err.Error())
		return
	}

	response := ModelActionResponse{
		Name:    modelName,
		Action:  "load",
		Success: true,
		Message: "Model '" + modelName + "' loaded successfully",
	}

	WriteJSON(w, http.StatusOK, response)
}

// UnloadModel handles POST /api/models/:name/unload.
// It unloads the specified model from memory.
func (h *ModelsHandler) UnloadModel(w http.ResponseWriter, r *http.Request) {
	modelName := PathParam(r, "name")
	if modelName == "" {
		WriteError(w, http.StatusBadRequest, "missing_model_name",
			"Model name is required in the URL path")
		return
	}

	h.mu.RLock()
	provider := h.provider
	h.mu.RUnlock()

	if provider == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_provider",
			"Model provider is not available")
		return
	}

	ctx := r.Context()
	if err := provider.UnloadModel(ctx, modelName); err != nil {
		WriteError(w, http.StatusInternalServerError, "unload_failed",
			"Failed to unload model '"+modelName+"': "+err.Error())
		return
	}

	response := ModelActionResponse{
		Name:    modelName,
		Action:  "unload",
		Success: true,
		Message: "Model '" + modelName + "' unloaded successfully",
	}

	WriteJSON(w, http.StatusOK, response)
}

// -----------------------------------------------------------------------------
// Mock Provider for Testing
// -----------------------------------------------------------------------------

// MockModelProvider is a mock implementation of ModelProvider for testing.
// It allows tests to run without a real backend.
type MockModelProvider struct {
	models map[string]*ModelInfo
	mu     sync.RWMutex

	// LoadError allows tests to simulate load failures
	LoadError error
	// UnloadError allows tests to simulate unload failures
	UnloadError error
	// ListError allows tests to simulate list failures
	ListError error
}

// Ensure MockModelProvider implements ModelProvider.
var _ ModelProvider = (*MockModelProvider)(nil)

// NewMockModelProvider creates a new mock model provider.
func NewMockModelProvider() *MockModelProvider {
	return &MockModelProvider{
		models: make(map[string]*ModelInfo),
	}
}

// AddMockModel adds a mock model to the provider.
func (m *MockModelProvider) AddMockModel(model ModelInfo) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.models[model.Name] = &model
}

// ListModels returns all mock models.
func (m *MockModelProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.ListError != nil {
		return nil, m.ListError
	}

	models := make([]ModelInfo, 0, len(m.models))
	for _, model := range m.models {
		models = append(models, *model)
	}
	return models, nil
}

// GetModel returns a specific mock model.
func (m *MockModelProvider) GetModel(ctx context.Context, name string) (*ModelInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	model, ok := m.models[name]
	if !ok {
		return nil, &modelNotFoundError{name: name}
	}
	return model, nil
}

// LoadModel simulates loading a model.
func (m *MockModelProvider) LoadModel(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.LoadError != nil {
		return m.LoadError
	}

	model, ok := m.models[name]
	if !ok {
		return &modelNotFoundError{name: name}
	}

	model.Loaded = true
	return nil
}

// UnloadModel simulates unloading a model.
func (m *MockModelProvider) UnloadModel(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.UnloadError != nil {
		return m.UnloadError
	}

	model, ok := m.models[name]
	if !ok {
		return &modelNotFoundError{name: name}
	}

	model.Loaded = false
	return nil
}

// modelNotFoundError is an error indicating a model was not found.
type modelNotFoundError struct {
	name string
}

func (e *modelNotFoundError) Error() string {
	return "model not found: " + e.name
}
