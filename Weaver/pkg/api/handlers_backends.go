// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"context"
	"net/http"
	"sort"
	"sync"

	"github.com/r3d91ll/weaver/pkg/backend"
)

// BackendRegistry is the interface for accessing backend status.
// This interface allows for dependency injection and testing.
type BackendRegistry interface {
	// Status returns availability status for all backends.
	Status(ctx context.Context) map[string]backend.Status
	// List returns all registered backend names.
	List() []string
}

// BackendsHandler handles backend-related API requests.
type BackendsHandler struct {
	// registry is the backend registry (backend.Registry or mock)
	registry BackendRegistry

	// mu protects concurrent access
	mu sync.RWMutex
}

// NewBackendsHandler creates a new BackendsHandler with the given registry.
func NewBackendsHandler(registry BackendRegistry) *BackendsHandler {
	return &BackendsHandler{
		registry: registry,
	}
}

// NewBackendsHandlerWithRegistry creates a new BackendsHandler with a backend.Registry.
// This is a convenience function for production use.
func NewBackendsHandlerWithRegistry(registry *backend.Registry) *BackendsHandler {
	return &BackendsHandler{
		registry: &backendRegistryAdapter{registry: registry},
	}
}

// backendRegistryAdapter adapts *backend.Registry to the BackendRegistry interface.
type backendRegistryAdapter struct {
	registry *backend.Registry
}

func (a *backendRegistryAdapter) Status(ctx context.Context) map[string]backend.Status {
	return a.registry.Status(ctx)
}

func (a *backendRegistryAdapter) List() []string {
	return a.registry.List()
}

// RegisterRoutes registers the backend API routes on the router.
func (h *BackendsHandler) RegisterRoutes(router *Router) {
	router.GET("/api/backends", h.ListBackends)
	router.GET("/api/backends/:name", h.GetBackend)
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// BackendListResponse is the JSON response for GET /api/backends.
type BackendListResponse struct {
	Backends []BackendInfo `json:"backends"`
}

// BackendInfo is the JSON representation of a backend's information.
type BackendInfo struct {
	Name         string              `json:"name"`
	Type         string              `json:"type"`
	Available    bool                `json:"available"`
	Capabilities BackendCapabilities `json:"capabilities"`
}

// BackendCapabilities is the JSON representation of backend capabilities.
type BackendCapabilities struct {
	ContextLimit      int  `json:"contextLimit"`
	SupportsTools     bool `json:"supportsTools"`
	SupportsStreaming bool `json:"supportsStreaming"`
	SupportsHidden    bool `json:"supportsHidden"`
	MaxTokens         int  `json:"maxTokens"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ListBackends handles GET /api/backends.
// It returns a list of all registered backends with their status.
func (h *BackendsHandler) ListBackends(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	registry := h.registry
	h.mu.RUnlock()

	if registry == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_registry",
			"Backend registry is not available")
		return
	}

	// Get backend status from registry
	ctx := r.Context()
	statusMap := registry.Status(ctx)

	// Convert to response format and sort by name for consistent output
	backends := make([]BackendInfo, 0, len(statusMap))
	for _, status := range statusMap {
		backends = append(backends, BackendInfo{
			Name:      status.Name,
			Type:      string(status.Type),
			Available: status.Available,
			Capabilities: BackendCapabilities{
				ContextLimit:      status.Capabilities.ContextLimit,
				SupportsTools:     status.Capabilities.SupportsTools,
				SupportsStreaming: status.Capabilities.SupportsStreaming,
				SupportsHidden:    status.Capabilities.SupportsHidden,
				MaxTokens:         status.Capabilities.MaxTokens,
			},
		})
	}

	// Sort by name for consistent ordering
	sort.Slice(backends, func(i, j int) bool {
		return backends[i].Name < backends[j].Name
	})

	response := BackendListResponse{
		Backends: backends,
	}

	WriteJSON(w, http.StatusOK, response)
}

// GetBackend handles GET /api/backends/:name.
// It returns details for a specific backend.
func (h *BackendsHandler) GetBackend(w http.ResponseWriter, r *http.Request) {
	// Get backend name from path
	backendName := PathParam(r, "name")
	if backendName == "" {
		WriteError(w, http.StatusBadRequest, "missing_backend_name",
			"Backend name is required in the URL path")
		return
	}

	h.mu.RLock()
	registry := h.registry
	h.mu.RUnlock()

	if registry == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_registry",
			"Backend registry is not available")
		return
	}

	// Get backend status from registry
	ctx := r.Context()
	statusMap := registry.Status(ctx)

	// Find the requested backend
	status, found := statusMap[backendName]
	if !found {
		availableBackends := registry.List()
		WriteError(w, http.StatusNotFound, "backend_not_found",
			"Backend '"+backendName+"' not found. Available backends: "+joinBackendNames(availableBackends))
		return
	}

	// Convert to response format
	backendInfo := BackendInfo{
		Name:      status.Name,
		Type:      string(status.Type),
		Available: status.Available,
		Capabilities: BackendCapabilities{
			ContextLimit:      status.Capabilities.ContextLimit,
			SupportsTools:     status.Capabilities.SupportsTools,
			SupportsStreaming: status.Capabilities.SupportsStreaming,
			SupportsHidden:    status.Capabilities.SupportsHidden,
			MaxTokens:         status.Capabilities.MaxTokens,
		},
	}

	WriteJSON(w, http.StatusOK, backendInfo)
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// joinBackendNames joins backend names into a comma-separated string.
func joinBackendNames(names []string) string {
	if len(names) == 0 {
		return "(none)"
	}

	// Sort for consistent output
	sorted := make([]string, len(names))
	copy(sorted, names)
	sort.Strings(sorted)

	result := ""
	for i, name := range sorted {
		if i > 0 {
			result += ", "
		}
		result += name
	}
	return result
}

// -----------------------------------------------------------------------------
// Mock Registry for Testing
// -----------------------------------------------------------------------------

// MockBackendRegistry is a mock implementation of BackendRegistry for testing.
// It allows tests to run without a real backend.
type MockBackendRegistry struct {
	backends map[string]backend.Status
	mu       sync.RWMutex
}

// Ensure MockBackendRegistry implements BackendRegistry.
var _ BackendRegistry = (*MockBackendRegistry)(nil)

// NewMockBackendRegistry creates a new mock backend registry.
func NewMockBackendRegistry() *MockBackendRegistry {
	return &MockBackendRegistry{
		backends: make(map[string]backend.Status),
	}
}

// AddMockBackend adds a mock backend to the registry.
func (m *MockBackendRegistry) AddMockBackend(status backend.Status) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.backends[status.Name] = status
}

// Status returns status for all mock backends.
func (m *MockBackendRegistry) Status(ctx context.Context) map[string]backend.Status {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]backend.Status, len(m.backends))
	for name, status := range m.backends {
		result[name] = status
	}
	return result
}

// List returns all mock backend names.
func (m *MockBackendRegistry) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.backends))
	for name := range m.backends {
		names = append(names, name)
	}
	return names
}
