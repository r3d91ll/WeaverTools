package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/r3d91ll/weaver/pkg/backend"
)

// -----------------------------------------------------------------------------
// NewBackendsHandler Tests
// -----------------------------------------------------------------------------

func TestNewBackendsHandler(t *testing.T) {
	t.Run("with nil registry", func(t *testing.T) {
		h := NewBackendsHandler(nil)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with mock registry", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		h := NewBackendsHandler(mockReg)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with real backend registry", func(t *testing.T) {
		registry := backend.NewRegistry()
		h := NewBackendsHandlerWithRegistry(registry)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})
}

// -----------------------------------------------------------------------------
// ListBackends Tests
// -----------------------------------------------------------------------------

func TestBackendsHandler_ListBackends(t *testing.T) {
	t.Run("returns empty list when no backends", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		h := NewBackendsHandler(mockReg)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends", nil)
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
		backends := data["backends"].([]interface{})
		if len(backends) != 0 {
			t.Errorf("Expected empty backends list, got %d backends", len(backends))
		}
	})

	t.Run("returns backends when registered", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		mockReg.AddMockBackend(backend.Status{
			Name:      "test-backend",
			Type:      backend.TypeLoom,
			Available: true,
			Capabilities: backend.Capabilities{
				ContextLimit:      32768,
				SupportsTools:     true,
				SupportsStreaming: true,
				SupportsHidden:    true,
				MaxTokens:         4096,
			},
		})

		h := NewBackendsHandler(mockReg)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends", nil)
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
		backends := data["backends"].([]interface{})
		if len(backends) != 1 {
			t.Errorf("Expected 1 backend, got %d", len(backends))
		}

		be := backends[0].(map[string]interface{})
		if be["name"] != "test-backend" {
			t.Errorf("Expected backend name 'test-backend', got %v", be["name"])
		}
		if be["type"] != "loom" {
			t.Errorf("Expected backend type 'loom', got %v", be["type"])
		}
		if be["available"] != true {
			t.Error("Expected backend to be available")
		}

		caps := be["capabilities"].(map[string]interface{})
		if caps["contextLimit"].(float64) != 32768 {
			t.Errorf("Expected contextLimit 32768, got %v", caps["contextLimit"])
		}
		if caps["supportsTools"] != true {
			t.Error("Expected supportsTools to be true")
		}
	})

	t.Run("returns multiple backends sorted by name", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		mockReg.AddMockBackend(backend.Status{
			Name:      "zebra-backend",
			Type:      backend.TypeLoom,
			Available: true,
		})
		mockReg.AddMockBackend(backend.Status{
			Name:      "alpha-backend",
			Type:      backend.TypeClaudeCode,
			Available: false,
		})

		h := NewBackendsHandler(mockReg)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends", nil)
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
		backends := data["backends"].([]interface{})
		if len(backends) != 2 {
			t.Errorf("Expected 2 backends, got %d", len(backends))
		}

		// Check alphabetical ordering
		first := backends[0].(map[string]interface{})
		second := backends[1].(map[string]interface{})
		if first["name"] != "alpha-backend" {
			t.Errorf("Expected first backend to be 'alpha-backend', got %v", first["name"])
		}
		if second["name"] != "zebra-backend" {
			t.Errorf("Expected second backend to be 'zebra-backend', got %v", second["name"])
		}
	})

	t.Run("returns error when registry is nil", func(t *testing.T) {
		h := NewBackendsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_registry" {
			t.Error("Expected error code 'no_registry'")
		}
	})

	t.Run("returns backends with real registry", func(t *testing.T) {
		registry := backend.NewRegistry()
		// Note: Not registering any backends to keep test isolated
		h := NewBackendsHandlerWithRegistry(registry)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}
	})
}

// -----------------------------------------------------------------------------
// GetBackend Tests
// -----------------------------------------------------------------------------

func TestBackendsHandler_GetBackend(t *testing.T) {
	t.Run("returns backend when found", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		mockReg.AddMockBackend(backend.Status{
			Name:      "test-backend",
			Type:      backend.TypeLoom,
			Available: true,
			Capabilities: backend.Capabilities{
				ContextLimit:      32768,
				SupportsTools:     true,
				SupportsStreaming: true,
				SupportsHidden:    true,
				MaxTokens:         4096,
			},
		})

		h := NewBackendsHandler(mockReg)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends/test-backend", nil)
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
		if data["name"] != "test-backend" {
			t.Errorf("Expected backend name 'test-backend', got %v", data["name"])
		}
		if data["type"] != "loom" {
			t.Errorf("Expected backend type 'loom', got %v", data["type"])
		}
		if data["available"] != true {
			t.Error("Expected backend to be available")
		}
	})

	t.Run("returns error when backend not found", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		mockReg.AddMockBackend(backend.Status{
			Name: "existing-backend",
			Type: backend.TypeLoom,
		})

		h := NewBackendsHandler(mockReg)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "backend_not_found" {
			t.Error("Expected error code 'backend_not_found'")
		}
	})

	t.Run("returns error when registry is nil", func(t *testing.T) {
		h := NewBackendsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends/test", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_registry" {
			t.Error("Expected error code 'no_registry'")
		}
	})

	t.Run("shows available backends in error message", func(t *testing.T) {
		mockReg := NewMockBackendRegistry()
		mockReg.AddMockBackend(backend.Status{Name: "backend-a", Type: backend.TypeLoom})
		mockReg.AddMockBackend(backend.Status{Name: "backend-b", Type: backend.TypeClaudeCode})

		h := NewBackendsHandler(mockReg)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/backends/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Error == nil {
			t.Fatal("Expected error in response")
		}

		// Error message should mention available backends
		if resp.Error.Message == "" {
			t.Error("Expected non-empty error message")
		}
	})
}

// -----------------------------------------------------------------------------
// Helper Function Tests
// -----------------------------------------------------------------------------

func TestJoinBackendNames(t *testing.T) {
	tests := []struct {
		names    []string
		expected string
	}{
		{nil, "(none)"},
		{[]string{}, "(none)"},
		{[]string{"backend1"}, "backend1"},
		{[]string{"backend1", "backend2"}, "backend1, backend2"},
		{[]string{"zebra", "alpha", "beta"}, "alpha, beta, zebra"}, // Should be sorted
	}

	for _, tt := range tests {
		result := joinBackendNames(tt.names)
		if result != tt.expected {
			t.Errorf("joinBackendNames(%v) = %q, expected %q", tt.names, result, tt.expected)
		}
	}
}

// -----------------------------------------------------------------------------
// Route Registration Tests
// -----------------------------------------------------------------------------

func TestBackendsHandler_RegisterRoutes(t *testing.T) {
	mockReg := NewMockBackendRegistry()
	h := NewBackendsHandler(mockReg)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/backends"},
		{http.MethodGet, "/api/backends/test"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			// Should not get 404 from router (route not found)
			// Note: getting 404 from handler (backend not found) is OK
			if rec.Code == http.StatusNotFound {
				resp := parseAPIResponse(t, rec.Body)
				// Only fail if this is the router's "not found" error, not our handler's error
				if resp.Error != nil && resp.Error.Code == "not_found" {
					t.Errorf("Route %s %s not found", tt.method, tt.path)
				}
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Mock Registry Tests
// -----------------------------------------------------------------------------

func TestMockBackendRegistry(t *testing.T) {
	t.Run("add and get mock backend", func(t *testing.T) {
		reg := NewMockBackendRegistry()
		reg.AddMockBackend(backend.Status{
			Name:      "test-backend",
			Type:      backend.TypeLoom,
			Available: true,
			Capabilities: backend.Capabilities{
				ContextLimit: 32768,
				MaxTokens:    4096,
			},
		})

		status := reg.Status(context.Background())
		if len(status) != 1 {
			t.Errorf("Expected 1 backend, got %d", len(status))
		}

		be, ok := status["test-backend"]
		if !ok {
			t.Error("Expected to find 'test-backend'")
		}
		if be.Name != "test-backend" {
			t.Errorf("Expected name 'test-backend', got %s", be.Name)
		}
		if be.Type != backend.TypeLoom {
			t.Errorf("Expected type 'loom', got %s", be.Type)
		}
		if !be.Available {
			t.Error("Expected backend to be available")
		}
	})

	t.Run("list backends", func(t *testing.T) {
		reg := NewMockBackendRegistry()
		reg.AddMockBackend(backend.Status{Name: "backend1"})
		reg.AddMockBackend(backend.Status{Name: "backend2"})

		names := reg.List()
		if len(names) != 2 {
			t.Errorf("Expected 2 backends, got %d", len(names))
		}
	})

	t.Run("empty registry", func(t *testing.T) {
		reg := NewMockBackendRegistry()

		status := reg.Status(context.Background())
		if len(status) != 0 {
			t.Errorf("Expected 0 backends, got %d", len(status))
		}

		names := reg.List()
		if len(names) != 0 {
			t.Errorf("Expected 0 names, got %d", len(names))
		}
	})
}

// -----------------------------------------------------------------------------
// Interface Verification Tests
// -----------------------------------------------------------------------------

func TestBackendRegistryInterface(t *testing.T) {
	// Verify MockBackendRegistry implements BackendRegistry
	var _ BackendRegistry = (*MockBackendRegistry)(nil)

	// Verify backendRegistryAdapter implements BackendRegistry
	var _ BackendRegistry = (*backendRegistryAdapter)(nil)
}

// -----------------------------------------------------------------------------
// Concurrency Tests
// -----------------------------------------------------------------------------

func TestBackendsHandler_Concurrent(t *testing.T) {
	mockReg := NewMockBackendRegistry()
	mockReg.AddMockBackend(backend.Status{
		Name:      "concurrent-backend",
		Type:      backend.TypeLoom,
		Available: true,
	})

	h := NewBackendsHandler(mockReg)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Run concurrent requests
	const numRequests = 100
	done := make(chan bool, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			req := httptest.NewRequest(http.MethodGet, "/api/backends", nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Errorf("Expected status 200, got %d", rec.Code)
			}
			done <- true
		}()
	}

	// Wait for all requests to complete
	for i := 0; i < numRequests; i++ {
		<-done
	}
}

// -----------------------------------------------------------------------------
// Backend Type Tests
// -----------------------------------------------------------------------------

func TestBackendInfo_Types(t *testing.T) {
	tests := []struct {
		backendType backend.Type
		expected    string
	}{
		{backend.TypeLoom, "loom"},
		{backend.TypeClaudeCode, "claudecode"},
	}

	for _, tt := range tests {
		t.Run(string(tt.backendType), func(t *testing.T) {
			mockReg := NewMockBackendRegistry()
			mockReg.AddMockBackend(backend.Status{
				Name: "test",
				Type: tt.backendType,
			})

			h := NewBackendsHandler(mockReg)
			router := NewRouter()
			h.RegisterRoutes(router)

			req := httptest.NewRequest(http.MethodGet, "/api/backends/test", nil)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Errorf("Expected status 200, got %d", rec.Code)
			}

			resp := parseAPIResponse(t, rec.Body)
			data := resp.Data.(map[string]interface{})
			if data["type"] != tt.expected {
				t.Errorf("Expected type %q, got %v", tt.expected, data["type"])
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Capabilities Tests
// -----------------------------------------------------------------------------

func TestBackendCapabilities(t *testing.T) {
	mockReg := NewMockBackendRegistry()
	mockReg.AddMockBackend(backend.Status{
		Name:      "full-featured",
		Type:      backend.TypeLoom,
		Available: true,
		Capabilities: backend.Capabilities{
			ContextLimit:      65536,
			SupportsTools:     true,
			SupportsStreaming: true,
			SupportsHidden:    true,
			MaxTokens:         8192,
		},
	})

	h := NewBackendsHandler(mockReg)
	router := NewRouter()
	h.RegisterRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/api/backends/full-featured", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", rec.Code)
	}

	resp := parseAPIResponse(t, rec.Body)
	data := resp.Data.(map[string]interface{})
	caps := data["capabilities"].(map[string]interface{})

	if caps["contextLimit"].(float64) != 65536 {
		t.Errorf("Expected contextLimit 65536, got %v", caps["contextLimit"])
	}
	if caps["supportsTools"] != true {
		t.Error("Expected supportsTools to be true")
	}
	if caps["supportsStreaming"] != true {
		t.Error("Expected supportsStreaming to be true")
	}
	if caps["supportsHidden"] != true {
		t.Error("Expected supportsHidden to be true")
	}
	if caps["maxTokens"].(float64) != 8192 {
		t.Errorf("Expected maxTokens 8192, got %v", caps["maxTokens"])
	}
}
