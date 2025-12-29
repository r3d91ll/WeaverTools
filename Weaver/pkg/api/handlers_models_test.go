package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// NewModelsHandler Tests
// -----------------------------------------------------------------------------

func TestNewModelsHandler(t *testing.T) {
	t.Run("with nil provider", func(t *testing.T) {
		h := NewModelsHandler(nil)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with mock provider", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		h := NewModelsHandler(mockProvider)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})
}

// -----------------------------------------------------------------------------
// ListModels Tests
// -----------------------------------------------------------------------------

func TestModelsHandler_ListModels(t *testing.T) {
	t.Run("returns empty list when no models", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
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
		models := data["models"].([]interface{})
		if len(models) != 0 {
			t.Errorf("Expected empty models list, got %d models", len(models))
		}
	})

	t.Run("returns models when available", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:       "test-model",
			Loaded:     true,
			Size:       1024 * 1024 * 100, // 100MB
			MemoryUsed: 1024 * 1024 * 200, // 200MB
			Backend:    "loom",
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
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
		models := data["models"].([]interface{})
		if len(models) != 1 {
			t.Errorf("Expected 1 model, got %d", len(models))
		}

		model := models[0].(map[string]interface{})
		if model["name"] != "test-model" {
			t.Errorf("Expected model name 'test-model', got %v", model["name"])
		}
		if model["loaded"] != true {
			t.Error("Expected model to be loaded")
		}
		if model["backend"] != "loom" {
			t.Errorf("Expected backend 'loom', got %v", model["backend"])
		}
	})

	t.Run("returns multiple models sorted by name", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:   "zebra-model",
			Loaded: false,
		})
		mockProvider.AddMockModel(ModelInfo{
			Name:   "alpha-model",
			Loaded: true,
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
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
		models := data["models"].([]interface{})
		if len(models) != 2 {
			t.Errorf("Expected 2 models, got %d", len(models))
		}

		// Check alphabetical ordering
		first := models[0].(map[string]interface{})
		second := models[1].(map[string]interface{})
		if first["name"] != "alpha-model" {
			t.Errorf("Expected first model to be 'alpha-model', got %v", first["name"])
		}
		if second["name"] != "zebra-model" {
			t.Errorf("Expected second model to be 'zebra-model', got %v", second["name"])
		}
	})

	t.Run("returns error when provider is nil", func(t *testing.T) {
		h := NewModelsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_provider" {
			t.Error("Expected error code 'no_provider'")
		}
	})

	t.Run("returns error when list fails", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.ListError = errors.New("connection failed")

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "list_failed" {
			t.Error("Expected error code 'list_failed'")
		}
	})
}

// -----------------------------------------------------------------------------
// GetModel Tests
// -----------------------------------------------------------------------------

func TestModelsHandler_GetModel(t *testing.T) {
	t.Run("returns model when found", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:       "test-model",
			Loaded:     true,
			Size:       1024 * 1024 * 100,
			MemoryUsed: 1024 * 1024 * 200,
			Backend:    "loom",
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models/test-model", nil)
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
		if data["name"] != "test-model" {
			t.Errorf("Expected model name 'test-model', got %v", data["name"])
		}
		if data["loaded"] != true {
			t.Error("Expected model to be loaded")
		}
	})

	t.Run("returns error when model not found", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name: "existing-model",
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "model_not_found" {
			t.Error("Expected error code 'model_not_found'")
		}
	})

	t.Run("returns error when provider is nil", func(t *testing.T) {
		h := NewModelsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/models/test", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_provider" {
			t.Error("Expected error code 'no_provider'")
		}
	})
}

// -----------------------------------------------------------------------------
// LoadModel Tests
// -----------------------------------------------------------------------------

func TestModelsHandler_LoadModel(t *testing.T) {
	t.Run("loads model successfully", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:   "test-model",
			Loaded: false,
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/test-model/load", nil)
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
		if data["name"] != "test-model" {
			t.Errorf("Expected model name 'test-model', got %v", data["name"])
		}
		if data["action"] != "load" {
			t.Errorf("Expected action 'load', got %v", data["action"])
		}
		if data["success"] != true {
			t.Error("Expected success to be true")
		}

		// Verify model is now loaded
		model, _ := mockProvider.GetModel(context.Background(), "test-model")
		if !model.Loaded {
			t.Error("Expected model to be loaded in provider")
		}
	})

	t.Run("accepts optional device parameter", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:   "test-model",
			Loaded: false,
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		body := strings.NewReader(`{"device":"cuda:0"}`)
		req := httptest.NewRequest(http.MethodPost, "/api/models/test-model/load", body)
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
	})

	t.Run("returns error when model not found", func(t *testing.T) {
		mockProvider := NewMockModelProvider()

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/nonexistent/load", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "load_failed" {
			t.Error("Expected error code 'load_failed'")
		}
	})

	t.Run("returns error when load fails", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:   "test-model",
			Loaded: false,
		})
		mockProvider.LoadError = errors.New("GPU out of memory")

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/test-model/load", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "load_failed" {
			t.Error("Expected error code 'load_failed'")
		}
	})

	t.Run("returns error when provider is nil", func(t *testing.T) {
		h := NewModelsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/test/load", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_provider" {
			t.Error("Expected error code 'no_provider'")
		}
	})
}

// -----------------------------------------------------------------------------
// UnloadModel Tests
// -----------------------------------------------------------------------------

func TestModelsHandler_UnloadModel(t *testing.T) {
	t.Run("unloads model successfully", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:   "test-model",
			Loaded: true,
		})

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/test-model/unload", nil)
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
		if data["name"] != "test-model" {
			t.Errorf("Expected model name 'test-model', got %v", data["name"])
		}
		if data["action"] != "unload" {
			t.Errorf("Expected action 'unload', got %v", data["action"])
		}
		if data["success"] != true {
			t.Error("Expected success to be true")
		}

		// Verify model is now unloaded
		model, _ := mockProvider.GetModel(context.Background(), "test-model")
		if model.Loaded {
			t.Error("Expected model to be unloaded in provider")
		}
	})

	t.Run("returns error when model not found", func(t *testing.T) {
		mockProvider := NewMockModelProvider()

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/nonexistent/unload", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "unload_failed" {
			t.Error("Expected error code 'unload_failed'")
		}
	})

	t.Run("returns error when unload fails", func(t *testing.T) {
		mockProvider := NewMockModelProvider()
		mockProvider.AddMockModel(ModelInfo{
			Name:   "test-model",
			Loaded: true,
		})
		mockProvider.UnloadError = errors.New("model is in use")

		h := NewModelsHandler(mockProvider)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/test-model/unload", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusInternalServerError {
			t.Errorf("Expected status 500, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "unload_failed" {
			t.Error("Expected error code 'unload_failed'")
		}
	})

	t.Run("returns error when provider is nil", func(t *testing.T) {
		h := NewModelsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/models/test/unload", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_provider" {
			t.Error("Expected error code 'no_provider'")
		}
	})
}

// -----------------------------------------------------------------------------
// Route Registration Tests
// -----------------------------------------------------------------------------

func TestModelsHandler_RegisterRoutes(t *testing.T) {
	mockProvider := NewMockModelProvider()
	h := NewModelsHandler(mockProvider)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/models"},
		{http.MethodGet, "/api/models/test"},
		{http.MethodPost, "/api/models/test/load"},
		{http.MethodPost, "/api/models/test/unload"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			// Should not get 404 from router (route not found)
			// Note: getting 404 from handler (model not found) is OK
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
// Mock Provider Tests
// -----------------------------------------------------------------------------

func TestMockModelProvider(t *testing.T) {
	t.Run("add and get mock model", func(t *testing.T) {
		provider := NewMockModelProvider()
		provider.AddMockModel(ModelInfo{
			Name:       "test-model",
			Loaded:     true,
			Size:       1024 * 1024 * 100,
			MemoryUsed: 1024 * 1024 * 200,
			Backend:    "loom",
		})

		model, err := provider.GetModel(context.Background(), "test-model")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if model.Name != "test-model" {
			t.Errorf("Expected name 'test-model', got %s", model.Name)
		}
		if !model.Loaded {
			t.Error("Expected model to be loaded")
		}
	})

	t.Run("list models", func(t *testing.T) {
		provider := NewMockModelProvider()
		provider.AddMockModel(ModelInfo{Name: "model1"})
		provider.AddMockModel(ModelInfo{Name: "model2"})

		models, err := provider.ListModels(context.Background())
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if len(models) != 2 {
			t.Errorf("Expected 2 models, got %d", len(models))
		}
	})

	t.Run("load and unload model", func(t *testing.T) {
		provider := NewMockModelProvider()
		provider.AddMockModel(ModelInfo{
			Name:   "test-model",
			Loaded: false,
		})

		// Load
		err := provider.LoadModel(context.Background(), "test-model")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		model, _ := provider.GetModel(context.Background(), "test-model")
		if !model.Loaded {
			t.Error("Expected model to be loaded")
		}

		// Unload
		err = provider.UnloadModel(context.Background(), "test-model")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		model, _ = provider.GetModel(context.Background(), "test-model")
		if model.Loaded {
			t.Error("Expected model to be unloaded")
		}
	})

	t.Run("empty provider", func(t *testing.T) {
		provider := NewMockModelProvider()

		models, err := provider.ListModels(context.Background())
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if len(models) != 0 {
			t.Errorf("Expected 0 models, got %d", len(models))
		}
	})
}

// -----------------------------------------------------------------------------
// Interface Verification Tests
// -----------------------------------------------------------------------------

func TestModelProviderInterface(t *testing.T) {
	// Verify MockModelProvider implements ModelProvider
	var _ ModelProvider = (*MockModelProvider)(nil)
}

// -----------------------------------------------------------------------------
// Concurrency Tests
// -----------------------------------------------------------------------------

func TestModelsHandler_Concurrent(t *testing.T) {
	mockProvider := NewMockModelProvider()
	mockProvider.AddMockModel(ModelInfo{
		Name:   "concurrent-model",
		Loaded: true,
	})

	h := NewModelsHandler(mockProvider)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Run concurrent requests
	const numRequests = 100
	errChan := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				errChan <- fmt.Errorf("expected status 200, got %d", rec.Code)
				return
			}
			errChan <- nil
		}()
	}

	// Wait for all requests and collect errors
	for i := 0; i < numRequests; i++ {
		if err := <-errChan; err != nil {
			t.Error(err)
		}
	}
}

// -----------------------------------------------------------------------------
// ModelInfo Field Tests
// -----------------------------------------------------------------------------

func TestModelInfo_AllFields(t *testing.T) {
	mockProvider := NewMockModelProvider()
	mockProvider.AddMockModel(ModelInfo{
		Name:       "full-model",
		Loaded:     true,
		Size:       1024 * 1024 * 500, // 500MB
		MemoryUsed: 1024 * 1024 * 750, // 750MB (models expand in memory)
		Backend:    "loom",
	})

	h := NewModelsHandler(mockProvider)
	router := NewRouter()
	h.RegisterRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/api/models/full-model", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", rec.Code)
	}

	resp := parseAPIResponse(t, rec.Body)
	data := resp.Data.(map[string]interface{})

	if data["size"].(float64) != 1024*1024*500 {
		t.Errorf("Expected size 524288000, got %v", data["size"])
	}
	if data["memoryUsed"].(float64) != 1024*1024*750 {
		t.Errorf("Expected memoryUsed 786432000, got %v", data["memoryUsed"])
	}
	if data["backend"] != "loom" {
		t.Errorf("Expected backend 'loom', got %v", data["backend"])
	}
}

// -----------------------------------------------------------------------------
// Error Message Tests
// -----------------------------------------------------------------------------

func TestModelNotFoundError(t *testing.T) {
	err := &modelNotFoundError{name: "missing-model"}
	expected := "model not found: missing-model"
	if err.Error() != expected {
		t.Errorf("Expected error message %q, got %q", expected, err.Error())
	}
}
