package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/r3d91ll/weaver/pkg/concepts"
	"github.com/r3d91ll/yarn"
)

// -----------------------------------------------------------------------------
// NewConceptsHandler Tests
// -----------------------------------------------------------------------------

func TestNewConceptsHandler(t *testing.T) {
	t.Run("with nil store", func(t *testing.T) {
		h := NewConceptsHandler(nil)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with mock store", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})
}

// -----------------------------------------------------------------------------
// ListConcepts Tests
// -----------------------------------------------------------------------------

func TestConceptsHandler_ListConcepts(t *testing.T) {
	t.Run("returns empty list when no concepts", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts", nil)
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
		conceptsList := data["concepts"].([]interface{})
		if len(conceptsList) != 0 {
			t.Errorf("Expected empty concepts list, got %d concepts", len(conceptsList))
		}
	})

	t.Run("returns concepts when present", func(t *testing.T) {
		store := NewMockConceptStore()
		now := time.Now().UTC()
		store.AddMockConcept(&concepts.Concept{
			Name: "recursion",
			Samples: []concepts.Sample{
				{ID: "sample-1", Content: "Example of recursion", ExtractedAt: now},
				{ID: "sample-2", Content: "Another example", ExtractedAt: now},
			},
			CreatedAt: now,
			UpdatedAt: now,
		})

		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts", nil)
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
		conceptsList := data["concepts"].([]interface{})
		if len(conceptsList) != 1 {
			t.Errorf("Expected 1 concept, got %d", len(conceptsList))
		}

		c := conceptsList[0].(map[string]interface{})
		if c["name"] != "recursion" {
			t.Errorf("Expected name 'recursion', got %v", c["name"])
		}
		if int(c["sampleCount"].(float64)) != 2 {
			t.Errorf("Expected sampleCount 2, got %v", c["sampleCount"])
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewConceptsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "no_store" {
			t.Error("Expected error code 'no_store'")
		}
	})
}

// -----------------------------------------------------------------------------
// GetConcept Tests
// -----------------------------------------------------------------------------

func TestConceptsHandler_GetConcept(t *testing.T) {
	t.Run("returns concept when found", func(t *testing.T) {
		store := NewMockConceptStore()
		now := time.Now().UTC()
		store.AddMockConcept(&concepts.Concept{
			Name: "sorting",
			Samples: []concepts.Sample{
				{
					ID:          "sample-1",
					Content:     "Bubble sort example",
					Model:       "llama-7b",
					ExtractedAt: now,
					HiddenState: &yarn.HiddenState{
						Vector: []float32{1.0, 2.0, 3.0},
						Layer:  0,
					},
				},
			},
			CreatedAt: now,
			UpdatedAt: now,
		})

		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts/sorting", nil)
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
		if data["name"] != "sorting" {
			t.Errorf("Expected name 'sorting', got %v", data["name"])
		}
		if int(data["sampleCount"].(float64)) != 1 {
			t.Errorf("Expected sampleCount 1, got %v", data["sampleCount"])
		}
		if int(data["dimension"].(float64)) != 3 {
			t.Errorf("Expected dimension 3, got %v", data["dimension"])
		}

		samples := data["samples"].([]interface{})
		if len(samples) != 1 {
			t.Errorf("Expected 1 sample, got %d", len(samples))
		}

		s := samples[0].(map[string]interface{})
		if s["content"] != "Bubble sort example" {
			t.Errorf("Expected content 'Bubble sort example', got %v", s["content"])
		}
		if s["hasVector"] != true {
			t.Errorf("Expected hasVector true, got %v", s["hasVector"])
		}
	})

	t.Run("returns 404 when concept not found", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "concept_not_found" {
			t.Error("Expected error code 'concept_not_found'")
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewConceptsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts/test", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// GetStats Tests
// -----------------------------------------------------------------------------

func TestConceptsHandler_GetStats(t *testing.T) {
	t.Run("returns empty stats when no concepts", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts/stats", nil)
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
		if int(data["conceptCount"].(float64)) != 0 {
			t.Errorf("Expected conceptCount 0, got %v", data["conceptCount"])
		}
		if int(data["totalSamples"].(float64)) != 0 {
			t.Errorf("Expected totalSamples 0, got %v", data["totalSamples"])
		}
	})

	t.Run("returns stats when concepts present", func(t *testing.T) {
		store := NewMockConceptStore()
		now := time.Now().UTC()
		store.AddMockConcept(&concepts.Concept{
			Name: "recursion",
			Samples: []concepts.Sample{
				{
					ID:          "sample-1",
					Content:     "Example 1",
					Model:       "llama-7b",
					ExtractedAt: now,
					HiddenState: &yarn.HiddenState{
						Vector: []float32{1.0, 2.0, 3.0, 4.0},
						Layer:  0,
					},
				},
				{
					ID:          "sample-2",
					Content:     "Example 2",
					Model:       "llama-7b",
					ExtractedAt: now,
					HiddenState: &yarn.HiddenState{
						Vector: []float32{4.0, 5.0, 6.0, 7.0},
						Layer:  0,
					},
				},
			},
			CreatedAt: now,
			UpdatedAt: now,
		})

		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts/stats", nil)
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
		if int(data["conceptCount"].(float64)) != 1 {
			t.Errorf("Expected conceptCount 1, got %v", data["conceptCount"])
		}
		if int(data["totalSamples"].(float64)) != 2 {
			t.Errorf("Expected totalSamples 2, got %v", data["totalSamples"])
		}
		if int(data["healthyConcepts"].(float64)) != 1 {
			t.Errorf("Expected healthyConcepts 1, got %v", data["healthyConcepts"])
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewConceptsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/concepts/stats", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// AddSample Tests
// -----------------------------------------------------------------------------

func TestConceptsHandler_AddSample(t *testing.T) {
	t.Run("creates new concept and adds sample", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := AddSampleRequest{
			Content: "Example of a linked list",
			Model:   "llama-7b",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/concepts/linked-list/samples", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", rec.Code)
			t.Logf("Response: %s", rec.Body.String())
		}

		resp := parseAPIResponse(t, rec.Body)
		if !resp.Success {
			t.Error("Expected success response")
		}

		data := resp.Data.(map[string]interface{})
		if data["conceptName"] != "linked-list" {
			t.Errorf("Expected conceptName 'linked-list', got %v", data["conceptName"])
		}
		if data["content"] != "Example of a linked list" {
			t.Errorf("Expected content 'Example of a linked list', got %v", data["content"])
		}
		if data["id"] == "" {
			t.Error("Expected sample ID to be generated")
		}

		// Verify sample was added to store
		concept, ok := store.Get("linked-list")
		if !ok {
			t.Error("Expected concept to be created in store")
		}
		if len(concept.Samples) != 1 {
			t.Errorf("Expected 1 sample, got %d", len(concept.Samples))
		}
	})

	t.Run("adds sample to existing concept", func(t *testing.T) {
		store := NewMockConceptStore()
		now := time.Now().UTC()
		store.AddMockConcept(&concepts.Concept{
			Name: "recursion",
			Samples: []concepts.Sample{
				{ID: "existing-sample", Content: "Existing example", ExtractedAt: now},
			},
			CreatedAt: now,
			UpdatedAt: now,
		})

		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := AddSampleRequest{
			Content: "New recursion example",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/concepts/recursion/samples", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", rec.Code)
		}

		// Verify sample was added
		concept, ok := store.Get("recursion")
		if !ok {
			t.Error("Expected concept to exist")
		}
		if len(concept.Samples) != 2 {
			t.Errorf("Expected 2 samples, got %d", len(concept.Samples))
		}
	})

	t.Run("adds sample with hidden state", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := AddSampleRequest{
			Content: "Example with vectors",
			Model:   "llama-7b",
			HiddenState: &HiddenStateAPI{
				Vector: []float32{1.0, 2.0, 3.0, 4.0},
				Layer:  5,
				Dtype:  "float32",
			},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/concepts/vectors/samples", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", rec.Code)
		}

		// Verify hidden state was stored
		concept, ok := store.Get("vectors")
		if !ok {
			t.Error("Expected concept to be created")
		}
		if len(concept.Samples) != 1 {
			t.Errorf("Expected 1 sample, got %d", len(concept.Samples))
		}
		if concept.Samples[0].HiddenState == nil {
			t.Error("Expected hidden state to be present")
		}
		if len(concept.Samples[0].HiddenState.Vector) != 4 {
			t.Errorf("Expected vector length 4, got %d", len(concept.Samples[0].HiddenState.Vector))
		}
	})

	t.Run("returns error for missing content", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := AddSampleRequest{
			Model: "llama-7b",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/concepts/test/samples", bytes.NewReader(bodyBytes))
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
		if resp.Error == nil || resp.Error.Code != "missing_content" {
			t.Error("Expected error code 'missing_content'")
		}
	})

	t.Run("returns error for invalid JSON", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/concepts/test/samples", bytes.NewReader([]byte("not valid json")))
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

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewConceptsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := AddSampleRequest{
			Content: "Test content",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/concepts/test/samples", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// DeleteConcept Tests
// -----------------------------------------------------------------------------

func TestConceptsHandler_DeleteConcept(t *testing.T) {
	t.Run("deletes existing concept", func(t *testing.T) {
		store := NewMockConceptStore()
		now := time.Now().UTC()
		store.AddMockConcept(&concepts.Concept{
			Name:      "to-delete",
			Samples:   []concepts.Sample{{ID: "sample-1", Content: "Example", ExtractedAt: now}},
			CreatedAt: now,
			UpdatedAt: now,
		})

		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodDelete, "/api/concepts/to-delete", nil)
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
		if data["name"] != "to-delete" {
			t.Errorf("Expected name 'to-delete', got %v", data["name"])
		}

		// Verify concept is actually deleted
		_, ok := store.Get("to-delete")
		if ok {
			t.Error("Expected concept to be deleted")
		}
	})

	t.Run("returns 404 when concept not found", func(t *testing.T) {
		store := NewMockConceptStore()
		h := NewConceptsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodDelete, "/api/concepts/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "concept_not_found" {
			t.Error("Expected error code 'concept_not_found'")
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewConceptsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodDelete, "/api/concepts/test", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// Route Registration Tests
// -----------------------------------------------------------------------------

func TestConceptsHandler_RegisterRoutes(t *testing.T) {
	store := NewMockConceptStore()
	h := NewConceptsHandler(store)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/concepts"},
		{http.MethodGet, "/api/concepts/stats"},
		{http.MethodGet, "/api/concepts/test-concept"},
		{http.MethodPost, "/api/concepts/test-concept/samples"},
		{http.MethodDelete, "/api/concepts/test-concept"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			var body *bytes.Reader
			if tt.method == http.MethodPost {
				body = bytes.NewReader([]byte(`{"content":"test"}`))
			}

			var req *http.Request
			if body != nil {
				req = httptest.NewRequest(tt.method, tt.path, body)
				req.Header.Set("Content-Type", "application/json")
			} else {
				req = httptest.NewRequest(tt.method, tt.path, nil)
			}
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			// Should not get 404 (not found means route isn't registered)
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
// MockConceptStore Tests
// -----------------------------------------------------------------------------

func TestMockConceptStore(t *testing.T) {
	t.Run("add and get concept", func(t *testing.T) {
		store := NewMockConceptStore()
		sample := concepts.Sample{
			ID:          "sample-1",
			Content:     "Test content",
			ExtractedAt: time.Now().UTC(),
		}

		err := store.Add("test-concept", sample)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		concept, ok := store.Get("test-concept")
		if !ok {
			t.Error("Expected to find concept")
		}
		if concept.Name != "test-concept" {
			t.Errorf("Expected name 'test-concept', got %s", concept.Name)
		}
		if len(concept.Samples) != 1 {
			t.Errorf("Expected 1 sample, got %d", len(concept.Samples))
		}
	})

	t.Run("add returns error for empty name", func(t *testing.T) {
		store := NewMockConceptStore()
		sample := concepts.Sample{
			ID:      "sample-1",
			Content: "Test content",
		}

		err := store.Add("", sample)
		if err == nil {
			t.Error("Expected error for empty concept name")
		}

		conceptErr, ok := err.(*ConceptError)
		if !ok {
			t.Error("Expected ConceptError")
		}
		if conceptErr.Code != "empty_name" {
			t.Errorf("Expected error code 'empty_name', got %s", conceptErr.Code)
		}
	})

	t.Run("add returns error for empty sample ID", func(t *testing.T) {
		store := NewMockConceptStore()
		sample := concepts.Sample{
			Content: "Test content",
		}

		err := store.Add("test-concept", sample)
		if err == nil {
			t.Error("Expected error for empty sample ID")
		}

		conceptErr, ok := err.(*ConceptError)
		if !ok {
			t.Error("Expected ConceptError")
		}
		if conceptErr.Code != "empty_sample_id" {
			t.Errorf("Expected error code 'empty_sample_id', got %s", conceptErr.Code)
		}
	})

	t.Run("get nonexistent concept", func(t *testing.T) {
		store := NewMockConceptStore()
		_, ok := store.Get("nonexistent")
		if ok {
			t.Error("Expected not to find concept")
		}
	})

	t.Run("list concepts", func(t *testing.T) {
		store := NewMockConceptStore()
		store.Add("concept-1", concepts.Sample{ID: "s1", Content: "c1"})
		store.Add("concept-1", concepts.Sample{ID: "s2", Content: "c2"})
		store.Add("concept-2", concepts.Sample{ID: "s3", Content: "c3"})

		list := store.List()
		if len(list) != 2 {
			t.Errorf("Expected 2 concepts, got %d", len(list))
		}
		if list["concept-1"] != 2 {
			t.Errorf("Expected concept-1 to have 2 samples, got %d", list["concept-1"])
		}
		if list["concept-2"] != 1 {
			t.Errorf("Expected concept-2 to have 1 sample, got %d", list["concept-2"])
		}
	})

	t.Run("clear concept", func(t *testing.T) {
		store := NewMockConceptStore()
		store.Add("to-clear", concepts.Sample{ID: "s1", Content: "c1"})

		ok := store.Clear("to-clear")
		if !ok {
			t.Error("Expected clear to succeed")
		}

		_, found := store.Get("to-clear")
		if found {
			t.Error("Expected concept to be cleared")
		}
	})

	t.Run("clear nonexistent concept", func(t *testing.T) {
		store := NewMockConceptStore()

		ok := store.Clear("nonexistent")
		if ok {
			t.Error("Expected clear to return false for nonexistent concept")
		}
	})

	t.Run("stats returns aggregate data", func(t *testing.T) {
		store := NewMockConceptStore()
		now := time.Now().UTC()
		store.AddMockConcept(&concepts.Concept{
			Name: "recursion",
			Samples: []concepts.Sample{
				{ID: "s1", Content: "c1", Model: "llama-7b", ExtractedAt: now},
				{ID: "s2", Content: "c2", Model: "llama-7b", ExtractedAt: now},
			},
			CreatedAt: now,
			UpdatedAt: now,
		})
		store.AddMockConcept(&concepts.Concept{
			Name: "sorting",
			Samples: []concepts.Sample{
				{ID: "s3", Content: "c3", Model: "mistral-7b", ExtractedAt: now},
			},
			CreatedAt: now,
			UpdatedAt: now,
		})

		stats := store.Stats()
		if stats.ConceptCount != 2 {
			t.Errorf("Expected conceptCount 2, got %d", stats.ConceptCount)
		}
		if stats.TotalSamples != 3 {
			t.Errorf("Expected totalSamples 3, got %d", stats.TotalSamples)
		}
		if stats.Models["llama-7b"] != 2 {
			t.Errorf("Expected llama-7b usage 2, got %d", stats.Models["llama-7b"])
		}
		if stats.Models["mistral-7b"] != 1 {
			t.Errorf("Expected mistral-7b usage 1, got %d", stats.Models["mistral-7b"])
		}
	})
}

// -----------------------------------------------------------------------------
// Interface Verification Tests
// -----------------------------------------------------------------------------

func TestConceptStoreInterface(t *testing.T) {
	// Verify MockConceptStore implements ConceptStore
	var _ ConceptStore = (*MockConceptStore)(nil)
}
