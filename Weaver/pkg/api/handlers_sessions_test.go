package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// -----------------------------------------------------------------------------
// NewSessionsHandler Tests
// -----------------------------------------------------------------------------

func TestNewSessionsHandler(t *testing.T) {
	t.Run("with nil store", func(t *testing.T) {
		h := NewSessionsHandler(nil)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})

	t.Run("with memory store", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		if h == nil {
			t.Error("Expected handler to be created")
		}
	})
}

// -----------------------------------------------------------------------------
// ListSessions Tests
// -----------------------------------------------------------------------------

func TestSessionsHandler_ListSessions(t *testing.T) {
	t.Run("returns empty list when no sessions", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
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
		sessions := data["sessions"].([]interface{})
		if len(sessions) != 0 {
			t.Errorf("Expected empty sessions list, got %d sessions", len(sessions))
		}
	})

	t.Run("returns sessions when present", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:          "test-session-1",
			Name:        "Test Session",
			Description: "A test session",
			StartedAt:   time.Now().UTC(),
			Config: SessionAPIConfig{
				MeasurementMode: "active",
				AutoExport:      true,
				ExportPath:      "./exports",
			},
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
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
		sessions := data["sessions"].([]interface{})
		if len(sessions) != 1 {
			t.Errorf("Expected 1 session, got %d", len(sessions))
		}

		s := sessions[0].(map[string]interface{})
		if s["name"] != "Test Session" {
			t.Errorf("Expected session name 'Test Session', got %v", s["name"])
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewSessionsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/sessions", nil)
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
// GetSession Tests
// -----------------------------------------------------------------------------

func TestSessionsHandler_GetSession(t *testing.T) {
	t.Run("returns session when found", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:          "test-session-1",
			Name:        "Test Session",
			Description: "A test session",
			StartedAt:   time.Now().UTC(),
			Config: SessionAPIConfig{
				MeasurementMode: "passive",
				AutoExport:      false,
				ExportPath:      "./custom",
			},
			Metadata: map[string]any{
				"key": "value",
			},
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/sessions/test-session-1", nil)
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
		if data["name"] != "Test Session" {
			t.Errorf("Expected name 'Test Session', got %v", data["name"])
		}
		if data["description"] != "A test session" {
			t.Errorf("Expected description 'A test session', got %v", data["description"])
		}

		config := data["config"].(map[string]interface{})
		if config["measurementMode"] != "passive" {
			t.Errorf("Expected measurementMode 'passive', got %v", config["measurementMode"])
		}
	})

	t.Run("returns 404 when not found", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/sessions/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "session_not_found" {
			t.Error("Expected error code 'session_not_found'")
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewSessionsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodGet, "/api/sessions/test-id", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// CreateSession Tests
// -----------------------------------------------------------------------------

func TestSessionsHandler_CreateSession(t *testing.T) {
	t.Run("creates session with minimal data", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := CreateSessionRequest{
			Name: "New Session",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions", bytes.NewReader(bodyBytes))
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
		if data["name"] != "New Session" {
			t.Errorf("Expected name 'New Session', got %v", data["name"])
		}
		if data["id"] == "" {
			t.Error("Expected session ID to be generated")
		}

		// Check defaults
		config := data["config"].(map[string]interface{})
		if config["measurementMode"] != "active" {
			t.Errorf("Expected default measurementMode 'active', got %v", config["measurementMode"])
		}
		if config["autoExport"] != true {
			t.Errorf("Expected default autoExport true, got %v", config["autoExport"])
		}
	})

	t.Run("creates session with full data", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := CreateSessionRequest{
			Name:        "Full Session",
			Description: "A complete session",
			Config: &SessionAPIConfig{
				MeasurementMode: "triggered",
				AutoExport:      false,
				ExportPath:      "/custom/path",
			},
			Metadata: map[string]any{
				"project": "test-project",
				"version": 1,
			},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusCreated {
			t.Errorf("Expected status 201, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})

		if data["description"] != "A complete session" {
			t.Errorf("Expected description 'A complete session', got %v", data["description"])
		}

		config := data["config"].(map[string]interface{})
		if config["measurementMode"] != "triggered" {
			t.Errorf("Expected measurementMode 'triggered', got %v", config["measurementMode"])
		}

		metadata := data["metadata"].(map[string]interface{})
		if metadata["project"] != "test-project" {
			t.Errorf("Expected metadata project 'test-project', got %v", metadata["project"])
		}
	})

	t.Run("returns error for missing name", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := CreateSessionRequest{
			Description: "No name provided",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions", bytes.NewReader(bodyBytes))
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
		if resp.Error == nil || resp.Error.Code != "missing_name" {
			t.Error("Expected error code 'missing_name'")
		}
	})

	t.Run("returns error for invalid measurement mode", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := CreateSessionRequest{
			Name: "Test Session",
			Config: &SessionAPIConfig{
				MeasurementMode: "invalid_mode",
			},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions", bytes.NewReader(bodyBytes))
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
		if resp.Error == nil || resp.Error.Code != "invalid_measurement_mode" {
			t.Error("Expected error code 'invalid_measurement_mode'")
		}
	})

	t.Run("returns error for invalid JSON", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions", bytes.NewReader([]byte("not valid json")))
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
		h := NewSessionsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := CreateSessionRequest{
			Name: "Test Session",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// UpdateSession Tests
// -----------------------------------------------------------------------------

func TestSessionsHandler_UpdateSession(t *testing.T) {
	t.Run("updates session name", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Original Name",
			StartedAt: time.Now().UTC(),
			Config: SessionAPIConfig{
				MeasurementMode: "active",
			},
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := UpdateSessionRequest{
			Name: "Updated Name",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPut, "/api/sessions/test-session-1", bytes.NewReader(bodyBytes))
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
		if data["name"] != "Updated Name" {
			t.Errorf("Expected name 'Updated Name', got %v", data["name"])
		}
	})

	t.Run("updates session config", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
			Config: SessionAPIConfig{
				MeasurementMode: "active",
				AutoExport:      true,
			},
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := UpdateSessionRequest{
			Config: &SessionAPIConfig{
				MeasurementMode: "passive",
				AutoExport:      false,
				ExportPath:      "/new/path",
			},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPut, "/api/sessions/test-session-1", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		config := data["config"].(map[string]interface{})

		if config["measurementMode"] != "passive" {
			t.Errorf("Expected measurementMode 'passive', got %v", config["measurementMode"])
		}
		if config["autoExport"] != false {
			t.Errorf("Expected autoExport false, got %v", config["autoExport"])
		}
	})

	t.Run("returns error for invalid measurement mode", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
			Config: SessionAPIConfig{
				MeasurementMode: "active",
			},
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := UpdateSessionRequest{
			Config: &SessionAPIConfig{
				MeasurementMode: "invalid",
			},
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPut, "/api/sessions/test-session-1", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Error == nil || resp.Error.Code != "invalid_measurement_mode" {
			t.Error("Expected error code 'invalid_measurement_mode'")
		}
	})

	t.Run("returns 404 when session not found", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		requestBody := UpdateSessionRequest{
			Name: "Updated Name",
		}
		bodyBytes, _ := json.Marshal(requestBody)

		req := httptest.NewRequest(http.MethodPut, "/api/sessions/nonexistent", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}
	})

	t.Run("returns error for invalid JSON", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPut, "/api/sessions/test-id", bytes.NewReader([]byte("not valid json")))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// DeleteSession Tests
// -----------------------------------------------------------------------------

func TestSessionsHandler_DeleteSession(t *testing.T) {
	t.Run("deletes existing session", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodDelete, "/api/sessions/test-session-1", nil)
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
		if data["id"] != "test-session-1" {
			t.Errorf("Expected id 'test-session-1', got %v", data["id"])
		}

		// Verify session is actually deleted
		_, ok := store.Get("test-session-1")
		if ok {
			t.Error("Expected session to be deleted")
		}
	})

	t.Run("returns 404 when session not found", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodDelete, "/api/sessions/nonexistent", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "session_not_found" {
			t.Error("Expected error code 'session_not_found'")
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewSessionsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodDelete, "/api/sessions/test-id", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("Expected status 503, got %d", rec.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// EndSession Tests
// -----------------------------------------------------------------------------

func TestSessionsHandler_EndSession(t *testing.T) {
	t.Run("ends active session", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions/test-session-1/end", nil)
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
		if data["endedAt"] == nil {
			t.Error("Expected endedAt to be set")
		}
	})

	t.Run("returns error when session already ended", func(t *testing.T) {
		store := NewMemorySessionStore()
		now := time.Now().UTC()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: now.Add(-1 * time.Hour),
			EndedAt:   &now,
		}
		store.Create(session)

		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions/test-session-1/end", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Errorf("Expected status 400, got %d", rec.Code)
		}

		resp := parseAPIResponse(t, rec.Body)
		if resp.Success {
			t.Error("Expected error response")
		}
		if resp.Error == nil || resp.Error.Code != "already_ended" {
			t.Error("Expected error code 'already_ended'")
		}
	})

	t.Run("returns 404 when session not found", func(t *testing.T) {
		store := NewMemorySessionStore()
		h := NewSessionsHandler(store)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions/nonexistent/end", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Errorf("Expected status 404, got %d", rec.Code)
		}
	})

	t.Run("returns error when store is nil", func(t *testing.T) {
		h := NewSessionsHandler(nil)
		router := NewRouter()
		h.RegisterRoutes(router)

		req := httptest.NewRequest(http.MethodPost, "/api/sessions/test-id/end", nil)
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

func TestSessionsHandler_RegisterRoutes(t *testing.T) {
	store := NewMemorySessionStore()
	h := NewSessionsHandler(store)
	router := NewRouter()
	h.RegisterRoutes(router)

	// Test that routes are registered by checking if they respond
	tests := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/sessions"},
		{http.MethodGet, "/api/sessions/test-id"},
		{http.MethodPost, "/api/sessions"},
		{http.MethodPut, "/api/sessions/test-id"},
		{http.MethodDelete, "/api/sessions/test-id"},
		{http.MethodPost, "/api/sessions/test-id/end"},
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			var body *bytes.Reader
			if tt.method == http.MethodPost || tt.method == http.MethodPut {
				body = bytes.NewReader([]byte(`{"name":"test"}`))
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
// MemorySessionStore Tests
// -----------------------------------------------------------------------------

func TestMemorySessionStore(t *testing.T) {
	t.Run("create and get session", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
		}

		err := store.Create(session)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		retrieved, ok := store.Get("test-session-1")
		if !ok {
			t.Error("Expected to find session")
		}
		if retrieved.Name != "Test Session" {
			t.Errorf("Expected name 'Test Session', got %s", retrieved.Name)
		}
	})

	t.Run("create duplicate returns error", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
		}

		store.Create(session)
		err := store.Create(session)

		if err == nil {
			t.Error("Expected error for duplicate session")
		}

		sessErr, ok := err.(*SessionError)
		if !ok {
			t.Error("Expected SessionError")
		}
		if sessErr.Code != "already_exists" {
			t.Errorf("Expected error code 'already_exists', got %s", sessErr.Code)
		}
	})

	t.Run("get nonexistent session", func(t *testing.T) {
		store := NewMemorySessionStore()
		_, ok := store.Get("nonexistent")
		if ok {
			t.Error("Expected not to find session")
		}
	})

	t.Run("list sessions", func(t *testing.T) {
		store := NewMemorySessionStore()
		store.Create(&Session{ID: "session-1", Name: "Session 1", StartedAt: time.Now().UTC()})
		store.Create(&Session{ID: "session-2", Name: "Session 2", StartedAt: time.Now().UTC()})

		sessions := store.List()
		if len(sessions) != 2 {
			t.Errorf("Expected 2 sessions, got %d", len(sessions))
		}
	})

	t.Run("update session", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Original Name",
			StartedAt: time.Now().UTC(),
		}
		store.Create(session)

		session.Name = "Updated Name"
		err := store.Update(session)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		retrieved, _ := store.Get("test-session-1")
		if retrieved.Name != "Updated Name" {
			t.Errorf("Expected name 'Updated Name', got %s", retrieved.Name)
		}
	})

	t.Run("update nonexistent session", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:   "nonexistent",
			Name: "Test",
		}

		err := store.Update(session)
		if err == nil {
			t.Error("Expected error for nonexistent session")
		}

		sessErr, ok := err.(*SessionError)
		if !ok {
			t.Error("Expected SessionError")
		}
		if sessErr.Code != "not_found" {
			t.Errorf("Expected error code 'not_found', got %s", sessErr.Code)
		}
	})

	t.Run("delete session", func(t *testing.T) {
		store := NewMemorySessionStore()
		session := &Session{
			ID:        "test-session-1",
			Name:      "Test Session",
			StartedAt: time.Now().UTC(),
		}
		store.Create(session)

		err := store.Delete("test-session-1")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		_, ok := store.Get("test-session-1")
		if ok {
			t.Error("Expected session to be deleted")
		}
	})

	t.Run("delete nonexistent session", func(t *testing.T) {
		store := NewMemorySessionStore()

		err := store.Delete("nonexistent")
		if err == nil {
			t.Error("Expected error for nonexistent session")
		}

		sessErr, ok := err.(*SessionError)
		if !ok {
			t.Error("Expected SessionError")
		}
		if sessErr.Code != "not_found" {
			t.Errorf("Expected error code 'not_found', got %s", sessErr.Code)
		}
	})
}

// -----------------------------------------------------------------------------
// Helper Function Tests
// -----------------------------------------------------------------------------

func TestIsValidMeasurementMode(t *testing.T) {
	tests := []struct {
		mode     string
		expected bool
	}{
		{"passive", true},
		{"active", true},
		{"triggered", true},
		{"invalid", false},
		{"", false},
		{"ACTIVE", false}, // Case sensitive
	}

	for _, tt := range tests {
		t.Run(tt.mode, func(t *testing.T) {
			result := isValidMeasurementMode(tt.mode)
			if result != tt.expected {
				t.Errorf("isValidMeasurementMode(%q) = %v, expected %v", tt.mode, result, tt.expected)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Interface Verification Tests
// -----------------------------------------------------------------------------

func TestSessionStoreInterface(t *testing.T) {
	// Verify MemorySessionStore implements SessionStore
	var _ SessionStore = (*MemorySessionStore)(nil)
}
