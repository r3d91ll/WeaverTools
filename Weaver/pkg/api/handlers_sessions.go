// Package api provides the HTTP/WebSocket server for the Weaver web UI.
package api

import (
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
)

// SessionStore is the interface for managing sessions.
// This interface allows for dependency injection and testing.
type SessionStore interface {
	// List returns all sessions.
	List() []*Session
	// Get retrieves a session by ID.
	Get(id string) (*Session, bool)
	// Create creates a new session.
	Create(session *Session) error
	// Update updates an existing session.
	Update(session *Session) error
	// Delete removes a session by ID.
	Delete(id string) error
}

// Session represents a research session for the API.
type Session struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	StartedAt   time.Time         `json:"startedAt"`
	EndedAt     *time.Time        `json:"endedAt,omitempty"`
	Config      SessionAPIConfig  `json:"config"`
	Metadata    map[string]any    `json:"metadata,omitempty"`
	Stats       *SessionAPIStats  `json:"stats,omitempty"`
}

// SessionAPIConfig holds session configuration for the API.
type SessionAPIConfig struct {
	MeasurementMode string `json:"measurementMode"`
	AutoExport      bool   `json:"autoExport"`
	ExportPath      string `json:"exportPath"`
}

// SessionAPIStats holds session statistics for the API.
type SessionAPIStats struct {
	ConversationCount int     `json:"conversationCount"`
	MessageCount      int     `json:"messageCount"`
	MeasurementCount  int     `json:"measurementCount"`
	BilateralCount    int     `json:"bilateralCount"`
	AvgDEff           float64 `json:"avgDEff"`
	AvgBeta           float64 `json:"avgBeta"`
	AvgAlignment      float64 `json:"avgAlignment"`
}

// SessionsHandler handles session-related API requests.
type SessionsHandler struct {
	// store is the session store (in-memory or persistent)
	store SessionStore

	// mu protects concurrent access
	mu sync.RWMutex
}

// NewSessionsHandler creates a new SessionsHandler with the given session store.
func NewSessionsHandler(store SessionStore) *SessionsHandler {
	return &SessionsHandler{
		store: store,
	}
}

// RegisterRoutes registers the session API routes on the router.
func (h *SessionsHandler) RegisterRoutes(router *Router) {
	router.GET("/api/sessions", h.ListSessions)
	router.GET("/api/sessions/:id", h.GetSession)
	router.POST("/api/sessions", h.CreateSession)
	router.PUT("/api/sessions/:id", h.UpdateSession)
	router.DELETE("/api/sessions/:id", h.DeleteSession)
	router.POST("/api/sessions/:id/end", h.EndSession)
	router.GET("/api/sessions/:id/messages", h.GetSessionMessages)
}

// -----------------------------------------------------------------------------
// API Request Types
// -----------------------------------------------------------------------------

// CreateSessionRequest is the expected JSON body for POST /api/sessions.
type CreateSessionRequest struct {
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Config      *SessionAPIConfig `json:"config,omitempty"`
	Metadata    map[string]any    `json:"metadata,omitempty"`
}

// UpdateSessionRequest is the expected JSON body for PUT /api/sessions/:id.
type UpdateSessionRequest struct {
	Name        string            `json:"name,omitempty"`
	Description string            `json:"description,omitempty"`
	Config      *SessionAPIConfig `json:"config,omitempty"`
	Metadata    map[string]any    `json:"metadata,omitempty"`
}

// -----------------------------------------------------------------------------
// API Response Types
// -----------------------------------------------------------------------------

// SessionListResponse is the JSON response for GET /api/sessions.
type SessionListResponse struct {
	Sessions []*Session `json:"sessions"`
	Total    int        `json:"total"`
}

// SingleSessionResponse is the JSON response for GET /api/sessions/:id.
type SingleSessionResponse struct {
	Session *Session `json:"session"`
}

// MessagesResponse is the JSON response for GET /api/sessions/:id/messages.
type MessagesResponse struct {
	Messages []any `json:"messages"`
	Total    int   `json:"total"`
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------

// ListSessions handles GET /api/sessions.
// It returns a list of all sessions.
func (h *SessionsHandler) ListSessions(w http.ResponseWriter, r *http.Request) {
	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	sessions := store.List()
	if sessions == nil {
		sessions = []*Session{}
	}

	response := SessionListResponse{
		Sessions: sessions,
		Total:    len(sessions),
	}

	WriteJSON(w, http.StatusOK, response)
}

// GetSession handles GET /api/sessions/:id.
// It returns a specific session by ID.
func (h *SessionsHandler) GetSession(w http.ResponseWriter, r *http.Request) {
	sessionID := PathParam(r, "id")
	if sessionID == "" {
		WriteError(w, http.StatusBadRequest, "missing_session_id",
			"Session ID is required in the URL path")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	session, ok := store.Get(sessionID)
	if !ok {
		WriteError(w, http.StatusNotFound, "session_not_found",
			"Session '"+sessionID+"' not found")
		return
	}

	response := SingleSessionResponse{
		Session: session,
	}
	WriteJSON(w, http.StatusOK, response)
}

// CreateSession handles POST /api/sessions.
// It creates a new session with the provided data.
func (h *SessionsHandler) CreateSession(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req CreateSessionRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	// Validate required fields
	if req.Name == "" {
		WriteError(w, http.StatusBadRequest, "missing_name",
			"Session name is required")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	// Create new session
	session := &Session{
		ID:          uuid.New().String(),
		Name:        req.Name,
		Description: req.Description,
		StartedAt:   time.Now().UTC(),
		Metadata:    req.Metadata,
	}

	// Apply config with defaults
	if req.Config != nil {
		session.Config = *req.Config
	} else {
		session.Config = SessionAPIConfig{
			MeasurementMode: "active",
			AutoExport:      true,
			ExportPath:      "./experiments",
		}
	}

	// Validate measurement mode
	if session.Config.MeasurementMode != "" && !isValidMeasurementMode(session.Config.MeasurementMode) {
		WriteError(w, http.StatusBadRequest, "invalid_measurement_mode",
			"Invalid measurement mode '"+session.Config.MeasurementMode+"', must be one of: passive, active, triggered")
		return
	}

	// Set empty measurement mode to default
	if session.Config.MeasurementMode == "" {
		session.Config.MeasurementMode = "active"
	}

	// Initialize metadata if nil
	if session.Metadata == nil {
		session.Metadata = make(map[string]any)
	}

	// Initialize stats
	session.Stats = &SessionAPIStats{}

	if err := store.Create(session); err != nil {
		WriteError(w, http.StatusInternalServerError, "create_error",
			"Failed to create session: "+err.Error())
		return
	}

	response := SingleSessionResponse{
		Session: session,
	}
	WriteJSON(w, http.StatusCreated, response)
}

// UpdateSession handles PUT /api/sessions/:id.
// It updates an existing session with the provided data.
func (h *SessionsHandler) UpdateSession(w http.ResponseWriter, r *http.Request) {
	sessionID := PathParam(r, "id")
	if sessionID == "" {
		WriteError(w, http.StatusBadRequest, "missing_session_id",
			"Session ID is required in the URL path")
		return
	}

	// Parse request body
	var req UpdateSessionRequest
	if err := ReadJSON(r, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_json",
			"Failed to parse request body: "+err.Error())
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	// Get existing session
	session, ok := store.Get(sessionID)
	if !ok {
		WriteError(w, http.StatusNotFound, "session_not_found",
			"Session '"+sessionID+"' not found")
		return
	}

	// Update fields if provided
	if req.Name != "" {
		session.Name = req.Name
	}
	if req.Description != "" {
		session.Description = req.Description
	}
	if req.Config != nil {
		// Validate measurement mode if provided
		if req.Config.MeasurementMode != "" && !isValidMeasurementMode(req.Config.MeasurementMode) {
			WriteError(w, http.StatusBadRequest, "invalid_measurement_mode",
				"Invalid measurement mode '"+req.Config.MeasurementMode+"', must be one of: passive, active, triggered")
			return
		}
		session.Config = *req.Config
	}
	if req.Metadata != nil {
		session.Metadata = req.Metadata
	}

	if err := store.Update(session); err != nil {
		// Check if session was deleted between Get and Update (TOCTOU race)
		var sessionErr *SessionError
		if errors.As(err, &sessionErr) && sessionErr.Code == "not_found" {
			WriteError(w, http.StatusNotFound, "session_not_found",
				"Session '"+sessionID+"' not found")
			return
		}
		WriteError(w, http.StatusInternalServerError, "update_error",
			"Failed to update session: "+err.Error())
		return
	}

	response := SingleSessionResponse{
		Session: session,
	}
	WriteJSON(w, http.StatusOK, response)
}

// DeleteSession handles DELETE /api/sessions/:id.
// It removes a session by ID.
func (h *SessionsHandler) DeleteSession(w http.ResponseWriter, r *http.Request) {
	sessionID := PathParam(r, "id")
	if sessionID == "" {
		WriteError(w, http.StatusBadRequest, "missing_session_id",
			"Session ID is required in the URL path")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	// Check if session exists
	_, ok := store.Get(sessionID)
	if !ok {
		WriteError(w, http.StatusNotFound, "session_not_found",
			"Session '"+sessionID+"' not found")
		return
	}

	if err := store.Delete(sessionID); err != nil {
		WriteError(w, http.StatusInternalServerError, "delete_error",
			"Failed to delete session: "+err.Error())
		return
	}

	WriteJSON(w, http.StatusOK, map[string]string{
		"message": "Session deleted successfully",
		"id":      sessionID,
	})
}

// EndSession handles POST /api/sessions/:id/end.
// It marks a session as ended.
func (h *SessionsHandler) EndSession(w http.ResponseWriter, r *http.Request) {
	sessionID := PathParam(r, "id")
	if sessionID == "" {
		WriteError(w, http.StatusBadRequest, "missing_session_id",
			"Session ID is required in the URL path")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	// Get existing session
	session, ok := store.Get(sessionID)
	if !ok {
		WriteError(w, http.StatusNotFound, "session_not_found",
			"Session '"+sessionID+"' not found")
		return
	}

	// Check if already ended
	if session.EndedAt != nil {
		WriteError(w, http.StatusBadRequest, "already_ended",
			"Session has already ended")
		return
	}

	// Set end time
	now := time.Now().UTC()
	session.EndedAt = &now

	if err := store.Update(session); err != nil {
		// Check if session was deleted between Get and Update (TOCTOU race)
		var sessionErr *SessionError
		if errors.As(err, &sessionErr) && sessionErr.Code == "not_found" {
			WriteError(w, http.StatusNotFound, "session_not_found",
				"Session '"+sessionID+"' not found")
			return
		}
		WriteError(w, http.StatusInternalServerError, "update_error",
			"Failed to end session: "+err.Error())
		return
	}

	response := SingleSessionResponse{
		Session: session,
	}
	WriteJSON(w, http.StatusOK, response)
}

// GetSessionMessages handles GET /api/sessions/:id/messages.
// It returns messages for a specific session.
// Note: Currently returns empty since sessions don't store messages yet.
func (h *SessionsHandler) GetSessionMessages(w http.ResponseWriter, r *http.Request) {
	sessionID := PathParam(r, "id")
	if sessionID == "" {
		WriteError(w, http.StatusBadRequest, "missing_session_id",
			"Session ID is required in the URL path")
		return
	}

	h.mu.RLock()
	store := h.store
	h.mu.RUnlock()

	if store == nil {
		WriteError(w, http.StatusServiceUnavailable, "no_store",
			"Session store is not available")
		return
	}

	// Check if session exists
	_, ok := store.Get(sessionID)
	if !ok {
		WriteError(w, http.StatusNotFound, "session_not_found",
			"Session '"+sessionID+"' not found")
		return
	}

	// Return empty messages list (messages not stored in current implementation)
	response := MessagesResponse{
		Messages: []any{},
		Total:    0,
	}
	WriteJSON(w, http.StatusOK, response)
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// isValidMeasurementMode checks if the measurement mode is valid.
func isValidMeasurementMode(mode string) bool {
	switch mode {
	case "passive", "active", "triggered":
		return true
	default:
		return false
	}
}

// -----------------------------------------------------------------------------
// In-Memory Session Store
// -----------------------------------------------------------------------------

// MemorySessionStore is an in-memory implementation of SessionStore.
// It is primarily useful for testing and development.
type MemorySessionStore struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

// Ensure MemorySessionStore implements SessionStore.
var _ SessionStore = (*MemorySessionStore)(nil)

// NewMemorySessionStore creates a new in-memory session store.
func NewMemorySessionStore() *MemorySessionStore {
	return &MemorySessionStore{
		sessions: make(map[string]*Session),
	}
}

// List returns all sessions in the store.
func (s *MemorySessionStore) List() []*Session {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sessions := make([]*Session, 0, len(s.sessions))
	for _, session := range s.sessions {
		sessions = append(sessions, session)
	}
	return sessions
}

// Get retrieves a session by ID.
func (s *MemorySessionStore) Get(id string) (*Session, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	session, ok := s.sessions[id]
	return session, ok
}

// Create adds a new session to the store.
func (s *MemorySessionStore) Create(session *Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.sessions[session.ID]; exists {
		return &SessionError{Code: "already_exists", Message: "session already exists"}
	}

	s.sessions[session.ID] = session
	return nil
}

// Update updates an existing session in the store.
func (s *MemorySessionStore) Update(session *Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.sessions[session.ID]; !exists {
		return &SessionError{Code: "not_found", Message: "session not found"}
	}

	s.sessions[session.ID] = session
	return nil
}

// Delete removes a session from the store.
func (s *MemorySessionStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.sessions[id]; !exists {
		return &SessionError{Code: "not_found", Message: "session not found"}
	}

	delete(s.sessions, id)
	return nil
}

// SessionError represents an error from session operations.
type SessionError struct {
	Code    string
	Message string
}

func (e *SessionError) Error() string {
	return e.Message
}
