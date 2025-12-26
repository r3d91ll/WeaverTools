package yarn

import (
	"fmt"
	"sync"
	"time"
)

// SessionRegistry manages multiple research sessions with thread-safe access.
type SessionRegistry struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

// NewSessionRegistry creates a new session registry.
func NewSessionRegistry() *SessionRegistry {
	return &SessionRegistry{
		sessions: make(map[string]*Session),
	}
}

// Register adds a session to the registry.
func (r *SessionRegistry) Register(name string, session *Session) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; exists {
		return fmt.Errorf("session %q already registered", name)
	}
	r.sessions[name] = session
	return nil
}

// Get retrieves a session by name.
func (r *SessionRegistry) Get(name string) (*Session, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	session, ok := r.sessions[name]
	return session, ok
}

// List returns all registered session names.
func (r *SessionRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]string, 0, len(r.sessions))
	for name := range r.sessions {
		result = append(result, name)
	}
	return result
}

// SessionStatus represents session status.
type SessionStatus struct {
	Name      string       `json:"name"`
	ID        string       `json:"id"`
	IsActive  bool         `json:"is_active"`
	StartedAt time.Time    `json:"started_at"`
	EndedAt   *time.Time   `json:"ended_at,omitempty"`
	Stats     SessionStats `json:"stats"`
}

// Status returns status for all registered sessions.
func (r *SessionRegistry) Status() map[string]SessionStatus {
	// Copy sessions to avoid holding lock during stats computation
	r.mu.RLock()
	sessions := make(map[string]*Session, len(r.sessions))
	for name, s := range r.sessions {
		sessions[name] = s
	}
	r.mu.RUnlock()

	result := make(map[string]SessionStatus)
	for name, session := range sessions {
		result[name] = SessionStatus{
			Name:      session.Name,
			ID:        session.ID,
			IsActive:  session.EndedAt == nil,
			StartedAt: session.StartedAt,
			EndedAt:   session.EndedAt,
			Stats:     session.Stats(),
		}
	}
	return result
}

// Active returns all sessions that have not ended.
func (r *SessionRegistry) Active() []*Session {
	// Copy sessions to avoid holding lock during filtering
	r.mu.RLock()
	sessions := make([]*Session, 0, len(r.sessions))
	for _, s := range r.sessions {
		sessions = append(sessions, s)
	}
	r.mu.RUnlock()

	var result []*Session
	for _, session := range sessions {
		if session.EndedAt == nil {
			result = append(result, session)
		}
	}
	return result
}

// Unregister removes a session from the registry.
func (r *SessionRegistry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; !exists {
		return fmt.Errorf("session %q not registered", name)
	}
	delete(r.sessions, name)
	return nil
}

// Create creates a new session and registers it in one operation.
func (r *SessionRegistry) Create(name, description string) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.sessions[name]; exists {
		return nil, fmt.Errorf("session %q already registered", name)
	}

	session := NewSession(name, description)
	r.sessions[name] = session
	return session, nil
}

// GetOrCreate returns an existing session or creates a new one.
// Returns the session and a bool indicating if it was created (true) or already existed (false).
func (r *SessionRegistry) GetOrCreate(name, description string) (*Session, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if session, exists := r.sessions[name]; exists {
		return session, false
	}

	session := NewSession(name, description)
	r.sessions[name] = session
	return session, true
}
